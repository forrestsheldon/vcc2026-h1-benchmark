"""Bounded-memory primitives for the CPU/pdex H1 scorer."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
from cell_eval2.de_compute import _bh_per_target, _clipped_log2fc
from cell_eval2.moments import GroupMoments
from cell_eval2.prep import pseudobulk_bulk_lognorm_with_moments
from pdex import pdex
from scipy import sparse


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class RowSource:
    """A logical cell matrix backed by selected rows of an AnnData object."""

    adata: ad.AnnData
    rows: np.ndarray
    labels: np.ndarray

    def __post_init__(self) -> None:
        self.rows = np.asarray(self.rows, dtype=np.int64)
        self.labels = np.asarray(self.labels, dtype=str)
        if len(self.rows) != len(self.labels):
            raise ValueError("row and label counts differ")

    @property
    def n_obs(self) -> int:
        return len(self.rows)

    @property
    def genes(self) -> list[str]:
        return self.adata.var_names.astype(str).tolist()

    def read(self, positions: np.ndarray) -> sparse.csr_matrix:
        """Read arbitrary logical rows with a small, sorted physical HDF5 request."""
        positions = np.asarray(positions, dtype=np.int64)
        physical = self.rows[positions]
        order = np.argsort(physical, kind="stable")
        ordered = physical[order]
        unique, inverse = np.unique(ordered, return_inverse=True)
        matrix = sparse.csr_matrix(self.adata.X[unique])[inverse]
        return matrix[np.argsort(order, kind="stable")]

    def blocks(self, block_rows: int = 256):
        for start in range(0, self.n_obs, block_rows):
            stop = min(start + block_rows, self.n_obs)
            positions = np.arange(start, stop)
            yield start, stop, self.read(positions)


def validate_counts(source: RowSource) -> None:
    for _, _, matrix in source.blocks():
        if not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
            raise ValueError("counts must be finite and non-negative")
        if not np.equal(matrix.data, np.floor(matrix.data)).all():
            raise ValueError("counts must be integer-valued")
        totals = np.asarray(matrix.astype(np.float64).sum(axis=1)).ravel()
        if (totals < 1).any() or (totals > 1_000_000).any():
            raise ValueError("prediction cell totals must be in [1, 1,000,000]")


def row_totals(source: RowSource) -> np.ndarray:
    totals = np.empty(source.n_obs, dtype=np.float64)
    for start, stop, matrix in source.blocks():
        totals[start:stop] = np.asarray(matrix.astype(np.float64).sum(axis=1)).ravel()
    if (totals <= 0).any():
        raise ValueError("empty cells cannot be scored")
    return totals


def load_reference_statistics(moment_path: Path, de_path: Path):
    with np.load(moment_path) as stored:
        perts = stored["perts"].astype(str)
        means = stored["means"].copy()
        moments = GroupMoments(
            perts=perts,
            counts=stored["counts"].copy(),
            sumsq=stored["sumsq"].copy(),
            jk=stored["jk"].copy(),
        )
    return (
        {"bulk_lognorm": (perts, means)},
        {"bulk_lognorm": moments},
        pl.read_parquet(de_path),
    )


def prediction_statistics(
    source: RowSource,
    targets: list[str],
    control: str,
    real_bulks: dict,
    real_moments: dict,
    *,
    bulk_target_sum: float,
):
    """Compute exact resident target reductions and reuse the identical real control row."""
    target_means = {}
    target_moments = {}
    for target_index, target in enumerate(targets):
        positions = np.flatnonzero(source.labels == target)
        matrix = source.read(positions)
        group = ad.AnnData(
            matrix,
            obs=pd.DataFrame(
                {"target_gene": [target] * len(positions)},
                index=[f"{target}:{index}" for index in range(len(positions))],
            ),
            var=pd.DataFrame(index=source.genes),
        )
        _, means, moments = pseudobulk_bulk_lognorm_with_moments(
            group, "target_gene", bulk_target_sum=bulk_target_sum
        )
        target_means[target] = means[0]
        target_moments[target] = (
            moments.counts[0],
            moments.sumsq[0],
            moments.jk[0],
        )
        if (target_index + 1) % 10 == 0 or target_index + 1 == len(targets):
            print(
                f"Prediction moments {target_index + 1} of {len(targets)} targets",
                flush=True,
            )

    real_perts, real_means = real_bulks["bulk_lognorm"]
    real_moment = real_moments["bulk_lognorm"]
    real_index = {str(label): index for index, label in enumerate(real_perts)}
    order = np.sort(np.asarray([control, *targets], dtype=str))
    means = []
    counts = []
    sumsq = []
    jk = []
    for label in order:
        if label == control:
            index = real_index[control]
            means.append(real_means[index])
            counts.append(real_moment.counts[index])
            sumsq.append(real_moment.sumsq[index])
            jk.append(real_moment.jk[index])
        else:
            means.append(target_means[label])
            count, square, correction = target_moments[label]
            counts.append(count)
            sumsq.append(square)
            jk.append(correction)
    pred_moment = GroupMoments(
        perts=order,
        counts=np.asarray(counts),
        sumsq=np.asarray(sumsq),
        jk=np.asarray(jk),
    )
    return {"bulk_lognorm": (order, np.vstack(means))}, {"bulk_lognorm": pred_moment}


def _fill_normalized_chunk(
    output: np.ndarray,
    source: RowSource,
    totals: np.ndarray,
    gene_indices: np.ndarray,
    group_index: dict[str, int],
    group_sums: np.ndarray,
    *,
    row_offset: int,
) -> None:
    for start, stop, matrix in source.blocks():
        linear = matrix[:, gene_indices].toarray().astype(np.float64)
        linear *= (1_000_000.0 / totals[start:stop])[:, None]
        block_labels = source.labels[start:stop]
        for label in np.unique(block_labels):
            group_sums[group_index[label]] += linear[block_labels == label].sum(axis=0)
        np.log1p(linear, out=linear)
        output[row_offset + start : row_offset + stop] = linear


def _de_chunk(
    prediction: RowSource,
    controls: RowSource,
    prediction_totals: np.ndarray,
    control_totals: np.ndarray,
    targets: list[str],
    control: str,
    gene_indices: np.ndarray,
    *,
    threads: int,
    epsilon: float,
) -> pl.DataFrame:
    genes = np.asarray(prediction.genes)[gene_indices]
    labels = np.concatenate([prediction.labels, controls.labels])
    groups = np.unique(labels)
    group_index = {label: index for index, label in enumerate(groups)}
    group_sums = np.zeros((len(groups), len(genes)), dtype=np.float64)
    values = np.zeros((len(labels), len(genes)), dtype=np.float64)
    _fill_normalized_chunk(
        values,
        prediction,
        prediction_totals,
        gene_indices,
        group_index,
        group_sums,
        row_offset=0,
    )
    _fill_normalized_chunk(
        values,
        controls,
        control_totals,
        gene_indices,
        group_index,
        group_sums,
        row_offset=prediction.n_obs,
    )
    chunk = ad.AnnData(
        values,
        obs=pd.DataFrame({"target_gene": labels}),
        var=pd.DataFrame(index=genes),
    )
    pvalues = pdex(
        chunk,
        groupby="target_gene",
        mode="ref",
        reference=control,
        geometric_mean=True,
        epsilon=0.0,
        is_log1p=True,
        threads=threads,
    ).select("target", "feature", "p_value")
    counts = {label: int((labels == label).sum()) for label in groups}
    means = {
        label: group_sums[index] / counts[label] for label, index in group_index.items()
    }
    lfc = pl.concat(
        [
            pl.DataFrame(
                {
                    "target": target,
                    "feature": genes,
                    "log2_fold_change": _clipped_log2fc(
                        means[target], means[control], epsilon=epsilon
                    ),
                }
            )
            for target in targets
        ]
    )
    return lfc.join(pvalues, on=["target", "feature"], how="inner")


def compute_de_bounded(
    prediction: RowSource,
    controls: RowSource,
    targets: list[str],
    control: str,
    kept_genes: list[str],
    cache_dir: Path,
    cache_identity: str,
    *,
    gene_chunk: int,
    threads: int,
    epsilon: float,
) -> pl.DataFrame:
    """Run pdex on bounded dense gene slices and checkpoint each raw-p-value slice."""
    gene_lookup = {gene: index for index, gene in enumerate(prediction.genes)}
    gene_indices = np.asarray([gene_lookup[gene] for gene in kept_genes])
    control_totals = row_totals(controls)
    if prediction.adata is controls.adata:
        positions = np.searchsorted(controls.rows, prediction.rows)
        if np.array_equal(controls.rows[positions], prediction.rows):
            prediction_totals = control_totals[positions]
        else:
            prediction_totals = row_totals(prediction)
    else:
        prediction_totals = row_totals(prediction)
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for chunk_index, start in enumerate(range(0, len(gene_indices), gene_chunk)):
        indices = gene_indices[start : start + gene_chunk]
        frame_path = cache_dir / f"chunk_{chunk_index:03d}.parquet"
        meta_path = cache_dir / f"chunk_{chunk_index:03d}.json"
        expected = {
            "identity": cache_identity,
            "genes": np.asarray(prediction.genes)[indices].tolist(),
        }
        if frame_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta == {**expected, "sha256": file_sha256(frame_path)}:
                frames.append(pl.read_parquet(frame_path))
                continue
        print(
            f"Prediction DE genes {start + 1}-{start + len(indices)} of {len(gene_indices)}",
            flush=True,
        )
        frame = _de_chunk(
            prediction,
            controls,
            prediction_totals,
            control_totals,
            targets,
            control,
            indices,
            threads=threads,
            epsilon=epsilon,
        )
        frame.write_parquet(frame_path)
        meta_path.write_text(
            json.dumps(
                {**expected, "sha256": file_sha256(frame_path)},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        frames.append(frame)
    de = pl.concat(frames).sort("target", "feature")
    return _bh_per_target(de)
