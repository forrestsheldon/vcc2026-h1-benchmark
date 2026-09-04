"""A lightweight VCC 2026 benchmark on the public H1 training data."""

from __future__ import annotations

import hashlib
import json
import shutil
import zlib
from dataclasses import replace
from datetime import UTC, datetime
from importlib.metadata import distribution, version
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
from cell_eval2 import (
    EvalConfig,
    aggregate_metrics_wide,
    precompute_cache,
    score_metrics,
)
from cell_eval2.cache import CacheStore, config_hash, fingerprint_adata
from cell_eval2.catalog import CATALOG, resolve_metrics
from cell_eval2.de import prepare_de
from cell_eval2.run import (
    _compute_de_side,
    dispatch_anndata_metrics,
    dispatch_de_metrics,
    metric_output_names,
)
from scipy import sparse

from .bounded import (
    RowSource,
    compute_de_bounded,
    load_reference_statistics,
    prediction_statistics,
    validate_counts,
)

CONTROL = "non-targeting"
CELLS_PER_TARGET = 400
N_TARGETS = 126
N_CONTROLS = 38_176
N_GENES = 18_080
N_DE_GENES = 10_780
CELL_EVAL2_VERSION = "0.16.0"
CELL_EVAL2_COMMIT = "5e64833518a6603a0301cbe28185d49c30f4a986"
CELL_EVAL2_WHEEL_SHA256 = (
    "c78428ba705a94536e4a55464a34d1905aa5730d4f7e52ea8dbef4e7171d4fbe"
)
PDEX_VERSION = "0.3.0"
PDEX_WHEEL_SHA256 = "fa7d805925c5ae16e3b060390bc47c47b7f79cd162cbd6a86544963ab6a44739"
SCORED_METRICS = (
    "pds_cosine",
    "expr_mse_unbiased_capped_norm",
    "de_wilcoxon_lfc_nmae",
    "de_wilcoxon_direction_fidelity_yield_raw",
    "de_wilcoxon_direction_reach_raw",
    "de_wilcoxon_sig_jaccard",
)
CONTROL_BASELINE_SEED_NAMESPACE = "h1-control-baseline-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve())


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, frame: pd.DataFrame | pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(frame, pl.DataFrame):
        frame.write_csv(path, float_precision=17)
    else:
        frame.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")


def evaluation_config(
    *, cache_real: Path | None = None, cache_pred: Path | None = None
) -> EvalConfig:
    """Load Arc's preset and change only the documented H1 execution fields."""
    base = EvalConfig.from_preset("vcc2026")
    return replace(
        base,
        pert_col="target_gene",
        device="cpu",
        cache_real=str(cache_real) if cache_real is not None else None,
        cache_pred=str(cache_pred) if cache_pred is not None else None,
        de=replace(base.de, backend="pdex"),
    )


def public_config() -> EvalConfig:
    """The portable scoring configuration; cache locations are operational, not scientific."""
    return evaluation_config(cache_real=None, cache_pred=None)


def target_contract(obs: pd.DataFrame, target_counts_path: Path) -> pd.DataFrame:
    source = pd.read_csv(target_counts_path)
    order = source["target_gene"].astype(str).tolist()
    counts = obs["target_gene"].astype(str).value_counts()
    if len(order) != 150 or len(set(order)) != 150:
        raise ValueError("pert_counts_Training.csv must contain 150 unique targets")
    if set(order) != set(counts.index) - {CONTROL}:
        raise ValueError("target list and H1 labels differ")
    frame = pd.DataFrame(
        {
            "target_gene": order,
            "source_order": np.arange(len(order)),
            "n_cells": [int(counts[target]) for target in order],
        }
    )
    if "n_cells" in source and not frame["n_cells"].equals(
        source["n_cells"].astype(int)
    ):
        raise ValueError("pert_counts_Training.csv counts do not match H1 labels")
    frame["in_benchmark"] = frame["n_cells"] >= CELLS_PER_TARGET
    if frame["in_benchmark"].sum() != N_TARGETS:
        raise ValueError(f"expected {N_TARGETS} targets with at least 400 cells")
    return frame


def reconstruct_reference_cells(
    obs: pd.DataFrame, target_counts_path: Path
) -> pd.DataFrame:
    """Reconstruct replicate 0 from celleval2_h1_subsample.py."""
    labels = obs["target_gene"].astype(str).to_numpy()
    records: list[dict] = []
    for row in target_contract(obs, target_counts_path).itertuples(index=False):
        if not row.in_benchmark:
            continue
        available = np.flatnonzero(labels == row.target_gene)
        selected = np.sort(_sample_target_rows(available, row.target_gene))
        for source_row in selected:
            records.append(
                {
                    "target_gene": row.target_gene,
                    "source_order": int(row.source_order),
                    "source_row": int(source_row),
                    "obs_name": str(obs.index[source_row]),
                }
            )
    result = pd.DataFrame(records)
    if len(result) != N_TARGETS * CELLS_PER_TARGET:
        raise ValueError("canonical split does not contain 126 x 400 cells")
    return result


def _sample_target_rows(available: np.ndarray, target: str) -> np.ndarray:
    rng = np.random.default_rng(zlib.crc32(target.encode()))
    return available[rng.choice(len(available), CELLS_PER_TARGET, replace=False)]


def _validate_count_matrix(matrix) -> None:
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix).ravel()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("counts must be finite and non-negative")
    if not np.equal(values, np.floor(values)).all():
        raise ValueError("counts must be integer-valued")


def make_adata(matrix, labels, genes, obs_names) -> ad.AnnData:
    _validate_count_matrix(matrix)
    obs = pd.DataFrame(
        {"target_gene": np.asarray(labels, dtype=str)},
        index=pd.Index(np.asarray(obs_names, dtype=str)),
    )
    var = pd.DataFrame(index=pd.Index(np.asarray(genes, dtype=str)))
    return ad.AnnData(sparse.csr_matrix(matrix), obs=obs, var=var)


def assemble_reference(data: ad.AnnData, cells: pd.DataFrame) -> ad.AnnData:
    labels = data.obs["target_gene"].astype(str).to_numpy()
    target_blocks = []
    for target, group in cells.groupby("target_gene", sort=False):
        available = np.flatnonzero(labels == target)
        selected = _sample_target_rows(available, target)
        if set(selected) != set(group["source_row"].astype(int)):
            raise ValueError(
                f"reference cell manifest does not match the CRC32 draw for {target}"
            )
        # The DE producer reduced cells in deterministic RNG order. Retaining that order here
        # keeps arithmetic CPM means bit-aligned; the public manifest remains source-row sorted.
        target_blocks.append(selected)
    target_rows = np.concatenate(target_blocks)
    control_rows = np.flatnonzero(labels == CONTROL)
    if len(control_rows) != N_CONTROLS:
        raise ValueError(
            f"expected {N_CONTROLS} H1 controls, found {len(control_rows)}"
        )
    rows = np.concatenate([target_rows, control_rows])
    selected_labels = labels[rows]
    expected = cells["target_gene"].astype(str).to_numpy()
    if not np.array_equal(selected_labels[: len(cells)], expected):
        raise ValueError("reference cell manifest does not match the H1 labels")
    return make_adata(
        data.X[rows], selected_labels, data.var_names, data.obs_names[rows]
    )


def _reference_gene_universe(data: ad.AnnData) -> list[str]:
    labels = data.obs["target_gene"].astype(str).to_numpy()
    controls = data.X[np.flatnonzero(labels == CONTROL)].tocsr().astype(np.float64)
    totals = np.asarray(controls.sum(axis=1)).ravel()
    if (totals == 0).any():
        raise ValueError("H1 contains an empty control cell")
    controls.data /= np.repeat(totals / 1_000_000, np.diff(controls.indptr))
    keep = np.asarray(controls.mean(axis=0)).ravel() > 5
    return data.var_names[keep].astype(str).tolist()


def validate_de_artifact(
    path: Path,
    summary_path: Path,
    targets: list[str],
    genes: list[str],
) -> pl.DataFrame:
    frame = pl.read_parquet(path)
    required = {
        "target",
        "gene",
        "log2_fold_change",
        "p_value",
        "p_adj",
    }
    if not required <= set(frame.columns):
        raise ValueError(
            f"DE artifact is missing {sorted(required - set(frame.columns))}"
        )
    if frame.height != len(targets) * len(genes):
        raise ValueError(
            f"DE artifact has {frame.height:,} rows; expected {len(targets) * len(genes):,}"
        )
    if set(frame["target"].unique().to_list()) != set(targets):
        raise ValueError("DE target set does not equal the canonical benchmark panel")
    observed_genes = frame.group_by("target").agg(
        pl.col("gene").n_unique().alias("n_genes"),
        pl.col("gene").sort().alias("genes"),
    )
    if not observed_genes["n_genes"].eq(len(genes)).all():
        raise ValueError(
            "each DE target must contain one row per reference-filtered gene"
        )
    expected_genes = sorted(genes)
    if any(value != expected_genes for value in observed_genes["genes"].to_list()):
        raise ValueError("DE feature universe does not match reference mean CPM > 5")

    summary = pd.read_csv(summary_path).set_index("target").loc[targets]
    if (
        len(summary) != len(targets)
        or not summary["n_cells"].eq(CELLS_PER_TARGET).all()
    ):
        raise ValueError(
            "frozen DE summary does not describe 400 cells for every target"
        )
    calculated = (
        frame.with_columns((pl.col("p_adj") < 0.05).alias("significant"))
        .group_by("target")
        .agg(
            pl.col("significant").sum().alias("n_de"),
            (pl.col("significant") & (pl.col("log2_fold_change") > 0))
            .sum()
            .alias("n_up"),
            (pl.col("significant") & (pl.col("log2_fold_change") < 0))
            .sum()
            .alias("n_down"),
        )
        .to_pandas()
        .set_index("target")
        .loc[targets]
    )
    for column in ("n_de", "n_up", "n_down"):
        if not calculated[column].astype(int).equals(summary[column].astype(int)):
            raise ValueError(f"DE artifact and summary disagree on {column}")

    on_target = (
        frame.filter(pl.col("target") == pl.col("gene")).to_pandas().set_index("target")
    )
    common = summary.index.intersection(on_target.index)
    for artifact_column, summary_column in (
        ("log2_fold_change", "target_log2_fold_change"),
        ("p_adj", "target_p_adj"),
    ):
        if not np.allclose(
            on_target.loc[common, artifact_column],
            summary.loc[common, summary_column],
            rtol=1e-12,
            atol=1e-15,
            equal_nan=True,
        ):
            raise ValueError(f"DE artifact and summary disagree on {summary_column}")
    missing = summary.index.difference(on_target.index)
    if len(missing) and not summary.loc[
        missing, ["target_log2_fold_change", "target_p_adj"]
    ].isna().all(axis=None):
        raise ValueError(
            "DE summary reports on-target values outside the filtered gene universe"
        )

    return frame.select(
        "target",
        pl.col("gene").alias("feature"),
        "log2_fold_change",
        "p_adj",
    )


class _CacheSeeder:
    """Capture Arc's own cache coordinates and insert a supplied full DE table."""

    def __init__(self, root: Path, value: pl.DataFrame):
        self.store = CacheStore(str(root))
        self.value = value

    def get_or_compute(self, key, *, fingerprint, params, kind, compute):
        self.store.put(
            key,
            self.value,
            fingerprint=fingerprint,
            params=params,
            kind=kind,
        )
        return self.value


def seed_reference_cache(
    real: ad.AnnData, de: pl.DataFrame, config: EvalConfig
) -> None:
    fingerprint = fingerprint_adata(
        real, pert_col=config.pert_col, strict=config.cache_strict
    )
    _compute_de_side(
        real,
        cfg=config,
        fp=fingerprint,
        store=_CacheSeeder(Path(config.cache_real), de),
        side="real",
    )
    precompute_cache(real, side="real", config=config, de=de, comparator="bulk_lognorm")


def _distribution_digest(name: str) -> dict:
    dist = distribution(name)
    digest = hashlib.sha256()
    files = []
    for item in sorted(dist.files or [], key=str):
        path = Path(dist.locate_file(item))
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        rel = str(item)
        files.append(rel)
        digest.update(rel.encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    record = next(
        (
            Path(dist.locate_file(item))
            for item in dist.files or []
            if str(item).endswith("RECORD")
        ),
        None,
    )
    return {
        "version": dist.version,
        "installed_files_sha256": digest.hexdigest(),
        "record_sha256": sha256(record) if record and record.exists() else None,
        "n_hashed_files": len(files),
    }


def package_provenance() -> dict:
    cell_eval = _distribution_digest("cell-eval2")
    cell_eval["source_commit"] = CELL_EVAL2_COMMIT
    cell_eval["wheel_sha256"] = CELL_EVAL2_WHEEL_SHA256
    pdex = _distribution_digest("pdex")
    pdex["wheel_sha256"] = PDEX_WHEEL_SHA256
    return {"cell-eval2": cell_eval, "pdex": pdex}


def prepare_reference(args) -> None:
    if version("cell-eval2") != CELL_EVAL2_VERSION or version("pdex") != PDEX_VERSION:
        raise RuntimeError("run with cell-eval2==0.16.0 and pdex==0.3.0")
    data = ad.read_h5ad(args.h1, backed="r")
    try:
        if data.shape[1] != N_GENES:
            raise ValueError(f"expected {N_GENES} H1 genes")
        cells = reconstruct_reference_cells(data.obs, args.target_counts)
        targets = cells.drop_duplicates("target_gene")["target_gene"].tolist()
        genes = _reference_gene_universe(data)
        if len(genes) != N_DE_GENES:
            raise ValueError(
                f"expected {N_DE_GENES} genes above reference mean CPM > 5"
            )
        de = validate_de_artifact(args.de, args.de_summary, targets, genes)
        real = assemble_reference(data, cells)
    finally:
        data.file.close()

    write_csv(args.reference_cells, cells)
    config = evaluation_config(cache_real=args.reference_cache)
    seed_reference_cache(real, de, config)
    manifest = {
        "benchmark": "VCC 2026 evaluation profile on a deterministic H1 resampling panel",
        "official_score": False,
        "withheld_truth": False,
        "panel": {
            "targets": N_TARGETS,
            "cells_per_target": CELLS_PER_TARGET,
            "controls": N_CONTROLS,
            "genes": N_GENES,
            "de_genes": N_DE_GENES,
            "target_order": targets,
            "target_order_source": relative(args.target_counts),
            "sampling": "default_rng(crc32(target)).choice(..., replace=False), then source-row sort",
            "pds_excluded_target_genes": targets,
            "excluded_targets": 24,
            "stage1_split": "superseded for this public benchmark",
        },
        "inputs": {
            "h1": {
                "path": relative(args.h1),
                "sha256": sha256(args.h1),
                "source_url": "gs://arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025/",
                "accessed": "2026-08-30",
            },
            "target_counts": {
                "path": relative(args.target_counts),
                "sha256": sha256(args.target_counts),
            },
            "de": {
                "path": relative(args.de),
                "sha256": sha256(args.de),
                "producer": "scripts/exploration/celleval2_h1_subsample.py, replicate 0",
            },
            "de_summary": {
                "path": relative(args.de_summary),
                "sha256": sha256(args.de_summary),
            },
        },
        "artifacts": {
            "reference_cells": {
                "path": relative(args.reference_cells),
                "sha256": sha256(args.reference_cells),
            },
            "reference_cache": relative(args.reference_cache),
            "reference_cache_files": {
                path.name: sha256(path)
                for path in sorted(args.reference_cache.iterdir())
                if path.is_file()
                and path.name not in {"manifest.json", "manifest.json.lock"}
            },
        },
        "configuration": public_config().to_dict(),
        "configuration_sha256": config_hash(public_config().to_dict()),
        "packages": package_provenance(),
    }
    write_json(args.manifest, manifest)
    print(
        f"Prepared {N_TARGETS} targets and {N_CONTROLS} controls; cache: {args.reference_cache}"
    )


def read_reference(args) -> tuple[ad.AnnData, list[str]]:
    if not args.reference_cells.exists() or not args.manifest.exists():
        raise FileNotFoundError("run prepare-reference before this command")
    cells = pd.read_csv(args.reference_cells)
    data = ad.read_h5ad(args.h1, backed="r")
    try:
        expected = reconstruct_reference_cells(data.obs, args.target_counts)
        if not cells.equals(expected):
            raise ValueError(
                "reference_cells.csv does not reconstruct from the canonical seed rule"
            )
        real = assemble_reference(data, cells)
    finally:
        data.file.close()
    targets = cells.drop_duplicates("target_gene")["target_gene"].astype(str).tolist()
    return real, targets


def validate_prediction(
    prediction: ad.AnnData,
    targets: list[str],
    genes: list[str],
    *,
    cells_per_target: int = CELLS_PER_TARGET,
) -> None:
    if (
        prediction.n_vars != len(genes)
        or prediction.var_names.astype(str).tolist() != genes
    ):
        raise ValueError("prediction gene axis must exactly match H1")
    if prediction.obs.columns.tolist() != ["target_gene"]:
        raise ValueError("prediction obs must contain exactly one column: target_gene")
    labels = prediction.obs["target_gene"].astype(str)
    counts = labels.value_counts()
    if CONTROL in counts or set(counts.index) != set(targets):
        raise ValueError(
            "prediction must contain the canonical targets and no controls"
        )
    if counts.reindex(targets).ne(cells_per_target).any():
        raise ValueError(
            f"prediction must contain exactly {cells_per_target} cells per target"
        )
    _validate_count_matrix(prediction.X)
    totals = np.asarray(prediction.X.sum(axis=1)).ravel()
    if (totals == 0).any() or (totals > 1_000_000).any():
        raise ValueError("prediction cell totals must be in [1, 1,000,000]")


def augment_prediction(prediction: ad.AnnData, real: ad.AnnData) -> ad.AnnData:
    control_mask = real.obs["target_gene"].astype(str).eq(CONTROL).to_numpy()
    controls = real[control_mask]
    matrix = sparse.vstack([prediction.X, controls.X], format="csr")
    labels = np.concatenate(
        [
            prediction.obs["target_gene"].astype(str).to_numpy(),
            np.repeat(CONTROL, controls.n_obs),
        ]
    )
    names = np.concatenate(
        [
            prediction.obs_names.astype(str),
            np.asarray([f"h1-control:{name}" for name in controls.obs_names]),
        ]
    )
    return make_adata(matrix, labels, real.var_names, names)


def sample_control_rows(
    control_rows: np.ndarray,
    targets: list[str],
    *,
    cells_per_target: int = CELLS_PER_TARGET,
) -> tuple[np.ndarray, np.ndarray]:
    """Select unchanged H1 rows independently and without replacement per target."""
    selected_rows = []
    labels = []
    for target in targets:
        seed = zlib.crc32(f"{CONTROL_BASELINE_SEED_NAMESPACE}:{target}".encode())
        selected = control_rows[
            np.random.default_rng(seed).choice(
                len(control_rows), cells_per_target, replace=False
            )
        ]
        selected_rows.extend(np.sort(selected))
        labels.extend([target] * cells_per_target)
    return np.asarray(selected_rows), np.asarray(labels)


def _ordered_results(
    results: pl.DataFrame, targets: list[str], metrics: list[str]
) -> pl.DataFrame:
    target_order = {target: index for index, target in enumerate(targets)}
    metric_order = {metric: index for index, metric in enumerate(metrics)}
    return (
        results.with_columns(
            pl.col("perturbation").replace_strict(target_order).alias("_target_order"),
            pl.col("metric").replace_strict(metric_order).alias("_metric_order"),
        )
        .sort("_target_order", "_metric_order")
        .drop("_target_order", "_metric_order")
        .rename({"perturbation": "target_gene"})
    )


def tidy_aggregate(
    results: pl.DataFrame, config: EvalConfig
) -> tuple[pl.DataFrame, pl.DataFrame]:
    metrics = metric_output_names(config)
    wide = aggregate_metrics_wide(results, metrics=metrics)
    mean = wide.filter(pl.col("statistic") == "mean")
    counts = results.group_by("metric").agg(pl.len().alias("n_targets"))
    lookup = dict(zip(counts["metric"].to_list(), counts["n_targets"].to_list()))
    rows = [
        {
            "metric": metric,
            "aggregation": CATALOG[metric].agg,
            "n_targets": int(lookup.get(metric, N_TARGETS)),
            "raw_value": float(mean[metric].item()),
            "scored": metric in SCORED_METRICS,
        }
        for metric in metrics
    ]
    return wide, pl.DataFrame(rows)


def read_benchmark_contract(data: ad.AnnData, args) -> tuple[list[str], np.ndarray]:
    cells = pd.read_csv(args.reference_cells)
    expected = reconstruct_reference_cells(data.obs, args.target_counts)
    if not cells.equals(expected):
        raise ValueError("reference_cells.csv does not match the canonical H1 split")
    targets = cells.drop_duplicates("target_gene")["target_gene"].astype(str).tolist()
    labels = data.obs["target_gene"].astype(str).to_numpy()
    control_rows = np.flatnonzero(labels == CONTROL)
    if len(control_rows) != N_CONTROLS:
        raise ValueError(f"expected {N_CONTROLS} H1 controls")
    return targets, control_rows


def read_scoring_contract(args) -> list[str]:
    benchmark = json.loads(args.manifest.read_text())
    cells = pd.read_csv(args.reference_cells)
    if (
        sha256(args.reference_cells)
        != benchmark["artifacts"]["reference_cells"]["sha256"]
    ):
        raise ValueError(
            "reference_cells.csv checksum differs from the benchmark manifest"
        )
    counts = cells["target_gene"].astype(str).value_counts()
    targets = cells.drop_duplicates("target_gene")["target_gene"].astype(str).tolist()
    if targets != benchmark["panel"]["target_order"]:
        raise ValueError(
            "reference cell target order differs from the benchmark manifest"
        )
    if len(targets) != N_TARGETS or counts.reindex(targets).ne(CELLS_PER_TARGET).any():
        raise ValueError("reference cell manifest must contain 126 x 400 cells")
    return targets


def open_controls(args) -> ad.AnnData:
    benchmark = json.loads(args.manifest.read_text())
    manifest = json.loads(args.controls_manifest.read_text())
    artifact = manifest["artifact"]
    if manifest["source"]["sha256"] != benchmark["inputs"]["h1"]["sha256"]:
        raise ValueError(
            "control artifact and benchmark derive from different H1 files"
        )
    if sha256(args.controls) != artifact["sha256"]:
        raise ValueError("control artifact checksum differs from its manifest")
    controls = ad.read_h5ad(args.controls, backed="r")
    labels = controls.obs["target_gene"].astype(str)
    if controls.shape != (N_CONTROLS, N_GENES) or not labels.eq(CONTROL).all():
        controls.file.close()
        raise ValueError("control artifact must contain the 38,176 H1 controls")
    return controls


def validate_source(source: RowSource, targets: list[str], genes: list[str]) -> None:
    if source.genes != genes:
        raise ValueError("prediction gene axis must exactly match H1")
    counts = pd.Series(source.labels).value_counts()
    if CONTROL in counts or set(counts.index) != set(targets):
        raise ValueError(
            "prediction must contain the canonical targets and no controls"
        )
    if counts.reindex(targets).ne(CELLS_PER_TARGET).any():
        raise ValueError(
            f"prediction must contain exactly {CELLS_PER_TARGET} cells per target"
        )
    validate_counts(source)


def reference_artifacts(args):
    manifest = json.loads(args.manifest.read_text())
    files = manifest["artifacts"]["reference_cache_files"]
    moment_name = next(name for name in files if name.endswith(".moments.npz"))
    de_name = next(name for name in files if "de_wilcoxon_table" in name)
    for name in (moment_name, de_name):
        path = args.reference_cache / name
        if sha256(path) != files[name]:
            raise ValueError(f"reference cache checksum mismatch: {name}")
    return args.reference_cache / moment_name, args.reference_cache / de_name


def scoring_meta(args, config: EvalConfig) -> dict:
    bundle = json.loads((args.scale_bundle / "manifest.json").read_text())
    return {
        "cell_eval2_version": CELL_EVAL2_VERSION,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "resolved_device": "cpu",
        "resolved_de_backend": "pdex",
        "de_real_fingerprint": None,
        "de_pred_fingerprint": None,
        "source_fingerprint": bundle["source_fingerprint"],
        "source_fingerprint_strict": bundle["source_fingerprint_strict"],
        "input_type_real_effective": "counts",
        "input_type_pred_effective": "counts",
        "comparator": "bulk_lognorm",
        "config_digest": bundle["config_digest"],
        "anchor_semantic_identity": bundle["anchor_semantic_identity"],
        "anchor_metric_names": bundle["anchor_metric_names"],
    }


def score_source(
    args,
    prediction: RowSource,
    controls: RowSource,
    targets: list[str],
    prediction_provenance: dict,
    cache_identity: str,
) -> None:
    config = replace(
        evaluation_config(cache_real=args.reference_cache),
        num_threads=args.de_threads,
    )
    moment_path, de_path = reference_artifacts(args)
    real_bulks, real_moments, real_de = load_reference_statistics(moment_path, de_path)
    pred_bulks, pred_moments = prediction_statistics(
        prediction,
        targets,
        CONTROL,
        real_bulks,
        real_moments,
        bulk_target_sum=config.bulk_target_sum,
    )
    kept = set(real_de["feature"].unique().to_list())
    kept_genes = [gene for gene in prediction.genes if gene in kept]
    cache_dir = args.score_cache / cache_identity[:20]
    pred_de = compute_de_bounded(
        prediction,
        controls,
        targets,
        CONTROL,
        kept_genes,
        cache_dir,
        cache_identity,
        gene_chunk=args.gene_chunk,
        threads=args.de_threads,
        epsilon=config.de.epsilon,
    )
    names = list(resolve_metrics(config.metrics, version=config.version)[0])
    rows = dispatch_anndata_metrics(
        names,
        pred_bulks,
        real_bulks,
        np.asarray(prediction.genes),
        config,
        comparator="bulk_lognorm",
        pred_moments=pred_moments,
        real_moments=real_moments,
        driver="bounded-memory H1 scorer",
    )
    prepared = prepare_de(
        pred_de,
        real_de,
        control=CONTROL,
        sort_by=config.de.sort_by,
        p_adj_threshold=config.de.p_adj_threshold,
        nan_lfc_policy=config.de.nan_lfc_policy,
        min_abs_log2fc=config.de.min_abs_log2fc,
    )
    rows.extend(dispatch_de_metrics(names, prepared, config))
    results = pl.DataFrame(
        rows,
        schema={"perturbation": pl.Utf8, "metric": pl.Utf8, "value": pl.Float64},
    )
    wide, aggregate = tidy_aggregate(results, config)
    run_meta = scoring_meta(args, config)
    scaled = score_metrics(wide, real_bundle=args.scale_bundle, user_meta=run_meta)

    args.output.mkdir(parents=True, exist_ok=True)
    metrics = metric_output_names(config)
    per_target = _ordered_results(results, targets, metrics)
    paths = {
        "per_target": args.output / "per_target.csv",
        "aggregates": args.output / "aggregates.csv",
        "scores": args.output / "scores.csv",
    }
    write_csv(paths["per_target"], per_target)
    write_csv(paths["aggregates"], aggregate)
    write_csv(paths["scores"], scaled)
    manifest = {
        "benchmark_manifest_sha256": sha256(args.manifest),
        "prediction": prediction_provenance,
        "controls": {
            "path": relative(args.controls),
            "sha256": json.loads(args.controls_manifest.read_text())["artifact"][
                "sha256"
            ],
        },
        "scale_bundle": {
            "path": relative(args.scale_bundle),
            "manifest_sha256": sha256(args.scale_bundle / "manifest.json"),
        },
        "configuration_sha256": config_hash(public_config().to_dict()),
        "driver": {
            "name": "bounded-memory CPU/pdex",
            "gene_chunk": args.gene_chunk,
            "de_threads": args.de_threads,
        },
        "metrics": metrics,
        "scored_metrics": list(SCORED_METRICS),
        "outputs": {
            name: {"path": path.name, "sha256": sha256(path)}
            for name, path in paths.items()
        },
        "packages": package_provenance(),
    }
    write_json(args.output / "manifest.json", manifest)
    shutil.rmtree(cache_dir)
    print(scaled)


def score(args) -> None:
    controls_data = open_controls(args)
    prediction = ad.read_h5ad(args.prediction, backed="r")
    try:
        targets = read_scoring_contract(args)
        source = RowSource(
            prediction,
            np.arange(prediction.n_obs),
            prediction.obs["target_gene"].astype(str).to_numpy(),
        )
        validate_source(source, targets, controls_data.var_names.astype(str).tolist())
        control_rows = np.arange(controls_data.n_obs)
        controls = RowSource(
            controls_data, control_rows, np.repeat(CONTROL, len(control_rows))
        )
        prediction_hash = sha256(args.prediction)
        controls_hash = json.loads(args.controls_manifest.read_text())["artifact"][
            "sha256"
        ]
        identity = hashlib.sha256(
            f"{prediction_hash}:{controls_hash}:{sha256(args.manifest)}:{CELL_EVAL2_VERSION}".encode()
        ).hexdigest()
        score_source(
            args,
            source,
            controls,
            targets,
            {"path": args.prediction.name, "sha256": prediction_hash},
            identity,
        )
    finally:
        prediction.file.close()
        controls_data.file.close()


def score_control_baseline(args) -> None:
    controls_data = open_controls(args)
    try:
        targets = read_scoring_contract(args)
        control_rows = np.arange(controls_data.n_obs)
        rows, labels = sample_control_rows(control_rows, targets)
        prediction = RowSource(controls_data, rows, labels)
        validate_source(
            prediction, targets, controls_data.var_names.astype(str).tolist()
        )
        controls = RowSource(
            controls_data, control_rows, np.repeat(CONTROL, len(control_rows))
        )
        provenance = {
            "generator": "deterministic size-matched H1 controls",
            "control_pool_cells": N_CONTROLS,
            "cells_per_target": CELLS_PER_TARGET,
            "seed_namespace": CONTROL_BASELINE_SEED_NAMESPACE,
        }
        identity = hashlib.sha256(
            json.dumps(
                {"prediction": provenance, "benchmark": sha256(args.manifest)},
                sort_keys=True,
            ).encode()
        ).hexdigest()
        score_source(args, prediction, controls, targets, provenance, identity)
    finally:
        controls_data.file.close()


def validate_command(args) -> None:
    controls = open_controls(args)
    prediction = ad.read_h5ad(args.prediction, backed="r")
    try:
        targets = read_scoring_contract(args)
        source = RowSource(
            prediction,
            np.arange(prediction.n_obs),
            prediction.obs["target_gene"].astype(str).to_numpy(),
        )
        validate_source(source, targets, controls.var_names.astype(str).tolist())
        print(
            f"Valid H1 benchmark prediction: {prediction.n_obs:,} cells x "
            f"{prediction.n_vars:,} genes"
        )
    finally:
        prediction.file.close()
        controls.file.close()

