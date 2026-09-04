"""Build the control-only H1 count artifact used by the local scorer."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
from scipy import sparse

CONTROL = "non-targeting"
CSR_CHUNK = 65_536


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve())


def _write_csr(
    source: ad.AnnData,
    rows: np.ndarray,
    output: Path,
    *,
    block_rows: int,
) -> int:
    obs = source.obs.iloc[rows].copy()
    var = source.var.copy()
    skeleton = ad.AnnData(
        sparse.csr_matrix((len(rows), source.n_vars), dtype=np.uint32),
        obs=obs,
        var=var,
    )
    skeleton.write_h5ad(output)

    with h5py.File(source.filename, "r") as handle:
        indptr = handle["X/indptr"][:]
    nnz = int(np.sum(indptr[rows + 1] - indptr[rows], dtype=np.int64))

    with h5py.File(output, "r+") as handle:
        del handle["X"]
        group = handle.create_group("X")
        group.attrs["encoding-type"] = "csr_matrix"
        group.attrs["encoding-version"] = "0.1.0"
        group.attrs["shape"] = np.asarray((len(rows), source.n_vars), dtype=np.int64)
        chunk = min(nnz, CSR_CHUNK)
        values = group.create_dataset(
            "data",
            (nnz,),
            dtype=np.uint32,
            chunks=(chunk,),
            compression="lzf",
            shuffle=True,
        )
        indices = group.create_dataset(
            "indices",
            (nnz,),
            dtype=np.int32,
            chunks=(chunk,),
            compression="lzf",
            shuffle=True,
        )
        output_indptr = group.create_dataset("indptr", (len(rows) + 1,), dtype=np.int64)
        output_indptr[0] = 0

        offset = 0
        for start in range(0, len(rows), block_rows):
            stop = min(start + block_rows, len(rows))
            matrix = sparse.csr_matrix(source.X[rows[start:stop]])
            if (
                not np.isfinite(matrix.data).all()
                or (matrix.data < 0).any()
                or not np.equal(matrix.data, np.floor(matrix.data)).all()
                or (matrix.data > np.iinfo(np.uint32).max).any()
            ):
                raise ValueError(
                    "H1 controls must contain finite non-negative integer counts"
                )
            end = offset + matrix.nnz
            values[offset:end] = matrix.data.astype(np.uint32)
            indices[offset:end] = matrix.indices.astype(np.int32)
            output_indptr[start + 1 : stop + 1] = offset + matrix.indptr[1:]
            offset = end
            print(f"Wrote {stop:,} of {len(rows):,} controls", flush=True)
        if offset != nnz:
            raise ValueError(f"wrote {offset:,} nonzeros; expected {nnz:,}")
    return nnz


def verify(source_path: Path, output_path: Path, *, block_rows: int) -> None:
    source = ad.read_h5ad(source_path, backed="r")
    output = ad.read_h5ad(output_path, backed="r")
    try:
        rows = np.flatnonzero(
            source.obs["target_gene"].astype(str).to_numpy() == CONTROL
        )
        if output.shape != (len(rows), source.n_vars):
            raise ValueError("control artifact has the wrong shape")
        if (
            output.var_names.astype(str).tolist()
            != source.var_names.astype(str).tolist()
        ):
            raise ValueError("control artifact gene axis differs from H1")
        for column in source.obs.columns:
            observed = output.obs[column].astype(str).to_numpy()
            expected = source.obs.iloc[rows][column].astype(str).to_numpy()
            if not np.array_equal(observed, expected):
                raise ValueError(f"control artifact differs in obs[{column!r}]")
        if (
            output.obs_names.astype(str).tolist()
            != source.obs_names[rows].astype(str).tolist()
        ):
            raise ValueError("control artifact observation names differ from H1")
        for start in range(0, len(rows), block_rows):
            stop = min(start + block_rows, len(rows))
            expected = sparse.csr_matrix(source.X[rows[start:stop]])
            observed = sparse.csr_matrix(output.X[start:stop])
            if (expected != observed).nnz:
                raise ValueError(f"control counts differ in rows {start}:{stop}")
            print(f"Verified {stop:,} of {len(rows):,} controls", flush=True)
    finally:
        output.file.close()
        source.file.close()


def build(
    source_path: Path, output_path: Path, manifest_path: Path, *, block_rows: int
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.h5ad")
    if temporary.exists():
        temporary.unlink()

    source = ad.read_h5ad(source_path, backed="r")
    try:
        labels = source.obs["target_gene"].astype(str).to_numpy()
        rows = np.flatnonzero(labels == CONTROL)
        nnz = _write_csr(source, rows, temporary, block_rows=block_rows)
        shape = [len(rows), source.n_vars]
        obs_columns = source.obs.columns.tolist()
        var_columns = source.var.columns.tolist()
    finally:
        source.file.close()

    temporary.replace(output_path)
    verify(source_path, output_path, block_rows=block_rows)
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "selection": "target_gene == 'non-targeting', retaining source-row order",
        "source": {
            "path": relative(source_path),
            "bytes": source_path.stat().st_size,
            "sha256": sha256(source_path),
        },
        "artifact": {
            "path": relative(output_path),
            "bytes": output_path.stat().st_size,
            "sha256": sha256(output_path),
            "shape": shape,
            "nnz": nnz,
            "matrix": "CSR raw counts",
            "dtype": "uint32",
            "compression": "lzf",
            "csr_data_chunk": CSR_CHUNK,
            "obs_columns": obs_columns,
            "var_columns": var_columns,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Created {output_path} ({output_path.stat().st_size / 2**30:.2f} GiB)")
