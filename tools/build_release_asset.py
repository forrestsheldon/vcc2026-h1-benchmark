"""Package the portable, non-count H1 benchmark artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_manifest(source: Path, cache_files: dict[str, str]) -> bytes:
    manifest = json.loads(source.read_text())
    manifest["artifacts"]["reference_cache"] = "reference_cache"
    manifest["artifacts"]["reference_cache_files"] = cache_files
    manifest["artifacts"]["reference_cells"]["path"] = "reference_cells.csv"
    manifest["inputs"]["h1"]["path"] = "Arc public H1 training object"
    manifest["inputs"]["h1"]["source_url"] = (
        "https://storage.googleapis.com/arc-institute-virtual-cell-atlas/"
        "virtual-cell-challenge/2025/train/adata_Training.h5ad"
    )
    manifest["inputs"]["target_counts"]["path"] = "pert_counts_Training.csv"
    manifest["inputs"]["de"]["path"] = "canonical frozen reference DE"
    manifest["inputs"]["de_summary"]["path"] = "canonical frozen DE summary"
    manifest["panel"]["target_order_source"] = "pert_counts_Training.csv"
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()


def add_file(bundle: zipfile.ZipFile, name: str, value: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    bundle.writestr(info, value, compresslevel=6)


def build(source_root: Path, output: Path) -> str:
    report = source_root / "reports/vcc2026-h1"
    cache = source_root / "data/derived/vcc2026_h1/reference_cache"
    files: dict[str, bytes] = {}

    for path in sorted(cache.iterdir()):
        if path.is_file() and path.name != "manifest.json.lock":
            files[f"reference_cache/{path.name}"] = path.read_bytes()
    cache_hashes = {
        Path(name).name: sha256_bytes(value)
        for name, value in files.items()
        if name.startswith("reference_cache/") and name != "reference_cache/manifest.json"
    }
    files["reference_cells.csv"] = (report / "reference_cells.csv").read_bytes()
    files["benchmark_manifest.json"] = portable_manifest(
        report / "benchmark_manifest.json", cache_hashes
    )
    for path in sorted((report / "scale").iterdir()):
        if path.is_file() and path.name != "config.yaml":
            files[f"scale/{path.name}"] = path.read_bytes()

    asset_manifest = {
        "benchmark_version": "v1",
        "files": {name: sha256_bytes(value) for name, value in sorted(files.items())},
    }
    files["asset_manifest.json"] = (
        json.dumps(asset_manifest, indent=2, sort_keys=True) + "\n"
    ).encode()

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w") as bundle:
        for name, value in sorted(files.items()):
            add_file(bundle, name, value)
    temporary.replace(output)
    return sha256(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.source_root.resolve(), args.output.resolve()))


if __name__ == "__main__":
    main()
