from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from importlib import resources
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import httpx
import pandas as pd

from .controls import build as build_controls
from .paths import BenchmarkPaths


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def registry() -> dict:
    resource = resources.files("vcc_h1_eval").joinpath("benchmark-v1.json")
    if resource.is_file():
        return json.loads(resource.read_text())
    source = Path(__file__).resolve().parents[2] / "assets" / "benchmark-v1.json"
    return json.loads(source.read_text())


def verify_file(path: Path, spec: dict) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    if "bytes" in spec and path.stat().st_size != spec["bytes"]:
        raise ValueError(f"wrong file size for {path}")
    if sha256(path) != spec["sha256"]:
        raise ValueError(f"checksum mismatch for {path}")


def download(spec: dict, destination: Path) -> Path:
    if destination.exists():
        try:
            verify_file(destination, spec)
            print(f"Using verified {destination}", flush=True)
            return destination
        except ValueError:
            destination.unlink()

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    if partial.exists() and "bytes" in spec and partial.stat().st_size >= spec["bytes"]:
        try:
            verify_file(partial, spec)
            partial.replace(destination)
            return destination
        except ValueError:
            partial.unlink()
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
        print(f"Resuming {destination.name} at {offset:,} bytes", flush=True)
    else:
        print(f"Downloading {destination.name}", flush=True)

    with httpx.stream(
        "GET", spec["url"], headers=headers, follow_redirects=True, timeout=None
    ) as response:
        response.raise_for_status()
        append = offset > 0 and response.status_code == 206
        if offset and not append:
            offset = 0
        written = offset
        next_report = written + 1024**3
        with partial.open("ab" if append else "wb") as handle:
            for chunk in response.iter_bytes(8 * 1024 * 1024):
                handle.write(chunk)
                written += len(chunk)
                if written >= next_report:
                    print(f"Downloaded {written / 2**30:.1f} GiB", flush=True)
                    next_report += 1024**3

    verify_file(partial, spec)
    partial.replace(destination)
    return destination


def _asset_files(root: Path) -> dict[str, str]:
    manifest = json.loads((root / "asset_manifest.json").read_text())
    return manifest["files"]


def verify_asset(root: Path) -> None:
    for relative, expected in _asset_files(root).items():
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"benchmark artifact mismatch: {relative}")


def install_asset(paths: BenchmarkPaths, archive: Path) -> None:
    temporary = paths.root / "benchmark.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (temporary / member.filename).resolve()
            if not target.is_relative_to(temporary.resolve()):
                raise ValueError(f"unsafe archive member: {member.filename}")
        bundle.extractall(temporary)
    verify_asset(temporary)
    if paths.benchmark_dir.exists():
        shutil.rmtree(paths.benchmark_dir)
    temporary.replace(paths.benchmark_dir)


def check(paths: BenchmarkPaths) -> dict:
    if version("cell-eval2") != "0.16.0" or version("pdex") != "0.3.0":
        raise RuntimeError("requires cell-eval2==0.16.0 and pdex==0.3.0")
    specs = registry()
    verify_file(paths.genes, specs["sources"]["genes"])
    verify_file(paths.target_counts, specs["sources"]["target_counts"])
    verify_asset(paths.benchmark_dir)

    benchmark = json.loads(paths.benchmark_manifest.read_text())
    controls_manifest = json.loads(paths.controls_manifest.read_text())
    artifact = controls_manifest["artifact"]
    if controls_manifest["source"]["sha256"] != benchmark["inputs"]["h1"]["sha256"]:
        raise ValueError("controls and benchmark use different H1 sources")
    verify_file(paths.controls, artifact)
    controls = ad.read_h5ad(paths.controls, backed="r")
    try:
        labels = controls.obs["target_gene"].astype(str)
        genes = pd.read_csv(paths.genes, header=None)[0].astype(str).tolist()
        if controls.shape != (38_176, 18_080):
            raise ValueError("control artifact has the wrong shape")
        if not labels.eq("non-targeting").all():
            raise ValueError("control artifact contains perturbed cells")
        if controls.var_names.astype(str).tolist() != genes:
            raise ValueError("control and published gene axes differ")
    finally:
        controls.file.close()
    return {
        "benchmark_version": specs["benchmark_version"],
        "controls_sha256": artifact["sha256"],
        "benchmark_manifest_sha256": sha256(paths.benchmark_manifest),
    }


def setup(
    paths: BenchmarkPaths,
    *,
    h1: Path | None = None,
    remove_source: bool = False,
    asset_path: Path | None = None,
) -> dict:
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.source_dir.mkdir(parents=True, exist_ok=True)
    specs = registry()
    download(specs["sources"]["genes"], paths.genes)
    download(specs["sources"]["target_counts"], paths.target_counts)

    managed_h1 = h1 is None
    if managed_h1:
        source = download(specs["sources"]["h1"], paths.h1)
    else:
        source = h1.resolve()
        verify_file(source, specs["sources"]["h1"])

    if asset_path is None:
        asset_spec = specs["asset"]
        if asset_spec["sha256"] == "PENDING":
            raise RuntimeError("benchmark release asset has not been published")
        archive = download(asset_spec, paths.root / asset_spec["filename"])
    else:
        archive = asset_path.resolve()
        verify_file(archive, {"sha256": specs["asset"]["sha256"]})
    install_asset(paths, archive)

    from .scorer import reconstruct_reference_cells

    data = ad.read_h5ad(source, backed="r")
    try:
        expected = reconstruct_reference_cells(data.obs, paths.target_counts)
    finally:
        data.file.close()
    published = pd.read_csv(paths.reference_cells)
    if not expected.equals(published):
        raise ValueError("published reference cells do not reconstruct from H1")

    rebuild = True
    if paths.controls.exists() and paths.controls_manifest.exists():
        try:
            manifest = json.loads(paths.controls_manifest.read_text())
            rebuild = manifest["source"]["sha256"] != specs["sources"]["h1"]["sha256"]
            if not rebuild:
                verify_file(paths.controls, manifest["artifact"])
        except (KeyError, ValueError):
            rebuild = True
    if rebuild:
        build_controls(source, paths.controls, paths.controls_manifest, block_rows=512)

    result = check(paths)
    installation = {
        **result,
        "data_dir": str(paths.root),
        "source": {
            "path": str(source),
            "sha256": specs["sources"]["h1"]["sha256"],
            "retained": not (remove_source and managed_h1),
        },
        "packages": {
            "cell-eval2": version("cell-eval2"),
            "pdex": version("pdex"),
        },
    }
    paths.installation_manifest.write_text(
        json.dumps(installation, indent=2, sort_keys=True) + "\n"
    )
    if remove_source and managed_h1:
        source.unlink()
    return installation
