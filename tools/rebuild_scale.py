"""Build the reusable H1 score scale with resumable split-half DE."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import numpy as np
import polars as pl
from cell_eval2 import build_generic_baseline, compute_metrics
from cell_eval2.anchor import (
    _ANCHOR_SCHEMA,
    _SPLITS_SCHEMA,
    ANCHOR_CACHE_KEY,
    FULL_GATE_RAW,
    SPLIT_HALF_RAW,
    AnchorExpect,
    _bundle_from_obj,
    _bundle_to_obj,
    _derive_seeds,
    _inner_config,
    _lfc_nmae_names,
    anchor_cache_params,
    build_meta,
    validate_anchor,
    write_anchor,
)
from cell_eval2.baseline import config_digest
from cell_eval2.cache import MISS, CacheStore
from cell_eval2.catalog import resolve_metrics
from cell_eval2.ceiling import _disjoint_halves
from cell_eval2.de import prep_de_side, resolve_target_genes
from cell_eval2.lfc_nmae_ref import _assert_disjoint_controls, _nmae_ref_from_tables
from cell_eval2.real_bundle import build_real_bundle, read_real_bundle
from cell_eval2.run import _compute_de_side, aggregate_metrics, metric_output_names

from vcc_h1_eval.scorer import (
    SCORED_METRICS,
    evaluation_config,
    read_reference,
    sha256,
    write_json,
)

ROOT = Path.cwd()
H1_PATH = ROOT / "adata_Training.h5ad"
TARGET_COUNTS_PATH = ROOT / "pert_counts_Training.csv"
DE_PATH = ROOT / "reference_de.parquet"
REFERENCE_CELLS_PATH = ROOT / "reference_cells.csv"
BENCHMARK_MANIFEST_PATH = ROOT / "benchmark_manifest.json"
REFERENCE_CACHE = ROOT / "reference_cache"
SCALE_BUILD_DIR = ROOT / "scale_build"
SCALE_BUNDLE = ROOT / "scale"

BASE_SEED = 0
N_SPLITS = 5
PARTS_DIR = SCALE_BUILD_DIR / "anchor_parts"
ANCHOR_DIR = SCALE_BUILD_DIR / "anchor"


def build_config(reference_cache: Path, de_threads: int):
    return replace(
        evaluation_config(cache_real=reference_cache), num_threads=de_threads
    )


def build_identity(manifest: Path, config) -> str:
    payload = {
        "benchmark_manifest_sha256": sha256(manifest),
        "config_digest": config_digest(config, comparator="bulk_lognorm"),
        "base_seed": BASE_SEED,
        "n_splits": N_SPLITS,
        "cell_eval2_version": version("cell-eval2"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_full_de(path: Path, manifest_path: Path) -> pl.DataFrame:
    manifest = json.loads(manifest_path.read_text())
    if sha256(path) != manifest["inputs"]["de"]["sha256"]:
        raise ValueError("full-reference DE does not match the benchmark manifest")
    frame = pl.read_parquet(path)
    return frame.rename({"gene": "feature"}) if "gene" in frame.columns else frame


def _obs_digest(data: ad.AnnData) -> str:
    digest = hashlib.sha256()
    for name, label in zip(data.obs_names, data.obs["target_gene"], strict=True):
        digest.update(str(name).encode())
        digest.update(b"\0")
        digest.update(str(label).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _atomic_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.write_parquet(temporary)
    temporary.replace(path)


def _atomic_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _half_paths(split_index: int, half: str) -> tuple[Path, Path]:
    root = PARTS_DIR / f"split_{split_index}"
    return root / f"half_{half}_de.parquet", root / f"half_{half}.json"


def _valid_half(
    data: ad.AnnData,
    *,
    split_index: int,
    seed: int,
    half: str,
    identity: str,
) -> bool:
    table_path, meta_path = _half_paths(split_index, half)
    if not table_path.exists() or not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text())
    expected = {
        "identity": identity,
        "split_index": split_index,
        "seed": seed,
        "half": half,
        "n_cells": data.n_obs,
        "cells_sha256": _obs_digest(data),
        "de_sha256": sha256(table_path),
    }
    return all(meta.get(key) == value for key, value in expected.items())


def half_de(
    data: ad.AnnData,
    config,
    *,
    split_index: int,
    seed: int,
    half: str,
    identity: str,
) -> pl.DataFrame:
    table_path, meta_path = _half_paths(split_index, half)
    if _valid_half(
        data,
        split_index=split_index,
        seed=seed,
        half=half,
        identity=identity,
    ):
        print(f"Reusing split {split_index} half {half} DE", flush=True)
        return pl.read_parquet(table_path)

    side = "real" if half == "a" else "pred"
    print(f"Computing split {split_index} half {half} DE", flush=True)
    frame = _compute_de_side(
        data, cfg=_inner_config(config), fp=None, store=None, side=side
    )
    _atomic_parquet(frame, table_path)
    _atomic_json(
        {
            "identity": identity,
            "split_index": split_index,
            "seed": seed,
            "half": half,
            "n_cells": data.n_obs,
            "cells_sha256": _obs_digest(data),
            "de_sha256": sha256(table_path),
        },
        meta_path,
    )
    return frame


def _prep_de(frame: pl.DataFrame, config, name: str) -> pl.DataFrame:
    return prep_de_side(
        frame,
        name=name,
        sort_by=config.de.sort_by,
        nan_lfc_policy=config.de.nan_lfc_policy,
        min_abs_log2fc=config.de.min_abs_log2fc,
    )[0]


def score_split(
    half_a: ad.AnnData,
    half_b: ad.AnnData,
    de_full: pl.DataFrame,
    de_a: pl.DataFrame,
    de_b: pl.DataFrame,
    config,
    *,
    split_index: int,
    seed: int,
) -> pl.DataFrame:
    available, _ = resolve_metrics(config.metrics, version=config.version)
    metrics = list(available)
    expected = metric_output_names(config)
    tidy = compute_metrics(
        half_b,
        half_a,
        config=_inner_config(config),
        de_real=de_a,
        de_pred=de_b,
    )
    aggregate = aggregate_metrics(tidy, metrics=metrics)
    counts = (
        tidy.select("metric", "value")
        .filter(pl.col("value").is_not_null() & pl.col("value").is_not_nan())
        .group_by("metric")
        .len()
        .rename({"len": "n_perturbations"})
    )
    values = dict(zip(aggregate["metric"], aggregate["mean"], strict=True))
    cohort_sizes = dict(zip(counts["metric"], counts["n_perturbations"], strict=True))

    full = _prep_de(de_full, config, "real")
    split_a = _prep_de(de_a, config, "real")
    split_b = _prep_de(de_b, config, "pred")
    target_resolution = resolve_target_genes(
        full,
        sorted(full["target"].unique().to_list()),
        target_gene_map=config.target_gene_map,
    )
    reference = _nmae_ref_from_tables(
        full,
        split_a,
        split_b,
        p_adj_threshold=config.de.p_adj_threshold,
        min_gate_size=10,
        target_resolution=target_resolution,
    )
    if reference.height == 0:
        raise ValueError(f"split {split_index} produced no LFC-NMAE reference")
    for metric in _lfc_nmae_names(expected):
        values[metric] = float(np.mean(reference["nmae_ref_raw"].to_numpy()))
        cohort_sizes[metric] = reference.height

    absent = [
        metric
        for metric in expected
        if values.get(metric) is None or not np.isfinite(float(values[metric]))
    ]
    if absent:
        raise ValueError(f"split {split_index} produced no usable value for {absent}")
    return pl.DataFrame(
        {
            "split_index": [split_index] * len(expected),
            "seed": [seed] * len(expected),
            "metric": expected,
            "value": [float(values[metric]) for metric in expected],
            "n_perturbations": [
                None if cohort_sizes.get(metric) is None else int(cohort_sizes[metric])
                for metric in expected
            ],
        },
        schema=_SPLITS_SCHEMA,
    )


def _split_paths(split_index: int) -> tuple[Path, Path]:
    root = PARTS_DIR / f"split_{split_index}"
    return root / "metrics.parquet", root / "metrics.json"


def _valid_split(split_index: int, seed: int, identity: str) -> bool:
    frame_path, meta_path = _split_paths(split_index)
    if not frame_path.exists() or not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text())
    return (
        meta.get("identity") == identity
        and meta.get("split_index") == split_index
        and meta.get("seed") == seed
        and meta.get("metrics_sha256") == sha256(frame_path)
    )


def build_split(args) -> None:
    real, _ = read_reference(args)
    config = build_config(args.reference_cache, args.de_threads)
    identity = build_identity(args.manifest, config)
    seed = _derive_seeds(BASE_SEED, N_SPLITS)[args.split_index]
    half_a, half_b = _disjoint_halves(real, config.pert_col, config.control, seed)
    _assert_disjoint_controls(
        half_a, half_b, pert_col=config.pert_col, control=config.control
    )

    frame_path, meta_path = _split_paths(args.split_index)
    if _valid_split(args.split_index, seed, identity):
        print(f"Reusing completed split {args.split_index}", flush=True)
        return

    de_a = half_de(
        half_a,
        config,
        split_index=args.split_index,
        seed=seed,
        half="a",
        identity=identity,
    )
    de_b = half_de(
        half_b,
        config,
        split_index=args.split_index,
        seed=seed,
        half="b",
        identity=identity,
    )
    de_full = load_full_de(args.de, args.manifest)
    frame = score_split(
        half_a,
        half_b,
        de_full,
        de_a,
        de_b,
        config,
        split_index=args.split_index,
        seed=seed,
    )
    _atomic_parquet(frame, frame_path)
    _atomic_json(
        {
            "identity": identity,
            "split_index": args.split_index,
            "seed": seed,
            "metrics_sha256": sha256(frame_path),
        },
        meta_path,
    )
    print(f"Completed split {args.split_index}", flush=True)


def assemble_anchor(real: ad.AnnData, config, identity: str):
    seeds = _derive_seeds(BASE_SEED, N_SPLITS)
    frames = []
    for split_index, seed in enumerate(seeds):
        if not _valid_split(split_index, seed, identity):
            raise FileNotFoundError(f"split {split_index} is incomplete")
        frames.append(pl.read_parquet(_split_paths(split_index)[0]))
    splits = pl.concat(frames).sort("split_index", "metric")
    nmae_names = _lfc_nmae_names(metric_output_names(config))
    anchor = (
        splits.group_by("metric")
        .agg(
            replicate=pl.col("value").mean(),
            replicate_sd=pl.col("value").std(ddof=0),
            replicate_min=pl.col("value").min(),
            replicate_max=pl.col("value").max(),
            n_perturbations_min=pl.col("n_perturbations").min(),
            n_perturbations_max=pl.col("n_perturbations").max(),
        )
        .with_columns(
            estimator=pl.when(pl.col("metric").is_in(nmae_names))
            .then(pl.lit(FULL_GATE_RAW, dtype=pl.Utf8))
            .otherwise(pl.lit(SPLIT_HALF_RAW, dtype=pl.Utf8))
        )
        .select(list(_ANCHOR_SCHEMA))
        .sort("metric")
    )

    names, _ = resolve_metrics(config.metrics, version=config.version)
    metrics = metric_output_names(config)
    meta = build_meta(
        real_ad=real,
        cfg=config,
        names=list(names),
        base_seed=BASE_SEED,
        n_splits=N_SPLITS,
        seeds=seeds,
        metrics=metrics,
    )
    expect = AnchorExpect(
        fingerprint=meta["real_fingerprint"],
        semantic_identity=meta["semantic_identity"],
        version=meta["cell_eval2_version"],
        metrics=tuple(metrics),
    )
    validate_anchor(anchor, meta, expect, source="resumable H1 build")
    ANCHOR_DIR.mkdir(parents=True, exist_ok=True)
    write_anchor(str(ANCHOR_DIR), splits, anchor, meta=meta)

    fingerprint = meta["real_fingerprint"]
    params = anchor_cache_params(
        config,
        real,
        list(names),
        base_seed=BASE_SEED,
        n_splits=N_SPLITS,
        metrics=metrics,
    )
    store = CacheStore(str(config.cache_real))
    store.put(
        ANCHOR_CACHE_KEY,
        _bundle_to_obj(splits, anchor, meta),
        fingerprint=fingerprint,
        params=params,
        kind="json",
    )
    cached = store.get(
        ANCHOR_CACHE_KEY,
        fingerprint=fingerprint,
        params=params,
        kind="json",
    )
    if cached is MISS:
        raise RuntimeError("the assembled anchor was not written to Arc's cache")
    cached_anchor, cached_splits, cached_meta = _bundle_from_obj(cached)
    if not anchor.equals(cached_anchor) or not splits.equals(cached_splits):
        raise RuntimeError("Arc's cache did not round-trip the assembled anchor")
    if cached_meta != meta:
        raise RuntimeError("Arc's cache did not round-trip the anchor metadata")
    return anchor, splits, meta


def validate_baseline(real: ad.AnnData, config) -> Path:
    baseline_path = SCALE_BUILD_DIR / "generic_baseline.h5ad"
    meta_path = SCALE_BUILD_DIR / "baseline_checkpoint.json"
    if not baseline_path.exists() or not meta_path.exists():
        print("Building Arc generic-response baseline", flush=True)
        result = build_generic_baseline(real, config=config, save_pred=baseline_path)
        write_json(meta_path, result.meta)

    meta = json.loads(meta_path.read_text())
    expected_digest = config_digest(config, comparator="bulk_lognorm")
    if meta.get("config_digest") != expected_digest:
        raise ValueError("baseline configuration does not match this build")
    baseline = ad.read_h5ad(baseline_path, backed="r")
    try:
        if baseline.shape != real.shape:
            raise ValueError("baseline shape does not match the H1 reference")
        if not np.array_equal(baseline.var_names, real.var_names):
            raise ValueError("baseline gene axis does not match the H1 reference")
        if not np.array_equal(
            baseline.obs[config.pert_col].astype(str),
            real.obs[config.pert_col].astype(str),
        ):
            raise ValueError("baseline labels do not match the H1 reference")
    finally:
        baseline.file.close()
    return baseline_path


def publish_bundle(args, real: ad.AnnData, config, baseline_path: Path) -> None:
    baseline = ad.read_h5ad(baseline_path)
    build_real_bundle(
        real,
        baseline,
        config=config,
        outdir=str(args.scale_bundle),
        bundle_id="vcc2026-h1-126x400-pdex",
        base_seed=BASE_SEED,
        n_splits=N_SPLITS,
        force=args.force,
    )
    bundle = read_real_bundle(args.scale_bundle)
    manifest = bundle.manifest
    if manifest["rule_digest"] is None or manifest["rule_mismatches"]:
        raise ValueError(
            "Arc did not recognize the H1 bundle as competition-compatible"
        )
    if manifest["n_splits"] != N_SPLITS:
        raise ValueError("published bundle has the wrong split count")
    if set(manifest["members"]) != set(SCORED_METRICS):
        raise ValueError("published bundle has the wrong scored metric set")
    if manifest["derived_seeds"] != _derive_seeds(BASE_SEED, N_SPLITS):
        raise ValueError("published bundle has the wrong anchor seeds")
    print(f"Published H1 score bundle: {args.scale_bundle}", flush=True)


def orchestrate(args) -> None:
    real, _ = read_reference(args)
    config = build_config(args.reference_cache, args.de_threads)
    baseline_path = validate_baseline(real, config)
    identity = build_identity(args.manifest, config)

    common = [
        "--h1",
        str(args.h1),
        "--target-counts",
        str(args.target_counts),
        "--reference-cells",
        str(args.reference_cells),
        "--manifest",
        str(args.manifest),
        "--de",
        str(args.de),
        "--reference-cache",
        str(args.reference_cache),
        "--scale-bundle",
        str(args.scale_bundle),
        "--de-threads",
        str(args.de_threads),
    ]
    for split_index, seed in enumerate(_derive_seeds(BASE_SEED, N_SPLITS)):
        if _valid_split(split_index, seed, identity):
            print(f"Reusing completed split {split_index}", flush=True)
            continue
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "_split",
                str(split_index),
                *common,
            ],
            check=True,
        )

    real, _ = read_reference(args)
    assemble_anchor(real, config, identity)
    publish_bundle(args, real, config, baseline_path)


def common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--h1", type=Path, default=H1_PATH)
    parser.add_argument("--target-counts", type=Path, default=TARGET_COUNTS_PATH)
    parser.add_argument("--reference-cells", type=Path, default=REFERENCE_CELLS_PATH)
    parser.add_argument("--manifest", type=Path, default=BENCHMARK_MANIFEST_PATH)
    parser.add_argument("--de", type=Path, default=DE_PATH)
    parser.add_argument("--reference-cache", type=Path, default=REFERENCE_CACHE)
    parser.add_argument("--scale-bundle", type=Path, default=SCALE_BUNDLE)
    parser.add_argument("--de-threads", type=int, default=4)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    common_arguments(build)
    build.add_argument("--force", action="store_true")
    build.set_defaults(function=orchestrate)
    split = commands.add_parser("_split", help=argparse.SUPPRESS)
    split.add_argument("split_index", type=int, choices=range(N_SPLITS))
    common_arguments(split)
    split.set_defaults(function=build_split)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
