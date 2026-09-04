import os
import zlib
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import pytest
from cell_eval2 import (
    EvalConfig,
    aggregate_metrics_wide,
    compute_metrics,
    score_metrics,
)
from cell_eval2.anchor import compute_replicate_anchor
from cell_eval2.cache import CacheStore, fingerprint_adata
from cell_eval2.catalog import CATALOG, resolve_metrics
from cell_eval2.ceiling import _disjoint_halves
from cell_eval2.de import prepare_de
from cell_eval2.de_compute import compute_de
from cell_eval2.prep import pseudobulk_bulk_lognorm_with_moments
from cell_eval2.real_bundle import check_submission, read_real_bundle
from cell_eval2.run import (
    _compute_de_side,
    dispatch_anndata_metrics,
    dispatch_de_metrics,
    metric_output_names,
)
from cell_eval2.score import _reference_column, _replicate_entries
from cell_eval2.scoring import score_one
from scipy import sparse

from tools import rebuild_scale as scale_builder
from vcc_h1_eval.bounded import (
    RowSource,
    compute_de_bounded,
    prediction_statistics,
)
from vcc_h1_eval.scorer import (
    CELL_EVAL2_COMMIT,
    CELLS_PER_TARGET,
    CONTROL,
    N_CONTROLS,
    N_TARGETS,
    SCORED_METRICS,
    _CacheSeeder,
    _sample_target_rows,
    assemble_reference,
    augment_prediction,
    evaluation_config,
    reconstruct_reference_cells,
    sample_control_rows,
    scoring_meta,
    tidy_aggregate,
    validate_de_artifact,
    validate_prediction,
)

FULL_DATA = Path(os.environ.get("VCC_H1_DATA", "/nonexistent"))
H1_PATH = FULL_DATA / "adata_Training.h5ad"
DE_PATH = FULL_DATA / "reference_de.parquet"
DE_SUMMARY_PATH = FULL_DATA / "reference_de_summary.csv"
SCALE_BUNDLE = FULL_DATA / "scale"


def synthetic_contract() -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = [f"g{index:03d}" for index in range(150)]
    labels = [CONTROL] * N_CONTROLS
    names = [f"control-{index}" for index in range(N_CONTROLS)]
    counts = []
    for index, target in enumerate(targets):
        n_cells = 401 if index < N_TARGETS else 399
        labels.extend([target] * n_cells)
        names.extend(f"{target}-{cell}" for cell in range(n_cells))
        counts.append(n_cells)
    obs = pd.DataFrame({"target_gene": labels}, index=names)
    target_counts = pd.DataFrame({"target_gene": targets, "n_cells": counts})
    return obs, target_counts


def synthetic_scoring_pair() -> tuple[ad.AnnData, ad.AnnData]:
    rng = np.random.default_rng(11)
    targets = ["A", "B", "C"]
    genes = targets + [f"x{index}" for index in range(27)]
    labels = np.repeat([CONTROL, *targets], 60)
    real_counts = rng.poisson(30, size=(len(labels), len(genes)))
    for target_index, target in enumerate(targets):
        mask = labels == target
        real_counts[mask, target_index] = rng.poisson(2, mask.sum())
        real_counts[np.ix_(mask, np.arange(3, 25))] += 30 + 5 * target_index
    pred_counts = real_counts.copy()
    pred_counts[labels != CONTROL, 3:25] += rng.poisson(
        2, size=((labels != CONTROL).sum(), 22)
    )
    obs = pd.DataFrame({"target_gene": labels})
    var = pd.DataFrame(index=genes)
    real = ad.AnnData(sparse.csr_matrix(real_counts), obs=obs.copy(), var=var.copy())
    pred = ad.AnnData(sparse.csr_matrix(pred_counts), obs=obs.copy(), var=var.copy())
    return pred, real


def test_config_changes_only_h1_cpu_pdex_and_cache(tmp_path: Path) -> None:
    expected = EvalConfig.from_preset("vcc2026")
    expected = replace(
        expected,
        pert_col="target_gene",
        device="cpu",
        cache_real=str(tmp_path),
        de=replace(expected.de, backend="pdex"),
    )
    assert asdict(evaluation_config(cache_real=tmp_path)) == asdict(expected)
    assert CELL_EVAL2_COMMIT == "5e64833518a6603a0301cbe28185d49c30f4a986"


def test_crc32_split_is_exact_unique_and_stable(tmp_path: Path) -> None:
    obs, counts = synthetic_contract()
    counts_path = tmp_path / "counts.csv"
    counts.to_csv(counts_path, index=False)

    first = reconstruct_reference_cells(obs, counts_path)
    second = reconstruct_reference_cells(obs, counts_path)

    assert first.equals(second)
    assert len(first) == N_TARGETS * CELLS_PER_TARGET
    assert first["source_row"].is_unique
    assert first.groupby("target_gene", sort=False).size().eq(CELLS_PER_TARGET).all()
    assert (
        first["target_gene"].drop_duplicates().tolist()
        == counts.target_gene[:N_TARGETS].tolist()
    )
    for target, group in first.groupby("target_gene", sort=False):
        assert group["source_row"].is_monotonic_increasing
        available = np.flatnonzero(obs.target_gene.to_numpy() == target)
        expected = np.sort(
            available[
                np.random.default_rng(zlib.crc32(target.encode())).choice(
                    len(available), CELLS_PER_TARGET, replace=False
                )
            ]
        )
        assert np.array_equal(group.source_row, expected)


def test_reference_uses_all_h1_controls(tmp_path: Path) -> None:
    obs, counts = synthetic_contract()
    counts_path = tmp_path / "counts.csv"
    counts.to_csv(counts_path, index=False)
    cells = reconstruct_reference_cells(obs, counts_path)
    matrix = sparse.csr_matrix(np.ones((len(obs), 3), dtype=np.int16))
    data = ad.AnnData(matrix, obs=obs, var=pd.DataFrame(index=["a", "b", "c"]))

    real = assemble_reference(data, cells)

    assert real.shape == (N_TARGETS * CELLS_PER_TARGET + N_CONTROLS, 3)
    assert real.obs.target_gene.value_counts()[CONTROL] == N_CONTROLS
    assert real.var_names.tolist() == ["a", "b", "c"]
    assert np.equal(real.X.data, np.floor(real.X.data)).all()


def test_de_artifact_schema_universe_and_summary(tmp_path: Path) -> None:
    targets = ["a", "b"]
    genes = ["a", "b", "x"]
    rows = []
    for target in targets:
        for gene in genes:
            lfc = -1.0 if target == gene else 0.1
            rows.append(
                {
                    "target": target,
                    "gene": gene,
                    "log2_fold_change": lfc,
                    "p_value": 0.01,
                    "p_adj": 0.03,
                }
            )
    de_path = tmp_path / "de.parquet"
    pl.DataFrame(rows).write_parquet(de_path)
    summary = pd.DataFrame(
        {
            "target": targets,
            "n_cells": [400, 400],
            "n_de": [3, 3],
            "n_up": [2, 2],
            "n_down": [1, 1],
            "target_log2_fold_change": [-1.0, -1.0],
            "target_p_adj": [0.03, 0.03],
        }
    )
    summary_path = tmp_path / "summary.csv"
    summary.to_csv(summary_path, index=False)

    validated = validate_de_artifact(de_path, summary_path, targets, genes)

    assert validated.columns == ["target", "feature", "log2_fold_change", "p_adj"]
    assert validated.shape == (6, 4)


def test_seeded_real_de_cache_does_not_execute_de(tmp_path: Path, monkeypatch) -> None:
    labels = np.repeat([CONTROL, "a"], 12)
    matrix = sparse.csr_matrix(np.arange(96).reshape(24, 4) % 7 + 1)
    real = ad.AnnData(
        matrix,
        obs=pd.DataFrame({"target_gene": labels}),
        var=pd.DataFrame(index=["a", "b", "c", "d"]),
    )
    config = evaluation_config(cache_real=tmp_path)
    de = compute_de(
        real,
        backend="pdex",
        groupby="target_gene",
        reference=CONTROL,
        mean_calc="arithmetic",
        epsilon=1e-9,
        input_type="counts",
        target_sum=1_000_000,
        filter_gene_min_cpm_cell=5,
        threads=1,
    )
    fp = fingerprint_adata(real, pert_col="target_gene", strict=True)
    _compute_de_side(
        real, cfg=config, fp=fp, store=_CacheSeeder(tmp_path, de), side="real"
    )

    def fail(*args, **kwargs):
        raise AssertionError("DE was recomputed")

    monkeypatch.setattr("cell_eval2.de_compute.compute_de", fail)
    cached = _compute_de_side(
        real, cfg=config, fp=fp, store=CacheStore(str(tmp_path)), side="real"
    )
    assert cached.sort(["target", "feature"]).equals(de.sort(["target", "feature"]))


def test_prediction_contract_and_real_control_augmentation() -> None:
    targets = ["a", "b"]
    genes = ["a", "b", "x"]
    pred = ad.AnnData(
        sparse.csr_matrix(np.ones((8, 3), dtype=np.int16)),
        obs=pd.DataFrame({"target_gene": np.repeat(targets, 4)}),
        var=pd.DataFrame(index=genes),
    )
    real = ad.AnnData(
        sparse.csr_matrix(np.ones((14, 3), dtype=np.int16)),
        obs=pd.DataFrame({"target_gene": ["a"] * 4 + ["b"] * 4 + [CONTROL] * 6}),
        var=pd.DataFrame(index=genes),
    )

    validate_prediction(pred, targets, genes, cells_per_target=4)
    augmented = augment_prediction(pred, real)

    assert augmented.obs.target_gene.value_counts().to_dict() == {
        "a": 4,
        "b": 4,
        CONTROL: 6,
    }
    assert augmented.var_names.tolist() == genes


def test_control_baseline_is_size_matched_and_deterministic() -> None:
    controls = np.arange(12)
    targets = ["A", "B"]

    first_rows, first_labels = sample_control_rows(controls, targets, cells_per_target=4)
    second_rows, second_labels = sample_control_rows(controls, targets, cells_per_target=4)

    assert pd.Series(first_labels).value_counts().reindex(targets).tolist() == [4, 4]
    assert np.array_equal(first_rows, second_rows)
    assert np.array_equal(first_labels, second_labels)
    for target in targets:
        rows = first_rows[first_labels == target]
        assert len(np.unique(rows)) == 4


def test_bounded_pdex_matches_arc_compute_de(tmp_path: Path) -> None:
    rng = np.random.default_rng(22)
    genes = [f"g{index}" for index in range(7)]
    targets = ["A", "B"]
    pred_labels = np.repeat(targets, 5)
    control_labels = np.repeat(CONTROL, 12)
    pred = ad.AnnData(
        sparse.csr_matrix(rng.poisson(20, size=(10, len(genes))).astype(np.float32)),
        obs=pd.DataFrame({"target_gene": pred_labels}),
        var=pd.DataFrame(index=genes),
    )
    controls = ad.AnnData(
        sparse.csr_matrix(rng.poisson(20, size=(12, len(genes))).astype(np.float32)),
        obs=pd.DataFrame({"target_gene": control_labels}),
        var=pd.DataFrame(index=genes),
    )
    combined = ad.concat([pred, controls], axis=0)
    direct = compute_de(
        combined,
        backend="pdex",
        groupby="target_gene",
        reference=CONTROL,
        mean_calc="arithmetic",
        epsilon=1e-9,
        input_type="counts",
        target_sum=1e6,
        clip_value=None,
        filter_gene_min_cpm_cell=5,
        fdr_scope="per_pert",
        threads=2,
    ).sort(["target", "feature"])
    bounded = compute_de_bounded(
        RowSource(pred, np.arange(pred.n_obs), pred_labels),
        RowSource(controls, np.arange(controls.n_obs), control_labels),
        targets,
        CONTROL,
        genes,
        tmp_path / "chunks",
        "synthetic-parity",
        gene_chunk=3,
        threads=2,
        epsilon=1e-9,
    ).sort(["target", "feature"])

    assert bounded.select("target", "feature").equals(
        direct.select("target", "feature")
    )
    for column in ("log2_fold_change", "p_value", "p_adj"):
        assert np.allclose(
            bounded[column], direct[column], rtol=1e-12, atol=1e-14, equal_nan=True
        )


def test_bounded_prediction_statistics_match_augmented_anndata() -> None:
    rng = np.random.default_rng(23)
    genes = [f"g{index}" for index in range(9)]
    targets = ["A", "B"]
    pred_labels = np.repeat(targets, 5)
    control_labels = np.repeat(CONTROL, 8)
    pred = ad.AnnData(
        sparse.csr_matrix(rng.poisson(12, size=(10, len(genes))).astype(np.float32)),
        obs=pd.DataFrame({"target_gene": pred_labels}),
        var=pd.DataFrame(index=genes),
    )
    controls = ad.AnnData(
        sparse.csr_matrix(rng.poisson(12, size=(8, len(genes))).astype(np.float32)),
        obs=pd.DataFrame({"target_gene": control_labels}),
        var=pd.DataFrame(index=genes),
    )
    real = ad.concat([pred, controls], axis=0)
    real_perts, real_means, real_moments = pseudobulk_bulk_lognorm_with_moments(
        real, "target_gene", bulk_target_sum=50_000
    )
    bulks, moments = prediction_statistics(
        RowSource(pred, np.arange(pred.n_obs), pred_labels),
        targets,
        CONTROL,
        {"bulk_lognorm": (real_perts, real_means)},
        {"bulk_lognorm": real_moments},
        bulk_target_sum=50_000,
    )
    direct_perts, direct_means, direct_moments = pseudobulk_bulk_lognorm_with_moments(
        real, "target_gene", bulk_target_sum=50_000
    )

    assert np.array_equal(bulks["bulk_lognorm"][0], direct_perts)
    assert np.array_equal(bulks["bulk_lognorm"][1], direct_means)
    bounded_moments = moments["bulk_lognorm"]
    assert np.array_equal(bounded_moments.counts, direct_moments.counts)
    assert np.array_equal(bounded_moments.sumsq, direct_moments.sumsq)
    assert np.allclose(bounded_moments.jk, direct_moments.jk, rtol=1e-12, atol=1e-14)


def test_bounded_raw_metrics_match_compute_metrics(tmp_path: Path) -> None:
    prediction, real = synthetic_scoring_pair()
    config = replace(evaluation_config(cache_real=None), num_threads=2)
    direct = compute_metrics(prediction, real, config=config).sort(
        ["perturbation", "metric"]
    )
    targets = ["A", "B", "C"]
    pred_mask = prediction.obs["target_gene"].astype(str).ne(CONTROL).to_numpy()
    control_mask = real.obs["target_gene"].astype(str).eq(CONTROL).to_numpy()
    pred_source = RowSource(
        prediction,
        np.flatnonzero(pred_mask),
        prediction.obs.loc[pred_mask, "target_gene"].astype(str).to_numpy(),
    )
    control_source = RowSource(
        real,
        np.flatnonzero(control_mask),
        np.repeat(CONTROL, control_mask.sum()),
    )
    real_perts, real_means, real_moments = pseudobulk_bulk_lognorm_with_moments(
        real, "target_gene", bulk_target_sum=config.bulk_target_sum
    )
    real_bulks = {"bulk_lognorm": (real_perts, real_means)}
    real_moments_by_norm = {"bulk_lognorm": real_moments}
    pred_bulks, pred_moments = prediction_statistics(
        pred_source,
        targets,
        CONTROL,
        real_bulks,
        real_moments_by_norm,
        bulk_target_sum=config.bulk_target_sum,
    )
    pred_de = compute_de_bounded(
        pred_source,
        control_source,
        targets,
        CONTROL,
        prediction.var_names.astype(str).tolist(),
        tmp_path / "chunks",
        "metric-parity",
        gene_chunk=11,
        threads=2,
        epsilon=config.de.epsilon,
    )
    real_de = compute_de(
        real,
        backend="pdex",
        groupby="target_gene",
        reference=CONTROL,
        mean_calc=config.de.mean_calc,
        epsilon=config.de.epsilon,
        input_type="counts",
        target_sum=config.target_sum,
        clip_value=config.de.clip_value,
        filter_gene_min_cpm_cell=config.filter.filter_gene_min_cpm_cell,
        fdr_scope=config.de.fdr_scope,
        threads=2,
    )
    names = list(resolve_metrics(config.metrics, version=config.version)[0])
    rows = dispatch_anndata_metrics(
        names,
        pred_bulks,
        real_bulks,
        prediction.var_names.astype(str).to_numpy(),
        config,
        comparator="bulk_lognorm",
        pred_moments=pred_moments,
        real_moments=real_moments_by_norm,
        driver="test bounded scorer",
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
    bounded = pl.DataFrame(rows).sort(["perturbation", "metric"])

    assert bounded.select("perturbation", "metric").equals(
        direct.select("perturbation", "metric")
    )
    assert np.allclose(
        bounded["value"], direct["value"], rtol=1e-11, atol=1e-12, equal_nan=True
    )


@pytest.mark.skipif(not SCALE_BUNDLE.exists(), reason="full benchmark bundle not supplied")
def test_bounded_run_metadata_enrols_with_h1_bundle() -> None:
    config = evaluation_config(cache_real=None)
    meta = scoring_meta(SimpleNamespace(scale_bundle=SCALE_BUNDLE), config)
    bundle = read_real_bundle(SCALE_BUNDLE)

    assert check_submission(bundle.manifest, meta) == []


def test_raw_and_aggregate_metrics_equal_direct_cell_eval2() -> None:
    prediction, real = synthetic_scoring_pair()
    config = evaluation_config(cache_real=None)

    raw = compute_metrics(prediction, real, config=config)
    wide, tidy = tidy_aggregate(raw, config)
    direct = aggregate_metrics_wide(raw, metrics=metric_output_names(config))

    assert wide.equals(direct)
    assert tidy.height == 10
    assert set(tidy.filter(pl.col("scored"))["metric"].to_list()) == set(SCORED_METRICS)
    assert np.isfinite(tidy["raw_value"].to_numpy()).all()


def test_half_de_checkpoint_is_reused(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(scale_builder, "PARTS_DIR", tmp_path)
    labels = np.repeat([CONTROL, "a"], 4)
    data = ad.AnnData(
        sparse.csr_matrix(np.ones((8, 3), dtype=np.int16)),
        obs=pd.DataFrame({"target_gene": labels}),
        var=pd.DataFrame(index=["a", "b", "c"]),
    )
    expected = pl.DataFrame(
        {
            "target": ["a"],
            "feature": ["a"],
            "log2_fold_change": [-1.0],
            "p_value": [0.01],
            "p_adj": [0.02],
        }
    )
    calls = 0

    def fake_de(*args, **kwargs):
        nonlocal calls
        calls += 1
        return expected

    monkeypatch.setattr(scale_builder, "_compute_de_side", fake_de)
    config = replace(evaluation_config(cache_real=None), num_threads=1)
    first = scale_builder.half_de(
        data,
        config,
        split_index=0,
        seed=7,
        half="a",
        identity="test",
    )
    second = scale_builder.half_de(
        data,
        config,
        split_index=0,
        seed=7,
        half="a",
        identity="test",
    )
    table_path, _ = scale_builder._half_paths(0, "a")
    table_path.write_bytes(b"corrupt")
    third = scale_builder.half_de(
        data,
        config,
        split_index=0,
        seed=7,
        half="a",
        identity="test",
    )

    assert calls == 2
    assert first.equals(expected)
    assert second.equals(expected)
    assert third.equals(expected)


def test_resumable_split_matches_arc_anchor() -> None:
    _, real = synthetic_scoring_pair()
    config = replace(evaluation_config(cache_real=None), num_threads=1)
    seed = scale_builder._derive_seeds(0, 1)[0]

    direct, _ = compute_replicate_anchor(real, config=config, base_seed=0, n_splits=1)
    half_a, half_b = _disjoint_halves(real, config.pert_col, config.control, seed)
    inner = scale_builder._inner_config(config)
    de_full = _compute_de_side(real, cfg=config, fp=None, store=None, side="real")
    de_a = _compute_de_side(half_a, cfg=inner, fp=None, store=None, side="real")
    de_b = _compute_de_side(half_b, cfg=inner, fp=None, store=None, side="pred")
    resumed = scale_builder.score_split(
        half_a,
        half_b,
        de_full,
        de_a,
        de_b,
        config,
        split_index=0,
        seed=seed,
    )

    direct = direct.sort("metric")
    resumed = resumed.sort("metric")
    assert direct.select("split_index", "seed", "metric", "n_perturbations").equals(
        resumed.select("split_index", "seed", "metric", "n_perturbations")
    )
    assert np.allclose(direct["value"], resumed["value"], rtol=0, atol=2e-15)


def test_assembled_anchor_round_trips_through_arc_cache(
    tmp_path: Path, monkeypatch
) -> None:
    parts = tmp_path / "parts"
    anchor_dir = tmp_path / "anchor"
    monkeypatch.setattr(scale_builder, "PARTS_DIR", parts)
    monkeypatch.setattr(scale_builder, "ANCHOR_DIR", anchor_dir)
    _, real = synthetic_scoring_pair()
    config = replace(evaluation_config(cache_real=tmp_path / "cache"), num_threads=1)
    metrics = metric_output_names(config)
    seeds = scale_builder._derive_seeds(0, scale_builder.N_SPLITS)

    for split_index, seed in enumerate(seeds):
        frame = pl.DataFrame(
            {
                "split_index": [split_index] * len(metrics),
                "seed": [seed] * len(metrics),
                "metric": metrics,
                "value": [0.1 * (split_index + 1)] * len(metrics),
                "n_perturbations": [3] * len(metrics),
            },
            schema=scale_builder._SPLITS_SCHEMA,
        )
        frame_path, meta_path = scale_builder._split_paths(split_index)
        scale_builder._atomic_parquet(frame, frame_path)
        scale_builder.write_json(
            meta_path,
            {
                "identity": "test",
                "split_index": split_index,
                "seed": seed,
                "metrics_sha256": scale_builder.sha256(frame_path),
            },
        )

    anchor, splits, meta = scale_builder.assemble_anchor(real, config, "test")

    assert splits.height == scale_builder.N_SPLITS * len(metrics)
    assert np.allclose(anchor["replicate"], 0.3)
    assert meta["n_splits"] == scale_builder.N_SPLITS
    assert (anchor_dir / "anchor_agg.parquet").exists()
    assert any((tmp_path / "cache").glob("replicate_anchor-*.json"))


def test_official_scaler_maps_baseline_to_zero_and_uses_six_members() -> None:
    metrics = metric_output_names(evaluation_config(cache_real=None))
    values = {
        metric: 0.8 if CATALOG[metric].scoring.direction == "lower" else 0.2
        for metric in metrics
    }
    frame = pl.DataFrame(
        {"statistic": ["mean"], **{metric: [value] for metric, value in values.items()}}
    )
    scored = score_metrics(frame, results_base=frame)
    members = scored.filter(pl.col("metric") != "avg_score")
    assert members.height == len(SCORED_METRICS)
    assert np.allclose(members["from_baseline"], 0)
    assert scored.filter(pl.col("metric") == "avg_score")["from_baseline"].item() == 0


def test_replicate_anchor_maps_to_one_and_mse_uses_arc_cap() -> None:
    metrics = metric_output_names(evaluation_config(cache_real=None))
    baseline = {
        metric: 0.8 if CATALOG[metric].scoring.direction == "lower" else 0.2
        for metric in metrics
    }
    replicate = {
        metric: 0.2 if CATALOG[metric].scoring.direction == "lower" else 0.8
        for metric in metrics
    }
    anchor = pl.DataFrame(
        {"metric": metrics, "replicate": [replicate[metric] for metric in metrics]}
    )
    entries = _replicate_entries(baseline, anchor)
    baseline_frame = pl.DataFrame(
        {"statistic": ["mean"], **{metric: [baseline[metric]] for metric in metrics}}
    )
    row_names = score_metrics(baseline_frame, results_base=baseline_frame)[
        "metric"
    ].to_list()
    scaled = _reference_column(
        row_names,
        [replicate[metric] for metric in metrics],
        metrics,
        entries,
        column="from_replicate",
        label="test anchor",
    )
    values = dict(zip(row_names, scaled.to_list()))
    assert all(values[metric] == 1 for metric in SCORED_METRICS)
    assert values["avg_score"] == 1

    mse_entry = entries["expr_mse_unbiased_capped_norm"]
    assert score_one(1.2, mse_entry.base, mse_entry.scoring) == 0


def frozen_artifact_complete() -> bool:
    if not DE_PATH.exists() or not DE_SUMMARY_PATH.exists():
        return False
    return (
        pl.scan_parquet(DE_PATH).select(pl.col("target").n_unique()).collect().item()
        == N_TARGETS
    )


@pytest.mark.skipif(
    not frozen_artifact_complete(),
    reason="the separately-run frozen DE artifact is not complete",
)
def test_frozen_acat2_de_matches_cell_eval2() -> None:
    data = ad.read_h5ad(H1_PATH, backed="r")
    try:
        available = np.flatnonzero(
            data.obs.target_gene.astype(str).to_numpy() == "ACAT2"
        )
        rows = _sample_target_rows(available, "ACAT2")
        controls = np.flatnonzero(
            data.obs.target_gene.astype(str).to_numpy() == CONTROL
        )
        selected = np.concatenate([rows, controls])
        real = ad.AnnData(
            data.X[selected].tocsr(),
            obs=pd.DataFrame(
                {"target_gene": ["ACAT2"] * len(rows) + [CONTROL] * len(controls)}
            ),
            var=pd.DataFrame(index=data.var_names.copy()),
        )
    finally:
        data.file.close()
    direct = compute_de(
        real,
        backend="pdex",
        groupby="target_gene",
        reference=CONTROL,
        mean_calc="arithmetic",
        epsilon=1e-9,
        input_type="counts",
        target_sum=1_000_000,
        clip_value=None,
        fdr_scope="per_pert",
        filter_gene_min_cpm_cell=5,
        threads=-1,
    ).filter(pl.col("target") == "ACAT2")
    frozen = (
        pl.read_parquet(DE_PATH)
        .filter(pl.col("target") == "ACAT2")
        .rename({"gene": "feature"})
    )
    assert direct["feature"].to_list() == frozen["feature"].to_list()
    for column in ("log2_fold_change", "p_value", "p_adj"):
        assert np.allclose(direct[column], frozen[column], rtol=0, atol=2e-15)
