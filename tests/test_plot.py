from pathlib import Path

import numpy as np
import pandas as pd

from vcc_h1_eval.plot import (
    METRICS,
    build_profiles,
    parse_results,
    plot_metric_profiles,
)


def write_reference(tmp_path: Path) -> tuple[Path, Path]:
    targets = ["A", "B", "C", "D"]
    cells = pd.DataFrame(
        {
            "target_gene": np.repeat(targets, 2),
            "source_order": np.repeat(np.arange(4), 2),
        }
    )
    reference_cells = tmp_path / "reference_cells.csv"
    cells.to_csv(reference_cells, index=False)

    counts = {"A": 1, "B": 3, "C": 2, "D": 0}
    rows = []
    for target in targets:
        for gene in range(4):
            rows.append(
                {
                    "target": target,
                    "p_adj": 0.01 if gene < counts[target] else 0.5,
                }
            )
    reference_de = tmp_path / "reference_de.parquet"
    pd.DataFrame(rows).to_parquet(reference_de)
    return reference_cells, reference_de


def write_result(tmp_path: Path, name: str, offset: float) -> Path:
    directory = tmp_path / name
    directory.mkdir()
    targets = ["A", "B", "C", "D"]
    rows = []
    values = {
        "pds_cosine": np.array([0.2, 0.4, 0.6, 0.8]) + offset,
        "expr_mse_unbiased_capped": np.array([2.0, 4.0, 6.0, 8.0]) + offset,
        "expr_distance_unbiased": np.array([4.0, 5.0, 8.0, 10.0]),
        "de_wilcoxon_lfc_nmae": np.array([np.nan, 0.9, 0.8, 0.7]) + offset,
        "de_wilcoxon_direction_fidelity_yield_raw": np.array([0.0, 0.1, 0.2, 0.3]),
        "de_wilcoxon_direction_reach_raw": np.array([0.1, 0.2, 0.3, 0.4]),
        "de_wilcoxon_sig_jaccard": np.array([0.0, 0.01, 0.02, 0.03]),
    }
    for metric, metric_values in values.items():
        for target, value in zip(targets, metric_values, strict=True):
            if pd.notna(value):
                rows.append({"target_gene": target, "metric": metric, "value": value})
    pd.DataFrame(rows).to_csv(directory / "per_target.csv", index=False)

    aggregates = []
    for metric, _, _ in METRICS:
        if metric == "expr_mse_unbiased_capped_norm":
            raw = values["expr_mse_unbiased_capped"].sum() / values[
                "expr_distance_unbiased"
            ].sum()
        else:
            raw = np.nanmean(values[metric])
        aggregates.append({"metric": metric, "raw_value": raw})
    pd.DataFrame(aggregates).to_csv(directory / "aggregates.csv", index=False)
    return directory


def test_profiles_follow_de_order_and_reproduce_aggregates(tmp_path: Path) -> None:
    reference_cells, reference_de = write_reference(tmp_path)
    result = write_result(tmp_path, "control", 0.0)
    profiles = build_profiles(
        {"control": result}, reference_cells, reference_de, ["B", "A", "C"]
    )

    pds = profiles.query("metric == 'pds_cosine'")
    assert pds["target_gene"].tolist() == ["A", "C", "B"]
    assert pds["n_de"].tolist() == [1, 2, 3]

    complete = build_profiles({"control": result}, reference_cells, reference_de)
    observed = complete.groupby("metric")["aggregate_contribution"].sum()
    expected = pd.read_csv(result / "aggregates.csv").set_index("metric")["raw_value"]
    assert np.allclose(observed.loc[expected.index], expected)
    lfc_a = complete.query(
        "metric == 'de_wilcoxon_lfc_nmae' and target_gene == 'A'"
    ).iloc[0]
    assert not lfc_a["eligible"] and pd.isna(lfc_a["value"])


def test_plot_writes_image_and_companion_table(tmp_path: Path) -> None:
    reference_cells, reference_de = write_reference(tmp_path)
    control = write_result(tmp_path, "control", 0.0)
    shifted = write_result(tmp_path, "global-shift", 0.02)
    output = tmp_path / "profiles.png"

    table = plot_metric_profiles(
        {"control": control, "global shift": shifted},
        reference_cells,
        reference_de,
        output,
    )

    assert output.stat().st_size > 10_000
    assert output.with_suffix(".csv").is_file()
    assert table["model"].drop_duplicates().tolist() == ["control", "global shift"]
    assert table["target_gene"].nunique() == 4


def test_result_labels_are_explicit_or_derived() -> None:
    assert parse_results(["control=/tmp/a", "/tmp/global-shift"]) == {
        "control": Path("/tmp/a"),
        "global-shift": Path("/tmp/global-shift"),
    }
