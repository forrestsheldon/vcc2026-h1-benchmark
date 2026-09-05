from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = (
    ("pds_cosine", "Perturbation discrimination (PDS) ↑", "higher"),
    ("expr_mse_unbiased_capped_norm", "Normalized expression MSE ↓", "lower"),
    ("de_wilcoxon_lfc_nmae", "DE log-fold-change NMAE ↓", "lower"),
    (
        "de_wilcoxon_direction_fidelity_yield_raw",
        "DE direction fidelity × yield ↑",
        "higher",
    ),
    ("de_wilcoxon_direction_reach_raw", "DE direction reach ↑", "higher"),
    ("de_wilcoxon_sig_jaccard", "Significant-gene Jaccard ↑", "higher"),
)
MSE_NUMERATOR = "expr_mse_unbiased_capped"
MSE_DENOMINATOR = "expr_distance_unbiased"
COLORS = ("#256abf", "#eb6834", "#16856b", "#8b5aa5", "#c49a15", "#555f6d")
STRENGTH_LINES = "#c3cbd6"
INK = "#131820"
MUTED = "#555f6d"


def parse_results(values: list[str]) -> dict[str, Path]:
    results = {}
    for value in values:
        if "=" in value:
            label, path = value.split("=", 1)
        else:
            path = value
            label = Path(path).name
        if not label or label in results:
            raise ValueError("result labels must be non-empty and unique")
        results[label] = Path(path)
    return results


def reference_de_path(benchmark_dir: Path) -> Path:
    manifest = json.loads((benchmark_dir / "benchmark_manifest.json").read_text())
    names = manifest["artifacts"]["reference_cache_files"]
    name = next(name for name in names if "de_wilcoxon_table" in name)
    return benchmark_dir / "reference_cache" / name


def target_strength(reference_cells: Path, reference_de: Path) -> pd.DataFrame:
    order = (
        pd.read_csv(reference_cells, usecols=["target_gene", "source_order"])
        .drop_duplicates("target_gene")
        .sort_values("source_order")
    )
    de = pd.read_parquet(reference_de, columns=["target", "p_adj"])
    counts = de.loc[de["p_adj"] < 0.05].groupby("target").size()
    order["n_de"] = order["target_gene"].map(counts).fillna(0).astype(int)
    return order.sort_values(["n_de", "source_order"]).reset_index(drop=True)


def _result_profiles(label: str, directory: Path) -> pd.DataFrame:
    long = pd.read_csv(directory / "per_target.csv")
    aggregates = pd.read_csv(directory / "aggregates.csv").set_index("metric")
    wide = long.pivot(index="target_gene", columns="metric", values="value")
    required = {name for name, _, _ in METRICS if name != "expr_mse_unbiased_capped_norm"}
    required |= {MSE_NUMERATOR, MSE_DENOMINATOR}
    missing = required - set(wide.columns)
    if missing:
        raise ValueError(f"{directory} is missing per-target metrics: {sorted(missing)}")

    rows = []
    for metric, _, direction in METRICS:
        if metric == "expr_mse_unbiased_capped_norm":
            numerator = wide[MSE_NUMERATOR]
            denominator = wide[MSE_DENOMINATOR]
            values = numerator / denominator
            contributions = numerator / denominator.sum()
        else:
            values = wide[metric]
            contributions = values / values.count()
        expected = float(aggregates.loc[metric, "raw_value"])
        if not np.isclose(contributions.sum(), expected, atol=1e-12):
            raise ValueError(f"{directory} {metric} does not reproduce its aggregate")
        for target, value in values.items():
            rows.append(
                {
                    "model": label,
                    "target_gene": target,
                    "metric": metric,
                    "value": value,
                    "aggregate_contribution": contributions[target],
                    "direction": direction,
                    "eligible": bool(pd.notna(value)),
                }
            )
    return pd.DataFrame(rows)


def build_profiles(
    results: dict[str, Path],
    reference_cells: Path,
    reference_de: Path,
    targets: list[str] | None = None,
) -> pd.DataFrame:
    strength = target_strength(reference_cells, reference_de)
    available = set(strength["target_gene"])
    selected = strength if targets is None else strength[strength["target_gene"].isin(targets)]
    if targets is not None and set(targets) != set(selected["target_gene"]):
        raise ValueError(f"unknown targets: {sorted(set(targets) - available)}")
    if selected.empty:
        raise ValueError("select at least one target")

    profiles = pd.concat(
        [_result_profiles(label, directory) for label, directory in results.items()],
        ignore_index=True,
    )
    profiles = profiles.merge(selected, on="target_gene", how="inner")
    model_order = {name: index for index, name in enumerate(results)}
    metric_order = {name: index for index, (name, _, _) in enumerate(METRICS)}
    profiles["model_order"] = profiles["model"].map(model_order)
    profiles["metric_order"] = profiles["metric"].map(metric_order)
    return (
        profiles.sort_values(["metric_order", "model_order", "n_de", "source_order"])
        .drop(columns=["metric_order", "model_order"])
        .reset_index(drop=True)
    )


def plot_metric_profiles(
    results: dict[str, Path],
    reference_cells: Path,
    reference_de: Path,
    output: Path,
    targets: list[str] | None = None,
) -> pd.DataFrame:
    profiles = build_profiles(results, reference_cells, reference_de, targets)
    order = (
        profiles[["target_gene", "n_de", "source_order"]]
        .drop_duplicates()
        .sort_values(["n_de", "source_order"])
        .reset_index(drop=True)
    )
    target_order = order["target_gene"].tolist()
    positions = pd.Series(np.arange(len(order)), index=target_order)

    figure, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    for axis, (metric, title, _) in zip(axes.flat, METRICS, strict=True):
        for boundary in (len(order) / 3, 2 * len(order) / 3):
            axis.axvline(boundary - 0.5, color=STRENGTH_LINES, linewidth=1, zorder=0)
        for color, (label, _) in zip(COLORS, results.items(), strict=False):
            frame = profiles.query("metric == @metric and model == @label").set_index(
                "target_gene"
            )
            values = frame["value"].reindex(target_order)
            axis.plot(
                positions,
                values,
                color=color,
                linewidth=1.35,
                alpha=0.9,
                label=label,
            )
            axis.scatter(positions, values, color=color, s=8, alpha=0.7, linewidths=0)
        axis.set_title(title, color=INK, fontsize=11)
        axis.set_ylabel("per-target value", color=MUTED)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(colors=MUTED)
        axis.grid(axis="y", color=STRENGTH_LINES, alpha=0.25, linewidth=0.7)

    ticks = np.arange(len(order)) if len(order) <= 20 else np.linspace(
        0, len(order) - 1, 7, dtype=int
    )
    for axis in axes[-1]:
        axis.set_xticks(ticks)
        axis.set_xticklabels(order.loc[ticks, "target_gene"], rotation=45, ha="right")
        axis.set_xlabel(
            f"{len(order)} perturbations, ordered by reference DE genes (FDR < 0.05)",
            color=MUTED,
        )
    figure.subplots_adjust(top=0.9, bottom=0.15, hspace=0.3, wspace=0.18)
    figure.suptitle(
        "Per-perturbation H1 metric profiles", color=INK, fontsize=15, y=0.985
    )
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=len(labels),
        frameon=False,
    )
    figure.text(
        0.5,
        0.018,
        "Normalized expression MSE is shown as the local numerator/denominator; "
        "its official panel aggregate remains the ratio of sums.",
        ha="center",
        color=MUTED,
        fontsize=9,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, facecolor="#fcfcfb")
    plt.close(figure)

    table = profiles.drop(columns="source_order")
    table.to_csv(output.with_suffix(".csv"), index=False, float_format="%.17g")
    return table
