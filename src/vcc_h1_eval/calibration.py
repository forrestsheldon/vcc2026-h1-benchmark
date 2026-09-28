"""Rescore H1 results against another panel's zero points (an inferred baseline calibration).

Every scored metric is ``score = f(raw, b, A)``: ``b`` is the generic-response baseline's
raw value and ``A`` the split-half replicate anchor, both computed on the panel being
scored. The H1 benchmark's ``b`` comes from H1, whose perturbations share unusually little
response, so its zero points can sit far from another panel's. On VCC 2026 validation, for
example, predicting no change scores -1.72 on DE direction fidelity against -0.08 here.

A calibration replaces each ``b`` with a value inferred from that panel's leaderboard score
for a *known* prediction: the control-resampling baseline, whose raw values are taken from
this benchmark's own unchanged-control run (resampled controls behave the same way on any
panel). The H1 anchor ``A`` stands in for the panel's unknown anchor. Scores are then
recomputed with cell-eval2's own ``score_one`` and each metric's catalog policy, so tails,
clamps and directions are exactly the scorer's. With the benchmark's own ``b`` this
reproduces ``scores.csv``.

These are estimates, not the panel's scores: H1 raw values are not the panel's raw values,
the anchor is borrowed, and a metric whose control score is clamped (MSE) cannot be
calibrated and keeps the H1 baseline.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from importlib.resources import files
from pathlib import Path

import pandas as pd
import polars as pl
from cell_eval2.catalog import CATALOG
from cell_eval2.scoring import score_one
from scipy.optimize import brentq

SCORED = (
    "pds_cosine",
    "expr_mse_unbiased_capped_norm",
    "de_wilcoxon_lfc_nmae",
    "de_wilcoxon_direction_fidelity_yield_raw",
    "de_wilcoxon_direction_reach_raw",
    "de_wilcoxon_sig_jaccard",
)


def available() -> list[str]:
    return sorted(p.name[:-5] for p in files("vcc_h1_eval.calibrations").iterdir() if p.name.endswith(".json"))


def load(name: str) -> dict:
    """A shipped calibration by name, or a calibration JSON file by path."""
    if Path(name).suffix == ".json" and Path(name).is_file():
        return json.loads(Path(name).read_text())
    path = files("vcc_h1_eval.calibrations") / f"{name}.json"
    if not path.is_file():
        raise ValueError(f"unknown calibration {name!r}; available: {available()}")
    return json.loads(path.read_text())


def scale_ends(scale_bundle: Path) -> tuple[dict, dict]:
    """The benchmark's own baseline (mean statistic) and replicate anchor per metric."""
    base = pl.read_csv(scale_bundle / "baseline_agg.csv").filter(pl.col("statistic") == "mean")
    anchor = pl.read_parquet(scale_bundle / "anchor_agg.parquet")
    b = {m: float(base[m][0]) for m in SCORED}
    a = dict(zip(anchor["metric"].to_list(), anchor["replicate"].to_list(), strict=True))
    return b, {m: float(a[m]) for m in SCORED}


def policy(metric: str, anchor: float):
    return replace(CATALOG[metric].scoring, anchor=float(anchor), allow_negative_baseline=False)


def score(metric: str, raw: float, base: float, anchor: float) -> float:
    return score_one(raw, base, policy(metric, anchor))


def infer_baseline(metric: str, raw: float, target: float, anchor: float) -> float | None:
    """The baseline b at which ``raw`` scores ``target``; None if the score is clamped."""
    spec = policy(metric, anchor)
    if spec.clamp_low is not None and target <= spec.clamp_low + 1e-12:
        return None  # a clamped score carries no information about b
    if spec.direction == "higher":
        lo, hi = -10.0 * max(1.0, abs(anchor)), anchor - 1e-9
    else:
        lo, hi = anchor + 1e-9, 10.0 * max(1.0, abs(anchor)) + abs(raw) * 10
    gap = lambda b: score_one(raw, b, spec) - target
    if gap(lo) * gap(hi) > 0:
        raise ValueError(
            f"{metric}: no baseline makes raw {raw:.4g} score {target:.4g} "
            f"against anchor {anchor:.4g}"
        )
    return brentq(gap, lo, hi, xtol=1e-12)


def derive(control_raw: dict, control_scores: dict, anchors: dict, fallback: dict) -> dict:
    """Per-metric baselines from a known prediction's raw values and its panel scores."""
    out = {}
    for m in SCORED:
        b = infer_baseline(m, control_raw[m], control_scores[m], anchors[m])
        out[m] = {"baseline": fallback[m] if b is None else b, "inferred": b is not None}
    return out


def rescore(results: Path, calibration: dict, scale_bundle: Path) -> pd.DataFrame:
    """Harness scores and calibrated scores for one results directory."""
    agg = pd.read_csv(results / "aggregates.csv").set_index("metric").raw_value
    own_b, anchors = scale_ends(scale_bundle)
    rows = []
    for m in SCORED:
        if m not in agg or not math.isfinite(float(agg[m])):
            continue
        cal_b = calibration["baselines"][m]["baseline"]
        rows.append(
            {
                "metric": m,
                "raw_value": float(agg[m]),
                "harness_score": score(m, float(agg[m]), own_b[m], anchors[m]),
                "calibrated_score": score(m, float(agg[m]), cal_b, anchors[m]),
                "calibrated_baseline": cal_b,
                "baseline_inferred": calibration["baselines"][m]["inferred"],
            }
        )
    table = pd.DataFrame(rows)
    if len(table) == len(SCORED):
        table.loc[len(table)] = {
            "metric": "avg_score",
            "harness_score": table.harness_score.mean(),
            "calibrated_score": table.calibrated_score.mean(),
        }
    return table


SHORT_NAMES = {  # the names `vcc status` prints
    "pds": "pds_cosine",
    "mse": "expr_mse_unbiased_capped_norm",
    "nmae": "de_wilcoxon_lfc_nmae",
    "fid": "de_wilcoxon_direction_fidelity_yield_raw",
    "reach": "de_wilcoxon_direction_reach_raw",
    "jaccard": "de_wilcoxon_sig_jaccard",
    "jac": "de_wilcoxon_sig_jaccard",
}


def calibrate_from_files(scores: Path, control_results: Path, name: str, scale_bundle: Path) -> dict:
    """Build a calibration document from a control submission's panel scores (JSON)."""
    given = json.loads(Path(scores).read_text())
    panel = {SHORT_NAMES.get(k, k): float(v) for k, v in given.items() if SHORT_NAMES.get(k, k) in SCORED}
    missing = [m for m in SCORED if m not in panel]
    if missing:
        raise ValueError(f"scores file is missing {missing}")
    raw = pd.read_csv(Path(control_results) / "aggregates.csv").set_index("metric").raw_value.to_dict()
    own_b, anchors = scale_ends(scale_bundle)
    return {
        "name": name,
        "method": "b solves score_one(raw_control, b, H1-anchored policy) = the panel score of the control submission",
        "reference_scores": panel,
        "baselines": derive(raw, panel, anchors, own_b),
        "h1_baselines": own_b,
        "h1_anchors": anchors,
    }
