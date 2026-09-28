import json

import pandas as pd
import polars as pl
import pytest

from vcc_h1_eval import calibration as cal

ANCHORS = {m: a for m, a in zip(cal.SCORED, [0.99, 0.19, 0.33, 0.87, 0.98, 0.43], strict=True)}
OWN_B = {m: b for m, b in zip(cal.SCORED, [0.51, 0.97, 0.96, 0.067, 0.082, 0.002], strict=True)}


@pytest.mark.parametrize("metric", [m for m in cal.SCORED if m != "expr_mse_unbiased_capped_norm"])
def test_inferred_baseline_reproduces_the_target_score(metric):
    raw = {"pds_cosine": 0.48, "de_wilcoxon_lfc_nmae": 1.0}.get(metric, 0.01)
    for target in (-1.7, -0.08, 0.0, 0.3):
        b = cal.infer_baseline(metric, raw, target, ANCHORS[metric])
        assert cal.score(metric, raw, b, ANCHORS[metric]) == pytest.approx(target, abs=1e-9)


def test_own_baseline_round_trip():
    for m in cal.SCORED:
        if m == "expr_mse_unbiased_capped_norm":
            continue
        s = cal.score(m, 0.2 if m != "de_wilcoxon_lfc_nmae" else 0.9, OWN_B[m], ANCHORS[m])
        b = cal.infer_baseline(m, 0.2 if m != "de_wilcoxon_lfc_nmae" else 0.9, s, ANCHORS[m])
        assert b == pytest.approx(OWN_B[m], rel=1e-7)


def test_clamped_score_is_not_identifiable():
    m = "expr_mse_unbiased_capped_norm"
    assert cal.infer_baseline(m, 1.0, 0.0, ANCHORS[m]) is None
    raw = dict(zip(cal.SCORED, [0.49, 1.0, 1.0, 0.0, 0.03, 0.0], strict=True))
    out = cal.derive(raw, {k: -0.1 for k in cal.SCORED} | {m: 0.0}, ANCHORS, OWN_B)
    assert out[m] == {"baseline": OWN_B[m], "inferred": False}
    assert all(out[k]["inferred"] for k in cal.SCORED if k != m)


def test_impossible_target_raises_a_clear_error():
    with pytest.raises(ValueError, match="no baseline"):
        cal.infer_baseline("de_wilcoxon_sig_jaccard", 0.5, -0.1, 0.43)


def test_shipped_calibration_loads():
    assert "vcc2026-val-1" in cal.available()
    doc = cal.load("vcc2026-val-1")
    assert set(doc["baselines"]) == set(cal.SCORED)
    fid = doc["baselines"]["de_wilcoxon_direction_fidelity_yield_raw"]
    assert fid["inferred"] and 0.5 < fid["baseline"] < 0.6


def test_calibrate_and_rescore_from_files(tmp_path):
    bundle = tmp_path / "scale"
    bundle.mkdir()
    pl.DataFrame({"statistic": ["mean"], **{m: [OWN_B[m]] for m in cal.SCORED}}).write_csv(bundle / "baseline_agg.csv")
    pl.DataFrame({"metric": list(ANCHORS), "replicate": list(ANCHORS.values())}).write_parquet(bundle / "anchor_agg.parquet")
    control = tmp_path / "control"
    control.mkdir()
    raw = dict(zip(cal.SCORED, [0.49, 1.0, 1.0, 0.0, 0.03, 0.0], strict=True))
    pd.DataFrame({"metric": list(raw), "raw_value": list(raw.values())}).to_csv(control / "aggregates.csv", index=False)
    scores = tmp_path / "scores.json"
    scores.write_text(json.dumps({"pds": -0.005, "mse": 0, "nmae": -0.005, "fid": -1.72, "reach": -0.008, "jaccard": -0.083}))
    doc = cal.calibrate_from_files(scores, control, "test", bundle)
    table = cal.rescore(control, doc, bundle).set_index("metric")
    # the control rescored under its own calibration reproduces the panel scores it was built from
    assert table.loc["de_wilcoxon_direction_fidelity_yield_raw", "calibrated_score"] == pytest.approx(-1.72)
    assert table.loc["pds_cosine", "calibrated_score"] == pytest.approx(-0.005)
    assert "avg_score" in table.index
