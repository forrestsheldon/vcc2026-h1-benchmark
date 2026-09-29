"""Build the H1 scale's zero point (baseline leg) the way the VCC 2026 scorer does.

The official 2026 baseline is the control-mean profile -- an equal-weight average of each
non-targeting guide's mean count vector -- emitted as IDENTICAL fractional cells for every
perturbation and scored as an ordinary prediction (cell-eval2 `real_bundle._baseline_leg`:
"The tiled arm is a mean and therefore fractional in any counts space"). Identical cells have
zero variance, so Wilcoxon calls most genes and direction fidelity sits at a coin flip.

This tool scores that arm with the benchmark's bounded-memory scorer (the dense tiled arm does
not fit comfortably in memory through `build_real_bundle`) and replaces `baseline_agg.csv` and
`baseline_meta.json` in an existing scale bundle. The anchor leg is untouched.

    python tools/build_control_mean_baseline.py --scale-bundle PATH [--data-dir DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from cell_eval2.run import aggregate_metrics_wide, metric_output_names
from scipy import sparse

from vcc_h1_eval import scorer
from vcc_h1_eval.bounded import RowSource
from vcc_h1_eval.cli import _score_args
from vcc_h1_eval.paths import BenchmarkPaths


class TiledMatrix:
    """Every row is the same profile; indexing returns that many copies."""

    def __init__(self, profile: np.ndarray):
        self.row = sparse.csr_matrix(profile[None, :])

    def __getitem__(self, index):
        return sparse.vstack([self.row] * len(np.atleast_1d(index)), format="csr")


class TiledData:
    def __init__(self, profile: np.ndarray, genes: list[str]):
        self.X = TiledMatrix(profile)
        self.var_names = pd.Index(genes)


def guide_balanced_mean(controls) -> tuple[np.ndarray, int]:
    guides = controls.obs["guide_id"].astype(str).to_numpy()
    names, member = np.unique(guides, return_inverse=True)
    sums = np.zeros((len(names), controls.n_vars))
    for start in range(0, controls.n_obs, 5000):
        stop = min(start + 5000, controls.n_obs)
        block = sparse.csr_matrix(controls.X[start:stop])
        indicator = sparse.csr_matrix(
            (np.ones(stop - start), (member[start:stop], np.arange(stop - start))),
            shape=(len(names), stop - start),
        )
        sums += np.asarray((indicator @ block).todense())
    means = sums / np.bincount(member, minlength=len(names))[:, None]
    return means.mean(axis=0), len(names)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale-bundle", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--de-threads", type=int, default=4)
    cli = parser.parse_args()

    paths = BenchmarkPaths.resolve(cli.data_dir)
    args = _score_args(argparse.Namespace(de_threads=cli.de_threads), paths)
    controls_data = scorer.open_controls(args)
    try:
        genes = controls_data.var_names.astype(str).tolist()
        profile, n_guides = guide_balanced_mean(controls_data)
        print(f"{n_guides} guides; profile {profile.sum():.0f} counts/cell", flush=True)
        targets = scorer.read_scoring_contract(args)
        labels = np.repeat(targets, scorer.CELLS_PER_TARGET)
        prediction = RowSource(TiledData(profile, genes), np.zeros(len(labels), np.int64), labels)
        controls = RowSource(
            controls_data,
            np.arange(controls_data.n_obs),
            np.repeat(scorer.CONTROL, controls_data.n_obs),
        )
        identity = hashlib.sha256(b"control-mean-tiled:" + profile.tobytes()).hexdigest()
        results, config = scorer.compute_results(args, prediction, controls, targets, identity)
    finally:
        controls_data.file.close()

    wide = aggregate_metrics_wide(results, metrics=metric_output_names(config))
    wide.write_csv(cli.scale_bundle / "baseline_agg.csv")
    meta_path = cli.scale_bundle / "baseline_meta.json"
    meta = json.loads(meta_path.read_text())
    meta.update(
        {
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "baseline_construction": {
                "profile": "equal-weight mean of per-guide control mean counts",
                "guides": n_guides,
                "emission": "identical fractional cells (tiled), 400 per target",
                "scorer": "vcc_h1_eval bounded-memory scorer (compute_results)",
            },
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(wide.filter(wide["statistic"] == "mean"), flush=True)


if __name__ == "__main__":
    main()
