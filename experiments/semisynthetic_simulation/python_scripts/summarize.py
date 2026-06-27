#!/usr/bin/env python3
"""
Summarize a semi-synthetic study into an estimator-performance table.

Reads every ``estimate_results.csv`` under a study directory (each already
carries the point estimate, its CI, and the appended ``true_effect``), groups
by model x method x outcome, and reports bias, empirical SD, mean estimated SE,
SE-calibration, and 95% CI coverage.

- One estimate per group (Phase 1, single run): bias + covered are meaningful;
  SD/calibration are undefined (need >1).
- Many estimates per group (bootstrap refits / outcome redraws): full table.

Usage:
    python -m experiments.semisynthetic_simulation.python_scripts.summarize \
        --study-dir <dir> [--out <path>]
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from corebehrt.constants.causal.data import EffectColumns as E
from corebehrt.constants.causal.data import OUTCOME
from corebehrt.constants.causal.paths import ESTIMATE_RESULTS_FILE

MODEL_NAMES = ("bert", "baseline")


def _tag_from_path(path: Path) -> dict:
    """Infer model / outer-run / inner-refit ids from a result file's path."""
    parts = [p.lower() for p in path.parts]
    model = next((m for m in MODEL_NAMES if m in parts), "unknown")
    run_match = re.search(r"run_\d+", str(path))
    inner_match = re.search(r"k_\d+", str(path))
    return {
        "model": model,
        "run_id": run_match.group(0) if run_match else "run_01",
        "inner_id": inner_match.group(0) if inner_match else "k_01",
    }


def load_results(study_dir: Path) -> pd.DataFrame:
    """Load and tag every estimate_results.csv under study_dir."""
    files = sorted(study_dir.rglob(ESTIMATE_RESULTS_FILE))
    if not files:
        raise FileNotFoundError(f"No {ESTIMATE_RESULTS_FILE} found under {study_dir}")
    frames = []
    for path in files:
        df = pd.read_csv(path)
        for key, value in _tag_from_path(path).items():
            df[key] = value
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate estimates into a per (model, method, outcome) performance table."""
    rows = []
    for (model, method, outcome), group in results.groupby(
        ["model", E.method, OUTCOME]
    ):
        effects = group[E.effect]
        ses = group[E.std_err]
        true = group[E.true_effect].iloc[0]
        covered = (group[E.CI95_lower] <= true) & (group[E.CI95_upper] >= true)
        n = len(group)
        sd_emp = effects.std(ddof=1) if n > 1 else np.nan
        mean_se = ses.mean()
        rows.append(
            {
                "model": model,
                "method": method,
                "outcome": outcome,
                "n": n,
                "true_effect": true,
                "mean_effect": effects.mean(),
                "bias": effects.mean() - true,
                "sd_emp": sd_emp,
                "mean_se": mean_se,
                "se_calibration": (sd_emp / mean_se)
                if n > 1 and mean_se > 0
                else np.nan,
                "coverage": covered.mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["outcome", "model", "method"])


def main():
    parser = argparse.ArgumentParser(description="Summarize a semi-synthetic study")
    parser.add_argument("--study-dir", required=True)
    parser.add_argument(
        "--out", default=None, help="Output CSV (default: <study-dir>/summary.csv)"
    )
    args = parser.parse_args()

    study_dir = Path(args.study_dir)
    results = load_results(study_dir)
    table = summarize(results)

    out = Path(args.out) if args.out else study_dir / "summary.csv"
    table.to_csv(out, index=False)

    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(table.to_string(index=False))
    print(f"\nSaved summary to {out}")


if __name__ == "__main__":
    main()
