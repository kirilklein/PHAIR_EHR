#!/usr/bin/env python3
"""Summarize nested model-refit and patient-bootstrap study results.

Usage:
    python -m experiments.semisynthetic_simulation.python_scripts.summarize \
        --study-dir <dir> [--out <path>]
"""

import argparse
from pathlib import Path

import pandas as pd

from experiments.semisynthetic_simulation.python_scripts.study_summary import (
    aggregate_replicates,
    load_bootstrap_results,
    load_results,
    summarize_performance,
)


def main():
    parser = argparse.ArgumentParser(description="Summarize a semi-synthetic study")
    parser.add_argument("--study-dir", required=True)
    parser.add_argument(
        "--out", default=None, help="Output CSV (default: <study-dir>/summary.csv)"
    )
    args = parser.parse_args()

    study_dir = Path(args.study_dir)
    results = load_results(study_dir)
    bootstrap_results = load_bootstrap_results(study_dir)
    replicates = aggregate_replicates(results, bootstrap_results)
    table = summarize_performance(replicates)

    out = Path(args.out) if args.out else study_dir / "summary.csv"
    replicates.to_csv(study_dir / "replicate_estimates.csv", index=False)
    table.to_csv(out, index=False)

    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(table.to_string(index=False))
    print(f"\nSaved summary to {out}")


if __name__ == "__main__":
    main()
