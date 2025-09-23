#!/usr/bin/env python3
"""
Analyze experiment results across different confounding and instrument configurations.
Creates a comprehensive 3x3 overview plot for TMLE and IPW methods,
including bias, precision, and coverage probability, averaged over all outcomes.

Usage:
    python analyze_experiment_results.py --results_dir outputs/causal/experiments
    python analyze_experiment_results.py --results_dir outputs/causal/experiments --output_dir plots/
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional

# Set plotting style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("colorblind")


def parse_experiment_name(exp_name: str) -> Dict[str, float]:
    """
    Parse experiment name to extract parameter values.
    Example: ce1p5_cy2_y0_i3 -> {ce: 1.5, cy: 2.0, y: 0.0, i: 3.0}
    """
    import re

    params = {}
    patterns = {
        "ce": r"ce(\d+(?:p\d+)?)",  # shared_to_exposure
        "cy": r"cy(\d+(?:p\d+)?)",  # shared_to_outcome
        "y": r"y(\d+(?:p\d+)?)",  # outcome_only_to_outcome
        "i": r"i(\d+(?:p\d+)?)",  # exposure_only_to_exposure
    }

    for param, pattern in patterns.items():
        match = re.search(pattern, exp_name)
        if match:
            value_str = match.group(1).replace("p", ".")
            params[param] = float(value_str)
        else:
            params[param] = 0.0
    return params


def load_experiment_results(
    results_dir: str, experiment_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """Load and combine results from multiple experiments."""
    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    all_results = []
    exp_dirs = (
        [results_path / name for name in experiment_names]
        if experiment_names
        else [d for d in results_path.iterdir() if d.is_dir()]
    )

    for exp_dir in exp_dirs:
        results_file = exp_dir / "estimate" / "estimate_results.csv"
        if not results_file.exists():
            print(f"Warning: Results file not found for {exp_dir.name}")
            continue

        try:
            df = pd.read_csv(results_file)
            df["experiment"] = exp_dir.name
            params = parse_experiment_name(exp_dir.name)
            for param, value in params.items():
                df[param] = value

            # --- Calculate Metrics ---
            # 1. Bias
            df["bias"] = df["effect"] - df["true_effect"]
            df["relative_bias"] = (df["bias"] / df["true_effect"]).replace(
                [np.inf, -np.inf], np.nan
            )

            # 2. Coverage Probability (NEW)
            df["covered"] = (df["true_effect"] >= df["CI95_lower"]) & (
                df["true_effect"] <= df["CI95_upper"]
            )

            all_results.append(df)
        except Exception as e:
            print(f"Error loading {exp_dir.name}: {e}")

    if not all_results:
        raise ValueError("No valid experiment results found.")

    combined_df = pd.concat(all_results, ignore_index=True)
    print(
        f"Combined results: {len(combined_df)} rows from {len(all_results)} experiments."
    )
    return combined_df


def create_overview_plot(df: pd.DataFrame, output_dir: str):
    """
    Creates a single 3x3 plot summarizing method performance (bias, precision, coverage).
    """
    os.makedirs(output_dir, exist_ok=True)

    methods_df = df[df["method"].isin(["TMLE", "IPW"])].copy()
    methods_df.dropna(subset=["relative_bias"], inplace=True)

    if methods_df.empty:
        print("Warning: No valid TMLE or IPW results found to plot.")
        return

    methods_df["avg_confounding"] = (methods_df["ce"] + methods_df["cy"]) / 2

    # --- Create the 3x3 plot ---
    fig, axes = plt.subplots(3, 3, figsize=(24, 20), constrained_layout=True)
    fig.suptitle(
        "Performance Overview: TMLE vs. IPW (Averaged Across All Outcomes)",
        fontsize=24,
        fontweight="bold",
    )

    colors = {"TMLE": "#0072B2", "IPW": "#D55E00"}

    # === ROW 1: RELATIVE BIAS (ACCURACY) ===
    bias_axes = axes[0]
    bias_axes[0].set_ylabel("Relative Bias", fontweight="bold", fontsize=14)
    for ax, param, marker in zip(
        bias_axes, ["avg_confounding", "i", "y"], ["o", "s", "^"]
    ):
        for method in ["TMLE", "IPW"]:
            grouped = (
                methods_df[methods_df["method"] == method]
                .groupby(param)["relative_bias"]
                .agg(["mean", "std"])
                .reset_index()
            )
            ax.errorbar(
                grouped[param],
                grouped["mean"],
                yerr=grouped["std"],
                label=method,
                color=colors[method],
                marker=marker,
                capsize=5,
                linestyle="-",
                linewidth=2.5,
            )
        ax.axhline(0, color="black", linestyle="--", alpha=0.7)
        ax.legend()
    bias_axes[0].set_title("Bias vs. Confounding", fontsize=16)
    bias_axes[0].set_xlabel("Average Confounding Strength")
    bias_axes[1].set_title("Bias vs. Instrument Strength", fontsize=16)
    bias_axes[1].set_xlabel("Instrument Strength (i)")
    bias_axes[2].set_title("Bias vs. Outcome-Only Confounder", fontsize=16)
    bias_axes[2].set_xlabel("Outcome-Only Strength (y)")

    # === ROW 2: STANDARD ERROR (PRECISION) ===
    se_axes = axes[1]
    se_axes[0].set_ylabel("Average Standard Error", fontweight="bold", fontsize=14)
    for ax, param, marker in zip(
        se_axes, ["avg_confounding", "i", "y"], ["o", "s", "^"]
    ):
        for method in ["TMLE", "IPW"]:
            grouped = (
                methods_df[methods_df["method"] == method]
                .groupby(param)["std_err"]
                .agg(["mean", "std"])
                .reset_index()
            )
            ax.errorbar(
                grouped[param],
                grouped["mean"],
                yerr=grouped["std"],
                label=method,
                color=colors[method],
                marker=marker,
                capsize=5,
                linestyle="--",
                linewidth=2.5,
            )
        ax.legend()
    se_axes[0].set_title("Precision vs. Confounding", fontsize=16)
    se_axes[0].set_xlabel("Average Confounding Strength")
    se_axes[1].set_title("Precision vs. Instrument Strength", fontsize=16)
    se_axes[1].set_xlabel("Instrument Strength (i)")
    se_axes[2].set_title("Precision vs. Outcome-Only Confounder", fontsize=16)
    se_axes[2].set_xlabel("Outcome-Only Strength (y)")

    # === ROW 3: COVERAGE PROBABILITY (INFERENCE VALIDITY) ===
    cov_axes = axes[2]
    cov_axes[0].set_ylabel("Coverage Probability", fontweight="bold", fontsize=14)
    for ax, param, marker in zip(
        cov_axes, ["avg_confounding", "i", "y"], ["o", "s", "^"]
    ):
        for method in ["TMLE", "IPW"]:
            # For a boolean, mean() gives the proportion.
            grouped = (
                methods_df[methods_df["method"] == method]
                .groupby(param)["covered"]
                .mean()
                .reset_index()
            )
            ax.plot(
                grouped[param],
                grouped["covered"],
                label=method,
                color=colors[method],
                marker=marker,
                linestyle="-",
                linewidth=2.5,
                markersize=8,
            )
        ax.axhline(0.95, color="black", linestyle="--", alpha=0.7, label="Target (95%)")
        ax.set_ylim(0, 1.05)  # Set y-axis from 0 to 105%
        ax.legend()
    cov_axes[0].set_title("Coverage vs. Confounding", fontsize=16)
    cov_axes[0].set_xlabel("Average Confounding Strength")
    cov_axes[1].set_title("Coverage vs. Instrument Strength", fontsize=16)
    cov_axes[1].set_xlabel("Instrument Strength (i)")
    cov_axes[2].set_title("Coverage vs. Outcome-Only Confounder", fontsize=16)
    cov_axes[2].set_xlabel("Outcome-Only Strength (y)")

    # Save the final plot
    output_path = Path(output_dir) / "experiment_overview_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Comprehensive 3x3 overview plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and summarize causal inference experiment results."
    )
    parser.add_argument(
        "--results_dir", required=True, help="Directory containing experiment results."
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        help="Specific experiments to analyze (default: all).",
    )
    parser.add_argument(
        "--output_dir",
        default="experiment_analysis_plots",
        help="Output directory for plots.",
    )
    args = parser.parse_args()

    print("Loading experiment results...")
    df = load_experiment_results(args.results_dir, args.experiments)

    print("Creating comprehensive 3x3 overview plot...")
    create_overview_plot(df, args.output_dir)

    print(f"\nAnalysis complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
