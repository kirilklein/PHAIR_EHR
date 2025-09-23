#!/usr/bin/env python3
"""
Analyze experiment results across different confounding and instrument configurations.
Creates three separate figures:
1. Relative Bias across confounding, instrument, and outcome-only strengths.
2. Standard Error across confounding, instrument, and outcome-only strengths.
3. Coverage Probability across confounding, instrument, and outcome-only strengths.

All plots are averaged over all outcomes for TMLE and IPW methods.

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
            df["bias"] = df["effect"] - df["true_effect"]
            df["relative_bias"] = (df["bias"] / df["true_effect"]).replace(
                [np.inf, -np.inf], np.nan
            )

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


def create_plots(df: pd.DataFrame, output_dir: str):
    """
    Creates separate figures for Relative Bias, Standard Error, and Coverage Probability.
    Each figure contains subplots for confounding, instrument, and outcome-only effects.
    """
    os.makedirs(output_dir, exist_ok=True)

    methods_df = df[df["method"].isin(["TMLE", "IPW"])].copy()
    methods_df.dropna(subset=["relative_bias"], inplace=True)

    if methods_df.empty:
        print("Warning: No valid TMLE or IPW results found to plot.")
        return

    methods_df["avg_confounding"] = (methods_df["ce"] + methods_df["cy"]) / 2

    colors = {"TMLE": "#0072B2", "IPW": "#D55E00"}
    param_configs = [
        ("avg_confounding", "Average Confounding Strength ((ce + cy) / 2)", "o"),
        ("i", "Instrument Strength (i)", "s"),
        ("y", "Outcome-Only Strength (y)", "^"),
    ]

    # --- Figure 1: Relative Bias ---
    fig_bias, axes_bias = plt.subplots(1, 3, figsize=(24, 7), constrained_layout=True)
    fig_bias.suptitle(
        "Relative Bias of Estimators (Averaged Across All Outcomes)",
        fontsize=20,
        fontweight="bold",
    )

    for i, (param, xlabel, marker) in enumerate(param_configs):
        ax = axes_bias[i]
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
        ax.set_title(f"Bias vs. {xlabel.split(' (')[0]}", fontsize=16)
        ax.set_xlabel(xlabel)
        if i == 0:
            ax.set_ylabel(
                "Relative Bias (Bias / True Effect)", fontweight="bold", fontsize=14
            )
        ax.legend()
    plt.savefig(
        Path(output_dir) / "relative_bias_overview.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig_bias)
    print(
        f"Relative Bias plot saved to: {Path(output_dir) / 'relative_bias_overview.png'}"
    )

    # --- Figure 2: Standard Error ---
    fig_se, axes_se = plt.subplots(1, 3, figsize=(24, 7), constrained_layout=True)
    fig_se.suptitle(
        "Standard Error of Estimators (Averaged Across All Outcomes)",
        fontsize=20,
        fontweight="bold",
    )

    for i, (param, xlabel, marker) in enumerate(param_configs):
        ax = axes_se[i]
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
        ax.set_title(f"Precision vs. {xlabel.split(' (')[0]}", fontsize=16)
        ax.set_xlabel(xlabel)
        if i == 0:
            ax.set_ylabel("Average Standard Error", fontweight="bold", fontsize=14)
        ax.legend()
    plt.savefig(
        Path(output_dir) / "standard_error_overview.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig_se)
    print(
        f"Standard Error plot saved to: {Path(output_dir) / 'standard_error_overview.png'}"
    )

    # --- Figure 3: Coverage Probability ---
    fig_cov, axes_cov = plt.subplots(1, 3, figsize=(24, 7), constrained_layout=True)
    fig_cov.suptitle(
        "Coverage Probability of 95% CIs (Averaged Across All Outcomes)",
        fontsize=20,
        fontweight="bold",
    )

    for i, (param, xlabel, marker) in enumerate(param_configs):
        ax = axes_cov[i]
        for method in ["TMLE", "IPW"]:
            grouped = (
                methods_df[methods_df["method"] == method]
                .groupby(param)["covered"]
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
        ax.axhline(0.95, color="black", linestyle="--", alpha=0.7, label="Target (95%)")
        ax.set_ylim(0, 1.05)  # Set y-axis from 0 to 105%
        ax.set_title(f"Coverage vs. {xlabel.split(' (')[0]}", fontsize=16)
        ax.set_xlabel(xlabel)
        if i == 0:
            ax.set_ylabel("Coverage Probability", fontweight="bold", fontsize=14)
        ax.legend()
    plt.savefig(
        Path(output_dir) / "coverage_probability_overview.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig_cov)
    print(
        f"Coverage Probability plot saved to: {Path(output_dir) / 'coverage_probability_overview.png'}"
    )


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

    print("Creating separate metric-specific overview plots...")
    create_plots(df, args.output_dir)

    print(f"\nAnalysis complete! Plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
