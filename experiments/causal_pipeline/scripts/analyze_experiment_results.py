#!/usr/bin/env python3
"""
Analyze experiment results across different confounding and instrument configurations.
Creates a single overview plot for TMLE and IPW methods, averaged over all outcomes.

Usage:
    python analyze_experiment_results.py --results_dir outputs/causal/experiments
    python analyze_experiment_results.py --results_dir outputs/causal/experiments --output_dir plots/
"""

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional

# Add project root to path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

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

            # Calculate bias metrics
            df["bias"] = df["effect"] - df["true_effect"]
            # Calculate relative bias, handling potential division by zero in true_effect
            df["relative_bias"] = (df["bias"] / df["true_effect"]).replace(
                [np.inf, -np.inf], np.nan
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
    Creates a single 2x2 plot summarizing method performance, averaged over all outcomes.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Filter to TMLE and IPW methods and drop rows with NaN relative_bias
    methods_df = df[df["method"].isin(["TMLE", "IPW"])].copy()
    methods_df.dropna(subset=["relative_bias"], inplace=True)

    if methods_df.empty:
        print("Warning: No valid TMLE or IPW results found to plot.")
        return

    # --- Prepare data for plotting ---
    methods_df["avg_confounding"] = (methods_df["ce"] + methods_df["cy"]) / 2
    methods_df["rmse"] = np.sqrt(methods_df["bias"] ** 2 + methods_df["std_err"] ** 2)

    # --- Create the plot ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    fig.suptitle(
        "Performance Overview: TMLE vs. IPW (Averaged Across All Outcomes)",
        fontsize=20,
        fontweight="bold",
    )

    colors = {"TMLE": "#0072B2", "IPW": "#D55E00"}

    # --- Plot 1: Relative Bias vs. Confounding Strength ---
    ax = axes[0, 0]
    for method in ["TMLE", "IPW"]:
        grouped = (
            methods_df[methods_df["method"] == method]
            .groupby("avg_confounding")["relative_bias"]
            .agg(["mean", "std"])
            .reset_index()
        )
        ax.errorbar(
            grouped["avg_confounding"],
            grouped["mean"],
            yerr=grouped["std"],
            label=method,
            color=colors[method],
            marker="o",
            capsize=5,
            linestyle="-",
            linewidth=2.5,
        )
    ax.axhline(0, color="black", linestyle="--", alpha=0.7)
    ax.set_title("Bias vs. Confounding Strength", fontsize=14)
    ax.set_xlabel("Average Confounding Strength ((ce + cy) / 2)")
    ax.set_ylabel("Relative Bias (Bias / True Effect)")
    ax.legend()

    # --- Plot 2: Relative Bias vs. Instrument Strength ---
    ax = axes[0, 1]
    for method in ["TMLE", "IPW"]:
        grouped = (
            methods_df[methods_df["method"] == method]
            .groupby("i")["relative_bias"]
            .agg(["mean", "std"])
            .reset_index()
        )
        ax.errorbar(
            grouped["i"],
            grouped["mean"],
            yerr=grouped["std"],
            label=method,
            color=colors[method],
            marker="s",
            capsize=5,
            linestyle="-",
            linewidth=2.5,
        )
    ax.axhline(0, color="black", linestyle="--", alpha=0.7)
    ax.set_title("Bias vs. Instrument Strength", fontsize=14)
    ax.set_xlabel("Instrument Strength (i)")
    ax.set_ylabel("Relative Bias (Bias / True Effect)")
    ax.legend()

    # --- Plot 3: Standard Error vs. Confounding Strength ---
    ax = axes[1, 0]
    for method in ["TMLE", "IPW"]:
        grouped = (
            methods_df[methods_df["method"] == method]
            .groupby("avg_confounding")["std_err"]
            .agg(["mean", "std"])
            .reset_index()
        )
        ax.errorbar(
            grouped["avg_confounding"],
            grouped["mean"],
            yerr=grouped["std"],
            label=method,
            color=colors[method],
            marker="D",
            capsize=5,
            linestyle="--",
            linewidth=2.5,
        )
    ax.set_title("Precision vs. Confounding Strength", fontsize=14)
    ax.set_xlabel("Average Confounding Strength ((ce + cy) / 2)")
    ax.set_ylabel("Average Standard Error")
    ax.legend()

    # --- Plot 4: RMSE vs. Confounding Strength ---
    ax = axes[1, 1]
    for method in ["TMLE", "IPW"]:
        grouped = (
            methods_df[methods_df["method"] == method]
            .groupby("avg_confounding")["rmse"]
            .agg(["mean", "std"])
            .reset_index()
        )
        ax.errorbar(
            grouped["avg_confounding"],
            grouped["mean"],
            yerr=grouped["std"],
            label=method,
            color=colors[method],
            marker="^",
            capsize=5,
            linestyle="-",
            linewidth=2.5,
        )
    ax.set_title("Overall Error (RMSE) vs. Confounding", fontsize=14)
    ax.set_xlabel("Average Confounding Strength ((ce + cy) / 2)")
    ax.set_ylabel("Average RMSE (sqrt(Bias² + SE²))")
    ax.legend()

    # Save the final plot
    output_path = Path(output_dir) / "experiment_overview_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Overview plot saved to: {output_path}")


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

    print("Creating simplified overview plot...")
    create_overview_plot(df, args.output_dir)

    # The summary report function can be kept if desired, but is not essential for the plot
    # print("Creating summary report...")
    # create_summary_report(df, args.output_dir)

    print(f"\nAnalysis complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
