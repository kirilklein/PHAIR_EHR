#!/usr/bin/env python3
"""
Creates simplified, clean plots for bias, coverage, variance, relative bias, and Z-score analysis.

Generates ten figures, two for each metric (vs. Confounding and vs. Instrument).
All plots skip creating subplots that would only contain a single data point.

Usage:
    python analyze_experiment_results.py --results_dir outputs/causal/experiments
    python analyze_experiment_results.py --results_dir outputs/causal/experiments --outcomes OUTCOME_1 OUTCOME_2
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional

# Set plotting style
plt.style.use("seaborn-v0_8-whitegrid")


def parse_experiment_name(exp_name: str) -> Dict[str, float]:
    """Parse experiment name to extract parameter values, including negative 'm' values."""
    import re

    params = {}
    patterns = {
        "ce": r"ce(m?\d+(?:p\d+)?)",
        "cy": r"cy(m?\d+(?:p\d+)?)",
        "y": r"y(\d+(?:p\d+)?)",
        "i": r"i(\d+(?:p\d+)?)",
    }
    for param, pattern in patterns.items():
        match = re.search(pattern, exp_name)
        if match:
            value_str = match.group(1).replace("m", "-").replace("p", ".")
            params[param] = float(value_str)
        else:
            params[param] = 0.0
    return params


def load_and_process_results(
    results_dir: str, experiment_names: Optional[list] = None
) -> pd.DataFrame:
    """Loads all data and calculates bias, coverage, relative bias, and z-score."""
    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    all_results = []
    run_dirs = [
        d for d in results_path.iterdir() if d.is_dir() and d.name.startswith("run_")
    ]
    if not run_dirs:
        print("No 'run_XX' directories found. Treating results_dir as a single run.")
        run_dirs = [results_path]

    for run_dir in run_dirs:
        exp_dirs = [d for d in run_dir.iterdir() if d.is_dir()]
        if experiment_names:
            exp_dirs = [d for d in exp_dirs if d.name in experiment_names]

        for exp_dir in exp_dirs:
            possible_paths = [
                exp_dir / "estimate" / "estimate_results.csv",
                exp_dir / "estimate" / "baseline" / "estimate_results.csv",
                exp_dir / "estimate" / "bert" / "estimate_results.csv",
            ]
            results_file = next(
                (path for path in possible_paths if path.exists()), None
            )

            if not results_file:
                continue

            try:
                df = pd.read_csv(results_file)
                df["run_id"] = run_dir.name
                params = parse_experiment_name(exp_dir.name)
                for param, value in params.items():
                    df[param] = value

                df["bias"] = df["effect"] - df["true_effect"]
                df["covered"] = (df["true_effect"] >= df["CI95_lower"]) & (
                    df["true_effect"] <= df["CI95_upper"]
                )
                df["relative_bias"] = (df["bias"] / df["true_effect"]).replace(
                    [np.inf, -np.inf], np.nan
                )
                df["z_score"] = (df["bias"] / df["std_err"]).replace(
                    [np.inf, -np.inf], np.nan
                )
                all_results.append(df)
            except Exception as e:
                print(f"Error loading {results_file}: {e}")

    if not all_results:
        raise ValueError("No valid results found to process.")

    combined_df = pd.concat(all_results, ignore_index=True)
    print(f"Loaded {len(combined_df)} total rows from {len(run_dirs)} runs.")
    return combined_df


# --- AGGREGATION FUNCTIONS ---


def perform_bias_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    """Performs two-step aggregation for absolute bias analysis."""
    methods_df = df[df["method"].isin(["TMLE", "IPW"])].copy()
    if methods_df.empty:
        return pd.DataFrame()
    mean_bias_per_run = (
        methods_df.groupby(["run_id", "method", "ce", "cy", "i", "y"])["bias"]
        .mean()
        .reset_index()
    )
    mean_bias_per_run["avg_confounding"] = (
        mean_bias_per_run["ce"] + mean_bias_per_run["cy"]
    ) / 2
    final_agg = (
        mean_bias_per_run.groupby(["method", "avg_confounding", "i"])["bias"]
        .agg(["mean", "std"])
        .reset_index()
    )
    return final_agg


def perform_relative_bias_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    """Performs two-step aggregation for relative bias, excluding true_effect=0 cases."""
    methods_df = df[df["method"].isin(["TMLE", "IPW"])].copy()
    methods_df.dropna(subset=["relative_bias"], inplace=True)
    if methods_df.empty:
        return pd.DataFrame()
    mean_rb_per_run = (
        methods_df.groupby(["run_id", "method", "ce", "cy", "i", "y"])["relative_bias"]
        .mean()
        .reset_index()
    )
    mean_rb_per_run["avg_confounding"] = (
        mean_rb_per_run["ce"] + mean_rb_per_run["cy"]
    ) / 2
    final_agg = (
        mean_rb_per_run.groupby(["method", "avg_confounding", "i"])["relative_bias"]
        .agg(["mean", "std"])
        .reset_index()
    )
    return final_agg


def perform_zscore_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    """Performs two-step aggregation for Z-score (Standardized Bias)."""
    methods_df = df[df["method"].isin(["TMLE", "IPW"])].copy()
    methods_df.dropna(subset=["z_score"], inplace=True)
    if methods_df.empty:
        return pd.DataFrame()
    mean_z_per_run = (
        methods_df.groupby(["run_id", "method", "ce", "cy", "i", "y"])["z_score"]
        .mean()
        .reset_index()
    )
    mean_z_per_run["avg_confounding"] = (
        mean_z_per_run["ce"] + mean_z_per_run["cy"]
    ) / 2
    final_agg = (
        mean_z_per_run.groupby(["method", "avg_confounding", "i"])["z_score"]
        .agg(["mean", "std"])
        .reset_index()
    )
    return final_agg


def perform_coverage_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    """Performs single-step aggregation for coverage analysis."""
    methods_df = df[df["method"].isin(["TMLE", "IPW"])].copy()
    if methods_df.empty:
        return pd.DataFrame()
    methods_df["avg_confounding"] = (methods_df["ce"] + methods_df["cy"]) / 2
    coverage_agg = (
        methods_df.groupby(["method", "avg_confounding", "i"])["covered"]
        .mean()
        .reset_index()
    )
    return coverage_agg


def perform_variance_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    """Performs aggregation for empirical variance analysis."""
    methods_df = df[df["method"].isin(["TMLE", "IPW"])].copy()
    if methods_df.empty:
        return pd.DataFrame()
    variance_per_outcome = (
        methods_df.groupby(["method", "ce", "cy", "i", "y", "outcome"])["effect"]
        .var(ddof=1)
        .reset_index()
    )
    variance_per_outcome.rename(columns={"effect": "variance"}, inplace=True)
    variance_per_outcome["avg_confounding"] = (
        variance_per_outcome["ce"] + variance_per_outcome["cy"]
    ) / 2
    final_variance_agg = (
        variance_per_outcome.groupby(["method", "avg_confounding", "i"])["variance"]
        .mean()
        .reset_index()
    )
    return final_variance_agg


# --- PLOTTING FUNCTIONS ---


def create_plot_from_agg(
    agg_data: pd.DataFrame,
    metric_col: str,
    y_label: str,
    title_prefix: str,
    output_dir: str,
    plot_type: str = "errorbar",
):
    """Generic plotting function for creating faceted 1xN plots."""
    if agg_data.empty:
        return

    # Plot vs Confounding
    all_instrument_levels = sorted(agg_data["i"].unique())
    instrument_levels_to_plot = [
        lvl
        for lvl in all_instrument_levels
        if agg_data[agg_data["i"] == lvl]["avg_confounding"].nunique() > 1
    ]
    if instrument_levels_to_plot:
        print(
            f"Skipping {title_prefix} vs. Confounder plots for i={set(all_instrument_levels) - set(instrument_levels_to_plot)} (only one data point)."
        )
        n_subplots = len(instrument_levels_to_plot)
        fig, axes = plt.subplots(
            1,
            n_subplots,
            figsize=(6 * n_subplots, 5),
            sharey=True,
            constrained_layout=True,
        )
        if n_subplots == 1:
            axes = [axes]
        fig.suptitle(
            f"{title_prefix} vs. Confounding Strength", fontsize=16, fontweight="bold"
        )
        for i, inst_level in enumerate(instrument_levels_to_plot):
            ax = axes[i]
            subplot_data = agg_data[agg_data["i"] == inst_level]
            for method, color in [("TMLE", "blue"), ("IPW", "red")]:
                method_data = subplot_data[subplot_data["method"] == method]
                if not method_data.empty:
                    if plot_type == "errorbar":
                        ax.errorbar(
                            x=method_data["avg_confounding"],
                            y=method_data["mean"],
                            yerr=method_data["std"],
                            label=method,
                            color=color,
                            marker="o",
                            capsize=5,
                            linestyle="-",
                        )
                    else:
                        ax.plot(
                            method_data["avg_confounding"],
                            method_data[metric_col],
                            label=method,
                            color=color,
                            marker="o",
                            markersize=7,
                            linestyle="-" if plot_type == "line" else "",
                        )
            if plot_type == "errorbar":
                ax.axhline(0, color="black", linestyle="--", alpha=0.7)
            if metric_col == "covered":
                ax.axhline(
                    0.95, color="gray", linestyle="--", alpha=0.9, label="95% Target"
                )
            ax.set_title(f"Instrument Strength (i) = {inst_level}")
            ax.set_xlabel("Average Confounding Strength")
            if i == 0:
                ax.set_ylabel(y_label)
            ax.legend()
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        output_path = Path(output_dir) / f"{metric_col}_vs_confounding_plot.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {title_prefix} vs. Confounding plot to: {output_path}")

    # Plot vs Instrument
    all_conf_levels = sorted(agg_data["avg_confounding"].unique())
    conf_levels_to_plot = [
        lvl
        for lvl in all_conf_levels
        if agg_data[agg_data["avg_confounding"] == lvl]["i"].nunique() > 1
    ]
    if conf_levels_to_plot:
        print(
            f"Skipping {title_prefix} vs. Instrument plots for avg_confounding={set(all_conf_levels) - set(conf_levels_to_plot)} (only one data point)."
        )
        n_subplots = len(conf_levels_to_plot)
        fig, axes = plt.subplots(
            1,
            n_subplots,
            figsize=(6 * n_subplots, 5),
            sharey=True,
            constrained_layout=True,
        )
        if n_subplots == 1:
            axes = [axes]
        fig.suptitle(
            f"{title_prefix} vs. Instrument Strength", fontsize=16, fontweight="bold"
        )
        for i, conf_level in enumerate(conf_levels_to_plot):
            ax = axes[i]
            subplot_data = agg_data[agg_data["avg_confounding"] == conf_level]
            for method, color in [("TMLE", "blue"), ("IPW", "red")]:
                method_data = subplot_data[subplot_data["method"] == method]
                if not method_data.empty:
                    if plot_type == "errorbar":
                        ax.errorbar(
                            x=method_data["i"],
                            y=method_data["mean"],
                            yerr=method_data["std"],
                            label=method,
                            color=color,
                            marker="o",
                            capsize=5,
                            linestyle="-",
                        )
                    else:
                        ax.plot(
                            method_data["i"],
                            method_data[metric_col],
                            label=method,
                            color=color,
                            marker="o",
                            markersize=7,
                            linestyle="-" if plot_type == "line" else "",
                        )
            if plot_type == "errorbar":
                ax.axhline(0, color="black", linestyle="--", alpha=0.7)
            if metric_col == "covered":
                ax.axhline(
                    0.95, color="gray", linestyle="--", alpha=0.9, label="95% Target"
                )
            ax.set_title(f"Confounding Strength = {conf_level:.2f}")
            ax.set_xlabel("Instrument Strength (i)")
            if i == 0:
                ax.set_ylabel(y_label)
            ax.legend()
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        output_path = Path(output_dir) / f"{metric_col}_vs_instrument_plot.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {title_prefix} vs. Instrument plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create simplified analysis plots for causal inference experiments."
    )
    parser.add_argument(
        "--results_dir", required=True, help="Directory containing run subdirectories."
    )
    parser.add_argument(
        "--output_dir",
        default="experiment_analysis_plots",
        help="Directory to save plots.",
    )
    # **NEW**: Add argument to filter by outcome
    parser.add_argument(
        "--outcomes",
        nargs="*",
        help="Specific outcomes to include in the analysis (default: all).",
    )
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Load raw data
    raw_data = load_and_process_results(args.results_dir)

    # **NEW**: Filter data based on the --outcomes argument
    if args.outcomes:
        print(f"Filtering results for specific outcomes: {', '.join(args.outcomes)}")
        initial_rows = len(raw_data)
        raw_data = raw_data[raw_data["outcome"].isin(args.outcomes)]
        print(f"Filtered data from {initial_rows} to {len(raw_data)} rows.")
        if raw_data.empty:
            print(
                "Warning: No data remains after filtering for specified outcomes. Exiting."
            )
            return

    # 2. Perform aggregations for each analysis type
    agg_bias_data = perform_bias_aggregation(raw_data)
    agg_relative_bias_data = perform_relative_bias_aggregation(raw_data)
    agg_zscore_data = perform_zscore_aggregation(raw_data)
    agg_coverage_data = perform_coverage_aggregation(raw_data)
    agg_variance_data = perform_variance_aggregation(raw_data)

    # 3. Create plots for each metric
    create_plot_from_agg(
        agg_bias_data,
        "bias",
        "Average Bias (± Std Dev)",
        "Average Bias",
        args.output_dir,
        "errorbar",
    )
    create_plot_from_agg(
        agg_relative_bias_data,
        "relative_bias",
        "Average Relative Bias (± Std Dev)",
        "Relative Bias",
        args.output_dir,
        "errorbar",
    )
    create_plot_from_agg(
        agg_zscore_data,
        "z_score",
        "Average Z-Score (± Std Dev)",
        "Standardized Bias (Z-Score)",
        args.output_dir,
        "errorbar",
    )
    create_plot_from_agg(
        agg_coverage_data,
        "covered",
        "Coverage Probability",
        "95% CI Coverage",
        args.output_dir,
        "dot",
    )
    create_plot_from_agg(
        agg_variance_data,
        "variance",
        "Average Empirical Variance",
        "Empirical Variance",
        args.output_dir,
        "line",
    )

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
