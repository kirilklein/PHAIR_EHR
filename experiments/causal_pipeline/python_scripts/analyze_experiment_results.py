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


def aggregate_across_runs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate results across multiple runs for the same experiment configuration.
    Computes mean of means and proper combined standard errors.
    """
    if 'run_id' not in df.columns:
        return df  # No run aggregation needed
    
    print("Aggregating results across runs...")
    
    # Group by everything except run_id and the metrics we want to aggregate
    grouping_cols = [col for col in df.columns if col not in 
                    ['run_id', 'effect', 'std_err', 'CI95_lower', 'CI95_upper', 
                     'bias', 'relative_bias', 'covered']]
    
    def aggregate_metrics(group):
        n_runs = len(group)
        
        # Mean of means
        mean_effect = group['effect'].mean()
        
        # Combined standard error: sqrt(mean(var_within_runs) + var(means_across_runs))
        # where var_within_runs = std_err^2
        mean_within_run_var = (group['std_err'] ** 2).mean()
        var_across_runs = group['effect'].var(ddof=1) if n_runs > 1 else 0
        combined_std_err = np.sqrt(mean_within_run_var + var_across_runs)
        
        # Recalculate confidence intervals with combined standard error
        # Assuming normal distribution (could be improved with t-distribution)
        margin = 1.96 * combined_std_err
        ci_lower = mean_effect - margin
        ci_upper = mean_effect + margin
        
        # Aggregate other metrics
        mean_bias = group['bias'].mean()
        mean_relative_bias = group['relative_bias'].mean()
        mean_covered = group['covered'].mean()  # Coverage probability
        
        return pd.Series({
            'effect': mean_effect,
            'std_err': combined_std_err,
            'CI95_lower': ci_lower,
            'CI95_upper': ci_upper,
            'bias': mean_bias,
            'relative_bias': mean_relative_bias,
            'covered': mean_covered,
            'n_runs': n_runs
        })
    
    aggregated = df.groupby(grouping_cols).apply(aggregate_metrics).reset_index()
    
    print(f"Aggregated {len(df)} individual run results into {len(aggregated)} experiment configurations")
    
    # Show aggregation summary
    run_counts = aggregated['n_runs'].value_counts().sort_index()
    print("Runs per experiment configuration:")
    for n_runs, count in run_counts.items():
        print(f"  {count} experiments with {n_runs} runs each")
    
    return aggregated


def load_experiment_results(
    results_dir: str, experiment_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """Load and combine results from multiple experiments and runs."""
    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    all_results = []
    
    # Check if we have the new run structure (run_01, run_02, etc.)
    run_dirs = [d for d in results_path.iterdir() if d.is_dir() and d.name.startswith('run_')]
    
    if run_dirs:
        # New structure: outputs/causal/sim_study/runs/run_XX/experiment_name/
        print(f"Found {len(run_dirs)} run directories: {[d.name for d in run_dirs]}")
        exp_dirs = []
        for run_dir in run_dirs:
            if experiment_names:
                exp_dirs.extend([run_dir / name for name in experiment_names if (run_dir / name).exists()])
            else:
                exp_dirs.extend([d for d in run_dir.iterdir() if d.is_dir()])
    else:
        # Legacy structure: outputs/causal/sim_study/runs/experiment_name/
        print("Using legacy directory structure")
        exp_dirs = (
            [results_path / name for name in experiment_names]
            if experiment_names
            else [d for d in results_path.iterdir() if d.is_dir()]
        )

    for exp_dir in exp_dirs:
        exp_name = exp_dir.name
        
        # Determine run ID and experiment name
        if exp_dir.parent.name.startswith('run_'):
            run_id = exp_dir.parent.name
            experiment_config = exp_name
        else:
            run_id = "run_01"  # Legacy structure
            experiment_config = exp_name

        # Check for both baseline and BERT results
        model_types = []
        if (exp_dir / "estimate" / "baseline" / "estimate_results.csv").exists():
            model_types.append("baseline")
        if (exp_dir / "estimate" / "bert" / "estimate_results.csv").exists():
            model_types.append("bert")

        # Handle legacy structure (direct estimate folder)
        if (exp_dir / "estimate" / "estimate_results.csv").exists() and not model_types:
            model_types.append(
                "baseline"
            )  # Treat as baseline for backward compatibility

        if not model_types:
            print(f"Warning: No results found for {run_id}/{exp_name}")
            continue

        # Load results for each model type
        for model_type in model_types:
            if (
                model_type == "baseline"
                and (exp_dir / "estimate" / "estimate_results.csv").exists()
            ):
                # Legacy structure
                results_file = exp_dir / "estimate" / "estimate_results.csv"
            else:
                # New structure
                results_file = (
                    exp_dir / "estimate" / model_type / "estimate_results.csv"
                )

            if not results_file.exists():
                print(f"Warning: {model_type} results file not found for {run_id}/{exp_name}")
                continue

            try:
                df = pd.read_csv(results_file)
                df["experiment"] = experiment_config
                df["run_id"] = run_id
                df["model_type"] = model_type
                params = parse_experiment_name(experiment_config)
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
                print(f"Loaded {model_type} results for {run_id}/{exp_name}: {len(df)} rows")
            except Exception as e:
                print(f"Error loading {model_type} results for {run_id}/{exp_name}: {e}")

    if not all_results:
        raise ValueError("No valid experiment results found.")

    # Group results by model type
    results_by_model = {}
    for df in all_results:
        model_type = df["model_type"].iloc[0]
        if model_type not in results_by_model:
            results_by_model[model_type] = []
        results_by_model[model_type].append(df)

    # Combine results for each model type
    combined_results = {}
    for model_type, dfs in results_by_model.items():
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Aggregate across runs for the same experiment configuration
        aggregated_df = aggregate_across_runs(combined_df)
        combined_results[model_type] = aggregated_df
        
        print(
            f"Combined {model_type} results: {len(aggregated_df)} final rows from {len(combined_df)} individual run results across {len(dfs)} experiment files."
        )

    if not combined_results:
        raise ValueError("No valid experiment results found after processing.")

    return combined_results


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
    results_by_model = load_experiment_results(args.results_dir, args.experiments)

    print("Creating separate analyses for each model type...")
    for model_type, df in results_by_model.items():
        model_output_dir = os.path.join(args.output_dir, model_type)
        print(f"\nAnalyzing {model_type} results...")
        print(f"Creating plots for {model_type} in: {model_output_dir}")
        create_plots(df, model_output_dir)

    print(f"\nAnalysis complete! Results saved to: {args.output_dir}")
    print("Subfolders created:")
    for model_type in results_by_model.keys():
        print(f"  - {model_type}/")


if __name__ == "__main__":
    main()
