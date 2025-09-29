#!/usr/bin/env python3
"""
Analyze experiment results across different confounding and instrument configurations.
Creates separate, detailed figures for Relative Bias, Standard Error, and Coverage Probability.

In each plot, separate lines are drawn for each unique combination of instrument and
outcome-only strengths, allowing for a detailed comparison without over-averaging.

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
        "y": r"y(\d+(?:p\d+)?)",    # outcome_only_to_outcome
        "i": r"i(\d+(?:p\d+)?)",    # exposure_only_to_exposure
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
    Computes mean of metrics and a combined standard error.
    """
    # Check if run_id exists and has more than one unique value
    if 'run_id' not in df.columns or df['run_id'].nunique() <= 1:
        return df  # No aggregation needed

    print("Aggregating results across multiple runs...")
    
    grouping_cols = [
        col for col in df.columns if col not in 
        ['run_id', 'effect', 'std_err', 'CI95_lower', 'CI95_upper', 
         'bias', 'relative_bias', 'covered']
    ]
    
    # Using pandas' built-in aggregation for speed and clarity
    agg_dict = {
        'effect': 'mean',
        'bias': 'mean',
        'relative_bias': 'mean',
        'covered': 'mean',
        'n_runs': ('run_id', 'size')
    }

    # Custom function for combined standard error
    def combined_se(group):
        n_runs = len(group)
        if n_runs <= 1:
            return group['std_err'].iloc[0]
        mean_within_run_var = (group['std_err'] ** 2).mean()
        var_across_runs = group['effect'].var(ddof=1)
        return np.sqrt(mean_within_run_var + var_across_runs)

    aggregated = df.groupby(grouping_cols).apply(lambda g: pd.Series({
        'effect': g['effect'].mean(),
        'bias': g['bias'].mean(),
        'relative_bias': g['relative_bias'].mean(),
        'covered': g['covered'].mean(),
        'std_err': combined_se(g),
        'n_runs': len(g)
    })).reset_index()
    
    # Recalculate CIs based on new aggregated values
    margin = 1.96 * aggregated['std_err']
    aggregated['CI95_lower'] = aggregated['effect'] - margin
    aggregated['CI95_upper'] = aggregated['effect'] + margin
    
    print(f"Aggregated {len(df)} rows into {len(aggregated)} unique configurations.")
    return aggregated


def load_experiment_results(
    results_dir: str, experiment_names: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """Load, combine, and aggregate results from multiple experiments and runs."""
    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    all_results = []
    
    run_dirs = [d for d in results_path.iterdir() if d.is_dir() and d.name.startswith('run_')]
    
    exp_dirs = []
    if run_dirs:
        print(f"Found {len(run_dirs)} run directories.")
        for run_dir in run_dirs:
            exp_paths = [d for d in run_dir.iterdir() if d.is_dir()]
            if experiment_names:
                exp_paths = [p for p in exp_paths if p.name in experiment_names]
            exp_dirs.extend(exp_paths)
    else:
        print("Using legacy directory structure (no 'run_XX' folders).")
        exp_dirs = [d for d in results_path.iterdir() if d.is_dir()]
        if experiment_names:
            exp_dirs = [p for p in exp_dirs if p.name in experiment_names]

    for exp_dir in exp_dirs:
        run_id = exp_dir.parent.name if exp_dir.parent.name.startswith('run_') else "run_01"
        
        # Check for both baseline and bert subfolders
        model_types_found = {
            model_type: (exp_dir / "estimate" / model_type / "estimate_results.csv")
            for model_type in ["baseline", "bert"]
        }
        model_types_found = {k: v for k, v in model_types_found.items() if v.exists()}

        # Handle legacy structure
        legacy_file = exp_dir / "estimate" / "estimate_results.csv"
        if legacy_file.exists() and not model_types_found:
            model_types_found["baseline"] = legacy_file

        if not model_types_found:
            continue

        for model_type, results_file in model_types_found.items():
            try:
                df = pd.read_csv(results_file)
                df["experiment"] = exp_dir.name
                df["run_id"] = run_id
                df["model_type"] = model_type
                params = parse_experiment_name(exp_dir.name)
                for param, value in params.items():
                    df[param] = value

                df["bias"] = df["effect"] - df["true_effect"]
                df["relative_bias"] = (df["bias"] / df["true_effect"]).replace([np.inf, -np.inf], np.nan)
                df["covered"] = (df["true_effect"] >= df["CI95_lower"]) & (df["true_effect"] <= df["CI95_upper"])
                all_results.append(df)
            except Exception as e:
                print(f"Error loading {results_file}: {e}")

    if not all_results:
        raise ValueError("No valid experiment results found.")

    # Group by model_type before aggregation
    full_df = pd.concat(all_results, ignore_index=True)
    results_by_model = {}
    for model_type, group in full_df.groupby('model_type'):
        print(f"\nProcessing model type: {model_type}")
        results_by_model[model_type] = aggregate_across_runs(group)
    
    return results_by_model


def create_plots(df: pd.DataFrame, output_dir: str):
    """
    Creates separate figures for key metrics, with detailed lines for each (i, y) setting.
    """
    os.makedirs(output_dir, exist_ok=True)

    methods_df = df[df["method"].isin(["TMLE", "IPW"])].copy()
    methods_df.dropna(subset=["relative_bias"], inplace=True)
    if methods_df.empty:
        print("Warning: No valid TMLE or IPW results found to plot.")
        return

    methods_df["avg_confounding"] = (methods_df["ce"] + methods_df["cy"]) / 2
    
    unique_combinations = methods_df[['i', 'y']].drop_duplicates().sort_values(['i', 'y'])
    n_combinations = len(unique_combinations)
    
    # Generate color palettes for TMLE and IPW
    tmle_palette = sns.color_palette("Blues_d", n_colors=n_combinations)
    ipw_palette = sns.color_palette("Reds_d", n_colors=n_combinations)
    
    color_map = {}
    for idx, (_, row) in enumerate(unique_combinations.iterrows()):
        key = (row['i'], row['y'])
        color_map[key] = {'TMLE': tmle_palette[idx], 'IPW': ipw_palette[idx]}
        
    plot_configs = [
        ('relative_bias', 'Relative Bias', 'Relative Bias (Bias / True Effect)'),
        ('std_err', 'Standard Error', 'Standard Error'),
        ('covered', 'Coverage Probability', 'Coverage Probability')
    ]

    for metric, title_prefix, ylabel in plot_configs:
        fig, axes = plt.subplots(1, 3, figsize=(28, 8), constrained_layout=True)
        fig.suptitle(f"{title_prefix} by Instrument/Outcome-Only Combinations", fontsize=20, fontweight="bold")

        for i, (param, xlabel_base) in enumerate([("avg_confounding", "Confounding"), ("i", "Instrument"), ("y", "Outcome-Only")]):
            ax = axes[i]
            
            for combo_idx, (_, combo_row) in enumerate(unique_combinations.iterrows()):
                i_val, y_val = combo_row['i'], combo_row['y']
                combo_key = (i_val, y_val)
                combo_data = methods_df[(methods_df['i'] == i_val) & (methods_df['y'] == y_val)]
                
                for method in ["TMLE", "IPW"]:
                    method_data = combo_data[combo_data["method"] == method]
                    if method_data.empty:
                        continue
                    
                    # For detailed plots, plotting individual points can be better than error bars
                    ax.plot(
                        method_data[param],
                        method_data[metric],
                        label=f"{method} (i={i_val}, y={y_val})",
                        color=color_map[combo_key][method],
                        marker="o" if method == "TMLE" else "s",
                        linestyle="-" if method == "TMLE" else "--",
                        markersize=6,
                        alpha=0.8
                    )
            
            if metric in ['relative_bias']:
                ax.axhline(0, color="black", linestyle=":", alpha=0.7)
            if metric == 'covered':
                ax.axhline(0.95, color="black", linestyle=":", alpha=0.7, label="95% Target")
                ax.set_ylim(0, 1.05)
            
            ax.set_title(f"{title_prefix} vs. {xlabel_base}", fontsize=14, fontweight="bold")
            ax.set_xlabel(f"{xlabel_base} Strength", fontsize=12)
            if i == 0:
                ax.set_ylabel(ylabel, fontweight="bold", fontsize=12)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Create a single, clean legend on the right
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(1.01, 0.95), loc='upper left', fontsize=10, title="Settings")
        
        filename = f"{metric}_detailed_overview.png"
        plt.savefig(Path(output_dir) / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved detailed plot to: {Path(output_dir) / filename}")


def main():
    parser = argparse.ArgumentParser(description="Analyze and summarize causal inference experiment results.")
    parser.add_argument("--results_dir", required=True, help="Directory containing experiment results.")
    parser.add_argument("--experiments", nargs="*", help="Specific experiments to analyze (default: all).")
    parser.add_argument("--output_dir", default="experiment_analysis_plots", help="Output directory for plots.")
    args = parser.parse_args()

    print("Loading experiment results...")
    results_by_model = load_experiment_results(args.results_dir, args.experiments)

    print("\n--- Starting Plot Generation ---")
    for model_type, df in results_by_model.items():
        model_output_dir = Path(args.output_dir) / model_type
        print(f"\nAnalyzing '{model_type}' results...")
        print(f"Creating plots in: {model_output_dir}")
        create_plots(df, str(model_output_dir))

    print(f"\nAnalysis complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()