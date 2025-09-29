#!/usr/bin/env python3
"""
Creates two simplified, clean plots for bias analysis:
1. Bias vs. Confounding Strength (faceted by instrument level).
2. Bias vs. Instrument Strength (faceted by confounding level).

Both plots skip creating subplots that would only contain a single data point.

Usage:
    python analyze_experiment_results.py --results_dir outputs/causal/experiments
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional

# Set plotting style
plt.style.use("seaborn-v0_8-whitegrid")


def parse_experiment_name(exp_name: str) -> Dict[str, float]:
    """Parse experiment name to extract parameter values, including negative 'm' values."""
    import re
    params = {}
    patterns = {
        "ce": r"ce(m?\d+(?:p\d+)?)", "cy": r"cy(m?\d+(?:p\d+)?)",
        "y": r"y(\d+(?:p\d+)?)", "i": r"i(\d+(?:p\d+)?)",
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
    """Loads all data from run directories, calculates bias, and adds parameters."""
    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    all_results = []
    run_dirs = [d for d in results_path.iterdir() if d.is_dir() and d.name.startswith('run_')]
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
                exp_dir / "estimate" / "bert" / "estimate_results.csv"
            ]
            results_file = next((path for path in possible_paths if path.exists()), None)

            if not results_file:
                continue
            
            try:
                df = pd.read_csv(results_file)
                df["run_id"] = run_dir.name
                params = parse_experiment_name(exp_dir.name)
                for param, value in params.items():
                    df[param] = value
                
                df["bias"] = df["effect"] - df["true_effect"]
                all_results.append(df)
            except Exception as e:
                print(f"Error loading {results_file}: {e}")

    if not all_results:
        raise ValueError("No valid results found to process.")

    combined_df = pd.concat(all_results, ignore_index=True)
    print(f"Loaded {len(combined_df)} total rows from {len(run_dirs)} runs.")
    return combined_df

def perform_bias_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    """Performs the two-step aggregation to get the final mean and std dev of bias."""
    methods_df = df[df["method"].isin(["TMLE", "IPW"])].copy()
    if methods_df.empty:
        print("No TMLE or IPW data found for aggregation.")
        return pd.DataFrame()

    # Step A: Average bias over all outcomes WITHIN each run and experiment.
    mean_bias_per_run = methods_df.groupby(
        ['run_id', 'method', 'ce', 'cy', 'i', 'y']
    )['bias'].mean().reset_index()

    mean_bias_per_run['avg_confounding'] = (mean_bias_per_run['ce'] + mean_bias_per_run['cy']) / 2

    # Step B: For each experiment setting, calculate the mean and std dev of the
    # mean biases ACROSS all runs. This averages over the 'y' dimension.
    final_agg = mean_bias_per_run.groupby(
        ['method', 'avg_confounding', 'i']
    )['bias'].agg(['mean', 'std']).reset_index()
    
    return final_agg

def create_bias_vs_confounder_plot(final_agg: pd.DataFrame, output_dir: str):
    """Generates plot of Bias vs. Confounding, faceted by Instrument."""
    if final_agg.empty: return

    all_instrument_levels = sorted(final_agg['i'].unique())
    instrument_levels_to_plot = [
        lvl for lvl in all_instrument_levels
        if final_agg[final_agg['i'] == lvl]['avg_confounding'].nunique() > 1
    ]

    if not instrument_levels_to_plot:
        print("No instrument levels with multiple confounder points found. Skipping Bias vs. Confounder plot.")
        return
        
    print(f"Skipping Bias vs. Confounder plots for i={set(all_instrument_levels) - set(instrument_levels_to_plot)} (only one data point).")

    n_subplots = len(instrument_levels_to_plot)
    fig, axes = plt.subplots(1, n_subplots, figsize=(6 * n_subplots, 5), sharey=True, constrained_layout=True)
    if n_subplots == 1: axes = [axes]

    fig.suptitle("Average Bias vs. Confounding Strength", fontsize=16, fontweight='bold')
    for i, inst_level in enumerate(instrument_levels_to_plot):
        ax = axes[i]
        subplot_data = final_agg[final_agg['i'] == inst_level]
        for method, color in [("TMLE", "blue"), ("IPW", "red")]:
            method_data = subplot_data[subplot_data['method'] == method]
            if not method_data.empty:
                ax.errorbar(x=method_data['avg_confounding'], y=method_data['mean'], yerr=method_data['std'],
                            label=method, color=color, marker='o', capsize=5, linestyle='-')
        ax.axhline(0, color='black', linestyle='--', alpha=0.7)
        ax.set_title(f'Instrument Strength (i) = {inst_level}')
        ax.set_xlabel('Average Confounding Strength')
        if i == 0: ax.set_ylabel('Average Bias (± Std Dev of Mean Bias)')
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    output_path = Path(output_dir) / "bias_vs_confounding_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Bias vs. Confounding plot to: {output_path}")

def create_bias_vs_instrument_plot(final_agg: pd.DataFrame, output_dir: str):
    """Generates plot of Bias vs. Instrument, faceted by Confounding."""
    if final_agg.empty: return

    all_conf_levels = sorted(final_agg['avg_confounding'].unique())
    conf_levels_to_plot = [
        lvl for lvl in all_conf_levels
        if final_agg[final_agg['avg_confounding'] == lvl]['i'].nunique() > 1
    ]

    if not conf_levels_to_plot:
        print("No confounding levels with multiple instrument points found. Skipping Bias vs. Instrument plot.")
        return

    print(f"Skipping Bias vs. Instrument plots for avg_confounding={set(all_conf_levels) - set(conf_levels_to_plot)} (only one data point).")

    n_subplots = len(conf_levels_to_plot)
    fig, axes = plt.subplots(1, n_subplots, figsize=(6 * n_subplots, 5), sharey=True, constrained_layout=True)
    if n_subplots == 1: axes = [axes]

    fig.suptitle("Average Bias vs. Instrument Strength", fontsize=16, fontweight='bold')
    for i, conf_level in enumerate(conf_levels_to_plot):
        ax = axes[i]
        subplot_data = final_agg[final_agg['avg_confounding'] == conf_level]
        for method, color in [("TMLE", "blue"), ("IPW", "red")]:
            method_data = subplot_data[subplot_data['method'] == method]
            if not method_data.empty:
                ax.errorbar(x=method_data['i'], y=method_data['mean'], yerr=method_data['std'],
                            label=method, color=color, marker='o', capsize=5, linestyle='-')
        ax.axhline(0, color='black', linestyle='--', alpha=0.7)
        ax.set_title(f'Confounding Strength = {conf_level:.2f}')
        ax.set_xlabel('Instrument Strength (i)')
        if i == 0: ax.set_ylabel('Average Bias (± Std Dev of Mean Bias)')
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    output_path = Path(output_dir) / "bias_vs_instrument_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Bias vs. Instrument plot to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Create simplified bias analysis plots.")
    parser.add_argument("--results_dir", required=True, help="Directory containing run subdirectories.")
    parser.add_argument("--output_dir", default="experiment_analysis_plots", help="Directory to save plots.")
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Load raw data
    processed_data = load_and_process_results(args.results_dir)

    # 2. Perform the aggregation once
    final_agg_data = perform_bias_aggregation(processed_data)

    # 3. Create the two separate plots from the same aggregated data
    create_bias_vs_confounder_plot(final_agg_data, args.output_dir)
    create_bias_vs_instrument_plot(final_agg_data, args.output_dir)

    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()