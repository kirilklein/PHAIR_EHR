#!/usr/bin/env python3
"""
Compare IPW vs TMLE performance across different simulation scenarios.

This script loads and visualizes the results from both the original simulation
(where TMLE outperforms IPW) and the new IPW-favorable simulation to demonstrate
how the presence of instrumental variables causes overadjustment bias in IPW,
while pure confounding structures favor IPW over TMLE.

Key Insight: IPW overadjustment occurs when propensity score models include
instruments (exposure-only factors) that shouldn't be adjusted for.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_results(results_dir: str) -> pd.DataFrame:
    """Load effect estimation results from a directory."""
    results_file = Path(results_dir) / "final_results.csv"
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    return pd.read_csv(results_file)


def calculate_bias_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate bias metrics for each method and outcome."""
    metrics = []

    for outcome in df["outcome"].unique():
        outcome_data = df[df["outcome"] == outcome]

        # Get true effect if available
        true_effect_row = outcome_data[outcome_data["method"] == "True Effect"]
        if len(true_effect_row) == 0:
            continue
        true_effect = true_effect_row["effect"].iloc[0]

        for method in ["IPW", "TMLE"]:
            method_data = outcome_data[outcome_data["method"] == method]
            if len(method_data) == 0:
                continue

            estimated_effect = method_data["effect"].iloc[0]
            bias = estimated_effect - true_effect
            abs_bias = abs(bias)

            metrics.append(
                {
                    "outcome": outcome,
                    "method": method,
                    "true_effect": true_effect,
                    "estimated_effect": estimated_effect,
                    "bias": bias,
                    "abs_bias": abs_bias,
                    "ci_lower": method_data["ci_lower"].iloc[0],
                    "ci_upper": method_data["ci_upper"].iloc[0],
                }
            )

    return pd.DataFrame(metrics)


def create_comparison_plot(
    metrics_original: pd.DataFrame,
    metrics_ipw_favorable: pd.DataFrame,
    save_path: str = None,
):
    """Create a comparison plot showing bias for both simulations."""

    # Add simulation labels
    metrics_original["simulation"] = "Original (TMLE-favorable)"
    metrics_ipw_favorable["simulation"] = "IPW-favorable"

    # Combine datasets
    combined_metrics = pd.concat(
        [metrics_original, metrics_ipw_favorable], ignore_index=True
    )

    # Create subplot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Absolute bias comparison
    sns.barplot(
        data=combined_metrics,
        x="outcome",
        y="abs_bias",
        hue="method",
        col="simulation",
        kind="bar",
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("Absolute Bias by Method and Simulation")
    axes[0, 0].set_ylabel("Absolute Bias")
    axes[0, 0].tick_params(axis="x", rotation=45)

    # 2. Bias (signed) comparison
    sns.barplot(
        data=combined_metrics, x="outcome", y="bias", hue="method", ax=axes[0, 1]
    )
    axes[0, 1].axhline(y=0, color="black", linestyle="--", alpha=0.5)
    axes[0, 1].set_title("Bias (Signed) - Combined Simulations")
    axes[0, 1].set_ylabel("Bias")
    axes[0, 1].tick_params(axis="x", rotation=45)

    # 3. Method performance by simulation
    avg_bias = (
        combined_metrics.groupby(["simulation", "method"])["abs_bias"]
        .mean()
        .reset_index()
    )
    sns.barplot(
        data=avg_bias, x="simulation", y="abs_bias", hue="method", ax=axes[1, 0]
    )
    axes[1, 0].set_title("Average Absolute Bias by Simulation Type")
    axes[1, 0].set_ylabel("Average Absolute Bias")
    axes[1, 0].tick_params(axis="x", rotation=45)

    # 4. Scatter plot: Original vs IPW-favorable bias
    pivot_original = metrics_original.pivot(
        index="outcome", columns="method", values="abs_bias"
    )
    pivot_ipw = metrics_ipw_favorable.pivot(
        index="outcome", columns="method", values="abs_bias"
    )

    if "IPW" in pivot_original.columns and "TMLE" in pivot_original.columns:
        axes[1, 1].scatter(
            pivot_original["IPW"],
            pivot_original["TMLE"],
            alpha=0.7,
            label="Original",
            s=80,
        )
    if "IPW" in pivot_ipw.columns and "TMLE" in pivot_ipw.columns:
        axes[1, 1].scatter(
            pivot_ipw["IPW"], pivot_ipw["TMLE"], alpha=0.7, label="IPW-favorable", s=80
        )

    # Add diagonal line
    max_val = max(axes[1, 1].get_xlim()[1], axes[1, 1].get_ylim()[1])
    axes[1, 1].plot([0, max_val], [0, max_val], "k--", alpha=0.5)
    axes[1, 1].set_xlabel("IPW Absolute Bias")
    axes[1, 1].set_ylabel("TMLE Absolute Bias")
    axes[1, 1].set_title("IPW vs TMLE Bias Comparison")
    axes[1, 1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Comparison plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    """Main function to run the comparison analysis."""

    # Define paths
    original_results_dir = "./outputs/causal/estimate/baseline/simulated"
    ipw_favorable_results_dir = "./outputs/causal/estimate/baseline/simulated_ipw"

    try:
        # Load results
        print("Loading original simulation results...")
        original_results = load_results(original_results_dir)

        print("Loading IPW-favorable simulation results...")
        ipw_favorable_results = load_results(ipw_favorable_results_dir)

        # Calculate bias metrics
        print("Calculating bias metrics...")
        metrics_original = calculate_bias_metrics(original_results)
        metrics_ipw_favorable = calculate_bias_metrics(ipw_favorable_results)

        # Create comparison plot
        print("Creating comparison visualization...")
        output_dir = Path("./outputs/figs/simulation_comparison")
        output_dir.mkdir(parents=True, exist_ok=True)

        create_comparison_plot(
            metrics_original,
            metrics_ipw_favorable,
            save_path=output_dir / "ipw_vs_tmle_simulation_comparison.png",
        )

        # Print summary statistics
        print("\n" + "=" * 60)
        print("SIMULATION COMPARISON SUMMARY")
        print("=" * 60)

        print("\nOriginal Simulation (Contains Instruments & Outcome-only factors):")
        orig_summary = metrics_original.groupby("method")["abs_bias"].agg(
            ["mean", "std"]
        )
        print(orig_summary)

        print("\nIPW-favorable Simulation (Pure Confounding Structure):")
        ipw_summary = metrics_ipw_favorable.groupby("method")["abs_bias"].agg(
            ["mean", "std"]
        )
        print(ipw_summary)

        # Check which method performs better in each simulation
        print("\nMethod Performance Comparison:")
        print("Original simulation (with instruments) - Average absolute bias:")
        for method in ["IPW", "TMLE"]:
            if method in orig_summary.index:
                print(f"  {method}: {orig_summary.loc[method, 'mean']:.4f}")

        print("IPW-favorable simulation (confounders only) - Average absolute bias:")
        for method in ["IPW", "TMLE"]:
            if method in ipw_summary.index:
                print(f"  {method}: {ipw_summary.loc[method, 'mean']:.4f}")

        # Highlight the overadjustment insight
        print("\n" + "=" * 60)
        print("KEY INSIGHT: IPW OVERADJUSTMENT")
        print("=" * 60)
        print("When propensity score models include instruments (variables that")
        print("affect treatment but not outcome), IPW suffers from overadjustment")
        print("bias. The pure confounding simulation eliminates this issue by")
        print("ensuring all features are legitimate targets for PS adjustment.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure both simulations have been run and results are available.")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
