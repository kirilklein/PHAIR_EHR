from os.path import join
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from scipy.special import expit

from corebehrt.functional.utils.azure_save import save_figure_with_azure_copy


def plot_hist(p_exposure, output_dir, is_exposed: Optional[np.ndarray] = None):
    """
    Plot histogram of exposure probabilities, optionally colored by actual exposure status.

    Args:
        p_exposure: Array of predicted exposure probabilities
        output_dir: Directory to save the plot
        is_exposed: Optional array of actual exposure status (boolean)
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    if is_exposed is not None:
        # Split probabilities by actual exposure status
        p_control = p_exposure[~is_exposed]  # Not exposed (control)
        p_treated = p_exposure[is_exposed]  # Exposed (treated)

        # Create overlaid histograms with different colors
        bins = np.linspace(0, 1, 51)
        ax.hist(
            p_control,
            bins=bins,
            alpha=0.7,
            label="Control (Not Exposed)",
            color="#3498db",  # Blue
            density=True,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.hist(
            p_treated,
            bins=bins,
            alpha=0.7,
            label="Treated (Exposed)",
            color="#e74c3c",  # Red
            density=True,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.legend()
        ax.set_ylabel("Density")
        ax.set_title("Exposure Probability Distribution by Actual Exposure Status")
    else:
        # Original single histogram
        ax.hist(p_exposure, bins=50)
        ax.set_ylabel("Count")
        ax.set_title("Exposure Probability Histogram")

    ax.set_xlabel("Predicted Probability")
    ax.set_xlim(0, 1)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    save_figure_with_azure_copy(
        fig, join(output_dir, "exposure_probability_histogram.png")
    )


def plot_probability_distributions(
    all_probas: Dict[str, Dict[str, np.ndarray]], output_dir: str, bins: int = 50
):
    """Plots overlaid histograms of P0 and P1 for each outcome for easier interpretability."""

    num_outcomes = len(all_probas)
    if num_outcomes == 0:
        return

    # Layout: 2 columns if more than 1 outcome, else 1 column
    ncols = 2 if num_outcomes > 1 else 1
    nrows = (num_outcomes + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(8 * ncols, 6 * nrows), squeeze=False
    )
    axes: List[Axes] = axes.flatten()
    bins = np.linspace(0, 1, bins)
    for i, (outcome_name, probas) in enumerate(all_probas.items()):
        ax: Axes = axes[i]

        # Histogram for treated (P1)
        ax.hist(
            probas["P1"],
            bins=bins,
            range=(0, 1),
            alpha=0.5,
            label="P1 (Treated)",
            edgecolor="black",
        )
        # Histogram for control (P0)
        ax.hist(
            probas["P0"],
            bins=bins,
            range=(0, 1),
            alpha=0.5,
            label="P0 (Control)",
            edgecolor="black",
        )

        ax.set_title(f"Probability Histograms for {outcome_name}")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 1)
        ax.legend()

    # Hide any unused subplots
    for j in range(num_outcomes, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plot_path = join(output_dir, "probability_histograms.png")
    save_figure_with_azure_copy(fig, plot_path)


def plot_true_effects_vs_risk_differences(
    ite_df: pd.DataFrame,
    cf_df: pd.DataFrame,
    true_effects_config: Dict,
    output_dir: str,
):
    """
    Compare three different effect measures:
    1. True Effect (Config): The exposure_effect parameter from configuration
    2. Observed Risk Difference: Calculated from actual 0/1 outcomes (exposed vs control)
    3. Average Treatment Effect (ATE): Mean of Individual Treatment Effects

    Args:
        ite_df: DataFrame with individual treatment effects (ITE)
        cf_df: DataFrame with counterfactuals and exposure status
        true_effects_config: Dictionary with outcome names and their true exposure_effect values
        output_dir: Directory to save the plot
    """
    # Extract outcome names from ITE columns
    outcome_names = [
        col.replace("ite_", "") for col in ite_df.columns if col.startswith("ite_")
    ]

    if not outcome_names:
        print("No ITE columns found in the data")
        return

    # Calculate all three effect measures for each outcome
    results = []

    for outcome_name in outcome_names:
        ite_col = f"ite_{outcome_name}"
        outcome_col = f"outcome_{outcome_name}"

        if ite_col not in ite_df.columns or outcome_col not in cf_df.columns:
            continue

        # 1. TRUE EFFECT (CONFIG): Convert logit effect to probability scale
        true_logit_effect = true_effects_config.get(outcome_name, {}).get(
            "exposure_effect", 0.0
        )
        p_base = true_effects_config.get(outcome_name, {}).get("p_base", 0.2)

        logit_p_base = np.log(p_base / (1 - p_base))
        true_p1 = expit(logit_p_base + true_logit_effect)
        true_p0 = expit(logit_p_base)
        true_risk_difference = true_p1 - true_p0

        # 2. AVERAGE TREATMENT EFFECT (ATE): Mean of ITEs
        ate_mean = ite_df[ite_col].mean()
        ate_std = ite_df[ite_col].std()
        ate_se = ate_std / np.sqrt(len(ite_df))  # Standard error of the mean

        # 3. OBSERVED RISK DIFFERENCE: From actual 0/1 outcomes
        exposed_mask = cf_df["exposure"] == 1
        control_mask = cf_df["exposure"] == 0

        # Get the actual outcomes for exposed and control groups
        exposed_outcomes = cf_df.loc[exposed_mask, outcome_col]
        control_outcomes = cf_df.loc[control_mask, outcome_col]

        if len(exposed_outcomes) > 0 and len(control_outcomes) > 0:
            exposed_rate = exposed_outcomes.mean()
            control_rate = control_outcomes.mean()
            observed_risk_difference = exposed_rate - control_rate

            # Calculate standard error for the risk difference
            # SE(p1 - p0) = sqrt(p1*(1-p1)/n1 + p0*(1-p0)/n0)
            n_exposed = len(exposed_outcomes)
            n_control = len(control_outcomes)

            exposed_var = exposed_rate * (1 - exposed_rate) / n_exposed
            control_var = control_rate * (1 - control_rate) / n_control
            observed_risk_diff_se = np.sqrt(exposed_var + control_var)
        else:
            observed_risk_difference = 0.0
            observed_risk_diff_se = 0.0

        results.append(
            {
                "outcome": outcome_name,
                "true_logit_effect": true_logit_effect,
                "true_risk_difference": true_risk_difference,
                "ate_mean": ate_mean,
                "ate_std": ate_std,
                "ate_se": ate_se,
                "observed_risk_difference": observed_risk_difference,
                "observed_risk_diff_se": observed_risk_diff_se,
                "n_exposed": len(exposed_outcomes) if len(exposed_outcomes) > 0 else 0,
                "n_control": len(control_outcomes) if len(control_outcomes) > 0 else 0,
            }
        )

    if not results:
        print("No valid outcomes found for comparison")
        return

    # Create single bar chart comparing all three measures
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Extract data for plotting
    outcome_labels = [r["outcome"] for r in results]
    true_effects = [r["true_risk_difference"] for r in results]
    ate_means = [r["ate_mean"] for r in results]
    ate_ses = [r["ate_se"] for r in results]
    observed_effects = [r["observed_risk_difference"] for r in results]
    observed_ses = [r["observed_risk_diff_se"] for r in results]

    x_pos = np.arange(len(outcome_labels))
    width = 0.25  # Width for three bars

    bars1 = ax.bar(
        x_pos - width,
        true_effects,
        width,
        label="True Effect (Config)",
        color="#2E86AB",
        alpha=0.8,
        edgecolor="black",
    )
    bars2 = ax.bar(
        x_pos,
        ate_means,
        width,
        label="ATE (Mean ITE)",
        color="#A23B72",
        alpha=0.8,
        edgecolor="black",
        yerr=ate_ses,
        capsize=3,
    )
    bars3 = ax.bar(
        x_pos + width,
        observed_effects,
        width,
        label="Observed Risk Diff",
        color="#F18F01",
        alpha=0.8,
        edgecolor="black",
        yerr=observed_ses,
        capsize=3,
    )

    ax.set_xlabel("Outcome")
    ax.set_ylabel("Effect Size")
    ax.set_title("Comparison of Effect Measures")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(outcome_labels, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

    # Add value labels on bars
    _ = max(
        max(ate_ses) if ate_ses else [0], max(observed_ses) if observed_ses else [0]
    )

    for bar1, bar2, bar3, true_val, ate_val, obs_val, ate_se, obs_se in zip(
        bars1,
        bars2,
        bars3,
        true_effects,
        ate_means,
        observed_effects,
        ate_ses,
        observed_ses,
    ):
        # True effect label
        ax.text(
            bar1.get_x() + bar1.get_width() / 2,
            bar1.get_height() + 0.005,
            f"{true_val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        # ATE label
        ax.text(
            bar2.get_x() + bar2.get_width() / 2,
            bar2.get_height() + ate_se + 0.005,
            f"{ate_val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        # Observed risk diff label
        ax.text(
            bar3.get_x() + bar3.get_width() / 2,
            bar3.get_height() + obs_se + 0.005,
            f"{obs_val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()

    # Save the plot
    plot_path = join(output_dir, "true_vs_observed_effects.png")
    save_figure_with_azure_copy(fig, plot_path)

    # Print detailed summary statistics
    print("\n" + "=" * 80)
    print("COMPREHENSIVE EFFECT COMPARISON SUMMARY")
    print("=" * 80)

    for r in results:
        print(f"\n{r['outcome']}:")
        print(f"  Sample sizes: Exposed={r['n_exposed']}, Control={r['n_control']}")
        print(
            f"  True Effect (Config): {r['true_logit_effect']:.3f} (logit) → {r['true_risk_difference']:.3f} (risk diff)"
        )
        print(
            f"  ATE (Mean ITE): {r['ate_mean']:.3f} ± {r['ate_se']:.3f} (SE), σ={r['ate_std']:.3f}"
        )
        print(
            f"  Observed Risk Diff: {r['observed_risk_difference']:.3f} ± {r['observed_risk_diff_se']:.3f} (SE)"
        )
        print(f"  Differences from True:")
        print(f"    ATE - True: {r['ate_mean'] - r['true_risk_difference']:.3f}")
        print(
            f"    Observed - True: {r['observed_risk_difference'] - r['true_risk_difference']:.3f}"
        )
        print(
            f"    ATE - Observed: {r['ate_mean'] - r['observed_risk_difference']:.3f}"
        )

    print("\n" + "=" * 80)

    return results
