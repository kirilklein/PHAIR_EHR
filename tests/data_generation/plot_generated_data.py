import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from corebehrt.constants.causal.data import (
    EXPOSURE_COL,
    PROBAS,
    PS_COL,
    SIMULATED_PROBAS_CONTROL,
    SIMULATED_PROBAS_EXPOSED,
    TARGETS,
)
from corebehrt.constants.data import PID_COL
from corebehrt.functional.utils.azure_save import save_figure_with_azure_copy


def save_ps_distribution_figure(subject_ids, ps_scores, exposures):
    """
    Create and save a figure showing the distribution of propensity scores
    colored by exposure status.
    """
    plt.figure(figsize=(10, 6))

    # Create a DataFrame for easier plotting
    plot_df = pd.DataFrame(
        {PID_COL: subject_ids, PS_COL: ps_scores, EXPOSURE_COL: exposures}
    )

    # Plot histogram of propensity scores colored by exposure
    sns.histplot(
        data=plot_df,
        x=PS_COL,
        hue=EXPOSURE_COL,
        bins=30,
        element="step",
        common_norm=False,
        alpha=0.6,
    )

    plt.title("Distribution of Propensity Scores by Exposure Status")
    plt.xlabel("Propensity Score")
    plt.ylabel("Count")

    # Create output directory if it doesn't exist
    figures_dir = "./outputs/causal/figures"
    Path(figures_dir).mkdir(parents=True, exist_ok=True)

    # Save figure
    save_figure_with_azure_copy(
        plt.gcf(),
        os.path.join(figures_dir, "ps_distribution.png"),
        dpi=300,
        bbox_inches="tight",
    )
    print(
        f"Propensity score distribution figure saved to {figures_dir}/ps_distribution.png"
    )


def save_outcome_probas_figures(subject_ids, cf_data, outcome_df):
    """
    Create and save figures showing:
    1. True outcome probabilities colored by exposure.
    2. Predicted outcome probabilities by actual outcome.
    """
    figures_dir = "./outputs/causal/figures"
    Path(figures_dir).mkdir(parents=True, exist_ok=True)

    # ---------------------------
    # Figure 1: True outcome probabilities by exposure
    # ---------------------------
    plt.figure(figsize=(10, 6))

    # Extract true probabilities from the simulation outcomes file (cf_data)
    true_p0 = cf_data[SIMULATED_PROBAS_CONTROL].values
    true_p1 = cf_data[SIMULATED_PROBAS_EXPOSED].values
    exposures = cf_data[EXPOSURE_COL].values
    observed_probas = cf_data[PROBAS].values

    true_plot_df = pd.DataFrame(
        {
            PID_COL: subject_ids,
            "True P(Y=1|do(A=0))": true_p0,
            "True P(Y=1|do(A=1))": true_p1,
            "Observed P(Y=1)": observed_probas,
            EXPOSURE_COL: exposures,
        }
    )

    true_plot_df_melted = pd.melt(
        true_plot_df,
        id_vars=[PID_COL, EXPOSURE_COL],
        value_vars=["True P(Y=1|do(A=0))", "True P(Y=1|do(A=1))", "Observed P(Y=1)"],
        var_name="Probability Type",
        value_name="Probability",
    )

    sns.boxplot(
        data=true_plot_df_melted,
        x="Probability Type",
        y="Probability",
        hue=EXPOSURE_COL,
    )
    plt.title("True Outcome Probabilities by Exposure Status")
    plt.ylabel("Probability")
    plt.xticks(rotation=45)
    save_figure_with_azure_copy(
        plt.gcf(),
        os.path.join(figures_dir, "true_outcome_probas.png"),
        dpi=300,
        bbox_inches="tight",
    )

    # ---------------------------
    # Figure 2: Predicted outcome probabilities by actual outcome
    # (Using the predictions file: outcome_df, which now contains 'probas' and 'cf_probas')
    # ---------------------------
    plt.figure(figsize=(10, 6))

    # Extract predicted probabilities from outcome_df
    pred_factual = outcome_df[PROBAS].values
    pred_counterfactual = outcome_df["cf_probas"].values
    outcomes = outcome_df[TARGETS].values

    pred_plot_df = pd.DataFrame(
        {
            PID_COL: subject_ids,
            "Pred P(Y=1|factual)": pred_factual,
            "Pred P(Y=1|counterfactual)": pred_counterfactual,
            "Actual Outcome": outcomes,
        }
    )

    pred_plot_df_melted = pd.melt(
        pred_plot_df,
        id_vars=[PID_COL, "Actual Outcome"],
        value_vars=["Pred P(Y=1|factual)", "Pred P(Y=1|counterfactual)"],
        var_name="Probability Type",
        value_name="Probability",
    )

    sns.boxplot(
        data=pred_plot_df_melted,
        x="Probability Type",
        y="Probability",
        hue="Actual Outcome",
    )
    plt.title("Predicted Outcome Probabilities by Actual Outcome")
    plt.ylabel("Probability")
    plt.xticks(rotation=45)
    save_figure_with_azure_copy(
        plt.gcf(),
        os.path.join(figures_dir, "predicted_outcome_probas.png"),
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Outcome probability figures saved to {figures_dir}")


def save_comparison_figures(subject_ids, cf_data, outcome_df):
    """
    Create and save side-by-side comparison plots of true vs. predicted probabilities.
    For the predicted outcomes, we derive the control/treatment probabilities using:
      - For subjects with exposure=0: factual (probas) is the control prediction, cf_probas is the treatment prediction.
      - For subjects with exposure=1: factual (probas) is the treatment prediction, cf_probas is the control prediction.
    """
    figures_dir = "./outputs/causal/figures"
    Path(figures_dir).mkdir(parents=True, exist_ok=True)

    exposures = cf_data[EXPOSURE_COL].values
    # True probabilities from simulation data
    true_p0 = cf_data[SIMULATED_PROBAS_CONTROL].values
    true_p1 = cf_data[SIMULATED_PROBAS_EXPOSED].values

    # Compute predicted control and treatment probabilities from predictions
    pred_probas = outcome_df[PROBAS].values
    pred_cf_probas = outcome_df["cf_probas"].values
    pred_control = np.where(exposures == 0, pred_probas, pred_cf_probas)
    pred_treatment = np.where(exposures == 1, pred_probas, pred_cf_probas)

    # ---------------------------
    # Comparison for control (A=0) probabilities
    # ---------------------------
    plt.figure(figsize=(12, 6))

    comparison_df = pd.DataFrame(
        {
            PID_COL: subject_ids,
            "True P(Y=1|do(A=0))": true_p0,
            "Predicted P(Y=1|do(A=0))": pred_control,
            EXPOSURE_COL: exposures,
        }
    )

    plt.subplot(1, 2, 1)
    sns.scatterplot(
        data=comparison_df,
        x="True P(Y=1|do(A=0))",
        y="Predicted P(Y=1|do(A=0))",
        hue=EXPOSURE_COL,
        alpha=0.6,
    )
    plt.plot([0, 1], [0, 1], "k--")  # Diagonal reference line
    plt.title("Control Outcome Probabilities (A=0)")
    plt.xlabel("True Probability")
    plt.ylabel("Predicted Probability")

    plt.subplot(1, 2, 2)
    differences = pred_control - true_p0
    sns.histplot(differences, bins=30)
    plt.axvline(x=0, color="r", linestyle="--")
    plt.title("Prediction Error Distribution (A=0)")
    plt.xlabel("Predicted - True Probability")
    plt.ylabel("Count")

    plt.tight_layout()
    save_figure_with_azure_copy(
        plt.gcf(),
        os.path.join(figures_dir, "control_probas_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )

    # ---------------------------
    # Comparison for treatment (A=1) probabilities
    # ---------------------------
    plt.figure(figsize=(12, 6))

    comparison_df = pd.DataFrame(
        {
            PID_COL: subject_ids,
            "True P(Y=1|do(A=1))": true_p1,
            "Predicted P(Y=1|do(A=1))": pred_treatment,
            EXPOSURE_COL: exposures,
        }
    )

    plt.subplot(1, 2, 1)
    sns.scatterplot(
        data=comparison_df,
        x="True P(Y=1|do(A=1))",
        y="Predicted P(Y=1|do(A=1))",
        hue=EXPOSURE_COL,
        alpha=0.6,
    )
    plt.plot([0, 1], [0, 1], "k--")
    plt.title("Treatment Outcome Probabilities (A=1)")
    plt.xlabel("True Probability")
    plt.ylabel("Predicted Probability")

    plt.subplot(1, 2, 2)
    differences = pred_treatment - true_p1
    sns.histplot(differences, bins=30)
    plt.axvline(x=0, color="r", linestyle="--")
    plt.title("Prediction Error Distribution (A=1)")
    plt.xlabel("Predicted - True Probability")
    plt.ylabel("Count")

    plt.tight_layout()
    save_figure_with_azure_copy(
        plt.gcf(),
        os.path.join(figures_dir, "treatment_probas_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )

    # ---------------------------
    # Comparison for observed outcome probabilities
    # ---------------------------
    plt.figure(figsize=(12, 6))

    comparison_df = pd.DataFrame(
        {
            PID_COL: subject_ids,
            "True Observed P(Y=1)": cf_data[PROBAS].values,
            "Predicted Observed P(Y=1)": outcome_df[PROBAS].values,
            EXPOSURE_COL: exposures,
        }
    )

    plt.subplot(1, 2, 1)
    sns.scatterplot(
        data=comparison_df,
        x="True Observed P(Y=1)",
        y="Predicted Observed P(Y=1)",
        hue=EXPOSURE_COL,
        alpha=0.6,
    )
    plt.plot([0, 1], [0, 1], "k--")
    plt.title("Observed Outcome Probabilities")
    plt.xlabel("True Probability")
    plt.ylabel("Predicted Probability")

    plt.subplot(1, 2, 2)
    differences = outcome_df[PROBAS].values - cf_data[PROBAS].values
    sns.histplot(differences, bins=30)
    plt.axvline(x=0, color="r", linestyle="--")
    plt.title("Prediction Error Distribution (Observed)")
    plt.xlabel("Predicted - True Probability")
    plt.ylabel("Count")

    plt.tight_layout()
    save_figure_with_azure_copy(
        plt.gcf(),
        os.path.join(figures_dir, "observed_probas_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Probability comparison figures saved to {figures_dir}")


def save_predicted_outcome_probas_distribution(cf_data, outcome_df):
    """
    Create and save a figure showing the distribution of the predicted (factual)
    outcome probabilities, colored by actual exposure.
    """
    figures_dir = "./outputs/causal/figures"
    Path(figures_dir).mkdir(parents=True, exist_ok=True)

    # Merge the exposure column from simulation data into the predictions for coloring
    merged_df = outcome_df.merge(
        cf_data[[PID_COL, EXPOSURE_COL]], on=PID_COL, how="left"
    )

    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=merged_df,
        x=PROBAS,  # 'probas' is the factual predicted probability
        hue=EXPOSURE_COL,
        bins=30,
        element="step",
        common_norm=False,
        alpha=0.6,
    )
    plt.title("Distribution of Predicted Outcome Probabilities by Exposure Status")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")

    save_figure_with_azure_copy(
        plt.gcf(),
        os.path.join(figures_dir, "predicted_outcome_probas_distribution.png"),
        dpi=300,
        bbox_inches="tight",
    )
    print(
        f"Predicted outcome probability distribution saved to {os.path.join(figures_dir, 'predicted_outcome_probas_distribution.png')}"
    )


def save_outcome_probas_by_exposure_figure(subject_ids, cf_data, outcome_df):
    """
    Create and save figures showing outcome probabilities grouped by exposure status.
    For both true and predicted probabilities.
    """
    figures_dir = "./outputs/causal/figures"
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    exposures = cf_data[EXPOSURE_COL].values

    # ---------------------------
    # True probabilities by exposure
    # ---------------------------
    plt.figure(figsize=(12, 6))

    true_p0 = cf_data[SIMULATED_PROBAS_CONTROL].values
    true_p1 = cf_data[SIMULATED_PROBAS_EXPOSED].values
    observed_probas = cf_data[PROBAS].values

    plot_df = pd.DataFrame(
        {
            PID_COL: subject_ids,
            "True P(Y=1|do(A=0))": true_p0,
            "True P(Y=1|do(A=1))": true_p1,
            "True Observed P(Y=1)": observed_probas,
            "Exposure": ["Treated" if exp == 1 else "Untreated" for exp in exposures],
        }
    )

    sns.boxplot(
        data=plot_df.melt(
            id_vars=[PID_COL, "Exposure"],
            value_vars=[
                "True P(Y=1|do(A=0))",
                "True P(Y=1|do(A=1))",
                "True Observed P(Y=1)",
            ],
            var_name="Probability Type",
            value_name="Probability",
        ),
        x="Exposure",
        y="Probability",
        hue="Probability Type",
    )

    plt.title("True Outcome Probabilities Grouped by Exposure Status")
    plt.xlabel("Exposure Status")
    plt.ylabel("Probability")
    plt.legend(title="Probability Type")
    save_figure_with_azure_copy(
        plt.gcf(),
        os.path.join(figures_dir, "true_outcome_probas_by_exposure.png"),
        dpi=300,
        bbox_inches="tight",
    )

    # ---------------------------
    # Predicted probabilities by exposure
    # ---------------------------
    plt.figure(figsize=(12, 6))

    # Compute predicted control and treatment probabilities based on exposure:
    # For subjects with exposure==0: factual (probas) is the control prediction,
    # for subjects with exposure==1: factual is the treatment prediction.
    pred_probas = outcome_df[PROBAS].values
    pred_cf_probas = outcome_df["cf_probas"].values
    pred_control = np.where(exposures == 0, pred_probas, pred_cf_probas)
    pred_treatment = np.where(exposures == 1, pred_probas, pred_cf_probas)

    pred_plot_df = pd.DataFrame(
        {
            PID_COL: subject_ids,
            "Pred P(Y=1|do(A=0))": pred_control,
            "Pred P(Y=1|do(A=1))": pred_treatment,
            "Pred Observed P(Y=1)": pred_probas,
            "Exposure": ["Treated" if exp == 1 else "Untreated" for exp in exposures],
        }
    )

    sns.boxplot(
        data=pred_plot_df.melt(
            id_vars=[PID_COL, "Exposure"],
            value_vars=[
                "Pred P(Y=1|do(A=0))",
                "Pred P(Y=1|do(A=1))",
                "Pred Observed P(Y=1)",
            ],
            var_name="Probability Type",
            value_name="Probability",
        ),
        x="Exposure",
        y="Probability",
        hue="Probability Type",
    )

    plt.title("Predicted Outcome Probabilities Grouped by Exposure Status")
    plt.xlabel("Exposure Status")
    plt.ylabel("Probability")
    plt.legend(title="Probability Type")
    save_figure_with_azure_copy(
        plt.gcf(),
        os.path.join(figures_dir, "predicted_outcome_probas_by_exposure.png"),
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Outcome probabilities by exposure figures saved to {figures_dir}")
