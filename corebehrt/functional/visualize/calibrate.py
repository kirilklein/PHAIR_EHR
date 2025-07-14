import os
from os.path import join
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from CausalEstimate.vis.plotting import plot_hist_by_groups
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score

from corebehrt.constants.causal.data import (
    EXPOSURE_COL,
    PROBAS,
    PS_COL,
    TARGETS,
)


def plot_probas_hist(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    group_labels: Tuple[str, str],
    title: str,
    xlabel: str,
    name: str,
    save_dir: str,
) -> None:
    """
    Plot the histogram of a value column grouped by a group column.
    Wrapper around CausalEstimate.vis.plotting.plot_hist_by_groups.
    """
    bin_edges = np.linspace(
        df[value_col].quantile(0.01), df[value_col].quantile(0.99), 51
    )
    fig, ax = plot_hist_by_groups(
        df=df,
        value_col=value_col,
        group_col=group_col,
        group_values=(0, 1),
        group_labels=group_labels,
        bin_edges=bin_edges,
        normalize=True,
        title=title,
        xlabel=xlabel,
    )
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.legend(title=group_col)
    fig.savefig(join(save_dir, f"{name}_hist.png"), bbox_inches="tight")
    plt.close(fig)


def plot_cf_probas_diff_vs_certainty_in_exposure(
    df: pd.DataFrame, save_dir: str, y_col: str
) -> None:
    """
    Plot the difference between counterfactual and factual probabilities vs certainty in actual exposure.
    """
    group_mask = df[EXPOSURE_COL] == 1
    # Compute x-axis as certainty in actual exposure
    df["x"] = np.where(group_mask, df[PS_COL] - 0.5, (1 - df[PS_COL]) - 0.5)

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        df.loc[group_mask, "x"],
        df.loc[group_mask, y_col],
        color="#e74c3c",  # red
        alpha=0.7,
        label="Exposed",
        s=1,
    )
    ax.scatter(
        df.loc[~group_mask, "x"],
        df.loc[~group_mask, y_col],
        color="#3498db",  # blue
        alpha=0.7,
        label="Control",
        s=1,
    )

    # Label axes
    ax.set_xlabel("Certainty in Actual Exposure")
    ax.set_ylabel("Counterfactual - Factual")

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(f"Counterfactual - Factual vs Certainty in Exposure")
    # Add legend without frame
    ax.legend(frameon=False, loc=0, title="Exposure")

    fig.savefig(
        join(save_dir, f"cf_diff_vs_certainty_in_exposure_by_exposure.png"),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_cf_diff_vs_probas_by_group(
    df: pd.DataFrame,
    save_dir: str,
    group_col: str,
    proba_col: str,
    group_labels: Tuple[str, str],
    y_col: str,
) -> None:
    """
    Plot the difference between counterfactual and factual probabilities vs outcome probability.
    Group columns used to get the group mask: group==1 first, group==0 second
    Group labels: 0 = Control/Negative, 1 = Exposed/Positive
    """
    group_mask = df[group_col] == 1
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        df.loc[group_mask, proba_col],
        df.loc[group_mask, y_col],
        color="#e74c3c",  # red
        alpha=0.6,
        label=group_labels[0],
        s=1,
    )
    ax.scatter(
        df.loc[~group_mask, proba_col],
        df.loc[~group_mask, y_col],
        color="#3498db",  # blue
        alpha=0.6,
        label=group_labels[1],
        s=1,
    )

    # Label axes
    ax.set_xlabel(f"{proba_col}")
    ax.set_ylabel("Counterfactual - Factual")

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_title(f"Diff vs {proba_col} by {group_col}")
    # Add legend without frame
    ax.legend(frameon=False)

    fig.savefig(
        join(save_dir, f"cf_diff_vs_{proba_col}_by_{group_col}.png"),
        bbox_inches="tight",
    )
    plt.close(fig)


def produce_calibration_plots(
    df_calibrated: pd.DataFrame, df: pd.DataFrame, fig_dir: str, title: str, name: str
) -> None:
    """
    Produce calibration plots for the original and calibrated probabilities.
    Args:
        df_calibrated: Dataframe with columns:
        - subject_id
        - probas
        - targets
    """

    cal_fig_dir = join(fig_dir, "calibration")
    os.makedirs(cal_fig_dir, exist_ok=True)

    n_bins = 20
    prob_true_orig, prob_pred_orig = calibration_curve(
        df[TARGETS],
        df[PROBAS],
        n_bins=n_bins,
        strategy="quantile",
    )
    prob_true_calibrated, prob_pred_calibrated = calibration_curve(
        df_calibrated[TARGETS],
        df_calibrated[PROBAS],
        n_bins=n_bins,
        strategy="quantile",
    )
    fig, ax = plt.subplots(figsize=(6, 6))

    cal_error_orig = brier_score_loss(df[TARGETS], df[PROBAS])
    cal_error_calibrated = brier_score_loss(
        df_calibrated[TARGETS], df_calibrated[PROBAS]
    )
    roc_auc_orig = roc_auc_score(df[TARGETS], df[PROBAS])
    roc_auc_calibrated = roc_auc_score(df_calibrated[TARGETS], df_calibrated[PROBAS])
    ax.plot(
        prob_pred_orig,
        prob_true_orig,
        label=f"Before Calibration (BS: {cal_error_orig:.3f}, AUC: {roc_auc_orig:.3f})",
        marker="o",
        color="tab:blue",
    )
    ax.plot(
        prob_pred_calibrated,
        prob_true_calibrated,
        label=f"After Calibration (BS: {cal_error_calibrated:.3f}, AUC: {roc_auc_calibrated:.3f})",
        marker="o",
        color="tab:orange",
    )

    ax.plot([0, 1], [0, 1], linestyle="--", label="Ideal", color="black")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(title)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect("equal")
    ax.legend()
    fig.savefig(join(cal_fig_dir, f"{name}.png"), bbox_inches="tight")
    plt.close(fig)
