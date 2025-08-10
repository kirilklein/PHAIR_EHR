from typing import Tuple

import matplotlib as mpl
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score

from corebehrt.constants.causal.data import (
    EXPOSURE_COL,
    PROBAS,
    PS_COL,
    TARGETS,
)


def plot_weights_hist(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    group_labels: Tuple[str, str],
    title: str,
    xlabel: str,
    ax: mpl.axes.Axes,
    min_quantile: float = 0.001,
    max_quantile: float = 0.999,
    num_bins: int = 51,
    y_max_quantile: float = 0.99,
) -> None:
    """
    Plot the histogram of a value column grouped by a group column.
    Clips the y-axis to a quantile of the histogram counts to avoid outliers.
    """
    bin_edges = np.linspace(
        df[value_col].quantile(min_quantile),
        df[value_col].quantile(max_quantile),
        num_bins,
    )

    group0_data = df[df[group_col] == 0][value_col]
    group1_data = df[df[group_col] == 1][value_col]

    # Calculate histograms to determine y-axis limit
    counts0, _ = np.histogram(group0_data, bins=bin_edges, density=True)
    counts1, _ = np.histogram(group1_data, bins=bin_edges, density=True)
    max_count = max(
        np.quantile(counts0, y_max_quantile), np.quantile(counts1, y_max_quantile)
    )
    if max_count == 0:
        max_count = max(counts0.max(), counts1.max())

    ax.hist(
        group0_data,
        bins=bin_edges,
        label=group_labels[0],
        density=True,
        alpha=0.7,
        color="#3498db",
    )
    ax.hist(
        group1_data,
        bins=bin_edges,
        label=group_labels[1],
        density=True,
        alpha=0.7,
        color="#e74c3c",
    )

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.legend(title=group_col)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylim(0, max_count * 1.1)


def plot_probas_hist(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    group_labels: Tuple[str, str],
    title: str,
    xlabel: str,
    ax: mpl.axes.Axes,
    min_quantile: float = 0.001,
    max_quantile: float = 0.999,
    num_bins: int = 51,
) -> None:
    """
    Plot the histogram of a value column grouped by a group column.
    """
    bin_edges = np.linspace(
        df[value_col].quantile(min_quantile),
        df[value_col].quantile(max_quantile),
        num_bins,
    )

    group0_data = df[df[group_col] == 0][value_col]
    group1_data = df[df[group_col] == 1][value_col]

    ax.hist(
        group0_data,
        bins=bin_edges,
        label=group_labels[0],
        density=True,
        alpha=0.7,
        color="#3498db",
    )  # blue
    ax.hist(
        group1_data,
        bins=bin_edges,
        label=group_labels[1],
        density=True,
        alpha=0.7,
        color="#e74c3c",
    )  # red

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.legend(title=group_col)
    ax.set_title(title)
    ax.set_xlabel(xlabel)


def plot_cf_probas_diff_vs_certainty_in_exposure(
    df: pd.DataFrame, y_col: str, ax: mpl.axes.Axes
) -> None:
    """
    Plot the difference between counterfactual and factual probabilities vs certainty in actual exposure.
    """
    group_mask = df[EXPOSURE_COL] == 1
    # Compute x-axis as certainty in actual exposure
    df["x"] = np.where(group_mask, df[PS_COL] - 0.5, (1 - df[PS_COL]) - 0.5)

    # Create the plot
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
    ax.set_title("Counterfactual - Factual vs Certainty in Exposure")
    # Add legend without frame
    ax.legend(frameon=False, loc=0, title="Exposure")


def plot_cf_diff_vs_probas_by_group(
    df: pd.DataFrame,
    group_col: str,
    proba_col: str,
    group_labels: Tuple[str, str],
    y_col: str,
    ax: mpl.axes.Axes,
) -> None:
    """
    Plot the difference between counterfactual and factual probabilities vs outcome probability.
    Group columns used to get the group mask: group==1 first, group==0 second
    Group labels: 0 = Control/Negative, 1 = Exposed/Positive
    """
    group_mask = df[group_col] == 1
    # Create the plot
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


def produce_calibration_plots(
    df_calibrated: pd.DataFrame, df: pd.DataFrame, name: str, ax: mpl.axes.Axes
) -> None:
    """
    Produce calibration plots for the original and calibrated probabilities.
    Args:
        df_calibrated: Dataframe with columns:
        - subject_id
        - probas
        - targets
    """
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

    cal_error_orig = brier_score_loss(df[TARGETS], df[PROBAS])
    cal_error_calibrated = brier_score_loss(
        df_calibrated[TARGETS], df_calibrated[PROBAS]
    )
    roc_auc_orig = roc_auc_score(df[TARGETS], df[PROBAS])
    roc_auc_calibrated = roc_auc_score(df_calibrated[TARGETS], df_calibrated[PROBAS])
    ax.plot(
        prob_pred_orig,
        prob_true_orig,
        label=f"Before (BS: {cal_error_orig:.3f}, AUC: {roc_auc_orig:.3f})",
        marker="o",
        color="tab:blue",
    )
    ax.plot(
        prob_pred_calibrated,
        prob_true_calibrated,
        label=f"After (BS: {cal_error_calibrated:.3f}, AUC: {roc_auc_calibrated:.3f})",
        marker="o",
        color="tab:orange",
    )

    ax.plot([0, 1], [0, 1], linestyle="--", label="Ideal", color="black")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(name)
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    ax.set_aspect("equal")
    ax.legend()
