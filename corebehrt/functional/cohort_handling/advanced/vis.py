from typing import Dict, Tuple, List
import os

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import logging
from corebehrt.constants.data import PID_COL, AGE_COL, TIMESTAMP_COL
from corebehrt.functional.utils.azure_save import save_figure_with_azure_copy
from matplotlib.axes import Axes
from matplotlib.container import BarContainer


def plot_cohort_stats(
    stats: Dict,
    title: str = "Cohort Selection Flow",
    figsize: Tuple[int, int] = (16, 10),
    save_path: str = None,
    show_plot: bool = True,
) -> None:
    """
    Create a comprehensive visualization of cohort selection statistics.

    Args:
        stats: Dictionary containing cohort selection statistics
        title: Title for the plot
        figsize: Figure size (width, height)
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot

    Example:
        >>> stats = {
        ...     "initial_total": 629,
        ...     "excluded_by_inclusion_criteria": {"min_age_0": 0, "criteria_1": 629, "total_excluded": 1},
        ...     "excluded_by_exclusion_criteria": {"min_age_99": 0, "total_excluded": 0},
        ...     "n_excluded_by_expression": 0,
        ...     "final_included": 629
        ... }
        >>> plot_cohort_stats(stats, "Exposed Patients Cohort Selection")
    """
    fig = plt.figure(figsize=figsize)

    # Create a 2x2 grid with the top row spanning for the flow chart
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.3, wspace=0.3)
    ax_flow = fig.add_subplot(gs[0, :])  # Top row spans both columns
    ax_inclusion = fig.add_subplot(gs[1, 0])  # Bottom left
    ax_exclusion = fig.add_subplot(gs[1, 1])  # Bottom right

    fig.suptitle(title, fontsize=16, fontweight="bold")

    # 1. Step-by-step flow chart
    _plot_step_by_step_flow(ax_flow, stats)

    # 2. Individual criteria breakdown
    _plot_individual_criteria_breakdown(ax_inclusion, ax_exclusion, stats)

    plt.tight_layout()

    if save_path:
        try:
            save_figure_with_azure_copy(
                fig, save_path, dpi=300, bbox_inches="tight", close=False
            )
            print(f"Plot saved to: {save_path}")
        except Exception as e:
            print(f"Error saving plot: {e}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def _plot_step_by_step_flow(ax: Axes, stats: Dict) -> None:
    """Plot the step-by-step patient flow through the selection process."""
    initial = stats.get("initial_total", 0)
    final = stats.get("final_included", 0)

    # Get totals reported by the stats object
    inclusion_stats = stats.get("excluded_by_inclusion_criteria", {})
    total_excluded_by_inclusion = inclusion_stats.get("total_excluded", 0)

    # Patients remaining after inclusion is simply those meeting the inclusion expression
    after_inclusion = initial - total_excluded_by_inclusion

    # Patients excluded at the exclusion step should be computed CONDITIONED on having
    # passed inclusion. This automatically accounts for overlaps between inclusion-violators
    # and exclusion-violators.
    excluded_at_exclusion_step = max(0, after_inclusion - final)

    # After exclusion equals the final included cohort
    # after_exclusion = after_inclusion - excluded_at_exclusion_step  # equals `final`

    # Use three bars to avoid confusion: Initial -> After Inclusion -> Final (After Exclusion)
    stages = [
        "Initial\nCohort",
        "After\nInclusion",
        "Final\nCohort",
    ]
    counts = [initial, after_inclusion, final]

    # Colors: blue for kept, progressively darker
    colors = ["#4A90E2", "#357ABD", "#2E6B9E", "#1F4F79"]

    # Create bars
    x_positions = np.arange(len(stages))
    bars: BarContainer = ax.bar(
        x_positions, counts, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5
    )

    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(counts) * 0.02,
            f"{count:,}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12,
        )

    # Add exclusion arrows and labels
    max_height = max(counts)
    arrow_height = max_height * 0.6

    # Inclusion exclusion arrow
    if total_excluded_by_inclusion > 0:
        ax.annotate(
            f"Excluded: {total_excluded_by_inclusion:,}",
            xy=(0.5, arrow_height),
            xytext=(0.5, arrow_height + max_height * 0.15),
            arrowprops=dict(arrowstyle="->", color="red", lw=3),
            ha="center",
            va="bottom",
            color="red",
            fontweight="bold",
            fontsize=11,
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", alpha=0.9
            ),
        )

    # Exclusion arrow (conditioned on having passed inclusion)
    if excluded_at_exclusion_step > 0:
        ax.annotate(
            f"Excluded: {excluded_at_exclusion_step:,}",
            xy=(1.5, arrow_height),
            xytext=(1.5, arrow_height + max_height * 0.15),
            arrowprops=dict(arrowstyle="->", color="red", lw=3),
            ha="center",
            va="bottom",
            color="red",
            fontweight="bold",
            fontsize=11,
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", alpha=0.9
            ),
        )

    # Styling
    ax.set_xticks(x_positions)
    ax.set_xticklabels(stages, fontsize=12, fontweight="bold")
    ax.set_title(
        "Patient Flow Through Selection Process", fontweight="bold", fontsize=14, pad=20
    )
    ax.set_ylabel("Number of Patients", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Add summary statistics box
    retention_rate = (final / initial * 100) if initial > 0 else 0
    summary_text = f"Overall Statistics:\n"
    summary_text += f"• Initial: {initial:,}\n"
    summary_text += f"• Final: {final:,}\n"
    summary_text += f"• Retention: {retention_rate:.1f}%"

    ax.text(
        1.0,
        1.0,
        summary_text,
        transform=ax.transAxes,
        verticalalignment="bottom",
        horizontalalignment="right",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8, edgecolor="navy"
        ),
        fontweight="bold",
    )


def _plot_individual_criteria_breakdown(
    ax_inclusion: Axes, ax_exclusion: Axes, stats: Dict
) -> None:
    """Plot individual criteria showing how many patients MET each criterion."""
    inclusion_criteria = stats.get("excluded_by_inclusion_criteria", {})
    exclusion_criteria = stats.get("excluded_by_exclusion_criteria", {})
    initial_total = stats.get("initial_total", 0)

    # Plot inclusion criteria (show patients who MET the criteria)
    _plot_criteria_met(
        ax_inclusion,
        inclusion_criteria,
        initial_total,
        "Inclusion",
        "#2E8B57",
        is_inclusion=True,
    )

    # Plot exclusion criteria (show patients who MET the criteria)
    _plot_criteria_met(
        ax_exclusion,
        exclusion_criteria,
        initial_total,
        "Exclusion",
        "#DC143C",
        is_inclusion=False,
    )


def _plot_criteria_met(
    ax: Axes,
    criteria_stats: Dict,
    initial_total: int,
    criteria_type: str,
    color: str,
    is_inclusion: bool,
) -> None:
    """Plot how many patients met each individual criterion."""
    # Remove 'total_excluded' from individual criteria
    individual_criteria = {
        k: v for k, v in criteria_stats.items() if k != "total_excluded"
    }

    if not individual_criteria:
        ax.text(
            0.5,
            0.5,
            f"No {criteria_type.lower()}\ncriteria applied",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
        )
        ax.set_title(f"{criteria_type} Criteria", fontweight="bold", fontsize=12)
        ax.axis("off")
        return

    criteria_names = list(individual_criteria.keys())

    if is_inclusion:
        # For inclusion criteria: stats show "excluded by" so patients who MET = total - excluded
        patients_met = [initial_total - count for count in individual_criteria.values()]
        ylabel = "Patients Meeting Criterion"
    else:
        # For exclusion criteria: stats show "excluded by" which means patients who met the exclusion
        patients_met = list(individual_criteria.values())
        ylabel = "Patients Meeting Criterion"

    # Sort by count (descending)
    sorted_data = sorted(
        zip(criteria_names, patients_met), key=lambda x: x[1], reverse=True
    )
    criteria_names, patients_met = zip(*sorted_data) if sorted_data else ([], [])

    if not criteria_names:
        ax.text(
            0.5,
            0.5,
            f"No {criteria_type.lower()}\ncriteria data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_title(f"{criteria_type} Criteria", fontweight="bold", fontsize=12)
        return

    # Create horizontal bar chart
    y_positions = np.arange(len(criteria_names))
    bars = ax.barh(
        y_positions,
        patients_met,
        color=color,
        alpha=0.7,
        edgecolor="black",
        linewidth=1,
    )

    # Add value labels
    max_count = max(patients_met) if patients_met else 0
    for bar, count in zip(bars, patients_met):
        width = bar.get_width()
        ax.text(
            width + max_count * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,}",
            ha="left",
            va="center",
            fontweight="bold",
            fontsize=10,
        )

        # Add percentage
        if initial_total > 0:
            percentage = (count / initial_total) * 100
            ax.text(
                width * 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{percentage:.1f}%",
                ha="center",
                va="center",
                fontweight="bold",
                fontsize=9,
                color="white" if count > max_count * 0.3 else "black",
            )

    # Styling
    ax.set_yticks(y_positions)
    ax.set_yticklabels(criteria_names, fontsize=10)
    ax.set_xlabel(ylabel, fontsize=11, fontweight="bold")
    ax.set_title(
        f"{criteria_type} Criteria\n(Patients Meeting Each)",
        fontweight="bold",
        fontsize=12,
    )
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    # Invert y-axis to show highest counts at top
    ax.invert_yaxis()


def plot_multiple_cohort_stats(
    stats_dict: Dict[str, Dict],
    figsize: Tuple[int, int] = (20, 12),
    save_path: str = None,
    show_plot: bool = True,
) -> None:
    """
    Plot statistics for multiple cohorts (e.g., exposed vs control) side by side.

    Args:
        stats_dict: Dictionary with cohort names as keys and stats as values
        figsize: Figure size (width, height)
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot

    Example:
        >>> stats_dict = {
        ...     "exposed": exposed_stats,
        ...     "control": control_stats
        ... }
        >>> plot_multiple_cohort_stats(stats_dict)
    """
    if not stats_dict:
        print("No statistics provided for comparison plot.")
        return

    n_cohorts = len(stats_dict)

    # Ensure figsize is properly formatted as tuple of integers
    if not isinstance(figsize, (tuple, list)) or len(figsize) != 2:
        figsize = (20, 12)
    figsize = (int(figsize[0]), int(figsize[1]))

    # Create figure with proper subplot layout
    fig = plt.figure(figsize=figsize)
    fig.suptitle("Cohort Selection Comparison", fontsize=18, fontweight="bold")

    # Create a grid for each cohort (3 columns: flow, inclusion, exclusion)
    n_rows = n_cohorts
    n_cols = 3

    for i, (cohort_name, stats) in enumerate(stats_dict.items()):
        # Flow chart (wider)
        ax_flow = plt.subplot2grid((n_rows, n_cols), (i, 0), colspan=1, fig=fig)
        _plot_step_by_step_flow(ax_flow, stats)
        ax_flow.set_title(
            f"{cohort_name.title()} - Patient Flow", fontweight="bold", fontsize=14
        )

        # Inclusion criteria
        ax_inclusion = plt.subplot2grid((n_rows, n_cols), (i, 1), colspan=1, fig=fig)

        # Exclusion criteria
        ax_exclusion = plt.subplot2grid((n_rows, n_cols), (i, 2), colspan=1, fig=fig)

        _plot_individual_criteria_breakdown(ax_inclusion, ax_exclusion, stats)

    plt.tight_layout()

    if save_path:
        try:
            save_figure_with_azure_copy(
                fig, save_path, dpi=300, bbox_inches="tight", close=False
            )
            print(f"Comparison plot saved to: {save_path}")
        except Exception as e:
            print(f"Error saving plot: {e}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_age_distribution(
    final_index_dates_with_age: pd.DataFrame,
    control_pids: List[int],
    save_path: str,
    logger: logging.Logger,
):
    """Plots the age distribution for exposed vs. control from a pre-calculated age column.

    Creates three plots:
    - Combined plot with both exposed and control groups
    - Separate plot for exposed patients only
    - Separate plot for control patients only
    """
    logger.info("Plotting age at index date distribution.")
    plot_df = final_index_dates_with_age.copy()

    # 1. Create a 'group' column by checking if a patient is in the control list
    plot_df["group"] = np.where(
        plot_df[PID_COL].isin(control_pids), "Control", "Exposed"
    )

    # 2. Create and save the combined histogram plot
    plt.figure(figsize=(12, 7))
    sns.histplot(
        data=plot_df, x=AGE_COL, hue="group", kde=True, bins=40, element="step"
    )
    plt.title("Age Distribution at Index Date (Final Cohorts)")
    plt.xlabel("Age (years)")
    plt.ylabel("Patient Count")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    save_figure_with_azure_copy(plt.gcf(), save_path, dpi=300)
    logger.info(f"Saved combined age distribution plot to {save_path}")

    # 3. Create and save exposed-only plot
    base_path, ext = os.path.splitext(save_path)
    exposed_save_path = f"{base_path}_exposed{ext}"

    exposed_df = plot_df[plot_df["group"] == "Exposed"]
    plt.figure(figsize=(12, 7))
    sns.histplot(
        data=exposed_df, x=AGE_COL, kde=True, bins=40, element="step", color="steelblue"
    )
    plt.title("Age Distribution at Index Date (Exposed)")
    plt.xlabel("Age (years)")
    plt.ylabel("Patient Count")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    save_figure_with_azure_copy(plt.gcf(), exposed_save_path, dpi=300)
    logger.info(f"Saved exposed age distribution plot to {exposed_save_path}")

    # 4. Create and save control-only plot
    control_save_path = f"{base_path}_control{ext}"

    control_df = plot_df[plot_df["group"] == "Control"]
    plt.figure(figsize=(12, 7))
    sns.histplot(
        data=control_df,
        x=AGE_COL,
        kde=True,
        bins=40,
        element="step",
        color="darkorange",
    )
    plt.title("Age Distribution at Index Date (Control)")
    plt.xlabel("Age (years)")
    plt.ylabel("Patient Count")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    save_figure_with_azure_copy(plt.gcf(), control_save_path, dpi=300)
    logger.info(f"Saved control age distribution plot to {control_save_path}")


def plot_index_date_distribution(
    final_index_dates: pd.DataFrame,
    control_pids: List[int],
    save_path: str,
    logger: logging.Logger,
):
    """Plots the index date distribution for exposed vs. control from a pre-calculated index date column."""
    logger.info("Plotting index date distribution.")
    plot_df = final_index_dates.copy()

    # 1. Create a 'group' column by checking if a patient is in the control list
    plot_df["group"] = np.where(
        plot_df[PID_COL].isin(control_pids), "Control", "Exposed"
    )

    # 2. Create and save the histogram plot
    plt.figure(figsize=(12, 7))
    try:
        sns.histplot(
            data=plot_df,
            x=TIMESTAMP_COL,
            hue="group",
            kde=True,
            bins=40,
            element="step",
        )
    except Exception as e:
        print(f"Error plotting index date distribution: {e}")
        sns.histplot(
            data=plot_df,
            x=TIMESTAMP_COL,
            hue="group",
            kde=False,
            bins=40,
            element="step",
        )
    plt.title("Index Date Distribution (Final Cohorts)")
    plt.xlabel("Index Date")
    plt.ylabel("Patient Count")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    save_figure_with_azure_copy(plt.gcf(), save_path, dpi=300)

    logger.info(f"Saved index date distribution plot to {save_path}")
