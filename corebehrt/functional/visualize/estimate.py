import logging
import os
from os.path import join
from typing import List
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from CausalEstimate.estimators.functional.ipw import compute_ipw_weights

from corebehrt.constants.causal.data import (
    EffectColumns,
    TMLEAnalysisColumns,
    EXPOSURE_COL,
    PS_COL,
)
from corebehrt.modules.plot.estimate import (
    ContingencyPlotConfig,
    EffectSizePlotConfig,
    EffectSizePlotter,
    ContingencyTablePlotter,
    AdjustmentPlotConfig,
    AdjustmentPlotter,
)
from corebehrt.functional.utils.azure_save import save_figure_with_azure_copy

logger = logging.getLogger(__name__)


def create_annotated_heatmap_matplotlib(
    df: pd.DataFrame,
    method_names: list,
    effect_name: str = "effect",
    save_path: str = None,
):
    """
    Creates an annotated heatmap using Matplotlib from a list of effect dictionaries.

    Args:
        df: A pandas DataFrame containing effect data with 'method', 'outcome',
            and the specified 'effect_name' columns.
        method_names: A list of method names to be displayed on the y-axis, maintaining their order.
        effect_name: The key in the effect dictionaries to visualize (e.g., 'effect', 'std_err').
        save_path: Optional. A string representing the file path where the plot should be saved
                   (e.g., 'heatmap.png', 'plots/my_heatmap.pdf'). If None, the plot is displayed.
    """  # Ensure 'method' and 'outcome' columns exist
    if (
        EffectColumns.method not in df.columns
        or EffectColumns.outcome not in df.columns
    ):
        raise ValueError("The DataFrame must contain 'method' and 'outcome' columns.")
    if effect_name not in df.columns:
        raise ValueError(
            f"'{effect_name}' column not found in the DataFrame. "
            f"Please ensure the DataFrame contains this column."
        )
    # Pivot the DataFrame
    heatmap_data = df.pivot_table(
        index=EffectColumns.method, columns=EffectColumns.outcome, values=effect_name
    ).reindex(method_names)

    # Determine if annotations should be displayed based on the number of outcomes
    num_outcomes = len(heatmap_data.columns)
    annotate_cells = num_outcomes <= 100

    plt.figure(
        figsize=(num_outcomes * 1.2, len(method_names) * 0.8)
    )  # Adjust figure size dynamically
    ax = sns.heatmap(
        heatmap_data,
        annot=False,  # We will manually control annotations
        fmt=".3f",  # Format for the annotation text
        cmap="plasma",  # Color map (you can choose others like "viridis", "YlGnBu", etc.)
        linewidths=0.5,
        linecolor="lightgray",
        cbar_kws={"label": effect_name.replace("_", " ").title()},
    )

    # Manually add annotations if annotate_cells is True
    if annotate_cells:
        for text in ax.texts:
            text.set_text(
                ""
            )  # Clear default annotations from seaborn's 'annot=True' if it was used

        for i in range(heatmap_data.shape[0]):
            for j in range(heatmap_data.shape[1]):
                value = heatmap_data.iloc[i, j]
                if pd.notnull(value):
                    ax.text(
                        j + 0.5,
                        i + 0.5,
                        f"{value:.3f}",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=8,
                    )

    ax.set_title(
        f"Heatmap of {effect_name.replace('_', ' ').title()} by Method and Outcome"
    )
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Method")

    # --- New logic for saving or showing the plot ---
    if save_path:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(save_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_figure_with_azure_copy(plt.gcf(), save_path, bbox_inches="tight")
    else:
        plt.show()
        plt.close()


def create_effect_size_plot(
    effects_df: pd.DataFrame,
    save_dir: str,
    config: EffectSizePlotConfig,
    title: str = "Effect estimates by outcome and method",
    methods: list[str] = None,
):
    """
    Creates a forest plot by initializing and running the EffectSizePlotter.
    """
    try:
        plotter = EffectSizePlotter(effects_df, save_dir, config, title, methods)
        plotter.run()
    except Exception as e:
        logger.error(f"Failed to create effect size plot. Error: {e}", exc_info=True)


def create_contingency_table_plot(
    data_df: pd.DataFrame,
    save_dir: str,
    config: ContingencyPlotConfig,
    title: str = "Patient Counts by Treatment Status and Outcome",
):
    """
    Creates stacked bar plots from contingency data by running the ContingencyTablePlotter.
    """
    try:
        plotter = ContingencyTablePlotter(data_df, save_dir, config, title)
        plotter.run()
    except Exception as e:
        logger.error(
            f"Failed to create contingency table plot. Error: {e}", exc_info=True
        )


def create_adjustment_plot(
    data_df: pd.DataFrame,
    save_dir: str,
    config: AdjustmentPlotConfig,
    title: str = "TMLE Adjustment Analysis",
):
    """
    Creates adjustment plots by initializing and running the AdjustmentPlotter.
    """
    if not all(
        col in data_df.columns
        for col in [
            TMLEAnalysisColumns.initial_effect,
            TMLEAnalysisColumns.adjustment,
            TMLEAnalysisColumns.adjustment_0,
            TMLEAnalysisColumns.adjustment_1,
            TMLEAnalysisColumns.initial_effect_0,
            TMLEAnalysisColumns.initial_effect_1,
        ]
    ):
        logger.warning(
            "Skipping adjustment plots: Required columns not found in dataframe."
        )
        return

    try:
        plotter = AdjustmentPlotter(data_df, save_dir, config, title)
        plotter.run()
    except Exception as e:
        logger.error(f"Failed to create adjustment plot. Error: {e}", exc_info=True)


def create_ipw_plot(exposure: pd.Series, propensity: pd.Series, save_dir: str):
    """
    Creates IPW weight distribution plots for ATE and ATT.
    Args:
        exposure: pd.Series of exposure values
        propensity: pd.Series of propensity scores
        save_dir: str path to save the figure in

    """
    if len(exposure) != len(propensity):
        raise ValueError("Exposure and propensity must have the same length")

    combos = [
        ("ATE", True),
        ("ATE", False),
        ("ATT", True),
        ("ATT", False),
    ]

    A = exposure.values
    exposure_labels = pd.Series(A).map({0: "Control", 1: "Exposed"}).values

    # Precompute weights and mask for ATT (omit treated)
    weights_data = []
    for wt, stab in combos:
        w = compute_ipw_weights(
            A=exposure,
            ps=propensity,
            weight_type=wt,
            stabilized=stab,
        )
        if wt == "ATT":
            mask = A == 0  # omit treated (A==1) for ATT plots
        else:
            mask = slice(None)
        weights_data.append((wt, stab, w, mask))

    # Determine global x-limit (robust against heavy tails)
    included_weights = []
    for wt, stab, w, mask in weights_data:
        ww = w if isinstance(mask, slice) else w[mask]
        included_weights.append(pd.Series(ww))
    p99 = pd.concat(included_weights).quantile(0.99)
    xmax = float(p99) if pd.notnull(p99) and p99 > 0 else None

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=120)
    axes: List[Axes] = axes.flatten()
    palette = {"Control": "#1f77b4", "Exposed": "#d62728"}

    for ax, (wt, stab, w, mask) in zip(axes, weights_data):
        if isinstance(mask, slice):
            w_plot = pd.Series(w, name="weight")
            labels = pd.Series(exposure_labels, name="Exposure")
        else:
            w_plot = pd.Series(w[mask], name="weight")
            labels = pd.Series(exposure_labels[mask], name="Exposure")

        df_plot = pd.concat([w_plot, labels], axis=1)

        sns.histplot(
            data=df_plot,
            x="weight",
            hue="Exposure",
            bins=50,
            element="step",
            stat="density",
            common_norm=False,
            palette=palette,
            alpha=0.5,
            ax=ax,
        )
        ax.set_title(f"{wt} — {'Stabilized' if stab else 'Unstabilized'}")
        ax.set_xlabel("IPW weight")
        ax.set_ylabel("Density")
        if xmax:
            ax.set_xlim(0, xmax)
        ax.grid(True, linestyle="--", alpha=0.3)

        # For ATT, legend will only show Control; keep consistent location
        ax.legend(title="Exposure", loc="upper right", frameon=False)

    plt.suptitle("IPW Weight Distributions by Exposure", y=0.98)
    save_path = join(save_dir, "ipw_weights.png")
    save_figure_with_azure_copy(fig, save_path, bbox_inches="tight")
    plt.close(fig)
