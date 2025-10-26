import logging
import os
from os.path import join
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from CausalEstimate.estimators.functional.ipw import compute_ipw_weights
from matplotlib.axes import Axes

from corebehrt.constants.causal.data import (
    EffectColumns,
    TMLEAnalysisColumns,
)
from corebehrt.functional.utils.azure_save import save_figure_with_azure_copy
from corebehrt.functional.visualize.calibrate import plot_probas_hist
from corebehrt.modules.plot.estimate import (
    AdjustmentPlotConfig,
    AdjustmentPlotter,
    ContingencyPlotConfig,
    ContingencyTablePlotter,
    EffectSizePlotConfig,
    EffectSizePlotter,
)

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


def create_ipw_plot(
    exposure: pd.Series,
    propensity: pd.Series,
    save_dir: str,
    clip_percentile: float = 1.0,
):
    """
    Creates IPW weight distribution plots for ATE and ATT, with a correct
    legend on each subplot showing "Exposed" and "Unexposed" labels.

    Args:
        exposure: pd.Series of exposure values (0 or 1).
        propensity: pd.Series of propensity scores.
        save_dir: str path to save the figure in.
        clip_percentile: float percentile for clipping weights (e.g., 0.99 for 99th percentile).
                         1.0 means no clipping.
    """
    if len(exposure) != len(propensity):
        raise ValueError("Exposure and propensity must have the same length")

    # --- 1. Data Preparation ---
    A = exposure.values
    exposure_labels = exposure.map({0: "Unexposed", 1: "Exposed"}).values
    palette = {"Unexposed": "#1f77b4", "Exposed": "#d62728"}

    weights = {
        ("ATE", "clipped"): compute_ipw_weights(
            exposure, propensity, "ATE", clip_percentile
        ),
        ("ATE", "unclipped"): compute_ipw_weights(exposure, propensity, "ATE", 1.0),
        ("ATT", "clipped"): compute_ipw_weights(
            exposure, propensity, "ATT", clip_percentile
        ),
        ("ATT", "unclipped"): compute_ipw_weights(exposure, propensity, "ATT", 1.0),
    }

    all_weights = np.concatenate(list(weights.values()))
    xmax = np.percentile(all_weights, 99.5)

    # --- 2. Plotting ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=120, sharex=True, sharey=True)

    plot_configs = [
        {
            "ax": axes[0, 0],
            "wt": "ATE",
            "clip_status": "clipped",
            "title": f"Clipped (p{int(clip_percentile * 100)})",
        },
        {
            "ax": axes[0, 1],
            "wt": "ATE",
            "clip_status": "unclipped",
            "title": "Unclipped",
        },
        {"ax": axes[1, 0], "wt": "ATT", "clip_status": "clipped", "title": None},
        {"ax": axes[1, 1], "wt": "ATT", "clip_status": "unclipped", "title": None},
    ]

    for config in plot_configs:
        ax = config["ax"]
        wt = config["wt"]
        clip_status = config["clip_status"]
        w = weights[(wt, clip_status)]

        mask = (A == 0) if wt == "ATT" else slice(None)
        df_plot = pd.DataFrame({"weight": w[mask], "Exposure": exposure_labels[mask]})
        unique_exposures = df_plot["Exposure"].unique()

        # Plot based on whether there are one or two exposure groups
        if len(unique_exposures) > 1:  # ATE plots
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
        else:  # ATT plots (only "Unexposed" group)
            group_label = unique_exposures[0]
            sns.histplot(
                data=df_plot,
                x="weight",
                bins=50,
                element="step",
                stat="density",
                color=palette[group_label],
                alpha=0.5,
                ax=ax,
                label=group_label,
            )

        # --- 3. Create a normal legend on EACH subplot ---
        # This is the key change: create a simple, robust legend for every plot.
        legend = ax.legend(title="Exposure Status", loc="upper right")
        if legend:
            legend.get_frame().set_alpha(0.8)

        # --- 4. Formatting & Labeling ---
        if config["title"]:
            ax.set_title(config["title"], fontsize=14)
        ax.set_xlabel("IPW Weight")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xlim(0, xmax)

    axes[0, 0].set_ylabel("ATE\nDensity", fontsize=12)
    axes[1, 0].set_ylabel("ATT (Unexposed Only)\nDensity", fontsize=12)

    fig.suptitle("IPW Weight Distributions", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = join(save_dir, "ipw_weights_comparison.png")
    save_figure_with_azure_copy(fig, save_path, bbox_inches="tight")
    plt.close(fig)


def create_ps_comparison_plot(df_before, df_after, ps_col, exposure_col, save_dir: str):
    try:
        fig, ax = plt.subplots(2, 1, figsize=(10, 6))
        ax: List[Axes] = ax.flatten()
        plot_probas_hist(
            df_before,
            ps_col,
            exposure_col,
            ("Control", "Exposed"),
            "Before Common Support",
            "Propensity Score",
            ax[0],
            min_quantile=0,
            max_quantile=1,
        )
        ax[0].legend_.remove()
        plot_probas_hist(
            df_after,
            ps_col,
            exposure_col,
            ("Control", "Exposed"),
            "After Common Support",
            "Propensity Score",
            ax[1],
            min_quantile=0,
            max_quantile=1,
        )
        fig.suptitle("Effect of Common Support on Propensity Score")
        save_figure_with_azure_copy(
            fig, join(save_dir, "ps_comparison.png"), bbox_inches="tight"
        )
        plt.close(fig)
    except Exception as e:
        logger.warning(f"Failed to create PS histograms: {e}")
