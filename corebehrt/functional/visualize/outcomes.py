import os
from os.path import join
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.patches import Rectangle

from corebehrt.functional.utils.azure_save import save_figure_with_azure_copy


def _generate_paginated_plots(
    items_to_plot: List[str],
    plot_callback: Callable,
    output_dir: str,
    base_filename: str,
    plot_title: str,
    max_items_per_plot: int,
    max_number_of_plots: int,
    **kwargs,
) -> None:
    """
    Orchestrates the creation of one or more plots from a list of items.

    This function handles data truncation, pagination, figure creation,
    saving, and title/filename generation.

    Args:
        items_to_plot (List[str]): A list of unique item names to be plotted.
        plot_callback (Callable): A function that performs the actual plotting on an Axes object.
                                 It will be called with (ax, item_chunk, title, **kwargs).
        output_dir (str): Directory where the plot(s) will be saved.
        base_filename (str): The base name for the output file(s).
        plot_title (str): The base title for the plot(s).
        max_items_per_plot (int): Maximum number of items per plot.
        max_number_of_plots (int): Maximum number of plots to generate.
        **kwargs: Additional keyword arguments to pass to the plot_callback.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not items_to_plot:
        print("No items to plot.")
        return

    if max_items_per_plot <= 0 or max_number_of_plots <= 0:
        print("max_items_per_plot and max_number_of_plots must be positive.")
        return

    # Truncate the list of items if it exceeds total capacity
    total_capacity = max_items_per_plot * max_number_of_plots
    if len(items_to_plot) > total_capacity:
        print(
            f"Number of items ({len(items_to_plot)}) exceeds total capacity ({total_capacity}). "
            f"Truncating to the first {total_capacity} items."
        )
        items_to_plot = items_to_plot[:total_capacity]

    num_items = len(items_to_plot)
    num_plots = (num_items + max_items_per_plot - 1) // max_items_per_plot

    if num_plots > 1:
        print(
            f"Generating {num_plots} plot(s) with up to {max_items_per_plot} items each."
        )

    for i in range(num_plots):
        start_index = i * max_items_per_plot
        end_index = start_index + max_items_per_plot
        item_chunk = items_to_plot[start_index:end_index]

        num_items_in_plot = len(item_chunk)
        fig_height = max(6, num_items_in_plot * 0.4)
        fig, ax = plt.subplots(figsize=(12, fig_height))

        # Prepare title and save path
        current_title = plot_title
        if num_plots > 1:
            current_title += f" (Part {i + 1}/{num_plots})"
            save_path = join(output_dir, f"{base_filename}_{i + 1}.png")
        else:
            save_path = join(output_dir, f"{base_filename}.png")

        # Execute the specific plotting logic
        plot_callback(ax=ax, item_chunk=item_chunk, title=current_title, **kwargs)

        fig.tight_layout()
        save_figure_with_azure_copy(fig, save_path)
        print(f"Plot saved to '{save_path}'")


def _plot_distribution_chunk(
    ax: Axes, item_chunk: List[str], title: str, proportions: pd.Series
) -> None:
    """Callback to plot a single chunk of outcome distributions."""
    subset_proportions = proportions.loc[item_chunk]

    bars: BarContainer = ax.barh(
        subset_proportions.index, subset_proportions.values, color="skyblue"
    )
    ax.set_xlabel("Proportion of Positive Outcomes")
    ax.set_ylabel("Outcomes")
    ax.set_title(title, fontsize=16)
    ax.set_xlim(0, max(1.0, subset_proportions.max() * 1.1))

    for bar in bars:
        bar: Rectangle
        width = bar.get_width()
        ax.text(
            width + 0.01,
            bar.get_y() + bar.get_height() / 2.0,
            f"{width:.2f}",
            va="center",
        )


def _plot_filtering_chunk(
    ax: Axes, item_chunk: List[str], title: str, df_plot: pd.DataFrame
) -> None:
    """Callback to plot a single chunk of filtering statistics."""
    subset_df = df_plot[df_plot["name"].isin(item_chunk)]
    sns.set_style("whitegrid")

    sns.barplot(x="count", y="name", hue="status", data=subset_df, orient="h", ax=ax)

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Number of Patients", fontsize=12)
    ax.set_ylabel("Exposure / Outcome", fontsize=12)
    ax.legend(title="Filtering Status")

    for p in ax.patches:
        p: Rectangle
        width = p.get_width()
        if width > 0:
            ax.annotate(
                f"{width:.0f}",
                (width, p.get_y() + p.get_height() / 2.0),
                ha="left",
                va="center",
                xytext=(5, 0),
                textcoords="offset points",
            )

    max_count = subset_df["count"].max()
    safe_right = 1 if pd.isna(max_count) or max_count <= 0 else max_count * 1.15
    ax.set_xlim(right=safe_right)


# --- Public API Functions ---
def plot_target_distribution(
    df: pd.DataFrame,
    outcome_dir: str,
    max_outcomes_per_plot: int = 15,
    max_number_of_plots: int = 10,
) -> None:
    """
    Plots the distribution of positive outcomes, splitting into multiple files if needed.

    Args:
        df (pd.DataFrame): DataFrame with outcome columns (0s and 1s).
        outcome_dir (str): Directory where the plot(s) will be saved.
        max_outcomes_per_plot (int): Maximum number of outcomes per plot.
        max_number_of_plots (int): Maximum number of plots to generate.
    """
    outcome_dir = join(outcome_dir, "target_distribution")
    os.makedirs(outcome_dir, exist_ok=True)
    # 1. Data Preparation
    if df.empty:
        print("Input DataFrame is empty. No outcomes to plot.")
        return
    outcome_proportions = df.mean().sort_values(ascending=True)
    outcomes_to_plot = outcome_proportions.index.tolist()

    # 2. Orchestration
    _generate_paginated_plots(
        items_to_plot=outcomes_to_plot,
        plot_callback=_plot_distribution_chunk,
        output_dir=outcome_dir,
        base_filename="target_distribution",
        plot_title="Distribution of Positive Outcomes",
        max_items_per_plot=max_outcomes_per_plot,
        max_number_of_plots=max_number_of_plots,
        # kwargs passed to the callback
        proportions=outcome_proportions,
    )


def plot_filtering_stats(
    stats: Dict[str, Dict],
    output_dir: str,
    max_items_per_plot: int = 15,
    max_number_of_plots: int = 10,
) -> None:
    """
    Plots patient counts before and after filtering, splitting into multiple files if needed.

    Args:
        stats (dict): A dictionary containing the before/after counts.
        output_dir (str): The directory to save the plot(s) in.
        max_items_per_plot (int): Maximum number of items per plot.
        max_number_of_plots (int): Maximum number of plots to generate.
    """
    # 1. Data Preparation
    if not stats:
        print("Statistics dictionary is empty, skipping plot generation.")
        return
    output_dir = join(output_dir, "filtering_counts")
    os.makedirs(output_dir, exist_ok=True)
    plot_data = []
    for name, values in stats.items():
        plot_data.append(
            {
                "name": name,
                "status": "Before Filtering",
                "count": values.get("before", 0),
            }
        )
        after_counts = values.get("after", {})
        positive_events_after = after_counts.get(
            1, 0
        )  # Assuming '1' is the key for positive cases
        plot_data.append(
            {
                "name": name,
                "status": "Within Follow-up Window",
                "count": positive_events_after,
            }
        )
    df_plot = pd.DataFrame(plot_data)

    if df_plot.empty:
        print("No data to plot after processing stats.")
        return

    sorted_names = (
        df_plot[df_plot["status"] == "Before Filtering"]
        .sort_values("count", ascending=False)["name"]
        .tolist()
    )
    df_plot["name"] = pd.Categorical(
        df_plot["name"], categories=sorted_names, ordered=True
    )

    # 2. Orchestration
    _generate_paginated_plots(
        items_to_plot=sorted_names,
        plot_callback=_plot_filtering_chunk,
        output_dir=output_dir,
        base_filename="filtering_counts",
        plot_title="Patient Counts Before and After Filtering",
        max_items_per_plot=max_items_per_plot,
        max_number_of_plots=max_number_of_plots,
        # kwargs passed to the callback
        df_plot=df_plot,
    )
