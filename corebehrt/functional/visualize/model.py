import torch
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Union, Dict, List, Optional
import pandas as pd


# --- Helper for plotting distributions for a single group ---
def _plot_distributions_for_group(
    weights_in_group: Dict[str, np.ndarray],
    group_name: str,
    base_title: str,
    save_dir: Optional[str],
    bins: int,
    max_subplots_per_fig: int,
    cols: int,
):
    """Helper to generate and save/show distribution plots for a group of layers."""
    if save_dir:
        save_dir = os.path.join(save_dir, "weight_distributions")
        os.makedirs(save_dir, exist_ok=True)

    layers_to_plot = list(weights_in_group.items())
    num_layers = len(layers_to_plot)
    if num_layers == 0:
        return

    print(f"\n--- Generating plots for '{group_name}' group... ---")
    num_figures = int(np.ceil(num_layers / max_subplots_per_fig))

    for fig_num in range(num_figures):
        start_idx = fig_num * max_subplots_per_fig
        end_idx = start_idx + max_subplots_per_fig
        chunk_items = layers_to_plot[start_idx:end_idx]
        num_plots_on_fig = len(chunk_items)

        rows = int(np.ceil(num_plots_on_fig / cols))

        # --- CHANGE: Make subplots shorter for a more compact view ---
        fig, axes = plt.subplots(
            rows, cols, figsize=(cols * 4, rows * 1.5), constrained_layout=True
        )
        axes = np.array(axes).flatten()

        for i, (name, data) in enumerate(chunk_items):
            ax = axes[i]
            sns.histplot(
                data,
                bins=bins,
                ax=ax,
                kde=True,
                color="#4c72b0",
                alpha=0.6,
                edgecolor=None,
            )
            ax.axvline(x=0, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
            # --- CHANGE: Smaller font size for titles to fit compact layout ---
            ax.set_title(name, fontsize=10, wrap=True)
            ax.set_ylabel("")
            ax.set_xlabel("")
            ax.set_yticks([])
            ax.tick_params(axis="x", labelsize=9)
            ax.ticklabel_format(style="sci", axis="x", scilimits=(-2, 2))

        for i in range(num_plots_on_fig, len(axes)):
            axes[i].set_visible(False)

        fig_title = f"{base_title} - {group_name}"
        if num_figures > 1:
            fig_title += f" (Part {fig_num + 1}/{num_figures})"
        fig.suptitle(fig_title, fontsize=16, weight="bold")

        if save_dir:
            filename = f"{group_name.lower()}_part_{fig_num + 1}.png"
            save_path = os.path.join(save_dir, filename)
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Saved figure to {save_path}")
            plt.close(fig)
        else:
            plt.show()


# --- Main visualization function, now an orchestrator ---
def visualize_weight_distributions(
    model_or_state_dict: Union[nn.Module, Dict[str, torch.Tensor]],
    bins: int = 75,
    max_subplots_per_fig: int = 16,  # Increased default
    cols: int = 4,
    base_title: str = "Weight Distributions",
    save_dir: Optional[str] = None,
):
    """
    Analyzes and visualizes the weight distributions and statistics of a PyTorch model,
    grouping layers by component.

    This function generates two sets of plots:
    1. Histograms of weight distributions for each layer.
    2. Bar charts of weight statistics (mean, std, sparsity) for each layer group.
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    if isinstance(model_or_state_dict, dict):
        state_dict = model_or_state_dict
    elif hasattr(model_or_state_dict, "state_dict"):
        state_dict = model_or_state_dict.state_dict()
    else:
        raise TypeError("Input must be a PyTorch model (nn.Module) or a state_dict.")

    all_weights = {
        name: param.data.cpu().numpy().flatten()
        for name, param in state_dict.items()
        if param.is_floating_point()
    }

    if not all_weights:
        print("No weight tensors found in the model/state_dict.")
        return

    # --- Group weights by component ---
    grouped_weights = {
        "Embeddings": {},
        "Poolers": {},
        "Heads": {},
        "Loss": {},
        "Body": {},
    }
    for name, data in all_weights.items():
        if name.startswith("embeddings."):
            grouped_weights["Embeddings"][name] = data
        elif "pooler" in name:
            grouped_weights["Poolers"][name] = data
        elif "head" in name:
            grouped_weights["Heads"][name] = data
        elif "loss" in name:
            grouped_weights["Loss"][name] = data
        else:
            grouped_weights["Body"][name] = data

    # --- Plot distributions for each group ---
    for group_name, weights_in_group in grouped_weights.items():
        if weights_in_group:
            _plot_distributions_for_group(
                weights_in_group,
                group_name,
                base_title,
                save_dir,
                bins,
                max_subplots_per_fig,
                cols,
            )
        else:
            print(f"Skipping '{group_name}' distribution plot as it has no layers.")

    # --- Calculate and plot weight statistics ---
    print("\n--- Calculating and plotting weight statistics... ---")
    stats = []
    for name, data in all_weights.items():
        if data.size == 0:
            continue
        stats.append(
            {
                "name": name,
                "mean": np.mean(data),
                "std": np.std(data),
                "zeros_pct": 100 * (np.sum(data == 0) / data.size),
            }
        )

    stats_output_path = None
    if save_dir:
        # --- CHANGE: Save stats plots in the same subdirectory as distributions ---
        stats_dir = os.path.join(save_dir, "weight_distributions")
        stats_output_path = os.path.join(stats_dir, "weight_statistics.png")

    _plot_weight_statistics(
        stats=stats,
        base_title="Model Weight Statistics",
        output_path=stats_output_path,
    )


def _create_single_stats_plot(df: pd.DataFrame, title: str):
    """
    Creates a single figure with 3 subplots for a given DataFrame of stats.
    This is a helper function to be called by the main plotter.
    """
    # --- CHANGE 1: Tighter vertical spacing ---
    # Reducing the multiplier from 0.5 to 0.4 makes the plot more compact.
    fig_height = max(4, len(df) * 0.4)

    fig, axes = plt.subplots(
        1, 3, figsize=(18, fig_height), sharey=True, constrained_layout=True
    )
    plt.style.use("seaborn-v0_8-whitegrid")

    # --- Use a thicker bar width to reduce whitespace between bars ---
    bar_width = 0.85

    # --- Plot 1: Mean (μ) ---
    ax1 = axes[0]
    colors_mean = plt.cm.RdBu_r(np.linspace(0.1, 0.9, len(df)))
    df["mean"].plot(kind="barh", ax=ax1, color=colors_mean, width=bar_width)
    ax1.set_title("Mean Weight Value (μ)", fontsize=14, weight="bold")
    ax1.set_xlabel("Value")
    ax1.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax1.grid(axis="x", linestyle="--", alpha=0.6)
    ax1.tick_params(axis="y", labelsize=12)

    # --- Plot 2: Standard Deviation (σ) ---
    ax2 = axes[1]
    colors_std = plt.cm.viridis_r(np.linspace(0.1, 0.8, len(df)))
    df["std"].plot(kind="barh", ax=ax2, color=colors_std, width=bar_width)
    ax2.set_title("Standard Deviation (σ)", fontsize=14, weight="bold")
    ax2.set_xlabel("Value")
    ax2.grid(axis="x", linestyle="--", alpha=0.6)

    # --- Plot 3: Sparsity (% Zeros) ---
    ax3 = axes[2]
    colors_zeros = plt.cm.Reds(np.linspace(0.2, 0.8, len(df)))
    df["zeros_pct"].plot(kind="barh", ax=ax3, color=colors_zeros, width=bar_width)
    ax3.set_title("Sparsity (% Zeros)", fontsize=14, weight="bold")
    ax3.set_xlabel("Percentage (%)")
    ax3.set_xlim(0, 101)
    ax3.grid(axis="x", linestyle="--", alpha=0.6)

    # Add value labels to the bars for clarity
    for ax in axes:
        for p in ax.patches:
            value = p.get_width()
            text_x = p.get_width()
            ha = "left"
            padding = ax.get_xlim()[1] * 0.01

            if abs(value) > ax.get_xlim()[1] * 0.4:
                text_x -= padding
                ha = "right"
            else:
                text_x += padding

            text = f"{value:.1f}%" if ax == ax3 else f"{value:.2e}"
            ax.text(
                text_x,
                p.get_y() + p.get_height() / 2.0,
                text,
                va="center",
                ha=ha,
                # --- CHANGE 2: Larger annotation font size ---
                fontsize=10,
                color="white" if ha == "right" else "black",
                weight="bold" if ha == "right" else "normal",
            )

    fig.suptitle(title, fontsize=20, weight="bold")
    return fig


# --- Main orchestrator function (with your new "Loss" group) ---
def _plot_weight_statistics(
    stats: List[Dict],
    base_title: str = "Model Weight Statistics",
    output_path: Optional[str] = None,
) -> None:
    """
    Creates high-quality, grouped bar charts of model weight statistics.
    Layers are grouped into 'Embeddings', 'Poolers', 'Heads', 'Loss', and 'Body'.
    This is intended to be a helper function.

    Args:
        stats (List[Dict]): A list of dictionaries with stats for all layers.
        base_title (str): The base title for all figures.
        output_path (Optional[str]): If provided, saves the figures to files
            with a group-specific suffix (e.g., 'path_embeddings.png').
            If None, displays the plots interactively.
    """
    if not stats:
        print("No statistics provided to plot.")
        return

    # --- 1. Group the stats based on layer names (with Loss group) ---
    grouped_stats = {
        "Embeddings": [],
        "Poolers": [],
        "Heads": [],
        "Loss": [],
        "Body": [],
    }
    for s in stats:
        name = s["name"]
        if name.startswith("embeddings."):
            grouped_stats["Embeddings"].append(s)
        elif "pooler" in name:
            grouped_stats["Poolers"].append(s)
        elif "head" in name:
            grouped_stats["Heads"].append(s)
        elif "loss" in name:
            grouped_stats["Loss"].append(s)
        else:
            grouped_stats["Body"].append(s)

    # --- 2. Iterate through groups and plot each one ---
    for group_name, group_data in grouped_stats.items():
        if not group_data:
            print(f"Skipping '{group_name}' group as it has no layers.")
            continue

        print(f"\n--- Generating plot for '{group_name}' group... ---")

        df = pd.DataFrame(group_data).set_index("name")
        df = df.iloc[::-1]  # Plot first layer at the top

        fig_title = f"{base_title} - {group_name}"
        fig = _create_single_stats_plot(df, title=fig_title)

        # --- 3. Save or show the figure for the current group ---
        if output_path:
            path_parts = os.path.splitext(output_path)
            group_output_path = f"{path_parts[0]}_{group_name.lower()}{path_parts[1]}"
            output_dir = os.path.dirname(group_output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            plt.savefig(group_output_path, dpi=200, bbox_inches="tight")
            print(f"Saved weight statistics plot to {group_output_path}")
            plt.close(fig)
        else:
            plt.show()
