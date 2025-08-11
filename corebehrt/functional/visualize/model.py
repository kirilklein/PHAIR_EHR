import torch
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Union, Dict, List, Optional
import pandas as pd


def visualize_weight_distributions(
    model_or_state_dict: Union[nn.Module, Dict[str, torch.Tensor]],
    layer_filter: Optional[Union[str, List[str]]] = None,
    bins: int = 100,
    max_subplots_per_fig: int = 15,
    cols: int = 5,
    plot_type: str = "hist_kde",
    title: str = "Weight Distributions",
    save_dir: Optional[str] = None,
) -> None:
    """
    Analyzes and visualizes the weight distributions of a PyTorch model.

    This function generates histograms and/or kernel density estimates for the
    weights of specified layers, and prints summary statistics (mean, std dev,
    and sparsity). If the number of layers to plot exceeds `max_subplots_per_fig`,
    multiple figures will be generated.

    Args:
        model_or_state_dict (Union[nn.Module, Dict[str, torch.Tensor]]):
            A PyTorch model (nn.Module) or its state_dict.
        layer_filter (Optional[Union[str, List[str]]], optional):
            A string or list of strings to filter layer names. Only layers
            whose names contain any of these substrings will be plotted.
            If None, all layers with weights are plotted. Defaults to None.
        bins (int, optional):
            Number of bins for the histograms. Defaults to 100.
        max_subplots_per_fig (int, optional):
            The maximum number of subplots to show in a single figure.
            Defaults to 9.
        cols (int, optional):
            Number of columns in the plot grid. Defaults to 3.
        plot_type (str, optional):
            Type of plot to generate. Options are:
            - 'hist_kde': Histogram with a KDE overlay (default).
            - 'hist': Histogram only.
            - 'kde': KDE only.
        title (str, optional):
            The main title for the entire figure. Defaults to 'Weight Distributions'.
        print_stats (bool, optional):
            If True, prints a table with summary statistics for each weight tensor.
            Defaults to True.
        save_dir (Optional[str], optional):
            If provided, the directory where plots will be saved. The directory
            will be created if it doesn't exist. Plots are saved as
            'weight_distributions_{i}.png'. If None, plots are shown directly.
            Defaults to None.
    """
    if save_dir:
        save_dir = os.path.join(save_dir, "weight_distributions")
        os.makedirs(save_dir, exist_ok=True)

    # --- 1. Get State Dictionary and Extract Weights ---
    if isinstance(model_or_state_dict, dict):
        state_dict = model_or_state_dict
    elif hasattr(model_or_state_dict, "state_dict"):
        state_dict = model_or_state_dict.state_dict()
    else:
        raise TypeError("Input must be a PyTorch model (nn.Module) or a state_dict.")

    weights_data = {}
    stats = []

    for name, param in state_dict.items():
        if param.is_floating_point():
            param_data = param.data.cpu().numpy()
            weights_data[name] = param_data.flatten()

            # Calculate stats
            mean = np.mean(param_data)
            std = np.std(param_data)
            close_to_zero_pct = (
                np.isclose(param_data, 0, atol=1e-5).sum() / param_data.size
            ) * 100
            stats.append(
                {
                    "name": name,
                    "shape": list(param.shape),
                    "mean": mean,
                    "std": std,
                    "zeros_pct": close_to_zero_pct,
                }
            )

    if not weights_data:
        print("No weight tensors found in the model/state_dict.")
        return

    plot_weight_statistics(
        stats,
        base_title="Model Weight Statistics",
        output_path=os.path.join(save_dir, "weight_statistics.png"),
    )

    # --- 3. Filter Layers for Plotting ---
    if layer_filter:
        if isinstance(layer_filter, str):
            layer_filter = [layer_filter]

        filtered_weights = {
            name: data
            for name, data in weights_data.items()
            if any(f in name for f in layer_filter)
        }
        if not filtered_weights:
            print(
                f"\nWarning: No layers found matching the filter: {layer_filter}. Nothing to plot."
            )
            return
    else:
        filtered_weights = weights_data

    # --- 4. Visualize Weight Distributions ---
    layers_to_plot = list(filtered_weights.items())
    num_layers = len(layers_to_plot)
    if num_layers == 0:
        return

    print(
        f"\nGenerating {num_layers} weight distribution plot(s) across multiple figures..."
    )

    num_figures = int(np.ceil(num_layers / max_subplots_per_fig))

    for fig_num in range(num_figures):
        start_idx = fig_num * max_subplots_per_fig
        end_idx = start_idx + max_subplots_per_fig
        chunk_items = layers_to_plot[start_idx:end_idx]
        num_plots_on_fig = len(chunk_items)

        rows = int(np.ceil(num_plots_on_fig / cols))

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, axes = plt.subplots(
            rows, cols, figsize=(cols * 4.5, rows * 3.5), constrained_layout=True
        )
        axes = np.array(axes).flatten()

        for i, (name, data) in enumerate(chunk_items):
            ax = axes[i]

            if plot_type == "hist_kde":
                sns.histplot(
                    data,
                    bins=bins,
                    ax=ax,
                    kde=True,
                    color="#4c72b0",
                    alpha=0.6,
                    edgecolor="white",
                )
            elif plot_type == "hist":
                sns.histplot(
                    data,
                    bins=bins,
                    ax=ax,
                    kde=False,
                    color="#55a868",
                    alpha=0.7,
                    edgecolor="white",
                )
            elif plot_type == "kde":
                sns.kdeplot(data, ax=ax, color="#c44e52", fill=True, alpha=0.5)
            else:
                raise ValueError(
                    "plot_type must be one of 'hist_kde', 'hist', or 'kde'"
                )

            ax.axvline(x=0, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
            ax.set_title(name, fontsize=18, wrap=True, y=1.02)
            ax.set_ylabel("")
            ax.set_yticks([])
            ax.tick_params(axis="x", labelsize=11)
            ax.ticklabel_format(style="sci", axis="x", scilimits=(-2, 2))

        for i in range(num_plots_on_fig, len(axes)):
            axes[i].set_visible(False)

        fig_title = (
            f"{title} (Part {fig_num + 1}/{num_figures})" if num_figures > 1 else title
        )
        fig.suptitle(fig_title, fontsize=18, weight="bold")

        if save_dir:
            save_path = os.path.join(
                save_dir, f"weight_distributions_part_{fig_num + 1}.png"
            )
            plt.savefig(save_path)
            print(f"Saved figure to {save_path}")
            plt.close(fig)
        else:
            plt.show()


def plot_weight_statistics(
    stats: List[Dict],
    base_title: str = "Model Weight Statistics",
    output_path: Optional[str] = None,
) -> None:
    """
    Creates high-quality, grouped bar charts of model weight statistics.
    Layers are grouped into 'Embeddings', 'Poolers', 'Heads', and 'Body'.

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

    # --- 1. Group the stats based on layer names ---
    grouped_stats = {
        "Embeddings": [],
        "Poolers": [],
        "Heads": [],
        "Body": [],
        "Loss": [],
    }
    for s in stats:
        name = s["name"]
        if name.startswith("embeddings."):
            grouped_stats["Embeddings"].append(s)
        elif "pooler" in name:
            grouped_stats["Poolers"].append(s)
        elif "heads" in name:
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
            # Create a unique filename for each group plot
            path_parts = os.path.splitext(output_path)
            group_output_path = f"{path_parts[0]}_{group_name.lower()}{path_parts[1]}"

            output_dir = os.path.dirname(group_output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            plt.savefig(group_output_path, dpi=200, bbox_inches="tight")
            print(f"Saved weight statistics plot to {group_output_path}")
            plt.close(fig)  # Close the figure to free memory
        else:
            plt.show()


def _create_single_stats_plot(df: pd.DataFrame, title: str):
    """
    Creates a single figure with 3 subplots for a given DataFrame of stats.
    This is a helper function to be called by the main plotter.
    """
    # Adjust figure height based on the number of layers for better readability
    fig_height = max(4, len(df) * 0.5)

    # Use constrained_layout=True for robust automatic layout management
    fig, axes = plt.subplots(
        1, 3, figsize=(18, fig_height), sharey=True, constrained_layout=True
    )
    plt.style.use("seaborn-v0_8-whitegrid")

    # --- Plot 1: Mean (μ) ---
    ax1 = axes[0]
    colors_mean = plt.cm.RdBu_r(np.linspace(0.1, 0.9, len(df)))
    df["mean"].plot(kind="barh", ax=ax1, color=colors_mean, width=0.5)
    ax1.set_title("Mean Weight Value (μ)", fontsize=14, weight="bold")
    ax1.set_xlabel("Value")
    ax1.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax1.grid(axis="x", linestyle="--", alpha=0.6)

    # --- Plot 2: Standard Deviation (σ) ---
    ax2 = axes[1]
    colors_std = plt.cm.viridis_r(np.linspace(0.1, 0.8, len(df)))
    df["std"].plot(kind="barh", ax=ax2, color=colors_std, width=0.5)
    ax2.set_title("Standard Deviation (σ)", fontsize=14, weight="bold")
    ax2.set_xlabel("Value")
    ax2.grid(axis="x", linestyle="--", alpha=0.6)

    # --- Plot 3: Sparsity (% Zeros) ---
    ax3 = axes[2]
    colors_zeros = plt.cm.Reds(np.linspace(0.2, 0.8, len(df)))
    df["zeros_pct"].plot(kind="barh", ax=ax3, color=colors_zeros, width=0.5)
    ax3.set_title("Sparsity (% Zeros)", fontsize=14, weight="bold")
    ax3.set_xlabel("Percentage (%)")
    ax3.set_xlim(0, 101)  # Set limit slightly past 100 for labels
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
                fontsize=9,
                color="white" if ha == "right" else "black",
            )

    fig.suptitle(title, fontsize=20, weight="bold")
    return fig
