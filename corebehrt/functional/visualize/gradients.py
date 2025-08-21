import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch.nn as nn
from torch.optim import Optimizer
from torch.cuda.amp import GradScaler
from typing import Callable, Optional, Dict
from corebehrt.azure.util import save_figure_with_azure_copy
from matplotlib.axes import Axes
from typing import List


def _plot_distributions_for_group(
    gradients_in_group: Dict[str, np.ndarray],
    group_name: str,
    base_title: str,
    save_dir: Optional[str],
    bins: int,
    max_subplots_per_fig: int,
    cols: int,
):
    """
    Helper to generate and save/show distribution plots for a group of gradients.
    This function is styled to match the weight distribution plotter.
    """
    if save_dir:
        # Create a specific subdirectory for gradient plots
        save_dir = os.path.join(save_dir, "gradient_distributions")
        os.makedirs(save_dir, exist_ok=True)

    layers_to_plot = list(gradients_in_group.items())
    num_layers = len(layers_to_plot)
    if num_layers == 0:
        return

    print(f"\n--- Generating gradient plots for '{group_name}' group... ---")
    num_figures = int(np.ceil(num_layers / max_subplots_per_fig))

    for fig_num in range(num_figures):
        start_idx = fig_num * max_subplots_per_fig
        end_idx = start_idx + max_subplots_per_fig
        chunk_items = layers_to_plot[start_idx:end_idx]
        num_plots_on_fig = len(chunk_items)

        rows = int(np.ceil(num_plots_on_fig / cols))

        # Use the same compact subplot sizing as the weight plotter
        fig, axes = plt.subplots(
            rows, cols, figsize=(cols * 4, rows * 1.5), constrained_layout=True
        )
        axes: List[Axes] = np.array(axes).flatten()

        for i, (name, data) in enumerate(chunk_items):
            ax = axes[i]
            # Use the same styling for the histogram plot
            sns.histplot(
                data,
                bins=bins,
                ax=ax,
                kde=True,
                color="#4c72b0",  # Matched color
                alpha=0.6,
                edgecolor=None,
            )
            # Add the vertical line at zero, just like the weight plots
            ax.axvline(x=0, color="red", linestyle="--", linewidth=1.2, alpha=0.7)

            # Use the same compact title and label styling
            ax.set_title(name, fontsize=10, wrap=True)
            ax.set_ylabel("")
            ax.set_xlabel("")
            ax.set_yticks([])
            ax.tick_params(axis="x", labelsize=9)
            ax.ticklabel_format(style="sci", axis="x", scilimits=(-2, 2))

        # Hide any unused subplots
        for i in range(num_plots_on_fig, len(axes)):
            axes[i].set_visible(False)

        # Create a title for the entire figure
        fig_title = f"{base_title} - {group_name}"
        if num_figures > 1:
            fig_title += f" (Part {fig_num + 1}/{num_figures})"
        fig.suptitle(fig_title, fontsize=16, weight="bold")

        if save_dir:
            # Matched file naming convention
            filename = f"{group_name.lower()}_part_{fig_num + 1}.png"
            save_path = os.path.join(save_dir, filename)
            save_figure_with_azure_copy(
                fig, save_path, dpi=200, bbox_inches="tight", close=False
            )
            print(f"Saved figure to {save_path}")
        else:
            plt.show()
    plt.close()


def plot_gradient_distributions(
    model: nn.Module,
    optimizer: Optimizer,
    scaler: Optional[GradScaler] = None,
    log: Callable[[str], None] = print,
    run_folder: str = ".",
    global_step: int = 0,
    bins: int = 75,
    max_subplots_per_fig: int = 16,
    cols: int = 4,
    allow_destructive_unscaling: bool = False,
):
    """
    Inspects and plots the distributions of gradients, grouped by model component,
    with styling and structure that mirrors the weight visualization functions.

    This function MUST be called after `loss.backward()` and before `optimizer.step()`.

    Args:
        model (nn.Module): The model whose gradients are to be plotted.
        optimizer (Optimizer): The optimizer, only needed if allow_destructive_unscaling=True.
        scaler (Optional[GradScaler]): The GradScaler for mixed precision.
        log (Callable[[str], None]): A logging function (e.g., `print`).
        run_folder (str): Base folder to save figures.
        global_step (int): The current training step, included in the plot title.
        bins (int): Number of bins for the histograms.
        max_subplots_per_fig (int): Maximum number of subplots on a single figure.
        cols (int): Number of columns in the subplot grid.
        allow_destructive_unscaling (bool): If True, allows calling scaler.unscale_(optimizer)
            which mutates optimizer state. If False (default), uses non-destructive approach.
    """
    # --- 1. Handle gradient scaling non-destructively ---
    save_dir = os.path.join(run_folder, "figs")  # Base directory for figures
    os.makedirs(save_dir, exist_ok=True)

    unscaled_str = ""
    scale_factor = 1.0

    if scaler and scaler.is_enabled():
        if allow_destructive_unscaling:
            # Original destructive approach (only if explicitly opted in)
            try:
                scaler.unscale_(optimizer)
                unscaled_str = " (Unscaled)"
            except Exception as e:
                log(f"Warning: Could not unscale gradients for plotting: {e}")
        else:
            # Non-destructive approach: get scale and divide gradient copies
            try:
                scale_factor = scaler.get_scale()
                unscaled_str = " (Unscaled)" if scale_factor != 1.0 else ""
            except Exception as e:
                log(f"Warning: Could not get scale factor for gradient plotting: {e}")
                scale_factor = 1.0

    # --- 2. Extract and group all available gradients ---
    # Use the same grouping logic as the weight visualizer
    grouped_gradients = {
        "Embeddings": {},
        "Poolers": {},
        "Heads": {},
        "Loss": {},
        "Body": {},
    }
    for name, param in model.named_parameters():
        if param.grad is not None and param.requires_grad:
            # Create a copy of the gradient and apply unscaling if needed
            grad_tensor = param.grad.detach().clone()
            if scale_factor != 1.0:
                grad_tensor = grad_tensor / scale_factor
            grad_data = grad_tensor.cpu().numpy().flatten()

            if name.startswith("embeddings."):
                grouped_gradients["Embeddings"][name] = grad_data
            elif "pooler" in name:
                grouped_gradients["Poolers"][name] = grad_data
            elif "head" in name:
                grouped_gradients["Heads"][name] = grad_data
            elif "loss" in name:
                grouped_gradients["Loss"][name] = grad_data
            else:
                grouped_gradients["Body"][name] = grad_data

    if not any(grouped_gradients.values()):
        log("No gradients found to plot. Ensure this is called after loss.backward().")
        return

    log(f"\n--- Plotting Gradient Distributions (Step {global_step}) ---")
    base_title = f"Gradient Distributions{unscaled_str} at Step {global_step}"

    # --- 3. Loop through each group and create figures using the helper ---
    for group_name, gradients_in_group in grouped_gradients.items():
        if gradients_in_group:
            _plot_distributions_for_group(
                gradients_in_group=gradients_in_group,
                group_name=group_name,
                base_title=base_title,
                save_dir=save_dir,
                bins=bins,
                max_subplots_per_fig=max_subplots_per_fig,
                cols=cols,
            )
        else:
            log(f"Skipping '{group_name}' gradient plot as it has no layers.")
