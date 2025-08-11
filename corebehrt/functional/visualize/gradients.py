import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch.nn as nn
from torch.optim import Optimizer
from torch.cuda.amp import GradScaler
from typing import Callable
from typing import Optional, Tuple


def plot_gradient_distributions(
    model: nn.Module,
    optimizer: Optimizer,
    scaler: Optional[GradScaler] = None,
    log: Callable[[str], None] = print,
    run_folder: str = ".",
    global_step: int = 0,
    grid_size: Tuple[int, int] = (5, 3),
) -> None:
    """
    Inspects and plots the distributions of gradients for all model parameters.

    This function automatically paginates the plots across multiple figures if the
    number of parameters exceeds the grid size. It MUST be called after
    `loss.backward()` and before `optimizer.step()`.

    Args:
        model (nn.Module): The model whose gradients are to be plotted.
        optimizer (Optimizer): The optimizer, needed for unscaling gradients.
        scaler (Optional[GradScaler]): The GradScaler, if using mixed precision.
                                       If None, unscaling is skipped.
        log (Callable[[str], None]): A logging function (e.g., `print`).
        run_folder (str): The base folder for the run, where figures will be saved.
        global_step (int): The current training step, used for file naming.
        grid_size (Tuple[int, int]): The (rows, cols) for subplots in each figure.
                                     Defaults to (3, 3).
    """
    # --- 1. Unscale gradients to view their true values ---
    save_dir = os.path.join(run_folder, "figs", "gradient_plots")
    os.makedirs(save_dir, exist_ok=True)

    if scaler:
        try:
            # This is an in-place operation that modifies the .grad attribute
            scaler.unscale_(optimizer)
        except Exception as e:
            log(f"Warning: Could not unscale gradients for plotting: {e}")
            # Continue without unscaling if it fails, but log the issue.

    # --- 2. Extract all available gradients ---
    all_gradients = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Use .detach() to ensure no computational graph is attached
            grad_data = param.grad.detach().cpu().numpy().flatten()
            all_gradients[name] = grad_data

    if not all_gradients:
        log("No gradients found to plot. Ensure this is called after loss.backward().")
        return

    # --- 3. Setup for plotting and pagination ---
    log(
        f"\n--- Plotting {len(all_gradients)} Gradient Distributions (Step {global_step}) ---"
    )
    gradient_items = list(all_gradients.items())
    rows, cols = grid_size
    subplots_per_fig = rows * cols

    # --- 4. Loop through chunks and create figures ---
    num_total_plots = len(gradient_items)
    for i in range(0, num_total_plots, subplots_per_fig):
        chunk = gradient_items[i : i + subplots_per_fig]
        fig_num = (i // subplots_per_fig) + 1

        # Create a new figure for the current chunk
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, axes = plt.subplots(
            rows, cols, figsize=(cols * 4.5, rows * 3.5), constrained_layout=True
        )
        axes = np.array(axes).flatten()

        # Plot each gradient distribution in the chunk
        for j, (name, grad_data) in enumerate(chunk):
            ax = axes[j]

            # Use a nice color and some transparency
            sns.histplot(
                grad_data,
                bins=80,
                ax=ax,
                kde=True,
                color="#2a9d8f",
                alpha=0.7,
                edgecolor="white",
            )

            grad_mean = np.mean(grad_data)
            grad_std = np.std(grad_data)

            # Format title with stats and wrap long names
            ax.set_title(
                f"{name}\nμ={grad_mean:.2e}, σ={grad_std:.2e}", fontsize=9, wrap=True
            )
            ax.set_xlabel("Gradient Value", fontsize=8)
            ax.set_ylabel("")  # De-clutter y-axis
            ax.set_yticks([])
            ax.tick_params(axis="x", labelsize=8)
            ax.ticklabel_format(style="sci", axis="x", scilimits=(-2, 2))

        # Hide any unused subplots on the last page
        for j in range(len(chunk), subplots_per_fig):
            axes[j].set_visible(False)

        unscaled_str = " (Unscaled)" if scaler else ""
        fig.suptitle(
            f"Gradient Distributions{unscaled_str} - Step {global_step} (Part {fig_num})",
            fontsize=16,
            weight="bold",
        )

        # Save the figure
        plot_path = os.path.join(
            save_dir, f"gradient_dist_step_{global_step}_part_{fig_num}.png"
        )
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)  # Close the figure to free up memory

    num_figures = int(np.ceil(num_total_plots / subplots_per_fig))
    log(
        f"Saved {num_total_plots} gradient plots across {num_figures} figure(s) in '{save_dir}'"
    )
