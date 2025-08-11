import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch.nn as nn
from torch.optim import Optimizer
from torch.cuda.amp import GradScaler
from typing import Callable, Optional, Tuple, Dict, List
from collections import defaultdict


def _get_parameter_group(name: str) -> str:
    """Categorizes a parameter name into a logical model group."""
    if name.startswith("bert.encoder") or name.startswith("encoder.layer"):
        return "Encoder"
    if name.startswith("bert.embeddings") or name.startswith("embeddings"):
        return "Embeddings"
    if name.startswith("pooler"):
        return "Shared Pooler"
    if name.startswith("exposure_pooler"):
        return "Exposure Pooler"
    if name.startswith("outcome_poolers"):
        return "Outcome Poolers"
    if name.startswith("encoder_bottleneck"):
        return "Bottleneck"
    if name.startswith("exposure_head"):
        return "Exposure Head"
    if name.startswith("outcome_heads"):
        # e.g., 'outcome_heads.MI.classifier.1.weight' -> 'Outcome Head: MI'
        parts = name.split(".")
        if len(parts) > 1:
            return f"Outcome Head ({parts[1]})"
        return "Outcome Heads"
    return "Other"


def plot_gradient_distributions(
    model: nn.Module,
    optimizer: Optimizer,
    scaler: Optional[GradScaler] = None,
    log: Callable[[str], None] = print,
    run_folder: str = ".",
    global_step: int = 0,
    grid_size: Tuple[int, int] = (4, 3),
) -> None:
    """
    Inspects and plots the distributions of gradients, grouped by model component.

    This function automatically paginates plots and saves them as separate figures
    for each logical group of parameters (Encoder, Heads, etc.). It MUST be
    called after `loss.backward()` and before `optimizer.step()`.

    Args:
        model (nn.Module): The model whose gradients are to be plotted.
        optimizer (Optimizer): The optimizer, needed for unscaling gradients.
        scaler (Optional[GradScaler]): The GradScaler for mixed precision.
        log (Callable[[str], None]): A logging function (e.g., `print`).
        run_folder (str): Base folder to save figures.
        global_step (int): The current training step for file naming.
        grid_size (Tuple[int, int]): The (rows, cols) for subplots per figure.
    """
    # --- 1. Unscale gradients to view their true values ---
    save_dir = os.path.join(run_folder, "figs", "gradient_plots")
    os.makedirs(save_dir, exist_ok=True)

    if scaler and scaler.is_enabled():
        try:
            scaler.unscale_(optimizer)
        except Exception as e:
            log(f"Warning: Could not unscale gradients for plotting: {e}")

    # --- 2. Extract and group all available gradients ---
    grouped_gradients: Dict[str, List[Tuple[str, np.ndarray]]] = defaultdict(list)
    for name, param in model.named_parameters():
        if param.grad is not None:
            group = _get_parameter_group(name)
            grad_data = param.grad.detach().cpu().numpy().flatten()
            grouped_gradients[group].append((name, grad_data))

    if not grouped_gradients:
        log("No gradients found to plot. Ensure this is called after loss.backward().")
        return

    log(
        f"\n--- Plotting Gradients for {len(grouped_gradients)} Groups (Step {global_step}) ---"
    )
    rows, cols = grid_size
    subplots_per_fig = rows * cols
    total_plots_saved = 0

    # --- 3. Loop through each group and create figures ---
    for group_name, gradient_items in sorted(grouped_gradients.items()):
        num_total_plots_in_group = len(gradient_items)

        # Paginate if the number of parameters in the group exceeds the grid size
        for i in range(0, num_total_plots_in_group, subplots_per_fig):
            chunk = gradient_items[i : i + subplots_per_fig]
            page_num = (i // subplots_per_fig) + 1

            plt.style.use("seaborn-v0_8-whitegrid")
            # Make subplots less tall
            fig, axes = plt.subplots(
                rows, cols, figsize=(cols * 4, rows * 2.8), constrained_layout=True
            )
            axes = np.array(axes).flatten()

            # Plot each gradient distribution in the chunk
            for j, (name, grad_data) in enumerate(chunk):
                ax = axes[j]
                sns.histplot(
                    grad_data,
                    bins=80,
                    ax=ax,
                    kde=True,
                    color="#2a9d8f",
                    alpha=0.7,
                    edgecolor="white",
                )
                # Set larger title, with no stats (mu/sigma)
                ax.set_title(f"{name}", fontsize=11, wrap=True)
                ax.set_xlabel("Gradient Value", fontsize=8)
                ax.set_ylabel("")
                ax.set_yticks([])
                ax.tick_params(axis="x", labelsize=8)
                ax.ticklabel_format(style="sci", axis="x", scilimits=(-2, 2))

            # Hide unused subplots
            for j in range(len(chunk), subplots_per_fig):
                axes[j].set_visible(False)

            unscaled_str = " (Unscaled)" if scaler else ""
            fig.suptitle(
                f"Gradient Distributions: {group_name}{unscaled_str}\nStep {global_step} (Page {page_num})",
                fontsize=16,
                weight="bold",
            )

            # Save the figure with a group-specific name
            safe_group_name = (
                group_name.replace(" ", "_")
                .replace(":", "")
                .replace("(", "")
                .replace(")", "")
            )
            plot_path = os.path.join(
                save_dir,
                f"grads_step_{global_step}_group_{safe_group_name}_p{page_num}.png",
            )
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            total_plots_saved += 1

    log(f"Saved gradient plots across {total_plots_saved} figure(s) in '{save_dir}'")
