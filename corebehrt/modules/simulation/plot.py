from os.path import join
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from corebehrt.functional.utils.azure_save import save_figure_with_azure_copy


def plot_hist(p_exposure, output_dir, is_exposed: Optional[np.ndarray] = None):
    """
    Plot histogram of exposure probabilities, optionally colored by actual exposure status.

    Args:
        p_exposure: Array of predicted exposure probabilities
        output_dir: Directory to save the plot
        is_exposed: Optional array of actual exposure status (boolean)
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    if is_exposed is not None:
        # Split probabilities by actual exposure status
        p_control = p_exposure[~is_exposed]  # Not exposed (control)
        p_treated = p_exposure[is_exposed]  # Exposed (treated)

        # Create overlaid histograms with different colors
        bins = np.linspace(0, 1, 51)
        ax.hist(
            p_control,
            bins=bins,
            alpha=0.7,
            label="Control (Not Exposed)",
            color="#3498db",  # Blue
            density=True,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.hist(
            p_treated,
            bins=bins,
            alpha=0.7,
            label="Treated (Exposed)",
            color="#e74c3c",  # Red
            density=True,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.legend()
        ax.set_ylabel("Density")
        ax.set_title("Exposure Probability Distribution by Actual Exposure Status")
    else:
        # Original single histogram
        ax.hist(p_exposure, bins=50)
        ax.set_ylabel("Count")
        ax.set_title("Exposure Probability Histogram")

    ax.set_xlabel("Predicted Probability")
    ax.set_xlim(0, 1)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    save_figure_with_azure_copy(
        fig, join(output_dir, "exposure_probability_histogram.png")
    )


def plot_probability_distributions(
    all_probas: Dict[str, Dict[str, np.ndarray]], output_dir: str, bins: int = 50
):
    """Plots overlaid histograms of P0 and P1 for each outcome for easier interpretability."""

    num_outcomes = len(all_probas)
    if num_outcomes == 0:
        return

    # Layout: 2 columns if more than 1 outcome, else 1 column
    ncols = 2 if num_outcomes > 1 else 1
    nrows = (num_outcomes + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(8 * ncols, 6 * nrows), squeeze=False
    )
    axes: List[Axes] = axes.flatten()
    bins = np.linspace(0, 1, bins)
    for i, (outcome_name, probas) in enumerate(all_probas.items()):
        ax: Axes = axes[i]

        # Histogram for treated (P1)
        ax.hist(
            probas["P1"],
            bins=bins,
            range=(0, 1),
            alpha=0.5,
            label="P1 (Treated)",
            edgecolor="black",
        )
        # Histogram for control (P0)
        ax.hist(
            probas["P0"],
            bins=bins,
            range=(0, 1),
            alpha=0.5,
            label="P0 (Control)",
            edgecolor="black",
        )

        ax.set_title(f"Probability Histograms for {outcome_name}")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 1)
        ax.legend()

    # Hide any unused subplots
    for j in range(num_outcomes, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plot_path = join(output_dir, "probability_histograms.png")
    save_figure_with_azure_copy(fig, plot_path)
