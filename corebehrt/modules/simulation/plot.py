from os.path import join
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def plot_hist(p_exposure, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(p_exposure, bins=50)
    ax.set_title("Exposure Probability Histogram")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Count")
    ax.set_xlim(0, 1)
    plt.savefig(join(output_dir, "exposure_probability_histogram.png"))
    plt.close(fig)


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
    axes = axes.flatten()
    bins = np.linspace(0, 1, bins)
    for i, (outcome_name, probas) in enumerate(all_probas.items()):
        ax = axes[i]

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
    plt.savefig(plot_path)
    plt.close(fig)
