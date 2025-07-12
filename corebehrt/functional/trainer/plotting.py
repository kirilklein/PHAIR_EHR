# corebehrt/evaluation/plotting.py
import os
from collections import defaultdict
import matplotlib.pyplot as plt


def plot_training_curves(
    metric_history: dict, epoch_history: list, figs_dir: str, logger=print
):
    """
    Plots all recorded training metrics and saves them as image files.
    """
    if len(epoch_history) < 2:
        return  # Not enough data to plot

    os.makedirs(figs_dir, exist_ok=True)
    metric_groups = _group_metrics(metric_history, epoch_history)

    for group_name, metrics_data in metric_groups.items():
        _plot_metric_group(group_name, metrics_data, epoch_history, figs_dir, logger)


def _group_metrics(metric_history: dict, epoch_history: list) -> dict:
    """Groups related metrics (e.g., train/val/test loss) for plotting."""
    groups = defaultdict(dict)
    for name, values in metric_history.items():
        if len(values) != len(epoch_history):
            continue  # Skip metrics that weren't recorded for all epochs

        parts = name.split("_")
        prefix = parts[0]  # e.g., 'val', 'train', 'test'
        base_name = "_".join(parts[1:])  # e.g., 'loss', 'roc_auc', 'exposure_loss'

        groups[base_name][prefix] = values
    return groups


def _plot_metric_group(
    group_name: str, data: dict, epochs: list, figs_dir: str, logger: callable
):
    """Plots a single figure for a group of metrics."""
    plt.figure(figsize=(10, 6))
    colors = {"train": "blue", "val": "orange", "test": "green"}

    for prefix, values in data.items():
        try:
            numeric_values = [float(v) for v in values]
            plt.plot(
                epochs,
                numeric_values,
                label=f"{prefix}",
                color=colors.get(prefix, "black"),
                marker="o",
                markersize=3,
            )
        except (ValueError, TypeError):
            logger(f"Warning: Skipping non-numeric plot for {prefix}_{group_name}")

    plt.xlabel("Epoch")
    plt.ylabel(group_name.replace("_", " ").title())
    plt.title(f"{group_name.replace('_', ' ').title()} Over Training")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(figs_dir, f"{group_name}.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()
