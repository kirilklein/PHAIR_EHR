import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import pandas as pd
import torch

from corebehrt.azure.util import log as azure_log
from corebehrt.constants.causal.data import EXPOSURE
from corebehrt.functional.visualize.calibrate import plot_probas_hist
from corebehrt.modules.trainer.causal.utils import CausalPredictionData

# --- Configuration ---
try:
    style.use("seaborn-v0_8-whitegrid")
except:
    pass
STYLE_CONFIG = {
    "colors": {
        "train": "#1f77b4",  # Muted Blue
        "val": "#ff7f0e",  # Safety Orange
        "test": "#2ca02c",  # Cooked Asparagus Green
        "default": "#7f7f7f",  # Middle Gray
    },
    "figure_size": (12, 7),
    "dpi": 100,
}


def plot_training_curves(
    metric_history: Dict[str, List[Any]],
    epoch_history: List[int],
    output_dir: Path | str,
    outcome_names: Optional[List[str]] = None,
    log_func: Callable[[str], None] = print,
):
    """
    Groups, plots, and saves training curves with clear directory organization.

    This function organizes plots as follows:
    - `loss` plots are saved directly in `output_dir`.
    - `exposure` metrics are saved in `output_dir/exposure/`.
    - All outcome metrics are grouped by metric type (e.g., a single 'auroc' plot
      for all outcomes) and saved in `output_dir/outcomes/`.

    Args:
        metric_history: Dictionary mapping metric names to their value history.
        epoch_history: List of epoch numbers corresponding to the metrics.
        output_dir: The root directory where figure directories will be created.
        outcome_names: A list of outcome names to identify and group metrics.
        log_func: A logging function (e.g., logger.info) to report progress.
    """
    if not epoch_history:
        log_func("⚠️ Skipping plotting; epoch history is empty.")
        return

    output_dir = Path(output_dir)

    grouped_plots = _group_metrics_for_plotting(metric_history, outcome_names, log_func)
    if not grouped_plots:
        log_func("ℹ️ No metrics found to plot.")
        return

    log_func(f"📈 Found {len(grouped_plots)} plot groups. Generating figures...")

    for (group, base_metric), data in grouped_plots.items():
        _create_metric_plot(
            group=group,
            base_metric=base_metric,
            data=data,
            epochs=epoch_history,
            output_dir=output_dir,
            log_func=log_func,
        )


# --- Helper Functions ---


def _group_metrics_for_plotting(
    metric_history: Dict[str, List[Any]],
    outcome_names: Optional[List[str]],
    log_func: Callable[[str], None],
) -> Dict:
    """
    Parses metric names to group them for plotting.

    - 'loss' metrics get their own plot.
    - 'exposure' metrics are grouped into an 'exposure' directory.
    - All specified 'outcome' metrics are grouped together by base metric
      (e.g., 'auroc'), resulting in a single plot per metric that contains
      lines for all outcomes.

    Returns:
        A dictionary where keys are (group, base_metric) tuples and values are
        the data for that plot. The 'group' corresponds to the subdirectory.
    """
    outcome_names = sorted(outcome_names or [], key=len, reverse=True)
    grouped_plots = defaultdict(dict)

    for name, values in metric_history.items():
        if not values:
            continue

        parts = name.split("_")
        if len(parts) < 2:
            continue

        prefix = parts[0]  # 'train', 'val', etc.
        metric_body = "_".join(parts[1:])

        # Case 1: Handle 'loss' metric
        if metric_body == "loss":
            group_key = (None, "loss")  # Group=None saves to root output_dir
            grouped_plots[group_key][prefix] = values
            continue

        # Case 2: Handle 'exposure' metrics
        if EXPOSURE in metric_body:
            base_metric = metric_body.replace(f"{EXPOSURE}_", "").strip("_") or EXPOSURE
            group_key = (EXPOSURE, base_metric)
            grouped_plots[group_key][prefix] = values
            continue

        # Case 3: Handle outcome metrics
        matched = False
        for outcome in outcome_names:
            if metric_body.startswith(f"{outcome}_") or metric_body == outcome:
                group_name = "outcomes"  # All outcomes go to the 'outcomes' group
                base_metric = (
                    metric_body.replace(f"{outcome}_", "").strip("_") or "value"
                )
                group_key = (group_name, base_metric)

                # Create a descriptive key for the line, e.g., "Train Diabetes"
                line_key = f"{prefix.title()} {outcome.replace('_', ' ').title()}"
                grouped_plots[group_key][line_key] = values
                matched = True
                break

        if not matched:
            log_func(f"⚠️ Skipping unrecognized metric: '{name}'")

    return grouped_plots


def _create_metric_plot(
    group: Optional[str],
    base_metric: str,
    data: Dict[str, List[Any]],
    epochs: List[int],
    output_dir: Path,
    log_func: Callable[[str], None],
):
    """Creates, styles, and saves a single plot for a group of metrics."""
    # Determine plot directory: 'output_dir/outcomes', 'output_dir/exposure', or 'output_dir'
    plot_dir = output_dir / group if group else output_dir
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig_path = plot_dir / f"{base_metric}.png"

    fig, ax = plt.subplots(figsize=STYLE_CONFIG["figure_size"])

    # --- Color mapping for outcomes ---
    color_map = {}
    if group == "outcomes":
        # Extract unique outcomes to assign a consistent color to each
        unique_outcomes = sorted(
            list(set(" ".join(label.split()[1:]) for label in data.keys()))
        )
        num_unique_outcomes = len(unique_outcomes)

        # Use a continuous colormap and discretize it for a large number of outcomes
        if num_unique_outcomes > 0:
            colormap = plt.colormaps.get("viridis")
            colors = colormap(np.linspace(0, 1, num_unique_outcomes))
            color_map = {
                outcome: colors[i] for i, outcome in enumerate(unique_outcomes)
            }

    # --- Plotting loop ---
    for line_label, values in data.items():
        try:
            numeric_values = [float(v) for v in values]
            if len(numeric_values) != len(epochs):
                log_func(f"⚠️ Length mismatch for '{line_label}'. Skipping plot line.")
                continue

            # Infer prefix ('train', 'val') from the line label to set style
            prefix = line_label.split()[0].lower()
            linestyle = "--" if prefix == "val" else "-"

            # Determine color: unique for each outcome, or standard for loss/exposure
            if group == "outcomes":
                outcome_name = " ".join(line_label.split()[1:])
                color = color_map.get(outcome_name, STYLE_CONFIG["colors"]["default"])
            else:
                color = STYLE_CONFIG["colors"].get(
                    prefix, STYLE_CONFIG["colors"]["default"]
                )

            ax.plot(
                epochs,
                numeric_values,
                label=line_label.title(),
                color=color,
                marker="o",
                linestyle=linestyle,
                markersize=4,
                alpha=0.8,
            )
        except (ValueError, TypeError):
            log_func(f"⚠️ Non-numeric data for '{line_label}'. Skipping plot line.")

    # --- Styling ---
    title_name = base_metric.replace("_", " ").title()
    plot_group_name = (
        f" for {group.replace('_', ' ').title()}"
        if group and group != "outcomes"
        else ""
    )
    plot_title = f"{title_name}{plot_group_name} Over Epochs"
    if group == "outcomes":
        plot_title = f"{title_name} for All Outcomes Over Epochs"

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(title_name, fontsize=12)
    ax.set_title(plot_title, fontsize=16, weight="bold")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    # --- Legend Handling ---
    num_lines = len(data)
    if num_lines > 100:
        log_func(
            f"ℹ️ Omitting legend for '{base_metric}' plot as it has {num_lines} lines (>100)."
        )
    else:
        legend_title = "Split / Outcome" if group == "outcomes" else "Split"
        ax.legend(title=legend_title)

    # --- Saving and Closing ---
    try:
        fig.savefig(fig_path, dpi=STYLE_CONFIG["dpi"], bbox_inches="tight")
        azure_log.log_figure(
            key=base_metric,
            figure=fig,
            artifact_file=f"figs/{group}/{base_metric}.png"
            if group
            else f"figs/{base_metric}.png",
        )
        log_func(f"✅ Plot saved to '{fig_path}'")
    except Exception as e:
        log_func(f"❌ Failed to save plot '{fig_path}'. Error: {e}")
    finally:
        plt.close(fig)


def plot_prediction_histograms(
    prediction_data: Dict[str, CausalPredictionData],
    run_folder: str,
    outcome_names: List[str],
    accumulate_logits: bool,
):
    """Plots prediction histograms for exposure and outcomes."""
    if not accumulate_logits:
        return

    hist_dir = os.path.join(run_folder, "figs", "histograms")
    os.makedirs(hist_dir, exist_ok=True)

    # Plot for exposure
    if EXPOSURE in prediction_data and prediction_data[EXPOSURE].logits_list:
        create_and_save_hist(
            logits=torch.cat(prediction_data[EXPOSURE].logits_list),
            targets=torch.cat(prediction_data[EXPOSURE].targets_list),
            title="Exposure Prediction Distribution",
            filename="exposure_predictions",
            save_dir=hist_dir,
        )

    # Plot for outcomes
    for outcome_name in outcome_names:
        if (
            outcome_name in prediction_data
            and prediction_data[outcome_name].logits_list
        ):
            create_and_save_hist(
                logits=torch.cat(prediction_data[outcome_name].logits_list),
                targets=torch.cat(prediction_data[outcome_name].targets_list),
                title=f"{outcome_name} Prediction Distribution",
                filename=f"{outcome_name}_predictions",
                save_dir=hist_dir,
            )


def create_and_save_hist(
    logits: torch.Tensor,
    targets: torch.Tensor,
    title: str,
    filename: str,
    save_dir: str,
):
    probas = torch.sigmoid(logits).squeeze().numpy()
    targets = targets.squeeze().numpy()
    df = pd.DataFrame({"probas": probas, "targets": targets})
    plot_probas_hist(
        df,
        value_col="probas",
        group_col="targets",
        group_labels=("Negative", "Positive"),
        title=title,
        xlabel="Predicted Probability",
        name=filename,
        save_dir=save_dir,
    )
