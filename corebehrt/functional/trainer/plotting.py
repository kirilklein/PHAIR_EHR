from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Callable, Optional

import matplotlib.pyplot as plt
import matplotlib.style as style
from corebehrt.constants.causal.data import EXPOSURE

# --- Configuration ---
style.use("seaborn-v0_8-whitegrid")
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
    - Each outcome's metrics are saved in `output_dir/{outcome_name}/`.

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
    Parses metric names using the explicit outcome_names list and groups them.

    Returns:
        A dictionary where keys are (group, base_metric) tuples and values are
        the data for that plot. The 'group' corresponds to the subdirectory.
    """
    outcome_names = outcome_names or []
    structured_metrics = defaultdict(dict)

    for name, values in metric_history.items():
        if not values:
            continue

        parts = name.split("_")
        if len(parts) < 2:
            continue

        prefix = parts[0]
        metric_body = "_".join(parts[1:])

        group_name: Optional[str] = None
        base_metric: Optional[str] = None

        # 1. Check for the 'loss' special case
        if metric_body == "loss":
            group_name = None  # Signals root directory
            base_metric = "loss"

        # 2. Check for the 'exposure' special case
        elif EXPOSURE in metric_body:
            group_name = EXPOSURE
            base_metric = metric_body.replace(f"{EXPOSURE}_", "").strip("_") or EXPOSURE

        # 3. Check against the provided list of outcome names
        else:
            for outcome in outcome_names:
                if outcome in metric_body:
                    group_name = outcome
                    base_metric = metric_body.replace(f"{outcome}_", "").strip("_")
                    break

        # Add the parsed metric to our structured dictionary
        if base_metric:
            structured_metrics[(group_name, base_metric)][prefix] = values
        else:
            log_func(f"⚠️ Skipping unrecognized metric: '{name}'")

    return structured_metrics


def _create_metric_plot(
    group: Optional[str],
    base_metric: str,
    data: Dict[str, List[Any]],
    epochs: List[int],
    output_dir: Path,
    log_func: Callable[[str], None],
):
    """Creates, styles, and saves a single plot for a group of metrics."""
    plot_dir = output_dir / group if group else output_dir
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig_path = plot_dir / f"{base_metric}.png"

    fig, ax = plt.subplots(figsize=STYLE_CONFIG["figure_size"])

    for prefix, values in data.items():
        try:
            numeric_values = [float(v) for v in values]
            if len(numeric_values) != len(epochs):
                log_func(
                    f"⚠️ Length mismatch for '{prefix}_{base_metric}'. Skipping plot."
                )
                continue

            ax.plot(
                epochs,
                numeric_values,
                label=prefix.title(),
                color=STYLE_CONFIG["colors"].get(
                    prefix, STYLE_CONFIG["colors"]["default"]
                ),
                marker="o",
                linestyle="-",
                markersize=4,
                alpha=0.8,
            )
        except (ValueError, TypeError):
            log_func(f"⚠️ Non-numeric data for '{prefix}_{base_metric}'. Skipping plot.")

    # --- Styling ---
    title_name = base_metric.replace("_", " ").title()
    plot_group_name = f" for {group.replace('_', ' ').title()}" if group else ""

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(title_name, fontsize=12)
    ax.set_title(
        f"{title_name}{plot_group_name} Over Epochs", fontsize=16, weight="bold"
    )
    ax.legend(title="Split")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    # --- Saving and Closing ---
    try:
        fig.savefig(fig_path, dpi=STYLE_CONFIG["dpi"], bbox_inches="tight")
        log_func(f"✅ Plot saved to '{fig_path}'")
    except Exception as e:
        log_func(f"❌ Failed to save plot '{fig_path}'. Error: {e}")
    finally:
        plt.close(fig)
