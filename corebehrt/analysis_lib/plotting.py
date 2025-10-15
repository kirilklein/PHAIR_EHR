from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import math

plt.style.use("seaborn-v0_8-whitegrid")


def calculate_subplot_layout(n_subplots):
    """
    Calculate optimal subplot grid layout.

    Returns (nrows, ncols) for the subplot grid.
    Prefers layouts like 2x1, 2x2, 3x2, 3x3, etc.
    """
    if n_subplots == 1:
        return (1, 1)
    elif n_subplots == 2:
        return (2, 1)
    elif n_subplots <= 4:
        return (2, 2)
    elif n_subplots <= 6:
        return (3, 2)
    elif n_subplots <= 9:
        return (3, 3)
    elif n_subplots <= 12:
        return (4, 3)
    elif n_subplots <= 16:
        return (4, 4)
    else:
        # For larger numbers, calculate square-ish layout
        ncols = math.ceil(math.sqrt(n_subplots))
        nrows = math.ceil(n_subplots / ncols)
        return (nrows, ncols)


def split_into_batches(items, max_per_batch):
    """Split a list into batches of maximum size."""
    if max_per_batch is None or max_per_batch <= 0:
        return [items]

    batches = []
    for i in range(0, len(items), max_per_batch):
        batches.append(items[i : i + max_per_batch])
    return batches


def create_plot_from_agg(
    agg_data: pd.DataFrame,
    metric_col: str,
    y_label: str,
    title_prefix: str,
    output_dir: str,
    plot_type: str = "errorbar",
    max_subplots_per_figure: int = None,
    min_points: int = 2,
):
    """
    Generic plotting function for creating faceted plots with optimal grid layout.

    Args:
        agg_data: Aggregated data to plot
        metric_col: Column name for the metric to plot
        y_label: Label for y-axis
        title_prefix: Prefix for plot titles
        output_dir: Directory to save plots
        plot_type: Type of plot ('errorbar', 'line', 'dot')
        max_subplots_per_figure: Maximum subplots per figure (creates multiple figures if exceeded)
        min_points: Minimum number of data points required to generate a plot (default: 2)
    """
    if agg_data.empty:
        return

    # Plot vs Confounding
    all_instrument_levels = sorted(agg_data["i"].unique())
    instrument_levels_to_plot = [
        lvl
        for lvl in all_instrument_levels
        if agg_data[agg_data["i"] == lvl]["avg_confounding"].nunique() >= min_points
    ]
    if instrument_levels_to_plot:
        skipped = set(all_instrument_levels) - set(instrument_levels_to_plot)
        if skipped:
            print(
                f"Skipping {title_prefix} vs. Confounder plots for i={skipped} (fewer than {min_points} data points)."
            )

        # Split into batches if max_subplots_per_figure is specified
        batches = split_into_batches(instrument_levels_to_plot, max_subplots_per_figure)

        # Create output directory for this metric
        confounding_dir = Path(output_dir) / f"{metric_col}_vs_confounding"
        confounding_dir.mkdir(parents=True, exist_ok=True)

        for batch_idx, batch in enumerate(batches, start=1):
            n_subplots = len(batch)
            nrows, ncols = calculate_subplot_layout(n_subplots)

            # Calculate figure size based on grid
            fig_width = 5 * ncols
            fig_height = 4 * nrows

            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(fig_width, fig_height),
                sharey=True,
                constrained_layout=True,
            )

            # Flatten axes array for easy indexing
            if n_subplots == 1:
                axes = [axes]
            else:
                axes = axes.flatten() if nrows > 1 or ncols > 1 else [axes]

            fig.suptitle(
                f"{title_prefix} vs. Confounding Strength",
                fontsize=16,
                fontweight="bold",
            )

            for i, inst_level in enumerate(batch):
                ax = axes[i]
                subplot_data = agg_data[agg_data["i"] == inst_level]
                for method, color in [("TMLE", "blue"), ("IPW", "red")]:
                    method_data = subplot_data[subplot_data["method"] == method]
                    if not method_data.empty:
                        if plot_type == "errorbar":
                            ax.errorbar(
                                x=method_data["avg_confounding"],
                                y=method_data["mean"],
                                yerr=method_data["std"],
                                label=method,
                                color=color,
                                marker="o",
                                capsize=5,
                                linestyle="-",
                            )
                        else:
                            ax.plot(
                                method_data["avg_confounding"],
                                method_data[metric_col],
                                label=method,
                                color=color,
                                marker="o",
                                markersize=7,
                                linestyle="-" if plot_type == "line" else "",
                            )
                if plot_type == "errorbar":
                    ax.axhline(0, color="black", linestyle="--", alpha=0.7)
                if metric_col == "covered":
                    ax.axhline(
                        0.95,
                        color="gray",
                        linestyle="--",
                        alpha=0.9,
                        label="95% Target",
                    )
                ax.set_title(f"Instrument Strength (i) = {inst_level}")
                ax.set_xlabel("Average Confounding Strength")
                ax.set_ylabel(y_label)
                ax.legend()
                ax.grid(True, which="both", linestyle="--", linewidth=0.5)

            # Hide unused subplots
            for i in range(n_subplots, len(axes)):
                axes[i].set_visible(False)

            output_path = confounding_dir / f"fig_{batch_idx:02d}.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {title_prefix} vs. Confounding plot to: {output_path}")

    # Plot vs Instrument
    all_conf_levels = sorted(agg_data["avg_confounding"].unique())
    conf_levels_to_plot = [
        lvl
        for lvl in all_conf_levels
        if agg_data[agg_data["avg_confounding"] == lvl]["i"].nunique() >= min_points
    ]
    if conf_levels_to_plot:
        skipped = set(all_conf_levels) - set(conf_levels_to_plot)
        if skipped:
            print(
                f"Skipping {title_prefix} vs. Instrument plots for avg_confounding={skipped} (fewer than {min_points} data points)."
            )

        # Split into batches if max_subplots_per_figure is specified
        batches = split_into_batches(conf_levels_to_plot, max_subplots_per_figure)

        # Create output directory for this metric
        instrument_dir = Path(output_dir) / f"{metric_col}_vs_instrument"
        instrument_dir.mkdir(parents=True, exist_ok=True)

        for batch_idx, batch in enumerate(batches, start=1):
            n_subplots = len(batch)
            nrows, ncols = calculate_subplot_layout(n_subplots)

            # Calculate figure size based on grid
            fig_width = 5 * ncols
            fig_height = 4 * nrows

            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(fig_width, fig_height),
                sharey=True,
                constrained_layout=True,
            )

            # Flatten axes array for easy indexing
            if n_subplots == 1:
                axes = [axes]
            else:
                axes = axes.flatten() if nrows > 1 or ncols > 1 else [axes]

            fig.suptitle(
                f"{title_prefix} vs. Instrument Strength",
                fontsize=16,
                fontweight="bold",
            )

            for i, conf_level in enumerate(batch):
                ax = axes[i]
                subplot_data = agg_data[agg_data["avg_confounding"] == conf_level]
                for method, color in [("TMLE", "blue"), ("IPW", "red")]:
                    method_data = subplot_data[subplot_data["method"] == method]
                    if not method_data.empty:
                        if plot_type == "errorbar":
                            ax.errorbar(
                                x=method_data["i"],
                                y=method_data["mean"],
                                yerr=method_data["std"],
                                label=method,
                                color=color,
                                marker="o",
                                capsize=5,
                                linestyle="-",
                            )
                        else:
                            ax.plot(
                                method_data["i"],
                                method_data[metric_col],
                                label=method,
                                color=color,
                                marker="o",
                                markersize=7,
                                linestyle="-" if plot_type == "line" else "",
                            )
                if plot_type == "errorbar":
                    ax.axhline(0, color="black", linestyle="--", alpha=0.7)
                if metric_col == "covered":
                    ax.axhline(
                        0.95,
                        color="gray",
                        linestyle="--",
                        alpha=0.9,
                        label="95% Target",
                    )
                ax.set_title(f"Confounding Strength = {conf_level:.2f}")
                ax.set_xlabel("Instrument Strength (i)")
                ax.set_ylabel(y_label)
                ax.legend()
                ax.grid(True, which="both", linestyle="--", linewidth=0.5)

            # Hide unused subplots
            for i in range(n_subplots, len(axes)):
                axes[i].set_visible(False)

            output_path = instrument_dir / f"fig_{batch_idx:02d}.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {title_prefix} vs. Instrument plot to: {output_path}")


def create_method_comparison_plot(
    agg_data: pd.DataFrame,
    metric_name: str,
    y_label: str,
    title: str,
    output_dir: str,
    plot_type: str = "errorbar",
    min_points: int = 2,
):
    """
    Creates a method comparison plot with subplots for each (ce, cy, i) combination.

    Args:
        agg_data: Aggregated data with columns [method, ce, cy, i, outcome, mean, std]
        metric_name: Name of the metric (for filename)
        y_label: Label for y-axis
        title: Title for the overall figure
        output_dir: Directory to save the plot
        plot_type: Type of plot ('errorbar', 'line', 'dot')
        min_points: Minimum number of data points required to generate a plot
    """
    if agg_data.empty:
        print(f"No data to plot for {metric_name}")
        return

    # Get unique combinations of (ce, cy, i)
    param_combinations = (
        agg_data[["ce", "cy", "i"]]
        .drop_duplicates()
        .sort_values(["ce", "cy", "i"])
        .reset_index(drop=True)
    )

    # Filter out combinations with too few data points
    valid_combinations = []
    for _, row in param_combinations.iterrows():
        subset = agg_data[
            (agg_data["ce"] == row["ce"])
            & (agg_data["cy"] == row["cy"])
            & (agg_data["i"] == row["i"])
        ]
        if len(subset) >= min_points:
            valid_combinations.append(row)

    if not valid_combinations:
        print(
            f"No valid parameter combinations with >= {min_points} data points for {metric_name}"
        )
        return

    n_subplots = len(valid_combinations)
    nrows, ncols = calculate_subplot_layout(n_subplots)

    # Calculate figure size
    fig_width = 6 * ncols
    fig_height = 5 * nrows

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_width, fig_height),
        sharey=True,
        constrained_layout=True,
        facecolor="white",
    )

    # Flatten axes array
    if n_subplots == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if nrows > 1 or ncols > 1 else [axes]

    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.995)

    # Define method colors and markers with improved color palette
    method_styles = {
        "TMLE": {
            "color": "#2E86AB",
            "marker": "o",
            "linestyle": "",
            "label": "TMLE",
        },  # Deep blue
        "IPW": {
            "color": "#E63946",
            "marker": "s",
            "linestyle": "",
            "label": "IPW",
        },  # Red
        "TMLE_TH": {
            "color": "#06A77D",
            "marker": "^",
            "linestyle": "",
            "label": "TMLE-TH",
        },  # Teal/green
    }

    # Get all unique methods in the data
    unique_methods = sorted(agg_data["method"].unique())

    # Plot each subplot
    for idx, params in enumerate(valid_combinations):
        ax = axes[idx]
        ce, cy, i = params["ce"], params["cy"], params["i"]

        # Filter data for this parameter combination
        subplot_data = agg_data[
            (agg_data["ce"] == ce) & (agg_data["cy"] == cy) & (agg_data["i"] == i)
        ].copy()

        # Get unique outcomes and sort them
        outcomes = sorted(subplot_data["outcome"].unique())

        # Plot each method
        for method in unique_methods:
            method_data = subplot_data[subplot_data["method"] == method].copy()
            if method_data.empty:
                continue

            # Sort by outcome to ensure consistent ordering
            method_data = method_data.sort_values("outcome")

            style = method_styles.get(
                method,
                {"color": "#6C757D", "marker": "o", "linestyle": "", "label": method},
            )

            if plot_type == "errorbar":
                # Plot with error bars (no connecting lines)
                ax.errorbar(
                    x=range(len(method_data)),
                    y=method_data["mean"],
                    yerr=method_data["std"] if "std" in method_data.columns else None,
                    label=style["label"],
                    color=style["color"],
                    marker=style["marker"],
                    linestyle="",  # No connecting lines
                    capsize=4,
                    capthick=1.5,
                    markersize=8,
                    markeredgewidth=1.5,
                    markeredgecolor="white",
                    elinewidth=2,
                    alpha=0.85,
                    zorder=3,
                )
            elif plot_type == "dot":
                # Plot dots without error bars (for coverage)
                ax.plot(
                    range(len(method_data)),
                    method_data["mean"],
                    label=style["label"],
                    color=style["color"],
                    marker=style["marker"],
                    linestyle="",
                    markersize=10,
                    markeredgewidth=1.5,
                    markeredgecolor="white",
                    alpha=0.85,
                    zorder=3,
                )
            else:  # line
                # Plot with markers only
                ax.plot(
                    range(len(method_data)),
                    method_data["mean"],
                    label=style["label"],
                    color=style["color"],
                    marker=style["marker"],
                    linestyle="",  # No connecting lines
                    markersize=8,
                    markeredgewidth=1.5,
                    markeredgecolor="white",
                    alpha=0.85,
                    zorder=3,
                )

        # Add reference lines
        if plot_type == "errorbar":
            ax.axhline(
                0, color="#2C3E50", linestyle="--", alpha=0.4, linewidth=1.5, zorder=1
            )

        if metric_name == "coverage":
            ax.axhline(
                0.95,
                color="#E74C3C",
                linestyle=":",
                alpha=0.6,
                linewidth=2,
                label="95% Target",
                zorder=1,
            )

        # Set subplot title with improved styling
        ax.set_title(
            f"ce={ce}, cy={cy}, i={i}", fontsize=12, fontweight="semibold", pad=10
        )
        ax.set_xlabel(
            "Outcome (Effect Strength)", fontsize=11, fontweight="medium", labelpad=8
        )
        ax.set_ylabel(y_label, fontsize=11, fontweight="medium", labelpad=8)

        # Set x-axis ticks to outcome names
        ax.set_xticks(range(len(outcomes)))
        ax.set_xticklabels(outcomes, rotation=45, ha="right", fontsize=9)

        # Improve legend styling
        legend = ax.legend(
            fontsize=10,
            loc="best",
            framealpha=0.95,
            edgecolor="#CCCCCC",
            fancybox=True,
            shadow=False,
        )
        legend.get_frame().set_linewidth(1.5)

        # Improve grid styling
        ax.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.3, zorder=0)
        ax.set_axisbelow(True)  # Ensure grid is below data points

        # Improve spine styling
        for spine in ax.spines.values():
            spine.set_edgecolor("#CCCCCC")
            spine.set_linewidth(1.2)

        # Set clean white background for subplot
        ax.set_facecolor("white")

    # Hide unused subplots and remove their borders
    for idx in range(n_subplots, len(axes)):
        axes[idx].set_visible(False)

    # Save the figure with high quality settings
    output_path = Path(output_dir) / f"{metric_name}_method_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        pad_inches=0.1,
    )
    plt.close(fig)
    print(f"Saved {metric_name} method comparison plot to: {output_path}")
