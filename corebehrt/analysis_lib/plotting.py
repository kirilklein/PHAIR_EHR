from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from matplotlib.offsetbox import AnchoredText
from matplotlib.lines import Line2D

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


def calculate_horizontal_subplot_layout(n_subplots):
    """
    Calculate horizontal-first subplot grid layout.
    
    Returns (nrows, ncols) for the subplot grid.
    Prefers layouts like 1x2, 2x2, 2x3, 3x3, etc.
    """
    if n_subplots == 1:
        return (1, 1)
    elif n_subplots == 2:
        return (1, 2)
    elif n_subplots <= 4:
        return (2, 2)
    elif n_subplots <= 6:
        return (2, 3)
    elif n_subplots <= 9:
        return (3, 3)
    elif n_subplots <= 12:
        return (3, 4)
    elif n_subplots <= 16:
        return (4, 4)
    else:
        # For larger numbers, prefer wider layouts
        ncols = math.ceil(math.sqrt(n_subplots * 1.2))  # Slightly wider
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
    max_subplots: int = None,
    description: str | None = None,
    desc_on: str = "first",          # "each" | "first"
    legend_location: str = "right",  # "right" | "bottom"
):
    """
    Create a method comparison plot with subplots for each (ce, cy, i) combination.
    - Adds an inset description box (inside the plot).
    - Moves the legend outside the plot area as a single figure-level legend.
    """
    if agg_data.empty:
        print(f"No data to plot for {metric_name}")
        return

    # method styling
    method_styles = {
        "TMLE":    {"color": "#2E86AB", "marker": "o", "label": "TMLE"},
        "IPW":     {"color": "#E63946", "marker": "s", "label": "IPW"},
        "TMLE_TH": {"color": "#06A77D", "marker": "^", "label": "TMLE-TH"},
    }

    # find valid parameter triplets
    param_combinations = (
        agg_data[["ce", "cy", "i"]]
        .drop_duplicates()
        .sort_values(["ce", "cy", "i"])
        .reset_index(drop=True)
    )
    valid = []
    for _, row in param_combinations.iterrows():
        subset = agg_data[(agg_data["ce"] == row["ce"]) & (agg_data["cy"] == row["cy"]) & (agg_data["i"] == row["i"])]
        if len(subset) >= min_points:
            valid.append(row)
    if not valid:
        print(f"No valid parameter combinations with >= {min_points} data points for {metric_name}")
        return

    # Split into batches if max_subplots is specified
    batches = split_into_batches(valid, max_subplots)
    
    for batch_idx, batch in enumerate(batches, start=1):
        n_subplots = len(batch)
        nrows, ncols = calculate_horizontal_subplot_layout(n_subplots)
        
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(6 * ncols, 5 * nrows),
            sharey=True,
            constrained_layout=False,
            facecolor="white",
        )
        fig.subplots_adjust(top=0.85)  # Add more top margin for suptitle
        axes = [axes] if n_subplots == 1 else axes.flatten()
        
        # Add batch number to title if multiple batches
        batch_title = title
        if len(batches) > 1:
            batch_title = f"{title} (Part {batch_idx}/{len(batches)})"
        fig.suptitle(batch_title, fontsize=14, fontweight="bold", y=0.92)

        methods_present = set()
        
        # Calculate method offsets for better visibility
        unique_methods = sorted(agg_data["method"].unique())
        method_offsets = {}
        if len(unique_methods) > 1:
            offset_range = 0.2  # Total offset range (reduced from 0.3)
            offsets = np.linspace(-offset_range/2, offset_range/2, len(unique_methods))
            method_offsets = {method: offset for method, offset in zip(unique_methods, offsets)}

        for idx, params in enumerate(batch):
            ax = axes[idx]
            ce, cy, i = params["ce"], params["cy"], params["i"]
            subplot_data = agg_data[(agg_data["ce"] == ce) & (agg_data["cy"] == cy) & (agg_data["i"] == i)].copy()

            # outcomes for ticks
            outcomes = sorted(subplot_data["outcome"].unique())

            for method in sorted(subplot_data["method"].unique()):
                md = subplot_data[subplot_data["method"] == method].sort_values("outcome").copy()
                if md.empty:
                    continue
                st = method_styles.get(method, {"color": "#6C757D", "marker": "o", "label": method})
                methods_present.add(method)

                # Apply method offset
                base_x = np.array(range(len(md)))
                x = base_x + method_offsets.get(method, 0)
                if plot_type == "errorbar":
                    # Only show error bars for metrics that have std (bias, relative_bias, z_score)
                    yerr = md["std"] if "std" in md.columns and not md["std"].isna().all() else None
                    ax.errorbar(
                        x, md["mean"],
                        yerr=yerr,
                        marker=st["marker"], linestyle="",
                        color=st["color"], capsize=4, capthick=1.5,
                        markersize=8, markeredgewidth=1.5, markeredgecolor="white",
                        elinewidth=2, alpha=0.85, zorder=3, label=st["label"]
                    )
                elif plot_type == "dot":
                    ax.plot(
                        x, md["mean"],
                        marker=st["marker"], linestyle="",
                        color=st["color"], markersize=10,
                        markeredgewidth=1.5, markeredgecolor="white",
                        alpha=0.85, zorder=3, label=st["label"]
                    )
                else:  # line
                    ax.plot(
                        x, md["mean"],
                        marker=st["marker"], linestyle="",
                        color=st["color"], markersize=8,
                        markeredgewidth=1.5, markeredgecolor="white",
                        alpha=0.85, zorder=3, label=st["label"]
                    )

            if plot_type == "errorbar":
                ax.axhline(0, color="#2C3E50", linestyle="--", alpha=0.4, linewidth=1.5, zorder=1)
            if metric_name.lower() == "coverage":
                ax.axhline(0.95, color="#E74C3C", linestyle=":", alpha=0.6, linewidth=2, zorder=1)

            ax.set_title(f"ce={ce}, cy={cy}, i={i}", fontsize=9, fontweight="medium", pad=6)
            ax.set_xlabel("Outcome (Effect Strength)", fontsize=11, fontweight="medium")
            ax.set_ylabel(y_label, fontsize=11, fontweight="medium")
            ax.set_xticks(range(len(outcomes)))
            ax.set_xticklabels(outcomes, rotation=45, ha="right", fontsize=9)
            _apply_axis_polish(ax)

            # description box
            if description and (desc_on == "each" or (desc_on == "first" and idx == 0)):
                _add_description_box(ax, description)

        # hide unused axes
        for idx in range(n_subplots, len(axes)):
            axes[idx].set_visible(False)

        # single figure-level legend outside plots
        _figure_level_legend(fig, method_styles, methods_present, legend_location)

        # save with batch number if multiple batches
        if len(batches) > 1:
            output_path = Path(output_dir) / f"{metric_name}_method_comparison_part_{batch_idx}.png"
        else:
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

def _add_description_box(ax, description: str):
    """Place a rounded description box inside an axis."""
    at = AnchoredText(
        description,
        loc="upper left",
        prop=dict(size=9),
        frameon=True,
        borderpad=0.6,
        pad=0.4,
    )
    at.patch.set_boxstyle("round,pad=0.5")
    at.patch.set_alpha(0.9)
    at.patch.set_edgecolor("#CCCCCC")
    ax.add_artist(at)


def _apply_axis_polish(ax):
    """Consistent small visual polish on axes."""
    ax.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor("#CCCCCC")
        spine.set_linewidth(1.2)
    ax.set_facecolor("white")


def _figure_level_legend(fig, method_styles, methods_present, location="right"):
    """
    Add a single figure-level legend using proxy artists so it never looks like data.
    location: "right" or "bottom"
    """
    proxies, labels = [], []
    for m in methods_present:
        st = method_styles[m]
        proxies.append(
            Line2D([0], [0], marker=st["marker"], linestyle="",
                   markerfacecolor=st["color"], markeredgecolor="white",
                   markeredgewidth=1.5, markersize=9)
        )
        labels.append(st["label"])

    if not proxies:
        return

    if location == "bottom":
        fig.legend(
            proxies, labels, title="Method",
            loc="lower center", ncol=len(labels),
            frameon=True, edgecolor="#CCCCCC", framealpha=0.95
        )
        fig.subplots_adjust(bottom=0.14, top=0.9)
    else:  # right
        fig.legend(
            proxies, labels, title="Method",
            loc="center left", bbox_to_anchor=(1.0, 0.5),
            frameon=True, edgecolor="#CCCCCC", framealpha=0.95
        )
        # fig.subplots_adjust(right=0.82, top=0.92)