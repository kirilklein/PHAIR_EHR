from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")


def create_plot_from_agg(
    agg_data: pd.DataFrame,
    metric_col: str,
    y_label: str,
    title_prefix: str,
    output_dir: str,
    plot_type: str = "errorbar",
):
    """Generic plotting function for creating faceted 1xN plots."""
    if agg_data.empty:
        return

    # Plot vs Confounding
    all_instrument_levels = sorted(agg_data["i"].unique())
    instrument_levels_to_plot = [
        lvl
        for lvl in all_instrument_levels
        if agg_data[agg_data["i"] == lvl]["avg_confounding"].nunique() > 1
    ]
    if instrument_levels_to_plot:
        print(
            f"Skipping {title_prefix} vs. Confounder plots for i={set(all_instrument_levels) - set(instrument_levels_to_plot)} (only one data point)."
        )
        n_subplots = len(instrument_levels_to_plot)
        fig, axes = plt.subplots(
            1,
            n_subplots,
            figsize=(6 * n_subplots, 5),
            sharey=True,
            constrained_layout=True,
        )
        if n_subplots == 1:
            axes = [axes]
        fig.suptitle(
            f"{title_prefix} vs. Confounding Strength", fontsize=16, fontweight="bold"
        )
        for i, inst_level in enumerate(instrument_levels_to_plot):
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
                    0.95, color="gray", linestyle="--", alpha=0.9, label="95% Target"
                )
            ax.set_title(f"Instrument Strength (i) = {inst_level}")
            ax.set_xlabel("Average Confounding Strength")
            if i == 0:
                ax.set_ylabel(y_label)
            ax.legend()
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        output_path = Path(output_dir) / f"{metric_col}_vs_confounding_plot.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {title_prefix} vs. Confounding plot to: {output_path}")

    # Plot vs Instrument
    all_conf_levels = sorted(agg_data["avg_confounding"].unique())
    conf_levels_to_plot = [
        lvl
        for lvl in all_conf_levels
        if agg_data[agg_data["avg_confounding"] == lvl]["i"].nunique() > 1
    ]
    if conf_levels_to_plot:
        print(
            f"Skipping {title_prefix} vs. Instrument plots for avg_confounding={set(all_conf_levels) - set(conf_levels_to_plot)} (only one data point)."
        )
        n_subplots = len(conf_levels_to_plot)
        fig, axes = plt.subplots(
            1,
            n_subplots,
            figsize=(6 * n_subplots, 5),
            sharey=True,
            constrained_layout=True,
        )
        if n_subplots == 1:
            axes = [axes]
        fig.suptitle(
            f"{title_prefix} vs. Instrument Strength", fontsize=16, fontweight="bold"
        )
        for i, conf_level in enumerate(conf_levels_to_plot):
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
                    0.95, color="gray", linestyle="--", alpha=0.9, label="95% Target"
                )
            ax.set_title(f"Confounding Strength = {conf_level:.2f}")
            ax.set_xlabel("Instrument Strength (i)")
            if i == 0:
                ax.set_ylabel(y_label)
            ax.legend()
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        output_path = Path(output_dir) / f"{metric_col}_vs_instrument_plot.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {title_prefix} vs. Instrument plot to: {output_path}")
