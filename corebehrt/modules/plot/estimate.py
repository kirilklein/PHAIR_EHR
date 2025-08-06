import logging
import os

import matplotlib.pyplot as plt
from typing import List
import numpy as np
import pandas as pd
from dataclasses import dataclass
import matplotlib as mpl
from corebehrt.constants.causal.data import OUTCOME, EffectColumns, STATUS

logger = logging.getLogger(__name__)


@dataclass
class EffectSizePlotConfig:
    max_outcomes_per_figure: int = 10
    max_number_of_figures: int = 10
    plot_individual_effects: bool = False


class EffectSizePlotter:
    """
    Orchestrates the creation of forest plots for effect sizes, handling
    pagination and styling in a structured way.
    """

    def __init__(
        self,
        effects_df: pd.DataFrame,
        save_dir: str,
        config: EffectSizePlotConfig,
        title: str,
        methods: List[str],
    ):
        self.df = effects_df
        self.save_dir = save_dir
        self.config = config
        self.title = title
        self.methods = methods

        self._validate_and_prepare()

    def _validate_and_prepare(self):
        """Handles all initial setup and pre-calculation."""
        os.makedirs(self.save_dir, exist_ok=True)

        # Validate if individual effects can be plotted
        if self.config.plot_individual_effects and (
            EffectColumns.effect_0 not in self.df.columns
            or EffectColumns.effect_1 not in self.df.columns
        ):
            logger.warning(
                "Individual effect columns not found. Disabling 'plot_individual_effects'."
            )
            self.config.plot_individual_effects = False

        # Determine methods and outcomes to plot
        self.all_outcomes = self.df[OUTCOME].unique()

        # Pre-calculate aesthetics
        self.method_colors = {
            method: color
            for i, (method, color) in enumerate(
                zip(self.methods, plt.cm.get_cmap("tab10").colors)
            )
        }
        dodge_vals = (
            np.linspace(-0.2, 0.2, len(self.methods)) if len(self.methods) > 1 else [0]
        )
        self.method_dodge = {
            method: val for method, val in zip(self.methods, dodge_vals)
        }

        # Calculate pagination
        required = (
            len(self.all_outcomes) + self.config.max_outcomes_per_figure - 1
        ) // self.config.max_outcomes_per_figure
        self.num_figures_to_generate = min(required, self.config.max_number_of_figures)
        if required > self.num_figures_to_generate:
            logger.warning(
                f"Plot generation capped at {self.num_figures_to_generate} figures."
            )

    def run(self):
        """Main entry point to generate all plot pages."""
        for page_num in range(self.num_figures_to_generate):
            self._plot_page(page_num)

    def _plot_page(self, page_num: int):
        """Creates and saves a single figure (page) of the plot."""
        start_idx = page_num * self.config.max_outcomes_per_figure
        end_idx = start_idx + self.config.max_outcomes_per_figure
        outcomes_on_this_page = self.all_outcomes[start_idx:end_idx]
        page_df = self.df[self.df[OUTCOME].isin(outcomes_on_this_page)]

        fig, ax = plt.subplots(figsize=(max(10, 2 * len(outcomes_on_this_page)), 6))

        for method in self.methods:
            self._draw_method_data(ax, method, page_df, outcomes_on_this_page)

        self._finalize_figure(fig, ax, page_num, outcomes_on_this_page)

    def _draw_method_data(
        self,
        ax: mpl.axes.Axes,
        method: str,
        page_df: pd.DataFrame,
        outcomes_on_page: List[str],
    ):
        """Extracts data for a single method and draws its points and error bars."""
        method_df = page_df[page_df[EffectColumns.method] == method]
        plot_data = {"x": [], "y": [], "y_err": [], "y0": [], "y1": []}

        for i, outcome in enumerate(outcomes_on_page):
            row = method_df[method_df[OUTCOME] == outcome]
            if row.empty:
                continue

            point = row.iloc[0]
            effect = point[EffectColumns.effect]

            plot_data["x"].append(i + self.method_dodge[method])
            plot_data["y"].append(effect)
            plot_data["y_err"].append(
                [
                    [effect - point[EffectColumns.CI95_lower]],
                    [point[EffectColumns.CI95_upper] - effect],
                ]
            )

            if self.config.plot_individual_effects:
                plot_data["y0"].append(point[EffectColumns.effect_0])
                plot_data["y1"].append(point[EffectColumns.effect_1])

        # Plot main effect estimates
        ax.errorbar(
            x=plot_data["x"],
            y=plot_data["y"],
            yerr=np.array(plot_data["y_err"]).T.reshape(2, -1),
            fmt="o",
            color=self.method_colors[method],
            label=method.upper(),
            capsize=5,
            markersize=8,
        )

        # Plot individual outcome rates
        if self.config.plot_individual_effects and plot_data["y0"]:
            ax.plot(
                plot_data["x"],
                plot_data["y1"],
                "x",
                color=self.method_colors[method],
                markersize=5,
                alpha=0.7,
                label=f"{method.upper()} Y(1)",
            )
            ax.plot(
                plot_data["x"],
                plot_data["y0"],
                "+",
                color=self.method_colors[method],
                markersize=5,
                alpha=0.7,
                label=f"{method.upper()} Y(0)",
            )

    def _finalize_figure(
        self,
        fig: mpl.figure.Figure,
        ax: mpl.axes.Axes,
        page_num: int,
        outcomes_on_page: List[str],
    ):
        """Sets final aesthetics and saves the figure."""
        ax.axhline(0, ls="--", color="grey")
        ax.set_xticks(range(len(outcomes_on_page)))
        ax.set_xticklabels(outcomes_on_page, fontsize=12)
        ax.set_xlabel("Outcome", fontsize=14)
        ax.set_ylabel("Effect size", fontsize=14)
        ax.set_title(
            f"{self.title} (Part {page_num + 1}/{self.num_figures_to_generate})",
            fontsize=16,
        )

        handles, labels = ax.get_legend_handles_labels()
        if self.config.plot_individual_effects:
            ax.legend(
                handles,
                labels,
                title="Estimation type",
                bbox_to_anchor=(1.02, 0.5),
                loc="center left",
            )
            plt.subplots_adjust(right=0.8)  # Adjust layout if legend is outside
        else:
            ax.legend(title="Estimation type", loc="lower right")

        save_path = f"{self.save_dir}/effect_sizes_plot_{page_num + 1}.png"
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Effect size plot saved to {save_path}")


@dataclass
class ContingencyPlotConfig:
    max_outcomes_per_figure: int = (
        5  # Fewer outcomes per plot works better for grouped bars
    )
    max_number_of_figures: int = 10


class ContingencyTablePlotter:
    """
    Orchestrates the creation of stacked bar charts from contingency table data.
    """

    def __init__(
        self,
        data_df: pd.DataFrame,
        save_dir: str,
        config: ContingencyPlotConfig,
        title: str,
    ):
        self.df = data_df[
            data_df[STATUS] != "Total"
        ]  # We only need Treated/Untreated for plotting
        self.save_dir = save_dir
        self.config = config
        self.title = title

        self._prepare()

    def _prepare(self):
        """Pre-calculates necessary attributes for plotting."""
        os.makedirs(self.save_dir, exist_ok=True)
        self.all_outcomes = self.df[OUTCOME].unique()

        # Calculate pagination
        required = (
            len(self.all_outcomes) + self.config.max_outcomes_per_figure - 1
        ) // self.config.max_outcomes_per_figure
        self.num_figures_to_generate = min(required, self.config.max_number_of_figures)
        if required > self.num_figures_to_generate:
            logger.warning(
                f"Plot generation capped at {self.num_figures_to_generate} figures."
            )

    def run(self):
        """Main entry point to generate all plot pages."""
        for page_num in range(self.num_figures_to_generate):
            self._plot_page(page_num)

    def _plot_page(self, page_num: int):
        """Creates and saves a single figure (page) of the plot."""
        start_idx = page_num * self.config.max_outcomes_per_figure
        end_idx = start_idx + self.config.max_outcomes_per_figure
        outcomes_on_page = self.all_outcomes[start_idx:end_idx]
        page_df = self.df[self.df[OUTCOME].isin(outcomes_on_page)]

        fig, ax = plt.subplots(figsize=(max(10, 2.5 * len(outcomes_on_page)), 7))

        self._draw_bars_for_page(ax, page_df, outcomes_on_page)
        self._finalize_figure(fig, ax, page_num, outcomes_on_page)

    def _draw_bars_for_page(
        self, ax: mpl.axes.Axes, page_df: pd.DataFrame, outcomes_on_page: List[str]
    ):
        """Draws the grouped, stacked bars for the outcomes on the current page."""
        x = np.arange(len(outcomes_on_page))  # the label locations
        width = 0.35  # the width of the bars

        # Data for Untreated bars
        untreated_df = (
            page_df[page_df[STATUS] == "Untreated"]
            .set_index(OUTCOME)
            .loc[outcomes_on_page]
        )
        no_outcome_untreated = untreated_df["No Outcome"]
        outcome_untreated = untreated_df["Outcome"]

        # Data for Treated bars
        treated_df = (
            page_df[page_df[STATUS] == "Treated"]
            .set_index(OUTCOME)
            .loc[outcomes_on_page]
        )
        no_outcome_treated = treated_df["No Outcome"]
        outcome_treated = treated_df["Outcome"]

        # Plotting the bars
        ax.bar(
            x - width / 2,
            no_outcome_untreated,
            width,
            label="No Outcome (Untreated)",
            color="lightblue",
        )
        ax.bar(
            x - width / 2,
            outcome_untreated,
            width,
            bottom=no_outcome_untreated,
            label="Outcome (Untreated)",
            color="steelblue",
        )

        ax.bar(
            x + width / 2,
            no_outcome_treated,
            width,
            label="No Outcome (Treated)",
            color="lightcoral",
        )
        ax.bar(
            x + width / 2,
            outcome_treated,
            width,
            bottom=no_outcome_treated,
            label="Outcome (Treated)",
            color="firebrick",
        )

    def _finalize_figure(
        self,
        fig: mpl.figure.Figure,
        ax: mpl.axes.Axes,
        page_num: int,
        outcomes_on_page: List[str],
    ):
        """Sets final aesthetics and saves the figure."""
        ax.set_ylabel("Number of Patients", fontsize=14)
        ax.set_title(
            f"{self.title} (Part {page_num + 1}/{self.num_figures_to_generate})",
            fontsize=16,
        )
        ax.set_xticks(np.arange(len(outcomes_on_page)))
        ax.set_xticklabels(outcomes_on_page, fontsize=12, rotation=15, ha="right")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        save_path = f"{self.save_dir}/contingency_counts_{page_num + 1}.png"
        fig.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Contingency plot saved to {save_path}")


@dataclass
class AdjustmentPlotConfig:
    max_outcomes_per_figure: int = 10
    max_number_of_figures: int = 10


class AdjustmentPlotter:
    """
    Orchestrates the creation of plots to visualize the TMLE adjustment process.
    """

    def __init__(
        self,
        data_df: pd.DataFrame,
        save_dir: str,
        config: AdjustmentPlotConfig,
        title: str,
    ):
        self.df = data_df
        self.save_dir = save_dir
        self.config = config
        self.title = title
        self._prepare()

    def _prepare(self):
        """Pre-calculates necessary attributes for plotting."""
        os.makedirs(self.save_dir, exist_ok=True)
        self.all_outcomes = self.df[OUTCOME].unique()
        self.methods = self.df[EffectColumns.method].unique()

        required = (
            len(self.all_outcomes) + self.config.max_outcomes_per_figure - 1
        ) // self.config.max_outcomes_per_figure
        self.num_figures_to_generate = min(required, self.config.max_number_of_figures)

    def run(self):
        """Main entry point to generate all plot pages."""
        if self.num_figures_to_generate > 0:
            logger.info("Generating adjustment composition and component plots...")
        for page_num in range(self.num_figures_to_generate):
            self._plot_page(page_num)

    def _plot_page(self, page_num: int):
        """Creates and saves all plots for a single page of outcomes."""
        start_idx = page_num * self.config.max_outcomes_per_figure
        end_idx = start_idx + self.config.max_outcomes_per_figure
        outcomes_on_page = self.all_outcomes[start_idx:end_idx]
        page_df = self.df[self.df[OUTCOME].isin(outcomes_on_page)]

        # Create both plots for this page of outcomes
        self._create_composition_plot(page_df, outcomes_on_page, page_num)
        self._create_components_plot(page_df, outcomes_on_page, page_num)

    def _create_composition_plot(self, page_df, outcomes_on_page, page_num):
        """Creates the stacked bar chart showing initial_effect + adjustment."""
        fig, ax = plt.subplots(figsize=(max(10, 2 * len(outcomes_on_page)), 6))
        x = np.arange(len(outcomes_on_page))
        width = 0.35

        # This assumes only one method (e.g., TMLE) in the input df.
        # If multiple methods, this would need dodging similar to other plots.
        method_df = page_df.set_index(OUTCOME).loc[outcomes_on_page]
        initial = method_df["initial_effect"]
        adjustment = method_df["adjustment"]

        ax.bar(x, initial, width, label="Initial Effect", color="lightgray")
        ax.bar(x, adjustment, width, bottom=initial, label="Adjustment", color="salmon")

        # Add a line for the final, adjusted effect
        ax.plot(
            x,
            initial + adjustment,
            color="black",
            marker="_",
            markersize=15,
            linestyle="None",
            label="Final Effect",
        )

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Effect Size (Risk Difference)", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(outcomes_on_page, rotation=15, ha="right")
        ax.set_title(
            f"Adjustment Composition (Part {page_num + 1}/{self.num_figures_to_generate})",
            fontsize=14,
        )
        ax.legend()
        fig.tight_layout()
        save_path = f"{self.save_dir}/adjustment_composition_{page_num + 1}.png"
        plt.savefig(save_path)
        plt.close(fig)
        logger.info(f"Saved adjustment plot to {save_path}")

    def _create_components_plot(self, page_df, outcomes_on_page, page_num):
        """Creates the dot plot showing adjustment_0 and adjustment_1."""
        fig, ax = plt.subplots(figsize=(max(10, 2 * len(outcomes_on_page)), 6))
        x = np.arange(len(outcomes_on_page))

        method_df = page_df.set_index("outcome").loc[outcomes_on_page]
        adj_0 = method_df["adjustment_0"]
        adj_1 = method_df["adjustment_1"]

        ax.plot(
            x - 0.1, adj_1, "o", label="Adjustment for Treated (Y1)", color="firebrick"
        )
        ax.plot(
            x + 0.1,
            adj_0,
            "x",
            label="Adjustment for Control (Y0)",
            color="darkslateblue",
        )

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_ylabel("Adjustment Value", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(outcomes_on_page, rotation=15, ha="right")
        ax.set_title(
            f"Adjustment Components (Part {page_num + 1}/{self.num_figures_to_generate})",
            fontsize=14,
        )
        ax.legend()
        fig.tight_layout()
        save_path = f"{self.save_dir}/adjustment_components_{page_num + 1}.png"
        plt.savefig(save_path)
        plt.close(fig)
        logger.info(f"Saved adjustment components plot to {save_path}")
