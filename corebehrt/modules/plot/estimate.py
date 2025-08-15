import logging
import os

import matplotlib.pyplot as plt
from typing import List
import numpy as np
import pandas as pd
from dataclasses import dataclass
import matplotlib as mpl
from abc import ABC, abstractmethod
from corebehrt.constants.causal.data import (
    OUTCOME,
    EffectColumns,
    STATUS,
    TMLEAnalysisColumns,
)

logger = logging.getLogger(__name__)


@dataclass
class BasePlotConfig:
    max_outcomes_per_figure: int = 10
    max_number_of_figures: int = 10


class BasePlotter(ABC):
    """Abstract base class for creating paginated plots."""

    def __init__(
        self,
        data_df: pd.DataFrame,
        save_dir: str,
        config: BasePlotConfig,
        title: str,
    ):
        self.df = data_df
        self.save_dir = save_dir
        self.config = config
        self.title = title
        self.all_outcomes: np.ndarray = np.array([])
        self.num_figures_to_generate: int = 0
        self._prepare()

    def _prepare(self):
        """Handles all initial setup and pre-calculation."""
        os.makedirs(self.save_dir, exist_ok=True)
        if not self.df.empty:
            self.all_outcomes = self.df[OUTCOME].unique()

        required = (
            (len(self.all_outcomes) + self.config.max_outcomes_per_figure - 1)
            // self.config.max_outcomes_per_figure
            if self.config.max_outcomes_per_figure > 0
            else 0
        )
        self.num_figures_to_generate = min(required, self.config.max_number_of_figures)
        if required > self.num_figures_to_generate:
            logger.warning(
                f"Plot generation capped at {self.num_figures_to_generate} figures."
            )

    def run(self):
        """Main entry point to generate all plot pages."""
        if self.num_figures_to_generate > 0:
            logger.info(f"Generating {self._get_plot_type()} plots...")
        for page_num in range(self.num_figures_to_generate):
            self._plot_page(page_num)

    def _plot_page(self, page_num: int):
        """Creates and saves a single figure (page) of the plot."""
        start_idx = page_num * self.config.max_outcomes_per_figure
        end_idx = start_idx + self.config.max_outcomes_per_figure
        outcomes_on_this_page = self.all_outcomes[start_idx:end_idx]
        page_df = self.df[self.df[OUTCOME].isin(outcomes_on_this_page)]

        fig, ax = plt.subplots(figsize=self._get_figsize(outcomes_on_this_page))

        self._draw_page_content(ax, page_df, outcomes_on_this_page)
        self._finalize_figure(fig, ax, page_num, outcomes_on_this_page)

    def _finalize_figure(
        self,
        fig: mpl.figure.Figure,
        ax: mpl.axes.Axes,
        page_num: int,
        outcomes_on_page: List[str],
    ):
        """Sets final aesthetics and saves the figure."""
        ax.set_title(
            f"{self.title} (Part {page_num + 1}/{self.num_figures_to_generate})",
            fontsize=16,
        )
        self._configure_ax(ax, outcomes_on_page)

        save_path = f"{self.save_dir}/{self._get_save_filename(page_num)}"
        fig.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {self._get_plot_type()} plot to {save_path}")

    @abstractmethod
    def _get_plot_type(self) -> str:
        """Return a string representing the type of plot for logging."""
        pass

    @abstractmethod
    def _get_figsize(self, outcomes_on_page: List[str]) -> tuple:
        """Return the figure size for the plot."""
        pass

    @abstractmethod
    def _get_save_filename(self, page_num: int) -> str:
        """Return the filename for the saved plot."""
        pass

    @abstractmethod
    def _draw_page_content(
        self, ax: mpl.axes.Axes, page_df: pd.DataFrame, outcomes_on_page: List[str]
    ):
        """Draw the main content of the plot for a single page."""
        pass

    @abstractmethod
    def _configure_ax(self, ax: mpl.axes.Axes, outcomes_on_page: List[str]):
        """Configure axis labels, ticks, legends, etc."""
        pass


@dataclass
class EffectSizePlotConfig(BasePlotConfig):
    plot_individual_effects: bool = False


class EffectSizePlotter(BasePlotter):
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
        self.methods = methods
        self.method_colors = {}
        self.method_dodge = {}
        super().__init__(effects_df, save_dir, config, title)

    def _prepare(self):
        """Handles all initial setup and pre-calculation."""
        super()._prepare()
        # Validate if individual effects can be plotted
        if self.config.plot_individual_effects and (
            EffectColumns.effect_0 not in self.df.columns
            or EffectColumns.effect_1 not in self.df.columns
        ):
            logger.warning(
                "Individual effect columns not found. Disabling 'plot_individual_effects'."
            )
            self.config.plot_individual_effects = False

        # Pre-calculate aesthetics
        self.method_colors = {
            method: color
            for _, (method, color) in enumerate(
                zip(self.methods, plt.cm.get_cmap("tab10").colors)
            )
        }
        dodge_vals = (
            np.linspace(-0.2, 0.2, len(self.methods)) if len(self.methods) > 1 else [0]
        )
        self.method_dodge = {
            method: val for method, val in zip(self.methods, dodge_vals)
        }

    def _get_plot_type(self) -> str:
        return "effect size"

    def _get_figsize(self, outcomes_on_page: List[str]) -> tuple:
        return (max(10, 2 * len(outcomes_on_page)), 6)

    def _get_save_filename(self, page_num: int) -> str:
        return f"effect_sizes_plot_{page_num + 1}.png"

    def _draw_page_content(
        self, ax: mpl.axes.Axes, page_df: pd.DataFrame, outcomes_on_page: List[str]
    ):
        for method in self.methods:
            self._draw_method_data(ax, method, page_df, outcomes_on_page)
        self._draw_true_effect(ax, page_df, outcomes_on_page)

    def _configure_ax(self, ax: mpl.axes.Axes, outcomes_on_page: List[str]):
        ax.axhline(0, ls="--", color="grey")
        ax.set_xticks(range(len(outcomes_on_page)))
        ax.set_xticklabels(outcomes_on_page, fontsize=12)
        ax.set_xlabel("Outcome", fontsize=14)
        ax.set_ylabel("Effect size", fontsize=14)
        handles, labels = ax.get_legend_handles_labels()

        # Deduplicate labels for clarity in the legend
        unique_labels = {}
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_labels[label] = handle

        legend_handles = list(unique_labels.values())
        legend_labels = list(unique_labels.keys())

        if self.config.plot_individual_effects:
            ax.legend(
                legend_handles,
                legend_labels,
                title="Estimation type",
                bbox_to_anchor=(1.02, 0.5),
                loc="center left",
            )
            plt.subplots_adjust(right=0.8)  # Adjust layout if legend is outside
        else:
            ax.legend(
                handles=legend_handles,
                labels=legend_labels,
                title="Estimation type",
                loc="best",
            )

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

    def _draw_true_effect(
        self, ax: mpl.axes.Axes, page_df: pd.DataFrame, outcomes_on_page: List[str]
    ):
        if EffectColumns.true_effect not in page_df.columns:
            return
        true_effects = page_df.groupby(OUTCOME)[EffectColumns.true_effect].first()
        x_coords, y_coords = [], []
        for i, outcome in enumerate(outcomes_on_page):
            if outcome in true_effects.index and pd.notna(true_effects[outcome]):
                x_coords.append(i)
                y_coords.append(true_effects[outcome])
        if x_coords:
            ax.plot(
                x_coords,
                y_coords,
                "*",
                markersize=12,
                color="black",
                label="True Effect",
                linestyle="None",
                zorder=5,
            )


@dataclass
class ContingencyPlotConfig(BasePlotConfig):
    max_outcomes_per_figure: int = (
        5  # Fewer outcomes per plot works better for grouped bars
    )


class ContingencyTablePlotter(BasePlotter):
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
        # We only need Treated/Untreated for plotting
        filtered_df = data_df[data_df[STATUS] != "Total"].copy()
        super().__init__(filtered_df, save_dir, config, title)

    def _get_plot_type(self) -> str:
        return "contingency"

    def _get_figsize(self, outcomes_on_page: List[str]) -> tuple:
        return (max(10, 2.5 * len(outcomes_on_page)), 7)

    def _get_save_filename(self, page_num: int) -> str:
        return f"contingency_counts_{page_num + 1}.png"

    def _draw_page_content(
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

    def _configure_ax(self, ax: mpl.axes.Axes, outcomes_on_page: List[str]):
        """Sets final aesthetics and saves the figure."""
        ax.set_ylabel("Number of Patients", fontsize=14)
        ax.set_xticks(np.arange(len(outcomes_on_page)))
        ax.set_xticklabels(outcomes_on_page, fontsize=12, rotation=15, ha="right")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.7)


@dataclass
class AdjustmentPlotConfig(BasePlotConfig):
    pass


class AdjustmentPlotter(BasePlotter):
    """
    Orchestrates the creation of detailed plots to visualize the TMLE adjustment process.
    """

    def _get_plot_type(self) -> str:
        return "detailed adjustment"

    def _get_figsize(self, outcomes_on_page: List[str]) -> tuple:
        return (max(12, 2.5 * len(outcomes_on_page)), 8)

    def _get_save_filename(self, page_num: int) -> str:
        return f"detailed_adjustment_{page_num + 1}.png"

    def _draw_page_content(
        self, ax: mpl.axes.Axes, page_df: pd.DataFrame, outcomes_on_page: List[str]
    ):
        """Draws the arrows, points, and difference bars for the page."""
        dodge = 0.15  # Offset for Y0 and Y1

        for i, outcome in enumerate(outcomes_on_page):
            outcome_df = page_df[page_df[OUTCOME] == outcome]
            if outcome_df.empty:
                logger.warning(f"No data found for outcome: {outcome}")
                continue
            row = outcome_df.iloc[0]
            y0_initial = row[TMLEAnalysisColumns.initial_effect_0]
            y0_adj = row[TMLEAnalysisColumns.adjustment_0]
            y0_final = y0_initial + y0_adj

            y1_initial = row[TMLEAnalysisColumns.initial_effect_1]
            y1_adj = row[TMLEAnalysisColumns.adjustment_1]
            y1_final = y1_initial + y1_adj

            # --- Plot Y0 (Control Group) ---
            ax.arrow(
                x=i - dodge,
                y=y0_initial,
                dx=0,
                dy=y0_adj,
                head_width=0.05,
                head_length=0.01,
                fc="royalblue",
                ec="royalblue",
                length_includes_head=True,
            )
            ax.plot(
                i - dodge,
                y0_initial,
                "o",
                color="royalblue",
                markersize=6,
            )

            # --- Plot Y1 (Treated Group) ---
            ax.arrow(
                x=i + dodge,
                y=y1_initial,
                dx=0,
                dy=y1_adj,
                head_width=0.05,
                head_length=0.01,
                fc="firebrick",
                ec="firebrick",
                length_includes_head=True,
            )
            ax.plot(
                i + dodge,
                y1_initial,
                "o",
                color="firebrick",
                markersize=6,
            )

            # --- Plot Final Risk Difference Bracket ---
            # The vertical line connecting the final points
            ax.vlines(
                i,
                ymin=y0_final,
                ymax=y1_final,
                color="black",
                linestyle="-",
                linewidth=1.5,
            )
            # Caps at the end of the vertical line
            cap_width = 0.1
            ax.hlines(
                y0_final, xmin=i - cap_width / 2, xmax=i + cap_width / 2, color="black"
            )
            ax.hlines(
                y1_final, xmin=i - cap_width / 2, xmax=i + cap_width / 2, color="black"
            )

    def _configure_ax(self, ax: mpl.axes.Axes, outcomes_on_page: List[str]):
        """Sets final aesthetics and saves the figure."""
        import matplotlib.lines as mlines

        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_ylabel("Outcome Probability (Risk)", fontsize=14)
        ax.set_xticks(np.arange(len(outcomes_on_page)))
        ax.set_xticklabels(outcomes_on_page, fontsize=12, rotation=15, ha="right")

        handles = [
            mlines.Line2D(
                [],
                [],
                color="royalblue",
                marker="o",
                linestyle="None",
                label="Control Initial",
            ),
            mlines.Line2D(
                [],
                [],
                color="firebrick",
                marker="o",
                linestyle="None",
                label="Treated Initial",
            ),
            mlines.Line2D(
                [],
                [],
                color="gray",
                marker="^",
                linestyle="None",
                markersize=7,
                label="Adjustment Arrow",
            ),
            mlines.Line2D(
                [], [], color="black", linestyle="-", label="Final Risk Difference"
            ),
        ]
        ax.legend(handles=handles)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
