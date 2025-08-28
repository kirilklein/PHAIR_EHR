import os

import matplotlib.pyplot as plt
import numpy as np

from corebehrt.constants.causal.data import (
    CF_PROBAS,
    EXPOSURE_COL,
    OUTCOME_COL,
    PROBAS,
    PS_COL,
)
from corebehrt.functional.visualize.calibrate import (
    plot_cf_diff_vs_probas_by_group,
    plot_cf_probas_diff_vs_certainty_in_exposure,
    plot_probas_hist,
    plot_weights_hist,
    produce_calibration_plots,
)
from corebehrt.functional.visualize.estimate import create_ipw_plot
from corebehrt.modules.setup.causal.artifacts import CalibrationArtifacts
from corebehrt.modules.setup.causal.path_manager import CalibrationPathManager
from corebehrt.functional.utils.azure_save import save_figure_with_azure_copy
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import List, Tuple


class PlottingManager:
    """
    Manages the generation and saving of all plots for the causal inference pipeline.
    """

    def __init__(
        self,
        path_manager: CalibrationPathManager,
        plot_all_outcomes: bool = False,
        num_outcomes_to_plot: int = 5,
    ):
        """Initializes the PlottingManager with a path manager instance."""
        self.paths = path_manager
        self.plot_all_outcomes = plot_all_outcomes
        self.num_outcomes_to_plot = num_outcomes_to_plot
        self.outcomes_to_plot: list[str] = []

    def generate_all_plots(self, data: CalibrationArtifacts):
        """
        Generates and saves all calibration and distribution plots.
        This is the main entry point for the class.
        """
        if (
            not self.plot_all_outcomes
            and len(data.outcome_names) > self.num_outcomes_to_plot
        ):
            self.outcomes_to_plot = data.outcome_names[: self.num_outcomes_to_plot]  # type: ignore
        else:
            self.outcomes_to_plot = data.outcome_names

        self._generate_calibration_plots(data)
        self._generate_distribution_plots(data)

    def _generate_calibration_plots(self, data: CalibrationArtifacts):
        """Generates and saves calibration reliability plots."""
        fig_dir = self.paths.get_figure_dir()
        cal_fig_dir = fig_dir / "calibration"
        os.makedirs(cal_fig_dir, exist_ok=True)

        # Plot calibration for propensity scores (exposure)
        fig, ax = plt.subplots(figsize=(6, 6))
        produce_calibration_plots(
            data.calibrated_exposure_df,
            data.exposure_df,
            "Propensity Score Calibration",
            ax=ax,
        )
        save_figure_with_azure_copy(
            fig, cal_fig_dir / f"{PS_COL}.png", bbox_inches="tight"
        )
        plt.close(fig)

        if not self.outcomes_to_plot:
            return

        # Plot calibration for each outcome in subplots
        num_outcomes = len(self.outcomes_to_plot)
        cols = min(3, num_outcomes)
        rows = (num_outcomes + cols - 1) // cols
        fig, axes = plt.subplots(
            rows, cols, figsize=(6 * cols, 6 * rows), squeeze=False
        )
        axes: List[Axes] = axes.flatten()

        for i, name in enumerate(self.outcomes_to_plot):
            produce_calibration_plots(
                data.calibrated_outcomes[name], data.outcomes[name], name, ax=axes[i]
            )

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Outcome Probability Calibration", fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_figure_with_azure_copy(
            fig, cal_fig_dir / "outcomes_calibration.png", bbox_inches="tight"
        )
        plt.close(fig)

    def _generate_distribution_plots(self, data: CalibrationArtifacts):
        """Generates and saves histogram and scatter plots for model outputs."""
        df = data.combined_df.copy()
        num_outcomes = len(self.outcomes_to_plot)
        if num_outcomes == 0:
            return

        hist_fig_dir = self.paths.get_figure_dir("histograms")
        cf_fig_dir = self.paths.get_figure_dir("cf_probas")
        os.makedirs(hist_fig_dir, exist_ok=True)
        os.makedirs(cf_fig_dir, exist_ok=True)

        create_ipw_plot(df[EXPOSURE_COL], df[PS_COL], hist_fig_dir)

        # Plot standalone propensity score distribution
        fig, ax = plt.subplots()
        plot_probas_hist(
            df,
            PS_COL,
            EXPOSURE_COL,
            ("Control", "Exposed"),
            "Propensity Score: Control vs Exposed",
            "Propensity Score",
            ax,
        )
        save_figure_with_azure_copy(
            fig, hist_fig_dir / f"{PS_COL}_by_exposure_hist.png", bbox_inches="tight"
        )

        # Pre-calculate diff columns
        for name in self.outcomes_to_plot:
            df[f"diff_{name}"] = df[f"{CF_PROBAS}_{name}"] - df[f"{PROBAS}_{name}"]

        # Avoid inf/NaN when PS hits 0 or 1
        ps = df[PS_COL].clip(1e-6, 1 - 1e-6)
        df["ipw"] = np.where(df[EXPOSURE_COL] == 1, 1.0 / ps, 1.0 / (1.0 - ps))
        # Plot IPW distribution
        fig, ax = plt.subplots()
        plot_weights_hist(
            df,
            "ipw",
            EXPOSURE_COL,
            ("Control", "Exposed"),
            "IPW: Control vs Exposed",
            "IPW",
            ax,
        )
        save_figure_with_azure_copy(
            fig, hist_fig_dir / "ipw_by_exposure_hist.png", bbox_inches="tight"
        )

        # --- Subplot Grid Helpers ---
        cols = min(3, num_outcomes)
        rows = (num_outcomes + cols - 1) // cols

        def create_grid(
            title: str, figsize: Tuple[int, int] = (6, 5)
        ) -> Tuple[Figure, List[Axes]]:
            fig, axes = plt.subplots(
                rows,
                cols,
                figsize=(figsize[0] * cols, figsize[1] * rows),
                squeeze=False,
            )
            fig.suptitle(title, fontsize=16)
            return fig, axes.flatten()

        def save_grid(fig: Figure, axes: List[Axes], filename: str):
            for i in range(num_outcomes, len(axes)):
                axes[i].set_visible(False)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            save_figure_with_azure_copy(fig, filename, bbox_inches="tight")

        # --- Plot Histograms ---
        plot_configs = [
            (
                "Outcome Probability by Exposure",
                PROBAS,
                EXPOSURE_COL,
                ("Control", "Exposed"),
                "Outcome Probability",
                "outcome_probas_by_exposure.png",
            ),
            (
                "Outcome Probability by Outcome",
                PROBAS,
                OUTCOME_COL,
                ("Negative", "Positive"),
                "Outcome Probability",
                "outcome_probas_by_outcome.png",
            ),
            (
                "CF Probability by Exposure",
                CF_PROBAS,
                EXPOSURE_COL,
                ("Control", "Exposed"),
                "CF Probability",
                "cf_probas_by_exposure.png",
            ),
            (
                "CF Probability by Outcome",
                CF_PROBAS,
                OUTCOME_COL,
                ("Negative", "Positive"),
                "CF Probability",
                "cf_probas_by_outcome.png",
            ),
            (
                "Difference (CF - Factual) by Exposure",
                "diff",
                EXPOSURE_COL,
                ("Control", "Exposed"),
                "Difference",
                "diff_by_exposure.png",
            ),
            (
                "Difference (CF - Factual) by Outcome",
                "diff",
                OUTCOME_COL,
                ("Negative", "Positive"),
                "Difference",
                "diff_by_outcome.png",
            ),
            (
                "Propensity Score by Outcome",
                PS_COL,
                OUTCOME_COL,
                ("Negative", "Positive"),
                "Propensity Score",
                "ps_by_outcome.png",
            ),
        ]
        for title, val_prefix, group_prefix, labels, xlabel, fname in plot_configs:
            fig, axes = create_grid(title)
            for i, name in enumerate(self.outcomes_to_plot):
                val_col = (
                    f"{val_prefix}_{name}"
                    if val_prefix not in [PS_COL, "diff"]
                    else val_prefix
                    if val_prefix == PS_COL
                    else f"{val_prefix}_{name}"
                )
                group_col = (
                    f"{group_prefix}_{name}"
                    if group_prefix == OUTCOME_COL
                    else group_prefix
                )
                plot_probas_hist(df, val_col, group_col, labels, name, xlabel, axes[i])
            save_grid(fig, axes, hist_fig_dir / fname)

        # --- Plot Scatter Plots ---
        fig, axes = create_grid("CF-Factual Diff vs. Certainty", figsize=(8, 6))
        for i, name in enumerate(self.outcomes_to_plot):
            plot_cf_probas_diff_vs_certainty_in_exposure(df, f"diff_{name}", axes[i])
            axes[i].set_title(name)
        save_grid(fig, axes, cf_fig_dir / "diff_vs_certainty.png")

        fig, axes = create_grid("CF-Factual Diff vs. Factual Probas", figsize=(8, 6))
        for i, name in enumerate(self.outcomes_to_plot):
            plot_cf_diff_vs_probas_by_group(
                df,
                EXPOSURE_COL,
                f"{PROBAS}_{name}",
                ("Exposed", "Control"),
                f"diff_{name}",
                axes[i],
            )
            axes[i].set_title(name)
        save_grid(fig, axes, cf_fig_dir / "diff_vs_probas_by_exposure.png")

        fig, axes = create_grid("CF-Factual Diff vs. Propensity Score", figsize=(8, 6))
        for i, name in enumerate(self.outcomes_to_plot):
            plot_cf_diff_vs_probas_by_group(
                df,
                EXPOSURE_COL,
                PS_COL,
                ("Exposed", "Control"),
                f"diff_{name}",
                axes[i],
            )
            axes[i].set_title(name)
        save_grid(fig, axes, cf_fig_dir / "diff_vs_ps_by_exposure.png")
