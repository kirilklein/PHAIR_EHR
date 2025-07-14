import os
from pathlib import Path

import pandas as pd

from corebehrt.constants.causal.data import (
    CF_PROBAS,
    EXPOSURE_COL,
    OUTCOME_COL,
    PROBAS,
    PS_COL,
)
from corebehrt.constants.causal.paths import OUTCOMES_DIR
from corebehrt.main_causal.helper.calibrate_plot import (
    produce_calibration_plots,
    plot_probas_hist,
    plot_cf_probas_diff_vs_certainty_in_exposure,
    plot_cf_diff_vs_probas_by_group,
)
from corebehrt.modules.setup.causal.artifacts import CalibrationArtifacts
from corebehrt.modules.setup.causal.path_manager import CalibrationPathManager


class PlottingManager:
    """
    Manages the generation and saving of all plots for the causal inference pipeline.
    """

    def __init__(self, path_manager: CalibrationPathManager):
        """Initializes the PlottingManager with a path manager instance."""
        self.paths = path_manager

    def generate_all_plots(self, data: CalibrationArtifacts):
        """
        Generates and saves all calibration and distribution plots.
        This is the main entry point for the class.
        """
        self._generate_calibration_plots(data)
        self._generate_distribution_plots(data)

    def _generate_calibration_plots(self, data: CalibrationArtifacts):
        """Generates and saves calibration reliability plots."""
        fig_dir = self.paths.get_figure_dir()

        # Plot calibration for propensity scores (exposure)
        produce_calibration_plots(
            data.calibrated_exposure_df,
            data.exposure_df,
            fig_dir,
            "Propensity Score Calibration",
            "ps",
        )

        # Plot calibration for each outcome
        outcomes_fig_dir = self.paths.get_figure_dir(OUTCOMES_DIR)
        for name in data.outcome_names:
            produce_calibration_plots(
                data.calibrated_outcomes[name],
                data.outcomes[name],
                outcomes_fig_dir,
                "Outcome Probability Calibration",
                name,
            )

    def _generate_distribution_plots(self, data: CalibrationArtifacts):
        """Generates and saves histogram and scatter plots for model outputs."""
        df = data.combined_df

        for name in data.outcome_names:
            # Create outcome-specific directories
            hist_fig_dir = self.paths.get_figure_dir(f"histograms/{name}")
            cf_fig_dir = self.paths.get_figure_dir(f"cf_probas/{name}")

            # Define column names for the current outcome
            outcome_probas_col = f"{PROBAS}_{name}"
            cf_probas_col = f"{CF_PROBAS}_{name}"
            outcome_col = f"{OUTCOME_COL}_{name}"
            df["diff"] = df[cf_probas_col] - df[outcome_probas_col]

            # --- Plot Histograms ---
            self._plot_histogram_group(
                df,
                outcome_probas_col,
                outcome_col,
                "probas",
                "Outcome Probability",
                hist_fig_dir,
            )
            self._plot_histogram_group(
                df,
                cf_probas_col,
                outcome_col,
                "cf_probas",
                "CF Outcome Probability",
                hist_fig_dir,
            )
            self._plot_histogram_group(
                df,
                "diff",
                outcome_col,
                "diff",
                "Counterfactual - Factual",
                hist_fig_dir,
            )
            self._plot_histogram_group(
                df, PS_COL, outcome_col, "ps", "Propensity Score", hist_fig_dir
            )

            # --- Plot Scatter Plots ---
            plot_cf_probas_diff_vs_certainty_in_exposure(df, cf_fig_dir)

            # Plot difference vs. factual probability, grouped by exposure/outcome
            plot_cf_diff_vs_probas_by_group(
                df, cf_fig_dir, EXPOSURE_COL, outcome_probas_col, ("Exposed", "Control")
            )
            plot_cf_diff_vs_probas_by_group(
                df,
                cf_fig_dir,
                outcome_col,
                outcome_probas_col,
                ("Positive", "Negative"),
            )

            # Plot difference vs. propensity score, grouped by exposure/outcome
            plot_cf_diff_vs_probas_by_group(
                df, cf_fig_dir, EXPOSURE_COL, PS_COL, ("Exposed", "Control")
            )
            plot_cf_diff_vs_probas_by_group(
                df, cf_fig_dir, outcome_col, PS_COL, ("Positive", "Negative")
            )

    def _plot_histogram_group(
        self,
        df: pd.DataFrame,
        value_col: str,
        outcome_col: str,
        group_name: str,
        x_label: str,
        base_dir: Path,
    ):
        """Helper to reduce repetition in plotting histograms, now more robust."""
        save_dir = base_dir / group_name  # Use Path operator instead of join
        os.makedirs(save_dir, exist_ok=True)

        # Plot histogram grouped by exposure status
        plot_probas_hist(
            df,
            value_col,
            EXPOSURE_COL,
            ("Control", "Exposed"),
            f"{x_label}: Control vs Exposed",
            x_label,
            f"{group_name}_by_exposure",
            save_dir,
        )

        # Plot histogram grouped by outcome status
        plot_probas_hist(
            df,
            value_col,
            outcome_col,
            ("Negative", "Positive"),
            f"{x_label}: Negative vs Positive",
            x_label,
            f"{group_name}_by_outcome",
            save_dir,
        )
