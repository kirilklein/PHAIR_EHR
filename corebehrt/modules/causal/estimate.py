from os.path import join
from typing import Any, Dict

import pandas as pd
from CausalEstimate.filter.propensity import filter_common_support
from CausalEstimate.interface.estimator import Estimator
from CausalEstimate.stats.stats import compute_treatment_outcome_table

from corebehrt.constants.causal import (
    CALIBRATED_PREDICTIONS_FILE,
    CF_PROBAS,
    ESTIMATE_RESULTS_FILE,
    EXPERIMENT_DATA_FILE,
    EXPERIMENT_STATS_FILE,
    EXPOSURE_COL,
    PROBAS,
    PROBAS_CONTROL,
    PROBAS_EXPOSED,
    PS_COL,
    SIMULATION_RESULTS_FILE,
    TARGETS,
    TRUE_EFFECT_COL,
)
from corebehrt.constants.data import PID_COL
from corebehrt.functional.causal.counterfactuals import expand_counterfactuals
from corebehrt.functional.causal.effect import compute_effect_from_counterfactuals
from corebehrt.modules.setup.config import Config


class EffectEstimator:
    """
    Orchestrates the loading of data, expansion of counterfactuals, and causal effect estimation,
    optionally applying common-support filtering and bootstrap methods for confidence intervals.
    """

    def __init__(self, cfg: Config, logger: Any):
        self.cfg = cfg
        self.logger = logger

        # Various paths
        self.exp_dir = self.cfg.paths.estimate
        self.exposure_pred_dir = self.cfg.paths.exposure_predictions
        self.outcome_pred_dir = self.cfg.paths.outcome_predictions
        self.counterfactual_outcomes_dir = self.cfg.paths.get("counterfactual_outcomes")

        # Estimator config: which methods, effect type, etc.
        self.estimator_cfg = self.cfg.get("estimator")

        # Parse any arguments for the estimation process
        self.estimation_args = self._initialize_estimation_args()

    def _initialize_estimation_args(self) -> Dict:
        """
        Initialize and store common estimation arguments used in both the primary and
        counterfactual effect estimation, including:
          - method_args
          - common support parameters
          - bootstrap settings
        """
        # Default method args for AIPW and TMLE
        method_args = {
            method: {
                "predicted_outcome_treated_col": PROBAS_EXPOSED,
                "predicted_outcome_control_col": PROBAS_CONTROL,
                "predicted_outcome_col": PROBAS,
            }
            for method in ["AIPW", "TMLE"]
        }
        # Merge user-provided overrides, if any
        if self.estimator_cfg.get("method_args"):
            method_args.update(self.estimator_cfg.method_args)

        common_support_flag = bool(
            self.estimator_cfg.get("common_support_threshold", False)
        )
        common_support_threshold = self.estimator_cfg.get("common_support_threshold")
        n_bootstrap = self.estimator_cfg.get("n_bootstrap", 0)

        return {
            "method_args": method_args,
            "common_support": common_support_flag,
            "common_support_threshold": common_support_threshold,
            "n_bootstrap": n_bootstrap,
        }

    def run(self) -> None:
        """Top-level method to load data, expand counterfactuals, and estimate causal effects."""
        self.logger.info("Loading data")
        df = self._load_data()

        # Persist data snapshots
        self._save_experiment_data(df)
        self._save_experiment_stats(df)

        # Expand counterfactual columns
        df = expand_counterfactuals(
            df, EXPOSURE_COL, CF_PROBAS, PROBAS_CONTROL, PROBAS_EXPOSED
        )

        self.logger.info("Estimating causal effect")
        effect_df = self._compute_causal_effect(df)

        # Optionally compute "true effect" from counterfactuals (if available)
        if self.counterfactual_outcomes_dir:
            counterfactual_outcomes = pd.read_csv(
                join(self.counterfactual_outcomes_dir, SIMULATION_RESULTS_FILE)
            )
            effect_df[TRUE_EFFECT_COL] = self._compute_true_effect_from_counterfactuals(
                df, counterfactual_outcomes
            )

        # Save final results
        self._save_estimate_results(effect_df)

    def _load_data(self) -> pd.DataFrame:
        """
        Load exposure and outcome predictions from disk, rename columns to standardized names,
        and merge on PID_COL.
        """
        exposure_preds = pd.read_csv(
            join(self.exposure_pred_dir, CALIBRATED_PREDICTIONS_FILE)
        )
        exposure_preds = exposure_preds.rename(
            columns={TARGETS: EXPOSURE_COL, PROBAS: PS_COL}
        )

        outcome_preds = pd.read_csv(
            join(self.outcome_pred_dir, CALIBRATED_PREDICTIONS_FILE)
        )

        df = pd.merge(exposure_preds, outcome_preds, on=PID_COL)
        self.logger.info("Data loaded successfully")
        return df

    def _compute_causal_effect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the causal effect on the provided DataFrame, optionally using bootstrap
        and common-support filtering.
        """
        # Build the estimator
        estimator = self._create_estimator()

        # Extract bootstrap flags from config
        bootstrap_flag = bool(self.estimation_args["n_bootstrap"] > 1)
        n_bootstraps = self.estimation_args["n_bootstrap"]

        # The Estimator can apply common support internally if 'apply_common_support' is True
        effect = estimator.compute_effect(
            df,
            treatment_col=EXPOSURE_COL,
            outcome_col=TARGETS,
            ps_col=PS_COL,
            bootstrap=bootstrap_flag,
            n_bootstraps=n_bootstraps,
            method_args=self.estimation_args["method_args"],
            apply_common_support=self.estimation_args["common_support"],
            common_support_threshold=self.estimation_args["common_support_threshold"],
        )

        return self._convert_effect_to_dataframe(effect)

    def _compute_true_effect_from_counterfactuals(
        self, df: pd.DataFrame, counterfactual_outcomes: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Takes the counterfactual outcomes DataFrame plus the original dataframe (for matching
        PID and PS), and returns a single column containing the true effect for each row
        (as computed from counterfactual data).
        """
        # Merge to attach PS scores to the counterfactual rows
        if self.estimation_args["common_support"]:
            counterfactual_outcomes = pd.merge(
                counterfactual_outcomes,
                df[[PID_COL, PS_COL]],
                on=PID_COL,
                validate="1:1",
            )

            # Now filter out rows that fall outside the overlap region
            counterfactual_outcomes = filter_common_support(
                counterfactual_outcomes,
                ps_col=PS_COL,
                treatment_col=EXPOSURE_COL,
                threshold=self.estimation_args["common_support_threshold"],
            )

        return compute_effect_from_counterfactuals(
            counterfactual_outcomes, self.estimator_cfg.effect_type
        )

    def _create_estimator(self) -> Estimator:
        """
        Create an Estimator instance configured with methods/effect_type from self.estimator_cfg.
        """
        return Estimator(
            methods=self.estimator_cfg.methods,
            effect_type=self.estimator_cfg.effect_type,
        )

    def _save_experiment_data(self, df: pd.DataFrame) -> None:
        """Save a Parquet copy of the combined data for later debugging or analysis."""
        filepath = join(self.exp_dir, EXPERIMENT_DATA_FILE)
        df.to_parquet(filepath, index=True)

    def _save_experiment_stats(self, df: pd.DataFrame) -> None:
        """Compute summary stats for the experiment and save them as CSV."""
        stats_table = compute_treatment_outcome_table(df, EXPOSURE_COL, TARGETS)
        filepath = join(self.exp_dir, EXPERIMENT_STATS_FILE)
        stats_table.to_csv(filepath)

    def _save_estimate_results(self, effect_df: pd.DataFrame) -> None:
        """Save the final effect estimates (and optionally true effect) as CSV."""
        filepath = join(self.exp_dir, ESTIMATE_RESULTS_FILE)
        effect_df.to_csv(filepath, index=False)

    @staticmethod
    def _convert_effect_to_dataframe(effect: dict) -> pd.DataFrame:
        """
        Convert the dictionary returned by Estimator to a standard DataFrame
        with one row per method.
        """
        return (
            pd.DataFrame.from_dict(effect, orient="index")
            .reset_index()
            .rename(columns={"index": "method"})
        )
