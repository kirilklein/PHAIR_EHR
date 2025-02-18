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
    def __init__(self, cfg: Config, logger: Any):
        self.cfg = cfg
        self.logger = logger
        self.exp_dir = self.cfg.paths.estimate
        self.exposure_pred_dir = self.cfg.paths.exposure_predictions
        self.outcome_pred_dir = self.cfg.paths.outcome_predictions
        self.counterfactual_outcomes_dir = self.cfg.paths.get("counterfactual_outcomes")
        self.estimator_cfg = self.cfg.get("estimator")
        self.estimation_args = self._initialize_estimation_args()

    def _initialize_estimation_args(self) -> Dict:
        """
        Initialize and store common estimation arguments used in both the primary and counterfactual
        effect estimation, including method_args, common support parameters, and bootstrap settings.
        """
        method_args = {
            method: {
                "predicted_outcome_treated_col": PROBAS_EXPOSED,
                "predicted_outcome_control_col": PROBAS_CONTROL,
                "predicted_outcome_col": PROBAS,
            }
            for method in ["AIPW", "TMLE"]
        }
        if self.estimator_cfg.get("method_args"):
            method_args.update(self.estimator_cfg.method_args)

        common_support = bool(self.estimator_cfg.get("common_support_threshold", False))
        common_support_threshold = self.estimator_cfg.get("common_support_threshold")
        n_bootstrap = self.estimator_cfg.get("n_bootstrap", 0)

        return {
            "method_args": method_args,
            "common_support": common_support,
            "common_support_threshold": common_support_threshold,
            "n_bootstrap": n_bootstrap,
        }

    def run(self) -> None:
        self.logger.info("Loading data")
        df = self._load_data()

        self._save_experiment_data(df)
        self._save_experiment_stats(df)

        # Expand counterfactual values in the data
        df = expand_counterfactuals(
            df, EXPOSURE_COL, CF_PROBAS, PROBAS_CONTROL, PROBAS_EXPOSED
        )

        self.logger.info("Estimating causal effect")
        effect_df = self._compute_causal_effect(df)

        if self.counterfactual_outcomes_dir:
            counterfactual_outcomes = pd.read_csv(
                join(self.counterfactual_outcomes_dir, SIMULATION_RESULTS_FILE)
            )
            effect_df[TRUE_EFFECT_COL] = self._compute_true_effect_from_counterfactuals(
                df, counterfactual_outcomes
            )

        self._save_estimate_results(effect_df)

    def _compute_causal_effect(self, df: pd.DataFrame) -> pd.DataFrame:
        estimator = Estimator(
            methods=self.estimator_cfg.methods,
            effect_type=self.estimator_cfg.effect_type,
        )
        args = self.estimation_args
        bootstrap_flag = bool(args["n_bootstrap"] > 1)
        n_bootstraps = args["n_bootstrap"]

        effect = estimator.compute_effect(
            df,
            treatment_col=EXPOSURE_COL,
            outcome_col=TARGETS,
            ps_col=PS_COL,
            bootstrap=bootstrap_flag,
            n_bootstraps=n_bootstraps,
            method_args=args["method_args"],
            apply_common_support=args["common_support"],
            common_support_threshold=args["common_support_threshold"],
        )

        return self._convert_effect_to_dataframe(effect)

    def _compute_true_effect_from_counterfactuals(
        self, df: pd.DataFrame, counterfactual_outcomes: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Takes the counterfactual outcomes, the original dataframe with PS (for common suport filtering), and computes the true effect
        Returns a column with the true effect for each row.
        """
        if self.estimation_args["common_support"]:
            counterfactual_outcomes = pd.merge(
                counterfactual_outcomes,
                df[[PID_COL, PS_COL]],
                on=PID_COL,
                validate="1:1",  # Enforce one-to-one merge
            )

            counterfactual_outcomes = filter_common_support(
                counterfactual_outcomes,
                ps_col=PS_COL,
                treatment_col=EXPOSURE_COL,
                threshold=self.estimation_args["common_support_threshold"],
            )

        return compute_effect_from_counterfactuals(
            counterfactual_outcomes, self.estimator_cfg.effect_type
        )

    def _load_data(self) -> pd.DataFrame:
        """Load exposure and outcome predictions and merge them."""
        exposure_preds = pd.read_csv(
            join(self.exposure_pred_dir, CALIBRATED_PREDICTIONS_FILE)
        )
        exposure_preds = exposure_preds.rename(
            columns={
                TARGETS: EXPOSURE_COL,
                PROBAS: PS_COL,
            }
        )
        outcome_preds = pd.read_csv(
            join(self.outcome_pred_dir, CALIBRATED_PREDICTIONS_FILE)
        )
        df = pd.merge(exposure_preds, outcome_preds, on=PID_COL)
        self.logger.info("Data loaded successfully")
        return df

    def _save_experiment_data(self, df: pd.DataFrame) -> None:
        filepath = join(self.exp_dir, EXPERIMENT_DATA_FILE)
        df.to_parquet(filepath, index=True)

    def _save_experiment_stats(self, df: pd.DataFrame) -> None:
        stats_table = compute_treatment_outcome_table(df, EXPOSURE_COL, TARGETS)
        filepath = join(self.exp_dir, EXPERIMENT_STATS_FILE)
        stats_table.to_csv(filepath)

    def _save_estimate_results(self, effect_df: pd.DataFrame) -> None:
        filepath = join(self.exp_dir, ESTIMATE_RESULTS_FILE)
        effect_df.to_csv(filepath, index=False)

    @staticmethod
    def _convert_effect_to_dataframe(effect: dict) -> pd.DataFrame:
        return (
            pd.DataFrame.from_dict(effect, orient="index")
            .reset_index()
            .rename(columns={"index": "method"})
        )
