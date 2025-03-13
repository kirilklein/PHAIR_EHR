from os.path import join
from typing import Any, Dict

import pandas as pd
from CausalEstimate import MultiEstimator
from CausalEstimate.estimators import AIPW, IPW, TMLE
from CausalEstimate.filter.propensity import filter_common_support
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
    Orchestrates data loading, counterfactual expansion, and causal effect estimation,
    with options for common-support filtering and bootstrap-based inference.
    """

    def __init__(self, cfg: Config, logger: Any):
        self.cfg = cfg
        self.logger = logger
        self._set_paths()
        self.estimator_cfg = self.cfg.get("estimator")
        self.estimation_args = self._get_estimation_args()
        self.estimator = self._build_multi_estimator()

    def _set_paths(self) -> None:
        self.exp_dir = self.cfg.paths.estimate
        self.exposure_pred_dir = self.cfg.paths.exposure_predictions
        self.outcome_pred_dir = self.cfg.paths.outcome_predictions
        self.counterfactual_outcomes_dir = self.cfg.paths.get("counterfactual_outcomes")

    def _get_estimation_args(self) -> Dict:
        # Default method arguments for AIPW and TMLE
        default_method_args = {
            "AIPW": {
                "predicted_outcome_treated_col": PROBAS_EXPOSED,
                "predicted_outcome_control_col": PROBAS_CONTROL,
            },
            "TMLE": {
                "predicted_outcome_col": PROBAS,
                "predicted_outcome_treated_col": PROBAS_EXPOSED,
                "predicted_outcome_control_col": PROBAS_CONTROL,
            },
        }
        # Merge any user-provided overrides
        user_method_args = self.estimator_cfg.get("method_args", {})
        default_method_args.update(user_method_args)

        return {
            "method_args": default_method_args,
            "common_support": bool(
                self.estimator_cfg.get("common_support_threshold", False)
            ),
            "common_support_threshold": self.estimator_cfg.get(
                "common_support_threshold"
            ),
            "n_bootstrap": self.estimator_cfg.get("n_bootstrap", 0),
        }

    def run(self) -> None:
        """Main workflow: prepare data, estimate effects, optionally append true effects, and save results."""
        self.logger.info("Starting effect estimation process")
        df = self._prepare_data()
        effect_df = self._estimate_effects(df)
        effect_df = self._append_true_effect(df, effect_df)
        self._save_results(df, effect_df)
        self.logger.info("Effect estimation complete.")

    def _prepare_data(self) -> pd.DataFrame:
        """Load raw data, save snapshots and summary stats, and expand counterfactual columns."""
        df = self._load_data()
        self._save_experiment_data(df)
        self._save_experiment_stats(df)
        return expand_counterfactuals(
            df, EXPOSURE_COL, CF_PROBAS, PROBAS_CONTROL, PROBAS_EXPOSED
        )

    def _load_data(self) -> pd.DataFrame:
        """Load exposure and outcome predictions, standardize column names, and merge on PID."""
        exposure_preds = pd.read_csv(
            join(self.exposure_pred_dir, CALIBRATED_PREDICTIONS_FILE)
        )
        exposure_preds.rename(
            columns={TARGETS: EXPOSURE_COL, PROBAS: PS_COL}, inplace=True
        )
        outcome_preds = pd.read_csv(
            join(self.outcome_pred_dir, CALIBRATED_PREDICTIONS_FILE)
        )
        df = pd.merge(exposure_preds, outcome_preds, on=PID_COL)
        self.logger.info("Data loaded successfully")
        return df

    def _estimate_effects(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate the causal effects using the configured MultiEstimator.
        Bootstrapping is triggered if n_bootstrap > 1.
        """
        effect_dict = self.estimator.compute_effects(
            df,
            n_bootstraps=self.estimation_args["n_bootstrap"],
            apply_common_support=self.estimation_args["common_support"],
            common_support_threshold=self.estimation_args["common_support_threshold"],
            return_bootstrap_samples=False,
        )
        return self._convert_effect_to_dataframe(effect_dict)

    def _append_true_effect(
        self, df: pd.DataFrame, effect_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        If counterfactual outcomes are available, compute and append the 'true effect'
        as derived from counterfactual data.
        """
        if self.counterfactual_outcomes_dir:
            cf_file = join(self.counterfactual_outcomes_dir, SIMULATION_RESULTS_FILE)
            cf_outcomes = pd.read_csv(cf_file)
            true_effect = self._compute_true_effect_from_counterfactuals(
                df, cf_outcomes
            )
            effect_df[TRUE_EFFECT_COL] = true_effect
        return effect_df

    def _build_multi_estimator(self) -> MultiEstimator:
        """
        Instantiate individual estimators (TMLE, IPW, AIPW) according to the configuration,
        and combine them into a MultiEstimator.
        """
        estimators = []
        method_args = self.estimation_args["method_args"]
        for method in self.estimator_cfg.methods:
            method_upper = method.upper()
            if method_upper == "TMLE":
                estimators.append(
                    TMLE(
                        effect_type=self.estimator_cfg.effect_type,
                        treatment_col=EXPOSURE_COL,
                        outcome_col=TARGETS,
                        ps_col=PS_COL,
                        probas_col=method_args.get("TMLE", {}).get(
                            "predicted_outcome_col", PROBAS
                        ),
                        probas_t1_col=method_args.get("TMLE", {}).get(
                            "predicted_outcome_treated_col", PROBAS_EXPOSED
                        ),
                        probas_t0_col=method_args.get("TMLE", {}).get(
                            "predicted_outcome_control_col", PROBAS_CONTROL
                        ),
                    )
                )
            elif method_upper == "IPW":
                estimators.append(
                    IPW(
                        effect_type=self.estimator_cfg.effect_type,
                        treatment_col=EXPOSURE_COL,
                        outcome_col=TARGETS,
                        ps_col=PS_COL,
                    )
                )
            elif method_upper == "AIPW":
                estimators.append(
                    AIPW(
                        effect_type=self.estimator_cfg.effect_type,
                        treatment_col=EXPOSURE_COL,
                        outcome_col=TARGETS,
                        ps_col=PS_COL,
                        probas_t1_col=method_args.get("AIPW", {}).get(
                            "predicted_outcome_treated_col", PROBAS_EXPOSED
                        ),
                        probas_t0_col=method_args.get("AIPW", {}).get(
                            "predicted_outcome_control_col", PROBAS_CONTROL
                        ),
                    )
                )
        return MultiEstimator(estimators=estimators, verbose=False)

    def _compute_true_effect_from_counterfactuals(
        self, df: pd.DataFrame, cf_outcomes: pd.DataFrame
    ) -> pd.Series:
        """Compute the true effect from counterfactual outcomes, applying common-support filtering if needed."""
        if self.estimation_args["common_support"]:
            cf_outcomes = pd.merge(
                cf_outcomes, df[[PID_COL, PS_COL]], on=PID_COL, validate="1:1"
            )
            cf_outcomes = filter_common_support(
                cf_outcomes,
                ps_col=PS_COL,
                treatment_col=EXPOSURE_COL,
                threshold=self.estimation_args["common_support_threshold"],
            )
        return compute_effect_from_counterfactuals(
            cf_outcomes, self.estimator_cfg.effect_type
        )

    def _save_results(self, df: pd.DataFrame, effect_df: pd.DataFrame) -> None:
        """Persist the experiment data, stats, and final estimation results."""
        self._save_experiment_data(df)
        self._save_experiment_stats(df)
        self._save_estimate_results(effect_df)

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
        """Convert the MultiEstimator's output dictionary to a DataFrame (one row per method)."""
        return (
            pd.DataFrame.from_dict(effect, orient="index")
            .reset_index()
            .rename(columns={"index": "method"})
        )
