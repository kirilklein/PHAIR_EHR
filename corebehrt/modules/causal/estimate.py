from os.path import join
from typing import Any, Dict

import pandas as pd
import numpy as np
import torch
from CausalEstimate import MultiEstimator
from CausalEstimate.estimators import AIPW, IPW, TMLE
from CausalEstimate.filter.propensity import filter_common_support
from CausalEstimate.stats.stats import compute_treatment_outcome_table
from corebehrt.constants.causal.paths import PATIENTS_FILE

from corebehrt.constants.causal.data import (
    CF_PROBAS,
    EXPOSURE_COL,
    PROB_C_KEY,
    PROB_KEY,
    PROB_T_KEY,
    PROBAS,
    PROBAS_CONTROL,
    PROBAS_EXPOSED,
    PS_COL,
    TARGETS,
    TRUE_EFFECT_COL,
)
from corebehrt.constants.causal.paths import (
    CALIBRATED_PREDICTIONS_FILE,
    ESTIMATE_RESULTS_FILE,
    EXPERIMENT_DATA_FILE,
    EXPERIMENT_STATS_FILE,
    SIMULATION_RESULTS_FILE,
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
        """
        Initialize estimation arguments.
        Uses shorter keys for predicted outcomes.
        """
        default_method_args = {
            "AIPW": {
                PROB_T_KEY: PROBAS_EXPOSED,
                PROB_C_KEY: PROBAS_CONTROL,
            },
            "TMLE": {
                PROB_KEY: PROBAS,
                PROB_T_KEY: PROBAS_EXPOSED,
                PROB_C_KEY: PROBAS_CONTROL,
            },
        }
        # Merge user-provided overrides, if any.
        user_args = self.estimator_cfg.get("method_args", {})
        default_method_args.update(user_args)

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
        """Main pipeline: prepare data, estimate effects, add benchmarks, and save results."""
        self.logger.info("Starting effect estimation process")
        df = self._prepare_data()

        # Estimate effects (CausalEstimate applies common support filtering internally)
        effect_df = self._estimate_effects(df)

        # Apply the same common support filtering to get analysis cohort
        analysis_df = self._get_analysis_cohort(df)

        # Use filtered cohort for benchmarks to ensure consistency
        effect_df = self._append_true_effect(analysis_df, effect_df)
        effect_df = self._append_unadjusted_effect(analysis_df, effect_df)

        self._save_results(df, effect_df)
        self.logger.info("Effect estimation complete.")

    def _prepare_data(self) -> pd.DataFrame:
        df = self._load_data()
        self._save_experiment_data(df)
        self._save_experiment_stats(df)
        return expand_counterfactuals(
            df, EXPOSURE_COL, PROBAS, CF_PROBAS, PROBAS_CONTROL, PROBAS_EXPOSED
        )

    def _load_data(self) -> pd.DataFrame:
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

    def _get_analysis_cohort(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the same common support filtering used by CausalEstimate estimators.

        This ensures that true effects and unadjusted effects are computed on the
        same cohort as the other estimators for fair comparison.
        """
        initial_len = len(df)
        if self.estimation_args["common_support"]:
            df = filter_common_support(
                df,
                ps_col=PS_COL,
                treatment_col=EXPOSURE_COL,
                threshold=self.estimation_args["common_support_threshold"],
            )
            self.logger.info(
                f"Analysis cohort after common support filtering: {initial_len} → {len(df)} observations"
            )

        torch.save(df[PID_COL].values, join(self.exp_dir, PATIENTS_FILE))
        return df

    def _estimate_effects(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estimate effects using CausalEstimate (applies common support filtering internally)."""
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
        Add ground truth effect estimates from simulated counterfactual outcomes.

        Uses the same analysis cohort as other estimators for consistency.
        Only available when counterfactual outcomes directory is provided (simulation data).
        """
        if self.counterfactual_outcomes_dir:
            cf_file = join(self.counterfactual_outcomes_dir, SIMULATION_RESULTS_FILE)
            cf_outcomes = pd.read_csv(cf_file)
            true_effect = self._compute_true_effect_from_counterfactuals(
                df, cf_outcomes
            )
            effect_df[TRUE_EFFECT_COL] = true_effect
        return effect_df

    def _append_unadjusted_effect(
        self, df: pd.DataFrame, effect_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Append unadjusted Risk Difference (RD) and Risk Ratio (RR) to the effect estimates.

        Uses the same analysis cohort as other estimators for consistency.
        Calculates simple two-sample comparisons between exposed and unexposed groups
        with standard errors and 95% confidence intervals.
        """
        # Split data by exposure status (uses same filtered cohort as other estimators)
        exposed = df[df[EXPOSURE_COL] == 1]
        unexposed = df[df[EXPOSURE_COL] == 0]

        # Calculate basic statistics
        risk_exposed = exposed[TARGETS].mean()
        risk_unexposed = unexposed[TARGETS].mean()
        n_exposed = len(exposed)
        n_unexposed = len(unexposed)

        # Calculate both measures
        rd_row = self._calculate_risk_difference(
            risk_exposed, risk_unexposed, n_exposed, n_unexposed
        )
        rr_row = self._calculate_risk_ratio(
            risk_exposed, risk_unexposed, n_exposed, n_unexposed
        )

        # Append both rows to existing results
        return pd.concat([effect_df, rd_row, rr_row], ignore_index=True)

    def _calculate_risk_difference(
        self,
        risk_exposed: float,
        risk_unexposed: float,
        n_exposed: int,
        n_unexposed: int,
    ) -> pd.DataFrame:
        """Calculate Risk Difference with 95% CI using delta method."""
        risk_difference = risk_exposed - risk_unexposed

        # Standard error using delta method
        variance_exposed = risk_exposed * (1 - risk_exposed) / n_exposed
        variance_unexposed = risk_unexposed * (1 - risk_unexposed) / n_unexposed
        se_rd = np.sqrt(variance_exposed + variance_unexposed)

        # 95% Confidence Interval
        ci_margin = 1.96 * se_rd
        ci_lower_rd = risk_difference - ci_margin
        ci_upper_rd = risk_difference + ci_margin

        return pd.DataFrame(
            {
                "method": ["RD"],
                "effect": [risk_difference],
                "std_err": [se_rd],
                "CI95_lower": [ci_lower_rd],
                "CI95_upper": [ci_upper_rd],
                "effect_1": [risk_exposed],
                "effect_0": [risk_unexposed],
                "n_bootstraps": [0],
            }
        )

    def _calculate_risk_ratio(
        self,
        risk_exposed: float,
        risk_unexposed: float,
        n_exposed: int,
        n_unexposed: int,
    ) -> pd.DataFrame:
        """Calculate Risk Ratio with 95% CI using log transformation."""
        # Handle division by zero
        if risk_unexposed == 0 or risk_exposed == 0:
            risk_ratio = np.inf if risk_exposed > 0 else np.nan
            se_log_rr = np.nan
            ci_lower_rr = np.nan
            ci_upper_rr = np.nan
        else:
            risk_ratio = risk_exposed / risk_unexposed

            # Standard error on log scale
            se_log_rr = np.sqrt(
                (1 - risk_exposed) / (risk_exposed * n_exposed)
                + (1 - risk_unexposed) / (risk_unexposed * n_unexposed)
            )

            # 95% CI on log scale, then exponentiate
            log_rr = np.log(risk_ratio)
            ci_margin_log = 1.96 * se_log_rr
            ci_lower_rr = np.exp(log_rr - ci_margin_log)
            ci_upper_rr = np.exp(log_rr + ci_margin_log)

        return pd.DataFrame(
            {
                "method": ["RR"],
                "effect": [risk_ratio],
                "std_err": [se_log_rr],
                "CI95_lower": [ci_lower_rr],
                "CI95_upper": [ci_upper_rr],
                "effect_1": [risk_exposed],
                "effect_0": [risk_unexposed],
                "n_bootstraps": [0],
            }
        )

    def _build_multi_estimator(self) -> MultiEstimator:
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
                        probas_col=method_args.get("TMLE", {}).get(PROB_KEY, PROBAS),
                        probas_t1_col=method_args.get("TMLE", {}).get(
                            PROB_T_KEY, PROBAS_EXPOSED
                        ),
                        probas_t0_col=method_args.get("TMLE", {}).get(
                            PROB_C_KEY, PROBAS_CONTROL
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
                            PROB_T_KEY, PROBAS_EXPOSED
                        ),
                        probas_t0_col=method_args.get("AIPW", {}).get(
                            PROB_C_KEY, PROBAS_CONTROL
                        ),
                    )
                )
        return MultiEstimator(estimators=estimators, verbose=False)

    def _compute_true_effect_from_counterfactuals(
        self, df: pd.DataFrame, cf_outcomes: pd.DataFrame
    ) -> pd.Series:
        """
        Compute ground truth effects from simulated counterfactual outcomes.

        Uses the analysis cohort to ensure consistency with other estimators.
        """
        # Merge with the analysis cohort to get the same subjects
        cf_outcomes = pd.merge(
            cf_outcomes, df[[PID_COL, PS_COL]], on=PID_COL, validate="1:1"
        )
        return compute_effect_from_counterfactuals(
            cf_outcomes, self.estimator_cfg.effect_type
        )

    def _save_results(self, df: pd.DataFrame, effect_df: pd.DataFrame) -> None:
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
        effect_df = effect_df.round(5)
        effect_df.to_csv(filepath, index=False)

    @staticmethod
    def _convert_effect_to_dataframe(effect: dict) -> pd.DataFrame:
        return (
            pd.DataFrame.from_dict(effect, orient="index")
            .reset_index()
            .rename(columns={"index": "method"})
        )
