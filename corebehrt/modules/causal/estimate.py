from os.path import join
from typing import Any, Dict, List

import pandas as pd
import torch
from CausalEstimate import MultiEstimator
from CausalEstimate.estimators import AIPW, IPW, TMLE
from CausalEstimate.filter.propensity import filter_common_support
from CausalEstimate.stats.stats import compute_treatment_outcome_table


from corebehrt.constants.causal.data import (
    CF_PROBAS,
    EXPOSURE_COL,
    OUTCOME,
    PROB_C_KEY,
    PROB_KEY,
    PROB_T_KEY,
    PROBAS,
    PROBAS_CONTROL,
    PROBAS_EXPOSED,
    PS_COL,
    TRUE_EFFECT_COL,
)
from corebehrt.constants.causal.paths import (
    ESTIMATE_RESULTS_FILE,
    EXPERIMENT_DATA_FILE,
    EXPERIMENT_STATS_FILE,
    PATIENTS_FILE,
    SIMULATION_RESULTS_FILE,
    COMBINED_CALIBRATED_PREDICTIONS_FILE,
)
from corebehrt.constants.data import PID_COL
from corebehrt.functional.causal.counterfactuals import expand_counterfactuals
from corebehrt.functional.causal.effect import compute_effect_from_counterfactuals
from corebehrt.functional.causal.estimate import (
    calculate_risk_difference,
    calculate_risk_ratio,
)
from corebehrt.modules.setup.config import Config


class EffectEstimator:
    """
    Orchestrates data loading, counterfactual expansion, and causal effect estimation
    for multiple outcomes, with options for filtering and bootstrapping.
    """

    def __init__(self, cfg: Config, logger: Any):
        self.cfg = cfg
        self.logger = logger
        self.exp_dir = self.cfg.paths.estimate

        self.predictions_file = join(
            self.cfg.paths.calibrated_predictions,
            COMBINED_CALIBRATED_PREDICTIONS_FILE,
        )
        self.estimator_cfg = self.cfg.estimator
        self.df = pd.read_csv(self.predictions_file)
        self.outcome_names = self._get_outcome_names(self.df)
        self._validate_columns(self.df, self.outcome_names)
        self.counterfactual_outcomes_dir = self.cfg.paths.get("counterfactual_outcomes")
        self.estimation_args = self._get_estimation_args()

    def run(self) -> None:
        """
        Main pipeline: Loops through each outcome to prepare data, estimate effects,
        add benchmarks, and save all results together.
        """
        self.logger.info("Starting effect estimation process for multiple outcomes.")
        all_effects = []
        all_stats = []

        for outcome_name in self.outcome_names:
            self.logger.info(f"--- Processing outcome: {outcome_name} ---")

            # 1. Prepare data with potential outcomes for the current outcome
            df_for_outcome = self._prepare_data_for_outcome(self.df, outcome_name)

            # 2. Build estimators with the correct outcome column
            estimator = self._build_multi_estimator()

            # 3. Estimate effects
            effect_df = self._estimate_effects(df_for_outcome, estimator)

            # 4. Get the analysis cohort and add benchmarks
            analysis_df = self._get_analysis_cohort(df_for_outcome)
            # effect_df = self._append_true_effect(analysis_df, effect_df, outcome_name)
            effect_df = self._append_unadjusted_effect(
                analysis_df, effect_df, outcome_name
            )

            # 5. Compute and collect stats for this outcome
            outcome_stats = self._compute_outcome_stats(analysis_df, outcome_name)
            all_stats.append(outcome_stats)

            # 6. Tag results with the outcome name and collect
            effect_df["outcome"] = outcome_name
            all_effects.append(effect_df)

        # Combine and save all results
        final_results_df = pd.concat(all_effects, ignore_index=True)
        combined_stats_df = pd.concat(all_stats, ignore_index=True)

        self._save_results(self.df, final_results_df, combined_stats_df)
        self.logger.info("Effect estimation complete for all outcomes.")

    @staticmethod
    def _get_outcome_names(df: pd.DataFrame) -> List[str]:
        prefix = OUTCOME + "_"
        return [
            col.removeprefix(prefix) for col in df.columns if col.startswith(prefix)
        ]

    @staticmethod
    def _validate_columns(df: pd.DataFrame, outcome_names: List[str]) -> None:
        """"""
        required_columns = [PID_COL, EXPOSURE_COL, PS_COL]
        for name in outcome_names:
            for prefix in [OUTCOME, PROBAS, CF_PROBAS]:
                required_columns.append(prefix + "_" + name)
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

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

    @staticmethod
    def _prepare_data_for_outcome(df: pd.DataFrame, outcome_name: str) -> pd.DataFrame:
        """
        Prepares the dataframe for a single outcome by creating potential outcome columns.
        """
        df = df.copy()
        # Define the specific source columns for this outcome
        probas_col = f"{PROBAS}_{outcome_name}"
        cf_probas_col = f"{CF_PROBAS}_{outcome_name}"
        outcome_col = f"{OUTCOME}_{outcome_name}"

        df.rename(
            columns={
                outcome_col: OUTCOME,
                cf_probas_col: CF_PROBAS,
                probas_col: PROBAS,
            },
            inplace=True,
        )
        df = df[[PID_COL, EXPOSURE_COL, PS_COL, PROBAS, CF_PROBAS, OUTCOME]]

        # Expand counterfactuals to create the generic PROBAS_EXPOSED and PROBAS_CONTROL columns
        # The underlying estimators will use these generic columns.
        return expand_counterfactuals(
            df,
            exposure_col=EXPOSURE_COL,
            factual_outcome_col=PROBAS,
            cf_outcome_col=CF_PROBAS,
            outcome_control_col=PROBAS_CONTROL,
            outcome_exposed_col=PROBAS_EXPOSED,
        )

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.predictions_file)
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

    def _estimate_effects(
        self, df: pd.DataFrame, estimator: MultiEstimator
    ) -> pd.DataFrame:
        """Estimate effects using the provided estimator instance."""
        # This method is now simpler as it just runs the given estimator.
        effect_dict = estimator.compute_effects(
            df,
            n_bootstraps=self.estimation_args["n_bootstrap"],
            apply_common_support=self.estimation_args["common_support"],
            common_support_threshold=self.estimation_args["common_support_threshold"],
            return_bootstrap_samples=False,
        )
        return self._convert_effect_to_dataframe(effect_dict)

    def _append_true_effect(
        self, df: pd.DataFrame, effect_df: pd.DataFrame, outcome_name: str
    ) -> pd.DataFrame:
        """
        ! This needs to be adjusted, depending on how we will save outcomes
        Add ground truth effect estimates from simulated counterfactual outcomes.

        Uses the same analysis cohort as other estimators for consistency.
        Only available when counterfactual outcomes directory is provided (simulation data).
        """
        if self.counterfactual_outcomes_dir:
            cf_file = join(
                self.counterfactual_outcomes_dir,
                outcome_name + "_" + SIMULATION_RESULTS_FILE,
            )
            cf_outcomes = pd.read_csv(cf_file)
            true_effect = self._compute_true_effect_from_counterfactuals(
                df, cf_outcomes
            )
            effect_df[TRUE_EFFECT_COL + "_" + outcome_name] = true_effect
        return effect_df

    def _append_unadjusted_effect(
        self, df: pd.DataFrame, effect_df: pd.DataFrame, outcome_name: str
    ) -> pd.DataFrame:
        """
        Append unadjusted Risk Difference (RD) and Risk Ratio (RR) to the effect estimates.

        Uses the same analysis cohort as other estimators for consistency.
        Calculates simple two-sample comparisons between exposed and unexposed groups
        with standard errors and 95% confidence intervals.
        """

        exposed = df[df[EXPOSURE_COL] == 1]
        unexposed = df[df[EXPOSURE_COL] == 0]

        # Calculate basic statistics
        risk_exposed = exposed[OUTCOME].mean()
        risk_unexposed = unexposed[OUTCOME].mean()
        n_exposed = len(exposed)
        n_unexposed = len(unexposed)

        # Calculate both measures
        rd_row = calculate_risk_difference(
            risk_exposed, risk_unexposed, n_exposed, n_unexposed
        )
        rr_row = calculate_risk_ratio(
            risk_exposed, risk_unexposed, n_exposed, n_unexposed
        )

        # Append both rows to existing results
        return pd.concat([effect_df, rd_row, rr_row], ignore_index=True)

    def _build_multi_estimator(self) -> MultiEstimator:
        """Builds a MultiEstimator for a specific outcome."""
        estimators = []
        method_args = self.estimation_args["method_args"]
        # Define the specific observed outcome column for this run

        for method in self.estimator_cfg.methods:
            method_upper = method.upper()
            if method_upper == "TMLE":
                estimators.append(
                    TMLE(
                        effect_type=self.cfg.estimator.effect_type,
                        treatment_col=EXPOSURE_COL,
                        outcome_col=OUTCOME,
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
                        effect_type=self.cfg.estimator.effect_type,
                        treatment_col=EXPOSURE_COL,
                        outcome_col=OUTCOME,
                        ps_col=PS_COL,
                    )
                )
            elif method_upper == "AIPW":
                estimators.append(
                    AIPW(
                        effect_type=self.cfg.estimator.effect_type,
                        treatment_col=EXPOSURE_COL,
                        outcome_col=OUTCOME,
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

    def _compute_outcome_stats(
        self, analysis_df: pd.DataFrame, outcome_name: str
    ) -> pd.DataFrame:
        """
        Compute treatment-outcome statistics for a specific outcome.

        Args:
            analysis_df: Analysis cohort dataframe with OUTCOME column
            outcome_name: Name of the outcome being analyzed

        Returns:
            DataFrame with statistics tagged with outcome name
        """
        stats_table = compute_treatment_outcome_table(
            analysis_df, EXPOSURE_COL, OUTCOME
        )
        stats_table = stats_table.reset_index(drop=False)
        stats_table.rename(columns={"index": "status"}, inplace=True)
        stats_table["outcome"] = outcome_name
        return stats_table

    def _save_results(
        self, df: pd.DataFrame, effect_df: pd.DataFrame, stats_df: pd.DataFrame
    ) -> None:
        self._save_experiment_data(df)
        self._save_experiment_stats_combined(stats_df)
        self._save_estimate_results(effect_df)

    def _save_experiment_data(self, df: pd.DataFrame) -> None:
        filepath = join(self.exp_dir, EXPERIMENT_DATA_FILE)
        df.to_parquet(filepath, index=True)

    def _save_experiment_stats_combined(self, stats_df: pd.DataFrame) -> None:
        """Save combined statistics for all outcomes."""
        filepath = join(self.exp_dir, EXPERIMENT_STATS_FILE)
        stats_df.to_csv(filepath, index=False)

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
