import os
from os.path import join
from typing import Any, Dict

import pandas as pd
import torch
from CausalEstimate import MultiEstimator
from CausalEstimate.estimators import AIPW, IPW, TMLE
from CausalEstimate.filter.propensity import filter_common_support
from corebehrt.modules.causal.bias import BiasConfig, BiasIntroducer
from corebehrt.constants.causal.data import (
    EXPOSURE_COL,
    OUTCOME,
    PROB_C_KEY,
    PROB_KEY,
    PROB_T_KEY,
    PROBAS,
    PROBAS_CONTROL,
    PROBAS_EXPOSED,
    PS_COL,
    EffectColumns,
)
from corebehrt.constants.causal.paths import (
    COMBINED_CALIBRATED_PREDICTIONS_FILE,
    COUNTERFACTUALS_FILE,
    PATIENTS_FILE,
)
from corebehrt.constants.data import PID_COL
from corebehrt.functional.estimate.data_handler import (
    get_outcome_names,
    prepare_data_for_outcome,
    validate_columns,
    prepare_tmle_analysis_df,
)
from corebehrt.functional.estimate.benchmarks import (
    append_true_effect,
    append_unadjusted_effect,
)
from corebehrt.functional.estimate.report import (
    compute_outcome_stats,
    convert_effect_to_dataframe,
)
from corebehrt.functional.io_operations.estimate import (
    save_all_results,
    save_tmle_analysis,
)
from corebehrt.functional.visualize.estimate import create_annotated_heatmap_matplotlib
from corebehrt.modules.setup.config import Config


class EffectEstimator:
    """
    Orchestrates data loading, counterfactual expansion, and causal effect estimation
    for multiple outcomes, with options for filtering and bootstrapping.
    """

    RELEVANT_COLUMNS = EffectColumns.get_columns()

    def __init__(self, cfg: Config, logger: Any):
        self.cfg = cfg
        self.logger = logger
        self.exp_dir = self.cfg.paths.estimate

        self.predictions_file = join(
            self.cfg.paths.calibrated_predictions,
            COMBINED_CALIBRATED_PREDICTIONS_FILE,
        )
        self.estimator_cfg = self.cfg.estimator
        self.effect_type = self.cfg.estimator.effect_type
        self.df = pd.read_csv(self.predictions_file)
        self.outcome_names = get_outcome_names(self.df)
        validate_columns(self.df, self.outcome_names)
        self.counterfactual_outcomes_dir = self.cfg.paths.get("counterfactual_outcomes")
        self.counterfactual_df = self._load_counterfactual_outcomes()
        self.estimation_args = self._get_estimation_args()
        self._init_bias_introducer()

    def _init_bias_introducer(self) -> BiasIntroducer:
        self.bias_introducer = None
        if (bias_cfg := self.estimator_cfg.get("bias")) is not None:
            bias_config = BiasConfig(
                **bias_cfg,
            )
            self.bias_introducer = BiasIntroducer(bias_config)

    def run(self) -> None:
        if self.bias_introducer is not None:
            print("Run bias simulation")
            self._run_bias_simulation()
        else:
            print("Run standard estimation")
            self.run_standard_estimation()

    def run_standard_estimation(self) -> None:
        """
        Main pipeline: Loops through each outcome to prepare data, estimate effects,
        add benchmarks, and save all results together.
        """
        self.logger.info("Starting effect estimation process for multiple outcomes.")
        all_effects = []
        all_stats = []
        initial_estimates = []
        for outcome_name in self.outcome_names:
            self.logger.info(f"--- Processing outcome: {outcome_name} ---")

            # 1. Prepare data with potential outcomes for the current outcome
            df_for_outcome = prepare_data_for_outcome(self.df, outcome_name)

            # 2. Build estimators with the correct outcome column
            estimator = self._build_multi_estimator()

            # 3. Estimate effects
            effect_df = self._estimate_effects(df_for_outcome, estimator)

            # 4. Get the analysis cohort and add benchmarks
            analysis_df = self._get_analysis_cohort(df_for_outcome)

            if self.counterfactual_df is not None:
                effect_df = append_true_effect(
                    analysis_df,
                    effect_df,
                    self.counterfactual_df,
                    outcome_name,
                    self.effect_type,
                    self.estimation_args["common_support_threshold"],
                )

            effect_df = append_unadjusted_effect(analysis_df, effect_df)

            # 5. Compute and collect stats for this outcome
            outcome_stats = compute_outcome_stats(analysis_df, outcome_name)
            all_stats.append(outcome_stats)

            # 6. Tag results with the outcome name and collect
            effect_df[OUTCOME] = outcome_name

            current_columns = effect_df.columns.tolist()
            filter_columns = [
                col for col in self.RELEVANT_COLUMNS if col in current_columns
            ]

            effect_df_clean = effect_df[filter_columns]
            effect_df_clean = effect_df_clean.round(5)
            initial_estimates.append(effect_df)
            all_effects.append(effect_df_clean)

        self._process_and_save_results(all_effects, all_stats, initial_estimates)
        self._visualize_effects(pd.concat(all_effects, ignore_index=True))
        self.logger.info("Effect estimation complete for all outcomes.")

    def _visualize_effects(self, effects: pd.DataFrame):
        fig_dir = join(self.exp_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        methods = self.estimator_cfg.methods
        create_annotated_heatmap_matplotlib(
            effects,
            methods + ["RD"],
            EffectColumns.effect,
            join(fig_dir, "effects.png"),
        )
        if EffectColumns.true_effect in effects.columns:
            # show true effects, only show one of the methods (theyre all the same for true effect)
            create_annotated_heatmap_matplotlib(
                effects,
                methods[:1],
                EffectColumns.true_effect,
                join(fig_dir, "true_effects.png"),
            )

            # show diff
            effects["diff"] = (
                effects[EffectColumns.true_effect] - effects[EffectColumns.effect]
            )
            create_annotated_heatmap_matplotlib(
                effects, methods, "diff", join(fig_dir, "diff.png")
            )

    def _process_and_save_results(
        self,
        all_effects: list,
        all_stats: list,
        initial_estimates: list,
    ) -> None:
        """Combines results from all outcomes and saves them to disk."""
        final_results_df = pd.concat(all_effects, ignore_index=True)
        combined_stats_df = pd.concat(all_stats, ignore_index=True)
        initial_estimates_df = pd.concat(initial_estimates, ignore_index=True)

        save_all_results(self.exp_dir, self.df, final_results_df, combined_stats_df)

        tmle_analysis_df = prepare_tmle_analysis_df(initial_estimates_df)
        save_tmle_analysis(tmle_analysis_df, self.exp_dir)

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
                        effect_type=self.effect_type,
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
                        effect_type=self.effect_type,
                        treatment_col=EXPOSURE_COL,
                        outcome_col=OUTCOME,
                        ps_col=PS_COL,
                    )
                )
            elif method_upper == "AIPW":
                estimators.append(
                    AIPW(
                        effect_type=self.effect_type,
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
        return convert_effect_to_dataframe(effect_dict)

    def _load_counterfactual_outcomes(self) -> pd.DataFrame:
        """Load combined counterfactual outcomes if available."""
        if not self.counterfactual_outcomes_dir:
            return None

        # Try to load combined counterfactual file first
        combined_cf_file = join(self.counterfactual_outcomes_dir, COUNTERFACTUALS_FILE)
        if os.path.exists(combined_cf_file):
            self.logger.info(
                f"Loading combined counterfactual outcomes from {combined_cf_file}"
            )
            return pd.read_csv(combined_cf_file)

        self.logger.info(
            "No combined counterfactual outcomes file found, will try individual files"
        )
        return None

    def _run_bias_simulation(self) -> None:
        """Runs the effect estimation across a grid of biases for each outcome."""
        self.logger.info("Starting bias simulation.")
        if self.counterfactual_df is None:
            self.logger.error(
                "Bias simulation requires counterfactual outcomes to calculate true effects. Aborting."
            )
            return

        all_biased_effects = []
        bias_grid = self.bias_introducer.get_bias_grid()

        for outcome_name in self.outcome_names:
            self.logger.info(
                f"--- Processing outcome: {outcome_name} for bias simulation ---"
            )

            df_for_outcome = prepare_data_for_outcome(self.df, outcome_name)
            estimator = self._build_multi_estimator()

            # Get unbiased cohort and calculate true effect once as a reference
            analysis_df_unbiased = self._get_analysis_cohort(df_for_outcome.copy())
            dummy_effect_df = pd.DataFrame([{"estimator": "placeholder"}])
            true_effect_df = append_true_effect(
                analysis_df_unbiased,
                dummy_effect_df,
                self.counterfactual_df,
                outcome_name,
                self.effect_type,
                self.estimation_args["common_support_threshold"],
            )
            true_effect_value = true_effect_df[EffectColumns.true_effect].iloc[0]

            for ps_bias, y_bias in bias_grid:
                self.logger.debug(f"Applying bias: ps_bias={ps_bias}, y_bias={y_bias}")
                df_biased = self.bias_introducer.apply_bias(
                    df_for_outcome.copy(), ps_bias, y_bias
                )

                effect_df = self._estimate_effects(df_biased, estimator)

                effect_df[EffectColumns.ps_bias] = ps_bias
                effect_df[EffectColumns.y_bias] = y_bias
                effect_df[OUTCOME] = outcome_name
                effect_df[EffectColumns.true_effect] = true_effect_value

                all_biased_effects.append(effect_df)

        if not all_biased_effects:
            self.logger.warning("Bias simulation finished with no results.")
            return

        final_results_df = pd.concat(all_biased_effects, ignore_index=True)
        self._save_bias_results(final_results_df)
        self.logger.info("Bias simulation complete.")

    def _save_bias_results(self, results_df: pd.DataFrame) -> None:
        """Saves the results of the bias simulation."""
        results_path = join(self.exp_dir, "bias_simulation_results.csv")
        results_df = results_df.round(5)
        results_df = results_df[
            [col for col in self.RELEVANT_COLUMNS if col in results_df.columns]
        ]
        results_df.to_csv(results_path, index=False)
        self.logger.info(f"Bias simulation results saved to {results_path}")
