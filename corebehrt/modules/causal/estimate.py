import logging
import os
from os.path import join

import pandas as pd
import torch
from CausalEstimate import MultiEstimator
from CausalEstimate.estimators import AIPW, IPW, TMLE
from CausalEstimate.filter.propensity import filter_common_support

from corebehrt.constants.causal.data import (
    EXPOSURE_COL,
    OUTCOME,
    PROBAS,
    PROBAS_CONTROL,
    PROBAS_EXPOSED,
    PS_COL,
    EffectColumns,
    EFFECT_ROUND_DIGIT,
)
from corebehrt.constants.causal.paths import (
    COMBINED_CALIBRATED_PREDICTIONS_FILE,
    COUNTERFACTUALS_FILE,
    PATIENTS_FILE,
)
from corebehrt.constants.data import PID_COL
from corebehrt.functional.estimate.benchmarks import (
    append_true_effect,
    append_unadjusted_effect,
)
from corebehrt.functional.estimate.data_handler import (
    get_outcome_names,
    prepare_data_for_outcome,
    prepare_tmle_analysis_df,
    validate_columns,
)
from corebehrt.functional.estimate.report import (
    compute_outcome_stats,
    convert_effect_to_dataframe,
)
from corebehrt.functional.io_operations.estimate import (
    save_all_results,
    save_tmle_analysis,
)
from corebehrt.modules.plot.estimate import (
    AdjustmentPlotConfig,
    ContingencyPlotConfig,
    EffectSizePlotConfig,
)
from corebehrt.functional.visualize.estimate import (
    create_ipw_plot,
    create_adjustment_plot,
    create_annotated_heatmap_matplotlib,
    create_contingency_table_plot,
    create_effect_size_plot,
    create_ps_comparison_plot,
)
from corebehrt.modules.causal.bias import BiasConfig, BiasIntroducer
from corebehrt.modules.setup.config import Config


class EffectEstimator:
    """
    Orchestrates data loading, counterfactual expansion, and causal effect estimation
    for multiple outcomes, with options for filtering and bootstrapping.
    """

    RELEVANT_COLUMNS = EffectColumns.get_columns()

    def __init__(self, cfg: Config, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self.exp_dir: str = self.cfg.paths.estimate

        self.predictions_file: str = join(
            self.cfg.paths.calibrated_predictions,
            COMBINED_CALIBRATED_PREDICTIONS_FILE,
        )
        self.estimator_cfg: dict = self.cfg.estimator
        self.init_estimator_args(self.estimator_cfg)
        self._init_plot_configs()
        self.effect_type: str = self.cfg.estimator.effect_type
        self.df = pd.read_csv(self.predictions_file)
        self.analysis_df = self._get_analysis_cohort(self.df)
        self.outcome_names = get_outcome_names(self.df)
        validate_columns(self.df, self.outcome_names)
        self.counterfactual_outcomes_dir: str = self.cfg.paths.get(
            "counterfactual_outcomes"
        )
        self.counterfactual_df = self._load_counterfactual_outcomes()

        self._init_bias_introducer()

    def _init_plot_configs(self) -> None:
        if self.cfg.get("plot", False):
            self.effect_size_plot_cfg = EffectSizePlotConfig(
                **self.cfg.plot.get("effect_size", {})
            )
            self.contingency_plot_cfg = ContingencyPlotConfig(
                **self.cfg.plot.get("contingency_table", {})
            )
            self.adjustment_plot_cfg = AdjustmentPlotConfig(
                **self.cfg.plot.get("adjustment", {})
            )
        else:
            self.effect_size_plot_cfg = EffectSizePlotConfig()
            self.contingency_plot_cfg = ContingencyPlotConfig()
            self.adjustment_plot_cfg = AdjustmentPlotConfig()

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
            df_for_outcome = prepare_data_for_outcome(self.analysis_df, outcome_name)

            # 2. Build estimators with the correct outcome column
            estimator = self._build_multi_estimator()

            # 3. Estimate effects
            effect_df = self._estimate_effects(df_for_outcome, estimator)

            if self.counterfactual_df is not None:
                effect_df = append_true_effect(
                    df_for_outcome,
                    effect_df,
                    self.counterfactual_df,
                    outcome_name,
                    self.effect_type,
                    self.common_support_threshold,
                )

            effect_df = append_unadjusted_effect(df_for_outcome, effect_df)

            # 5. Compute and collect stats for this outcome
            outcome_stats = compute_outcome_stats(df_for_outcome, outcome_name)
            all_stats.append(outcome_stats)

            # 6. Tag results with the outcome name and collect
            effect_df[OUTCOME] = outcome_name

            current_columns = effect_df.columns.tolist()
            filter_columns = [
                col for col in self.RELEVANT_COLUMNS if col in current_columns
            ]

            effect_df_clean = effect_df[filter_columns]
            effect_df_clean = effect_df_clean.round(EFFECT_ROUND_DIGIT)
            initial_estimates.append(effect_df)
            all_effects.append(effect_df_clean)

        final_results_df, combined_stats_df, tmle_analysis_df = (
            self._process_and_save_results(all_effects, all_stats, initial_estimates)
        )
        self._visualize_effects(final_results_df, combined_stats_df, tmle_analysis_df)
        self.logger.info("Effect estimation complete for all outcomes.")

    def _estimate_effects(
        self, df: pd.DataFrame, estimator: MultiEstimator
    ) -> pd.DataFrame:
        """Estimate effects using the provided estimator instance."""
        effect_dict = estimator.compute_effects(
            df,
            n_bootstraps=self.n_bootstrap,
            apply_common_support=False,  # we already filtered for common support
            common_support_threshold=None,
            return_bootstrap_samples=False,
        )
        return convert_effect_to_dataframe(effect_dict)

    def _visualize_effects(
        self,
        final_results_df: pd.DataFrame,
        combined_stats_df: pd.DataFrame,
        tmle_analysis_df: pd.DataFrame,
    ):
        fig_dir = join(self.exp_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)

        create_ps_comparison_plot(
            self.df, self.analysis_df, PS_COL, EXPOSURE_COL, fig_dir
        )
        create_ipw_plot(
            self.analysis_df[EXPOSURE_COL], self.analysis_df[PS_COL], fig_dir
        )
        methods = self.estimator_cfg.methods
        create_annotated_heatmap_matplotlib(
            final_results_df,
            methods + ["RD"],
            EffectColumns.effect,
            join(fig_dir, "effects.png"),
        )
        create_contingency_table_plot(
            combined_stats_df,
            join(fig_dir, "contingency_table"),
            self.contingency_plot_cfg,
            "Patient Counts by Treatment Status and Outcome",
        )
        create_effect_size_plot(
            effects_df=final_results_df,
            save_dir=join(fig_dir, "effects_scatter"),
            title=f"Effect estimates by outcome and method",
            methods=methods + ["RD"],
            config=self.effect_size_plot_cfg,
        )

        if tmle_analysis_df is not None:
            self.logger.info("Generating TMLE adjustment visualizations...")
            create_adjustment_plot(
                data_df=tmle_analysis_df,
                save_dir=join(fig_dir, "adjustment_analysis"),
                config=self.adjustment_plot_cfg,
                title=f"Adjustment Analysis",
            )

        if EffectColumns.true_effect in final_results_df.columns:
            # show true effects, only show one of the methods (theyre all the same for true effect)
            create_annotated_heatmap_matplotlib(
                final_results_df,
                methods[:1],
                EffectColumns.true_effect,
                join(fig_dir, "true_effects.png"),
            )

            # show diff
            final_results_df["diff"] = (
                final_results_df[EffectColumns.true_effect]
                - final_results_df[EffectColumns.effect]
            )
            create_annotated_heatmap_matplotlib(
                final_results_df, methods, "diff", join(fig_dir, "diff.png")
            )

    def _process_and_save_results(
        self,
        all_effects: list,
        all_stats: list,
        initial_estimates: list,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Combines results from all outcomes and saves them to disk."""
        final_results_df = pd.concat(all_effects, ignore_index=True)
        combined_stats_df = pd.concat(all_stats, ignore_index=True)
        initial_estimates_df = pd.concat(initial_estimates, ignore_index=True)

        save_all_results(
            self.exp_dir, self.analysis_df, final_results_df, combined_stats_df
        )

        tmle_analysis_df = (
            prepare_tmle_analysis_df(initial_estimates_df)
            if any(m.upper() == "TMLE" for m in self.estimator_cfg.methods)
            else None
        )
        save_tmle_analysis(
            tmle_analysis_df, self.exp_dir
        ) if tmle_analysis_df is not None else None
        return final_results_df, combined_stats_df, tmle_analysis_df

    def _build_multi_estimator(self) -> MultiEstimator:
        """Builds a MultiEstimator for a specific outcome."""
        estimators = []

        for method in self.estimator_cfg.methods:
            method_upper = method.upper()
            if method_upper == "TMLE":
                estimators.append(
                    TMLE(
                        effect_type=self.effect_type,
                        treatment_col=EXPOSURE_COL,
                        outcome_col=OUTCOME,
                        ps_col=PS_COL,
                        probas_col=PROBAS,
                        probas_t1_col=PROBAS_EXPOSED,
                        probas_t0_col=PROBAS_CONTROL,
                        clip_percentile=self.clip_percentile,
                    )
                )
            elif method_upper == "IPW":
                estimators.append(
                    IPW(
                        effect_type=self.effect_type,
                        treatment_col=EXPOSURE_COL,
                        outcome_col=OUTCOME,
                        ps_col=PS_COL,
                        clip_percentile=self.clip_percentile,
                    )
                )
            elif method_upper == "AIPW":
                estimators.append(
                    AIPW(
                        effect_type=self.effect_type,
                        treatment_col=EXPOSURE_COL,
                        outcome_col=OUTCOME,
                        ps_col=PS_COL,
                        probas_t1_col=PROBAS_EXPOSED,
                        probas_t0_col=PROBAS_CONTROL,
                    )
                )
        return MultiEstimator(estimators=estimators, verbose=False)

    def init_estimator_args(self, cfg) -> None:
        """
        Initialize estimation arguments.
        Uses shorter keys for predicted outcomes.
        """
        self.common_support_threshold: float = cfg.get("common_support_threshold", None)
        self.common_support: bool = True if self.common_support_threshold else False
        self.n_bootstrap: int = cfg.get("n_bootstrap", 0)
        self.clip_percentile: float = cfg.get("clip_percentile", 1)

    def _get_analysis_cohort(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the same common support filtering used by CausalEstimate estimators.

        This ensures that true effects and unadjusted effects are computed on the
        same cohort as the other estimators for fair comparison.
        """
        initial_len = len(df)
        if self.common_support:
            df = filter_common_support(
                df,
                ps_col=PS_COL,
                treatment_col=EXPOSURE_COL,
                threshold=self.common_support_threshold,
            )
            self.logger.info(
                f"Analysis cohort after common support filtering: {initial_len} → {len(df)} observations"
            )

        torch.save(df[PID_COL].values, join(self.exp_dir, PATIENTS_FILE))
        return df

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

            df_for_outcome = prepare_data_for_outcome(self.analysis_df, outcome_name)
            estimator = self._build_multi_estimator()

            # Get unbiased cohort and calculate true effect once as a reference

            dummy_effect_df = pd.DataFrame([{"estimator": "placeholder"}])
            true_effect_df = append_true_effect(
                df_for_outcome,
                dummy_effect_df,
                self.counterfactual_df,
                outcome_name,
                self.effect_type,
                self.common_support_threshold,
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
        results_df = results_df.round(EFFECT_ROUND_DIGIT)
        results_df = results_df[
            [col for col in self.RELEVANT_COLUMNS if col in results_df.columns]
        ]
        results_df.to_csv(results_path, index=False)
        self.logger.info(f"Bias simulation results saved to {results_path}")
