from os.path import join
from typing import Any, Optional, Tuple

import pandas as pd
from CausalEstimate.interface.estimator import Estimator
from CausalEstimate.stats.stats import compute_treatment_outcome_table
from corebehrt.functional.causal.counterfactuals import expand_counterfactuals

from corebehrt.constants.causal import (
    CALIBRATED_PREDICTIONS_FILE,
    EXPERIMENT_DATA_FILE,
    EXPERIMENT_STATS_FILE,
    EXPOSURE_COL,
    PROBAS,
    PS_COL,
    TARGETS,
    ESTIMATE_RESULTS_FILE,
    CF_PROBAS,
    PROBAS_CONTROL_COL,
    PROBAS_EXPOSED_COL,
)
from corebehrt.constants.data import PID_COL
from corebehrt.modules.setup.config import Config


class EffectEstimator:
    def __init__(self, cfg: Config, logger: Any):
        self.cfg = cfg
        self.logger = logger
        self.exp_dir = self.cfg.paths.estimate
        self.exposure_pred_dir = self.cfg.paths.exposure_predictions
        self.outcome_pred_dir = self.cfg.paths.outcome_predictions
        self.estimator_cfg = self.cfg.get("estimator")

    def run(self):
        self.logger.info("Loading data")
        df = self._load_data()

        df.to_parquet(join(self.exp_dir, EXPERIMENT_DATA_FILE), index=True)
        stats_table = compute_treatment_outcome_table(df, EXPOSURE_COL, TARGETS)
        stats_table.to_csv(join(self.exp_dir, EXPERIMENT_STATS_FILE))

        effect, common_support, common_support_threshold = self._compute_causal_effect(
            df
        )

        effect.to_csv(join(self.exp_dir, ESTIMATE_RESULTS_FILE), index=False)

    def _compute_causal_effect(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, bool, Optional[float]]:
        estimator = Estimator(
            methods=self.estimator_cfg.methods,
            effect_type=self.estimator_cfg.effect_type,
        )
        df = expand_counterfactuals(
            df, EXPOSURE_COL, CF_PROBAS, PROBAS_CONTROL_COL, PROBAS_EXPOSED_COL
        )

        # Get args for compute effect
        method_args = {
            method: {
                "predicted_outcome_treated_col": PROBAS_EXPOSED_COL,
                "predicted_outcome_control_col": PROBAS_CONTROL_COL,
                "predicted_outcome_col": PROBAS,
            }
            for method in ["AIPW", "TMLE"]
        }

        if self.estimator_cfg.get("method_args"):
            method_args.update(self.estimator_cfg.method_args)

        common_support = bool(self.estimator_cfg.get("common_support_threshold", False))
        common_support_threshold = self.estimator_cfg.get("common_support_threshold")

        effect = estimator.compute_effect(
            df,
            treatment_col=EXPOSURE_COL,
            outcome_col=TARGETS,
            ps_col=PS_COL,
            bootstrap=bool(self.estimator_cfg.get("n_bootstrap", 0) > 1),
            n_bootstraps=self.estimator_cfg.get("n_bootstrap", 0),
            method_args=method_args,
            apply_common_support=common_support,
            common_support_threshold=common_support_threshold,
        )

        return (
            self._convert_effect_to_dataframe(effect),
            common_support,
            common_support_threshold,
        )

    def _load_data(self) -> pd.DataFrame:
        """Load exposure and outcome predictions and combine them."""
        exposure_preds = pd.read_csv(
            join(self.exposure_pred_dir, CALIBRATED_PREDICTIONS_FILE)
        )
        exposure_preds = exposure_preds.rename(
            columns={
                TARGETS: EXPOSURE_COL,
                PROBAS: PS_COL,
            }
        )

        # Load outcome predictions
        outcome_preds = pd.read_csv(
            join(self.outcome_pred_dir, CALIBRATED_PREDICTIONS_FILE)
        )

        # Combine data
        df = pd.merge(exposure_preds, outcome_preds, on=PID_COL)
        return df

    @staticmethod
    def _convert_effect_to_dataframe(effect: dict) -> pd.DataFrame:
        """
        Convert the effect estimates to a DataFrame.

        Args:
            effect (dict): The effect estimates.

        Returns:
            pd.DataFrame: The effect estimates as a DataFrame.
        """
        return (
            pd.DataFrame.from_dict(effect, orient="index")
            .reset_index()
            .rename(columns={"index": "method"})
        )
