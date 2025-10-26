"""
Prediction accumulator for causal inference models.

This module provides functionality to collect and combine predictions from
multiple folds during the finetune process, creating the same combined
dataframe structure that was previously done in CalibrationProcessor.
"""

import logging
from os.path import join
from typing import Dict, List

import pandas as pd

from corebehrt.constants.causal.data import (
    CF_OUTCOME,
    CF_PROBAS,
    EXPOSURE,
    EXPOSURE_COL,
    OUTCOME_COL,
    PROBAS,
    PS_COL,
    TARGETS,
    PROBAS_ROUND_DIGIT,
)
from corebehrt.constants.data import PID_COL, VAL_KEY
from corebehrt.constants.causal.paths import COMBINED_PREDICTIONS_FILE
from corebehrt.functional.io_operations.causal.predictions import collect_fold_data


class PredictionAccumulator:
    """
    Accumulates and combines predictions from all folds after finetune validation.

    Creates the same combined dataframe structure as CalibrationProcessor but
    operates as an integrated step in the finetune pipeline.
    """

    def __init__(self, finetune_dir: str, outcome_names: List[str]):
        """
        Initialize the accumulator.

        Args:
            finetune_dir: Directory containing fold predictions
            outcome_names: List of outcome names to process
        """
        self.finetune_dir = finetune_dir
        self.outcome_names = outcome_names
        self.logger = logging.getLogger(self.__class__.__name__)

    def accumulate_and_save_predictions(self) -> str:
        """
        Accumulate all predictions from folds and save as combined dataframe.

        Returns:
            Path to the saved combined predictions file
        """
        self.logger.info("Starting prediction accumulation across all folds...")

        # Collect exposure predictions
        df_exposure = self._collect_exposure_predictions()

        # Collect outcome predictions
        outcome_dataframes = {}
        for outcome_name in self.outcome_names:
            df_outcome = self._collect_outcome_predictions(outcome_name)
            outcome_dataframes[outcome_name] = df_outcome

        # Combine all predictions
        combined_df = self._combine_predictions(df_exposure, outcome_dataframes)

        # Save combined predictions
        output_path = join(self.finetune_dir, COMBINED_PREDICTIONS_FILE)
        combined_df = combined_df.round(PROBAS_ROUND_DIGIT)
        combined_df.to_csv(output_path, index=False)

        self.logger.info(f"Combined predictions saved to: {output_path}")
        self.logger.info(f"Combined dataframe shape: {combined_df.shape}")
        self.logger.info(f"Columns: {list(combined_df.columns)}")

        return output_path

    def _collect_exposure_predictions(self) -> pd.DataFrame:
        """Collect exposure predictions from all folds."""
        self.logger.info("Collecting exposure predictions...")

        pids, preds, targets = collect_fold_data(
            self.finetune_dir, EXPOSURE, VAL_KEY, collect_targets=True
        )

        df = pd.DataFrame({PID_COL: pids, PROBAS: preds, TARGETS: targets.astype(int)})

        self.logger.info(f"Exposure predictions: {len(df)} patients")
        return df

    def _collect_outcome_predictions(self, outcome_name: str) -> pd.DataFrame:
        """Collect outcome and counterfactual predictions for a specific outcome."""
        self.logger.info(f"Collecting predictions for outcome: {outcome_name}")

        # Collect factual outcome predictions
        pids, preds, targets = collect_fold_data(
            self.finetune_dir, outcome_name, VAL_KEY, collect_targets=True
        )

        df_outcome = pd.DataFrame(
            {PID_COL: pids, PROBAS: preds, TARGETS: targets.astype(int)}
        )

        # Collect counterfactual outcome predictions
        cf_pids, cf_preds, _ = collect_fold_data(
            self.finetune_dir,
            f"{CF_OUTCOME}_{outcome_name}",
            VAL_KEY,
            collect_targets=False,
        )

        df_cf = pd.DataFrame({PID_COL: cf_pids, CF_PROBAS: cf_preds})

        # Combine factual and counterfactual predictions
        combined = pd.merge(df_outcome, df_cf, on=PID_COL, how="inner", validate="1:1")

        self.logger.info(
            f"Outcome {outcome_name} predictions: {len(combined)} patients"
        )
        return combined

    def _combine_predictions(
        self, exposure_df: pd.DataFrame, outcome_dataframes: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Combine exposure and outcome predictions into a single dataframe.

        Resulting dataframe has columns:
        - PID_COL: Patient ID
        - PS_COL: Propensity score (from exposure predictions)
        - EXPOSURE_COL: Exposure target
        - For each outcome: OUTCOME_COL_{name}, CF_PROBAS_{name}, PROBAS_{name}
        """
        self.logger.info("Combining exposure and outcome predictions...")

        # Rename exposure columns to match expected format
        exposure_df = exposure_df.rename(
            columns={PROBAS: PS_COL, TARGETS: EXPOSURE_COL}
        )

        # Start with exposure dataframe
        combined_df = exposure_df.copy()

        # Merge each outcome dataframe
        for outcome_name, outcome_df in outcome_dataframes.items():
            # Rename outcome columns to include outcome name
            outcome_df = outcome_df.rename(
                columns={
                    TARGETS: f"{OUTCOME_COL}_{outcome_name}",
                    CF_PROBAS: f"{CF_PROBAS}_{outcome_name}",
                    PROBAS: f"{PROBAS}_{outcome_name}",
                }
            )

            # Merge with combined dataframe
            combined_df = pd.merge(
                combined_df, outcome_df, on=PID_COL, how="inner", validate="1:1"
            )

        self.logger.info(f"Final combined predictions: {len(combined_df)} patients")
        return combined_df
