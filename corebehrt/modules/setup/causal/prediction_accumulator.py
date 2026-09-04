"""
Prediction accumulator for causal inference models.

This module provides functionality to collect and combine predictions from
multiple folds during the finetune process, creating the same combined
dataframe structure that was previously done in CalibrationProcessor.
"""

import logging
from os.path import join
from typing import Dict, List, Set

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


def _n_exposed(df: pd.DataFrame, exposure_col: str = EXPOSURE_COL) -> int:
    if exposure_col not in df.columns:
        return -1
    return int((df[exposure_col] == 1).sum())


def _emit(logger: logging.Logger, msg: str, *args, level: int = logging.INFO) -> None:
    """Log and print so Azure console always shows attrition messages."""
    text = msg % args if args else msg
    logger.log(level, text)
    print(f"[PredictionAccumulator] {text}", flush=True)


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
        _emit(self.logger, "Starting prediction accumulation across all folds...")

        # Collect exposure predictions
        df_exposure = self._collect_exposure_predictions()
        n_exp_start = _n_exposed(df_exposure.rename(columns={TARGETS: EXPOSURE_COL}))
        _emit(
            self.logger,
            "Exposure collect: patients=%d, exposed=%d, unique_pids=%d, duplicate_pids=%d",
            len(df_exposure),
            int((df_exposure[TARGETS] == 1).sum()),
            df_exposure[PID_COL].nunique(),
            len(df_exposure) - df_exposure[PID_COL].nunique(),
        )

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

        n_exp_final = _n_exposed(combined_df)
        _emit(self.logger, "Combined predictions saved to: %s", output_path)
        _emit(
            self.logger,
            "ATTRITION SUMMARY: exposure_collect patients=%d exposed=%d → "
            "combined patients=%d exposed=%d (lost patients=%d, lost exposed=%d)",
            len(df_exposure),
            int((df_exposure[TARGETS] == 1).sum())
            if TARGETS in df_exposure.columns
            else n_exp_start,
            len(combined_df),
            n_exp_final,
            len(df_exposure) - len(combined_df),
            (
                int((df_exposure[TARGETS] == 1).sum()) - n_exp_final
                if TARGETS in df_exposure.columns
                else -1
            ),
        )
        _emit(self.logger, "Combined dataframe shape: %s", combined_df.shape)
        _emit(self.logger, "Columns: %s", list(combined_df.columns))

        return output_path

    def _collect_exposure_predictions(self) -> pd.DataFrame:
        """Collect exposure predictions from all folds."""
        _emit(self.logger, "Collecting exposure predictions...")

        pids, preds, targets = collect_fold_data(
            self.finetune_dir, EXPOSURE, VAL_KEY, collect_targets=True
        )

        df = pd.DataFrame({PID_COL: pids, PROBAS: preds, TARGETS: targets.astype(int)})
        n_exposed = int((df[TARGETS] == 1).sum())
        n_unexposed = int((df[TARGETS] == 0).sum())
        n_other = len(df) - n_exposed - n_unexposed
        _emit(
            self.logger,
            "Exposure predictions: patients=%d, exposed=%d, unexposed=%d, other=%d",
            len(df),
            n_exposed,
            n_unexposed,
            n_other,
        )
        return df

    def _collect_outcome_predictions(self, outcome_name: str) -> pd.DataFrame:
        """Collect outcome and counterfactual predictions for a specific outcome."""
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
        before_cf_merge = len(df_outcome)
        combined = pd.merge(df_outcome, df_cf, on=PID_COL, how="inner", validate="1:1")
        cf_drop = before_cf_merge - len(combined)
        if cf_drop:
            missing_cf = set(df_outcome[PID_COL]) - set(df_cf[PID_COL])
            _emit(
                self.logger,
                "Outcome %s: dropped %d patients missing counterfactual predictions "
                "(factual=%d, cf=%d, sample_missing=%s)",
                outcome_name,
                cf_drop,
                before_cf_merge,
                len(df_cf),
                sorted(missing_cf)[:5],
                level=logging.WARNING,
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
        _emit(self.logger, "Combining exposure and outcome predictions...")

        # Rename exposure columns to match expected format
        exposure_df = exposure_df.rename(
            columns={PROBAS: PS_COL, TARGETS: EXPOSURE_COL}
        )

        # Start with exposure dataframe
        combined_df = exposure_df.copy()
        _emit(
            self.logger,
            "Starting combine from exposure predictions: patients=%d, exposed=%d",
            len(combined_df),
            _n_exposed(combined_df),
        )

        total_dropped = 0
        total_exposed_dropped = 0
        drop_events = []

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
            before_merge = len(combined_df)
            before_exposed = _n_exposed(combined_df)
            before_pids = set(combined_df[PID_COL])
            combined_df = pd.merge(
                combined_df, outcome_df, on=PID_COL, how="inner", validate="1:1"
            )
            merge_drop = before_merge - len(combined_df)
            exposed_drop = before_exposed - _n_exposed(combined_df)
            if merge_drop:
                missing = before_pids - set(outcome_df[PID_COL])
                missing_exposed = self._count_exposed_in_pids(
                    exposure_df, missing
                )
                total_dropped += merge_drop
                total_exposed_dropped += exposed_drop
                drop_events.append(
                    (outcome_name, merge_drop, exposed_drop, missing_exposed)
                )
                _emit(
                    self.logger,
                    "Merge with outcome %s dropped %d patients "
                    "(%d exposed by label; before=%d/%d exposed, outcome_rows=%d, "
                    "sample_missing=%s)",
                    outcome_name,
                    merge_drop,
                    exposed_drop,
                    before_merge,
                    before_exposed,
                    len(outcome_df),
                    sorted(missing)[:5],
                    level=logging.WARNING,
                )

        if drop_events:
            _emit(
                self.logger,
                "Merge-drop totals across %d outcomes with drops: "
                "patients_dropped=%d, exposed_dropped=%d; first drops: %s",
                len(drop_events),
                total_dropped,
                total_exposed_dropped,
                drop_events[:10],
                level=logging.WARNING,
            )
        else:
            _emit(
                self.logger,
                "No patients dropped during outcome merges (%d outcomes).",
                len(outcome_dataframes),
            )

        _emit(
            self.logger,
            "Final combined predictions: patients=%d, exposed=%d",
            len(combined_df),
            _n_exposed(combined_df),
        )
        return combined_df

    @staticmethod
    def _count_exposed_in_pids(
        exposure_df: pd.DataFrame, pids: Set
    ) -> int:
        if not pids:
            return 0
        mask = exposure_df[PID_COL].isin(pids)
        return int((exposure_df.loc[mask, EXPOSURE_COL] == 1).sum())
