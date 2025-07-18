from typing import Tuple

import pandas as pd
import torch

from corebehrt.constants.causal.data import (
    CF_OUTCOME,
    CF_PROBAS,
    EXPOSURE,
    EXPOSURE_COL,
    OUTCOME_COL,
    PROBAS,
    PS_COL,
    TARGETS,
)
from corebehrt.constants.data import PID_COL, VAL_KEY
from corebehrt.functional.causal.calibration import calibrate_folds
from corebehrt.functional.io_operations.causal.predictions import collect_fold_data
from corebehrt.modules.setup.causal.artifacts import CalibrationArtifacts
from corebehrt.modules.setup.causal.path_manager import CalibrationPathManager


class CalibrationProcessor:
    """
    Handles the collection, calibration, and combination of predictions.
    """

    def __init__(self, path_manager: CalibrationPathManager, finetune_dir: str):
        self.paths = path_manager
        self.finetune_dir = finetune_dir
        self.folds = torch.load(self.paths.get_folds_path())
        self.outcome_names = torch.load(self.paths.get_outcome_names_path())

    def collect_and_save_all_predictions(self):
        """Collects and saves both exposure and outcome predictions."""
        # Collect and save exposure predictions
        df_exp = self._collect_single_prediction(
            prediction_type=EXPOSURE, probas_name=PROBAS
        )
        df_exp.to_csv(self.paths.get_predictions_path("exposure"), index=False)

        # Collect and save outcome predictions
        for name in self.outcome_names:
            df_outcome = self._collect_single_prediction(
                prediction_type=name, probas_name=PROBAS
            )
            df_cf_outcome = self._collect_single_prediction(
                prediction_type=f"{CF_OUTCOME}_{name}",
                probas_name=CF_PROBAS,
                collect_targets=False,
            )
            combined = pd.merge(
                df_outcome, df_cf_outcome, on=PID_COL, how="inner", validate="1:1"
            )
            path = self.paths.get_predictions_path("outcome", name)
            combined.to_csv(path, index=False)

    def load_calibrate_and_save_all(self) -> CalibrationArtifacts:
        """Loads, calibrates, and saves all predictions, then returns them."""
        # Calibrate exposure
        df_exp, df_exp_calibrated = self._read_calibrate_write(
            "exposure",
        )

        # Calibrate outcomes
        outcomes, outcomes_calibrated = {}, {}
        for name in self.outcome_names:
            df_outcome, df_outcome_calibrated = self._read_calibrate_write(
                "outcome", outcome_name=name
            )
            outcomes[name] = df_outcome
            outcomes_calibrated[name] = df_outcome_calibrated

        # Combine and save final calibrated data
        combined_df = self._combine_predictions(df_exp_calibrated, outcomes_calibrated)
        combined_df.to_csv(self.paths.get_combined_calibrated_path(), index=False)

        return CalibrationArtifacts(
            combined_df=combined_df,
            exposure_df=df_exp,
            calibrated_exposure_df=df_exp_calibrated,
            outcomes=outcomes,
            calibrated_outcomes=outcomes_calibrated,
            outcome_names=self.outcome_names,
        )

    def _collect_single_prediction(
        self, prediction_type: str, probas_name: str, collect_targets: bool = True
    ) -> pd.DataFrame:
        pids, preds, targets = collect_fold_data(
            self.finetune_dir, prediction_type, VAL_KEY, collect_targets
        )
        df = pd.DataFrame({PID_COL: pids, probas_name: preds})
        if collect_targets:
            df[TARGETS] = targets.astype(int)
        return df

    def _read_calibrate_write(
        self, pred_type: str, outcome_name: str = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        file_path = self.paths.get_predictions_path(pred_type, outcome_name)
        write_path = self.paths.get_calibrated_predictions_path(pred_type, outcome_name)

        df = pd.read_csv(file_path)
        df_calibrated = self._calibrate_folds(df)
        df_calibrated.to_csv(write_path, index=False)
        return df, df_calibrated

    def _calibrate_folds(self, df: pd.DataFrame) -> pd.DataFrame:
        return calibrate_folds(df, self.folds)

    def _combine_predictions(
        self, exposure: pd.DataFrame, outcomes: dict
    ) -> pd.DataFrame:
        """
        Combine exposure and outcome predictions.
        Resulting dataframe has columns: PID_COL, PS_COL, EXPOSURE_COL,
        and for each outcome_name: OUTCOME_COL_outcome_name, CF_PROBAS_outcome_name, PROBAS_outcome_name
        """
        exposure = exposure.rename(columns={PROBAS: PS_COL, TARGETS: EXPOSURE_COL})
        df = exposure
        for outcome_name, outcome in outcomes.items():
            outcome = outcome.rename(
                columns={
                    TARGETS: OUTCOME_COL + "_" + outcome_name,
                    CF_PROBAS: CF_PROBAS + "_" + outcome_name,
                    PROBAS: PROBAS + "_" + outcome_name,
                }
            )
            df = pd.merge(df, outcome, on=PID_COL, how="inner", validate="1:1")
        return df
