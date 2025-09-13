from typing import Tuple
from os.path import join

import pandas as pd
import torch

from corebehrt.constants.causal.data import (
    CF_PROBAS,
    EXPOSURE_COL,
    OUTCOME_COL,
    PROBAS,
    PS_COL,
    TARGETS,
    PROBAS_ROUND_DIGIT,
)
from corebehrt.constants.data import PID_COL
from corebehrt.functional.causal.calibration import calibrate_folds
from corebehrt.modules.setup.causal.artifacts import CalibrationArtifacts
from corebehrt.modules.setup.causal.path_manager import CalibrationPathManager
from corebehrt.constants.causal.paths import COMBINED_PREDICTIONS_FILE


class CalibrationProcessor:
    """
    Simplified calibration processor that loads pre-combined predictions
    from finetune, calibrates them, and saves them in the expected format
    for backward compatibility with subsequent pipeline steps.
    """

    def __init__(self, path_manager: CalibrationPathManager, finetune_dir: str):
        self.paths = path_manager
        self.finetune_dir = finetune_dir
        self.folds = torch.load(self.paths.get_folds_path())
        self.outcome_names = torch.load(self.paths.get_outcome_names_path())

    def load_and_save_predictions(self):
        """
        Load pre-combined predictions from finetune and save them in the
        expected individual file format for backward compatibility.
        """
        # Load the pre-combined predictions from finetune
        combined_predictions_path = join(self.finetune_dir, COMBINED_PREDICTIONS_FILE)
        combined_df = pd.read_csv(combined_predictions_path)
        combined_df = combined_df.round(PROBAS_ROUND_DIGIT)

        # Extract and save exposure predictions
        self._extract_and_save_exposure_predictions(combined_df)

        # Extract and save outcome predictions
        self._extract_and_save_outcome_predictions(combined_df)

    def load_and_calibrate_predictions(self) -> CalibrationArtifacts:
        """
        Loads pre-combined predictions, saves them in expected format,
        calibrates them, and returns calibrated artifacts.

        Returns:
            CalibrationArtifacts containing original and calibrated predictions
        """
        # First, save predictions in the expected format
        self.load_and_save_predictions()

        # Load the pre-combined predictions from finetune
        combined_predictions_path = join(self.finetune_dir, COMBINED_PREDICTIONS_FILE)
        combined_df = pd.read_csv(combined_predictions_path)
        combined_df = combined_df.round(PROBAS_ROUND_DIGIT)
        # Calibrate exposure
        df_exp, df_exp_calibrated = self._calibrate_exposure_from_combined(combined_df)

        # Calibrate outcomes
        outcomes, outcomes_calibrated = self._calibrate_outcomes_from_combined(
            combined_df
        )

        # Create and save the final combined calibrated dataframe
        combined_calibrated_df = self._create_combined_calibrated_df(
            df_exp_calibrated, outcomes_calibrated
        )
        combined_calibrated_df = combined_calibrated_df.round(PROBAS_ROUND_DIGIT)
        combined_calibrated_df.to_csv(
            self.paths.get_combined_calibrated_path(), index=False
        )

        return CalibrationArtifacts(
            combined_df=combined_calibrated_df,
            exposure_df=df_exp,
            calibrated_exposure_df=df_exp_calibrated,
            outcomes=outcomes,
            calibrated_outcomes=outcomes_calibrated,
            outcome_names=self.outcome_names,
        )

    def _extract_and_save_exposure_predictions(self, combined_df: pd.DataFrame):
        """Extract exposure predictions and save to expected file location."""
        exposure_cols = [PID_COL, PS_COL, EXPOSURE_COL]
        exposure_df = combined_df[exposure_cols].copy()

        # Rename columns to match expected format
        exposure_df = exposure_df.rename(
            columns={PS_COL: PROBAS, EXPOSURE_COL: TARGETS}
        )

        # Save to expected location
        exposure_path = self.paths.get_predictions_path("exposure")
        exposure_df.to_csv(exposure_path, index=False)

    def _extract_and_save_outcome_predictions(self, combined_df: pd.DataFrame):
        """Extract outcome predictions and save to expected file locations."""
        for outcome_name in self.outcome_names:
            outcome_cols = [
                PID_COL,
                f"{OUTCOME_COL}_{outcome_name}",
                f"{PROBAS}_{outcome_name}",
                f"{CF_PROBAS}_{outcome_name}",
            ]

            # Filter to only include columns that exist
            existing_cols = [col for col in outcome_cols if col in combined_df.columns]
            outcome_df = combined_df[existing_cols].copy()

            # Rename columns to match expected format
            rename_map = {
                f"{OUTCOME_COL}_{outcome_name}": TARGETS,
                f"{PROBAS}_{outcome_name}": PROBAS,
                f"{CF_PROBAS}_{outcome_name}": CF_PROBAS,
            }
            outcome_df = outcome_df.rename(columns=rename_map)

            # Save to expected location
            outcome_path = self.paths.get_predictions_path("outcome", outcome_name)
            outcome_df.to_csv(outcome_path, index=False)

    def _calibrate_exposure_from_combined(
        self, combined_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract exposure data, calibrate it, and save calibrated version."""
        # Extract exposure data
        exposure_cols = [PID_COL, PS_COL, EXPOSURE_COL]
        exposure_df = combined_df[exposure_cols].copy()
        exposure_df = exposure_df.rename(
            columns={PS_COL: PROBAS, EXPOSURE_COL: TARGETS}
        )

        # Calibrate exposure predictions
        calibrated_exposure_df = calibrate_folds(exposure_df, self.folds)

        # Save calibrated exposure predictions
        calibrated_path = self.paths.get_calibrated_predictions_path("exposure")
        calibrated_exposure_df.to_csv(calibrated_path, index=False)

        return exposure_df, calibrated_exposure_df

    def _calibrate_outcomes_from_combined(
        self, combined_df: pd.DataFrame
    ) -> Tuple[dict, dict]:
        """Extract outcome data, calibrate it, and save calibrated versions."""
        outcomes = {}
        calibrated_outcomes = {}

        for outcome_name in self.outcome_names:
            # Extract outcome data
            outcome_cols = [
                PID_COL,
                f"{OUTCOME_COL}_{outcome_name}",
                f"{PROBAS}_{outcome_name}",
                f"{CF_PROBAS}_{outcome_name}",
            ]

            existing_cols = [col for col in outcome_cols if col in combined_df.columns]
            outcome_df = combined_df[existing_cols].copy()

            # Rename to standard format
            rename_map = {
                f"{OUTCOME_COL}_{outcome_name}": TARGETS,
                f"{PROBAS}_{outcome_name}": PROBAS,
                f"{CF_PROBAS}_{outcome_name}": CF_PROBAS,
            }
            outcome_df = outcome_df.rename(columns=rename_map)

            # Calibrate outcome predictions
            calibrated_outcome_df = calibrate_folds(outcome_df, self.folds)

            # Save calibrated outcome predictions
            calibrated_path = self.paths.get_calibrated_predictions_path(
                "outcome", outcome_name
            )
            calibrated_outcome_df.to_csv(calibrated_path, index=False)

            outcomes[outcome_name] = outcome_df
            calibrated_outcomes[outcome_name] = calibrated_outcome_df

        return outcomes, calibrated_outcomes

    def _create_combined_calibrated_df(
        self, calibrated_exposure_df: pd.DataFrame, calibrated_outcomes: dict
    ) -> pd.DataFrame:
        """Create the combined calibrated dataframe from calibrated components."""
        # Start with calibrated exposure (rename back to combined format)
        combined_df = calibrated_exposure_df.rename(
            columns={PROBAS: PS_COL, TARGETS: EXPOSURE_COL}
        )

        # Add calibrated outcomes
        for outcome_name, calibrated_outcome_df in calibrated_outcomes.items():
            # Rename outcome columns back to combined format
            outcome_df = calibrated_outcome_df.rename(
                columns={
                    TARGETS: f"{OUTCOME_COL}_{outcome_name}",
                    PROBAS: f"{PROBAS}_{outcome_name}",
                    CF_PROBAS: f"{CF_PROBAS}_{outcome_name}",
                }
            )

            # Merge with combined dataframe
            combined_df = pd.merge(
                combined_df, outcome_df, on=PID_COL, how="inner", validate="1:1"
            )

        return combined_df

    # Deprecated methods - kept for backward compatibility but redirect to new methods
    def collect_and_save_all_predictions(self):
        """
        DEPRECATED: Redirects to load_and_save_predictions() which uses pre-combined data.
        """
        self.load_and_save_predictions()

    def load_calibrate_and_save_all(self) -> CalibrationArtifacts:
        """
        DEPRECATED: Use load_and_calibrate_predictions() instead.
        Kept for backward compatibility.
        """
        return self.load_and_calibrate_predictions()
