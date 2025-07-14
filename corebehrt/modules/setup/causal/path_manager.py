import os
from pathlib import Path
from corebehrt.constants.causal.paths import (
    PREDICTIONS_DIR_EXPOSURE,
    PREDICTIONS_DIR_OUTCOME,
    PREDICTIONS_FILE,
    CALIBRATED_PREDICTIONS_FILE,
)
from corebehrt.constants.paths import FOLDS_FILE, OUTCOME_NAMES_FILE


class CalibrationPathManager:
    """
    Manages all file and directory paths for the prediction pipeline.

    This class centralizes path construction, making the code cleaner
    and easier to maintain.
    """

    def __init__(self, finetune_dir: str, write_dir: str):
        self.finetune_dir = Path(finetune_dir)
        self.write_dir = Path(write_dir)
        self.figures_dir = self.write_dir / "figures"

    def get_folds_path(self) -> Path:
        return self.finetune_dir / FOLDS_FILE

    def get_outcome_names_path(self) -> Path:
        return self.finetune_dir / OUTCOME_NAMES_FILE

    def get_predictions_path(self, pred_type: str, outcome_name: str = None) -> Path:
        if pred_type == "exposure":
            path = self.write_dir / PREDICTIONS_DIR_EXPOSURE / PREDICTIONS_FILE
            os.makedirs(path.parent, exist_ok=True)
            return path
        elif outcome_name:
            path = (
                self.write_dir
                / PREDICTIONS_DIR_OUTCOME
                / outcome_name
                / PREDICTIONS_FILE
            )
            os.makedirs(path.parent, exist_ok=True)
            return path
        raise ValueError("Invalid prediction type or missing outcome name.")

    def get_calibrated_predictions_path(
        self, pred_type: str, outcome_name: str = None
    ) -> Path:
        if pred_type == "exposure":
            path = (
                self.write_dir / PREDICTIONS_DIR_EXPOSURE / CALIBRATED_PREDICTIONS_FILE
            )
            os.makedirs(path.parent, exist_ok=True)
            return path
        elif outcome_name:
            path = (
                self.write_dir
                / PREDICTIONS_DIR_OUTCOME
                / outcome_name
                / CALIBRATED_PREDICTIONS_FILE
            )
            os.makedirs(path.parent, exist_ok=True)
            return path
        raise ValueError("Invalid prediction type or missing outcome name.")

    def get_combined_calibrated_path(self) -> Path:
        path = self.write_dir / "combined_predictions_and_targets_calibrated.csv"
        os.makedirs(path.parent, exist_ok=True)
        return path

    def get_figure_dir(self, subfolder: str = None) -> Path:
        path = self.figures_dir
        if subfolder:
            path = path / subfolder
        os.makedirs(path, exist_ok=True)
        return path
