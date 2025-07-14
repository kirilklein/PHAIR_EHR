from dataclasses import dataclass
from typing import Dict, List
import pandas as pd


@dataclass
class CalibrationArtifacts:
    """
    Represents the complete set of artifacts produced by the calibration process.

    Attributes:
        combined_df: Combined dataframe with all calibrated predictions
        exposure_df: Raw exposure predictions
        calibrated_exposure_df: Calibrated exposure predictions
        outcomes: Dictionary of raw outcome predictions by outcome name
        calibrated_outcomes: Dictionary of calibrated outcome predictions by outcome name
        outcome_names: List of outcome names
    """

    combined_df: pd.DataFrame
    exposure_df: pd.DataFrame
    calibrated_exposure_df: pd.DataFrame
    outcomes: Dict[str, pd.DataFrame]
    calibrated_outcomes: Dict[str, pd.DataFrame]
    outcome_names: List[str]
