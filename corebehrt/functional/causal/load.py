from os.path import join
from typing import Tuple

import pandas as pd
import torch

from corebehrt.constants.causal.data import TARGETS
from corebehrt.constants.causal.paths import CALIBRATED_PREDICTIONS_FILE, ENCODINGS_FILE
from corebehrt.constants.paths import PID_FILE
from corebehrt.functional.causal.data_utils import align_df_with_pids


def load_encodings_and_pids_from_encoded_dir(
    encoded_dir: str,
) -> Tuple[torch.Tensor, list]:
    """Load encodings and patient IDs from the encoded directory.

    Args:
        encoded_dir: Path to directory containing encoded data files

    Returns:
        Tuple of (encodings array, patient IDs array)
    """
    encodings = torch.load(join(encoded_dir, ENCODINGS_FILE))
    pids = torch.load(join(encoded_dir, PID_FILE))
    return encodings, pids


def load_exposure_from_predictions(
    calibrated_predictions_dir: str, pids: list
) -> torch.Tensor:
    predictions = load_and_align_calibrated_predictions(
        calibrated_predictions_dir, pids
    )
    return torch.tensor(predictions[TARGETS].values)


def load_and_align_calibrated_predictions(
    calibrated_predictions_dir: str,
    pids: list,
) -> pd.DataFrame:
    """Load calibrated predictions and align them with patient IDs.

    Args:
        calibrated_predictions_dir: Path to directory containing calibrated predictions
        pids: Array of patient IDs to align predictions with

    Returns:
        DataFrame of predictions aligned with patient IDs
    """
    predictions = pd.read_csv(
        join(calibrated_predictions_dir, CALIBRATED_PREDICTIONS_FILE)
    )
    predictions = align_df_with_pids(predictions, pids)
    return predictions
