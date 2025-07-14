import warnings
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from corebehrt.constants.causal.data import (
    CALIBRATION_COLLAPSE_THRESHOLD,
    CF_PROBAS,
    PROBAS,
    TARGETS,
)
from corebehrt.constants.data import PID_COL, TRAIN_KEY, VAL_KEY
from corebehrt.functional.causal.data_utils import split_data
from corebehrt.functional.trainer.calibrate import train_calibrator


def calibrate_folds(
    df: pd.DataFrame,
    folds: list,
    epsilon: float = 1e-8,
    calibration_collapse_threshold: float = CALIBRATION_COLLAPSE_THRESHOLD,
) -> pd.DataFrame:
    """
    Calibrate predictions across cross-validation folds.

    For each fold:
    1. Splits data into train/validation sets
    2. Trains a calibrator on training data
    3. Applies calibration to validation predictions (factual and counterfactual)
    4. Applies safety bounds to prevent extreme probabilities

    Returns a concatenated DataFrame of all calibrated validation predictions.
    """
    all_calibrated_dfs = []
    for fold in folds:
        train_pids = fold[TRAIN_KEY]
        val_pids = fold[VAL_KEY]

        train_df, val_df = split_data(df, train_pids, val_pids)
        calibrator = train_calibrator(train_df[PROBAS], train_df[TARGETS])

        calibrated = calibrator.predict(val_df[PROBAS])
        calibrated, calibration_performed = robust_calibration_with_fallback(
            calibrated,
            val_df[PROBAS],
            val_df[TARGETS],
            epsilon,
            calibration_collapse_threshold,
        )
        calibrated_cf = None
        if CF_PROBAS in val_df.columns:
            if calibration_performed:
                calibrated_cf = calibrator.predict(val_df[CF_PROBAS])
            else:
                calibrated_cf = val_df[CF_PROBAS]

        calibrated = pd.DataFrame(
            {
                PID_COL: val_df[PID_COL].values,
                PROBAS: calibrated,
                TARGETS: val_df[TARGETS].values,
            }
        )
        if calibrated_cf is not None:
            calibrated[CF_PROBAS] = calibrated_cf
        all_calibrated_dfs.append(calibrated)
    return pd.concat(all_calibrated_dfs)


def robust_calibration_with_fallback(
    calibrated_probas: np.ndarray,
    preds: np.ndarray,
    targets: np.ndarray,
    epsilon: float = 1e-8,
    collapse_threshold: float = CALIBRATION_COLLAPSE_THRESHOLD,
) -> Tuple[np.ndarray, bool]:
    """
    If the calibrated probabilities appear to be collapsed (have very low variance),
    keeps the original probabilities instead. Returns the updated validation dataframe.
    """
    calibrated_probas = np.clip(calibrated_probas, epsilon, 1 - epsilon)
    if np.std(calibrated_probas) < collapse_threshold:
        warnings.warn(
            f"Calibrated probabilities appear to be collapsed (std={np.std(calibrated_probas):.6f}). Using original probabilities instead."
        )
        return preds, False
    original_brier_score = brier_score_loss(targets, preds)
    calibrated_brier_score = brier_score_loss(targets, calibrated_probas)
    if calibrated_brier_score > original_brier_score:
        warnings.warn(
            f"Calibrated Brier score ({calibrated_brier_score:.6f}) is higher than original Brier score ({original_brier_score:.6f}). Using original probabilities instead."
        )
        return preds, False
    else:
        return calibrated_probas, True
