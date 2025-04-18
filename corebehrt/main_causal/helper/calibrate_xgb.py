from typing import Any

import numpy as np
import pandas as pd

from corebehrt.constants.causal.data import (
    CF_PROBAS,
    PROBAS,
    TARGETS,
    CALIBRATION_COLLAPSE_THRESHOLD,
)
from corebehrt.constants.data import PID_COL
from corebehrt.functional.trainer.calibrate import train_calibrator
from corebehrt.main_causal.helper.utils import safe_assign_calibrated_probas


def get_predictions(
    model: Any,
    X: np.ndarray,
) -> np.ndarray:
    """Get positive class probabilities from model."""
    predictions = model.predict_proba(X)[:, 1]  # Get positive class probabilities
    return predictions


def calibrate_predictions(
    model: Any,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_val_counter: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    val_pids: list,
    epsilon: float = 1e-8,
    collapse_threshold: float = CALIBRATION_COLLAPSE_THRESHOLD,
) -> pd.DataFrame:
    """
    Calibrates validation predictions using an isotonic regression calibrator
    trained on training predictions.
    If the calibrated probabilities appear to be collapsed (have very low variance),
    keeps the original probabilities instead.

    Args:
        model: The trained XGBoost model
        X_train: Training features
        X_val: Validation features
        X_val_counter: Counterfactual validation features
        y_train: Training targets
        y_val: Validation targets
        val_pids: List of patient IDs for the validation set
        epsilon: Clipping epsilon for calibration

    Returns:
        DataFrame with raw predictions, targets, calibrated predictions for both
        factual and counterfactual scenarios
    """
    # Collect training predictions and targets
    train_preds = get_predictions(model, X_train)
    calibrator = train_calibrator(train_preds, y_train)

    # Collect validation predictions and targets
    val_preds = get_predictions(model, X_val)
    calibrated_val = calibrator.predict(val_preds)
    calibrated_val = safe_assign_calibrated_probas(
        calibrated_val, val_preds, epsilon, collapse_threshold
    )

    # Collect counterfactual validation predictions
    val_cf_preds = get_predictions(model, X_val_counter)
    calibrated_cf = calibrator.predict(val_cf_preds)
    calibrated_cf = safe_assign_calibrated_probas(
        calibrated_cf, val_cf_preds, epsilon, collapse_threshold
    )

    # Create DataFrame with results
    val_df = pd.DataFrame(
        {
            PID_COL: val_pids,
            PROBAS: calibrated_val,
            CF_PROBAS: calibrated_cf,
            TARGETS: y_val,
        }
    )

    return val_df
