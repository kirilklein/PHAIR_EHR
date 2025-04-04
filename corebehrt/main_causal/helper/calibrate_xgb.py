from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from corebehrt.constants.causal.data import CF_PROBAS, PROBAS, TARGETS
from corebehrt.constants.data import PID_COL


def train_calibrator_from_data(
    predictions: np.ndarray, targets: np.ndarray
) -> IsotonicRegression:
    """Train an isotonic regression calibrator."""
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(predictions, targets)
    return calibrator


def apply_calibration_to_predictions(
    calibrator: IsotonicRegression, predictions: np.ndarray, epsilon: float = 1e-8
) -> np.ndarray:
    """Apply calibration to predictions and clip values."""
    calibrated = calibrator.predict(predictions)
    return np.clip(calibrated, epsilon, 1 - epsilon)


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
) -> pd.DataFrame:
    """
    Calibrates validation predictions using an isotonic regression calibrator
    trained on training predictions.

    Args:
        model: The trained XGBoost model
        train_data: Dictionary containing training features and labels
        val_data: Dictionary containing validation features and labels
        val_cf_data: Dictionary containing counterfactual validation features
        val_pids: List of patient IDs for the validation set
        epsilon: Clipping epsilon for calibration

    Returns:
        DataFrame with raw predictions, targets, calibrated predictions for both
        factual and counterfactual scenarios
    """
    # Collect training predictions and targets
    train_preds = get_predictions(model, X_train)
    calibrator = train_calibrator_from_data(train_preds, y_train)

    # Collect validation predictions and targets
    val_preds = get_predictions(model, X_val)
    calibrated_val = apply_calibration_to_predictions(calibrator, val_preds, epsilon)

    # Collect counterfactual validation predictions
    val_cf_preds = get_predictions(model, X_val_counter)
    calibrated_cf = apply_calibration_to_predictions(calibrator, val_cf_preds, epsilon)

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
