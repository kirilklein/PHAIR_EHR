from betacal import BetaCalibration
import numpy as np


def train_calibrator(
    train_preds: np.ndarray, train_targets: np.ndarray
) -> BetaCalibration:
    """Train the calibrator on the given dataframe."""
    calibrator = BetaCalibration("abm")
    calibrator.fit(train_preds, train_targets)
    return calibrator
