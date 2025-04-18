import numpy as np
import warnings

from corebehrt.constants.causal.data import CALIBRATION_COLLAPSE_THRESHOLD


def safe_assign_calibrated_probas(
    calibrated_probas: np.ndarray,
    preds: np.ndarray,
    epsilon: float = 1e-8,
    collapse_threshold: float = CALIBRATION_COLLAPSE_THRESHOLD,
) -> np.ndarray:
    """
    If the calibrated probabilities appear to be collapsed (have very low variance),
    keeps the original probabilities instead. Returns the updated validation dataframe.
    """
    calibrated_probas = np.clip(calibrated_probas, epsilon, 1 - epsilon)
    if np.std(calibrated_probas) < collapse_threshold:
        warnings.warn(
            f"Calibrated probabilities appear to be collapsed (std={np.std(calibrated_probas):.6f}). Using original probabilities instead."
        )
        return preds
    else:
        return calibrated_probas
