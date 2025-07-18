import logging
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from typing import List, Tuple, Dict, Any
from corebehrt.constants.causal.data import (
    CALIBRATION_COLLAPSE_THRESHOLD,
    CF_PROBAS,
    PROBAS,
    TARGETS,
)
from corebehrt.constants.data import PID_COL, TRAIN_KEY, VAL_KEY
from corebehrt.functional.causal.data_utils import split_data
from corebehrt.functional.trainer.calibrate import train_calibrator

EPSILON = 1e-8  # Small value to clip probabilities, preventing log loss errors.
logger = logging.getLogger(__name__)


def calibrate_folds(
    df: pd.DataFrame,
    folds: List[Dict[str, np.ndarray]],
    epsilon: float = EPSILON,
    calibration_collapse_threshold: float = CALIBRATION_COLLAPSE_THRESHOLD,
) -> pd.DataFrame:
    """
    Calibrates predictions across cross-validation folds.

    For each fold, this function:
    1. Splits data into training and validation sets.
    2. Trains a calibrator on the training data.
    3. Applies robust calibration to the validation predictions.
    4. Concatenates the results from all folds.

    Args:
        df: The main DataFrame containing all data.
        folds: A list of dictionaries, each representing a fold with 'train' and 'val' keys.
        epsilon: A small value to clip probabilities, preventing log loss errors.
        calibration_collapse_threshold: The standard deviation threshold below which
                                        calibrated probabilities are considered collapsed.

    Returns:
        A DataFrame with calibrated predictions for all validation sets.
    """
    logger.info("🚀 Starting cross-fold calibration...")
    calibrated_dfs = []

    for i, fold in enumerate(folds):
        fold_num = i + 1
        logger.info(f"--- Processing Fold {fold_num}/{len(folds)} ---")

        train_pids, val_pids = fold[TRAIN_KEY], fold[VAL_KEY]
        train_df, val_df = split_data(df, train_pids, val_pids)

        calibrator = train_calibrator(train_df[PROBAS], train_df[TARGETS])

        initial_calibrated_probas = calibrator.predict(val_df[PROBAS])

        calibrated_probas, calibration_applied = _robust_calibration_with_fallback(
            initial_calibrated_probas,
            val_df[PROBAS].values,
            val_df[TARGETS].values,
            epsilon,
            calibration_collapse_threshold,
            fold_num,
        )

        fold_results = {
            PID_COL: val_df[PID_COL].values,
            PROBAS: calibrated_probas,
            TARGETS: val_df[TARGETS].values,
        }

        if CF_PROBAS in val_df.columns:
            if calibration_applied:
                fold_results[CF_PROBAS] = calibrator.predict(val_df[CF_PROBAS])
            else:
                fold_results[CF_PROBAS] = val_df[CF_PROBAS].values

        calibrated_dfs.append(pd.DataFrame(fold_results))

    logger.info("✅ Cross-fold calibration complete.")
    return pd.concat(calibrated_dfs, ignore_index=True)


def _evaluate_calibration_performance(
    calibrated_probas: np.ndarray,
    original_probas: np.ndarray,
    targets: np.ndarray,
    epsilon: float = EPSILON,
) -> Dict[str, Any]:
    """
    Evaluates and compares calibration performance using Brier score and ROC AUC.

    Args:
        calibrated_probas: The calibrated probability predictions.
        original_probas: The original probability predictions.
        targets: The true binary targets.
        epsilon: A small value to clip probabilities.

    Returns:
        A dictionary containing performance metrics and improvement flags.
    """
    # Clip probabilities to prevent errors in metric calculations
    calibrated_probas = np.clip(calibrated_probas, epsilon, 1 - epsilon)
    original_probas = np.clip(original_probas, epsilon, 1 - epsilon)

    # --- Metric Calculation ---
    original_brier = brier_score_loss(targets, original_probas)
    calibrated_brier = brier_score_loss(targets, calibrated_probas)
    original_auc = roc_auc_score(targets, original_probas)
    calibrated_auc = roc_auc_score(targets, calibrated_probas)
    calibrated_std = np.std(calibrated_probas)

    # --- Performance Comparison ---
    brier_improvement = original_brier - calibrated_brier
    auc_improvement = calibrated_auc - original_auc

    return {
        "original_brier": original_brier,
        "calibrated_brier": calibrated_brier,
        "brier_improvement": brier_improvement,
        "original_auc": original_auc,
        "calibrated_auc": calibrated_auc,
        "auc_improvement": auc_improvement,
        "calibrated_std": calibrated_std,
        "brier_improved": brier_improvement > 0,
        "auc_tolerable": auc_improvement >= -0.001,  # Allow for minor AUC degradation
    }


def _robust_calibration_with_fallback(
    calibrated_probas: np.ndarray,
    original_probas: np.ndarray,
    targets: np.ndarray,
    epsilon: float,
    collapse_threshold: float,
    fold_num: int,
) -> Tuple[np.ndarray, bool]:
    """
    Applies calibration if it improves metrics and doesn't collapse probabilities.
    Otherwise, it falls back to the original probabilities.

    Args:
        calibrated_probas: The post-calibration probabilities.
        original_probas: The pre-calibration probabilities.
        targets: The true labels.
        epsilon: A small value for clipping probabilities.
        collapse_threshold: The standard deviation threshold for detecting collapse.
        fold_num: The current fold number for logging.

    Returns:
        A tuple containing the final probabilities (calibrated or original) and a
        boolean indicating if calibration was applied.
    """
    performance = _evaluate_calibration_performance(
        calibrated_probas, original_probas, targets, epsilon
    )

    # --- Log Performance Summary ---
    logger.info(
        f"[Fold {fold_num}] Performance | "
        f"Brier: {performance['original_brier']:.4f} -> {performance['calibrated_brier']:.4f} "
        f"({'+' if performance['brier_improvement'] > 0 else ''}{performance['brier_improvement']:.4f}) | "
        f"AUC: {performance['original_auc']:.4f} -> {performance['calibrated_auc']:.4f} "
        f"({'+' if performance['auc_improvement'] > 0 else ''}{performance['auc_improvement']:.4f})"
    )

    # --- Decision Logic ---
    is_collapsed = performance["calibrated_std"] < collapse_threshold
    if is_collapsed:
        logger.warning(
            f"[Fold {fold_num}] ⚠️ Calibration collapsed (std={performance['calibrated_std']:.4f}). "
            "Falling back to original probabilities."
        )
        return original_probas, False

    if not performance["brier_improved"]:
        logger.warning(
            f"[Fold {fold_num}] 📉 Brier score did not improve. "
            "Falling back to original probabilities."
        )
        return original_probas, False

    if not performance["auc_tolerable"]:
        logger.warning(
            f"[Fold {fold_num}] 📉 AUC degraded significantly (by {abs(performance['auc_improvement']):.4f}). "
            "Despite Brier improvement, this may indicate an issue."
        )

    logger.info(f"[Fold {fold_num}] ✨ Calibration successful and applied.")
    return np.clip(calibrated_probas, epsilon, 1 - epsilon), True
