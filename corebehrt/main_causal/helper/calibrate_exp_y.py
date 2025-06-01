"""
Prediction collection and calibration for causal inference models.
This module provides utilities for collecting, merging, and calibrating
factual and counterfactual predictions from trained causal models. It handles:
- Gathering predictions across cross-validation folds
- Processing exposure predictions and outcome predictions
- Combining factual and counterfactual outcomes
- Preparing prediction data for calibration procedures
- Calibrating raw model outputs for improved estimation
"""

import os
import warnings
from os.path import join
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss

from corebehrt.constants.causal.data import (
    CALIBRATION_COLLAPSE_THRESHOLD,
    CF_OUTCOME,
    CF_PROBAS,
    EXPOSURE,
    EXPOSURE_COL,
    OUTCOME,
    OUTCOME_COL,
    PROBAS,
    PS_COL,
    TARGETS,
)
from corebehrt.constants.causal.paths import (
    CALIBRATED_PREDICTIONS_FILE,
    PREDICTIONS_DIR_EXPOSURE,
    PREDICTIONS_DIR_OUTCOME,
    PREDICTIONS_FILE,
)
from corebehrt.constants.data import PID_COL, TRAIN_KEY, VAL_KEY
from corebehrt.constants.paths import FOLDS_FILE
from corebehrt.functional.causal.data_utils import split_data
from corebehrt.functional.io_operations.causal.predictions import collect_fold_data
from corebehrt.functional.trainer.calibrate import train_calibrator
from corebehrt.main_causal.helper.calibrate_plot import (
    produce_calibration_plots,
    produce_plots,
)


def load_calibrate_and_save(finetune_dir: str, write_dir: str) -> None:
    """
    Calibrate exposure and outcome predictions across cross-validation folds.

    This function operates on predictions previously collected and saved by
    collect_and_save_predictions(). It expects predictions to be already organized
    in the standard directory structure with separate folders for exposure and
    outcome predictions.

    The function:
    1. Loads previously collected predictions from write_dir
    2. Applies probability calibration using CV folds from finetune_dir
    3. Saves calibrated results to the same directory structure with a different filename

    Args:
        finetune_dir: Directory containing fine-tuned model folds for calibration
        write_dir: Directory where predictions were saved and where calibrated
                  predictions will be stored, following the established hierarchy:
                  write_dir/
                    ├── predictions_exposure/
                    │   ├── predictions_and_targets.csv
                    │   └── predictions_and_targets_calibrated.csv
                    └── predictions_outcome/
                        ├── predictions_and_targets.csv
                        └── predictions_and_targets_calibrated.csv
    """
    # Load folds
    folds = torch.load(join(finetune_dir, FOLDS_FILE))

    # Load collected predictions
    df_exp = pd.read_csv(join(write_dir, PREDICTIONS_DIR_EXPOSURE, PREDICTIONS_FILE))
    df_outcome = pd.read_csv(join(write_dir, PREDICTIONS_DIR_OUTCOME, PREDICTIONS_FILE))

    # Calibrate
    df_exp_calibrated = calibrate_folds(df_exp, folds)
    df_outcome_calibrated = calibrate_folds(df_outcome, folds)

    # Save calibrated predictions
    df_exp_calibrated.to_csv(
        join(write_dir, PREDICTIONS_DIR_EXPOSURE, CALIBRATED_PREDICTIONS_FILE),
        index=False,
    )
    df_outcome_calibrated.to_csv(
        join(write_dir, PREDICTIONS_DIR_OUTCOME, CALIBRATED_PREDICTIONS_FILE),
        index=False,
    )

    df = combine_predictions(df_exp_calibrated, df_outcome_calibrated)
    df.to_csv(
        join(write_dir, "combined_predictions_and_targets_calibrated.csv"), index=False
    )

    fig_dir = join(write_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    produce_calibration_plots(
        df_exp_calibrated, df_exp, fig_dir, "Propensity Score Calibration", "ps"
    )
    produce_calibration_plots(
        df_outcome_calibrated,
        df_outcome,
        fig_dir,
        "Outcome Probability Calibration",
        "outcome",
    )

    produce_plots(df, fig_dir)


def combine_predictions(exp: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
    exp = exp.rename(columns={PROBAS: PS_COL, TARGETS: EXPOSURE_COL})
    out = out.rename(columns={TARGETS: OUTCOME_COL})
    df = pd.merge(exp, out, on=PID_COL, how="inner", validate="1:1")
    return df


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
        if (CF_PROBAS in val_df.columns) and calibration_performed:
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


def collect_and_save_predictions(
    finetune_dir: str,
    write_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Collect and save exposure and outcome predictions from all folds.

    Gathers exposure predictions, factual outcome predictions, and counterfactual
    outcome predictions from trained models across all folds. Merges factual and
    counterfactual outcomes into a single dataset. Saves results to separate
    directories for exposure and outcome predictions.

    Args:
        finetune_dir: Directory containing fine-tuned model folds
        write_dir: Directory to save the collected predictions
    """
    exposure_predictions_dir = join(write_dir, PREDICTIONS_DIR_EXPOSURE)
    outcome_predictions_dir = join(write_dir, PREDICTIONS_DIR_OUTCOME)
    os.makedirs(exposure_predictions_dir, exist_ok=True)
    os.makedirs(outcome_predictions_dir, exist_ok=True)
    df_exp: pd.DataFrame = collect_combined_predictions(
        finetune_model_dir=finetune_dir,
        prediction_type=EXPOSURE,
        mode=VAL_KEY,
        probas_name=PROBAS,
    )
    df_exp.to_csv(join(exposure_predictions_dir, PREDICTIONS_FILE), index=False)
    df_outcome: pd.DataFrame = collect_combined_predictions(
        finetune_model_dir=finetune_dir,
        prediction_type=OUTCOME,
        mode=VAL_KEY,
        probas_name=PROBAS,
    )
    df_cf_outcome: pd.DataFrame = collect_combined_predictions(
        finetune_model_dir=finetune_dir,
        prediction_type=CF_OUTCOME,
        mode=VAL_KEY,
        probas_name=CF_PROBAS,
        collect_targets=False,
    )

    df_outcome_combined = pd.merge(
        df_outcome, df_cf_outcome, on=PID_COL, how="inner", validate="1:1"
    )
    df_outcome_combined.to_csv(
        join(outcome_predictions_dir, PREDICTIONS_FILE), index=False
    )
    return df_exp, df_outcome_combined


def collect_combined_predictions(
    finetune_model_dir: str,
    prediction_type: str,
    mode: str = VAL_KEY,
    probas_name: str = PROBAS,
    targets_name: str = TARGETS,
    collect_targets: bool = True,
) -> pd.DataFrame:
    """
    Combine predictions from all folds and save to output folder.
    prediction_types: outcome, exposure, cf_outcome (used to construct name and acces the correct files.)
    """
    # Step 1: Collect data from all folds
    pids, predictions, targets = collect_fold_data(
        finetune_model_dir, prediction_type, mode, collect_targets
    )
    df = pd.DataFrame({PID_COL: pids, probas_name: predictions})
    if collect_targets:
        df[targets_name] = targets.astype(int)
    return df


def robust_calibration_with_fallback(
    calibrated_probas: np.ndarray,
    preds: np.ndarray,
    targets: np.ndarray,
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
        return preds, False
    original_brier_score = brier_score_loss(preds, targets)
    calibrated_brier_score = brier_score_loss(calibrated_probas, targets)
    if calibrated_brier_score > original_brier_score:
        warnings.warn(
            f"Calibrated Brier score ({calibrated_brier_score:.6f}) is higher than original Brier score ({original_brier_score:.6f}). Using original probabilities instead."
        )
        return preds, False
    else:
        return calibrated_probas, True
