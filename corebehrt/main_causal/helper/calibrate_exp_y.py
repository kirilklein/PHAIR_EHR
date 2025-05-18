"""
Prediction collection and calibration for causal inference models.
This module provides utilities for collecting, merging, and calibrating
factual and counterfactual predictions from trained causal models. It handles:
- Gathering predictions across cross-validation folds
- Processing exposure predictions and outcome predictions
- Combining factual and counterfactual outcomes
- Preparing prediction data for calibration procedures
- Calibrating raw model outputs for improved estimation
The module serves as a bridge between model training and causal effect estimation.
"""

import os
from os.path import join
from typing import Tuple

import numpy as np
import pandas as pd

from corebehrt.constants.causal.data import (
    CALIBRATION_COLLAPSE_THRESHOLD,
    CF_OUTCOME,
    CF_PROBAS,
    EXPOSURE,
    OUTCOME,
    PROBAS,
    TARGETS,
)
from corebehrt.constants.causal.paths import (
    PREDICTIONS_DIR_EXPOSURE,
    PREDICTIONS_DIR_OUTCOME,
    PREDICTIONS_FILE,
)
from corebehrt.constants.data import PID_COL, TRAIN_KEY, VAL_KEY
from corebehrt.functional.causal.data_utils import split_data
from corebehrt.functional.io_operations.causal.predictions import collect_fold_data
from corebehrt.functional.trainer.calibrate import train_calibrator
from corebehrt.main_causal.helper.utils import safe_assign_calibrated_probas


def calibrate_folds(
    df: pd.DataFrame,
    folds: list,
    epsilon: float = 1e-8,
    calibration_collapse_threshold: float = CALIBRATION_COLLAPSE_THRESHOLD,
):
    """
    Calibrate predictions using the calibration method.
    """
    calibrated_dfs = []
    for fold in folds:
        train_pids = fold[TRAIN_KEY]
        val_pids = fold[VAL_KEY]
        train, val = split_data(df, train_pids, val_pids)
        calibrator = train_calibrator(train[PROBAS], train[TARGETS])
        calibrated = calibrator.predict(val[PROBAS])
        calibrated = safe_assign_calibrated_probas(
            calibrated, val[PROBAS], epsilon, calibration_collapse_threshold
        )
        if CF_PROBAS in val.columns:
            calibrated_cf = calibrator.predict(val[CF_PROBAS])
            calibrated_cf = safe_assign_calibrated_probas(
                calibrated_cf,
                val[CF_PROBAS],
                epsilon,
                calibration_collapse_threshold,
            )
        calibrated = pd.DataFrame({PID_COL: val_pids, PROBAS: calibrated})
        if CF_PROBAS in val.columns:
            calibrated[CF_PROBAS] = calibrated_cf
        calibrated_dfs.append(calibrated)
    return pd.concat(calibrated_dfs)


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
