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

import pandas as pd

from corebehrt.constants.causal.data import (
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
from corebehrt.constants.data import PID_COL, VAL_KEY
from corebehrt.functional.io_operations.causal.predictions import (
    collect_fold_data,
)


def collect_and_save_predictions(
    finetune_dir: str,
    write_dir: str,
):
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

    df = pd.merge(df_outcome, df_cf_outcome, on=PID_COL, how="inner", validate="1:1")
    df.to_csv(join(outcome_predictions_dir, PREDICTIONS_FILE), index=False)


def collect_combined_predictions(
    finetune_model_dir: str,
    prediction_type: str,
    mode: str = VAL_KEY,
    probas_name: str = PROBAS,
    targets_name: str = TARGETS,
    collect_targets: bool = True,
) -> pd.DataFrame:
    """Combine predictions from all folds and save to output folder.
    prediction_types: outcome, exposure, cf_outcome (used to construct name and acces the correct files.)
    """
    # Step 1: Collect data from all folds
    pids, predictions, targets = collect_fold_data(
        finetune_model_dir, prediction_type, mode, collect_targets
    )
    df = pd.DataFrame({PID_COL: pids, probas_name: predictions, targets_name: targets})
    return df
