from os.path import join
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from betacal import BetaCalibration
from tqdm import tqdm

from corebehrt.constants.causal.data import PROBAS, TARGETS
from corebehrt.constants.causal.paths import (
    CALIBRATED_PREDICTIONS_FILE,
    PREDICTIONS_FILE,
)
from corebehrt.constants.data import PID_COL, TRAIN_KEY, VAL_KEY
from corebehrt.constants.paths import CHECKPOINTS_DIR
from corebehrt.functional.io_operations.load import get_pids_file
from corebehrt.functional.io_operations.paths import (
    get_checkpoint_predictions_path,
    get_fold_folders,
)
from corebehrt.functional.causal.stats import compute_probas_stats
from corebehrt.functional.setup.model import get_last_checkpoint_epoch
from corebehrt.functional.trainer.calibrate import train_calibrator


def save_combined_predictions(logger, write_dir: str, finetune_model_dir: str) -> None:
    """Combine predictions from all folds and save to finetune folder.
    Args:
        finetune_folder (str): The folder containing the finetuned models.
        mode (str): The mode to combine predictions for.
    """
    fold_folders = get_fold_folders(finetune_model_dir)

    predictions = []
    targets = []
    pids = []

    for fold_folder in tqdm(fold_folders, desc="Saving combined predictions"):
        # Get directories and last epoch
        fold_dir = join(finetune_model_dir, fold_folder)
        fold_checkpoints_folder = join(fold_dir, CHECKPOINTS_DIR)
        last_epoch = get_last_checkpoint_epoch(fold_checkpoints_folder)

        # Get paths to the predictions and targets
        predictions_path = get_checkpoint_predictions_path(
            PROBAS, fold_dir, VAL_KEY, last_epoch
        )
        targets_path = get_checkpoint_predictions_path(
            TARGETS, fold_dir, VAL_KEY, last_epoch
        )

        # Get the pids, predictions and targets
        fold_pids = torch.load(get_pids_file(fold_dir, mode=VAL_KEY), weights_only=True)
        fold_predictions = load_flattened_array_from_npz(
            predictions_path, PROBAS
        ).flatten()
        fold_targets = load_flattened_array_from_npz(targets_path, TARGETS).flatten()

        # Append to the lists
        predictions.append(fold_predictions)
        targets.append(fold_targets)
        pids.extend(fold_pids)

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    pre_cal_probas_stats = compute_probas_stats(predictions, targets)
    print("\nPre-calibration probas stats:")
    print(pre_cal_probas_stats)
    logger.info("Saving combined predictions")
    pd.DataFrame({PID_COL: pids, TARGETS: targets, PROBAS: predictions}).to_csv(
        join(write_dir, PREDICTIONS_FILE), index=False
    )


def compute_and_save_calibration(
    logger, write_dir: str, finetune_model_dir: str
) -> None:
    """
    Compute calibration for the predictions and save the results in a csv file: predictions_and_targets_calibrated.csv
    """
    preds = pd.read_csv(join(write_dir, PREDICTIONS_FILE))
    fold_folders = get_fold_folders(finetune_model_dir)

    all_calibrated_predictions: list[pd.DataFrame] = []

    for fold_folder in tqdm(fold_folders, desc="Computing calibration"):
        fold_dir = join(finetune_model_dir, fold_folder)

        train_pids = torch.load(
            get_pids_file(fold_dir, mode=TRAIN_KEY), weights_only=True
        )
        val_pids = torch.load(get_pids_file(fold_dir, mode=VAL_KEY), weights_only=True)
        train_data, val_data = split_data(preds, train_pids, val_pids)

        calibrated_probas = val_data[PROBAS]  # calibrate(train_data, val_data)
        val_data[PROBAS] = calibrated_probas  # Update the probabilities
        all_calibrated_predictions.append(val_data)

    combined_calibrated_df = pd.concat(all_calibrated_predictions, ignore_index=True)
    post_cal_probas_stats = compute_probas_stats(
        combined_calibrated_df[PROBAS], combined_calibrated_df[TARGETS]
    )
    print("\nPost-calibration probas stats:")
    print(post_cal_probas_stats)
    logger.info("Saving calibrated predictions")
    combined_calibrated_df.to_csv(
        join(write_dir, CALIBRATED_PREDICTIONS_FILE),
        index=False,
    )


def calibrate(
    train_data: pd.DataFrame, val_data: pd.DataFrame, epsilon=1e-8
) -> np.ndarray:
    """Calibrate the probabilities of the given dataframe using the calibrator."""
    calibrator = train_calibrator_from_data(train_data)
    calibrated_probas = calibrator.predict(val_data[PROBAS])
    calibrated_probas = np.clip(calibrated_probas, epsilon, 1 - epsilon)
    return calibrated_probas


def train_calibrator_from_data(train_data: pd.DataFrame) -> BetaCalibration:
    """Train the calibrator on the given dataframe."""
    return train_calibrator(train_data[PROBAS], train_data[TARGETS])


def split_data(
    predictions_df: pd.DataFrame, train_pids: torch.Tensor, val_pids: torch.Tensor
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the predictions dataframe into train and val dataframes based on the given PIDs."""
    train_data: pd.DataFrame = predictions_df[predictions_df[PID_COL].isin(train_pids)]
    val_data: pd.DataFrame = predictions_df[predictions_df[PID_COL].isin(val_pids)]
    return train_data, val_data


def get_folds_pids(finetune_model_dir: str, mode: str) -> List[List[str]]:
    """Get the pids for each fold."""
    fold_folders = get_fold_folders(finetune_model_dir)
    return [
        torch.load(
            get_pids_file(join(finetune_model_dir, fold_folder), mode),
            weights_only=True,
        )
        for fold_folder in fold_folders
    ]


def load_flattened_array_from_npz(file_path: str, field: str) -> np.ndarray:
    """Load a numpy file."""
    return np.load(file_path, allow_pickle=True)[field].flatten()
