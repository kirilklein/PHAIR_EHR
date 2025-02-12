import os
from os.path import join

import numpy as np
import pandas as pd
import torch

from corebehrt.constants.causal import PREDICTIONS_FILE, PROBAS, TARGETS
from corebehrt.constants.data import PID_COL
from corebehrt.constants.paths import CHECKPOINTS_DIR
from corebehrt.functional.io_operations.load import get_pids_file
from corebehrt.functional.setup.model import get_last_checkpoint_epoch


def save_combined_predictions(finetune_folder: str, mode="val") -> None:
    """Combine predictions from all folds and save to finetune folder.
    Args:
        finetune_folder (str): The folder containing the finetuned models.
        mode (str): The mode to combine predictions for.
    """
    fold_folders = get_fold_folders(finetune_folder)

    predictions = []
    targets = []
    pids = []

    for fold_folder in fold_folders:
        fold_dir = join(finetune_folder, fold_folder)
        fold_checkpoints_folder = join(fold_dir, CHECKPOINTS_DIR)
        last_epoch = get_last_checkpoint_epoch(fold_checkpoints_folder)

        predictions_path = get_file_path(PROBAS, fold_dir, mode, last_epoch)
        targets_path = get_file_path(TARGETS, fold_dir, mode, last_epoch)

        fold_pids = torch.load(get_pids_file(fold_dir, mode="val"), weights_only=True)
        fold_predictions = load_numpy_file(predictions_path, PROBAS).flatten()
        fold_targets = load_numpy_file(targets_path, TARGETS).flatten()

        predictions.append(fold_predictions)
        targets.append(fold_targets)
        pids.extend(fold_pids)

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)

    pd.DataFrame({PID_COL: pids, TARGETS: targets, PROBAS: predictions}).to_csv(
        join(finetune_folder, PREDICTIONS_FILE), index=False
    )


def load_numpy_file(file_path: str, field: str) -> np.ndarray:
    """Load a numpy file."""
    return np.load(file_path, allow_pickle=True)[field].flatten()


def get_fold_folders(finetune_folder: str) -> list[str]:
    """Get the number of folds from the finetune folder."""
    return [f for f in os.listdir(finetune_folder) if f.startswith("fold_")]


def get_file_path(file_type: str, fold_dir: str, mode: str, epoch: int) -> str:
    """Get the path to the file for a given fold and mode."""
    return join(fold_dir, CHECKPOINTS_DIR, f"{file_type}_{mode}_{epoch}.npz")
