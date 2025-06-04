"""
Prediction data I/O utilities for causal inference models.

This module provides functions for loading, processing, and managing prediction
data across model folds, including access to prediction probabilities, targets,
patient IDs, and file path handling.
"""

import os
from os.path import join
from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from corebehrt.constants.causal.data import EXPOSURE, OUTCOME, PROBAS, TARGETS
from corebehrt.constants.paths import CHECKPOINTS_DIR
from corebehrt.functional.io_operations.load import get_pids_file
from corebehrt.functional.io_operations.paths import get_fold_folders
from corebehrt.functional.setup.model import get_last_checkpoint_epoch


def collect_fold_data(
    finetune_model_dir: str,
    prediction_type: str,
    mode: str,
    collect_targets: bool = True,
) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """Collect predictions and optionally targets from all folds."""
    fold_folders = get_fold_folders(finetune_model_dir)

    # Initialize collections
    pids = []
    predictions = []
    targets = []

    # Process each fold
    for fold_folder in tqdm(fold_folders, desc="Processing folds"):
        # Get directories and last epoch
        fold_dir = join(finetune_model_dir, fold_folder)
        fold_checkpoints_folder = join(fold_dir, CHECKPOINTS_DIR)
        last_epoch = get_last_checkpoint_epoch(fold_checkpoints_folder)

        # Get the PIDs for this fold
        fold_pids = load_fold_pids(fold_dir, mode)
        pids.extend(fold_pids)

        # Process each prediction type
        fold_predictions, fold_targets = load_fold_predictions(
            fold_dir, prediction_type, mode, last_epoch, collect_targets
        )
        predictions.append(fold_predictions)

        if collect_targets:
            targets.append(fold_targets)

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets) if collect_targets else None
    return pids, predictions, targets


def load_fold_predictions(
    fold_dir: str, pred_type: str, mode: str, epoch: int, collect_targets: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load predictions and optionally targets for a single fold and prediction type.
    If collect_targets is False, only predictions are loaded and None is returned for targets.
    """
    predictions_path = get_prediction_file_path(
        fold_dir, PROBAS, mode, pred_type, epoch
    )

    # Load predictions (required)
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
    predictions = load_flattened_array_from_npz(predictions_path, PROBAS)

    # Skip targets if not needed
    if not collect_targets:
        return predictions, None

    # Load targets
    targets_path = get_prediction_file_path(fold_dir, TARGETS, mode, pred_type, epoch)
    if not os.path.exists(targets_path):
        raise FileNotFoundError(f"Targets file not found: {targets_path}")
    targets = load_flattened_array_from_npz(targets_path, TARGETS)

    return predictions, targets


def load_fold_pids(fold_dir: str, mode: str) -> List[int]:
    """Construct pid file path based on fold directory and data split mode."""
    pids_file = get_pids_file(fold_dir, mode)
    return torch.load(pids_file)


def load_flattened_array_from_npz(file_path: str, field: str) -> np.ndarray:
    """Load npz file and access field as numpy array."""
    return np.load(file_path, allow_pickle=True)[field].flatten()


def get_prediction_file_path(
    fold_dir: str, type_prefix: str, mode: str, pred_type: str, epoch: int
) -> str:
    """
    Builds a standardized path to prediction files based on fold directory,
    data split mode, prediction type, and model checkpoint epoch.
    """
    file_suffix = pred_type if pred_type in [OUTCOME, EXPOSURE] else pred_type

    return join(
        fold_dir, CHECKPOINTS_DIR, f"{type_prefix}_{mode}_{file_suffix}_{epoch}.npz"
    )
