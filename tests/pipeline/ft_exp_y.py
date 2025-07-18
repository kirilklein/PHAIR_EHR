#!/usr/bin/env python
"""
Test script to validate consistency between fold PIDs and prediction arrays.

This script checks:
1. Whether the PIDs in folds.pt match the PIDs in each fold directory
2. Whether the prediction and target arrays match the length of PIDs
3. Whether the folds.pt list holds the same PIDs as stored in each fold

Usage:
    python check_finetune_exp_y.py path/to/finetune_dir
"""

import argparse
import sys
from os.path import exists, join

import numpy as np
import torch

from corebehrt.constants.causal.data import (
    CF_OUTCOME,
    EXPOSURE,
    OUTCOME,
    PROBAS,
    TARGETS,
)
from corebehrt.constants.data import VAL_KEY
from corebehrt.constants.paths import CHECKPOINTS_DIR, FOLDS_FILE, OUTCOME_NAMES_FILE
from corebehrt.functional.io_operations.paths import get_fold_folders
from corebehrt.functional.setup.model import get_last_checkpoint_epoch


def check_fold_pids_match(finetune_dir: str, fold_name: str, mode: str) -> bool:
    """Check if PIDs in folds file match the PIDs saved in each fold directory."""
    # Load main folds file
    folds = torch.load(join(finetune_dir, FOLDS_FILE))

    # Map fold_1 naming to 0-indexed fold list
    fold_idx = int(fold_name.split("_")[-1]) - 1
    fold_mode_key = mode

    # Get expected PIDs from main folds file
    if fold_idx >= len(folds):
        print(f"Error: Fold {fold_name} not found in folds file")
        return False

    expected_pids = folds[fold_idx][fold_mode_key]

    # Get actual PIDs from fold directory
    fold_dir = join(finetune_dir, fold_name)
    actual_pids_path = join(fold_dir, f"{mode}_pids.pt")

    if not exists(actual_pids_path):
        print(f"Error: {actual_pids_path} not found")
        return False

    actual_pids = torch.load(actual_pids_path)

    # Compare PIDs
    expected_set = set(expected_pids)
    actual_set = set(actual_pids)

    if expected_set != actual_set:
        print(f"Error: PIDs mismatch for {fold_name} ({mode})")
        print(f"  Expected {len(expected_pids)} PIDs, got {len(actual_pids)} PIDs")
        print(f"  Missing from fold_dir: {len(expected_set - actual_set)}")
        print(f"  Extra in fold_dir: {len(actual_set - expected_set)}")
        return False

    return True


def check_predictions_match_pids(
    fold_dir: str, mode: str, pred_type: str, epoch: int, outcome_name: str = None
) -> bool:
    """Check if prediction arrays match the number of PIDs."""
    # Load PIDs
    pids_path = join(fold_dir, f"{mode}_pids.pt")
    if not exists(pids_path):
        print(f"Error: {pids_path} not found")
        return False

    pids = torch.load(pids_path)

    # Build prediction file name
    if outcome_name and pred_type == CF_OUTCOME:
        pred_filename = f"{PROBAS}_{mode}_{pred_type}_{outcome_name}_{epoch}.npz"
        target_filename = f"{TARGETS}_{mode}_{pred_type}_{outcome_name}_{epoch}.npz"
    elif outcome_name and pred_type == OUTCOME:
        pred_filename = f"{PROBAS}_{mode}_{outcome_name}_{epoch}.npz"
        target_filename = f"{TARGETS}_{mode}_{outcome_name}_{epoch}.npz"
    else:
        pred_filename = f"{PROBAS}_{mode}_{pred_type}_{epoch}.npz"
        target_filename = f"{TARGETS}_{mode}_{pred_type}_{epoch}.npz"

    # Load predictions
    predictions_path = join(fold_dir, CHECKPOINTS_DIR, pred_filename)

    if not exists(predictions_path):
        print(f"Error: {predictions_path} not found")
        return False

    predictions = np.load(predictions_path, allow_pickle=True)[PROBAS]

    # Load targets if available
    targets_path = join(fold_dir, CHECKPOINTS_DIR, target_filename)

    targets = None
    if exists(targets_path):
        targets = np.load(targets_path, allow_pickle=True)[TARGETS]

    # Compare lengths
    if len(predictions) != len(pids):
        print(
            f"Error: Predictions length ({len(predictions)}) doesn't match PIDs length ({len(pids)})"
        )
        return False

    if targets is not None and len(targets) != len(pids):
        print(
            f"Error: Targets length ({len(targets)}) doesn't match PIDs length ({len(pids)})"
        )
        return False

    return True


def main(finetune_dir: str):
    """Main validation function."""
    # Verify folds file exists
    folds_path = join(finetune_dir, FOLDS_FILE)
    if not exists(folds_path):
        print(f"Error: Folds file {folds_path} not found")
        return False

    # Load outcome names
    outcome_names_path = join(finetune_dir, OUTCOME_NAMES_FILE)
    if not exists(outcome_names_path):
        print(f"Error: Outcome names file {outcome_names_path} not found")
        return False

    outcome_names = torch.load(outcome_names_path)
    print(f"Loaded outcome names: {outcome_names}")

    # Get fold directories
    fold_folders = get_fold_folders(finetune_dir)
    if not fold_folders:
        print(f"Error: No fold directories found in {finetune_dir}")
        return False

    all_checks_passed = True

    # Check each fold
    for fold_name in fold_folders:
        fold_dir = join(finetune_dir, fold_name)

        # Get last checkpoint epoch
        checkpoints_dir = join(fold_dir, CHECKPOINTS_DIR)
        try:
            last_epoch = get_last_checkpoint_epoch(checkpoints_dir)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error getting last checkpoint for {fold_name}: {e}")
            all_checks_passed = False
            continue

        # Check validation PIDs match between folds.pt and fold_x/val_pids.pt
        pids_match = check_fold_pids_match(finetune_dir, fold_name, VAL_KEY)
        all_checks_passed = all_checks_passed and pids_match

        # Check EXPOSURE predictions (no outcome name needed)
        try:
            predictions_match = check_predictions_match_pids(
                fold_dir, VAL_KEY, EXPOSURE, last_epoch
            )
            all_checks_passed = all_checks_passed and predictions_match

            if not predictions_match:
                print(f"Prediction mismatch in {fold_name} for {EXPOSURE}")

        except Exception as e:
            print(f"Error checking predictions for {fold_name}, {EXPOSURE}: {e}")
            all_checks_passed = False

        # Check OUTCOME and CF_OUTCOME predictions for each outcome name
        for outcome_name in outcome_names:
            for pred_type in [OUTCOME, CF_OUTCOME]:
                try:
                    predictions_match = check_predictions_match_pids(
                        fold_dir, VAL_KEY, pred_type, last_epoch, outcome_name
                    )
                    all_checks_passed = all_checks_passed and predictions_match

                    if not predictions_match:
                        print(
                            f"Prediction mismatch in {fold_name} for {pred_type}_{outcome_name}"
                        )

                except Exception as e:
                    print(
                        f"Error checking predictions for {fold_name}, {pred_type}_{outcome_name}: {e}"
                    )
                    all_checks_passed = False

    if all_checks_passed:
        print("✅ All checks passed! PIDs and predictions are consistent.")
        return True
    else:
        print("❌ Some checks failed. See errors above.")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate consistency between fold PIDs and predictions"
    )
    parser.add_argument("finetune_dir", help="Path to finetune directory")
    args = parser.parse_args()

    success = main(args.finetune_dir)
    sys.exit(0 if success else 1)
