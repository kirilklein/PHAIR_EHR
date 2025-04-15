import argparse
import os
from os.path import join

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import KFold

from corebehrt.constants.causal.data import PROBAS, TARGETS
from corebehrt.constants.data import PID_COL
from corebehrt.constants.paths import FINETUNE_CFG


def generate_mock_finetune_output(
    output_dir: str, n_folds: int = 3, n_subjects: int = 100, random_seed: int = 42
):
    """
    Generate mock finetuning output files for testing calibration.
    Uses sklearn's KFold for proper cross-validation splits.
    """
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Create all subject IDs
    all_pids = np.arange(n_subjects)

    # Initialize K-fold splitter
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    # Generate data for all subjects once
    all_probas = np.random.beta(5, 2, size=n_subjects).astype(np.float32)
    all_targets = np.random.binomial(
        1, all_probas + np.random.normal(0, 0.01, size=n_subjects)
    ).astype(np.float32)

    all_val_pids = []
    all_val_probas = []
    all_val_targets = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_pids), 1):
        fold_dir = join(output_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        checkpoints_dir = join(fold_dir, "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)

        # Get train/val split for this fold
        train_pids = all_pids[train_idx]
        val_pids = all_pids[val_idx]
        all_val_pids.extend(val_pids.tolist())
        # Get corresponding probas and targets for validation set
        val_probas = all_probas[val_idx]
        val_targets = all_targets[val_idx]
        all_val_probas.extend(val_probas.tolist())
        all_val_targets.extend(val_targets.tolist())
        print(f"Fold {fold}: {len(train_pids)} train, {len(val_pids)} validation")

        # Save pids
        torch.save(train_pids.tolist(), join(fold_dir, "train_pids.pt"))
        torch.save(val_pids.tolist(), join(fold_dir, "val_pids.pt"))

        # Save fold-specific files with correct shapes
        val_targets_reshaped = val_targets.reshape(-1, 1)
        val_probas_reshaped = val_probas.reshape(-1, 1)

        np.savez_compressed(
            join(checkpoints_dir, "targets_val_999.npz"), targets=val_targets_reshaped
        )
        np.savez_compressed(
            join(checkpoints_dir, "probas_val_999.npz"), probas=val_probas_reshaped
        )

        # this is needed by calibrate to get epoch number
        mock_model_checkpoint = torch.zeros(1)
        torch.save(
            mock_model_checkpoint, join(checkpoints_dir, "checkpoint_epoch999_end.pt")
        )

        # Save fold-specific validation data
        pd.DataFrame(
            {"subject_id": val_pids, "probas": val_probas, "targets": val_targets}
        ).to_csv(join(fold_dir, "mock_validation_data.csv"), index=False)

    predictions_df = pd.DataFrame(
        {
            PID_COL: np.array(all_val_pids),
            PROBAS: np.array(all_val_probas),
            TARGETS: np.array(all_val_targets),
        }
    )

    # Save combined predictions
    predictions_df.to_csv(
        join(output_dir, "mock_predictions_and_targets.csv"), index=False
    )

    # Save empty config file
    empty_config = {}
    with open(join(output_dir, FINETUNE_CFG), "w") as f:
        yaml.dump(empty_config, f)

    print(f"Mock finetuning output generated at {output_dir} with {n_folds} folds")


#  Example usage
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate mock finetuning outputs for testing"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where mock outputs will be saved",
    )
    parser.add_argument(
        "--n_folds", type=int, default=5, help="Number of folds to generate"
    )
    parser.add_argument(
        "--n_subjects", type=int, default=100, help="Number of subjects to simulate"
    )
    parser.add_argument(
        "--random_seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    generate_mock_finetune_output(
        args.output_dir, args.n_folds, args.n_subjects, args.random_seed
    )
