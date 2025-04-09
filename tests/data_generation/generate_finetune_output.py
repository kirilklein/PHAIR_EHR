import argparse
import os
from os.path import join

import numpy as np
import pandas as pd
import torch
import yaml

from corebehrt.constants.causal.data import PROBAS, TARGETS
from corebehrt.constants.data import PID_COL
from corebehrt.constants.paths import FINETUNE_CFG


def generate_mock_finetune_output(
    output_dir: str,
    n_folds: int = 3,
    n_subjects: int = 100,
    random_seed: int = 42
):
    """
    Generate mock finetuning output files for testing calibration.
    
    This function creates a directory structure mimicking the output of finetuning:
    - fold_i/
      - train_pids.pt
      - val_pids.pt
      - checkpoints/
        - targets_val999.npz
        - probas_val999.npz
    
    Args:
        output_dir (str): Directory where mock outputs will be saved
        n_folds (int): Number of folds to generate
        n_subjects (int): Total number of subjects to simulate
        random_seed (int): Random seed for reproducibility
    """

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Create all subject IDs - ensure at least 5 subjects per fold
    min_subjects = max(n_subjects, n_folds * 5)
    all_pids = list(range(1, min_subjects + 1))

    # Instead of dividing into non-overlapping segments, 
    # use a different approach with some overlapping patients
    # but ensure no validation set is empty
    all_val_pids = []
    all_val_predictions = []
    all_val_targets = []
    for fold in range(1, n_folds + 1):
        # Create fold directory
        fold_dir = join(output_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        # Create checkpoints directory
        checkpoints_dir = join(fold_dir, "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)

        # For more realistic distribution, we'll use a different approach
        # Shuffle all pids and take first 80% for training, next 20% for validation
        np.random.shuffle(all_pids)

        # Ensure we have at least 5 samples in validation
        train_size = int(len(all_pids) * 0.8)
        train_pids = all_pids[:train_size]
        val_pids = all_pids[train_size:train_size + max(5, len(all_pids) - train_size)]

        print(f"Fold {fold}: {len(train_pids)} train, {len(val_pids)} validation")

        # Save pids
        torch.save(train_pids, join(fold_dir, "train_pids.pt"))
        torch.save(val_pids, join(fold_dir, "val_pids.pt"))
        all_val_pids.extend(val_pids)
        # Generate targets - binary outcomes (0 or 1)
        # Use float32 to match what the real model would output
        val_targets = np.random.randint(0, 2, size=len(val_pids)).astype(np.float32)
        all_val_targets.extend(val_targets)
        # Generate probabilities - values between 0 and 1
        val_probas = np.random.random(size=len(val_pids)).astype(np.float32)
        all_val_predictions.extend(val_probas)

        # Reshape to match expected dimensions (N, 1)
        val_targets = val_targets.reshape(-1, 1)
        val_probas = val_probas.reshape(-1, 1)

        # Save targets and probabilities
        np.savez(join(checkpoints_dir, "targets_val999.npz"), val_targets)
        np.savez(join(checkpoints_dir, "probas_val999.npz"), val_probas)

        # Write a simple CSV with subject_id, probas, targets to verify the data
        with open(join(fold_dir, "validation_data.csv"), "w") as f:
            f.write("subject_id,probas,targets\n")
            for i in range(len(val_pids)):
                f.write(f"{val_pids[i]},{val_probas[i][0]},{val_targets[i][0]}\n")

    print(f"Mock finetuning output generated at {output_dir} with {n_folds} folds")
    # it will also expect a finetune config file
    empty_config = {}
    with open(join(output_dir, FINETUNE_CFG), "w") as f:
        yaml.dump(empty_config, f)
    # Save combined predictions and targets with subject IDs for later comparison
    predictions_df = pd.DataFrame(
        {PID_COL: all_val_pids, PROBAS: all_val_predictions, TARGETS: all_val_targets}
    )
    predictions_df.to_csv(join(output_dir, "predictions_and_targets.csv"), index=False)



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
