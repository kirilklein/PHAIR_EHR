"""
Baseline training script for causal inference using CatBoost with one-hot/multi-hot encoding.

This script is an updated version that uses CatBoost, includes robust parameter handling
with defaults, and integrates Optuna for automated hyperparameter tuning. It includes age
features and supports cross-validation with the same directory structure and configuration system.
"""

import logging
import os
from datetime import datetime
from os.path import join
from typing import List, Dict, NamedTuple

import optuna
import pandas as pd
import torch
import numpy as np

from corebehrt.constants.paths import (
    PREPARED_ALL_PATIENTS,
    TEST_PIDS_FILE,
)
from corebehrt.constants.data import PID_COL
from corebehrt.constants.causal.data import (
    EXPOSURE,
    PS_COL,
    EXPOSURE_COL,
    OUTCOME_COL,
    PROBAS,
    CF_PROBAS,
)
from corebehrt.constants.causal.paths import COMBINED_PREDICTIONS_FILE
from corebehrt.functional.io_operations.load import load_vocabulary
from corebehrt.functional.setup.args import get_args
from corebehrt.main_causal.helper.train_baseline import (
    handle_folds,
    nested_cv_loop,
    FoldPredictionData,
)
from corebehrt.modules.preparation.causal.dataset import CausalPatientDataset
from corebehrt.modules.setup.causal.directory import CausalDirectoryPreparer
from corebehrt.modules.setup.config import load_config

CONFIG_PATH = "./corebehrt/configs/causal/finetune/baseline.yaml"
ROUND_DIGIT = 7


class FoldPredictions(NamedTuple):
    """Container for predictions from a single fold."""

    pids: List[int]
    predictions: Dict[str, np.ndarray]  # target_name -> predictions
    targets: Dict[str, np.ndarray]  # target_name -> actual values


def save_nested_cv_summary(
    all_results: List[pd.DataFrame], baseline_folder: str
) -> None:
    """Combines all target results and saves the final summary."""
    logger = logging.getLogger("save_summary")

    if not all_results:
        logger.warning("No nested CV results to save.")
        return

    # Combine all results into a single DataFrame
    all_results_df = pd.concat(all_results, ignore_index=True)
    all_results_df = all_results_df.sort_values(by="mean_auc", ascending=False)

    # Save the combined report
    scores_folder = join(baseline_folder, "scores")
    os.makedirs(scores_folder, exist_ok=True)
    final_report_path = join(
        scores_folder, f"scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    all_results_df.to_csv(final_report_path, index=False)

    logger.info("===== Nested CV Final Summary =====")
    logger.info(f"\n{all_results_df.to_string(index=False)}")
    logger.info(f"Final summary report saved to {final_report_path}")


def save_combined_predictions(
    prediction_storage: List[FoldPredictionData],
    baseline_folder: str,
    outcome_names: List[str],
) -> None:
    """Combines predictions from all folds and saves in the same format as finetune_exp_y.py."""
    logger = logging.getLogger("save_predictions")

    if not prediction_storage:
        logger.warning("No fold predictions to save.")
        return

    # Group predictions by fold and organize by target
    fold_data = {}  # fold_idx -> {target_name: (pids, predictions, targets)}

    for pred_data in prediction_storage:
        fold_idx = pred_data.fold_idx
        if fold_idx not in fold_data:
            fold_data[fold_idx] = {}

        fold_data[fold_idx][pred_data.target_name] = (
            pred_data.pids,
            pred_data.predictions,
            pred_data.targets,
        )

    # Combine all folds
    all_data = []

    for fold_idx in sorted(fold_data.keys()):
        fold_targets = fold_data[fold_idx]

        # Get PIDs from the first available target (should be consistent across targets)
        first_target = next(iter(fold_targets.keys()))
        pids, _, _ = fold_targets[first_target]

        # Create row data for each patient in this fold
        for i, pid in enumerate(pids):
            row_data = {PID_COL: pid}

            # Add exposure data (propensity score and target)
            if EXPOSURE in fold_targets:
                _, predictions, targets = fold_targets[EXPOSURE]
                row_data[PS_COL] = predictions[i]
                row_data[EXPOSURE_COL] = int(targets[i])

            # Add outcome data
            for outcome_name in outcome_names:
                if outcome_name in fold_targets:
                    _, predictions, targets = fold_targets[outcome_name]
                    row_data[f"{PROBAS}_{outcome_name}"] = predictions[i]
                    row_data[f"{OUTCOME_COL}_{outcome_name}"] = int(targets[i])
                    # For baseline, we use the same prediction as counterfactual
                    row_data[f"{CF_PROBAS}_{outcome_name}"] = predictions[i]

            all_data.append(row_data)

    # Create DataFrame and save
    combined_df = pd.DataFrame(all_data)
    output_path = join(baseline_folder, COMBINED_PREDICTIONS_FILE)
    combined_df = combined_df.round(ROUND_DIGIT)
    combined_df.to_csv(output_path, index=False)

    logger.info(f"Combined predictions saved to: {output_path}")
    logger.info(f"Combined dataframe shape: {combined_df.shape}")
    logger.info(f"Columns: {list(combined_df.columns)}")


def main_baseline(config_path: str):
    """Main baseline training function using Nested CV."""
    cfg = load_config(config_path)

    # Setup directories
    CausalDirectoryPreparer(cfg).setup_train_baseline()

    # Logger
    logger = logging.getLogger("train_baseline")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    patients = torch.load(join(cfg.paths.prepared_data, PREPARED_ALL_PATIENTS))
    vocab = load_vocabulary(cfg.paths.prepared_data)
    data = CausalPatientDataset(patients, vocab)

    if os.path.exists(join(cfg.paths.prepared_data, TEST_PIDS_FILE)):
        logger.warning(
            "A test PID file exists, but it will be IGNORED in Nested CV mode."
        )

    # Use folds from prepared data
    folds = handle_folds(cfg, logger)
    all_pids_in_folds = {
        pid for fold in folds for split in fold.values() for pid in split
    }
    cv_data = data.filter_by_pids(list(all_pids_in_folds))

    # Run the main Nested CV loop and collect results and predictions
    all_results, prediction_storage = nested_cv_loop(cfg, logger, cv_data, folds)

    # Save only the combined final summary
    save_nested_cv_summary(all_results, cfg.paths.model)

    # Save combined predictions in the same format as finetune_exp_y.py
    outcome_names = data.get_outcome_names()
    save_combined_predictions(prediction_storage, cfg.paths.model, outcome_names)

    logger.info("Baseline training with Nested CV completed.")


if __name__ == "__main__":
    # To avoid cluttered logs from Optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    args = get_args(CONFIG_PATH)
    main_baseline(args.config_path)
