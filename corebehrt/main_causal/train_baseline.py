"""
Baseline training script for causal inference using CatBoost with one-hot/multi-hot encoding.

This script is an updated version that uses CatBoost, includes robust parameter handling
with defaults, and integrates Optuna for automated hyperparameter tuning. It includes age
features and supports cross-validation with the same directory structure and configuration system.
"""

import logging
import os
from os.path import join

import optuna
import torch

from corebehrt.constants.paths import (
    PREPARED_ALL_PATIENTS,
    OUTCOME_NAMES_FILE,
    TEST_PIDS_FILE,
)
from corebehrt.functional.io_operations.load import load_vocabulary
from corebehrt.functional.setup.args import get_args
from corebehrt.main_causal.helper.train_baseline import (
    handle_folds,
    nested_cv_loop,
    save_nested_cv_summary,
    save_combined_predictions,
)
from corebehrt.modules.preparation.causal.dataset import CausalPatientDataset
from corebehrt.modules.setup.causal.directory import CausalDirectoryPreparer
from corebehrt.modules.setup.config import load_config

CONFIG_PATH = "./corebehrt/configs/causal/finetune/baseline.yaml"


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
    prepared_data_path: str = cfg.paths.prepared_data
    if not os.path.exists(prepared_data_path):
        raise FileNotFoundError(
            f"Prepared data path {prepared_data_path} does not exist"
        )

    patients = torch.load(join(prepared_data_path, PREPARED_ALL_PATIENTS))
    vocab = load_vocabulary(prepared_data_path)
    data = CausalPatientDataset(patients, vocab)

    if os.path.exists(join(prepared_data_path, TEST_PIDS_FILE)):
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
    model_path = cfg.paths.model
    save_nested_cv_summary(all_results, model_path)

    # Save combined predictions in the same format as finetune_exp_y.py
    outcome_names = data.get_outcome_names()
    torch.save(outcome_names, join(model_path, OUTCOME_NAMES_FILE))
    save_combined_predictions(prediction_storage, model_path, outcome_names)

    logger.info("Baseline training with Nested CV completed.")


if __name__ == "__main__":
    # To avoid cluttered logs from Optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    args = get_args(CONFIG_PATH)
    main_baseline(args.config_path)
