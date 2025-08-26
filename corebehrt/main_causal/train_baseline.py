"""
Baseline training script for causal inference using CatBoost with one-hot/multi-hot encoding.

This script is an updated version that uses CatBoost, includes robust parameter handling
with defaults, and integrates Optuna for automated hyperparameter tuning. It includes age
features and supports cross-validation with the same directory structure and configuration system.
"""

import logging
import os
from os.path import join

import torch
import optuna


from corebehrt.constants.paths import (
    PREPARED_ALL_PATIENTS,
    TEST_PIDS_FILE,
)
from corebehrt.functional.setup.args import get_args
import pandas as pd
import glob
from corebehrt.modules.preparation.causal.dataset import CausalPatientDataset
from corebehrt.functional.io_operations.load import load_vocabulary
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.causal.directory import CausalDirectoryPreparer
from corebehrt.main_causal.helper.train_baseline import handle_folds, nested_cv_loop

CONFIG_PATH = "./corebehrt/configs/causal/finetune/baseline.yaml"


def aggregate_nested_cv_results(baseline_folder: str) -> None:
    """Finds all nested_cv_results CSVs and combines them into a single report."""
    logger = logging.getLogger("aggregate_results")
    search_path = join(baseline_folder, "nested_cv_results_*.csv")
    result_files = glob.glob(search_path)

    if not result_files:
        logger.warning("No Nested CV result files found to aggregate.")
        return

    all_results_df = pd.concat(
        [pd.read_csv(f) for f in result_files], ignore_index=True
    )
    all_results_df = all_results_df.sort_values(by="mean_auc", ascending=False)

    # Save the combined report
    final_report_path = join(baseline_folder, "final_nested_cv_summary.csv")
    all_results_df.to_csv(final_report_path, index=False)

    logger.info("===== Nested CV Final Summary =====")
    logger.info(f"\n{all_results_df.to_string(index=False)}")
    logger.info(f"Final summary report saved to {final_report_path}")


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

    # Run the main Nested CV loop
    nested_cv_loop(cfg, logger, cfg.paths.model, cv_data, folds)

    # Aggregate the final results into a single report
    aggregate_nested_cv_results(cfg.paths.model)

    logger.info("Baseline training with Nested CV completed.")


if __name__ == "__main__":
    # To avoid cluttered logs from Optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    args = get_args(CONFIG_PATH)
    main_baseline(args.config_path)
