"""
Subpopulation fine-tuning for causal inference models.

Loads prepared data from a main run, filters to a subpopulation,
creates fresh bootstrap folds, and continues fine-tuning from the
main run's per-fold checkpoints with the encoder frozen.
"""

import logging
from os.path import join

import torch

from corebehrt.constants.paths import (
    FOLDS_FILE,
    OUTCOME_NAMES_FILE,
    PREPARED_ALL_PATIENTS,
)
from corebehrt.functional.features.split import create_folds
from corebehrt.functional.io_operations.load import load_vocabulary
from corebehrt.functional.setup.args import get_args
from corebehrt.main.helper.finetune_cv import check_for_overlap
from corebehrt.main_causal.finetune_exp_y import validate_folds
from corebehrt.main_causal.helper.finetune_exp_y import cv_loop
from corebehrt.modules.monitoring.causal.metric_aggregation import (
    compute_and_save_combined_scores_mean_std,
)
from corebehrt.modules.preparation.causal.dataset import CausalPatientDataset
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.directory import DirectoryPreparer
from corebehrt.modules.setup.causal.prediction_accumulator import PredictionAccumulator

CONFIG_PATH = "./corebehrt/configs/causal/finetune/ft_subpop.yaml"


def main_finetune_subpop(config_path):
    cfg = load_config(config_path)
    DirectoryPreparer(cfg).setup_finetune(check_pretrain=False)

    logger = logging.getLogger("finetune_subpop")

    # Load data and filter to subpopulation
    loaded_data = torch.load(join(cfg.paths.prepared_data, PREPARED_ALL_PATIENTS))
    vocab = load_vocabulary(cfg.paths.prepared_data)
    data = CausalPatientDataset(loaded_data, vocab)

    subpop_pids = torch.load(cfg.paths.subpopulation_pids)
    logger.info(f"Loaded {len(subpop_pids)} subpopulation PIDs")

    data = data.filter_by_pids(subpop_pids)
    # dropped_outcomes = data.drop_constant_outcomes()
    # if dropped_outcomes:
    #     logger.info(
    #         "Dropped %d outcome(s) with no label variation in subpopulation: %s",
    #         len(dropped_outcomes),
    #         dropped_outcomes,
    #     )
    train_val_pids = data.get_pids()
    logger.info(f"Filtered to {len(train_val_pids)} patients in prepared data")

    # Create fresh bootstrap folds from subpopulation
    data_cfg = cfg.get("data", {})
    n_folds = data_cfg.get("n_folds", 5)
    seed = data_cfg.get("seed", 42)
    bootstrap = cfg.get("bootstrap", True)

    folds = create_folds(train_val_pids, n_folds, seed, bootstrap=bootstrap)
    validate_folds(folds, set(train_val_pids), logger, bootstrap=bootstrap)
    check_for_overlap(folds, [], logger)
    torch.save(folds, join(cfg.paths.model, FOLDS_FILE))
    logger.info(f"Created {n_folds} folds (bootstrap={bootstrap}, seed={seed})")

    # Run CV loop (loads per-fold checkpoints via restart_model)
    test_data = CausalPatientDataset([], vocab)
    cv_loop(cfg, logger, cfg.paths.model, data, folds, test_data)

    # Post-processing
    outcome_names = data.get_outcome_names()
    PredictionAccumulator(
        cfg.paths.model, outcome_names
    ).accumulate_and_save_predictions()
    torch.save(outcome_names, join(cfg.paths.model, OUTCOME_NAMES_FILE))
    logger.info(f"Saved outcome names: {outcome_names}")

    compute_and_save_combined_scores_mean_std(
        len(folds), cfg.paths.model, mode="val", outcome_names=outcome_names
    )

    logger.info("Done")


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_finetune_subpop(args.config_path)
