import logging
import os
from os.path import join

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import random
import time

from corebehrt.constants.paths import (
    FOLDS_FILE,
    OUTCOME_NAMES_FILE,
    PREPARED_ALL_PATIENTS,
    TEST_PIDS_FILE,
)
from corebehrt.functional.setup.args import get_args
from corebehrt.main.helper.finetune_cv import check_for_overlap
from corebehrt.main_causal.helper.finetune_exp_y import cv_loop
from corebehrt.modules.monitoring.causal.metric_aggregation import (
    compute_and_save_combined_scores_mean_std,
)
from corebehrt.constants.data import TRAIN_KEY, VAL_KEY

from corebehrt.modules.preparation.causal.dataset import CausalPatientDataset
from corebehrt.functional.io_operations.load import load_vocabulary
from corebehrt.modules.setup.config import Config, load_config
from corebehrt.modules.setup.directory import DirectoryPreparer
from corebehrt.modules.setup.causal.prediction_accumulator import PredictionAccumulator

CONFIG_PATH = "./corebehrt/configs/causal/finetune/ft_exp_y.yaml"


def main_finetune(config_path):
    cfg = load_config(config_path)

    # Setup directories
    DirectoryPreparer(cfg).setup_finetune()

    # Logger
    logger = logging.getLogger("finetune_exp_y")

    loaded_data = torch.load(join(cfg.paths.prepared_data, PREPARED_ALL_PATIENTS))
    vocab = load_vocabulary(cfg.paths.prepared_data)
    data = CausalPatientDataset(loaded_data, vocab)
    test_data = CausalPatientDataset([], vocab)

    # Initialize test and train/val pid lists
    test_pids = []
    train_val_pids = data.get_pids()

    # If evaluation is desired, then:
    if cfg.get("evaluate", False):
        if os.path.exists(join(cfg.paths.prepared_data, TEST_PIDS_FILE)):
            test_pids = torch.load(join(cfg.paths.prepared_data, TEST_PIDS_FILE))
            test_data = data.filter_by_pids(test_pids)

    # Use folds from prepared data
    folds = handle_folds(cfg, test_pids, logger)
    train_val_data = data.filter_by_pids(train_val_pids)
    cv_loop(
        cfg,
        logger,
        cfg.paths.model,
        train_val_data,
        folds,
        test_data,
    )

    outcome_names = data.get_outcome_names()
    # Save combined predictions
    PredictionAccumulator(
        cfg.paths.model, outcome_names
    ).accumulate_and_save_predictions()

    # Save outcome names to the model directory
    torch.save(outcome_names, join(cfg.paths.model, OUTCOME_NAMES_FILE))
    logger.info(f"Saved outcome names: {outcome_names}")

    compute_and_save_combined_scores_mean_std(
        len(folds), cfg.paths.model, mode="val", outcome_names=outcome_names
    )

    if len(test_data) > 0:
        compute_and_save_combined_scores_mean_std(
            len(folds), cfg.paths.model, mode="test", outcome_names=outcome_names
        )

    logger.info("Done")


def handle_folds(cfg: Config, test_pids: list, logger: logging.Logger) -> list:
    """
    Load folds and optionally reshuffle PIDs across them.
    Save folds to model directory.
    Return folds.

    If cfg.data.reshuffle is True, shuffles PIDs across the loaded folds
    using cfg.data.reshuffle_seed (or auto-generated seed if not provided).
    This allows running multiple experiments with different fold splits
    without re-preparing data.
    """
    # Always load folds from prepared data
    folds_path = join(cfg.paths.prepared_data, FOLDS_FILE)
    folds = torch.load(folds_path)
    n_folds = len(folds)
    logger.info(f"Loaded {n_folds} folds from prepared data")
    data_cfg = cfg.get("data", {})
    # Check if we should reshuffle
    reshuffle = data_cfg.get("reshuffle", False)

    if reshuffle:
        # Get or generate seed
        reshuffle_seed = data_cfg.get("reshuffle_seed", None)
        if reshuffle_seed is None:
            # Auto-generate time-based seed
            reshuffle_seed = int(time.time() * 1000) % (2**32)

        logger.info(f"Reshuffling folds with seed={reshuffle_seed}")

        # Collect all PIDs from all folds
        random.seed(reshuffle_seed)

        all_train_pids = []
        all_val_pids = []

        for fold in folds:
            all_train_pids.extend(fold[TRAIN_KEY])
            all_val_pids.extend(fold[VAL_KEY])

        # Combine and shuffle all PIDs
        all_pids = all_train_pids + all_val_pids
        random.shuffle(all_pids)

        # Redistribute PIDs back into folds with same structure
        pid_idx = 0
        for fold in folds:
            n_train = len(fold[TRAIN_KEY])
            n_val = len(fold[VAL_KEY])

            fold[TRAIN_KEY] = all_pids[pid_idx : pid_idx + n_train]
            pid_idx += n_train

            fold[VAL_KEY] = all_pids[pid_idx : pid_idx + n_val]
            pid_idx += n_val

        logger.info(f"Reshuffled {len(all_pids)} PIDs across {n_folds} folds")
    else:
        logger.info("Using folds as loaded (no reshuffling)")

    check_for_overlap(folds, test_pids, logger)
    torch.save(folds, join(cfg.paths.model, FOLDS_FILE))
    return folds


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_finetune(args.config_path)
