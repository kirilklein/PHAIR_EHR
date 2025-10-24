import logging
import os
from os.path import join

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch

from corebehrt.constants.paths import (
    FOLDS_FILE,
    OUTCOME_NAMES_FILE,
    PREPARED_ALL_PATIENTS,
    TEST_PIDS_FILE,
)
from corebehrt.functional.setup.args import get_args
from corebehrt.functional.features.split import create_folds
from corebehrt.main.helper.finetune_cv import check_for_overlap
from corebehrt.main_causal.helper.finetune_exp_y import cv_loop
from corebehrt.modules.monitoring.causal.metric_aggregation import (
    compute_and_save_combined_scores_mean_std,
)

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
    Load or regenerate folds and check for overlap with test pids.
    Save folds to model directory.
    Return folds.

    If cfg.data.reshuffle_seed is provided, regenerates folds with that seed
    instead of loading from prepared_data. This allows running multiple
    experiments with different fold splits without re-preparing data.
    """
    reshuffle_seed = cfg.data.get("reshuffle_seed", None)

    if reshuffle_seed is not None:
        # Regenerate folds with new seed
        logger.info(f"Regenerating folds with reshuffle_seed={reshuffle_seed}")

        # Load prepared data to get all PIDs
        loaded_data = torch.load(join(cfg.paths.prepared_data, PREPARED_ALL_PATIENTS))
        all_pids = [p.pid for p in loaded_data]

        # Remove test pids if they exist
        train_val_pids = [pid for pid in all_pids if pid not in test_pids]

        # Create new folds
        folds = create_folds(
            train_val_pids,
            cfg.data.get("cv_folds", 5),
            reshuffle_seed,
            cfg.data.get("val_ratio", 0.2),
        )
        n_folds = len(folds)
        logger.info(f"Created {n_folds} new folds with seed {reshuffle_seed}")
    else:
        # Load existing folds from prepared data
        folds_path = join(cfg.paths.prepared_data, FOLDS_FILE)
        folds = torch.load(folds_path)
        n_folds = len(folds)
        logger.info(f"Using {n_folds} predefined folds from prepared data")

    check_for_overlap(folds, test_pids, logger)
    torch.save(folds, join(cfg.paths.model, FOLDS_FILE))
    return folds


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_finetune(args.config_path)
