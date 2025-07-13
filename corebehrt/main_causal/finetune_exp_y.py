import logging
import os
from os.path import join
from corebehrt.modules.setup.config import Config

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch

from corebehrt.constants.causal.data import EXPOSURE, OUTCOME
from corebehrt.constants.paths import FOLDS_FILE, PREPARED_ALL_PATIENTS, TEST_PIDS_FILE
from corebehrt.functional.setup.args import get_args
from corebehrt.main.helper.finetune_cv import check_for_overlap
from corebehrt.main_causal.helper.finetune_exp_y import cv_loop
from corebehrt.modules.monitoring.causal.metric_aggregation import (
    compute_and_save_scores_mean_std,
)
from corebehrt.modules.preparation.causal.dataset import CausalPatientDataset
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.directory import DirectoryPreparer

CONFIG_PATH = "./corebehrt/configs/causal/finetune/ft_exp_y.yaml"


def main_finetune(config_path):
    cfg = load_config(config_path)

    # Setup directories
    DirectoryPreparer(cfg).setup_finetune()

    # Logger
    logger = logging.getLogger("finetune_exp_y")

    loaded_data = torch.load(join(cfg.paths.prepared_data, PREPARED_ALL_PATIENTS))
    data = CausalPatientDataset(loaded_data)
    test_data = CausalPatientDataset([])

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
    save_combined_scores(cfg, len(folds), outcome_names, "val")
    if len(test_data) > 0:
        save_combined_scores(cfg, len(folds), outcome_names, "test")

    logger.info("Done")


def handle_folds(cfg: Config, test_pids: list, logger: logging.Logger) -> list:
    """
    Load folds and check for overlap with test pids.
    Save folds to model directory.
    Return folds.
    """
    folds_path = join(cfg.paths.prepared_data, FOLDS_FILE)
    folds = torch.load(folds_path)
    check_for_overlap(folds, test_pids, logger)
    n_folds = len(folds)
    logger.info(f"Using {n_folds} predefined folds")
    torch.save(folds, join(cfg.paths.model, FOLDS_FILE))
    return folds


def save_combined_scores(
    cfg: Config, n_folds: int, outcome_names: list, mode: str
) -> None:
    """
    Save combined scores for exposure and outcomes.
    """
    compute_and_save_scores_mean_std(
        n_folds, cfg.paths.model, mode=mode, target_type=EXPOSURE
    )
    compute_and_save_scores_mean_std(
        n_folds,
        cfg.paths.model,
        mode=mode,
        target_type=OUTCOME,
        outcome_names=outcome_names,
    )


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_finetune(args.config_path)
