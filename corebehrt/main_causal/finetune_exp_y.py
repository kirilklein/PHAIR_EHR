import logging
import os
from os.path import join

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import time

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


def validate_folds(folds: list, expected_pids: set, logger: logging.Logger) -> None:
    """
    Validate fold structure for correctness.

    Checks:
    - All PIDs are present (no loss)
    - Each fold has unique PIDs (no duplicates within)
    - Validation sets don't overlap across folds
    - Train/val PIDs sum to total PIDs in each fold
    """
    all_val_pids = set()

    for i, fold in enumerate(folds):
        train_pids = set(fold[TRAIN_KEY])
        val_pids = set(fold[VAL_KEY])

        # Check: No duplicates within fold
        assert len(train_pids) == len(fold[TRAIN_KEY]), (
            f"Fold {i}: Duplicate train PIDs"
        )
        assert len(val_pids) == len(fold[VAL_KEY]), f"Fold {i}: Duplicate val PIDs"

        # Check: No overlap between train and val
        assert train_pids.isdisjoint(val_pids), f"Fold {i}: Train/val overlap"

        # Check: Train + val = all PIDs
        fold_total = train_pids | val_pids
        assert fold_total == expected_pids, f"Fold {i}: Missing or extra PIDs"

        # Track validation PIDs across folds
        assert val_pids.isdisjoint(all_val_pids), (
            f"Fold {i}: Val PIDs overlap with other folds"
        )
        all_val_pids.update(val_pids)

    # Check: All PIDs appear in exactly one validation set
    assert all_val_pids == expected_pids, "Not all PIDs covered in validation sets"
    logger.info(
        f"✓ Folds validated: {len(folds)} folds, {len(expected_pids)} unique PIDs"
    )


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

    # Extract unique PIDs from first fold (each fold contains all PIDs split into train/val)
    all_pids = folds[0][TRAIN_KEY] + folds[0][VAL_KEY]
    expected_pids = set(all_pids)

    # Validate loaded folds
    logger.info("Validating loaded folds...")
    validate_folds(folds, expected_pids, logger)

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

        # Recreate folds with new seed using existing create_folds function
        # This ensures proper handling of uneven fold sizes and correct KFold splitting
        folds = create_folds(all_pids, n_folds, reshuffle_seed)

        logger.info(f"Reshuffled {len(all_pids)} unique PIDs across {n_folds} folds")

        # Validate reshuffled folds
        logger.info("Validating reshuffled folds...")
        validate_folds(folds, expected_pids, logger)
    else:
        logger.info("Using folds as loaded (no reshuffling)")

    check_for_overlap(folds, test_pids, logger)
    torch.save(folds, join(cfg.paths.model, FOLDS_FILE))
    return folds


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_finetune(args.config_path)
