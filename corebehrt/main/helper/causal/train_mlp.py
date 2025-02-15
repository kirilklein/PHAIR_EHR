import warnings
from datetime import datetime
from os.path import join
from typing import Dict, List, Tuple

import pandas as pd
import torch

from corebehrt.constants.data import ABSPOS_COL, TIMESTAMP_COL, VAL_KEY
from corebehrt.constants.paths import DATA_CFG, FOLDS_FILE, INDEX_DATES_FILE
from corebehrt.functional.causal.load import (
    load_encodings_and_pids_from_encoded_dir,
    load_exposure_from_predictions,
)
from corebehrt.functional.cohort_handling.outcomes import get_binary_outcomes
from corebehrt.functional.utils.time import get_abspos_from_origin_point
from corebehrt.main.helper.causal.train_mlp import (
    check_val_fold_pids,
    combine_encodings_and_exposures,
)
from corebehrt.modules.setup.config import load_config


def combine_encodings_and_exposures(
    encodings: torch.Tensor, exposures: torch.Tensor
) -> torch.Tensor:
    """Combines input features with exposure values centered at 0.5.

    Args:
        x: Input feature matrix
        exposures: Exposure values to combine with features
    Returns:
        Combined matrix of features and centered exposures
    """
    return torch.cat([encodings, exposures.unsqueeze(1) - 0.5], dim=1)


def check_val_fold_pids(folds: List[Dict], pids: List[str]):
    """Validates that validation fold PIDs are a subset of provided PIDs.

    Args:
        folds: List of dictionaries containing validation fold PIDs
        pids: List of allowed patient IDs
    """
    pids = set(pids)
    for fold in folds:
        if not set(fold[VAL_KEY]).issubset(pids):
            warnings.warn(
                "Some validation fold PIDs are not present in the provided PIDs list. Perhaps Exclusion was performed during fine-tuning."
            )


def prepare_data(
    cfg, logger
) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[dict]]:
    """Prepare data for training by loading features, temporal info and outcomes.

    Returns:
        X: Feature matrix
        y: Binary Outcome labels
        pids: Patient IDs
        folds: Cross-validation folds
    """
    # Step 1: Load and combine feature data
    logger.info("Load encodings and exposure")
    X = _prepare_feature_data(
        encoded_data_path=cfg.paths.encoded_data,
        calibrated_predictions_path=cfg.paths.calibrated_predictions,
    )

    # Step 2: Load and process temporal information
    logger.info("Load index dates and folds")
    index_dates, folds, pids = _prepare_index_dates_and_folds(
        cohort_dir=cfg.paths.cohort,
        encoded_data_path=cfg.paths.encoded_data,
    )

    # Step 3: Process outcomes
    logger.info("Load outcomes")
    y = _prepare_outcomes(
        outcomes_path=cfg.paths.outcomes,
        index_dates=index_dates,
        pids=pids,
        n_hours_start_follow_up=cfg.outcome.n_hours_start_follow_up,
        n_hours_end_follow_up=cfg.outcome.n_hours_end_follow_up,
    )

    return X, y, pids, folds


def _prepare_feature_data(
    encoded_data_path: str,
    calibrated_predictions_path: str,
) -> torch.Tensor:
    encodings, pids = load_encodings_and_pids_from_encoded_dir(encoded_data_path)
    exposure = load_exposure_from_predictions(calibrated_predictions_path, pids)
    return combine_encodings_and_exposures(encodings, exposure)


def _prepare_index_dates_and_folds(
    cohort_dir: str, encoded_data_path: str
) -> Tuple[pd.DataFrame, torch.Tensor, List[str]]:
    # Load and process index dates
    index_dates = pd.read_csv(
        join(cohort_dir, INDEX_DATES_FILE), parse_dates=[TIMESTAMP_COL]
    )
    origin_point = load_config(join(encoded_data_path, DATA_CFG)).features.origin_point
    index_dates[ABSPOS_COL] = get_abspos_from_origin_point(
        index_dates[TIMESTAMP_COL], datetime(**origin_point)
    )

    # Load and validate folds
    folds = torch.load(join(cohort_dir, FOLDS_FILE))
    _, pids = load_encodings_and_pids_from_encoded_dir(encoded_data_path)
    check_val_fold_pids(folds, pids)

    return index_dates, folds, pids


def _prepare_outcomes(
    outcomes_path: str,
    index_dates: pd.DataFrame,
    pids: List[str],
    n_hours_start_follow_up: int,
    n_hours_end_follow_up: int,
) -> torch.Tensor:
    outcomes = pd.read_csv(outcomes_path)

    binary_outcomes = get_binary_outcomes(
        index_dates,
        outcomes,
        n_hours_start_follow_up,
        n_hours_end_follow_up,
    )
    binary_outcomes = binary_outcomes.loc[pids]
    return torch.tensor(binary_outcomes.values, dtype=torch.float32)
