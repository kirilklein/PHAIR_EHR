import os
from os.path import join
from typing import List
import pandas as pd
import torch

from corebehrt.constants.cohort import CODE_PATTERNS, CRITERIA_DEFINITIONS, DELAYS
from corebehrt.constants.paths import (
    FOLDS_FILE,
    TEST_PIDS_FILE,
)
from corebehrt.constants.data import PID_COL
from corebehrt.constants.paths import INDEX_DATES_FILE, PID_FILE
from corebehrt.functional.features.split import create_folds, split_test
from corebehrt.functional.io_operations.meds import iterate_splits_and_shards
from corebehrt.modules.cohort_handling.advanced.data.patient import (
    patients_to_dataframe,
)
from corebehrt.modules.cohort_handling.advanced.extract import extract_patient_criteria
import logging

logger = logging.getLogger("select_cohort_advanced")


def extract_and_save_criteria(
    meds_path: str,
    index_dates: pd.DataFrame,
    cfg: dict,
    save_path: str,
    splits: list[str],
) -> pd.DataFrame:
    """Extracts criteria from medical event data and saves the results to a CSV file."""

    if CRITERIA_DEFINITIONS not in cfg:
        raise ValueError(f"Configuration missing required key: {CRITERIA_DEFINITIONS}")

    patients = {}
    for shard_path in iterate_splits_and_shards(meds_path, splits):
        print(f"Processing shard: {os.path.basename(shard_path)}")
        shard = pd.read_parquet(shard_path)
        patients.update(
            extract_patient_criteria(
                shard,
                index_dates,
                cfg.get(CRITERIA_DEFINITIONS),
                cfg.get(CODE_PATTERNS),
                cfg.get(DELAYS),
            )
        )

    df = patients_to_dataframe(patients)
    df.to_csv(join(save_path, "criteria_flags.csv"))
    return df


def split_and_save(
    filtered_pids: List[int],
    save_path: str,
    test_ratio: float,
    cv_folds: int,
    val_ratio: float,
    seed: int = 42,
) -> None:
    """Split the cohort into test and train/val, create folds if applicable  and save the results."""
    logger.info("Splitting test and train/val")
    train_val_pids, test_pids = split_test(filtered_pids, test_ratio)
    if len(test_pids) > 0:
        torch.save(test_pids, join(save_path, TEST_PIDS_FILE))

    if len(train_val_pids) > 0:
        logger.info("Creating folds")
        folds = create_folds(
            train_val_pids,
            cv_folds,
            seed,
            val_ratio,
        )
        torch.save(folds, join(save_path, FOLDS_FILE))


def filter_and_save_cohort(
    df: pd.DataFrame, index_dates: pd.DataFrame, save_path: str
) -> List[int]:
    """Filter and save the filtered pids and filtered index dates."""
    filtered_pids = df[PID_COL].tolist()
    torch.save(filtered_pids, join(save_path, PID_FILE))
    filtered_index_dates = index_dates[index_dates[PID_COL].isin(filtered_pids)]
    filtered_index_dates.to_csv(join(save_path, INDEX_DATES_FILE))
    return filtered_pids
