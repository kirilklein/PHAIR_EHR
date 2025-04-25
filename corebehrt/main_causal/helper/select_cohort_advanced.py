import logging
import os
from os.path import join
from typing import List

import pandas as pd
import torch

from corebehrt.constants.cohort import CRITERIA_DEFINITIONS, DELAYS
from corebehrt.constants.data import PID_COL
from corebehrt.constants.paths import (
    FOLDS_FILE,
    INDEX_DATES_FILE,
    PID_FILE,
    TEST_PIDS_FILE,
)
from corebehrt.functional.features.split import create_folds, split_test
from corebehrt.functional.io_operations.meds import iterate_splits_and_shards
from corebehrt.modules.cohort_handling.advanced.extract import CohortExtractor

logger = logging.getLogger("select_cohort_advanced")


def extract_and_save_criteria(
    meds_path: str,
    index_dates: pd.DataFrame,
    cfg: dict,
    save_path: str,
    splits: list[str],
    pids: List[int] = None,
) -> pd.DataFrame:
    """Extracts criteria from medical event data and saves the results to a CSV file."""

    if CRITERIA_DEFINITIONS not in cfg:
        raise ValueError(f"Configuration missing required key: {CRITERIA_DEFINITIONS}")

    if pids is not None:
        index_dates = index_dates[index_dates[PID_COL].isin(pids)]

    criteria_dfs = []
    for shard_path in iterate_splits_and_shards(meds_path, splits):
        logger.info(f"Processing shard: {os.path.basename(shard_path)}")
        shard = pd.read_parquet(shard_path)
        if pids is not None:
            logger.info(f"Filtering shard for {len(pids)} patients")
            shard = shard[shard[PID_COL].isin(pids)]
        logger.info(f"Extracting criteria for {shard[PID_COL].nunique()} patients")
        cohort_extractor = CohortExtractor(
            cfg.get(CRITERIA_DEFINITIONS),
            cfg.get(DELAYS),
        )
        criteria_df = cohort_extractor.extract(
            shard,
            index_dates,
        )
        criteria_dfs.append(criteria_df)
    criteria_df = pd.concat(criteria_dfs)
    criteria_df.to_csv(join(save_path, "criteria_flags.csv"), index=False)
    return criteria_df


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
