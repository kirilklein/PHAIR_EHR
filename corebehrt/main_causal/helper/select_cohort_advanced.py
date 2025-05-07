"""Advanced cohort selection and extraction utilities for medical event data.

This module provides functionality for extracting and validating patient cohorts based on
medical event criteria. It supports:
- Complex criteria definitions with inclusion/exclusion rules
- Multi-shard data processing
- Age-based filtering
- Train/test/validation splits
- Cohort persistence

The module is designed to work with medical event data in parquet format, where each event
is associated with a patient ID, timestamp, and medical concept code.
"""

import logging
import os
from os.path import join
from typing import List

import pandas as pd
import torch

from corebehrt.constants.cohort import (
    CRITERIA_DEFINITIONS,
    EXCLUSION,
    INCLUSION,
    UNIQUE_CODE_LIMITS,
)
from corebehrt.constants.data import CONCEPT_COL, PID_COL
from corebehrt.constants.paths import (
    FOLDS_FILE,
    INDEX_DATES_FILE,
    PID_FILE,
    TEST_PIDS_FILE,
)
from corebehrt.functional.cohort_handling.advanced.checks import (
    check_criteria_definitions,
    check_expression,
    check_unique_code_limits,
)
from corebehrt.functional.features.split import create_folds, split_test
from corebehrt.functional.io_operations.meds import iterate_splits_and_shards
from corebehrt.modules.cohort_handling.advanced.extract import CohortExtractor

logger = logging.getLogger("select_cohort_advanced")


def extract_criteria_from_shards(
    meds_path: str,
    index_dates: pd.DataFrame,
    criteria_definitions_cfg: dict,
    splits: list[str],
    pids: List[int] = None,
) -> pd.DataFrame:
    """Extract criteria from medical event data across multiple shards.

    Args:
        meds_path (str): Path to the medical events data
        index_dates (pd.DataFrame): DataFrame containing index dates for patients
        criteria_definitions_cfg (dict): Configuration for criteria definitions
        delays_cfg (dict): Configuration for delays
        splits (list[str]): List of splits to process
        pids (List[int], optional): List of patient IDs to filter. Defaults to None.

    Returns:
        pd.DataFrame: Combined DataFrame containing extracted criteria for all patients
    """
    cohort_extractor = CohortExtractor(
        criteria_definitions_cfg,
    )

    criteria_dfs = []
    for shard_path in iterate_splits_and_shards(meds_path, splits):
        logger.info(
            f"========== Processing shard: {os.path.basename(shard_path)} =========="
        )
        shard = pd.read_parquet(shard_path)
        shard[CONCEPT_COL] = shard[CONCEPT_COL].astype("category")

        if pids is not None:
            logger.info(f"Filtering shard for {len(pids)} patients")
            shard = shard[shard[PID_COL].isin(pids)]

        logger.info(f"Extracting criteria for {shard[PID_COL].nunique()} patients")
        criteria_df = cohort_extractor.extract(
            shard,
            index_dates,
        )
        criteria_dfs.append(criteria_df)

    return pd.concat(criteria_dfs)


def check_criteria_cfg(cfg: dict) -> None:
    """Validate the cohort selection criteria configuration.

    This function performs comprehensive validation of the configuration dictionary
    used for cohort selection, including:
    - Presence of required criteria definitions
    - Validation of criteria definitions and delays configuration
    - Checking inclusion and exclusion expressions
    - Validation of unique code limits if specified

    Args:
        cfg (dict): Configuration dictionary containing cohort selection criteria

    Raises:
        ValueError: If required configuration keys are missing or if validation fails
    """
    if CRITERIA_DEFINITIONS not in cfg:
        raise ValueError(f"Configuration missing required key: {CRITERIA_DEFINITIONS}")

    logger.info("Checking criteria definitions")
    criteria_definitions_cfg = cfg.get(CRITERIA_DEFINITIONS)
    check_criteria_definitions(criteria_definitions_cfg)

    logger.info("Checking inclusion and exclusion expressions")
    criteria_names = list(criteria_definitions_cfg.keys())
    if UNIQUE_CODE_LIMITS in cfg:
        logger.info("Checking unique code limits")
        check_unique_code_limits(cfg.get(UNIQUE_CODE_LIMITS), criteria_names)


def check_inclusion_exclusion(cfg: dict) -> None:
    """Check inclusion and exclusion expressions."""
    criteria_definitions_cfg = cfg.get(CRITERIA_DEFINITIONS)
    criteria_names = list(criteria_definitions_cfg.keys())
    if INCLUSION in cfg:
        check_expression(cfg.get(INCLUSION), criteria_names)
    else:
        raise ValueError(f"Configuration missing required key: {INCLUSION}")
    if EXCLUSION in cfg:
        check_expression(cfg.get(EXCLUSION), criteria_names)
    else:
        raise ValueError(f"Configuration missing required key: {EXCLUSION}")


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
