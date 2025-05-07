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
)
from corebehrt.constants.data import CONCEPT_COL, PID_COL
from corebehrt.constants.paths import (
    FOLDS_FILE,
    INDEX_DATES_FILE,
    PID_FILE,
    TEST_PIDS_FILE,
)
from corebehrt.modules.cohort_handling.advanced.validator import CriteriaValidator
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
        splits (list[str]): List of splits to process
        pids (List[int], optional): List of patient IDs to filter. Defaults to None.

    Returns:
        pd.DataFrame: Combined DataFrame containing extracted criteria for all patients

    Raises:
        ValueError: If no criteria data is found for any patients
    """
    cohort_extractor = CohortExtractor(
        criteria_definitions_cfg,
    )

    criteria_dfs = []
    total_patients_processed = 0

    for shard_path in iterate_splits_and_shards(meds_path, splits):
        logger.info(
            f"========== Processing shard: {os.path.basename(shard_path)} =========="
        )
        shard = pd.read_parquet(shard_path)
        shard[CONCEPT_COL] = shard[CONCEPT_COL].astype("category")

        if pids is not None:
            logger.info(f"Filtering shard for {len(pids)} patients")
            shard = shard[shard[PID_COL].isin(pids)]
            if shard.empty:
                logger.warning(f"No matching patients found in shard {shard_path}")
                continue

        if shard.empty:
            logger.warning(f"Empty shard found: {shard_path}")
            continue

        logger.info(f"Extracting criteria for {shard[PID_COL].nunique()} patients")
        criteria_df = cohort_extractor.extract(
            shard,
            index_dates,
        )

        if not criteria_df.empty:
            criteria_dfs.append(criteria_df)
            total_patients_processed += len(criteria_df)
        else:
            logger.warning(
                f"No criteria matched for any patients in shard {shard_path}"
            )

    if not criteria_dfs:
        error_msg = "No criteria data found for any patients"
        if pids is not None:
            error_msg += f" in the provided list of {len(pids)} patient IDs"
        raise ValueError(error_msg)

    logger.info(
        f"Successfully processed criteria for {total_patients_processed} patients"
    )
    return pd.concat(criteria_dfs)


def check_criteria_cfg(cfg: dict) -> None:
    """Validate the cohort selection criteria configuration.

    This function performs comprehensive validation of the configuration dictionary
    used for cohort selection, including:
    - Presence of required criteria definitions
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
    validator = CriteriaValidator(criteria_definitions_cfg)
    validator.validate()


def check_inclusion_exclusion(validator: CriteriaValidator, cfg: dict) -> None:
    """Check inclusion and exclusion expressions."""
    if INCLUSION in cfg:
        validator.validate_expression(INCLUSION, cfg[INCLUSION])
    else:
        raise ValueError(f"Configuration missing required key: {INCLUSION}")
    if EXCLUSION in cfg:
        validator.validate_expression(EXCLUSION, cfg[EXCLUSION])
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
