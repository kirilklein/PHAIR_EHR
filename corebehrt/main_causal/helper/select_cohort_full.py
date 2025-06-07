"""
Cohort Selection for Causal Studies with Index Date Matching.

This module implements a comprehensive pipeline for selecting matched cohorts in causal
inference studies using EHR data. It identifies exposed and control patients while
ensuring proper temporal alignment and applying inclusion/exclusion criteria.

Main Workflow:
1. Load patient data and identify first exposure events as index dates
2. Filter exposed patients by time windows and advanced criteria
3. Assign index dates to control patients by sampling from exposed patients (with death date validation)
4. Apply same filtering criteria to control patients
5. Combine results and save outputs with detailed statistics

Key Functions:
- select_cohort: Main cohort selection pipeline
- draw_index_dates_for_control: Index date assignment for control patients with death validation

The module ensures control patients receive valid index dates (not after death) through
retry mechanisms and maintains detailed statistics throughout the filtering process.
"""

import json
import logging
import os
from os.path import join
from typing import List, Tuple

import numpy as np
import pandas as pd

from corebehrt.constants.causal.paths import STATS_PATH
from corebehrt.constants.cohort import CRITERIA_DEFINITIONS, EXCLUSION, INCLUSION
from corebehrt.constants.data import PID_COL, TIMESTAMP_COL
from corebehrt.constants.paths import INDEX_DATES_FILE
from corebehrt.functional.causal.checks import check_time_windows
from corebehrt.functional.cohort_handling.advanced.index_dates import (
    draw_index_dates_for_control_with_redraw,
    select_time_eligible_exposed,
)
from corebehrt.functional.preparation.filter import select_first_event
from corebehrt.main_causal.helper.select_cohort_advanced import (
    check_inclusion_exclusion,
    extract_criteria_from_shards,
)
from corebehrt.modules.cohort_handling.advanced.apply import apply_criteria_with_stats
from corebehrt.modules.cohort_handling.advanced.validator import CriteriaValidator
from corebehrt.modules.cohort_handling.patient_filter import (
    exclude_pids_from_df,
    filter_df_by_pids,
)
from corebehrt.modules.features.loader import ConceptLoader
from corebehrt.modules.setup.config import load_config


def select_cohort(
    features_path: str,
    meds_path: str,
    splits: List[str],
    exposures_path: str,
    exposure: str,
    save_path: str,
    time_windows: dict,
    criteria_definitions_path: str,
    logger: logging.Logger,
) -> List[str]:
    """
    Select cohort by applying multiple filtering steps.

    The process includes:
      1. Loading patient data.
      2. Determine index dates for exposed (first exposure).
      3. Filtering by time windows (sufficient follow-up and lookback).
      4. Advanced filtering based on age, comorbities, prior medication.
      5. Draw index dates for unexposed patients from exposed index dates.
        5.1. For dead patients, redraw up to 2 times, otherwise exclude.
        5.2. Keep the connection to which exposed has the same index date.
      6. Perform filtering by age, comorbities, prior medication.
      7. Optional compliance filtering (if excluded, exclude also control with same index date to avoid survivor bias)
      8. Split into train/val/test sets.

    Args:
        path_cfg: Configuration dictionary
        selection_cfg: Configuration dictionary
        index_date_cfg: Configuration dictionary
        logger: Logger object
    Returns:
        Tuple of:
          - Final patient IDs (list)
          - Series of index dates (with potential test shift applied)
          - Train/validation patient IDs (list)
          - Test patient IDs (list)
    """
    check_time_windows(time_windows)
    patients_info, exposures, index_dates = _load_data(
        features_path, exposures_path, exposure, logger
    )
    control_patients_info = exclude_pids_from_df(
        patients_info, index_dates[PID_COL].unique()
    )

    criteria_config = load_config(criteria_definitions_path)
    criteria_exposed, index_dates_filtered_exposed = _prepare_exposed(
        index_dates,
        time_windows,
        logger,
        criteria_config,
        meds_path,
        splits,
        save_path,
    )

    criteria_control, index_dates_filtered_control = _prepare_control(
        control_patients_info,
        index_dates_filtered_exposed,
        logger,
        criteria_config,
        meds_path,
        splits,
        save_path,
    )

    criteria = pd.concat([criteria_exposed, criteria_control])
    criteria.to_csv(join(save_path, STATS_PATH, "criteria.csv"), index=False)

    final_index_dates = pd.concat(
        [index_dates_filtered_exposed, index_dates_filtered_control]
    )
    final_index_dates.to_csv(join(save_path, INDEX_DATES_FILE), index=False)

    pids = final_index_dates[PID_COL].unique()

    return pids


def _prepare_control(
    control_patients_info: pd.DataFrame,
    index_dates: pd.DataFrame,
    logger: logging.Logger,
    criteria_config: dict,
    meds_path: str,
    splits: List[str],
    save_path: str,
):
    """
    Prepare control patients for cohort selection.
    Return criteria and index dates.
    Save stats.
    """
    # Now we need to draw index dates for unexposed patients from exposed index dates, taking death date into account
    control_index_dates, exposure_matching = draw_index_dates_for_control_with_redraw(
        control_patients_info[PID_COL].unique(),
        index_dates,
        control_patients_info,
    )
    exposure_matching.to_csv(join(save_path, "index_date_matching.csv"), index=False)
    log_patient_num(logger, control_index_dates, "control_index_dates")

    criteria_control, control_stats = filter_by_criteria(
        criteria_config,
        meds_path,
        control_index_dates,
        splits,
        control_index_dates.index.tolist(),
        logger,
        "control",
    )
    _save_stats(control_stats, save_path, "control", logger)
    control_index_date_filtered = filter_df_by_pids(
        control_index_dates, criteria_control[PID_COL]
    )
    return criteria_control, control_index_date_filtered


def _prepare_exposed(
    index_dates: pd.DataFrame,
    time_windows: dict,
    logger: logging.Logger,
    criteria_config: dict,
    meds_path: str,
    splits: List[str],
    save_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare exposed patients for cohort selection.
    Return criteria and index dates.
    Save stats.
    """
    time_eligible_exposed = select_time_eligible_exposed(index_dates, time_windows)
    criteria_exposed, exposed_stats = filter_by_criteria(
        criteria_config,
        meds_path,
        index_dates,
        splits,
        time_eligible_exposed,
        logger,
        "exposed",
    )
    _save_stats(exposed_stats, save_path, "exposed", logger)

    index_dates_filtered_exposed = filter_df_by_pids(
        index_dates, criteria_exposed[PID_COL]
    )
    return criteria_exposed, index_dates_filtered_exposed


def filter_by_criteria(
    criteria_config: dict,
    meds_path: str,
    index_dates: pd.DataFrame,
    splits: List[str],
    pids: List[str],
    logger: logging.Logger,
    description: str,
):
    """
    Filter patients by criteria.
    Save stats.
    Return filtered criteria.
    """
    validator = CriteriaValidator(criteria_config.get(CRITERIA_DEFINITIONS))
    validator.validate()
    check_inclusion_exclusion(validator, criteria_config)
    criteria = extract_criteria_from_shards(
        meds_path,
        index_dates,
        criteria_config.get(CRITERIA_DEFINITIONS),
        splits,
        pids=pids,
    )
    log_patient_num(logger, index_dates, "index_dates")
    logger.info(f"N pids {description}: {len(pids)}")
    log_patient_num(logger, criteria, "criteria")
    logger.info(f"Applying criteria to {description} and saving stats")
    criteria_filtered, stats = apply_criteria_with_stats(
        criteria,
        criteria_config.get(INCLUSION),
        criteria_config.get(EXCLUSION),
    )
    return criteria_filtered, stats


def _save_stats(stats: dict, save_path: str, description: str, logger: logging.Logger):
    stats = _ensure_stats_format(stats)
    logger.info("Saving stats")
    os.makedirs(join(save_path, STATS_PATH), exist_ok=True)
    with open(join(save_path, STATS_PATH, f"{description}.json"), "w") as f:
        json.dump(stats, f)


def _load_data(
    features_path: str, exposures_path: str, exposure: str, logger: logging.Logger
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Loading data")
    patients_info = pd.read_parquet(
        join(features_path, "patient_info.parquet")
    ).drop_duplicates(subset=PID_COL, keep="first")
    log_patient_num(logger, patients_info, "patients_info")
    exposures = ConceptLoader.read_file(join(exposures_path, exposure))
    log_patient_num(logger, exposures, "exposures")
    index_dates = select_first_event(exposures, PID_COL, TIMESTAMP_COL)
    return patients_info, exposures, index_dates


def _ensure_stats_format(stats: dict) -> dict:
    return {
        k: int(v) if isinstance(v, (np.int32, np.int64)) else v
        for k, v in stats.items()
    }


def log_patient_num(logger: logging.Logger, df: pd.DataFrame, name: str):
    logger.info(f"N {name}: {len(df[PID_COL].unique())}")
