import logging
from datetime import datetime
from os.path import join
from typing import List, Tuple

import pandas as pd

from corebehrt.constants.cohort import CRITERIA_DEFINITIONS
from corebehrt.constants.data import PID_COL, TIMESTAMP_COL
from corebehrt.functional.features.split import split_test
from corebehrt.functional.preparation.filter import select_first_event
from corebehrt.main_causal.helper.select_cohort_advanced import (
    extract_criteria_from_shards,
)
from corebehrt.modules.cohort_handling.advanced.validator import CriteriaValidator
from corebehrt.modules.cohort_handling.index_dates import IndexDateHandler
from corebehrt.modules.cohort_handling.patient_filter import exclude_pids_from_df
from corebehrt.modules.features.loader import ConceptLoader
from corebehrt.modules.setup.config import load_config


def select_cohort(
    features_path: str,
    meds_path: str,
    splits: List[str],
    exposures_path: str,
    exposure: str,
    time_windows: dict,
    test_ratio: float,
    criteria_definitions_path: str,
    logger: logging.Logger,
) -> Tuple[List[str], pd.Series, List[str], List[str]]:
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
    logger.info("Loading data")
    patients_info = pd.read_parquet(
        join(features_path, "patient_info.parquet")
    ).drop_duplicates(subset=PID_COL, keep="first")
    logger.info("N patients_info: %d", len(patients_info))
    exposures = ConceptLoader.read_file(join(exposures_path, exposure))
    logger.info("N exposures: %d", len(exposures))
    index_dates = select_first_event(exposures, PID_COL, TIMESTAMP_COL)

    time_eligible_exposed = select_time_eligible_exposed(index_dates, time_windows)

    excluded_exposed = exclude_pids_from_df(index_dates, time_eligible_exposed)[
        PID_COL
    ].unique()
    logger.info("N excluded_exposed: %d", len(excluded_exposed))
    patients_info = exclude_pids_from_df(patients_info, excluded_exposed)
    logger.info(
        "N patients_info after excluding by time windows: %d", len(patients_info)
    )

    criteria_definitions_cfg = load_config(criteria_definitions_path)
    validator = CriteriaValidator(criteria_definitions_cfg.get(CRITERIA_DEFINITIONS))
    validator.validate()
    criteria_exposed = extract_criteria_from_shards(
        meds_path,
        index_dates,
        criteria_definitions_cfg.get(CRITERIA_DEFINITIONS),
        splits,
        pids=time_eligible_exposed,
    )
    logger.info("N index_dates: %d", len(index_dates))
    logger.info("N time_eligible_exposed: %d", len(time_eligible_exposed))
    logger.info("N criteria_exposed: %d", len(criteria_exposed))
    assert False


def log_patient_num(logger, patients_info):
    logger.info(f"Patient number: {len(patients_info[PID_COL].unique())}")


def select_time_eligible_exposed(index_dates: pd.DataFrame, time_windows: dict) -> list:
    """
    We check whether the lookback and followup time windows are satisfied.
    We exclude patients that do not satisfy the time windows.
    Return remaining exposed patients.
    """
    if index_dates.duplicated(subset=PID_COL).any():
        raise ValueError("Duplicate patient IDs found in index_dates")
    if index_dates.empty:
        return []
    sufficient_follow_up = index_dates[TIMESTAMP_COL] + pd.Timedelta(
        **time_windows["min_follow_up"]
    ) <= datetime(**time_windows["data_end"])
    sufficient_lookback = index_dates[TIMESTAMP_COL] - pd.Timedelta(
        **time_windows["min_lookback"]
    ) >= datetime(**time_windows["data_start"])
    filtered_index_dates = index_dates[sufficient_follow_up & sufficient_lookback]

    return filtered_index_dates[PID_COL].unique()


def check_time_windows(time_windows):
    """
    Check that the time windows are valid.
    """
    if "data_end" not in time_windows:
        raise ValueError("data_end must be provided")
    if "data_start" not in time_windows:
        raise ValueError("data_start must be provided")

    try:
        data_end = datetime(**time_windows["data_end"])
    except KeyError:
        raise ValueError("data_end must be provided as year, month, day")
    try:
        data_start = datetime(**time_windows["data_start"])
    except KeyError:
        raise ValueError("data_start must be provided as year, month, day")

    if data_end < data_start:
        raise ValueError("data_end must be greater than data_start")

    if "min_follow_up" not in time_windows:
        raise ValueError("min_follow_up must be provided")
    try:
        pd.Timedelta(**time_windows["min_follow_up"])
    except KeyError:
        raise ValueError(
            "min_follow_up can be given in weeks, days, seconds, minutes, hours"
        )
    if "min_lookback" not in time_windows:
        raise ValueError(
            "min_lookback can be given in weeks, days, seconds, minutes, hours, or years"
        )
    try:
        pd.Timedelta(**time_windows["min_lookback"])
    except KeyError:
        raise ValueError(
            "min_lookback can be given in weeks, days, seconds, minutes, hours, or years"
        )
