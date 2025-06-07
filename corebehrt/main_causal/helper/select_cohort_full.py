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
from datetime import datetime
from os.path import join
from typing import List, Tuple

import numpy as np
import pandas as pd

from corebehrt.constants.causal.data import CONTROL_PID_COL, EXPOSED_PID_COL
from corebehrt.constants.causal.paths import STATS_PATH
from corebehrt.constants.cohort import CRITERIA_DEFINITIONS, EXCLUSION, INCLUSION
from corebehrt.constants.data import DEATHDATE_COL, PID_COL, TIMESTAMP_COL
from corebehrt.constants.paths import INDEX_DATES_FILE
from corebehrt.functional.preparation.filter import select_first_event
from corebehrt.main_causal.helper.select_cohort_advanced import (
    check_inclusion_exclusion,
    extract_criteria_from_shards,
)
from corebehrt.modules.cohort_handling.advanced.apply import apply_criteria_with_stats
from corebehrt.modules.cohort_handling.advanced.validator import CriteriaValidator
from corebehrt.modules.cohort_handling.patient_filter import exclude_pids_from_df
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

    time_eligible_exposed = select_time_eligible_exposed(index_dates, time_windows)

    excluded_exposed = exclude_pids_from_df(index_dates, time_eligible_exposed)[
        PID_COL
    ].unique()
    logger.info(f"N excluded_exposed: {len(excluded_exposed)}")
    patients_info = exclude_pids_from_df(patients_info, excluded_exposed)
    patients_info_control = exclude_pids_from_df(
        patients_info, index_dates[PID_COL].unique()
    )
    log_patient_num(
        logger, patients_info, "patients_info after excluding by time windows"
    )
    criteria_config = load_config(criteria_definitions_path)
    criteria_exposed = filter_by_criteria(
        criteria_config,
        meds_path,
        index_dates,
        splits,
        time_eligible_exposed,
        logger,
        save_path,
        "exposed",
    )

    index_dates_filtered_exposed = index_dates[
        index_dates[PID_COL].isin(criteria_exposed[PID_COL])
    ]
    log_patient_num(
        logger, index_dates_filtered_exposed, "index_dates_filtered_exposed"
    )

    # Now we need to draw index dates for unexposed patients from exposed index dates, taking death date into account
    control_index_dates, exposure_matching = draw_index_dates_for_control(
        patients_info_control[PID_COL].unique(),
        index_dates_filtered_exposed,
        patients_info_control,
    )
    exposure_matching.to_csv(join(save_path, "index_date_matching.csv"), index=False)
    log_patient_num(logger, control_index_dates, "control_index_dates")

    criteria_control = filter_by_criteria(
        criteria_config,
        meds_path,
        control_index_dates,
        splits,
        control_index_dates.index.tolist(),
        logger,
        save_path,
        "control",
    )

    criteria = pd.concat([criteria_exposed, criteria_control])
    criteria.to_csv(join(save_path, STATS_PATH, "criteria.csv"), index=False)

    control_index_date_filtered = control_index_dates[
        control_index_dates[PID_COL].isin(criteria_control[PID_COL])
    ]
    index_dates = pd.concat([index_dates_filtered_exposed, control_index_date_filtered])
    index_dates.to_csv(join(save_path, INDEX_DATES_FILE), index=False)

    pids = index_dates[PID_COL].unique()

    return pids


def filter_by_criteria(
    criteria_config: dict,
    meds_path: str,
    index_dates: pd.DataFrame,
    splits: List[str],
    pids: List[str],
    logger: logging.Logger,
    save_path: str,
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

    stats = _ensure_stats_format(stats)
    logger.info("Saving stats")
    os.makedirs(join(save_path, STATS_PATH), exist_ok=True)
    with open(join(save_path, STATS_PATH, f"{description}.json"), "w") as f:
        json.dump(stats, f)
    return criteria_filtered


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


def log_patient_num(logger, df, name: str):
    logger.info(f"N {name}: {len(df[PID_COL].unique())}")


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


def draw_index_dates_for_control(
    control_pids: List[str],
    exposed_index_dates: pd.DataFrame,
    patients_info: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Draw index dates for unexposed patients by randomly sampling from exposed patients' index dates.

    This function assigns index dates to unexposed (control) patients by randomly sampling
    from the index dates of exposed patients. It ensures that assigned index dates do not
    occur after a patient's death date. If an invalid date is drawn (after death), the
    function will attempt to redraw up to 2 additional times before excluding the patient.

    Parameters
    ----------
    control_pids : List[str]
        List of patient IDs for unexposed/control patients who need index dates assigned.
    exposed_index_dates : pd.DataFrame
        DataFrame with exposed patient data, must include PID_COL and TIMESTAMP_COL columns.
    patients_info : pd.DataFrame
        DataFrame containing patient information, must include PID_COL and DEATHDATE_COL columns.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - pd.DataFrame: Index dates for valid unexposed patients with columns [PID_COL, TIMESTAMP_COL]
        - pd.DataFrame: Matching information with columns ['exposed_pid', 'index_date'] showing
          which exposed patient each unexposed patient was matched to and their assigned index date

    Notes
    -----
    - Patients who die before their assigned index date are excluded after 2 failed attempts
    - The matching process uses random sampling with replacement from exposed patients
    - The function prioritizes validity over maintaining exact sample size
    """

    # Get death dates for unexposed patients
    death_info = patients_info.set_index(PID_COL)[DEATHDATE_COL].reindex(control_pids)

    # Convert exposed info to arrays for faster sampling
    exposed_dates_array = exposed_index_dates[TIMESTAMP_COL].values
    exposed_pids_array = exposed_index_dates[PID_COL].values
    n_exposed = len(exposed_dates_array)
    n_unexposed = len(control_pids)

    # Draw initial random indices
    sampled_indices = np.random.choice(n_exposed, size=n_unexposed, replace=True)

    # Create DataFrame with all necessary info
    temp_df = pd.DataFrame(
        {
            TIMESTAMP_COL: exposed_dates_array[sampled_indices],
            DEATHDATE_COL: death_info.values,
            EXPOSED_PID_COL: exposed_pids_array[sampled_indices],
            CONTROL_PID_COL: control_pids,
        },
        index=control_pids,
    )

    # Perform up to 2 additional attempts for invalid dates
    for attempt in range(2):
        # Create mask vectorized: invalid where index_date > death_date (and death_date is not null)
        invalid_mask = (temp_df[TIMESTAMP_COL] > temp_df[DEATHDATE_COL]) & pd.notna(
            temp_df[DEATHDATE_COL]
        )

        # Break if no invalid dates
        if not invalid_mask.any():
            break

        # Redraw indices for invalid patients and update both columns
        n_invalid = invalid_mask.sum()
        new_indices = np.random.choice(n_exposed, size=n_invalid, replace=True)
        temp_df.loc[invalid_mask, TIMESTAMP_COL] = exposed_dates_array[new_indices]
        temp_df.loc[invalid_mask, EXPOSED_PID_COL] = exposed_pids_array[new_indices]

    # Final check and exclusion vectorized
    final_invalid_mask = (temp_df[TIMESTAMP_COL] > temp_df[DEATHDATE_COL]) & pd.notna(
        temp_df[DEATHDATE_COL]
    )

    # Keep only valid patients
    temp_df = temp_df.loc[~final_invalid_mask]

    # Extract results as DataFrame with same structure as exposed_index_dates
    control_index_dates = pd.DataFrame(
        {PID_COL: temp_df.index, TIMESTAMP_COL: temp_df[TIMESTAMP_COL]}
    )

    # Create matching information
    exposure_matching = temp_df[[EXPOSED_PID_COL, TIMESTAMP_COL]].reset_index(
        drop=False, names=CONTROL_PID_COL
    )

    return control_index_dates, exposure_matching
