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
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from corebehrt.functional.cohort_handling.advanced.vis import (
    plot_cohort_stats,
    plot_multiple_cohort_stats,
    plot_age_distribution,
    plot_index_date_distribution,
)
import shutil
from corebehrt.constants.causal.paths import (
    EXPOSURES_FILE,
    INDEX_DATE_MATCHING_FILE,
    STATS_PATH,
    CRITERIA_FLAGS_FILE,
    CRITERIA_DEFINITIONS_FILE,
)
from corebehrt.constants.cohort import CRITERIA_DEFINITIONS, EXCLUSION, INCLUSION
from corebehrt.constants.data import (
    ABSPOS_COL,
    CONCEPT_COL,
    PID_COL,
    TIMESTAMP_COL,
    AGE_COL,
    BIRTHDATE_COL,
)
from corebehrt.constants.paths import FOLDS_FILE, INDEX_DATES_FILE, TEST_PIDS_FILE
from corebehrt.constants.causal.data import CONTROL_PID_COL
from corebehrt.functional.causal.checks import check_time_windows
from corebehrt.functional.cohort_handling.advanced.index_dates import (
    draw_index_dates_for_control_with_redraw,
    select_time_eligible_exposed,
)
from corebehrt.functional.features.split import create_folds, split_test
from corebehrt.functional.io_operations.meds import iterate_splits_and_shards
from corebehrt.functional.preparation.filter import select_first_event
from corebehrt.functional.utils.time import get_hours_since_epoch
from corebehrt.modules.cohort_handling.advanced.apply import apply_criteria_with_stats
from corebehrt.modules.cohort_handling.advanced.extract import CohortExtractor
from corebehrt.modules.cohort_handling.advanced.validator import CriteriaValidator
from corebehrt.modules.cohort_handling.patient_filter import (
    filter_df_by_pids,
)
from corebehrt.modules.features.loader import ConceptLoader
from corebehrt.modules.setup.config import load_config

logger = logging.getLogger(__name__)


def select_cohort(
    features_path: str,
    meds_path: str,
    splits: List[str],
    exposures_path: str,
    exposure: str,
    save_path: str,
    time_windows: dict,
    criteria_definitions_path: str,
    index_date_matching_cfg: dict,
    logger: logging.Logger,
) -> List[str]:
    """
    Select cohort by applying multiple filtering steps.

    The process includes:
      1. Loading patient data and determining index dates for exposed patients
      2. Filtering by time windows (sufficient follow-up and lookback)
      3. Advanced filtering based on age, comorbidities, prior medication
      4. Drawing index dates for unexposed patients from exposed index dates
      5. Applying same filtering criteria to control patients
      6. Combining results and saving outputs

    Parameters
    ----------
    features_path : str
        Path to patient features directory
    meds_path : str
        Path to medication data directory
    splits : List[str]
        Data splits to process (e.g., ["train", "val"])
    exposures_path : str
        Path to exposure data directory
    exposure : str
        Exposure file name
    save_path : str
        Directory to save results and statistics
    time_windows : dict
        Time window requirements for eligibility
    criteria_definitions_path : str
        Path to criteria configuration file
    index_date_matching_cfg : dict
        Configuration for index date matching with keys:
        - birth_year_tolerance: int, default=3
        - redraw_attempts: int, default=3
    logger : logging.Logger
        Logger for tracking progress

    Returns
    -------
    List[str]
        Final patient IDs after all filtering steps
    """
    check_time_windows(time_windows)
    patients_info, exposures, index_dates, index_date_matching = _load_data(
        features_path, exposures_path, exposure, logger
    )

    criteria_config = load_config(criteria_definitions_path)
    shutil.copy(criteria_definitions_path, join(save_path, CRITERIA_DEFINITIONS_FILE))
    logger.info("Preparing exposed patients")
    criteria_exposed, index_dates_filtered_exposed, exposed_stats = _prepare_exposed(
        index_dates,
        time_windows,
        logger,
        criteria_config,
        meds_path,
        splits,
        save_path,
    )
    logger.info("Preparing control patients")
    criteria_control, index_dates_filtered_control, control_stats = _prepare_control(
        patients_info,
        index_dates_filtered_exposed,
        logger,
        criteria_config,
        meds_path,
        splits,
        save_path,
        index_date_matching=index_date_matching,
        index_date_matching_cfg=index_date_matching_cfg,
    )
    logger.info("Creating combined visualization")
    # Create combined visualization
    combined_stats = {"exposed": exposed_stats, "control": control_stats}

    plot_multiple_cohort_stats(
        stats_dict=combined_stats,
        figsize=(20, 12),
        save_path=join(save_path, STATS_PATH, "cohort_comparison.png"),
        show_plot=False,
    )
    logger.info("Saving data")
    criteria = pd.concat([criteria_exposed, criteria_control])
    criteria.to_csv(join(save_path, STATS_PATH, CRITERIA_FLAGS_FILE), index=False)

    final_index_dates = pd.concat(
        [index_dates_filtered_exposed, index_dates_filtered_control]
    )
    final_index_dates[ABSPOS_COL] = get_hours_since_epoch(
        final_index_dates[TIMESTAMP_COL]
    )
    final_index_dates[AGE_COL] = _compute_age(final_index_dates, patients_info)
    final_index_dates.to_csv(join(save_path, INDEX_DATES_FILE), index=False)

    pids = final_index_dates[PID_COL].unique()
    exposures = exposures.loc[exposures[PID_COL].isin(pids)]
    exposures.to_csv(join(save_path, EXPOSURES_FILE), index=False)

    plot_age_distribution(
        final_index_dates_with_age=final_index_dates,
        control_pids=index_dates_filtered_control[PID_COL].unique(),
        save_path=join(save_path, STATS_PATH, "age_distribution.png"),
        logger=logger,
    )
    plot_index_date_distribution(
        final_index_dates,
        control_pids=index_dates_filtered_control[PID_COL].unique(),
        save_path=join(save_path, STATS_PATH, "index_date_distribution.png"),
        logger=logger,
    )

    return pids


def _prepare_control(
    patients_info: pd.DataFrame,
    index_dates: pd.DataFrame,
    logger: logging.Logger,
    criteria_config: dict,
    meds_path: str,
    splits: List[str],
    save_path: str,
    index_date_matching: pd.DataFrame = None,
    index_date_matching_cfg: dict = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Prepare control patients for cohort selection.
    Return criteria and index dates.
    Save stats.
    Return control criteria and index dates and index dates for controls.
    """
    # Now we need to draw index dates for unexposed patients from exposed index dates, taking death date into account
    if index_date_matching is None:
        control_pids = list(
            set(patients_info[PID_COL].unique()) - set(index_dates[PID_COL].unique())
        )
        if index_date_matching_cfg is None:
            index_date_matching_cfg = {"birth_year_tolerance": 3, "redraw_attempts": 3}
        control_index_dates, index_date_matching = (
            draw_index_dates_for_control_with_redraw(
                control_pids,
                index_dates,
                patients_info,
                birth_year_tolerance=index_date_matching_cfg.get(
                    "birth_year_tolerance"
                ),
                redraw_attempts=index_date_matching_cfg.get("redraw_attempts"),
            )
        )
        index_date_matching[ABSPOS_COL] = get_hours_since_epoch(
            index_date_matching[TIMESTAMP_COL]
        )
    else:
        # Validate required columns exist
        required_cols = [CONTROL_PID_COL, TIMESTAMP_COL]
        missing_cols = [
            col for col in required_cols if col not in index_date_matching.columns
        ]
        if missing_cols:
            raise ValueError(
                f"Index date matching file missing required columns: {missing_cols}"
            )

        control_index_dates = index_date_matching.rename(
            columns={CONTROL_PID_COL: PID_COL}
        )
        control_index_dates = control_index_dates[[PID_COL, TIMESTAMP_COL]]
        # Add ABSPOS_COL for consistency with the other branch
        control_index_dates[ABSPOS_COL] = get_hours_since_epoch(
            control_index_dates[TIMESTAMP_COL]
        )
    index_date_matching.to_csv(join(save_path, INDEX_DATE_MATCHING_FILE), index=False)
    log_patient_num(logger, control_index_dates, "control_index_dates")
    criteria_control, included_pids_control, control_stats = filter_by_criteria(
        criteria_config,
        meds_path,
        control_index_dates,
        splits,
        control_index_dates.index.tolist(),
        logger,
        "control",
    )
    _save_stats(control_stats, save_path, "control", logger)
    plot_cohort_stats(
        stats=control_stats,
        title="Control Patients Cohort Selection",
        save_path=join(save_path, STATS_PATH, "control_flow.png"),
        show_plot=False,
    )
    control_index_date_filtered = filter_df_by_pids(
        control_index_dates, included_pids_control
    )
    return criteria_control, control_index_date_filtered, control_stats


def _prepare_exposed(
    index_dates: pd.DataFrame,
    time_windows: dict,
    logger: logging.Logger,
    criteria_config: dict,
    meds_path: str,
    splits: List[str],
    save_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Prepare exposed patients for cohort selection.
    Return criteria and index dates.
    Save stats.
    Return exposed criteria and index dates.
    """
    time_eligible_exposed = select_time_eligible_exposed(index_dates, time_windows)
    criteria_exposed, included_pids_exposed, exposed_stats = filter_by_criteria(
        criteria_config,
        meds_path,
        index_dates,
        splits,
        time_eligible_exposed,
        logger,
        "exposed",
    )
    _save_stats(exposed_stats, save_path, "exposed", logger)
    plot_cohort_stats(
        stats=exposed_stats,
        title="Exposed Patients Cohort Selection",
        save_path=join(save_path, STATS_PATH, "exposed_flow.png"),
        show_plot=False,
    )
    index_dates_filtered_exposed = filter_df_by_pids(index_dates, included_pids_exposed)
    return criteria_exposed, index_dates_filtered_exposed, exposed_stats


def filter_by_criteria(
    criteria_config: dict,
    meds_path: str,
    index_dates: pd.DataFrame,
    splits: List[str],
    pids: List[str],
    logger: logging.Logger,
    description: str,
) -> Tuple[pd.DataFrame, List[str], dict]:
    """
    Filter patients based on inclusion/exclusion criteria and extract relevant data.

    This function validates criteria configuration, extracts criteria data from medication
    shards, applies inclusion/exclusion filters, and logs statistics throughout the process.

    Args:
        criteria_config: Dictionary containing criteria definitions, inclusion, and exclusion rules
        meds_path: Path to medication data shards
        index_dates: DataFrame with patient index dates
        splits: List of data splits to process
        pids: List of patient IDs to filter
        logger: Logger instance for recording statistics and progress
        description: Description string for logging purposes

    Returns:
        tuple: (criteria_data, included_pids, stats)
            - criteria_data: pd.DataFrame with extracted criteria data for all patients
            - included_pids: List of patient IDs that passed the inclusion/exclusion criteria
            - stats: Dictionary containing filtering statistics and counts

    Notes
    -----
    The function performs the following steps:
    1. Validates criteria configuration and checks inclusion/exclusion logic
    2. Extracts criteria data from shards for specified patients
    3. Logs patient counts at various stages
    4. Applies inclusion/exclusion criteria to filter patients
    5. Returns (unfiltered) criteria flags, included patient IDs, and statistics
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
    included_pids, stats = apply_criteria_with_stats(
        criteria,
        criteria_config.get(INCLUSION),
        criteria_config.get(EXCLUSION),
    )
    return criteria, included_pids, stats


def _save_stats(stats: dict, save_path: str, description: str, logger: logging.Logger):
    stats = _ensure_stats_format(stats)
    logger.info("Saving stats")
    os.makedirs(join(save_path, STATS_PATH), exist_ok=True)
    with open(join(save_path, STATS_PATH, f"{description}.json"), "w") as f:
        json.dump(stats, f)


def _load_data(
    features_path: str, exposures_path: str, exposure: str, logger: logging.Logger
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load data from features and exposures.
    Return patients info, exposures, and index dates.
    """
    logger.info("Loading data")
    patients_info = pd.read_parquet(
        join(features_path, "patient_info.parquet")
    ).drop_duplicates(subset=PID_COL, keep="first")
    log_patient_num(logger, patients_info, "patients_info")
    exposures = ConceptLoader.read_file(join(exposures_path, exposure))
    log_patient_num(logger, exposures, "exposures")
    index_dates = select_first_event(exposures, PID_COL, TIMESTAMP_COL)

    index_date_matching = None
    if os.path.exists(join(exposures_path, INDEX_DATE_MATCHING_FILE)):
        logger.info("Loading index date matching file")
        index_date_matching = pd.read_csv(
            join(exposures_path, INDEX_DATE_MATCHING_FILE), parse_dates=[TIMESTAMP_COL]
        )

    return patients_info, exposures, index_dates, index_date_matching


def _ensure_stats_format(stats: dict) -> dict:
    return {
        k: int(v) if isinstance(v, (np.int32, np.int64)) else v
        for k, v in stats.items()
    }


def log_patient_num(logger: logging.Logger, df: pd.DataFrame, name: str):
    logger.info(f"N {name}: {len(df[PID_COL].unique())}")


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


def _compute_age(
    index_dates_df: pd.DataFrame, patients_info: pd.DataFrame
) -> pd.Series:
    """
    Computes age at index date for each patient.

    Args:
        index_dates_df: DataFrame with at least PID_COL and TIMESTAMP_COL.
        patients_info: DataFrame with at least PID_COL and BIRTHDATE_COL.

    Returns:
        A pandas Series containing the calculated age in years, aligned with index_dates_df.
    """
    # Create a mapping from patient ID to their birthdate for efficient lookup
    birthdate_map = patients_info.set_index(PID_COL)[BIRTHDATE_COL]

    # Map the birthdates onto the index_dates DataFrame using the patient ID
    birthdates = index_dates_df[PID_COL].map(birthdate_map)

    # Ensure columns are datetime objects before subtraction
    index_times = pd.to_datetime(index_dates_df[TIMESTAMP_COL])
    birth_times = pd.to_datetime(birthdates)

    # Calculate and return the age
    age_in_years = (index_times - birth_times) / pd.Timedelta(days=365.25)

    return age_in_years
