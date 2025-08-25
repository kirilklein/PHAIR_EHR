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
from corebehrt.functional.utils.filter import safe_control_pids
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
    logger.info("=" * 80)
    logger.info("STARTING COHORT SELECTION")
    logger.info("=" * 80)

    check_time_windows(time_windows)
    patients_info, exposures, index_dates, index_date_matching = _load_data(
        features_path, exposures_path, exposure, logger
    )

    # Initial counts - these are important for understanding the cohort
    exposed_pids = index_dates[PID_COL].unique()
    logger.info(f"Initial exposed patients from index dates: {len(exposed_pids)}")
    logger.info(
        f"Total patients available in features: {patients_info[PID_COL].nunique()}"
    )
    logger.info(f"Total exposure events: {len(exposures)}")

    criteria_config = load_config(criteria_definitions_path)
    shutil.copy(criteria_definitions_path, join(save_path, CRITERIA_DEFINITIONS_FILE))

    logger.info("=" * 50)
    logger.info("PREPARING EXPOSED PATIENTS")
    logger.info("=" * 50)
    criteria_exposed, index_dates_filtered_exposed, exposed_stats = _prepare_exposed(
        index_dates,
        time_windows,
        logger,
        criteria_config,
        meds_path,
        splits,
        save_path,
    )

    # Key outcome metrics - important to track
    exposed_final_count = len(index_dates_filtered_exposed[PID_COL].unique())
    logger.info(f"Final exposed patients after all filtering: {exposed_final_count}")
    exposed_loss = len(exposed_pids) - exposed_final_count
    exposed_loss_pct = (
        (exposed_loss / len(exposed_pids) * 100) if len(exposed_pids) else 0.0
    )
    logger.info(f"Exposed patient exclusions: {exposed_loss} ({exposed_loss_pct:.1f}%)")

    logger.info("=" * 50)
    logger.info("PREPARING CONTROL PATIENTS")
    logger.info("=" * 50)
    criteria_control, index_dates_filtered_control, control_stats = _prepare_control(
        patients_info,
        index_dates_filtered_exposed,
        exposed_pids,
        logger,
        criteria_config,
        meds_path,
        splits,
        save_path,
        index_date_matching=index_date_matching,
        index_date_matching_cfg=index_date_matching_cfg,
    )

    # Key outcome metrics
    control_final_count = len(index_dates_filtered_control[PID_COL].unique())
    logger.info(f"Final control patients after all filtering: {control_final_count}")

    logger.info("Creating combined visualization")
    # Create combined visualization
    combined_stats = {"exposed": exposed_stats, "control": control_stats}

    plot_multiple_cohort_stats(
        stats_dict=combined_stats,
        figsize=(20, 12),
        save_path=join(save_path, STATS_PATH, "figs", "cohort_comparison.png"),
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
        save_path=join(save_path, STATS_PATH, "figs", "age_distribution.png"),
        logger=logger,
    )
    plot_index_date_distribution(
        final_index_dates,
        control_pids=index_dates_filtered_control[PID_COL].unique(),
        save_path=join(save_path, STATS_PATH, "figs", "index_date_distribution.png"),
        logger=logger,
    )

    logger.info("=" * 80)
    logger.info("COHORT SELECTION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Final exposed patients: {exposed_final_count}")
    logger.info(f"Final control patients: {control_final_count}")
    logger.info(f"Total final cohort: {exposed_final_count + control_final_count}")
    logger.info("=" * 80)

    return pids


def _prepare_control(
    patients_info: pd.DataFrame,
    filtered_exposed_index_dates: pd.DataFrame,
    exposed_pids: List[str],
    logger: logging.Logger,
    criteria_config: dict,
    meds_path: str,
    splits: List[str],
    save_path: str,
    index_date_matching: pd.DataFrame | None = None,
    index_date_matching_cfg: dict | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Prepare control patients for cohort selection.
    """
    logger.info("Starting control patient preparation")

    # Now we need to draw index dates for unexposed patients from exposed index dates, taking death date into account
    if index_date_matching is None:
        logger.info("Creating new index date matching for controls")

        pid_series = patients_info[PID_COL]
        total_patients_available = len(pid_series.unique())
        logger.info(f"Total patients available in features: {total_patients_available}")

        control_pids = safe_control_pids(pid_series, exposed_pids)
        logger.info(
            f"Potential control patients (excluding exposed): {len(control_pids)}"
        )
        logger.info(
            f"Control pool reduction: {total_patients_available - len(control_pids)} patients excluded (exposed)"
        )

        default_cfg = {
            "birth_year_tolerance": 3,
            "redraw_attempts": 3,
            "age_adjusted": True,
        }
        index_date_matching_cfg = {**default_cfg, **(index_date_matching_cfg or {})}
        logger.info(f"Index date matching config: {index_date_matching_cfg}")

        logger.info("Assigning index dates to control patients...")
        control_index_dates, index_date_matching = (
            draw_index_dates_for_control_with_redraw(
                control_pids,
                filtered_exposed_index_dates,
                patients_info,
                birth_year_tolerance=index_date_matching_cfg["birth_year_tolerance"],
                redraw_attempts=index_date_matching_cfg["redraw_attempts"],
                age_adjusted=index_date_matching_cfg["age_adjusted"],
            )
        )

        controls_with_index_dates = index_date_matching[CONTROL_PID_COL].nunique()
        logger.info(
            f"Controls successfully assigned index dates: {controls_with_index_dates} / {len(control_pids)}"
        )
        excluded_during_matching = len(control_pids) - controls_with_index_dates
        if excluded_during_matching > 0:
            matching_loss_pct = excluded_during_matching / len(control_pids) * 100
            logger.warning(
                f"Controls excluded during index date matching: {excluded_during_matching} ({matching_loss_pct:.1f}%)"
            )

        index_date_matching[ABSPOS_COL] = get_hours_since_epoch(
            index_date_matching[TIMESTAMP_COL]
        )
    else:
        logger.info("Using pre-existing index date matching")
        # Validate required columns exist
        required_cols = [CONTROL_PID_COL, TIMESTAMP_COL]
        missing_cols = [
            col for col in required_cols if col not in index_date_matching.columns
        ]
        if missing_cols:
            raise ValueError(
                f"Index date matching file missing required columns: {missing_cols}"
            )

        controls_with_index_dates = index_date_matching[CONTROL_PID_COL].nunique()
        logger.info(
            f"Pre-existing controls with index dates: {controls_with_index_dates}"
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

    logger.info("Applying criteria filtering to control patients...")
    pre_criteria_count = control_index_dates[PID_COL].nunique()
    logger.info(f"Control patients before criteria filtering: {pre_criteria_count}")

    criteria_control, included_pids_control, control_stats = filter_by_criteria(
        criteria_config,
        meds_path,
        control_index_dates,
        splits,
        control_index_dates[PID_COL].unique().tolist(),
        logger,
        "control",
    )

    post_criteria_count = len(included_pids_control)
    logger.info(f"Control patients after criteria filtering: {post_criteria_count}")
    excluded_by_criteria = pre_criteria_count - post_criteria_count
    if excluded_by_criteria > 0:
        criteria_loss_pct = (
            (excluded_by_criteria / pre_criteria_count * 100)
            if pre_criteria_count
            else 0.0
        )
        logger.info(
            f"Controls excluded by criteria: {excluded_by_criteria} ({criteria_loss_pct:.1f}%)"
        )

    _save_stats(control_stats, save_path, "control", logger)
    plot_cohort_stats(
        stats=control_stats,
        title="Control Patients Cohort Selection",
        save_path=join(save_path, STATS_PATH, "figs", "control_flow.png"),
        show_plot=False,
    )
    control_index_date_filtered = filter_df_by_pids(
        control_index_dates, included_pids_control
    )

    logger.info(
        f"Final control index dates after filtering: {len(control_index_date_filtered)}"
    )
    logger.info(
        f"Final unique control patients: {control_index_date_filtered[PID_COL].nunique()}"
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
    """
    logger.info("Starting exposed patient preparation")
    initial_exposed_count = index_dates[PID_COL].nunique()
    logger.info(f"Initial exposed patients: {initial_exposed_count}")

    logger.info("Applying time window eligibility criteria...")
    logger.info(f"Time windows: {time_windows}")

    time_eligible_exposed = select_time_eligible_exposed(index_dates, time_windows)
    logger.info(
        f"Time eligible exposed patients: {len(time_eligible_exposed)} / {initial_exposed_count}"
    )
    excluded_by_time = initial_exposed_count - len(time_eligible_exposed)
    if excluded_by_time > 0:
        time_loss_pct = excluded_by_time / initial_exposed_count * 100
        logger.info(
            f"Exposed excluded by time windows: {excluded_by_time} ({time_loss_pct:.1f}%)"
        )

    logger.info("Applying criteria filtering to exposed patients...")
    criteria_exposed, included_pids_exposed, exposed_stats = filter_by_criteria(
        criteria_config,
        meds_path,
        index_dates,
        splits,
        time_eligible_exposed,
        logger,
        "exposed",
    )

    post_criteria_count = len(included_pids_exposed)
    logger.info(f"Exposed patients after criteria filtering: {post_criteria_count}")
    excluded_by_criteria = len(time_eligible_exposed) - post_criteria_count
    if excluded_by_criteria > 0:
        criteria_loss_pct = excluded_by_criteria / len(time_eligible_exposed) * 100
        logger.info(
            f"Time-eligible exposed excluded by criteria: {excluded_by_criteria} ({criteria_loss_pct:.1f}%)"
        )

    total_excluded = initial_exposed_count - post_criteria_count
    if total_excluded > 0:
        total_loss_pct = total_excluded / initial_exposed_count * 100
        logger.info(
            f"Total exposed excluded (time + criteria): {total_excluded} ({total_loss_pct:.1f}%)"
        )

    _save_stats(exposed_stats, save_path, "exposed", logger)
    plot_cohort_stats(
        stats=exposed_stats,
        title="Exposed Patients Cohort Selection",
        save_path=join(save_path, STATS_PATH, "figs", "exposed_flow.png"),
        show_plot=False,
    )
    index_dates_filtered_exposed = filter_df_by_pids(index_dates, included_pids_exposed)

    logger.info(
        f"Final exposed index dates after filtering: {len(index_dates_filtered_exposed)}"
    )
    logger.info(
        f"Final unique exposed patients: {index_dates_filtered_exposed[PID_COL].nunique()}"
    )

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
    """
    logger.info(f"Starting criteria filtering for {description} patients")
    logger.debug(f"Input patient count for {description}: {len(pids)}")
    logger.debug(f"Index dates shape for {description}: {index_dates.shape}")

    validator = CriteriaValidator(criteria_config.get(CRITERIA_DEFINITIONS))
    validator.validate()
    check_inclusion_exclusion(validator, criteria_config)

    logger.info(f"Extracting criteria from medical data for {description} patients...")
    criteria = extract_criteria_from_shards(
        meds_path,
        index_dates,
        criteria_config.get(CRITERIA_DEFINITIONS),
        splits,
        pids=pids,
    )

    logger.debug(f"Criteria extraction completed for {description}")
    log_patient_num(logger, index_dates, f"{description}_index_dates")
    log_patient_num(logger, criteria, f"{description}_criteria")

    logger.info(f"Applying inclusion/exclusion criteria to {description} patients...")
    inclusion_config = criteria_config.get(INCLUSION)
    exclusion_config = criteria_config.get(EXCLUSION)

    included_pids, stats = apply_criteria_with_stats(
        criteria,
        inclusion_config,
        exclusion_config,
    )

    logger.info(f"Criteria filtering completed for {description}")
    logger.info(
        f"Patients passing all criteria for {description}: {len(included_pids)}"
    )
    logger.debug(f"Detailed filtering statistics for {description}: {stats}")

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
    """Extract criteria from medical event data across multiple shards."""
    logger.info("Starting criteria extraction from medical data shards")
    logger.info(
        f"Target patient count for extraction: {len(pids) if pids is not None else 'All'}"
    )
    logger.info(f"Processing splits: {splits}")
    logger.debug(
        f"Criteria definitions: {list(criteria_definitions_cfg.keys()) if criteria_definitions_cfg else 'None'}"
    )

    cohort_extractor = CohortExtractor(
        criteria_definitions_cfg,
    )

    criteria_dfs = []
    total_patients_processed = 0
    total_shards_processed = 0
    empty_shards = 0

    for shard_path in iterate_splits_and_shards(meds_path, splits):
        total_shards_processed += 1
        logger.info(
            f"Processing shard {total_shards_processed}: {os.path.basename(shard_path)}"
        )

        shard = pd.read_parquet(shard_path)
        logger.info(
            f"Shard loaded with {len(shard)} events and {shard[PID_COL].nunique()} unique patients"
        )

        shard[CONCEPT_COL] = shard[CONCEPT_COL].astype("category")

        if pids is not None:
            logger.info(f"Filtering shard for {len(pids)} target patients")
            patients_in_shard_before = shard[PID_COL].nunique()
            shard = shard[shard[PID_COL].isin(pids)]
            patients_in_shard_after = shard[PID_COL].nunique() if not shard.empty else 0
            logger.info(
                f"Patients in shard after filtering: {patients_in_shard_after} / {patients_in_shard_before}"
            )

            if shard.empty:
                logger.info(f"No matching patients found in shard {shard_path}")
                empty_shards += 1
                continue

        if shard.empty:
            logger.warning(f"Empty shard found: {shard_path}")
            empty_shards += 1
            continue

        patients_for_extraction = shard[PID_COL].nunique()
        logger.info(f"Extracting criteria for {patients_for_extraction} patients")

        criteria_df = cohort_extractor.extract(
            shard,
            index_dates,
        )

        if not criteria_df.empty:
            patients_with_criteria = criteria_df[PID_COL].nunique()
            logger.info(
                f"Successfully extracted criteria for {patients_with_criteria} patients"
            )
            criteria_dfs.append(criteria_df)
            total_patients_processed += patients_with_criteria
        else:
            logger.warning(
                f"No criteria matched for any patients in shard {shard_path}"
            )

    logger.info(f"Criteria extraction summary:")
    logger.info(f"  - Total shards processed: {total_shards_processed}")
    logger.info(f"  - Shards with criteria data: {len(criteria_dfs)}")
    logger.info(f"  - Total patients with criteria: {total_patients_processed}")

    if empty_shards > 0:
        logger.info(f"  - Empty shards: {empty_shards}")

    if not criteria_dfs:
        error_msg = "No criteria data found for any patients"
        if pids is not None:
            error_msg += f" in the provided list of {len(pids)} patient IDs"
        raise ValueError(error_msg)

    combined_criteria = pd.concat(criteria_dfs)
    logger.info(
        f"Combined criteria extracted for {combined_criteria[PID_COL].nunique()} unique patients"
    )
    logger.info(f"Combined criteria DataFrame shape: {combined_criteria.shape}")

    return combined_criteria


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
