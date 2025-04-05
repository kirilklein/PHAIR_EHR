from datetime import timedelta

import pandas as pd

from corebehrt.constants.cohort import (
    CODE_ENTRY,
    CODE_PATTERNS,
    CRITERIA_DEFINITIONS,
    DELAYS,
    THRESHOLD,
    TIME_WINDOW_DAYS,
)
from corebehrt.constants.data import PID_COL, TIMESTAMP_COL
from corebehrt.functional.cohort_handling.advanced.calculations import calculate_age
from corebehrt.functional.cohort_handling.advanced.match import (
    evaluate_numeric_criteria,
    get_all_codes_for_criterion,
    match_codes,
)
from corebehrt.modules.cohort_handling.advanced.data.patient import Patient


def extract_patient_criteria(
    df: pd.DataFrame, index_dates: pd.DataFrame, config: dict
) -> dict[int, Patient]:
    """
    Extract and evaluate inclusion/exclusion criteria for each patient.
    """
    # Validate required configuration
    if CRITERIA_DEFINITIONS not in config:
        raise ValueError(f"Configuration missing required key: {CRITERIA_DEFINITIONS}")

    # Get code patterns for reuse
    code_patterns = config.get(CODE_PATTERNS, {})

    # Get unique patient IDs in the current shard
    shard_patient_ids = df[PID_COL].unique()

    # Filter index_dates to only include patients present in the shard
    relevant_index_dates = index_dates[index_dates[PID_COL].isin(shard_patient_ids)]

    patients = {}
    delays_config = config.get(DELAYS, {})

    for _, row in relevant_index_dates.iterrows():
        patient = Patient(row[PID_COL], row[TIMESTAMP_COL])
        patient_data = df[df[PID_COL] == patient.subject_id]
        if patient_data.empty:
            continue

        patient.age = calculate_age(patient_data, patient.index_date)

        for criteria, criteria_cfg in config[CRITERIA_DEFINITIONS].items():
            min_timestamp = patient.index_date - timedelta(
                days=criteria_cfg.get(TIME_WINDOW_DAYS, 36500)
            )

            if THRESHOLD in criteria_cfg:
                value, matched = evaluate_numeric_criteria(
                    patient_data, criteria_cfg, patient.index_date, min_timestamp
                )
                patient.values[criteria] = value
                patient.criteria_flags[criteria] = matched
            else:
                # Get all codes including referenced patterns
                all_codes = get_all_codes_for_criterion(criteria_cfg, code_patterns)
                criteria_cfg = (
                    criteria_cfg.copy()
                )  # Create a copy to not modify original
                criteria_cfg[CODE_ENTRY] = all_codes

                patient.criteria_flags[criteria] = match_codes(
                    patient_data,
                    criteria_cfg,
                    delays_config,
                    patient.index_date,
                    min_timestamp,
                )

        patients[patient.subject_id] = patient

    return patients
