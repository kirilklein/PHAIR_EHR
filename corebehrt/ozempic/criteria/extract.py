from datetime import timedelta

import pandas as pd

from corebehrt.constants.data import PID_COL
from corebehrt.ozempic.criteria.match import evaluate_numeric_criteria, match_codes
from corebehrt.ozempic.data.patient import Patient
from corebehrt.ozempic.utils.calculations import calculate_age
from corebehrt.ozempic.utils.definitions import (
    CRITERIA_DEFINITIONS,
    DELAYS,
    THRESHOLD,
    TIME_WINDOW_DAYS,
)


def extract_patient_criteria(
    df: pd.DataFrame, index_dates: pd.DataFrame, config: dict
) -> dict[int, Patient]:
    """
    Extract and evaluate inclusion/exclusion criteria for each patient.

    This function processes patient data and evaluates various criteria based on the provided
    configuration. It handles both numeric criteria (e.g., HbA1c thresholds) and code-based
    criteria (e.g., diagnoses, medications). Criteria evaluation considers time windows and
    code delays as specified in the configuration.

    Args:
        df (pd.DataFrame): Patient event data with columns:
            - PID_COL: Patient identifier
            - CONCEPT_COL: Event code (diagnoses, medications, etc.)
            - VALUE_COL: Numeric values for measurements
            - TIMESTAMP_COL: Event timestamps
        index_dates (pd.DataFrame): Reference dates for each patient with columns:
            - PID_COL: Patient identifier
            - index_date: Reference date for criteria evaluation
        config (dict): Configuration dictionary containing:
            - DELAYS: Dictionary specifying code groups and their delays
            - CRITERIA_DEFINITIONS: Dictionary of criteria to evaluate, each containing:
                - codes: List of regex patterns to match
                - threshold: Optional numeric threshold
                - operator: Optional comparison operator (">=", "<=", ">", "<")
                - time_window_days: Optional time window for event consideration
                - exclude_codes: Optional list of codes to exclude

    Returns:
        dict[int, Patient]: Dictionary mapping patient IDs to Patient objects containing:
            - criteria_flags: Dictionary of boolean flags for each criterion
            - values: Dictionary of numeric values (age, measurements, etc.)

    Example config:
        {
            "delays": {
                "days": 14,
                "code_groups": ["D/", "RD/"]
            },
            "criteria_definitions": {
                "type2_diabetes": {
                    "codes": ["^D/C11.*"]
                },
                "HbA1c": {
                    "codes": ["(?i).*hba1c.*"],
                    "threshold": 7.0,
                    "operator": ">="
                }
            }
        }
    """
    patients = {}
    delays_config = config.get(DELAYS, {})

    for _, row in index_dates.iterrows():
        patient = Patient(row[PID_COL], row.index_date)
        patient_data = df[df[PID_COL] == patient.subject_id]

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
                patient.criteria_flags[criteria] = match_codes(
                    patient_data,
                    criteria_cfg,
                    delays_config,
                    patient.index_date,
                    min_timestamp,
                )

        patients[patient.subject_id] = patient

    return patients
