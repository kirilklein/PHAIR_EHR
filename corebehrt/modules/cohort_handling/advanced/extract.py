from datetime import timedelta
import pandas as pd
from tqdm import tqdm
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

    # Get unique patient IDs in the current shard and filter index_dates
    shard_patient_ids = df[PID_COL].unique()
    relevant_index_dates = index_dates[index_dates[PID_COL].isin(shard_patient_ids)]
    print(f"Processing {len(relevant_index_dates)} patients")

    patients = {}
    delays_config = config.get(DELAYS, {})

    # Pre-group the DataFrame by patient id to avoid repeated filtering
    grouped_events = {pid: group for pid, group in df.groupby(PID_COL)}

    # Cache criteria definitions locally for speed
    criteria_definitions = config[CRITERIA_DEFINITIONS]

    # Precompute configurations for non-numeric criteria (ones without THRESHOLD)
    precomputed_non_numeric_cfg = {}
    for criterion, crit_cfg in criteria_definitions.items():
        if THRESHOLD not in crit_cfg:
            cfg_copy = crit_cfg.copy()
            cfg_copy[CODE_ENTRY] = get_all_codes_for_criterion(crit_cfg, code_patterns)
            precomputed_non_numeric_cfg[criterion] = cfg_copy

    # Use itertuples (faster than iterrows) for index_date iteration.
    # Assumes that the column names in index_dates match PID_COL and TIMESTAMP_COL.
    for row in tqdm(
        relevant_index_dates.itertuples(index=False),
        total=len(relevant_index_dates),
        desc="Processing patients",
    ):
        pid = getattr(row, PID_COL)
        ts = getattr(row, TIMESTAMP_COL)
        patient = Patient(pid, ts)

        # Skip patients without events.
        if pid not in grouped_events:
            continue
        patient_data = grouped_events[pid]

        # Calculate patient age.
        patient.age = calculate_age(patient_data, patient.index_date)

        # Iterate through each criterion
        for criterion, crit_cfg in criteria_definitions.items():
            # Compute the minimum timestamp for filtering based on the criterion time window.
            min_timestamp = patient.index_date - timedelta(
                days=crit_cfg.get(TIME_WINDOW_DAYS, 36500)
            )

            if THRESHOLD in crit_cfg:
                # Numeric criteria evaluation
                value, matched = evaluate_numeric_criteria(
                    patient_data, crit_cfg, patient.index_date, min_timestamp
                )
                patient.values[criterion] = value
                patient.criteria_flags[criterion] = matched
            else:
                # Non-numeric criteria: use the precomputed configuration.
                pre_cfg = precomputed_non_numeric_cfg[criterion]
                patient.criteria_flags[criterion] = match_codes(
                    patient_data,
                    pre_cfg,
                    delays_config,
                    patient.index_date,
                    min_timestamp,
                )

        patients[pid] = patient

    return patients
