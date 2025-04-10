import pandas as pd
from tqdm import tqdm

from corebehrt.constants.data import PID_COL, TIMESTAMP_COL
from corebehrt.functional.cohort_handling.advanced.calculations import calculate_age
from corebehrt.functional.cohort_handling.advanced.checks import check_code_patterns
from corebehrt.functional.cohort_handling.advanced.extract import (
    evaluate_criterion,
    precompute_non_numeric_criteria,
)
from corebehrt.modules.cohort_handling.advanced.data.patient import Patient


def extract_patient_criteria(
    df: pd.DataFrame,
    index_dates: pd.DataFrame,
    criteria_definitions: dict,
    delays_config: dict = None,
    code_patterns: dict = None,
) -> dict[int, Patient]:
    """
    Extract and evaluate inclusion/exclusion criteria for each patient.
    """
    if code_patterns is None:
        code_patterns = {}
    if delays_config is None:
        delays_config = {}

    check_code_patterns(criteria_definitions, code_patterns)

    # Get unique patient IDs in the current shard and filter index_dates
    shard_patient_ids = df[PID_COL].unique()
    relevant_index_dates = index_dates[index_dates[PID_COL].isin(shard_patient_ids)]
    print(f"Processing {len(relevant_index_dates)} patients")

    patients = {}

    # Pre-group the DataFrame by patient id to avoid repeated filtering
    grouped_events = {pid: group for pid, group in df.groupby(PID_COL)}

    precomputed_non_numeric_cfg = precompute_non_numeric_criteria(
        criteria_definitions, code_patterns
    )

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
            value, matched = evaluate_criterion(
                criterion,
                crit_cfg,
                patient_data,
                patient.index_date,
                precomputed_non_numeric_cfg,
                delays_config,
            )

            if value is not None:
                patient.values[criterion] = value
            patient.criteria_flags[criterion] = matched

        patients[pid] = patient

    return patients
