from typing import List

import pandas as pd
from tqdm import tqdm

from corebehrt.constants.causal.data import END_COL, START_COL
from corebehrt.constants.data import (
    ABSPOS_COL,
    AGE_COL,
    CONCEPT_COL,
    PID_COL,
    SEGMENT_COL,
)
from corebehrt.modules.preparation.causal.dataset import CausalPatientData


def dataframe_to_causal_patient_list(df: pd.DataFrame) -> List[CausalPatientData]:
    """Convert a DataFrame containing patient data into a list of PatientData objects.
    Args:
        df (pd.DataFrame): DataFrame containing patient data with columns:
            - PID: Patient ID
            - concept: Medical concepts/tokens
            - abspos: Absolute positions/timestamps
            - segment: Segment IDs
            - age: Patient ages
    Returns:
        List[PatientData]: List of PatientData objects, where each object contains:
            - pid (str): Patient ID
            - concepts (List[int]): List of medical concept tokens
            - abspos (List[float]): List of absolute positions/timestamps
            - segments (List[int]): List of segment IDs
            - ages (List[float]): List of patient ages
    """
    patients_data = []

    grouped = df.groupby(PID_COL, sort=False)
    loop = tqdm(
        grouped, total=len(grouped), desc="Converting to patient list", mininterval=1
    )
    for pid, group in loop:
        # Convert each column to a Python list
        concepts_list = group[CONCEPT_COL].tolist()
        abspos_list = group[ABSPOS_COL].tolist()
        segments_list = group[SEGMENT_COL].tolist()
        ages_list = group[AGE_COL].tolist()

        # Create a PatientData instance
        patient = CausalPatientData(
            pid=pid,
            concepts=concepts_list,
            abspos=abspos_list,
            segments=segments_list,
            ages=ages_list,
        )

        patients_data.append(patient)

    return patients_data


def abspos_to_binary_outcome(
    follow_ups: pd.DataFrame, outcomes: pd.DataFrame
) -> pd.Series:
    """
    Create binary outcome indicators for patients based on whether outcomes occur within their follow-up periods.

    Args:
        follow_ups: DataFrame with columns 'pid', 'start', 'end' (from prepare_follow_ups_adjusted)
        outcomes: DataFrame with columns 'pid', 'abspos' (absolute position of outcome)

    Returns:
        pd.Series: Binary outcome indicator for each patient.
            - Index: patient IDs (pid)
            - Values: 1 if patient had outcome during follow-up, 0 otherwise
            - Name: 'has_outcome'

    Example:
        >>> follow_ups = pd.DataFrame({
        ...     'pid': [1, 2], 'start': [100, 200], 'end': [400, 500]
        ... })
        >>> outcomes = pd.DataFrame({
        ...     'pid': [1, 1, 2, 3], 'abspos': [50, 150, 250, 350]
        ... })
        >>> abspos_to_binary_outcome(follow_ups, outcomes)
        pid
        1    1    # Patient 1: outcome at 150 within 100-400 ✓
        2    1    # Patient 2: outcome at 250 within 200-500 ✓
        Name: has_outcome, dtype: int64
    """
    # Initialize result with 0 for all patients in follow_ups
    result = pd.Series(0, index=follow_ups[PID_COL], name="has_outcome", dtype=int)
    # result.index.name = PID_COL

    # Merge outcomes with follow_ups
    merged = follow_ups.merge(outcomes, on=PID_COL, how="left")

    # Find outcomes within follow-up periods
    within_followup = (merged[ABSPOS_COL] > merged[START_COL]) & (
        merged[ABSPOS_COL] < merged[END_COL]
    )

    # Get unique patient IDs who had outcomes within follow-up
    patients_with_outcomes = merged.loc[within_followup, PID_COL].unique()

    # Set those patients to 1
    result.loc[patients_with_outcomes] = 1

    return result
