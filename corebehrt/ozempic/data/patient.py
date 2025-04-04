from datetime import datetime

import pandas as pd

from corebehrt.constants.data import PID_COL, AGE_COL
from corebehrt.constants.causal.data import INDEX_DATE


class Patient:
    def __init__(self, subject_id: int, index_date: datetime):
        self.subject_id = subject_id
        self.index_date = index_date
        self.age = None
        self.criteria_flags = {}
        self.values = {}

    def __repr__(self):
        return f"Patient({self.subject_id}, flags={self.criteria_flags}, values={self.values})"


def patients_to_dataframe(patients: dict[int, Patient]) -> pd.DataFrame:
    """
    Convert a dictionary of Patient objects into a DataFrame.
    Each row represents a patient, columns are criteria flags, values, and age.

    Args:
        patients: Dictionary mapping subject_id to Patient objects

    Returns:
        pd.DataFrame: DataFrame with columns for each criterion flag and value
    """
    # Initialize lists to store data
    rows = []

    # Get all possible columns from all patients
    flag_columns = set()
    value_columns = set()

    # First pass: collect all possible column names
    for patient in patients.values():
        flag_columns.update(patient.criteria_flags.keys())
        value_columns.update(patient.values.keys())

    # Second pass: create rows with all columns
    for subject_id, patient in patients.items():
        row = {PID_COL: subject_id, INDEX_DATE: patient.index_date}

        # Add criteria flags (True/False)
        for flag in flag_columns:
            row[flag] = patient.criteria_flags.get(flag, False)

        # Add values (numeric values)
        for val in value_columns:
            col_name = f"value_{val}"
            row[col_name] = patient.values.get(val, None)

        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Set column order
    column_order = [PID_COL, INDEX_DATE, AGE_COL]
    column_order.extend([col for col in sorted(flag_columns)])
    column_order.extend([f"value_{col}" for col in sorted(value_columns)])

    return df[column_order]
