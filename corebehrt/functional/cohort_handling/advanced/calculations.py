import pandas as pd
from datetime import datetime
from typing import Optional

from corebehrt.constants.data import (
    BIRTH_CODE,
    CONCEPT_COL,
    TIMESTAMP_COL,
)


def calculate_age(patient_data: pd.DataFrame, index_date: datetime) -> Optional[int]:
    """
    Calculate the age of a patient at a given index date.
    """
    birthdate = patient_data.loc[
        patient_data[CONCEPT_COL] == BIRTH_CODE, TIMESTAMP_COL
    ].min()
    if pd.notna(birthdate):
        return (index_date - birthdate).days // 365.25
    return None
