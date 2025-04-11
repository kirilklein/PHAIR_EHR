import re

import numpy as np
import pandas as pd

from corebehrt.constants.cohort import (
    AGE_AT_INDEX_DATE,
    CRITERION_FLAG,
    DELAY,
    FINAL_MASK,
    INDEX_DATE,
    MAX_TIME,
    MIN_TIME,
    NUMERIC_VALUE,
    NUMERIC_VALUE_SUFFIX,
)
from corebehrt.constants.data import (
    BIRTH_CODE,
    BIRTHDATE_COL,
    CONCEPT_COL,
    PID_COL,
    TIMESTAMP_COL,
)


def compute_age_at_index_date(
    index_dates: pd.DataFrame, events: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute the patient age at index date.
    Extracts birth dates from events and merges them with index_dates.
    Returns a DataFrame with columns: subject_id and age_at_index_date.
    """
    birth_dates = get_birth_date_for_each_patient(events)
    birth_dates_df = birth_dates.reset_index().rename(
        columns={TIMESTAMP_COL: BIRTHDATE_COL}
    )
    merged = index_dates.merge(birth_dates_df, on=PID_COL, how="left")
    merged[AGE_AT_INDEX_DATE] = (
        merged[INDEX_DATE] - merged[BIRTHDATE_COL]
    ).dt.days / 365.25
    return merged[[PID_COL, AGE_AT_INDEX_DATE]]


def get_birth_date_for_each_patient(events: pd.DataFrame) -> pd.Series:
    """
    Extract the birth date for each patient from the events DataFrame.
    Assumes that birth date events have CONCEPT_COL equal to "DOB".
    Returns a Series with index=subject_id and values as birth_date.
    """
    # Here we take the first occurrence (earliest time) of a DOB event.
    birth_dates = (
        events[events[CONCEPT_COL] == BIRTH_CODE].groupby(PID_COL)[TIMESTAMP_COL].min()
    )
    return birth_dates


def compute_delay_column(
    df: pd.DataFrame, code_groups: list, delay_in_days: int
) -> pd.DataFrame:
    """
    Compute a 'delay' column in df based on whether its code starts with any prefix in delays_config["code_groups"].
    If a match is found the delay (in days) is set to delays_config["days"], otherwise 0.
    """
    if code_groups:
        prefixes = tuple(code_groups)
        df[DELAY] = np.where(df[CONCEPT_COL].str.startswith(prefixes), delay_in_days, 0)
    else:
        df[DELAY] = 0
    return df


def compute_time_window_columns(
    df: pd.DataFrame, time_window_days: float = 36500
) -> pd.DataFrame:
    """
    Compute two new columns:
      - 'min_time': index_date minus the time_window_days (default 36500 days if not specified).
      - 'max_time': index_date plus the computed delay (if delay > 0; otherwise, just index_date).
    """
    df[MIN_TIME] = df[INDEX_DATE] - pd.to_timedelta(time_window_days, unit="D")
    df[MAX_TIME] = np.where(
        df[DELAY] > 0,
        df[INDEX_DATE] + pd.to_timedelta(df[DELAY], unit="D"),
        df[INDEX_DATE],
    )
    return df


def compute_code_masks(df: pd.DataFrame, codes: list, exclude_codes: list) -> pd.Series:
    """
    Build a Boolean mask for allowed codes and then exclude any matching exclusion patterns.

    Args:
        df: DataFrame with columns [subject_id, time, code, numeric_value, ...]
        codes: List of allowed codes
        exclude_codes: List of exclusion codes
    Returns:
        Boolean mask indicating whether each event's code is allowed.
    """
    if codes:
        allowed_regex = re.compile("|".join(codes))
        allowed_mask = df[CONCEPT_COL].str.contains(allowed_regex, na=False)
    else:
        allowed_mask = pd.Series(False, index=df.index)

    if exclude_codes:
        exclude_regex = re.compile("|".join(exclude_codes))
        exclude_mask = df[CONCEPT_COL].str.contains(exclude_regex, na=False)
    else:
        exclude_mask = pd.Series(False, index=df.index)

    return allowed_mask & (~exclude_mask)


def merge_index_dates(events: pd.DataFrame, index_dates: pd.DataFrame) -> pd.DataFrame:
    """
    Merge index_dates into the events DataFrame so that each event row gets its corresponding index_date.
    """
    index_dates = index_dates.rename(columns={TIMESTAMP_COL: "index_date"})
    return events.merge(index_dates, on=PID_COL, how="left")


def compute_time_mask(df: pd.DataFrame) -> pd.Series:
    """
    Create a Boolean mask indicating whether each event's time is between min_timestamp and max_timestamp.
    """
    return (df[TIMESTAMP_COL] >= df[MIN_TIME]) & (df[TIMESTAMP_COL] <= df[MAX_TIME])


def rename_result(df: pd.DataFrame, criterion: str, has_numeric: bool) -> pd.DataFrame:
    """Rename CRITERION_FLAG and, if applicable, NUMERIC_VALUE columns to use the criterion name."""
    new_cols = {CRITERION_FLAG: criterion}
    if has_numeric:
        new_cols[NUMERIC_VALUE] = criterion + NUMERIC_VALUE_SUFFIX
    return df.rename(columns=new_cols)[
        [PID_COL, criterion]
        + ([criterion + NUMERIC_VALUE_SUFFIX] if has_numeric else [])
    ]


def extract_numeric_values(
    df: pd.DataFrame,
    flag_df: pd.DataFrame,
    min_value: float = None,
    max_value: float = None,
) -> pd.DataFrame:
    """
    Extract the most recent numeric value for each patient where FINAL_MASK is True
    and the numeric value falls within [min_value, max_value] if specified.

    Args:
        df: DataFrame containing the filtered events with FINAL_MASK and NUMERIC_VALUE columns.
        flag_df: DataFrame containing the patient IDs and criterion flags (based solely on code matching).
        min_value: Optional minimum threshold for numeric_value.
        max_value: Optional maximum threshold for numeric_value.

    Returns:
        DataFrame with PID_COL, CRITERION_FLAG, and NUMERIC_VALUE columns.
        The CRITERION_FLAG is updated to True only if a numeric value in the required range is found.
    """
    # Start with events that are marked True by the final mask and have a numeric value.
    num_df = df[(df[FINAL_MASK]) & (df[NUMERIC_VALUE].notna())].copy()

    # Apply numeric range filtering, if thresholds are specified.
    if min_value is not None:
        num_df = num_df[num_df[NUMERIC_VALUE] >= min_value]
    if max_value is not None:
        num_df = num_df[num_df[NUMERIC_VALUE] <= max_value]

    # Group by patient and select the most recent event (by TIMESTAMP_COL)
    if not num_df.empty:
        num_df = num_df.sort_values(TIMESTAMP_COL).groupby(PID_COL).last().reset_index()
        num_df = num_df[[PID_COL, NUMERIC_VALUE]]
        result = flag_df.merge(num_df, on=PID_COL, how="left")
    else:
        result = flag_df.copy()
        result[NUMERIC_VALUE] = None

    # For numeric criteria, update the criterion flag: it is True only if a numeric value in the desired range exists.
    result[CRITERION_FLAG] = result[NUMERIC_VALUE].notna()
    return result
