"""Core functions for extracting and evaluating patient criteria from medical event data.

This module provides vectorized functions for:
- Computing patient ages at index dates
- Matching medical codes with regex patterns
- Filtering events by time windows
- Extracting numeric values with thresholds
- Evaluating boolean expressions for composite criteria

The functions are designed to work with pandas DataFrames containing medical events
and support efficient processing of large datasets through vectorized operations.
"""

import warnings
import re
from functools import lru_cache

import pandas as pd

from corebehrt.constants.cohort import (
    AGE_AT_INDEX_DATE,
    ALLOWED_OPERATORS,
    CRITERION_FLAG,
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


@lru_cache(maxsize=128)
def _compile_regex(patterns: tuple) -> re.Pattern:
    """
    Cache compiled regex patterns.
    Wrap each pattern in a non-capturing group.
    """
    if len(patterns) == 1:
        return re.compile(patterns[0])
    if any("?i" in p for p in patterns):
        warnings.warn(
            "Case-insensitive matching only supported for single pattern. Split the pattern into single criterion."
        )
        patterns = tuple(p.replace("?i", "") for p in patterns)
    pattern = "|".join(f"(?:{p})" for p in patterns)
    return re.compile(pattern)


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
        allowed_regex = _compile_regex(tuple(codes))
        allowed_mask = df[CONCEPT_COL].str.contains(allowed_regex, na=False)
    else:
        allowed_mask = pd.Series(False, index=df.index)

    if exclude_codes:
        # Convert list to tuple for hashing
        exclude_regex = _compile_regex(tuple(exclude_codes))
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


def compute_time_mask_exclusive(df: pd.DataFrame) -> pd.Series:
    """
    Create a Boolean mask indicating whether each event's time is between min_timestamp and max_timestamp.
    Require the columns TIMESTAMP_COL, MIN_TIME and MAX_TIME to be present in the DataFrame.
    """
    return (df[MIN_TIME] < df[TIMESTAMP_COL]) & (df[TIMESTAMP_COL] < df[MAX_TIME])


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
    extract_value: bool = False,
) -> pd.DataFrame:
    """
    Extract the most recent numeric value for each patient where FINAL_MASK is True
    and the numeric value falls within [min_value, max_value] if specified.

    Args:
        df: DataFrame containing the filtered events with FINAL_MASK and NUMERIC_VALUE columns.
        flag_df: DataFrame containing the patient IDs and criterion flags (based solely on code matching).
        min_value: Optional minimum threshold for numeric_value.
        max_value: Optional maximum threshold for numeric_value.
        extract_value: If True, extract raw numeric values without threshold filtering for statistics.
                      The CRITERION_FLAG will still respect thresholds if specified.

    Returns:
        DataFrame with PID_COL, CRITERION_FLAG, and NUMERIC_VALUE columns.
        The CRITERION_FLAG is updated to True only if a numeric value in the required range is found.
        If extract_value=True, NUMERIC_VALUE contains the raw value regardless of thresholds.
    """
    # Start with events that are marked True by the final mask and have a numeric value.
    mask = (df[FINAL_MASK]) & (df[NUMERIC_VALUE].notna())
    num_df = df[mask].copy()

    # Early return if no matching rows
    if not mask.any():
        result = flag_df.copy()
        result[NUMERIC_VALUE] = None
        result[CRITERION_FLAG] = False
        return result

    if extract_value:
        # When extract_value=True: get most recent value WITHOUT threshold filtering
        # But flag should be based on whether THAT most recent value passes thresholds
        num_df_recent = (
            num_df.sort_values(TIMESTAMP_COL).groupby(PID_COL).tail(1).reset_index()
        )
        num_df_recent = num_df_recent[[PID_COL, NUMERIC_VALUE]]

        # Merge with flag_df
        result = flag_df.merge(num_df_recent, on=PID_COL, how="left")

        # Set flag based on whether the most recent value passes thresholds
        if min_value is not None or max_value is not None:
            # Check if most recent value is in range
            value_col = result[NUMERIC_VALUE]
            in_range = pd.Series(True, index=result.index)

            if min_value is not None:
                in_range &= value_col >= min_value
            if max_value is not None:
                in_range &= value_col <= max_value

            # Flag is True only if value exists AND is in range
            result[CRITERION_FLAG] = value_col.notna() & in_range
        else:
            # No thresholds, flag is True if value exists
            result[CRITERION_FLAG] = result[NUMERIC_VALUE].notna()
    else:
        # Original behavior: find ANY value in range, take most recent of those
        # Apply numeric range filtering for the criterion flag, if thresholds are specified.
        if min_value is not None:
            num_df = num_df[num_df[NUMERIC_VALUE] >= min_value]
        if max_value is not None:
            num_df = num_df[num_df[NUMERIC_VALUE] <= max_value]

        # Group by patient and select the most recent event (by TIMESTAMP_COL)
        if not num_df.empty:
            num_df = (
                num_df.sort_values(TIMESTAMP_COL).groupby(PID_COL).tail(1).reset_index()
            )
            num_df = num_df[[PID_COL, NUMERIC_VALUE]]
            result = flag_df.merge(num_df, on=PID_COL, how="left")
        else:
            result = flag_df.copy()
            result[NUMERIC_VALUE] = None

        # For numeric criteria, update the criterion flag: it is True only if a numeric value in the desired range exists.
        result[CRITERION_FLAG] = result[NUMERIC_VALUE].notna()

    return result


@lru_cache(maxsize=256)
def extract_criteria_names_from_expression(expression: str) -> tuple:
    """Cache parsed expressions since they're often reused."""
    # Add spaces around operators and parentheses to ensure they're separated
    expression = expression.replace("~", " ~ ")
    expression = expression.replace("(", " ( ")
    expression = expression.replace(")", " ) ")

    tokens = expression.split()
    # Exclude operators and parentheses from the results
    return tuple(token for token in tokens if token.lower() not in ALLOWED_OPERATORS)
