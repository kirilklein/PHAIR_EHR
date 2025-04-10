import re
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import pandas as pd

from corebehrt.constants.cohort import (
    CODE_ENTRY,
    CODE_GROUPS,
    DAYS,
    EXCLUDE_CODES,
    NUMERIC_VALUE,
    OPERATOR,
    THRESHOLD,
    USE_PATTERNS,
)
from corebehrt.constants.data import CONCEPT_COL, TIMESTAMP_COL


def matches_pattern(code: str, patterns: list[str]) -> bool:
    """
    Check if any of the patterns match the code.
    """
    return any(re.match(pat, code) for pat in patterns)


def evaluate_numeric_criteria(
    patient_data: pd.DataFrame,
    criteria_cfg: dict,
    index_date: datetime,
    min_timestamp: datetime,
) -> Tuple[Optional[float], bool]:
    pattern = criteria_cfg[CODE_ENTRY][0]  # Assumes single numeric pattern
    numeric_data = patient_data[
        patient_data[CONCEPT_COL].str.contains(pattern, case=False, na=False)
    ]
    numeric_data = numeric_data[
        (numeric_data[TIMESTAMP_COL] >= min_timestamp)
        & (numeric_data[TIMESTAMP_COL] < index_date)
    ]

    if not numeric_data.empty:
        if NUMERIC_VALUE not in numeric_data.columns:
            raise ValueError(
                "Data is missing the numeric_value column required for numeric criteria."
            )
        latest = numeric_data.sort_values(TIMESTAMP_COL).iloc[-1]
        value = latest.numeric_value
        threshold = criteria_cfg[THRESHOLD]
        operator = criteria_cfg[OPERATOR]

        operators = {
            ">=": lambda x, y: x >= y,
            "<=": lambda x, y: x <= y,
            ">": lambda x, y: x > y,
            "<": lambda x, y: x < y,
        }
        if operator in operators:
            return value, operators[operator](value, threshold)
        else:
            raise ValueError(f"Invalid operator: {operator}")

    return None, False


def get_code_delay(code: str, delays: dict) -> int:
    """
    Get the delay for a specific code based on code groups.
    Code groups are simple string prefixes (e.g., "D/", "RD/")

    Args:
        code: The regex pattern or actual code to check
        delays: Dictionary with days and code_groups

    Returns:
        int: Delay in days (0 if no matching group)
    """
    code_groups = delays.get(CODE_GROUPS, [])
    for prefix in code_groups:
        if code.startswith(prefix):
            return delays[DAYS]
    return 0


def group_codes_by_delay(codes: list, delays: dict) -> dict:
    """
    Group regex patterns by their delay values based on their code prefixes.

    Args:
        codes: List of regex patterns from criteria
        delays: Dictionary with days and code_groups

    Returns:
        dict: Dictionary mapping delays to lists of patterns
    """
    delay_groups = {}
    for pattern in codes:
        # Remove regex start marker if present to check prefix
        check_pattern = pattern[1:] if pattern.startswith("^") else pattern

        delay = get_code_delay(check_pattern, delays)
        if delay not in delay_groups:
            delay_groups[delay] = []
        delay_groups[delay].append(pattern)

    return delay_groups


def match_codes(
    patient_data: pd.DataFrame,
    criteria_cfg: dict,
    delays_config: dict,
    index_date: datetime,
    min_timestamp: datetime,
) -> bool:
    """
    Match codes considering different delays for different code patterns.

    Args:
        patient_data: DataFrame with patient events
        criteria_cfg: Criteria configuration containing codes and exclude_codes
        delays_config: Configuration containing code_groups
        index_date: Reference date
        min_timestamp: Earliest date to consider
    """
    codes = criteria_cfg[CODE_ENTRY]
    exclude_codes = criteria_cfg.get(EXCLUDE_CODES, [])

    # Group codes by their delays
    delay_groups = group_codes_by_delay(codes, delays=delays_config)

    for delay_days, delayed_codes in delay_groups.items():
        max_timestamp = (
            (index_date + timedelta(days=delay_days)) if delay_days else index_date
        )

        filtered_data = patient_data[
            (patient_data[TIMESTAMP_COL] >= min_timestamp)
            & (patient_data[TIMESTAMP_COL] <= max_timestamp)
        ]
        if filtered_data.empty:
            continue

        # Exclude codes in a vectorized way: compile a combined regex.
        if exclude_codes:
            exclude_regex = re.compile("|".join(exclude_codes), flags=re.IGNORECASE)
            # Keep rows that do not contain any of the exclude codes.
            filtered_data = filtered_data[
                ~filtered_data[CONCEPT_COL].str.contains(exclude_regex, na=False)
            ]
            if filtered_data.empty:
                continue

        # Pre-compile the regex for the delayed_codes.
        code_regex = re.compile("|".join(delayed_codes), flags=re.IGNORECASE)
        if filtered_data[CONCEPT_COL].str.contains(code_regex, na=False).any():
            return True

    return False
