from datetime import datetime, timedelta
from typing import Optional, Tuple, List

import pandas as pd

from corebehrt.constants.cohort import (
    CODE_ENTRY,
    THRESHOLD,
    TIME_WINDOW_DAYS,
    USE_PATTERNS,
)
from corebehrt.functional.cohort_handling.advanced.match import (
    evaluate_numeric_criteria,
    match_codes,
)


def precompute_non_numeric_criteria(
    criteria_definitions: dict, code_patterns: dict
) -> dict:
    """Precompute configurations for non-numeric criteria (ones without THRESHOLD)."""
    precomputed_non_numeric_cfg = {}
    for criterion, crit_cfg in criteria_definitions.items():
        if THRESHOLD not in crit_cfg:
            cfg_copy = crit_cfg.copy()
            cfg_copy[CODE_ENTRY] = get_all_codes_for_criterion(crit_cfg, code_patterns)
            precomputed_non_numeric_cfg[criterion] = cfg_copy
    return precomputed_non_numeric_cfg


def check_code_patterns(criteria_definitions: dict, code_patterns: dict) -> None:
    """Check if any criteria use patterns and validate code_patterns."""
    has_patterns = any(
        USE_PATTERNS in crit_cfg for crit_cfg in criteria_definitions.values()
    )
    if not has_patterns:  # if no patterns are used, return 0
        return
    required_patterns = set()
    for crit_cfg in criteria_definitions.values():
        if USE_PATTERNS in crit_cfg:
            required_patterns.update(crit_cfg[USE_PATTERNS])
    for required_pattern in required_patterns:
        if required_pattern not in code_patterns:
            raise ValueError(
                f"Criteria definitions use pattern {required_pattern} but code_patterns do not contain it"
            )
    return


def evaluate_criterion(
    criterion: str,
    crit_cfg: dict,
    patient_data: pd.DataFrame,
    patient_index_date: datetime,
    precomputed_non_numeric_cfg: dict,
    delays_config: dict,
) -> Tuple[Optional[float], bool]:
    """
    Evaluate a single criterion for a patient.

    Args:
        criterion: Name of the criterion being evaluated
        crit_cfg: Configuration for this criterion
        patient_data: DataFrame containing patient's data
        patient_index_date: Reference date for the patient
        precomputed_non_numeric_cfg: Precomputed configuration for non-numeric criteria
        delays_config: Configuration for code delays

    Returns:
        Tuple[Optional[float], bool]: (value, matched) where value is None for non-numeric criteria
    """
    # Compute the minimum timestamp for filtering
    min_timestamp = patient_index_date - timedelta(
        days=crit_cfg.get(TIME_WINDOW_DAYS, 36500)
    )

    if THRESHOLD in crit_cfg:
        # Numeric criteria evaluation
        value, matched = evaluate_numeric_criteria(
            patient_data, crit_cfg, patient_index_date, min_timestamp
        )
        return value, matched
    else:
        # Non-numeric criteria using precomputed configuration
        pre_cfg = precomputed_non_numeric_cfg[criterion]
        matched = match_codes(
            patient_data,
            pre_cfg,
            delays_config,
            patient_index_date,
            min_timestamp,
        )
        return None, matched


def get_all_codes_for_criterion(
    criterion_config: dict, code_patterns: dict
) -> List[str]:
    """
    Get all code patterns for a criterion, including referenced patterns.

    Args:
        criterion_config: Configuration for a single criterion
        code_patterns: Dictionary of predefined code patterns

    Returns:
        List of all code patterns for this criterion
    """
    # Get both direct codes and pattern-based codes
    direct_codes = criterion_config.get(CODE_ENTRY, [])
    pattern_codes = []

    # Add codes from referenced patterns
    for pattern_name in criterion_config.get(USE_PATTERNS, []):
        if pattern_name in code_patterns:
            pattern_codes.extend(code_patterns[pattern_name][CODE_ENTRY])
        else:
            raise ValueError(
                f"Pattern {pattern_name} not found in code_patterns. Available patterns: {code_patterns.keys()}"
            )

    # Combine both types of codes
    all_codes = direct_codes + pattern_codes

    return all_codes
