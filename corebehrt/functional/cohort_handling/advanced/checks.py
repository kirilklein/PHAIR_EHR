import pandas as pd

from corebehrt.constants.causal.data import UNIQUE_CODE_LIMITS
from corebehrt.constants.cohort import (
    CRITERIA,
    EXCLUSION_CRITERIA,
    INCLUSION_CRITERIA,
    MINIMUM_ONE,
    STRICT,
    USE_PATTERNS,
)
from corebehrt.constants.data import AGE_COL, PID_COL


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


def check_criteria_columns(df: pd.DataFrame, config: dict) -> None:
    """
    Check if all required criteria columns exist in the DataFrame.
    """
    required_columns = [PID_COL, AGE_COL]

    # Add strict inclusion criteria columns
    for criterion in config[INCLUSION_CRITERIA].get(STRICT, []):
        required_columns.append(criterion)

    # Add minimum_one criteria columns if present
    if MINIMUM_ONE in config[INCLUSION_CRITERIA]:
        for criterion in config[INCLUSION_CRITERIA].get(MINIMUM_ONE, []):
            required_columns.append(criterion)

    # Add unique code limit criteria columns
    if UNIQUE_CODE_LIMITS in config:
        for limit_config in config[UNIQUE_CODE_LIMITS].values():
            criteria = limit_config.get(CRITERIA, [])
            required_columns.extend(criteria)

    # Add exclusion criteria columns
    for criterion in config[EXCLUSION_CRITERIA]:
        required_columns.append(criterion)

    # Check if any columns are missing
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required criteria columns: {missing_columns}")
