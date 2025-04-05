from typing import Dict, Tuple

import numpy as np
import pandas as pd

from corebehrt.constants.causal.data import (
    EXCLUDED_BY,
    EXCLUSION,
    FLOW,
    FLOW_AFTER_AGE,
    FLOW_AFTER_MINIMUM_ONE,
    FLOW_AFTER_STRICT,
    FLOW_FINAL,
    FLOW_INITIAL,
    FLOW_AFTER_UNIQUE_CODES,
    INCLUDED,
    STRICT_INCLUSION,
    UNIQUE_CODE_LIMITS,
    TOTAL,
)
from corebehrt.constants.cohort import (
    EXCLUSION_CRITERIA,
    INCLUSION_CRITERIA,
    MAX_AGE,
    MIN_AGE,
    MINIMUM_ONE,
    STRICT,
)
from corebehrt.constants.data import AGE_COL, PID_COL


def apply_criteria(df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply inclusion and exclusion criteria to a DataFrame of patients.
    Tracks patient counts at each step for CONSORT diagram creation.

    Args:
        df (pd.DataFrame): DataFrame with patient criteria
        config (dict): Configuration with criteria definitions

    Returns:
        tuple[pd.DataFrame, dict]: Included patients and statistics including flow counts
    """
    check_criteria_columns(
        df, config
    )  # This will raise ValueError if columns are missing

    stats = {
        TOTAL: len(df),
        FLOW: {
            FLOW_INITIAL: len(df),
            FLOW_AFTER_AGE: 0,
            FLOW_AFTER_STRICT: 0,
            FLOW_AFTER_MINIMUM_ONE: 0,
            FLOW_AFTER_UNIQUE_CODES: 0,
            FLOW_FINAL: 0,
        },
        EXCLUDED_BY: {
            AGE_COL: 0,
            STRICT_INCLUSION: {},
            MINIMUM_ONE: 0,
            UNIQUE_CODE_LIMITS: {},
            EXCLUSION: {},
        },
        INCLUDED: 0,
    }

    included = df.copy()

    # 1. Apply age criterion
    if MIN_AGE in config:
        age_mask = included[AGE_COL] >= config[MIN_AGE]
        stats[EXCLUDED_BY][AGE_COL] = (~age_mask).sum()
        included = included[age_mask]

    if MAX_AGE in config:
        age_mask = included[AGE_COL] <= config[MAX_AGE]
        stats[EXCLUDED_BY][AGE_COL] = (~age_mask).sum()
        included = included[age_mask]

    stats[FLOW][FLOW_AFTER_AGE] = len(included)

    # 2. Apply strict inclusion criteria
    for criterion in config[INCLUSION_CRITERIA][STRICT]:
        mask = included[criterion]
        excluded_count = (~mask).sum()
        stats[EXCLUDED_BY][STRICT_INCLUSION][criterion] = excluded_count
        included = included[mask]
    stats[FLOW][FLOW_AFTER_STRICT] = len(included)

    # 3. Apply minimum_one criteria
    if MINIMUM_ONE in config[INCLUSION_CRITERIA]:
        minimum_one_mask = included[config[INCLUSION_CRITERIA][MINIMUM_ONE]].any(axis=1)
        stats[EXCLUDED_BY][MINIMUM_ONE] = (~minimum_one_mask).sum()
        included = included[minimum_one_mask]
    stats[FLOW][FLOW_AFTER_MINIMUM_ONE] = len(included)

    # Apply unique code limits
    if UNIQUE_CODE_LIMITS in config:
        for limit_name, limit_config in config[UNIQUE_CODE_LIMITS].items():
            max_count = limit_config["max_count"]
            criteria_cols = limit_config["criteria"]

            # Count how many criteria are True for each patient
            unique_count = included[criteria_cols].sum(axis=1)
            exceeds_limit = unique_count > max_count

            # Track exclusions
            excluded_count = exceeds_limit.sum()
            stats[EXCLUDED_BY][UNIQUE_CODE_LIMITS][limit_name] = excluded_count

            # Remove patients exceeding the limit
            included = included[~exceeds_limit]

    stats[FLOW][FLOW_AFTER_UNIQUE_CODES] = len(included)

    # 4. Apply exclusion criteria
    for criterion in config[EXCLUSION_CRITERIA]:
        excluded_count = included[criterion].sum()
        stats[EXCLUDED_BY][EXCLUSION][criterion] = excluded_count

    # Remove excluded patients
    included = included[included[config[EXCLUSION_CRITERIA]].eq(False).all(axis=1)]

    stats[FLOW][FLOW_FINAL] = len(included)
    stats[INCLUDED] = len(included)

    return included, prettify_stats(stats)


def check_criteria_columns(df: pd.DataFrame, config: dict) -> None:
    """
    Check if all required criteria columns exist in the DataFrame.
    """
    required_columns = [PID_COL, AGE_COL]

    # Add strict inclusion criteria columns
    for criterion in config[INCLUSION_CRITERIA][STRICT]:
        required_columns.append(criterion)

    # Add minimum_one criteria columns if present
    if MINIMUM_ONE in config[INCLUSION_CRITERIA]:
        for criterion in config[INCLUSION_CRITERIA][MINIMUM_ONE]:
            required_columns.append(criterion)

    # Add unique code limit criteria columns
    if UNIQUE_CODE_LIMITS in config:
        for limit_config in config[UNIQUE_CODE_LIMITS].values():
            required_columns.extend(limit_config["criteria"])

    # Add exclusion criteria columns
    for criterion in config[EXCLUSION_CRITERIA]:
        required_columns.append(criterion)

    # Check if any columns are missing
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required criteria columns: {missing_columns}")


def prettify_stats(stats: dict) -> dict:
    """
    Convert all numpy numeric types to Python native types recursively through the dictionary.

    Args:
        stats (dict): Dictionary containing statistics, potentially with numpy numeric types
                     and nested dictionaries

    Returns:
        dict: Dictionary with all numpy numeric types converted to Python native types
    """

    def convert_value(v):
        if isinstance(v, (np.integer, np.floating)):
            return int(v) if isinstance(v, np.integer) else float(v)
        elif isinstance(v, dict):
            return prettify_stats(v)
        elif isinstance(v, (list, tuple)):
            return type(v)(convert_value(x) for x in v)
        return v

    return {k: convert_value(v) for k, v in stats.items()}
