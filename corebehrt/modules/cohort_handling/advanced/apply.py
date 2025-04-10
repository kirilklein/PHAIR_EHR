from typing import Dict, Tuple

import pandas as pd

from corebehrt.constants.causal.data import (
    EXCLUDED_BY,
    EXCLUSION,
    FLOW,
    FLOW_AFTER_AGE,
    FLOW_AFTER_MINIMUM_ONE,
    FLOW_AFTER_STRICT,
    FLOW_AFTER_UNIQUE_CODES,
    FLOW_FINAL,
    FLOW_INITIAL,
    INCLUDED,
    STRICT_INCLUSION,
    TOTAL,
    UNIQUE_CODE_LIMITS,
    MAX_COUNT,
)
from corebehrt.constants.cohort import (
    CRITERIA,
    EXCLUSION_CRITERIA,
    INCLUSION_CRITERIA,
    MAX_AGE,
    MIN_AGE,
    MINIMUM_ONE,
    STRICT,
)
from corebehrt.constants.data import AGE_COL

from corebehrt.functional.cohort_handling.advanced.checks import check_criteria_columns
from corebehrt.functional.cohort_handling.advanced.utills import prettify_stats


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

    included, stats = apply_age_criteria(
        included, stats, config.get(MIN_AGE, None), config.get(MAX_AGE, None)
    )
    stats[FLOW][FLOW_AFTER_AGE] = len(included)

    # 2. Apply strict inclusion criteria
    for criterion in config[INCLUSION_CRITERIA][STRICT]:
        included, stats = apply_strict_inclusion_criteria(included, stats, criterion)
    stats[FLOW][FLOW_AFTER_STRICT] = len(included)

    # 3. Apply minimum_one criteria
    if MINIMUM_ONE in config[INCLUSION_CRITERIA]:
        included, stats = apply_minimum_one_criteria(
            included, stats, config[INCLUSION_CRITERIA][MINIMUM_ONE]
        )
        stats[FLOW][FLOW_AFTER_MINIMUM_ONE] = len(included)

    # 4. Apply unique code limits
    if UNIQUE_CODE_LIMITS in config:
        included, stats = apply_unique_code_limits(included, stats, config)
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


def apply_unique_code_limits(
    df: pd.DataFrame, stats: dict, config: dict
) -> Tuple[pd.DataFrame, dict]:
    """
    Apply unique code limits to a DataFrame and update stats.
    Args:
        df (pd.DataFrame): DataFrame with patient criteria
        stats (dict): Statistics dictionary
        config (dict): Configuration with unique code limits

    Returns:
        tuple[pd.DataFrame, dict]: DataFrame with unique code limits applied and stats
    """
    for limit_name, limit_config in config[UNIQUE_CODE_LIMITS].items():
        max_count = limit_config[MAX_COUNT]
        criteria_cols = limit_config.get(CRITERIA, [])

        # Count how many criteria are True for each patient
        unique_count = df[criteria_cols].sum(axis=1)
        exceeds_limit = unique_count > max_count

        # Track exclusions
        excluded_count = exceeds_limit.sum()
        stats[EXCLUDED_BY][UNIQUE_CODE_LIMITS][limit_name] = excluded_count

        # Remove patients exceeding the limit
        df = df[~exceeds_limit]
    return df, stats


def apply_minimum_one_criteria(
    df: pd.DataFrame, stats: dict, criteria: list[str]
) -> Tuple[pd.DataFrame, dict]:
    """At least one of the criteria must be true."""
    minimum_one_mask = df[criteria].any(axis=1)
    stats[EXCLUDED_BY][MINIMUM_ONE] = (~minimum_one_mask).sum()
    df = df[minimum_one_mask]
    return df, stats


def apply_strict_inclusion_criteria(
    df: pd.DataFrame, stats: dict, criterion: str
) -> Tuple[pd.DataFrame, dict]:
    """Apply strict inclusion criteria to a DataFrame and update stats."""
    mask = df[criterion]
    excluded_count = (~mask).sum()
    stats[EXCLUDED_BY][STRICT_INCLUSION][criterion] = excluded_count
    df = df[mask]
    return df, stats


def apply_age_criteria(
    df: pd.DataFrame, stats: dict, min_age: int, max_age: int
) -> Tuple[pd.DataFrame, dict]:
    """Apply age criteria to a DataFrame and update stats."""
    if min_age is not None:
        age_mask = df[AGE_COL] >= min_age
        stats[EXCLUDED_BY][AGE_COL] = (~age_mask).sum()
        df = df[age_mask]

    if max_age is not None:
        age_mask = df[AGE_COL] <= max_age
        stats[EXCLUDED_BY][AGE_COL] = (~age_mask).sum()
        df = df[age_mask]
    return df, stats
