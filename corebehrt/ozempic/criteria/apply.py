from typing import Dict, Tuple

import pandas as pd

from corebehrt.constants.data import AGE_COL, PID_COL
from corebehrt.ozempic.utils.definitions import (
    EXCLUSION_CRITERIA,
    INCLUSION_CRITERIA,
    MIN_AGE,
    MAX_AGE,
    MINIMUM_ONE,
    STRICT,
)
from corebehrt.constants.causal.data import (
    EXCLUDED_BY,
    STRICT_INCLUSION,
    EXCLUSION,
    INCLUDED,
    TOTAL,
)


def apply_criteria(df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply inclusion and exclusion criteria to a DataFrame of patients.

    Args:
        df (pd.DataFrame): DataFrame with columns:
            - subject_id: Patient identifier
            - flag_*: Boolean columns for each criterion
            - value_age: Patient age
        config (dict): Configuration containing:
            - min_age: Minimum age requirement
            - inclusion_criteria:
                - strict: List of criteria that must all be True
                - minimum_one: List of criteria where at least one must be True
            - exclusion_criteria: List of criteria that must all be False

    Returns:
        tuple[pd.DataFrame, dict]:
            - DataFrame with only included patients
            - Dictionary with statistics about exclusions
    """

    stats = {
        TOTAL: len(df),
        EXCLUDED_BY: {AGE_COL: 0, STRICT_INCLUSION: 0, MINIMUM_ONE: 0, EXCLUSION: {}},
        INCLUDED: 0,
    }

    # Start with all patients
    included = df.copy()
    # Verify all required criteria columns exist
    check_criteria_columns(included, config)
    # 1. Apply age criterion
    if MIN_AGE in config:
        age_mask = included[AGE_COL] >= config[MIN_AGE]
        stats[EXCLUDED_BY][AGE_COL] = (~age_mask).sum()
        included = included[age_mask]

    if MAX_AGE in config:
        age_mask = included[AGE_COL] <= config[MAX_AGE]
        stats[EXCLUDED_BY][AGE_COL] = (~age_mask).sum()
        included = included[age_mask]

    # 2. Apply strict inclusion criteria
    for criterion in config[INCLUSION_CRITERIA][STRICT]:
        mask = included[criterion] == True
        stats[EXCLUDED_BY][STRICT_INCLUSION] += (~mask).sum()
        included = included[mask]

    # 3. Apply minimum_one criteria
    if MINIMUM_ONE in config[INCLUSION_CRITERIA]:
        # Create combined mask where at least one criterion is True
        minimum_one_mask = False
        for criterion in config[INCLUSION_CRITERIA][MINIMUM_ONE]:
            minimum_one_mask |= included[criterion] == True

        stats[EXCLUDED_BY][MINIMUM_ONE] = (~minimum_one_mask).sum()
        included = included[minimum_one_mask]

    # 4. Apply exclusion criteria
    ## First count the number of patients excluded by each criterion
    for criterion in config[EXCLUSION_CRITERIA]:
        excluded_count = included[criterion].sum()
        stats[EXCLUDED_BY][EXCLUSION][criterion] = excluded_count

    ## Then remove the excluded patients from the included DataFrame
    included = included[included[config[EXCLUSION_CRITERIA]].eq(False).all(axis=1)]

    stats[INCLUDED] = len(included)

    return included, stats


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

    # Add exclusion criteria columns
    for criterion in config[EXCLUSION_CRITERIA]:
        required_columns.append(criterion)

    # Check if any columns are missing
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required criteria columns: {missing_columns}")
