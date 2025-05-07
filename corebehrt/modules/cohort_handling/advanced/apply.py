"""
Module for applying complex inclusion/exclusion criteria to patient cohorts.

This module provides functions to:
1. Apply boolean expressions for inclusion/exclusion criteria (e.g., "diabetes & (stroke | mi)")
2. Track patient flow statistics through each filtering step
3. Apply limits on combinations of criteria (e.g., max number of medications)

Main function `apply_criteria_with_stats` returns both the filtered cohort and detailed statistics:
- Patients excluded by each inclusion criterion
- Patients excluded by each exclusion criterion
- Patients excluded by unique code limits
- Total patients at each stage

Example usage:
```python
df, stats = apply_criteria_with_stats(
    df=patient_criteria_df,
    inclusion_expression="type2_diabetes & (mi | stroke)",
    exclusion_expression="cancer | pregnancy",
    unique_code_limits={
        "medications": {
            "max_count": 2,
            "criteria": ["med_a", "med_b", "med_c"]
        }
    }
)
```
"""

from typing import Dict, Tuple

import pandas as pd

from corebehrt.constants.cohort import (
    CRITERIA,
    EXCLUDED_BY_EXCLUSION_CRITERIA,
    EXCLUDED_BY_INCLUSION_CRITERIA,
    FINAL_INCLUDED,
    INITIAL_TOTAL,
    MAX_COUNT,
    N_EXCLUDED_BY_CODE_LIMITS,
    N_EXCLUDED_BY_EXPRESSION,
)
from corebehrt.functional.cohort_handling.advanced.extract import (
    extract_criteria_names_from_expression,
)
from corebehrt.functional.cohort_handling.advanced.utills import print_stats


def apply_criteria_with_stats(
    df: pd.DataFrame,
    inclusion_expression: str,
    exclusion_expression: str,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply a composite inclusion/exclusion expression and optional unique code limits.

    Args:
        df (pd.DataFrame): Criteria flags per patient.
        inclusion_expression (str): Boolean expression combining inclusion criteria.
        exclusion_expression (str): Boolean expression combining exclusion criteria.
        verbose (bool): If True, prints the flow summary.

    Returns:
        Tuple containing:
            - Filtered DataFrame of included patients
            - Dictionary with flow statistics.
    """
    stats = {
        INITIAL_TOTAL: len(df),
        EXCLUDED_BY_INCLUSION_CRITERIA: {},
        EXCLUDED_BY_EXCLUSION_CRITERIA: {},
        N_EXCLUDED_BY_EXPRESSION: 0,
        N_EXCLUDED_BY_CODE_LIMITS: {},
        FINAL_INCLUDED: 0,
    }

    # --- Extract criteria names from expressions ---
    inclusion_criteria_names = extract_criteria_names_from_expression(
        inclusion_expression
    )
    exclusion_criteria_names = extract_criteria_names_from_expression(
        exclusion_expression
    )
    # --- Compute criteria-specific statistics ---
    # For inclusion: count how many patients DO NOT meet the criteria.
    for crit in inclusion_criteria_names:
        stats[EXCLUDED_BY_INCLUSION_CRITERIA][crit] = int(
            (~df[crit].astype(bool)).sum()
        )
    # For exclusion: count how many patients DO meet the criteria.
    for crit in exclusion_criteria_names:
        stats[EXCLUDED_BY_EXCLUSION_CRITERIA][crit] = int((df[crit].astype(bool)).sum())

    # --- Build a local dictionary covering all criteria for eval ---
    all_criteria = set(inclusion_criteria_names) | set(exclusion_criteria_names)
    local_dict = {crit: df[crit].astype(bool) for crit in all_criteria}

    # --- Evaluate the inclusion and exclusion expressions ---
    inclusion_mask = pd.eval(inclusion_expression, local_dict=local_dict)
    exclusion_mask = pd.eval(exclusion_expression, local_dict=local_dict)

    # --- Combine expressions: final mask includes patients who satisfy inclusion and do not satisfy exclusion ---
    final_mask = inclusion_mask & ~exclusion_mask
    stats[N_EXCLUDED_BY_EXPRESSION] = len(df) - final_mask.sum()

    # --- Subset the DataFrame ---
    included = df[final_mask].copy()
    stats[FINAL_INCLUDED] = len(included)

    if verbose:
        print_stats(stats)

    return included, stats


def apply_unique_code_limits(
    df: pd.DataFrame,
    limits_config: Dict,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Remove patients exceeding the allowed number of positive flags in certain groups.

    Args:
        df (pd.DataFrame): DataFrame of patients.
        limits_config (dict): Configuration, e.g.:
            { "antidiabetics": { "criteria": [...], "max_count": 2 } }

    Returns:
        Tuple: filtered DataFrame and dictionary of counts for exclusions per limit name.
    """
    stats = {}
    for name, cfg in limits_config.items():
        criteria_cols = cfg[CRITERIA]
        max_count = cfg[MAX_COUNT]

        # Sum positive flags and identify patients exceeding the limit.
        pos_counts = df[criteria_cols].sum(axis=1)
        mask_exceeds = pos_counts > max_count

        stats[name] = int(mask_exceeds.sum())
        df = df[~mask_exceeds]  # Remove patients exceeding limit.

    return df, stats
