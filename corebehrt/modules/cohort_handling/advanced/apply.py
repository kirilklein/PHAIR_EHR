"""
Module for applying complex inclusion/exclusion criteria to patient cohorts.

This module provides functions to:
1. Apply boolean expressions for inclusion/exclusion criteria (e.g., "diabetes & (stroke | mi)")
2. Track patient flow statistics through each filtering step
3. Apply limits on combinations of criteria (e.g., max number of medications)

Main function `apply_criteria_with_stats` returns both the filtered cohort and detailed statistics:
- Patients excluded by each inclusion criterion
- Patients excluded by each exclusion criterion
- Total patients at each stage

Example usage:
```python
df, stats = apply_criteria_with_stats(
    df=patient_criteria_df,
    inclusion_expression="type2_diabetes & (mi | stroke)",
    exclusion_expression="cancer | pregnancy",
)
```
"""

from typing import Dict, List, Tuple

import pandas as pd

from corebehrt.constants.cohort import (
    EXCLUDED_BY_EXCLUSION_CRITERIA,
    EXCLUDED_BY_INCLUSION_CRITERIA,
    FINAL_INCLUDED,
    INITIAL_TOTAL,
    N_EXCLUDED_BY_EXPRESSION,
)
from corebehrt.functional.cohort_handling.advanced.extract import (
    extract_criteria_names_from_expression,
)
from corebehrt.functional.cohort_handling.advanced.utills import print_stats
from corebehrt.constants.data import PID_COL


def apply_criteria_with_stats(
    criteria_flags: pd.DataFrame,
    inclusion_expression: str,
    exclusion_expression: str,
    verbose: bool = True,
) -> Tuple[List[str], Dict]:
    """
    Apply a composite inclusion/exclusion expression and optional unique code limits.

    Args:
        df (pd.DataFrame): Criteria flags per patient.
        inclusion_expression (str): Boolean expression combining inclusion criteria.
        exclusion_expression (str): Boolean expression combining exclusion criteria.
        verbose (bool): If True, prints the flow summary.

    Returns:
        Tuple containing:
            - List of included patient IDs
            - Dictionary with flow statistics.
    """
    stats = {
        INITIAL_TOTAL: len(criteria_flags),
        EXCLUDED_BY_INCLUSION_CRITERIA: {},
        EXCLUDED_BY_EXCLUSION_CRITERIA: {},
        N_EXCLUDED_BY_EXPRESSION: 0,
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
            (~criteria_flags[crit].astype(bool)).sum()
        )
    # For exclusion: count how many patients DO meet the criteria.
    for crit in exclusion_criteria_names:
        stats[EXCLUDED_BY_EXCLUSION_CRITERIA][crit] = int(
            (criteria_flags[crit].astype(bool)).sum()
        )

    # --- Build a local dictionary covering all criteria for eval ---
    all_criteria = set(inclusion_criteria_names) | set(exclusion_criteria_names)
    local_dict = {crit: criteria_flags[crit].astype(bool) for crit in all_criteria}

    # --- Evaluate the inclusion and exclusion expressions ---
    inclusion_mask = pd.eval(inclusion_expression, local_dict=local_dict)
    exclusion_mask = pd.eval(exclusion_expression, local_dict=local_dict)

    # --- Combine expressions: final mask includes patients who satisfy inclusion and do not satisfy exclusion ---
    final_mask = inclusion_mask & ~exclusion_mask
    stats[N_EXCLUDED_BY_EXPRESSION] = len(criteria_flags) - final_mask.sum()

    # --- Subset the DataFrame ---
    included_pids = criteria_flags.loc[final_mask, PID_COL].unique().tolist()
    stats[FINAL_INCLUDED] = len(included_pids)

    if verbose:
        print_stats(stats)

    return included_pids, stats
