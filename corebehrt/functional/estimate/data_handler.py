from typing import List

import pandas as pd

from corebehrt.constants.causal.data import (
    CF_PROBAS,
    EXPOSURE_COL,
    OUTCOME,
    PROBAS,
    PROBAS_CONTROL,
    PROBAS_EXPOSED,
    PS_COL,
)
from corebehrt.constants.data import PID_COL
from corebehrt.functional.causal.counterfactuals import expand_counterfactuals

TMLE_ANALYSIS_INPUT_COLS = [
    "initial_effect_1",
    "initial_effect_0",
    "adjustment_1",
    "adjustment_0",
]


def get_outcome_names(df: pd.DataFrame) -> List[str]:
    """
    Extract outcome names from DataFrame columns by removing the outcome prefix.

    Args:
        df: DataFrame containing columns with outcome prefixes

    Returns:
        List of outcome names with prefixes removed

    Example:
        If OUTCOME = "outcome" and df has columns:
        ['patient_id', 'outcome_diabetes', 'outcome_hypertension', 'age']

        Returns: ['diabetes', 'hypertension']
    """
    prefix = OUTCOME + "_"
    return [col.removeprefix(prefix) for col in df.columns if col.startswith(prefix)]


def validate_columns(df: pd.DataFrame, outcome_names: List[str]) -> None:
    """
    Validate that the DataFrame contains all required columns for estimation.
    Args:
        df: DataFrame to validate
        outcome_names: List of outcome names to check for
    """
    required_columns = [PID_COL, EXPOSURE_COL, PS_COL]
    for name in outcome_names:
        for prefix in [OUTCOME, PROBAS, CF_PROBAS]:
            required_columns.append(prefix + "_" + name)
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def prepare_data_for_outcome(df: pd.DataFrame, outcome_name: str) -> pd.DataFrame:
    """
    Prepares the dataframe for a single outcome by creating potential outcome columns.

    This function transforms a multi-outcome DataFrame into a single-outcome format
    by renaming outcome-specific columns to generic names and expanding counterfactuals
    to create potential outcome columns for causal estimation.

    Args:
        df: DataFrame containing multiple outcomes with prefixed columns
        outcome_name: Name of the specific outcome to prepare data for

    Returns:
        DataFrame with generic column names and potential outcome columns

    Example:
        Input DataFrame with outcome_name="diabetes":
        ```
        patient_id | exposure | ps_score | probas_diabetes | cf_probas_diabetes | outcome_diabetes
        1          | 1        | 0.3      | 0.8            | 0.2               | 1
        2          | 0        | 0.7      | 0.1            | 0.9               | 0
        ```

        Output DataFrame:
        ```
        patient_id | exposure | ps_score | probas | cf_probas | outcome | probas_exposed | probas_control
        1          | 1        | 0.3      | 0.8    | 0.2       | 1       | 0.8           | 0.2
        2          | 0        | 0.7      | 0.1    | 0.9       | 0       | 0.9           | 0.1
        ```
    """
    df = df.copy()
    # Define the specific source columns for this outcome
    probas_col = f"{PROBAS}_{outcome_name}"
    cf_probas_col = f"{CF_PROBAS}_{outcome_name}"
    outcome_col = f"{OUTCOME}_{outcome_name}"

    df.rename(
        columns={
            outcome_col: OUTCOME,
            cf_probas_col: CF_PROBAS,
            probas_col: PROBAS,
        },
        inplace=True,
    )
    df = df[[PID_COL, EXPOSURE_COL, PS_COL, PROBAS, CF_PROBAS, OUTCOME]]

    # Expand counterfactuals to create the generic PROBAS_EXPOSED and PROBAS_CONTROL columns
    # The underlying estimators will use these generic columns.
    return expand_counterfactuals(
        df,
        exposure_col=EXPOSURE_COL,
        factual_outcome_col=PROBAS,
        cf_outcome_col=CF_PROBAS,
        outcome_control_col=PROBAS_CONTROL,
        outcome_exposed_col=PROBAS_EXPOSED,
    )


def prepare_tmle_analysis_df(initial_estimates_df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Filters for TMLE results and calculates specific analysis columns.
    Returns a dataframe ready for saving, or None if conditions are not met.
    """
    required_cols = TMLE_ANALYSIS_INPUT_COLS + ["method", "outcome", "effect"]

    if not all(col in initial_estimates_df.columns for col in required_cols):
        print("Skipping TMLE analysis: required columns not found.")
        return None

    tmle_df = initial_estimates_df[initial_estimates_df["method"] == "TMLE"].copy()

    if tmle_df.empty:
        print("Skipping TMLE analysis: no TMLE results found.")
        return None

    print("Preparing TMLE-specific analysis file.")
    tmle_df["initial_effect"] = (
        tmle_df["initial_effect_1"] - tmle_df["initial_effect_0"]
    )
    tmle_df["adjustment"] = tmle_df["adjustment_1"] - tmle_df["adjustment_0"]

    final_cols = required_cols + ["initial_effect", "adjustment"]
    return tmle_df[final_cols]
