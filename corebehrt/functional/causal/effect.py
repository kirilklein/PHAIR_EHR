import pandas as pd
import numpy as np
from corebehrt.constants.data import PID_COL
from corebehrt.constants.causal.data import (
    EXPOSURE_COL,
    SIMULATED_PROBAS_CONTROL,
    SIMULATED_PROBAS_EXPOSED,
)


def compute_effect_from_ite(
    ite_df: pd.DataFrame, analysis_pids: np.ndarray, outcome_name: str
) -> float:
    """
    Computes the true effect from ITE data for the analysis cohort only.

    Args:
        ite_df: DataFrame containing ITE data with patient IDs
        analysis_pids: Array of patient IDs in the analysis cohort
        outcome_name: Name of the outcome

    Returns:
        float: Mean ITE for the analysis cohort
    """

    # Filter ITE data to analysis cohort only
    analysis_ite_df = ite_df[ite_df[PID_COL].isin(analysis_pids)]

    ite_col = f"ite_{outcome_name}"
    if ite_col not in analysis_ite_df.columns:
        raise ValueError(f"ITE column {ite_col} not found in ITE data")

    # Return mean ITE for selected patients
    return analysis_ite_df[ite_col].mean()


def compute_effect_from_counterfactuals(df: pd.DataFrame, effect_type: str) -> float:
    """
    Computes the effect from counterfactual probabilities (not binary outcomes).

    Args:
        df (pd.DataFrame): DataFrame containing columns 'P1', 'P0', and 'treatment'.
        effect_type (str): The type of effect to compute. Options are 'ATE', 'ATT', 'ATC', 'RR', 'OR'.

    Returns:
        float: The computed effect.

    Raises:
        ValueError: If the effect type is not recognized.
    """
    # Use probabilities instead of binary outcomes for true causal effect
    y1_mean = df[SIMULATED_PROBAS_EXPOSED].mean()
    y0_mean = df[SIMULATED_PROBAS_CONTROL].mean()

    if effect_type == "ATE":
        effect = y1_mean - y0_mean
    elif effect_type in ["ATT", "ATC"]:
        treated_flag = 1 if effect_type == "ATT" else 0
        subset = df[df[EXPOSURE_COL] == treated_flag]
        effect = (
            subset[SIMULATED_PROBAS_EXPOSED].mean()
            - subset[SIMULATED_PROBAS_CONTROL].mean()
        )
    elif effect_type == "RR":
        effect = (y1_mean + 1) / (y0_mean + 1)
    elif effect_type == "OR":
        effect = (y1_mean / (1 - y1_mean)) / (y0_mean / (1 - y0_mean))
    else:
        raise ValueError(f"Effect type '{effect_type}' is not recognized.")

    return effect
