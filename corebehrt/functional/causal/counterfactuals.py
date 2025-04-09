import numpy as np
import pandas as pd
import torch


def get_true_outcome(exposure, exposed_values, control_values):
    """
    Returns true outcomes based on exposure status (exposed_values if exposure=1, control_values if exposure=0).

    Args:
        exposure (numpy.ndarray): Binary exposure status (1=exposed, 0=control)
        exposed_values (numpy.ndarray): Values under exposure
        control_values (numpy.ndarray): Values under control

    Returns:
        numpy.ndarray: Combined true outcomes
    """
    return torch.where(exposure == 1, exposed_values, control_values)


def expand_counterfactuals(
    df: pd.DataFrame,
    exposure_col: str,
    factual_outcome_col: str,  # This will use the "probas" column from outcome predictions
    cf_outcome_col: str,  # This is the "cf_probas" column
    outcome_control_col: str,  # Name for the potential outcome column under control
    outcome_exposed_col: str,  # Name for the potential outcome column under treatment
):
    """
    For each individual:
      - If exposed, the outcome under exposure is taken from the factual prediction,
        and the counterfactual (under control) comes from cf_outcome_col.
      - If not exposed, the outcome under control is the factual prediction,
        and the counterfactual (under exposure) comes from cf_outcome_col.
    """
    df = df.copy()
    # For subjects who are exposed, factual prediction (probas) is their potential outcome under exposure.
    # For subjects who are not exposed, their potential outcome under exposure is given by the cf prediction.
    df[outcome_exposed_col] = np.where(
        df[exposure_col] == 1, df[factual_outcome_col], df[cf_outcome_col]
    )
    # For subjects who are not exposed, factual prediction (probas) is their potential outcome under control.
    # For subjects who are exposed, their potential outcome under control is given by the cf prediction.
    df[outcome_control_col] = np.where(
        df[exposure_col] == 0, df[factual_outcome_col], df[cf_outcome_col]
    )
    return df
