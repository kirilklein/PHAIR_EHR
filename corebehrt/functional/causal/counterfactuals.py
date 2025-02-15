import numpy as np
import pandas as pd
import torch


def combine_counterfactuals(exposure, exposed_values, control_values):
    """Combines counterfactual values based on exposure status.

    For each individual, returns the opposite of their actual exposure:
    - If exposed (exposure=1), returns their control value
    - If not exposed (exposure=0), returns their exposed value

    Args:
        exposure (numpy.ndarray): Binary array indicating exposure status (1=exposed, 0=control)
        exposed_values (numpy.ndarray): Values under exposure condition
        control_values (numpy.ndarray): Values under control condition

    Returns:
        numpy.ndarray: Combined array where each element is the counterfactual value
            based on the opposite of the actual exposure status
    """
    return torch.where(exposure == 1, control_values, exposed_values)


def expand_counterfactuals(
    df: pd.DataFrame,
    exposure_col: str,
    cf_probas_col: str,
    outcome_control_col: str,
    outcome_exposed_col: str,
):
    """Expands counterfactual values based on exposure status.

    For each individual:
    - If exposed (exposure==1), their observed outcome under exposure is cf_probas
      and their counterfactual outcome under control is 1 - cf_probas.
    - If not exposed (exposure==0), their observed outcome under control is cf_probas
      and their counterfactual outcome under exposure is 1 - cf_probas.

    Args:
        df (pandas.DataFrame): Input dataframe.
        exposure_col (str): Name of column containing exposure status (1=exposed, 0=control).
        cf_probas_col (str): Name of column containing the observed outcome probability,
                             which corresponds to the actual exposure received.
        outcome_control_col (str): Name of new column for the counterfactual outcome under control.
        outcome_exposed_col (str): Name of new column for the counterfactual outcome under exposure.


    Returns:
        pandas.DataFrame: DataFrame with two new columns:
                          'outcome_under_exposure' and 'outcome_under_control'.
    """
    df = df.copy()
    df[outcome_exposed_col] = np.where(
        df[exposure_col] == 1, df[cf_probas_col], 1 - df[cf_probas_col]
    )
    df[outcome_control_col] = np.where(
        df[exposure_col] == 0, df[cf_probas_col], 1 - df[cf_probas_col]
    )
    return df
