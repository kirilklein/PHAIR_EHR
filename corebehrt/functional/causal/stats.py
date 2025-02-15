import pandas as pd


def compute_treatment_outcome_table(
    df: pd.DataFrame, exposure_col: str, outcome_col: str
) -> pd.DataFrame:
    """
    Compute a 2x2 contingency table for binary exposure and outcome.

    Parameters:
    df (pd.DataFrame): The dataframe containing the data
    exposure_col (str): The name of the column containing the binary treatment indicator
    outcome_col (str): The name of the column containing the binary outcome indicator

    Returns:
    pd.DataFrame: A 2x2 contingency table with rows as treatment (0/1) and columns as outcome (0/1)
    """
    table = pd.crosstab(
        df[exposure_col], df[outcome_col], margins=True, margins_name="Total"
    )
    table.index = ["Untreated", "Treated", "Total"]
    table.columns = ["No Outcome", "Outcome", "Total"]
    return table
