import pandas as pd

from corebehrt.constants.data import PID_COL


def align_df_with_pids(df: pd.DataFrame, pids: list) -> pd.DataFrame:
    """
    Aligns a DataFrame with a list of patient IDs.

    Args:
        df: The DataFrame to align.
        pids: The list of patient IDs to align with.

    Returns:
        The aligned DataFrame.
    """
    return df[df[PID_COL].isin(pids)].reset_index(drop=True)
