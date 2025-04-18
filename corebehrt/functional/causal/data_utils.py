import pandas as pd

from corebehrt.constants.data import PID_COL


def align_df_with_pids(df: pd.DataFrame, pids: list) -> pd.DataFrame:
    """
    Aligns a DataFrame with a list of patient IDs.
    The DataFrame must have the PID_COL column.
    The result will have the same order as the list of patient IDs.

    Args:
        df: The DataFrame to align.
        pids: The list of patient IDs to align with.

    Returns:
        The aligned DataFrame.
    """
    df2 = df.set_index(PID_COL)
    df2 = df2.reindex(pids)
    return df2.reset_index()
