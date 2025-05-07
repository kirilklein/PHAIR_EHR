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

    # Check if PID_COL is in df
    if PID_COL not in df.columns:
        raise ValueError(f"PID_COL is not in the DataFrame: {df.columns}")

    # Validate that all patient IDs exist in the DataFrame
    missing_pids = set(pids) - set(df[PID_COL])
    if missing_pids:
        raise ValueError(
            f"The following patient IDs are not in the DataFrame: {missing_pids}"
        )

    df2 = df.set_index(PID_COL)
    df2 = df2.loc[pids]
    return df2.reset_index()
