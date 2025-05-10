from dataclasses import dataclass, field
from typing import Dict, Literal, Set

import numpy as np
import pandas as pd

from corebehrt.constants.causal.data import EXPOSURE_COL, PS_COL
from corebehrt.constants.causal.stats import (
    BINARY,
    CONTROL,
    COUNT,
    CRIT,
    EXPOSED,
    GROUP,
    MEAN,
    MEDIAN,
    NUMERIC,
    OVERALL,
    P25,
    P75,
    PERCENTAGE,
    STD,
)
from corebehrt.constants.data import PID_COL

SPECIAL_COLS = {PID_COL, EXPOSURE_COL, PS_COL}
STAT_COLS = [GROUP, CRIT, COUNT, PERCENTAGE, MEAN, STD, MEDIAN, P25, P75]


GroupType = Literal["overall", "exposed", "control"]


def effective_sample_size(w: np.ndarray):
    """
    Compute the effective sample size as dervied in
    Given sample weights w, the effective sample size is defined as
    N_e = (sum(w)^2) / (sum(w^2))

    Shook‐Sa, Bonnie E., and Michael G. Hudgens.
    "Power and sample size for observational studies of point exposure effects." Biometrics 78.1 (2022): 388-398.
    """
    return np.sum(w) ** 2 / np.sum(w**2)


@dataclass
class StatConfig:
    """Configuration for statistics calculation."""

    special_cols: Set[str] = field(default_factory=lambda: SPECIAL_COLS)
    decimal_places: int = 2
    percentage_decimal_places: int = 1


def format_stats_table(
    stats_dict: Dict[str, pd.DataFrame], config: StatConfig = StatConfig()
) -> Dict[str, pd.DataFrame]:
    """
    Format statistics tables for binary and numeric columns.
    """
    formatted = {}
    # Format binary
    binary_df = stats_dict[BINARY].copy()
    if not binary_df.empty:
        binary_df[PERCENTAGE] = binary_df[PERCENTAGE].apply(
            lambda x: f"{x:.{config.percentage_decimal_places}f}%"
            if pd.notna(x)
            else "N/A"
        )
        formatted[BINARY] = binary_df[[GROUP, CRIT, COUNT, PERCENTAGE]]
    else:
        formatted[BINARY] = binary_df
    # Format numeric
    numeric_df = stats_dict[NUMERIC].copy()
    if not numeric_df.empty:
        for col in [MEAN, STD, MEDIAN, P25, P75]:
            numeric_df[col] = numeric_df[col].apply(
                lambda x: f"{x:.{config.decimal_places}f}" if pd.notna(x) else "N/A"
            )
        formatted[NUMERIC] = numeric_df[
            [GROUP, CRIT, COUNT, MEAN, STD, MEDIAN, P25, P75]
        ]
    else:
        formatted[NUMERIC] = numeric_df
    return formatted


def get_stratified_stats(
    df: pd.DataFrame, config: StatConfig = StatConfig()
) -> Dict[str, pd.DataFrame]:
    """
    Compute statistics for the overall cohort and stratified by exposure status,
    split into binary and numeric DataFrames.
    """
    all_binary = []
    all_numeric = []
    # Overall
    stats = get_stats(df, OVERALL, config)
    if not stats[BINARY].empty:
        all_binary.append(stats[BINARY])
    if not stats[NUMERIC].empty:
        all_numeric.append(stats[NUMERIC])
    # By exposure
    if EXPOSURE_COL in df.columns:
        for exposure_value, group_name in [(1, EXPOSED), (0, CONTROL)]:
            group_df = df[df[EXPOSURE_COL] == exposure_value]
            if not group_df.empty:
                stats = get_stats(group_df, group_name, config)
                if not stats[BINARY].empty:
                    all_binary.append(stats[BINARY])
                if not stats[NUMERIC].empty:
                    all_numeric.append(stats[NUMERIC])
    binary_df = (
        pd.concat(all_binary, ignore_index=True) if all_binary else pd.DataFrame()
    )
    numeric_df = (
        pd.concat(all_numeric, ignore_index=True) if all_numeric else pd.DataFrame()
    )
    binary_df = _sort_stats_table(binary_df)
    numeric_df = _sort_stats_table(numeric_df)
    return {BINARY: binary_df, NUMERIC: numeric_df}


def get_stats_for_binary_column(
    s: pd.Series, n: int, criterion: str, group: GroupType
) -> Dict:
    """Get statistics for a binary column."""
    non_null = s.dropna()
    count = non_null.sum()
    return {
        CRIT: criterion,
        GROUP: group,
        COUNT: int(count),
        PERCENTAGE: 100 * float(count) / n if n > 0 else float("nan"),
    }


def get_stats_for_numeric_column(
    s: pd.Series, criterion: str, group: GroupType
) -> Dict:
    non_null = s.dropna()
    return {
        CRIT: criterion,
        GROUP: group,
        COUNT: non_null.count(),
        MEAN: non_null.mean(),
        STD: non_null.std(),
        MEDIAN: non_null.median(),
        P25: non_null.quantile(0.25),
        P75: non_null.quantile(0.75),
    }


def get_stats(
    df: pd.DataFrame, group: GroupType = OVERALL, config: StatConfig = StatConfig()
) -> Dict[str, pd.DataFrame]:
    """
    Compute statistics for all relevant columns in a DataFrame, split into binary and numeric.
    Returns a dict with keys 'binary' and 'numeric'.
    """
    if df.empty:
        return {
            BINARY: pd.DataFrame(columns=[GROUP, CRIT, COUNT, PERCENTAGE]),
            NUMERIC: pd.DataFrame(
                columns=[GROUP, CRIT, COUNT, MEAN, STD, MEDIAN, P25, P75]
            ),
        }
    stat_cols = [col for col in df.columns if col not in config.special_cols]
    n = len(df)
    binary_stats = []
    numeric_stats = []
    for col in stat_cols:
        s = df[col]
        non_null = s.dropna()
        if pd.api.types.is_bool_dtype(s) or set(non_null.unique()) <= {
            0,
            1,
            True,
            False,
        }:
            binary_stats.append(get_stats_for_binary_column(s, n, col, group))
        elif pd.api.types.is_numeric_dtype(s):
            numeric_stats.append(get_stats_for_numeric_column(s, col, group))
    return {
        BINARY: pd.DataFrame(binary_stats),
        NUMERIC: pd.DataFrame(numeric_stats),
    }


def _sort_stats_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort statistics tables for binary and numeric columns.
    """
    return df.sort_values(by=[CRIT, GROUP]).reset_index(drop=True)
