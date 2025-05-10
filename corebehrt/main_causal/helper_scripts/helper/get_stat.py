from dataclasses import dataclass, field
from typing import Dict, Literal, Set

import pandas as pd

from corebehrt.constants.causal.data import EXPOSURE_COL, PS_COL
from corebehrt.constants.causal.stats import (COUNT, CRIT, GROUP, MEDIAN,
                                              MEAN, PERCENTAGE, P25, P75, STD)
from corebehrt.constants.data import PID_COL

# Type definitions
GroupType = Literal["Overall", "Exposed", "Control"]

# Output/statistics columns


SPECIAL_COLS = {PID_COL, EXPOSURE_COL, PS_COL}
STAT_COLS = [GROUP, CRIT, COUNT, PERCENTAGE, MEAN, STD, MEDIAN, P25, P75]

@dataclass
class StatConfig:
    """Configuration for statistics calculation."""
    special_cols: Set[str] = field(default_factory=lambda: SPECIAL_COLS)
    decimal_places: int = 2
    percentage_decimal_places: int = 1

def analyze_cohort(
    df: pd.DataFrame, config: StatConfig = StatConfig(), return_raw: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Comprehensive cohort analysis with customizable configuration.
    """
    raw_stats = get_stratified_stats(df, config)
    formatted_stats = format_stats_table(raw_stats, config)
    result = {"formatted": formatted_stats}
    if return_raw:
        result["raw"] = raw_stats
    return result

def get_stats_for_column(
    s: pd.Series, n: int, criterion: str, group: GroupType
) -> Dict:
    """
    Compute statistics for a single column.
    """
    non_null = s.dropna()
    stats = {
        CRIT: criterion,
        GROUP: group,
        COUNT: 0,
        PERCENTAGE: float("nan"),
        MEAN: float("nan"),
        STD: float("nan"),
        MEDIAN: float("nan"),
        P25: float("nan"),
        P75: float("nan"),
    }
    if pd.api.types.is_bool_dtype(s) or set(non_null.unique()) <= {0, 1, True, False}:
        count = non_null.sum()
        stats[COUNT] = int(count)
        stats[PERCENTAGE] = 100 * float(count) / n if n > 0 else float("nan")
    elif pd.api.types.is_numeric_dtype(s):
        stats[COUNT] = non_null.count()
        stats[MEAN] = non_null.mean()
        stats[STD] = non_null.std()
        stats[MEDIAN] = non_null.median()
        stats[P25] = non_null.quantile(0.25)
        stats[P75] = non_null.quantile(0.75)
    return stats

def get_stats(
    df: pd.DataFrame, group: GroupType = "Overall", config: StatConfig = StatConfig()
) -> pd.DataFrame:
    """
    Compute statistics for all relevant columns in a DataFrame.
    """
    if df.empty:
        return pd.DataFrame(columns=STAT_COLS)
    stat_cols = [col for col in df.columns if col not in config.special_cols]
    n = len(df)
    stats_list = [get_stats_for_column(df[col], n, col, group) for col in stat_cols]
    return pd.DataFrame(stats_list)

def get_stratified_stats(
    df: pd.DataFrame, config: StatConfig = StatConfig()
) -> pd.DataFrame:
    """
    Compute statistics for the overall cohort and stratified by exposure status.
    """
    all_stats = [get_stats(df, "Overall", config)]
    if EXPOSURE_COL in df.columns:
        for exposure_value, group_name in [(1, "Exposed"), (0, "Control")]:
            group_df = df[df[EXPOSURE_COL] == exposure_value]
            if not group_df.empty:
                all_stats.append(get_stats(group_df, group_name, config))
    stats_df = pd.concat(all_stats, ignore_index=True) if all_stats else pd.DataFrame()
    stats_df.sort_values(by=[CRIT, GROUP], ascending=True, inplace=True)
    stats_df.reset_index(drop=True, inplace=True)
    return stats_df

def format_stats_table(
    stats_df: pd.DataFrame, config: StatConfig = StatConfig()
) -> pd.DataFrame:
    """
    Format statistics table for better readability.
    """
    if stats_df.empty:
        return stats_df.copy()
    formatted_df = stats_df.copy()
    # Format numeric columns
    for col in [MEAN, STD, MEDIAN, P25, P75]:
        formatted_df[col] = formatted_df[col].apply(
            lambda x: f"{x:.{config.decimal_places}f}" if pd.notna(x) else "N/A"
        )
    # Format percentage
    formatted_df[PERCENTAGE] = formatted_df[PERCENTAGE].apply(
        lambda x: f"{x:.{config.percentage_decimal_places}f}%" if pd.notna(x) else "N/A"
    )
    # Select and reorder columns
    return formatted_df[STAT_COLS]