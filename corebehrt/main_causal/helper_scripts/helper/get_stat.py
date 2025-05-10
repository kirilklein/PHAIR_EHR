from dataclasses import dataclass, field
from typing import Dict, Literal, Set

import pandas as pd

from corebehrt.constants.causal.data import EXPOSURE_COL, PS_COL
from corebehrt.constants.data import PID_COL

# Type definitions
GroupType = Literal["Overall", "Exposed", "Control"]

SPECIAL_COLS = {PID_COL, EXPOSURE_COL, PS_COL}

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

    Args:
        df: DataFrame containing cohort data
        config: Configuration for statistics calculation and formatting
        return_raw: Whether to include raw statistics in the result

    Returns:
        Dictionary with formatted (and optionally raw) statistics
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

    Args:
        s: Series to compute statistics for
        n: Total number of rows in the DataFrame
        criterion: Name of the criterion (column)
        group: Group identifier ("Overall", "Exposed", or "Unexposed")

    Returns:
        Dictionary with criterion, group, count, fraction, mean, and std statistics
    """
    # Only consider non-null values
    non_null = s.dropna()

    # Initialize stats dictionary with criterion and group
    stats = {
        "criterion": criterion,
        "group": group,
        "count": 0,
        "percentage": float("nan"),
        "mean": float("nan"),
        "std": float("nan"),
    }

    if pd.api.types.is_bool_dtype(s) or set(non_null.unique()) <= {0, 1, True, False}:
        # Binary column
        count = non_null.sum()
        stats["count"] = int(count)
        stats["percentage"] = 100 * float(count) / n if n > 0 else float("nan")
    elif pd.api.types.is_numeric_dtype(s):
        # Numeric column
        stats["count"] = non_null.count()
        stats["mean"] = non_null.mean()
        stats["std"] = non_null.std()

    return stats


def get_stats(
    df: pd.DataFrame, group: GroupType = "Overall", config: StatConfig = StatConfig()
) -> pd.DataFrame:
    """
    Compute statistics for all relevant columns in a DataFrame.

    Args:
        df: DataFrame containing the criteria columns
        group: Group identifier ("Overall", "Exposed", or "Unexposed")
        config: Configuration for statistics calculation

    Returns:
        A DataFrame with one row per criterion and columns: group, criterion, count, percentage, mean, std
    """
    if df.empty:
        return pd.DataFrame(
            columns=["criterion", "group", "count", "percentage", "mean", "std"]
        )

    # Only consider columns that are not special
    stat_cols = [col for col in df.columns if col not in config.special_cols]
    n = len(df)

    stats_list = []
    for col in stat_cols:
        stats = get_stats_for_column(df[col], n, col, group)
        stats_list.append(stats)

    return pd.DataFrame(stats_list)


def get_stratified_stats(
    df: pd.DataFrame, config: StatConfig = StatConfig()
) -> pd.DataFrame:
    """
    Compute statistics for the overall cohort and stratified by exposure status.

    Args:
        df: DataFrame containing criteria columns
        config: Configuration for statistics calculation

    Returns:
        DataFrame with statistics for overall, exposed, and unexposed groups
    """
    # Get overall stats
    all_stats = [get_stats(df, "Overall", config)]

    # Add exposure-stratified stats if exposure column exists
    if EXPOSURE_COL in df.columns:
        for exposure_value, group_name in [(1, "Exposed"), (0, "Control")]:
            group_df = df[df[EXPOSURE_COL] == exposure_value]
            if not group_df.empty:
                group_stats = get_stats(group_df, group_name, config)
                all_stats.append(group_stats)

    # Combine all stats
    stats_df = pd.concat(all_stats, ignore_index=True) if all_stats else pd.DataFrame()
    stats_df.sort_values(by=["criterion", "group"], ascending=True, inplace=True)
    stats_df.reset_index(drop=True, inplace=True)
    return stats_df


def format_stats_table(
    stats_df: pd.DataFrame, config: StatConfig = StatConfig()
) -> pd.DataFrame:
    """
    Format statistics table for better readability.

    Args:
        stats_df: DataFrame with statistics
        config: Configuration for output formatting

    Returns:
        Formatted DataFrame with readable statistics
    """
    if stats_df.empty:
        return stats_df

    # Make a copy to avoid modifying the original
    formatted_df = stats_df.copy()

    # Format numeric columns
    for col in ["mean", "std"]:
        formatted_df[col] = formatted_df[col].apply(
            lambda x: f"{x:.{config.decimal_places}f}" if pd.notna(x) else "N/A"
        )

    # Format percentage
    formatted_df["percentage"] = formatted_df["percentage"].apply(
        lambda x: f"{x:.{config.percentage_decimal_places}f}%" if pd.notna(x) else "N/A"
    )

    # Select and reorder columns
    result = formatted_df[["group", "criterion", "count", "percentage", "mean", "std"]]

    return result


