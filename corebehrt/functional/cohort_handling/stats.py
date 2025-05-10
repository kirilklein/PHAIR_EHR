from dataclasses import dataclass, field
from typing import Dict, Literal, Set, Optional

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
    NON_NULL_COUNT,
)
from corebehrt.constants.data import PID_COL

SPECIAL_COLS = {PID_COL, EXPOSURE_COL, PS_COL}
STAT_COLS = [GROUP, CRIT, COUNT, PERCENTAGE, MEAN, STD, MEDIAN, P25, P75]


GroupType = Literal["Overall", "Exposed", "Control"]


def effective_sample_size(w: np.ndarray):
    """
    Compute the effective sample size as dervied in
    Given sample weights w, the effective sample size is defined as
    N_e = (sum(w)^2) / (sum(w^2))

    Shook‐Sa, Bonnie E., and Michael G. Hudgens.
    "Power and sample size for observational studies of point exposure effects." Biometrics 78.1 (2022): 388-398.
    """
    return np.sum(w) ** 2 / np.sum(w**2)


def compute_weights(
    criteria: pd.DataFrame, weights_type: Literal["ATE", "ATT", "ATC"]
) -> pd.Series:
    """Compute weights for the cohort based on the weights type (ATE, ATT, ATC)."""
    exposure = criteria[EXPOSURE_COL]
    ps = criteria[PS_COL]
    if weights_type == "ATE":
        # Treated: 1/p, Control: 1/(1-p)
        weights = exposure / ps + (1 - exposure) / (1 - ps)
    elif weights_type == "ATT":
        # Treated: 1, Control: p/(1-p)
        weights = exposure * 1 + (1 - exposure) * (ps / (1 - ps))
    elif weights_type == "ATC":
        # Control: 1, Treated: (1-p)/p
        weights = (1 - exposure) * 1 + exposure * ((1 - ps) / ps)
    else:
        raise ValueError(f"Unknown weights_type: {weights_type}")
    return weights


@dataclass
class StatConfig:
    """Configuration for statistics calculation."""

    special_cols: Set[str] = field(default_factory=lambda: SPECIAL_COLS)
    decimal_places: int = 2
    percentage_decimal_places: int = 1
    weights_col: Optional[str] = None


def format_stats_table(
    stats_dict: Dict[str, pd.DataFrame], config: StatConfig = StatConfig()
) -> Dict[str, Dict[str, pd.DataFrame]]:
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
            [GROUP, CRIT, MEAN, STD, MEDIAN, P25, P75, NON_NULL_COUNT]
        ]
    else:
        formatted[NUMERIC] = numeric_df
    return formatted


def get_stratified_stats(
    df: pd.DataFrame, config: StatConfig = StatConfig()
) -> Dict[str, Dict[str, pd.DataFrame]]:
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
    s: pd.Series,
    n: int,
    criterion: str,
    group: GroupType,
    weights: Optional[pd.Series] = None,
) -> Dict:
    """Get statistics for a binary column."""
    non_null = s.dropna()
    if weights is not None:
        weights = weights[non_null.index]
        count = (non_null * weights).sum()
        total_weight = weights.sum()
        percentage = (
            100 * float(count) / total_weight if total_weight > 0 else float("nan")
        )
    else:
        count = non_null.sum()
        percentage = 100 * float(count) / n if n > 0 else float("nan")
    return {
        CRIT: criterion,
        GROUP: group,
        COUNT: int(count),
        PERCENTAGE: percentage,
    }


def get_stats_for_numeric_column(
    s: pd.Series, criterion: str, group: GroupType, weights: Optional[pd.Series] = None
) -> Dict:
    non_null = s.dropna()

    if weights is not None:
        # align weights with non-null values
        weights = weights.loc[non_null.index]

        # weighted mean & std
        weighted_mean = np.average(non_null, weights=weights)
        weighted_var = np.average((non_null - weighted_mean) ** 2, weights=weights)
        weighted_std = np.sqrt(weighted_var)

        # weighted quantiles
        # 1) sort data and corresponding weights
        sorted_idx = np.argsort(non_null)
        sorted_data = non_null.iloc[sorted_idx]
        sorted_weights_arr = weights.iloc[sorted_idx].values  # as plain numpy array

        # 2) cumulative sum and normalize
        cumw = np.cumsum(sorted_weights_arr)
        cumw = cumw / cumw[-1]  # now cumw[-1] is the total normalized weight (=1)

        # 3) find positions for quantiles
        median_idx = np.searchsorted(cumw, 0.5)
        p25_idx = np.searchsorted(cumw, 0.25)
        p75_idx = np.searchsorted(cumw, 0.75)

        median = sorted_data.iloc[median_idx]
        p25 = sorted_data.iloc[p25_idx]
        p75 = sorted_data.iloc[p75_idx]

    else:
        weighted_mean = non_null.mean()
        weighted_std = non_null.std()
        median = non_null.median()
        p25 = non_null.quantile(0.25)
        p75 = non_null.quantile(0.75)
    count = non_null.count()

    return {
        CRIT: criterion,
        GROUP: group,
        MEAN: weighted_mean,
        STD: weighted_std,
        MEDIAN: median,
        P25: p25,
        P75: p75,
        NON_NULL_COUNT: count,
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
    weights = df[config.weights_col] if config.weights_col else None

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
            binary_stats.append(get_stats_for_binary_column(s, n, col, group, weights))
        elif pd.api.types.is_numeric_dtype(s):
            numeric_stats.append(get_stats_for_numeric_column(s, col, group, weights))
    return {
        BINARY: pd.DataFrame(binary_stats),
        NUMERIC: pd.DataFrame(numeric_stats),
    }


def _sort_stats_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort statistics tables for binary and numeric columns.
    """
    return df.sort_values(by=[CRIT, GROUP]).reset_index(drop=True)
