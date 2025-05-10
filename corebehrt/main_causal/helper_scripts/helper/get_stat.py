import logging
from os.path import join
from typing import Dict

import numpy as np
import pandas as pd
import torch

from corebehrt.constants.causal.data import EXPOSURE_COL, PROBAS, PS_COL, TARGETS
from corebehrt.constants.causal.paths import (
    CALIBRATED_PREDICTIONS_FILE,
    CRITERIA_FLAGS_FILE,
    PS_PLOT_FILE,
    STATS_FILE_BINARY,
    STATS_FILE_NUMERIC,
    STATS_RAW_FILE_BINARY,
    STATS_RAW_FILE_NUMERIC,
)
from corebehrt.constants.causal.stats import (
    BINARY,
    CONTROL,
    EXPOSED,
    FORMATTED,
    NUMERIC,
    OVERALL,
    RAW,
)
from corebehrt.constants.data import PID_COL
from corebehrt.constants.paths import PID_FILE
from corebehrt.functional.cohort_handling.stats import (
    StatConfig,
    effective_sample_size,
    format_stats_table,
    get_stratified_stats,
)
from corebehrt.functional.cohort_handling.ps_stats import (
    common_support_interval,
    overlap_coefficient,
    ks_statistic,
    standardized_mean_difference,
)


def check_ps_columns(criteria: pd.DataFrame):
    if PS_COL not in criteria.columns:
        raise ValueError(f"PS_COL {PS_COL} not found in criteria")
    if EXPOSURE_COL not in criteria.columns:
        raise ValueError(f"EXPOSURE_COL {EXPOSURE_COL} not found in criteria")


def ps_plot(criteria: pd.DataFrame, save_path: str, filename: str):
    from CausalEstimate.vis.plotting import plot_hist_by_groups

    bin_edges = np.percentile(criteria[PS_COL], [0.1, 99.9])
    fig, _ = plot_hist_by_groups(
        df=criteria,
        value_col=PS_COL,
        group_col=EXPOSURE_COL,
        group_values=(0, 1),
        group_labels=("Control", "Exposed"),
        bin_edges=bin_edges,
        normalize=True,
        alpha=0.5,
    )
    fig.savefig(join(save_path, filename), dpi=200)


def analyze_cohort(
    df: pd.DataFrame, decimal_places: int = 2, percentage_decimal_places: int = 1
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Analyze cohort and return formatted (and optionally raw) binary and numeric stats.
    """
    config = StatConfig(
        decimal_places=decimal_places,
        percentage_decimal_places=percentage_decimal_places,
    )
    raw_stats = get_stratified_stats(df, config)
    result = {RAW: raw_stats}
    formatted_stats = format_stats_table(raw_stats, config)
    result[FORMATTED] = formatted_stats
    return result


def analyze_cohort_with_weights(
    df: pd.DataFrame,
    weights_col: str,
    decimal_places: int = 2,
    percentage_decimal_places: int = 1,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Analyze cohort with weights and return formatted (and optionally raw) binary and numeric stats."""
    config = StatConfig(
        decimal_places=decimal_places,
        percentage_decimal_places=percentage_decimal_places,
        weights_col=weights_col,
    )
    raw_stats = get_stratified_stats(df, config)
    result = {RAW: raw_stats}
    formatted_stats = format_stats_table(raw_stats, config)
    result[FORMATTED] = formatted_stats
    return result


def positivity_summary(ps: pd.Series, exposure: pd.Series) -> pd.DataFrame:
    """
    Gather all overlap/positivity diagnostics into a single-row DataFrame.
    """
    cs = common_support_interval(ps, exposure)
    ovl = overlap_coefficient(ps, exposure)
    ks_stat, ks_pval = ks_statistic(ps, exposure)
    std_diff = standardized_mean_difference(ps, exposure)

    data = {
        "cs_low": cs["cs_low"],
        "cs_high": cs["cs_high"],
        "pct_outside_control": cs["pct_outside_control"],
        "pct_outside_treated": cs["pct_outside_treated"],
        "overlap_coefficient": ovl,
        "ks_statistic": ks_stat,
        "ks_pvalue": ks_pval,
        "std_mean_diff": std_diff,
    }
    return pd.DataFrame([data])


def print_stats(stats: Dict[str, pd.DataFrame]):
    """Print formatted statistics tables."""
    print("================================================")
    print("Formatted stats:")
    print(stats[FORMATTED][BINARY].head(30))
    print(stats[FORMATTED][NUMERIC].head(30))


def save_stats(stats: Dict[str, pd.DataFrame], save_path: str, weighted: bool = False):
    """Save statistics tables to csv files."""
    if weighted:
        prefix = "weighted_"
    else:
        prefix = ""
    stats[FORMATTED][BINARY].to_csv(
        join(save_path, prefix + STATS_FILE_BINARY), index=False
    )
    stats[RAW][BINARY].to_csv(
        join(save_path, prefix + STATS_RAW_FILE_BINARY), index=False
    )
    stats[FORMATTED][NUMERIC].to_csv(
        join(save_path, prefix + STATS_FILE_NUMERIC), index=False
    )
    stats[RAW][NUMERIC].to_csv(
        join(save_path, prefix + STATS_RAW_FILE_NUMERIC), index=False
    )


def load_data(
    criteria_path: str,
    cohort_path: str,
    ps_calibrated_predictions_path: str,
    outcome_model_path: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Load and merge cohort criteria, patient IDs, propensity scores, and predictions as needed."""

    # Load main criteria DataFrame
    logger.info("Loading criteria DataFrame")
    criteria = pd.read_csv(join(criteria_path, CRITERIA_FLAGS_FILE))
    logger.info(f"Loaded {len(criteria)} criteria")

    # Optionally filter by patient IDs
    if cohort_path:
        pids = torch.load(join(cohort_path, PID_FILE))
        logger.info(f"Loaded {len(pids)} patient IDs")
        criteria = criteria[criteria[PID_COL].isin(pids)]
        logger.info(f"Filtered criteria to {len(criteria)} patients")

    # Optionally merge propensity scores and exposures
    if ps_calibrated_predictions_path:
        ps_path = join(ps_calibrated_predictions_path, CALIBRATED_PREDICTIONS_FILE)
        ps_df = pd.read_csv(ps_path).rename(
            columns={TARGETS: EXPOSURE_COL, PROBAS: PS_COL}
        )
        ps_df = _convert_to_int(ps_df, EXPOSURE_COL)
        criteria = pd.merge(criteria, ps_df, on=PID_COL, how="left")
        logger.info("Merged with propensity scores and exposures")

    # Optionally merge predictions and targets
    if outcome_model_path:
        outcome_path = join(outcome_model_path, CALIBRATED_PREDICTIONS_FILE)
        outcome_df = pd.read_csv(outcome_path)[[PID_COL, TARGETS]]
        outcome_df = _convert_to_int(outcome_df, TARGETS)
        criteria = pd.merge(criteria, outcome_df, on=PID_COL, how="left")
        logger.info("Merged with predictions and targets")

    return criteria


def get_effective_sample_size_df(df: pd.DataFrame, weights_col: str) -> pd.DataFrame:
    """
    Compute effective sample size for overall, treated (cases), and controls.
    Returns a DataFrame with columns: group, effective_sample_size
    """

    groups = [
        (OVERALL, df),
        (EXPOSED, df[df[EXPOSURE_COL] == 1]),
        (CONTROL, df[df[EXPOSURE_COL] == 0]),
    ]
    results = []
    for group_name, subdf in groups:
        w = subdf[weights_col].values
        ess = effective_sample_size(w)
        ess = round(ess, 2)
        results.append({"group": group_name, "effective_sample_size": ess})
    return pd.DataFrame(results)


def _convert_to_int(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Convert a column to integer type."""
    df[col] = df[col].astype(int)
    return df
