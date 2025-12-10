import logging
from os.path import join
from typing import Dict

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from corebehrt.constants.causal.data import EXPOSURE_COL, PROBAS, PS_COL, TARGETS
from corebehrt.constants.causal.paths import (
    CALIBRATED_PREDICTIONS_FILE,
    CRITERIA_FLAGS_FILE,
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
from corebehrt.functional.cohort_handling.ps_stats import (
    common_support_interval,
    ks_statistic,
    overlap_coefficient,
    standardized_mean_difference,
)
from corebehrt.functional.cohort_handling.stats import (
    StatConfig,
    effective_sample_size,
    format_stats_table,
    get_stratified_stats,
)
from corebehrt.functional.utils.log import log_table
from corebehrt.functional.utils.azure_save import save_figure_with_azure_copy

logger = logging.getLogger("get_stat")


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
    save_figure_with_azure_copy(fig, join(save_path, filename), dpi=200)


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

    logger.info("Computing positivity summary")
    check_ps_group_variance(ps, exposure)

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


def log_stats(stats: Dict[str, pd.DataFrame]):
    """Log formatted statistics tables."""
    logger.info("================================================")
    logger.info("Formatted stats:")
    log_table(stats[FORMATTED][BINARY].head(30), logger)
    log_table(stats[FORMATTED][NUMERIC].head(30), logger)


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

def compute_smd(df):
    """Compute the standardized mean difference for a dataframe between Exposed and Control."""
    p1 = df['Exposed']
    p0 = df['Control']
    pooled_sd = ((p1*(1-p1) + p0*(1-p0)) / 2) ** 0.5
    df['smd'] = (p1 - p0) / pooled_sd
    return df

def make_love_plot(stats: Dict[str, pd.DataFrame], weighted_stats: Dict[str, pd.DataFrame], save_path: str, filename: str):
    """Make a love plot of the statistics."""
    binary_stats = stats[RAW][BINARY]
    weighted_binary_stats = weighted_stats[RAW][BINARY]

    # Pivot and convert percentages to proportions
    pivot = binary_stats.pivot(index='criterion', columns='group', values='percentage')
    pivot[['Exposed', 'Control']] = pivot[['Exposed', 'Control']] / 100

    pivot_weighted = weighted_binary_stats.pivot(index='criterion', columns='group', values='percentage')
    pivot_weighted[['Exposed', 'Control']] = pivot_weighted[['Exposed', 'Control']] / 100

    # Compute SMD and sort
    pivot_smd = compute_smd(pivot).sort_values('smd')
    pivot_weighted_smd = compute_smd(pivot_weighted).sort_values('smd')

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(pivot_smd['smd'], pivot_smd.index, label='Unweighted', alpha=0.7)
    ax.scatter(pivot_weighted_smd['smd'], pivot_weighted_smd.index, label='Weighted', alpha=0.7)
    ax.axvline(0, linestyle='--', linewidth=1)
    ax.set_xlabel("Standardized Mean Difference")
    ax.set_ylabel("Confounder")
    ax.set_title("Love Plot (SMD): Exposure vs Control")
    ax.legend()
    plt.subplots_adjust(left=0.25)  # Increase left margin for y-axis labels
    save_figure_with_azure_copy(fig, join(save_path, filename), dpi=200)

def load_data(
    criteria_path: str,
    cohort_path: str,
    ps_calibrated_predictions_path: str,
    outcome_model_path: str,
) -> pd.DataFrame:
    """
    Load and merge cohort data, patient IDs, propensity scores, and outcome predictions as needed.

    This function:
        - Loads the main criteria DataFrame (patient-level flags and variables).
        - Optionally filters patients by a provided cohort (patient IDs).
        - Optionally merges in propensity scores and exposures (inner join).
        - Optionally merges in outcome predictions and targets (inner join).
        - Ensures no missing values are introduced for key columns.

    Args:
        criteria_path (str): Path to the directory containing the criteria CSV file.
        cohort_path (str): Path to the directory containing patient IDs to filter on (optional).
        ps_calibrated_predictions_path (str): Path to the directory with propensity score predictions (optional).
        outcome_model_path (str): Path to the directory with outcome predictions and targets (optional).

    Returns:
        pd.DataFrame: The merged and filtered cohort DataFrame, ready for analysis.
    """

    logger.info("Loading criteria DataFrame")
    criteria = pd.read_csv(join(criteria_path, CRITERIA_FLAGS_FILE))
    logger.info(f"Loaded {len(criteria)} criteria")

    if cohort_path:
        logger.info("Cohort path provided - filtering criteria")
        pids = torch.load(join(cohort_path, PID_FILE))
        logger.info(f"Loaded {len(pids)} patient IDs")
        criteria = criteria[criteria[PID_COL].isin(pids)]
        logger.info(f"Filtered criteria to {len(criteria)} patients")

    # Optionally merge propensity scores and exposures
    if ps_calibrated_predictions_path:
        logger.info("Propensity scores and exposures path provided - merging")
        ps_path = join(ps_calibrated_predictions_path, CALIBRATED_PREDICTIONS_FILE)
        ps_df = pd.read_csv(ps_path).rename(
            columns={TARGETS: EXPOSURE_COL, PROBAS: PS_COL}
        )
        ps_df = _convert_to_int(ps_df, EXPOSURE_COL)
        check_binary(ps_df, EXPOSURE_COL)
        logger.info(
            f"{len(criteria)} Patients in Criteria before merging with propensity scores and exposures"
        )
        criteria = _inner_merge_data_on_pid(criteria, ps_df)
        logger.info(
            f"{len(criteria)} Patients in Criteria after merging with propensity scores and exposures"
        )

    # Optionally merge predictions and targets
    if outcome_model_path:
        logger.info("Predictions and targets path provided - merging")
        outcome_path = join(outcome_model_path, CALIBRATED_PREDICTIONS_FILE)
        outcome_df = pd.read_csv(outcome_path)[[PID_COL, TARGETS]]
        outcome_df = _convert_to_int(outcome_df, TARGETS)
        check_binary(outcome_df, TARGETS)
        logger.info(
            f"{len(criteria)} Patients in Criteria before merging with predictions and targets"
        )
        criteria = _inner_merge_data_on_pid(criteria, outcome_df)
        logger.info(
            f"{len(criteria)} Patients in Criteria after merging with predictions and targets"
        )

    return criteria


def _inner_merge_data_on_pid(criteria: pd.DataFrame, data: pd.DataFrame):
    criteria = pd.merge(criteria, data, on=PID_COL, how="inner")
    logger.info(
        f"{len(criteria)} Patients in Criteria after merging with {data.columns[0]}"
    )
    return criteria


def check_binary(df: pd.DataFrame, col: str):
    if not df[col].isin([0, 1]).all():
        raise ValueError(
            f"Column {col} is not binary. Example values: {df[col].value_counts()}"
        )


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


def check_ps_group_variance(ps: pd.Series, exposure: pd.Series):
    """
    Check if the propensity score has zero variance in either group.
    Issues a warning if so, and returns True if an edge case is detected.
    """
    ps_control = ps[exposure == 0]
    ps_treated = ps[exposure == 1]
    std_control = np.std(ps_control)
    std_treated = np.std(ps_treated)
    edge_case = False

    if std_control == 0 or std_treated == 0:
        msg = (
            "Warning: Propensity score has zero standard deviation in "
            f"{'control' if std_control == 0 else ''}"
            f"{' and ' if std_control == 0 and std_treated == 0 else ''}"
            f"{'treated' if std_treated == 0 else ''} group(s). "
            "Some diagnostics will be undefined."
        )
        logger.warning(msg)
        edge_case = True
    return edge_case
