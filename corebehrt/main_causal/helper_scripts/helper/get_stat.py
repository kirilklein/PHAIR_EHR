import logging
from os.path import join
from typing import Dict
import pandas as pd
import torch

from corebehrt.constants.causal.data import EXPOSURE_COL, PROBAS, PS_COL, TARGETS
from corebehrt.constants.causal.paths import (
    CALIBRATED_PREDICTIONS_FILE,
    CRITERIA_FLAGS_FILE,
    STATS_FILE_BINARY,
    STATS_FILE_NUMERIC,
    STATS_RAW_FILE_BINARY,
    STATS_RAW_FILE_NUMERIC,
)
from corebehrt.constants.causal.stats import BINARY, FORMATTED, NUMERIC, RAW
from corebehrt.constants.data import PID_COL
from corebehrt.constants.paths import PID_FILE
from corebehrt.functional.cohort_handling.stats import (
    StatConfig,
    format_stats_table,
    get_stratified_stats,
)


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


def print_stats(stats: Dict[str, pd.DataFrame]):
    """Print formatted statistics tables."""
    print("================================================")
    print("Formatted stats:")
    print(stats[FORMATTED][BINARY].head(30))
    print(stats[FORMATTED][NUMERIC].head(30))


def save_stats(stats: Dict[str, pd.DataFrame], save_path: str, weighted: bool = False):
    """Save statistics tables to csv files."""
    if weighted:
        suffix = "_weighted"
    else:
        suffix = ""
    stats[FORMATTED][BINARY].to_csv(
        join(save_path, STATS_FILE_BINARY + suffix), index=False
    )
    stats[RAW][BINARY].to_csv(
        join(save_path, STATS_RAW_FILE_BINARY + suffix), index=False
    )
    stats[FORMATTED][NUMERIC].to_csv(
        join(save_path, STATS_FILE_NUMERIC + suffix), index=False
    )
    stats[RAW][NUMERIC].to_csv(
        join(save_path, STATS_RAW_FILE_NUMERIC + suffix), index=False
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


def _convert_to_int(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Convert a column to integer type."""
    df[col] = df[col].astype(int)
    return df
