"""
I/O operations for saving causal estimation results and experiment data.

This module provides functions to save experiment data, statistics, and effect estimates
to various file formats (parquet, csv) in the experiment directory structure.
"""

from os.path import join

import pandas as pd

from corebehrt.constants.causal.data import EFFECT_ROUND_DIGIT
from corebehrt.constants.causal.paths import (
    ESTIMATE_RESULTS_FILE,
    EXPERIMENT_DATA_FILE,
    EXPERIMENT_STATS_FILE,
)


def save_all_results(
    exp_dir: str, df: pd.DataFrame, results_df: pd.DataFrame, stats_df: pd.DataFrame
):
    """Save all results to CSV files."""
    results_df.to_csv(join(exp_dir, "estimate_results.csv"), index=False)
    stats_df.to_csv(join(exp_dir, "estimate_stats.csv"), index=False)
    df.to_csv(join(exp_dir, "analysis_df.csv"), index=False)


def save_experiment_data(df: pd.DataFrame, exp_dir: str) -> None:
    """Save experiment data as parquet file."""
    filepath = join(exp_dir, EXPERIMENT_DATA_FILE)
    df.to_parquet(filepath, index=True)


def save_experiment_stats_combined(stats_df: pd.DataFrame, exp_dir: str) -> None:
    """Save combined statistics for all outcomes as CSV file."""
    filepath = join(exp_dir, EXPERIMENT_STATS_FILE)
    stats_df.to_csv(filepath, index=False)


def save_estimate_results(effect_df: pd.DataFrame, exp_dir: str) -> None:
    """Save effect estimates as CSV file with 5 decimal precision."""
    filepath = join(exp_dir, ESTIMATE_RESULTS_FILE)
    effect_df = effect_df.round(EFFECT_ROUND_DIGIT)
    effect_df.to_csv(filepath, index=False)


def save_tmle_analysis(tmle_analysis_df: pd.DataFrame | None, exp_dir: str) -> None:
    """Saves the pre-computed TMLE analysis dataframe to a CSV file."""
    if tmle_analysis_df is None or tmle_analysis_df.empty:
        return
    tmle_analysis_df.to_csv(join(exp_dir, "tmle_analysis.csv"), index=False)
