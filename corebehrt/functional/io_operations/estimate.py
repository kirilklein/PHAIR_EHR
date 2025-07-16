"""
I/O operations for saving causal estimation results and experiment data.

This module provides functions to save experiment data, statistics, and effect estimates
to various file formats (parquet, csv) in the experiment directory structure.
"""

from os.path import join
import pandas as pd

from corebehrt.constants.causal.paths import (
    EXPERIMENT_DATA_FILE,
    EXPERIMENT_STATS_FILE,
    ESTIMATE_RESULTS_FILE,
)


def save_all_results(
    exp_dir: str, df: pd.DataFrame, effect_df: pd.DataFrame, stats_df: pd.DataFrame
) -> None:
    """
    Save all experiment results including data, statistics, and effect estimates.

    Args:
        exp_dir: Experiment directory path
        df: Experiment data to save
        effect_df: Effect estimates dataframe
        stats_df: Combined statistics dataframe
    """
    save_experiment_data(df, exp_dir)
    save_experiment_stats_combined(stats_df, exp_dir)
    save_estimate_results(effect_df, exp_dir)


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
    effect_df = effect_df.round(5)
    effect_df.to_csv(filepath, index=False)
