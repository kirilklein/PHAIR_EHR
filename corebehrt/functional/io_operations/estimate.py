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

INITIAL_ESTIMATES_COLUMNS = [
    "initial_effect_1",
    "initial_effect_0",
    "adjustment_1",
    "adjustment_0",
]


def save_all_results(
    exp_dir: str, df: pd.DataFrame, results_df: pd.DataFrame, stats_df: pd.DataFrame
):
    """Save all results to CSV files."""
    results_df.to_csv(join(exp_dir, "results.csv"), index=False)
    stats_df.to_csv(join(exp_dir, "stats.csv"), index=False)
    df.to_csv(join(exp_dir, "cohort.csv"), index=False)


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


def save_tmle_analysis(initial_estimates_df: pd.DataFrame, exp_dir: str) -> None:
    """Handle and save TMLE analysis."""
    if not all(
        col in initial_estimates_df.columns
        for col in INITIAL_ESTIMATES_COLUMNS + ["method", "outcome", "effect"]
    ):
        return
    initial_estimates_df = initial_estimates_df[
        INITIAL_ESTIMATES_COLUMNS + ["method", "effect", "outcome"]
    ]
    initial_estimates_df = initial_estimates_df[
        initial_estimates_df["method"] == "TMLE"
    ]
    initial_estimates_df["initial_effect"] = (
        initial_estimates_df["initial_effect_1"]
        - initial_estimates_df["initial_effect_0"]
    )
    initial_estimates_df["adjustment"] = (
        initial_estimates_df["adjustment_1"] - initial_estimates_df["adjustment_0"]
    )
    initial_estimates_df.to_csv(join(exp_dir, "tmle_analysis.csv"), index=False)
