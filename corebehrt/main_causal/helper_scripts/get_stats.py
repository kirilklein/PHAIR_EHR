"""
Cohort Statistics Calculator for Causal Inference Analysis.

Extracts descriptive statistics from patient cohorts, supporting stratification (e.g., by exposure), weighted analysis, and optional integration of predictions or propensity scores.

Inputs:
    - Required: Criteria DataFrame (e.g., from extract_criteria.py) a csv file with the criteria flags + age.
    - Optional:
        - PS: Predictions or calibrated predictions (for exposed/unexposed stats + weighted stats + common support filtering)
        - PT: Predictions and targets (for additional stats on outcome)
        - Cohort (to filter on pids)

Outputs:
    - Summary statistics by group and overall, for reporting and cohort characterization.
"""

import logging
from os.path import join
from typing import Dict

import pandas as pd
import torch

from corebehrt.constants.causal.data import EXPOSURE_COL, PROBAS, PS_COL, TARGETS
from corebehrt.constants.causal.paths import (
    CALIBRATED_PREDICTIONS_FILE,
    CRITERIA_FLAGS_FILE,
    STATS_FILE,
    STATS_RAW_FILE,
)
from corebehrt.constants.data import PID_COL
from corebehrt.constants.paths import PID_FILE
from corebehrt.functional.setup.args import get_args
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.directory_causal import CausalDirectoryPreparer

CONFIG_PATH = "./corebehrt/configs/causal/helper/get_stats.yaml"


def main(config_path: str):
    """Execute cohort selection and save results."""
    cfg = load_config(config_path)
    CausalDirectoryPreparer(cfg).setup_get_stats()

    logger = logging.getLogger("get_stats")

    logger.info("Starting get stats")
    path_cfg = cfg.paths
    criteria_path = path_cfg.criteria

    # optional
    cohort_path = path_cfg.get("cohort", None)
    ps_calibrated_predictions_path = path_cfg.get("ps_calibrated_predictions", None)
    outcome_model_path = path_cfg.get("outcome_model", None)

    # output
    save_path = path_cfg.stats

    criteria = load_data(
        criteria_path,
        cohort_path,
        ps_calibrated_predictions_path,
        outcome_model_path,
        logger,
    )

    print("================================================")
    print("Criteria:")
    print(criteria.head())
    print("================================================")

    stats = analyze_cohort(criteria, detailed=True)
    print("================================================")
    print("Raw stats:")
    print(stats["raw"].head(20))
    print("================================================")
    print("Formatted stats:")
    print(stats["formatted"].head(20))
    stats["formatted"].to_csv(join(save_path, STATS_FILE), index=False)
    stats["raw"].to_csv(join(save_path, STATS_RAW_FILE), index=False)


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
        ps_df[EXPOSURE_COL] = ps_df[EXPOSURE_COL].astype(int)
        criteria = pd.merge(criteria, ps_df, on=PID_COL, how="left")
        logger.info("Merged with propensity scores and exposures")

    # Optionally merge predictions and targets
    if outcome_model_path:
        outcome_path = join(outcome_model_path, CALIBRATED_PREDICTIONS_FILE)
        outcome_df = pd.read_csv(outcome_path)[[PID_COL, TARGETS]]
        outcome_df[TARGETS] = outcome_df[TARGETS].astype(int)
        criteria = pd.merge(criteria, outcome_df, on=PID_COL, how="left")
        logger.info("Merged with predictions and targets")

    return criteria


def get_stats(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """
    For each criterion column (excluding subject_id, exposure, ps, targets), compute:
      - count and fraction (if binary)
      - mean and std (if numeric)

    Args:
        df: DataFrame containing the criteria columns
        prefix: Optional prefix to add to the criterion name (used for exposed/unexposed stratification)

    Returns:
        A DataFrame with one row per criterion and columns: criterion, count, fraction, mean, std
    """
    # Columns to exclude from stats
    special_cols = {PID_COL, EXPOSURE_COL, PS_COL}

    # Only consider columns that are not special
    stat_cols = [col for col in df.columns if col not in special_cols]

    stats = []
    n = len(df)

    for col in stat_cols:
        s = df[col]
        # Only consider non-null values for stats
        non_null = s.dropna()

        # Create the criterion name with prefix if provided
        criterion_name = f"{prefix}{col}" if prefix else col
        entry = {"criterion": criterion_name}

        if pd.api.types.is_bool_dtype(s) or set(non_null.unique()) <= {
            0,
            1,
            True,
            False,
        }:
            # Binary: count and fraction of True/1
            count = non_null.sum()
            entry["count"] = int(count)
            entry["fraction"] = float(count) / n if n > 0 else float("nan")
            entry["mean"] = float("nan")
            entry["std"] = float("nan")
        elif pd.api.types.is_numeric_dtype(s):
            # Numeric: mean and std
            entry["count"] = non_null.count()
            entry["fraction"] = float("nan")
            entry["mean"] = non_null.mean()
            entry["std"] = non_null.std()
        else:
            # Skip non-binary, non-numeric columns
            continue

        stats.append(entry)

    return pd.DataFrame(stats)


def get_stratified_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes statistics for the overall cohort and, if the exposure column exists,
    separately for exposed and unexposed groups.

    Args:
        df: DataFrame containing criteria columns and potentially an exposure column

    Returns:
        A DataFrame with statistics for the overall cohort and stratified by exposure status
        if the exposure column exists.
    """
    # Get overall stats
    overall_stats = get_stats(df)

    # Check if exposure column exists in the DataFrame
    if EXPOSURE_COL in df.columns:
        # Get stats for exposed group
        exposed_df = df[df[EXPOSURE_COL] == 1]
        if len(exposed_df) > 0:
            exposed_stats = get_stats(exposed_df, prefix="exposed_")
        else:
            exposed_stats = pd.DataFrame(columns=overall_stats.columns)

        # Get stats for unexposed group
        unexposed_df = df[df[EXPOSURE_COL] == 0]
        if len(unexposed_df) > 0:
            unexposed_stats = get_stats(unexposed_df, prefix="unexposed_")
        else:
            unexposed_stats = pd.DataFrame(columns=overall_stats.columns)

        # Combine all stats
        combined_stats = pd.concat(
            [overall_stats, exposed_stats, unexposed_stats], ignore_index=True
        )
    else:
        # If no exposure column, just return overall stats
        combined_stats = overall_stats

    return combined_stats


def format_stats_table(stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Formats the statistics table for better readability:
    - Rounds numeric values
    - Formats fractions as percentages
    - Adds group information

    Args:
        stats_df: DataFrame with statistics as returned by get_stratified_stats

    Returns:
        A formatted DataFrame ready for presentation
    """
    # Make a copy to avoid modifying the original
    formatted_df = stats_df.copy()

    # Add group column based on criterion prefix
    formatted_df["group"] = "Overall"
    formatted_df.loc[formatted_df["criterion"].str.startswith("exposed_"), "group"] = (
        "Exposed"
    )
    formatted_df.loc[
        formatted_df["criterion"].str.startswith("unexposed_"), "group"
    ] = "Unexposed"

    # Remove prefix from criterion names for exposed/unexposed groups
    formatted_df["criterion"] = formatted_df["criterion"].str.replace("exposed_", "")
    formatted_df["criterion"] = formatted_df["criterion"].str.replace("unexposed_", "")

    # Format numeric columns
    for col in ["mean", "std"]:
        formatted_df[col] = formatted_df[col].apply(
            lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A"
        )

    # Format fraction as percentage
    formatted_df["percentage"] = formatted_df["fraction"].apply(
        lambda x: f"{x * 100:.1f}%" if not pd.isna(x) else "N/A"
    )

    # Reorder and select columns
    result = formatted_df[["group", "criterion", "count", "percentage", "mean", "std"]]

    return result


def analyze_cohort(df: pd.DataFrame, detailed: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Comprehensive cohort analysis function that computes and formats statistics.

    Args:
        df: DataFrame containing the criteria columns and potentially exposure column
        detailed: If True, returns both raw and formatted statistics

    Returns:
        A dictionary with formatted (and raw if detailed=True) statistics DataFrames
    """
    raw_stats = get_stratified_stats(df)
    formatted_stats = format_stats_table(raw_stats)

    result = {"formatted": formatted_stats}
    if detailed:
        result["raw"] = raw_stats

    return result


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main(args.config_path)
