"""
Cohort Statistics Calculator for Causal Inference Analysis.

Computes descriptive statistics for patient cohorts, supporting stratification by exposure,
weighted analysis (ATE, ATT, ATC), and optional integration of propensity scores and outcome predictions.

Inputs:
    - Criteria DataFrame (required): Patient-level criteria flags and numeric variables.
    - Propensity Scores (optional): For weighting and stratification.
    - Outcome Predictions/Targets (optional): For outcome statistics.
    - Cohort (optional): Patient IDs to filter.
    - Weights (optional): Weighting scheme ("ATE", "ATT", "ATC").

Outputs:
    - Summary statistics (binary and numeric) for overall and by exposure group.
    - Weighted statistics and effective sample size (ESS) if weights are used.
    - All outputs can be printed or saved as CSV files.
"""

import logging
from os.path import join

import pandas as pd

from corebehrt.constants.causal.data import EXPOSURE_COL, PS_COL
from corebehrt.constants.data import PID_COL
from corebehrt.constants.causal.paths import (
    EFFECTIVE_SAMPLE_SIZE_FILE,
    PS_PLOT_FILE,
    PS_PLOT_FILE_FILTERED,
    PS_SUMMARY_FILE,
    PS_SUMMARY_FILE_FILTERED,
    LOVE_PLOT_FILE,
    LOVE_PLOT_FILE_SECONDARY,
)
from corebehrt.constants.causal.stats import WEIGHTS_COL
from corebehrt.functional.cohort_handling.stats import compute_weights
from corebehrt.functional.setup.args import get_args
from corebehrt.functional.utils.log import log_table
from corebehrt.main_causal.helper_scripts.helper.get_stat import (
    analyze_cohort,
    analyze_cohort_with_weights,
    check_ps_columns,
    get_effective_sample_size_df,
    load_data,
    log_stats,
    positivity_summary,
    ps_plot,
    make_love_plot,
    save_stats,
)
from corebehrt.modules.cohort_handling.advanced.apply import apply_criteria_with_stats
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.causal.directory import CausalDirectoryPreparer

CONFIG_PATH = "./corebehrt/configs/causal/helper/get_stats.yaml"
EPS = 1e-6


def _convert_boolean_to_expression(value, df: pd.DataFrame) -> str:
    """
    Convert boolean or string boolean to a safe expression that references a dummy column.

    For boolean True/False, creates temporary dummy columns in the dataframe that are
    always True/False. These should only be used for filtering and not appear in final output.

    Args:
        value: Boolean, string boolean ("true"/"false"), or expression string
        df: DataFrame to add dummy columns to if needed

    Returns:
        Expression string referencing a dummy column or the original expression
    """
    # Ensure dummy columns exist
    if "_always_true_dummy" not in df.columns:
        df["_always_true_dummy"] = True
    if "_always_false_dummy" not in df.columns:
        df["_always_false_dummy"] = False

    # Handle boolean values
    if isinstance(value, bool):
        return "_always_true_dummy" if value else "_always_false_dummy"

    # Handle string representations of booleans
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return (
            "_always_true_dummy" if value.lower() == "true" else "_always_false_dummy"
        )

    # Otherwise, assume it's a valid expression string
    return str(value)


def process_cohort(
    criteria: pd.DataFrame,
    cfg: dict,
    logger: logging.Logger,
    save_path: str,
    cohort_name: str = "",
    save_stats_files: bool = True,
) -> tuple:
    """
    Process a cohort: clip PS, filter, compute stats and weighted stats.

    Args:
        criteria: DataFrame with criteria data
        cfg: Configuration dictionary
        logger: Logger instance
        save_path: Path to save output files
        cohort_name: Name of the cohort for logging
        save_stats_files: Whether to save stats CSV files (default: True)

    Returns:
        tuple: (stats, weighted_stats, criteria_processed, filtered)
    """
    criteria_processed = criteria.copy()

    # Clip PS if needed
    if (PS_COL in criteria_processed.columns) and cfg.get("clip_ps", True):
        outside_count = (criteria_processed[PS_COL] < EPS).sum() + (
            criteria_processed[PS_COL] > 1 - EPS
        ).sum()
        if outside_count > 0:
            logger.info(f"{outside_count} ps outside clipping range - clipping")
            logger.info("Clipping PS")
            criteria_processed[PS_COL] = criteria_processed[PS_COL].clip(
                lower=EPS, upper=1 - EPS
            )

    # Track if filtering was done
    filtered = False
    if cfg.get("common_support_threshold", None) is not None:
        from CausalEstimate.filter.propensity import filter_common_support

        criteria_processed = filter_common_support(
            criteria_processed,
            ps_col=PS_COL,
            treatment_col=EXPOSURE_COL,
            threshold=cfg.common_support_threshold,
        )
        filtered = True

    # Compute stats
    stats = analyze_cohort(criteria_processed)
    if cohort_name:
        logger.info(f"--------------------------------")
        logger.info(f"Stats for {cohort_name}:")
    log_stats(stats)
    if save_stats_files:
        save_stats(stats, save_path, weighted=False)

    # Compute weighted stats if weights are provided
    weighted_stats = None
    if cfg.get("weights", None) is not None:
        check_ps_columns(criteria_processed)
        criteria_processed[WEIGHTS_COL] = compute_weights(
            criteria_processed, cfg.weights
        )
        weighted_stats = analyze_cohort_with_weights(criteria_processed, WEIGHTS_COL)
        if cohort_name:
            logger.info(f"--------------------------------")
            logger.info(f"Weighted stats ({cfg.weights}) for {cohort_name}:")
        else:
            logger.info("--------------------------------")
            logger.info(f"Weighted stats ({cfg.weights}):")
        log_stats(weighted_stats)
        if save_stats_files:
            save_stats(weighted_stats, save_path, weighted=True)
            ess_df = get_effective_sample_size_df(criteria_processed, WEIGHTS_COL)
            logger.info(f"True sample size: {len(criteria_processed)}")
            log_table(ess_df, logger)
            ess_df.to_csv(join(save_path, EFFECTIVE_SAMPLE_SIZE_FILE), index=False)

    return stats, weighted_stats, criteria_processed, filtered


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
    if ps_calibrated_predictions_path is not None:
        ps_calibrated_predictions_path = join(
            ps_calibrated_predictions_path, "predictions_exposure"
        )
    outcome_model_path = path_cfg.get("outcome_model", None)

    # Secondary cohort config (optional)
    # If provided, should point to a YAML file with inclusion/exclusion expressions
    # to filter the primary cohort
    secondary_cohort_config_path = path_cfg.get("secondary_cohort_config", None)

    # output
    save_path = path_cfg.stats

    # Process primary cohort
    criteria = load_data(
        criteria_path,
        cohort_path,
        ps_calibrated_predictions_path,
        outcome_model_path,
    )

    if PS_COL in criteria.columns:
        ps_summary = positivity_summary(criteria[PS_COL], criteria[EXPOSURE_COL])
        logger.info("--------------------------------")
        logger.info("Positivity summary before filtering:")
        log_table(ps_summary, logger)
        ps_summary.to_csv(join(save_path, PS_SUMMARY_FILE), index=False)

    if cfg.get("plot_ps", False):
        try:
            check_ps_columns(criteria)
            ps_plot(criteria, save_path, PS_PLOT_FILE)
        except Exception as e:
            logger.warning(f"Error plotting PS: {e}")

    # Process primary cohort
    stats, weighted_stats, criteria_processed, filtered = process_cohort(
        criteria, cfg, logger, save_path
    )

    if filtered:
        if PS_COL in criteria_processed.columns:
            ps_summary = positivity_summary(
                criteria_processed[PS_COL], criteria_processed[EXPOSURE_COL]
            )
            logger.info("--------------------------------")
            logger.info("Positivity summary after filtering:")
            log_table(ps_summary, logger)
            ps_summary.to_csv(join(save_path, PS_SUMMARY_FILE_FILTERED), index=False)

    if cfg.get("plot_ps", False) and filtered:
        try:
            check_ps_columns(criteria_processed)
            ps_plot(criteria_processed, save_path, PS_PLOT_FILE_FILTERED)
        except Exception as e:
            logger.warning(f"Error plotting PS: {e}")

    if cfg.get("make_love_plot", False) and cfg.get("weights", None) is not None:
        make_love_plot(stats, weighted_stats, save_path, LOVE_PLOT_FILE)

    # Process secondary cohort if provided
    if secondary_cohort_config_path is not None:
        logger.info("--------------------------------")
        logger.info("Processing secondary cohort")
        logger.info(
            f"Loading secondary cohort config from: {secondary_cohort_config_path}"
        )

        # Load the secondary cohort config with inclusion/exclusion expressions
        secondary_cfg = load_config(secondary_cohort_config_path)
        inclusion_raw = secondary_cfg.get("inclusion", True)  # Default to include all
        exclusion_raw = secondary_cfg.get("exclusion", False)  # Default to exclude none

        # Work on a copy to avoid modifying the original dataframe
        criteria_for_filtering = criteria_processed.copy()

        # Convert boolean values to expressions that reference temporary dummy columns
        inclusion_expression = _convert_boolean_to_expression(
            inclusion_raw, criteria_for_filtering
        )
        exclusion_expression = _convert_boolean_to_expression(
            exclusion_raw, criteria_for_filtering
        )

        logger.info(f"Inclusion expression: {inclusion_expression}")
        logger.info(f"Exclusion expression: {exclusion_expression}")
        logger.info(
            f"Starting with {len(criteria_processed)} patients from primary cohort"
        )

        # Filter the primary cohort using the inclusion/exclusion expressions
        included_pids, filter_stats = apply_criteria_with_stats(
            criteria_for_filtering,
            inclusion_expression,
            exclusion_expression,
            verbose=True,
        )

        logger.info(f"Filtered to {len(included_pids)} patients for secondary cohort")

        # Create secondary cohort dataframe from filtered patient IDs
        # Use the original criteria_processed (without dummy columns) to preserve all original criteria
        criteria_secondary = criteria_processed[
            criteria_processed[PID_COL].isin(included_pids)
        ].copy()

        logger.info(f"Secondary cohort contains {len(criteria_secondary)} patients")

        stats_secondary, weighted_stats_secondary, criteria_secondary_processed, _ = (
            process_cohort(
                criteria_secondary,
                cfg,
                logger,
                save_path,
                "secondary cohort",
                save_stats_files=False,
            )
        )

        if cfg.get("make_love_plot", False) and cfg.get("weights", None) is not None:
            make_love_plot(
                stats_secondary,
                weighted_stats_secondary,
                save_path,
                LOVE_PLOT_FILE_SECONDARY,
            )

    logger.info("Done")


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main(args.config_path)
