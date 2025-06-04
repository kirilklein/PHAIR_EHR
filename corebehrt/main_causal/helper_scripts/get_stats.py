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

from corebehrt.constants.causal.data import EXPOSURE_COL, PS_COL
from corebehrt.constants.causal.paths import (
    EFFECTIVE_SAMPLE_SIZE_FILE,
    PS_PLOT_FILE,
    PS_PLOT_FILE_FILTERED,
    PS_SUMMARY_FILE,
    PS_SUMMARY_FILE_FILTERED,
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
    save_stats,
)
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.causal.directory import CausalDirectoryPreparer

CONFIG_PATH = "./corebehrt/configs/causal/helper/get_stats.yaml"
EPS = 1e-6


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
    )
    if (PS_COL in criteria.columns) and cfg.get("clip_ps", True):
        outside_count = (criteria[PS_COL] < EPS).sum() + (
            criteria[PS_COL] > 1 - EPS
        ).sum()
        logger.info(f"{outside_count} ps outside clipping range - clipping")
        logger.info("Clipping PS")
        criteria[PS_COL] = criteria[PS_COL].clip(lower=EPS, upper=1 - EPS)

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

    # Track if filtering was done
    filtered = False
    if cfg.get("common_support_threshold", None) is not None:
        from CausalEstimate.filter.propensity import filter_common_support

        criteria = filter_common_support(
            criteria,
            ps_col=PS_COL,
            treatment_col=EXPOSURE_COL,
            threshold=cfg.common_support_threshold,
        )
        filtered = True

    stats = analyze_cohort(criteria)
    log_stats(stats)
    save_stats(stats, save_path)

    if filtered:
        if PS_COL in criteria.columns:
            ps_summary = positivity_summary(criteria[PS_COL], criteria[EXPOSURE_COL])
            logger.info("--------------------------------")
            logger.info("Positivity summary after filtering:")
            log_table(ps_summary, logger)
            ps_summary.to_csv(join(save_path, PS_SUMMARY_FILE_FILTERED), index=False)

    if cfg.get("weights", None) is not None:
        check_ps_columns(criteria)
        criteria[WEIGHTS_COL] = compute_weights(criteria, cfg.weights)
        stats = analyze_cohort_with_weights(criteria, WEIGHTS_COL)
        logger.info("--------------------------------")
        logger.info(f"Weighted stats ({cfg.weights}):")
        log_stats(stats)
        save_stats(stats, save_path, weighted=True)
        ess_df = get_effective_sample_size_df(criteria, WEIGHTS_COL)
        logger.info(f"True sample size: {len(criteria)}")
        log_table(ess_df, logger)
        ess_df.to_csv(join(save_path, EFFECTIVE_SAMPLE_SIZE_FILE), index=False)

    if cfg.get("plot_ps", False) and filtered:
        try:
            check_ps_columns(criteria)
            ps_plot(criteria, save_path, PS_PLOT_FILE_FILTERED)
        except Exception as e:
            logger.warning(f"Error plotting PS: {e}")

    logger.info("Done")


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main(args.config_path)
