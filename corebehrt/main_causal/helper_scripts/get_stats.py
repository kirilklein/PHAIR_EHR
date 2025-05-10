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
from corebehrt.constants.causal.paths import EFFECTIVE_SAMPLE_SIZE_FILE
from corebehrt.constants.causal.stats import WEIGHTS_COL
from corebehrt.functional.cohort_handling.stats import compute_weights
from corebehrt.functional.setup.args import get_args
from corebehrt.main_causal.helper_scripts.helper.get_stat import (
    analyze_cohort,
    analyze_cohort_with_weights,
    get_effective_sample_size_df,
    load_data,
    print_stats,
    save_stats,
    ps_plot,
    check_ps_columns,
)
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

    if cfg.get("common_support_threshold", None) is not None:
        from CausalEstimate.filter.propensity import filter_common_support

        criteria = filter_common_support(
            criteria,
            ps_col=PS_COL,
            treatment_col=EXPOSURE_COL,
            threshold=cfg.common_support_threshold,
        )

    stats = analyze_cohort(criteria)
    print_stats(stats)
    save_stats(stats, save_path)

    if cfg.get("weights", None) is not None:
        check_ps_columns(criteria)
        criteria[WEIGHTS_COL] = compute_weights(criteria, cfg.weights)
        stats = analyze_cohort_with_weights(criteria, WEIGHTS_COL)
        print("--------------------------------")
        print(f"Weighted stats ({cfg.weights}):")
        print_stats(stats)
        save_stats(stats, save_path, weighted=True)
        ess_df = get_effective_sample_size_df(criteria, WEIGHTS_COL)
        print("True sample size: ", len(criteria))
        print(ess_df)

        ess_df.to_csv(join(save_path, EFFECTIVE_SAMPLE_SIZE_FILE), index=False)

    if cfg.get("plot_ps", False):
        check_ps_columns(criteria)
        ps_plot(criteria, save_path)

    logger.info("Done")


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main(args.config_path)
