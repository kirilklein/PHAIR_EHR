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

    stats = analyze_cohort(criteria)
    print_stats(stats)
    save_stats(stats, save_path)

    if cfg.get("weights", None) is not None:
        if PS_COL not in criteria.columns:
            raise ValueError(f"PS_COL {PS_COL} not found in criteria")
        if EXPOSURE_COL not in criteria.columns:
            raise ValueError(f"EXPOSURE_COL {EXPOSURE_COL} not found in criteria")
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


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main(args.config_path)
