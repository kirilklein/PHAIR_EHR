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

import pandas as pd
import torch

from corebehrt.constants.causal.data import (EXPOSURE_COL, PROBAS, PS_COL,
                                             TARGETS)
from corebehrt.constants.causal.paths import (CALIBRATED_PREDICTIONS_FILE,
                                              CRITERIA_FLAGS_FILE, STATS_FILE,
                                              STATS_RAW_FILE)
from corebehrt.constants.data import PID_COL
from corebehrt.constants.paths import PID_FILE
from corebehrt.functional.setup.args import get_args
from corebehrt.main_causal.helper_scripts.helper.get_stat import (
    StatConfig, analyze_cohort)
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
    custom_config = StatConfig(decimal_places=3, percentage_decimal_places=2)

    stats = analyze_cohort(criteria, config=custom_config, return_raw=True)
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
    df[col] = df[col].astype(int)
    return df


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main(args.config_path)
