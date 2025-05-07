"""
Get any statistic about a cohort, e.g. comorbidities (arbitrary definition) + gender/age distribution.
"""

import logging
from os.path import join

import pandas as pd
import torch

from corebehrt.constants.data import TIMESTAMP_COL
from corebehrt.constants.paths import INDEX_DATES_FILE, PID_FILE
from corebehrt.functional.setup.args import get_args
from corebehrt.main_causal.helper.select_cohort_advanced import (
    extract_criteria,
    check_criteria_cfg,
)
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.directory_causal import CausalDirectoryPreparer

CONFIG_PATH = "./corebehrt/configs/causal/get_stats.yaml"


def main(config_path: str):
    """Execute cohort selection and save results."""
    cfg = load_config(config_path)
    CausalDirectoryPreparer(cfg).setup_get_stats()

    logger = logging.getLogger("get_stats")

    logger.info("Starting get stats")
    path_cfg = cfg.paths
    cohort_path = path_cfg.cohort
    meds_path = path_cfg.meds
    save_path = path_cfg.cohort_stats
    criteria_config_path = path_cfg.criteria_config

    logger.info("Loading index dates")
    index_dates = pd.read_csv(
        join(cohort_path, INDEX_DATES_FILE), parse_dates=[TIMESTAMP_COL]
    )
    logger.info("Loading patient IDs")
    pids = torch.load(join(cohort_path, PID_FILE))
    logger.info(f"Loaded {len(pids)} patient IDs")

    logger.info("Loading criteria config")
    criteria_config = load_config(criteria_config_path)
    criteria_config.save_to_yaml(join(save_path, "criteria_config.yaml"))
    check_criteria_cfg(criteria_config)
    logger.info("Extracting criteria")
    extract_criteria(meds_path, index_dates, criteria_config, save_path, pids)


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main(args.config_path)
