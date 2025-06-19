"""
Get any statistic about a cohort, e.g. comorbidities (arbitrary definition) + gender/age distribution.
"""

import logging
from os.path import join

import pandas as pd
import torch

from corebehrt.constants.causal.paths import (
    CRITERIA_DEFINITIONS_CFG,
    CRITERIA_FLAGS_FILE,
)
from corebehrt.constants.cohort import CRITERIA_DEFINITIONS
from corebehrt.constants.data import TIMESTAMP_COL
from corebehrt.constants.paths import INDEX_DATES_FILE, PID_FILE
from corebehrt.functional.preparation.filter import filter_table_by_pids
from corebehrt.functional.setup.args import get_args
from corebehrt.main_causal.helper.select_cohort_full import (
    extract_criteria_from_shards,
)
from corebehrt.modules.cohort_handling.advanced.validator import CriteriaValidator
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.causal.directory import CausalDirectoryPreparer

CONFIG_PATH = "./corebehrt/configs/causal/helper/extract_criteria.yaml"


def main(config_path: str):
    """Execute cohort selection and save results."""
    cfg = load_config(config_path)
    CausalDirectoryPreparer(cfg).setup_extract_criteria()

    logger = logging.getLogger("extract_criteria")

    logger.info("Starting extract criteria")
    path_cfg = cfg.paths
    cohort_path = path_cfg.cohort
    meds_path = path_cfg.meds
    save_path = path_cfg.criteria
    criteria_definitions_config_path = path_cfg.criteria_definitions_config
    splits = path_cfg.get("splits", ["tuning"])

    logger.info("Loading index dates")
    index_dates = pd.read_csv(
        join(cohort_path, INDEX_DATES_FILE), parse_dates=[TIMESTAMP_COL]
    )
    logger.info("Loading patient IDs")
    pids = torch.load(join(cohort_path, PID_FILE))
    logger.info(f"Loaded {len(pids)} patient IDs")

    logger.info("Loading criteria config")
    criteria_definitions_config = load_config(criteria_definitions_config_path)
    criteria_definitions_config.save_to_yaml(join(save_path, CRITERIA_DEFINITIONS_CFG))
    CriteriaValidator(criteria_definitions_config.get(CRITERIA_DEFINITIONS)).validate()
    logger.info("Extracting criteria")
    index_dates = filter_table_by_pids(index_dates, pids)
    criteria_df = extract_criteria_from_shards(
        meds_path=meds_path,
        index_dates=index_dates,
        criteria_definitions_cfg=criteria_definitions_config.get(CRITERIA_DEFINITIONS),
        splits=splits,
        pids=pids,
    )
    criteria_df.to_csv(join(save_path, CRITERIA_FLAGS_FILE), index=False)


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main(args.config_path)
