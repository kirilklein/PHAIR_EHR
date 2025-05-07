"""
Advanced cohort selection script that applies complex inclusion/exclusion criteria to an initial patient cohort.

This script performs a second stage of patient filtering after the basic cohort selection. It:
1. Takes an initial cohort directory containing patient IDs and index dates
2. Processes raw medical event data (MEDS) to evaluate complex criteria for each patient
3. Extracts relevant clinical values and flags based on configurable criteria definitions
4. Applies inclusion/exclusion rules using the extracted values and flags
5. Tracks patient flow through each filtering step for CONSORT diagram creation

The script outputs:
- A DataFrame with criteria flags and clinical values for each patient
- Statistics showing patient counts at each filtering stage:
  * Initial total patients
  * Patients excluded by individual inclusion criteria
  * Patients excluded by individual exclusion criteria
  * Patients excluded by unique code limits
  * Final included patients count
- Final cohort files matching the format of the initial selection
  (patient IDs, index dates, train/test splits)

The criteria evaluation is highly configurable through YAML files that can specify:
- Code patterns to match clinical events
- Time windows relative to index dates
- Numeric thresholds and comparisons
- Required combinations of multiple criteria

"""

import json
import logging
from os.path import join

import numpy as np
import pandas as pd
import torch

from corebehrt.constants.cohort import (
    CRITERIA_DEFINITIONS,
    EXCLUSION,
    INCLUSION,
    UNIQUE_CODE_LIMITS,
)
from corebehrt.constants.causal.paths import CRITERIA_FLAGS_FILE, CRITERIA_CONFIG_FILE
from corebehrt.constants.data import TIMESTAMP_COL
from corebehrt.constants.paths import INDEX_DATES_FILE, PID_FILE
from corebehrt.functional.setup.args import get_args
from corebehrt.main_causal.helper.select_cohort_advanced import (
    extract_criteria_from_shards,
    filter_and_save_cohort,
    split_and_save,
    check_inclusion_exclusion,
)
from corebehrt.modules.cohort_handling.advanced.validator import CriteriaValidator
from corebehrt.functional.preparation.filter import filter_table_by_pids
from corebehrt.modules.cohort_handling.advanced.apply import apply_criteria_with_stats
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.directory_causal import CausalDirectoryPreparer

CONFIG_PATH = "./corebehrt/configs/causal/select_cohort/advanced_extraction.yaml"


def main(config_path: str):
    """Execute cohort selection and save results."""
    cfg = load_config(config_path)
    CausalDirectoryPreparer(cfg).setup_select_cohort_advanced()

    logger = logging.getLogger("select_cohort_advanced")

    logger.info("Starting advanced cohort selection")
    path_cfg = cfg.paths
    cohort_path = path_cfg.cohort
    meds_path = path_cfg.meds
    save_path = path_cfg.cohort_advanced
    criteria_config_path = path_cfg.criteria_config
    splits = path_cfg.get("splits", ["tuning"])

    logger.info("Loading index dates")
    index_dates = pd.read_csv(
        join(cohort_path, INDEX_DATES_FILE), parse_dates=[TIMESTAMP_COL]
    )
    logger.info("Loading patient IDs")
    pids = torch.load(join(cohort_path, PID_FILE))
    logger.info(f"Loaded {len(pids)} patient IDs")

    logger.info("Loading criteria config")
    criteria_config = load_config(criteria_config_path)
    # Write criteria config to output directory
    criteria_config.save_to_yaml(join(save_path, CRITERIA_CONFIG_FILE))

    validator = CriteriaValidator(criteria_config.get(CRITERIA_DEFINITIONS))
    validator.validate()
    check_inclusion_exclusion(validator, criteria_config)
    logger.info("Checks successful, extracting criteria")
    index_dates = filter_table_by_pids(index_dates, pids)
    criteria_df = extract_criteria_from_shards(
        meds_path=meds_path,
        index_dates=index_dates,
        criteria_definitions_cfg=criteria_config.get(CRITERIA_DEFINITIONS),
        splits=splits,
        pids=pids,
    )
    criteria_df.to_csv(
        join(save_path, CRITERIA_FLAGS_FILE), index=False
    )  # TODO: change to parquet

    logger.info("Applying criteria and saving stats")
    df, stats = apply_criteria_with_stats(
        criteria_df,
        criteria_config[INCLUSION],
        criteria_config[EXCLUSION],
        criteria_config.get(UNIQUE_CODE_LIMITS, {}),
    )

    stats = {
        k: int(v) if isinstance(v, (np.int32, np.int64)) else v
        for k, v in stats.items()
    }
    logger.info("Saving stats")
    with open(join(save_path, "stats.json"), "w") as f:
        json.dump(stats, f)

    logger.info("Saving cohort")
    filtered_pids = filter_and_save_cohort(df, index_dates, save_path)
    split_and_save(
        filtered_pids,
        save_path,
        cfg.get("test_ratio", 0),
        cfg.get("cv_folds", 1),
        cfg.get("val_ratio", 0.1),
        cfg.get("seed", 42),
    )


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main(args.config_path)
