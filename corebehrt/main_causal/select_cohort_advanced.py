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
- Statistics showing patient counts at each filtering stage
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
import os
from os.path import join

import pandas as pd
import torch

from corebehrt.constants.data import PID_COL, TIMESTAMP_COL
from corebehrt.constants.paths import (
    FOLDS_FILE,
    INDEX_DATES_FILE,
    PID_FILE,
    TEST_PIDS_FILE,
)
from corebehrt.functional.features.split import create_folds, split_test
from corebehrt.functional.io_operations.meds import iterate_splits_and_shards
from corebehrt.functional.setup.args import get_args
from corebehrt.modules.cohort_handling.advanced.apply import apply_criteria
from corebehrt.modules.cohort_handling.advanced.data.patient import (
    patients_to_dataframe,
)
from corebehrt.modules.cohort_handling.advanced.extract import extract_patient_criteria
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.directory_causal import CausalDirectoryPreparer

CONFIG_PATH = "./corebehrt/configs/causal/advanced_extraction.yaml"


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
    splits = path_cfg.get("splits", ["tuning"])

    index_dates = pd.read_csv(
        join(cohort_path, INDEX_DATES_FILE), parse_dates=[TIMESTAMP_COL]
    )

    patients = {}
    for shard_path in iterate_splits_and_shards(meds_path, splits):
        print(f"Processing shard: {os.path.basename(shard_path)}")
        shard = pd.read_parquet(shard_path)
        patients.update(extract_patient_criteria(shard, index_dates, cfg))

    df = patients_to_dataframe(patients)
    df.to_csv(join(save_path, "criteria_flags.csv"))

    df, stats = apply_criteria(df, cfg)
    with open(join(save_path, "stats.json"), "w") as f:
        json.dump(stats, f)

    filtered_pids = df[PID_COL].tolist()

    logger.info("Saving cohort")
    torch.save(filtered_pids, join(save_path, PID_FILE))
    filtered_index_dates = index_dates[index_dates[PID_COL].isin(filtered_pids)]
    filtered_index_dates.to_csv(join(save_path, INDEX_DATES_FILE))

    train_val_pids, test_pids = split_test(filtered_pids, cfg.get("test_ratio", 0))
    if len(test_pids) > 0:
        torch.save(test_pids, join(save_path, TEST_PIDS_FILE))

    if len(train_val_pids) > 0:
        folds = create_folds(
            train_val_pids,
            cfg.get("cv_folds", 1),
            cfg.get("seed", 42),
            cfg.get("val_ratio", 0.1),
        )
    torch.save(folds, join(save_path, FOLDS_FILE))


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main(args.config_path)
