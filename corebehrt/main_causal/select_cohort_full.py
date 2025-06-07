import logging
from os.path import join

import torch

from corebehrt.constants.paths import (
    FOLDS_FILE,
    INDEX_DATES_FILE,
    PID_FILE,
    TEST_PIDS_FILE,
)
from corebehrt.functional.features.split import create_folds
from corebehrt.functional.setup.args import get_args
from corebehrt.main_causal.helper.select_cohort_full import select_cohort
from corebehrt.modules.setup.causal.directory import CausalDirectoryPreparer
from corebehrt.modules.setup.config import load_config

CONFIG_PATH = "./corebehrt/configs/causal/select_cohort_full/extract.yaml"


def main_select_cohort(config_path: str):
    """Execute cohort selection and save results."""
    cfg = load_config(config_path)
    CausalDirectoryPreparer(cfg).setup_select_cohort_full()

    logger = logging.getLogger("select_cohort_full")

    path_cfg = cfg.paths

    logger.info("Starting cohort selection")
    pids, index_dates, train_val_pids, test_pids = select_cohort(
        path_cfg.features,
        path_cfg.meds,
        path_cfg.get("splits", ["tuning"]),
        path_cfg.exposures,
        path_cfg.exposure,
        cfg.time_windows,
        test_ratio=cfg.test_ratio,
        criteria_definitions_path=path_cfg.criteria_config,
        logger=logger,
    )
    logger.info("Saving cohort")
    torch.save(pids, join(path_cfg.cohort, PID_FILE))
    index_dates.to_csv(join(path_cfg.cohort, INDEX_DATES_FILE))

    if len(test_pids) > 0:
        torch.save(test_pids, join(path_cfg.cohort, TEST_PIDS_FILE))

    if len(train_val_pids) > 0:
        folds = create_folds(
            train_val_pids,
            cfg.get("cv_folds", 1),
            cfg.get("seed", 42),
            cfg.get("val_ratio", 0.1),
        )
        torch.save(folds, join(path_cfg.cohort, FOLDS_FILE))


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_select_cohort(args.config_path)
