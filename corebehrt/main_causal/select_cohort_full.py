import logging
from os.path import join

import torch

from corebehrt.constants.paths import PID_FILE
from corebehrt.functional.setup.args import get_args
from corebehrt.main_causal.helper.select_cohort_full import (
    select_cohort,
    split_and_save,
)
from corebehrt.modules.setup.causal.directory import CausalDirectoryPreparer
from corebehrt.modules.setup.config import load_config

CONFIG_PATH = "./corebehrt/configs/causal/select_cohort_full/extract.yaml"


def main(config_path: str):
    """Execute cohort selection and save results."""
    cfg = load_config(config_path)
    CausalDirectoryPreparer(cfg).setup_select_cohort_full()

    logger = logging.getLogger("select_cohort_full")

    path_cfg = cfg.paths

    logger.info("Starting cohort selection")
    filtered_pids = select_cohort(
        path_cfg.features,
        path_cfg.meds,
        path_cfg.get("splits", ["tuning"]),
        path_cfg.exposures,
        path_cfg.exposure,
        path_cfg.cohort,
        cfg.time_windows,
        criteria_definitions_path=path_cfg.criteria_config,
        index_date_matching_cfg=cfg.get("index_date_matching"),
        logger=logger,
    )
    torch.save(filtered_pids, join(path_cfg.cohort, PID_FILE))
    logger.info("Saving cohort")
    split_and_save(
        filtered_pids,
        path_cfg.cohort,
        cfg.get("test_ratio", 0),
        cfg.get("cv_folds", 1),
        cfg.get("val_ratio", 0.1),
        cfg.get("seed", 42),
    )


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main(args.config_path)
