"""
Prepare data for finetune with exposure and outcome.
"""

import logging
import torch
from os.path import join
import os

from corebehrt.functional.setup.args import get_args
from corebehrt.modules.preparation.prepare_data_causal import CausalDatasetPreparer
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.directory_causal import CausalDirectoryPreparer
from corebehrt.main.helper.pretrain import (
    get_splits_path,
)
from corebehrt.constants.paths import FOLDS_FILE, TEST_PIDS_FILE


CONFIG_PATH = "./corebehrt/configs/causal/finetune/prepare_ft_exp_y.yaml"


def main(config_path):
    cfg = load_config(config_path)

    # Setup directories
    CausalDirectoryPreparer(cfg).setup_prepare_finetune_exposure_outcome()
    logger = logging.getLogger("prepare finetune data")
    logger.info("Preparing finetune data")
    # Prepare data
    _ = CausalDatasetPreparer(cfg).prepare_finetune_data(mode="tuning")

    # Save splits from cohort selection
    folds_path = get_splits_path(cfg.paths)
    folds = torch.load(folds_path)
    torch.save(folds, join(cfg.paths.prepared_data, FOLDS_FILE))
    test_pids_file = join(cfg.paths.cohort, TEST_PIDS_FILE)
    if os.path.exists(test_pids_file):
        test_pids = torch.load(test_pids_file)
        torch.save(test_pids, join(cfg.paths.prepared_data, TEST_PIDS_FILE))


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    config_path = args.config_path
    main(config_path)
