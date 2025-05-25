"""
Prepare data for finetune with exposure and outcome.
"""

import logging
from os.path import join

import torch

from corebehrt.constants.paths import FOLDS_FILE
from corebehrt.functional.setup.args import get_args
from corebehrt.functional.features.split import create_folds
from corebehrt.modules.preparation.causal.prepare_data import CausalDatasetPreparer
from corebehrt.modules.setup.causal.directory import CausalDirectoryPreparer
from corebehrt.modules.setup.config import load_config

CONFIG_PATH = "./corebehrt/configs/causal/finetune/prepare/ft_exp_y.yaml"


def main(config_path):
    cfg = load_config(config_path)

    # Setup directories
    CausalDirectoryPreparer(cfg).setup_prepare_finetune_exposure_outcome()
    logger = logging.getLogger("prepare finetune data")
    logger.info("Preparing finetune data")
    # Prepare data
    data = CausalDatasetPreparer(cfg).prepare_finetune_data(mode="tuning")
    # Save splits from cohort selection
    pids = data.get_pids()
    folds = create_folds(
        pids,
        cfg.data.get("cv_folds", 5),
        cfg.data.get("seed", 42),
        cfg.data.get("val_ratio", 0.2),
    )
    torch.save(folds, join(cfg.paths.prepared_data, FOLDS_FILE))


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    config_path = args.config_path
    main(config_path)
