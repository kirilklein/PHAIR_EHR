import logging
from os.path import join

import torch

from corebehrt.constants.data import TRAIN_KEY, VAL_KEY
from corebehrt.constants.paths import FOLDS_FILE
from corebehrt.functional.setup.args import get_args
from corebehrt.functional.utils.filter import filter_folds_by_pids
from corebehrt.main_causal.helper.encode import encode_loop
from corebehrt.modules.preparation.dataset import PatientDataset
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.directory_causal import CausalDirectoryPreparer

CONFIG_PATH = "./corebehrt/configs/causal/finetune/encode.yaml"


def main_encode(config_path):
    cfg = load_config(config_path)

    # Setup directories
    CausalDirectoryPreparer(cfg).setup_encode()

    # Logger
    logger = logging.getLogger("encode")

    # Load data
    prepared_data_dir = cfg.paths.prepared_data
    data = PatientDataset.load(prepared_data_dir)
    logger.info(f"Data patients: {len(data.patients)}")
    folds = torch.load(join(prepared_data_dir, FOLDS_FILE))
    folds = filter_folds_by_pids(folds, data.get_pids())
    logger.info(
        f"Folds patients: {sum(len(fold[TRAIN_KEY]) + len(fold[VAL_KEY]) for fold in folds)}"
    )
    # Run encoding loop
    encode_loop(
        logger,
        cfg.paths.finetune_model,
        cfg.paths.encoded_data,
        data,
        folds,
        cfg.loader,
    )

    logger.info("Done")


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_encode(args.config_path)
