import logging
from os.path import join

import torch

from corebehrt.constants.data import TRAIN_KEY, VAL_KEY
from corebehrt.constants.paths import FOLDS_FILE, PROCESSED_DATA_DIR
from corebehrt.functional.setup.args import get_args
from corebehrt.functional.utils.filter import filter_folds_by_pids
from corebehrt.main.helper.causal.encode import encode_loop
from corebehrt.modules.preparation.dataset import PatientDataset
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.directory import DirectoryPreparer

CONFIG_PATH = "./corebehrt/configs/causal/encode.yaml"


def main_encode(config_path):
    cfg = load_config(config_path)

    # Setup directories
    DirectoryPreparer(cfg).setup_encode()

    # Logger
    logger = logging.getLogger("encode")

    # Load data
    processed_data_dir = join(cfg.paths.finetune_model, PROCESSED_DATA_DIR)
    data = PatientDataset.load(processed_data_dir)
    logger.info(f"Data patients: {len(data.patients)}")
    folds = torch.load(join(processed_data_dir, FOLDS_FILE))
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
