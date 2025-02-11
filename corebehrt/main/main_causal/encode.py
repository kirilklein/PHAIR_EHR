import logging
from os.path import join

import torch
from corebehrt.constants.paths import FOLDS_FILE, PROCESSED_DATA_DIR
from corebehrt.functional.setup.args import get_args
from corebehrt.modules.preparation.dataset import PatientDataset
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.directory import DirectoryPreparer
from corebehrt.main.helper.causal.encode import encode_loop

CONFIG_PATH = "./corebehrt/configs/encode.yaml"


def main_encode(config_path):
    cfg = load_config(config_path)

    # Setup directories
    DirectoryPreparer(cfg).setup_encode()

    # Logger
    logger = logging.getLogger("encode")

    # Load data
    processed_data_dir = join(cfg.paths.finetune_model, PROCESSED_DATA_DIR)
    data = PatientDataset.load(processed_data_dir)
    folds = torch.load(join(processed_data_dir, FOLDS_FILE))

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
