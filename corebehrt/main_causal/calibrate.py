import logging

from corebehrt.functional.setup.args import get_args
from corebehrt.main_causal.helper.calibrate import (
    compute_and_save_calibration,
    save_combined_predictions,
)
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.directory_causal import CausalDirectoryPreparer

CONFIG_PATH = "./corebehrt/configs/causal/finetune/calibrate.yaml"


def main_calibrate(config_path):
    cfg = load_config(config_path)

    # Setup directories
    CausalDirectoryPreparer(cfg).setup_calibrate()

    # Logger
    logger = logging.getLogger("calibrate")
    write_dir = cfg.paths.calibrated_predictions
    finetune_dir = cfg.paths.finetune_model

    logger.info("Saving combined predictions")
    save_combined_predictions(logger, write_dir, finetune_dir)

    logger.info("Computing and saving calibration")
    compute_and_save_calibration(logger, write_dir, finetune_dir)

    logger.info("Done")


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_calibrate(args.config_path)
