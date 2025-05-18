import logging

from corebehrt.functional.setup.args import get_args
from corebehrt.main_causal.helper.calibrate_exp_y import collect_and_save_predictions
from corebehrt.modules.setup.causal.directory import CausalDirectoryPreparer
from corebehrt.modules.setup.config import load_config

CONFIG_PATH = "./corebehrt/configs/causal/finetune/calibrate_exp_y.yaml"


def main_calibrate(config_path):
    cfg = load_config(config_path)

    # Setup directories
    CausalDirectoryPreparer(cfg).setup_calibrate()

    # Logger
    logger = logging.getLogger("calibrate")

    write_dir = cfg.paths.calibrated_predictions
    finetune_dir = cfg.paths.finetune_model

    logger.info("Collecting predictions")
    collect_and_save_predictions(finetune_dir, write_dir)
    logger.info("Computing and saving calibration")

    logger.info("Done")


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_calibrate(args.config_path)
