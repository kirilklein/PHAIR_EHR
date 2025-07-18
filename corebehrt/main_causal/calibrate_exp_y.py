import logging

from corebehrt.functional.setup.args import get_args
from corebehrt.modules.plot.calibration import PlottingManager
from corebehrt.modules.setup.causal.artifacts import CalibrationArtifacts
from corebehrt.modules.setup.causal.directory import CausalDirectoryPreparer
from corebehrt.modules.setup.causal.path_manager import CalibrationPathManager
from corebehrt.modules.setup.causal.prediction_processor import CalibrationProcessor
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

    # 1. Setup managers
    path_manager = CalibrationPathManager(finetune_dir, write_dir)
    prediction_processor = CalibrationProcessor(path_manager, finetune_dir)
    plotting_manager = PlottingManager(path_manager)

    # 2. Collect raw predictions
    logger.info("Collecting and saving predictions...")
    prediction_processor.collect_and_save_all_predictions()

    # 3. Calibrate predictions
    logger.info("Calibrating predictions...")
    calibrated_data: CalibrationArtifacts = (
        prediction_processor.load_calibrate_and_save_all()
    )

    # 4. Generate plots
    logger.info("Generating plots...")
    plotting_manager.generate_all_plots(calibrated_data)

    logger.info("Pipeline finished successfully!")


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_calibrate(args.config_path)
