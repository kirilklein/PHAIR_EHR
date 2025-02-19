import logging
from os.path import join

from corebehrt.constants.causal import SIMULATION_RESULTS_FILE, TIMESTAMP_OUTCOME_FILE
from corebehrt.constants.paths import DATA_CFG
from corebehrt.functional.causal.load import (
    load_encodings_and_pids_from_encoded_dir,
    load_exposure_from_predictions,
)
from corebehrt.functional.setup.args import get_args
from corebehrt.main.helper.causal.simulate import add_abspos_to_df, simulate
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.directory import DirectoryPreparer

CONFIG_PATH = "./corebehrt/configs/causal/simulate.yaml"


def main_simulate(config_path):
    cfg = load_config(config_path)

    # Setup directories
    DirectoryPreparer(cfg).setup_simulate()

    # Logger
    logger = logging.getLogger("simulate")

    logger.info("Load data")
    encodings, pids = load_encodings_and_pids_from_encoded_dir(cfg.paths.encoded_data)
    logger.info("Load calibrated predictions")

    exposure = load_exposure_from_predictions(cfg.paths.calibrated_predictions, pids)

    logger.info("Simulate")
    result_df, timestamp_df = simulate(
        logger, pids, encodings, exposure, cfg.simulation
    )

    # Post-process timestamps using the provided origin point.
    data_cfg = load_config(join(cfg.paths.encoded_data, DATA_CFG))
    timestamp_df = add_abspos_to_df(timestamp_df, data_cfg.features.origin_point)

    logger.info("Save results")
    result_df.to_csv(
        join(cfg.paths.simulated_outcome, SIMULATION_RESULTS_FILE), index=False
    )
    timestamp_df.to_csv(
        join(cfg.paths.simulated_outcome, TIMESTAMP_OUTCOME_FILE), index=False
    )

    logger.info("Done")


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_simulate(args.config_path)
