import logging
from os.path import join

import pandas as pd
import torch

from corebehrt.constants.causal import (
    CALIBRATED_PREDICTIONS_FILE,
    SIMULATION_RESULTS_FILE,
    TIMESTAMP_OUTCOME_FILE,
)
from corebehrt.constants.paths import ENCODINGS_FILE, PID_FILE
from corebehrt.functional.causal.data_utils import align_df_with_pids
from corebehrt.functional.setup.args import get_args
from corebehrt.main.helper.causal.simulate import simulate
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
    encodings = torch.load(join(cfg.paths.encoded_data, ENCODINGS_FILE)).numpy()
    pids = torch.load(join(cfg.paths.encoded_data, PID_FILE))

    logger.info("Load calibrated predictions")
    predictions = pd.read_csv(
        join(cfg.paths.calibrated_predictions, CALIBRATED_PREDICTIONS_FILE)
    )
    predictions = align_df_with_pids(predictions, pids)

    logger.info("Simulate")
    result_df, timestamp_df = simulate(logger, encodings, predictions, cfg.simulation)
    logger.info("Done")

    logger.info("Save results")
    result_df.to_csv(
        join(cfg.paths.simulated_outcome, SIMULATION_RESULTS_FILE), index=False
    )
    timestamp_df.to_csv(
        join(cfg.paths.simulated_outcome, TIMESTAMP_OUTCOME_FILE), index=False
    )


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_simulate(args.config_path)
