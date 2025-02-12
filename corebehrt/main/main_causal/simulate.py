import logging
from os.path import join

import torch
from corebehrt.constants.paths import ENCODINGS_FILE, PID_FILE
from corebehrt.functional.setup.args import get_args
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.directory import DirectoryPreparer
from corebehrt.main.helper.causal.simulate import simulate

CONFIG_PATH = "./corebehrt/configs/simulate.yaml"


def main_simulate(config_path):
    cfg = load_config(config_path)

    # Setup directories
    DirectoryPreparer(cfg).setup_simulate()

    # Logger
    logger = logging.getLogger("simulate")

    # Load data
    encoded_path = join(cfg.paths.encoded_data, ENCODINGS_FILE)
    data = torch.load(encoded_path)
    pids = torch.load(join(cfg.paths.encoded_data, PID_FILE))

    # Simulate
    simulate(logger, data, pids, cfg.simulation)

    logger.info("Done")


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_simulate(args.config_path)
