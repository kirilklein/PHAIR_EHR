import logging

from corebehrt.functional.setup.args import get_args
from corebehrt.modules.causal.estimate import EffectEstimator
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.directory_causal import CausalDirectoryPreparer

CONFIG_PATH = "./corebehrt/configs/causal/estimate/estimate.yaml"


def main_estimate(config_path):
    cfg = load_config(config_path)

    # Setup directories
    CausalDirectoryPreparer(cfg).setup_estimate()

    # Logger
    logger = logging.getLogger("estimate")

    estimator = EffectEstimator(cfg, logger)
    estimator.run()
    # Load data
    logger.info("Done")


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_estimate(args.config_path)
