import logging

from corebehrt.functional.setup.args import get_args
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.directory import DirectoryPreparer

CONFIG_PATH = "./corebehrt/configs/causal/train_mlp.yaml"


def main_train(config_path):
    cfg = load_config(config_path)

    # Setup directories
    DirectoryPreparer(cfg).setup_train()

    # Logger
    logger = logging.getLogger("train_mlp")

    logger.info("Load data")


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_train(args.config_path)
