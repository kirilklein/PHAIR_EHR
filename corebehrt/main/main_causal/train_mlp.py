import logging
import os
from os.path import join, split

from corebehrt.functional.setup.args import get_args
from corebehrt.main.helper.causal.train_mlp import (
    prepare_data,
    setup_data_loaders,
    setup_model,
    setup_trainer,
    split_fold_data,
)
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.directory import DirectoryPreparer

CONFIG_PATH = "./corebehrt/configs/causal/train_mlp.yaml"


def main_train(config_path):
    cfg = load_config(config_path)

    # Setup directories
    DirectoryPreparer(cfg).setup_train_mlp()

    # Logger
    logger = logging.getLogger("train_mlp")

    X, y, pids, folds = prepare_data(cfg, logger)

    for i, fold in enumerate(folds):
        fold_folder = join(cfg.paths.trained_mlp, f"fold_{i+1}")
        os.makedirs(fold_folder, exist_ok=True)
        logger.info(f"Training {split(fold_folder)[-1]} of {len(folds)}")

        logger.info("Splitting data")
        train_data, val_data = split_fold_data(X, y, fold, pids)

        logger.info("Setting up model")
        model = setup_model(cfg, train_data[0])

        train_loader, val_loader = setup_data_loaders(cfg, train_data, val_data)

        trainer = setup_trainer(fold_folder, cfg.trainer_args)
        trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_train(args.config_path)
