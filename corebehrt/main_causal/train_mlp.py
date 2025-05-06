import logging
import os
from os.path import join

import pandas as pd
import torch

from corebehrt.constants.causal.paths import CALIBRATED_PREDICTIONS_FILE
from corebehrt.constants.data import TRAIN_KEY, VAL_KEY
from corebehrt.functional.setup.args import get_args
from corebehrt.main_causal.helper.calibrate_mlp import calibrate_predictions
from corebehrt.main_causal.helper.train_mlp import setup_model, setup_trainer
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.directory_causal import CausalDirectoryPreparer
from corebehrt.modules.trainer.data_module import EncodedDataModule

CONFIG_PATH = "./corebehrt/configs/causal/double_robust/train_mlp.yaml"


def main_train(config_path):
    cfg = load_config(config_path)

    # Setup directories
    CausalDirectoryPreparer(cfg).setup_train_mlp()

    # Logger
    logger = logging.getLogger("train_mlp")

    # Log GPU information
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    else:
        logger.info("No GPU available, using CPU")

    data_module = EncodedDataModule(cfg, logger)
    data_module.setup()

    all_fold_results = []
    for fold_idx, fold in enumerate(data_module.folds):
        logger.info(f"FOLD {fold_idx + 1}/{len(data_module.folds)}")
        logger.info(f"validation patients: {len(fold[VAL_KEY])}")
        logger.info(f"training patients: {len(fold[TRAIN_KEY])}")

        fold_folder = join(cfg.paths.trained_mlp, f"fold_{fold_idx + 1}")
        os.makedirs(fold_folder, exist_ok=True)

        logger.info("Model input dim: %d", data_module.input_dim)
        logger.info("Get Data...")
        X_train, X_val, X_val_counter, y_train, y_val = data_module.get_fold_data(fold)
        logger.info(f"X_train shape: {X_train.shape}")
        logger.info(f"Num positive samples: {y_train.sum()}")

        logger.info("Initializing Data loaders...")
        train_loader, val_loader, val_cf_loader = data_module.get_fold_dataloaders(
            X_train,
            X_val,
            X_val_counter,
            y_train,
            y_val,
        )

        logger.info("Initializing model...")
        model = setup_model(
            cfg,
            num_features=data_module.input_dim,
            num_train_samples=len(fold[TRAIN_KEY]),
            y_train=y_train,
        )

        logger.info(f"Train loader size: {len(train_loader.dataset)}")
        logger.info(f"Validation loader size: {len(val_loader.dataset)}")

        logger.info("Setting up trainer...")
        trainer = setup_trainer(fold_folder, cfg.trainer_args)
        logger.info("Training...")
        trainer.fit(model, train_loader, val_loader)

        logger.info("Calibrating predictions...")

        fold_results = calibrate_predictions(
            model,
            train_loader,
            val_loader,
            val_cf_loader,
            val_pids=fold[VAL_KEY],
        )
        all_fold_results.append(fold_results)

    all_fold_results = pd.concat(all_fold_results)
    all_fold_results.to_csv(
        join(cfg.paths.trained_mlp, CALIBRATED_PREDICTIONS_FILE), index=False
    )


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_train(args.config_path)
