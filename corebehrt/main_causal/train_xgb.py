import json
import logging
import os
from os.path import join

import pandas as pd

from corebehrt.constants.causal.paths import CALIBRATED_PREDICTIONS_FILE
from corebehrt.constants.data import TRAIN_KEY, VAL_KEY
from corebehrt.functional.setup.args import get_args
from corebehrt.main_causal.helper.calibrate_xgb import calibrate_predictions
from corebehrt.main_causal.helper.train_xgb import (
    calculate_metrics,
    initialize_metrics,
    setup_xgb_params,
    train_xgb_model,
)
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.directory_causal import CausalDirectoryPreparer
from corebehrt.modules.trainer.data_module import EncodedDataModule

CONFIG_PATH = "./corebehrt/configs/causal/double_robust/train_xgb.yaml"


def main_train(config_path: str):
    cfg = load_config(config_path)

    # Setup directories
    CausalDirectoryPreparer(cfg).setup_train_xgb()

    # Logger
    logger = logging.getLogger("train_xgb")

    # Setup data
    data_module = EncodedDataModule(cfg, logger)
    data_module.setup()

    # Setup XGBoost parameters
    params, param_space = setup_xgb_params(cfg.model)

    all_fold_results = []
    for fold_idx, fold in enumerate(data_module.folds):
        logger.info(f"FOLD {fold_idx + 1}/{len(data_module.folds)}")
        logger.info(f"validation patients: {len(fold[VAL_KEY])}")
        logger.info(f"training patients: {len(fold[TRAIN_KEY])}")

        fold_folder = join(cfg.paths.trained_xgb, f"fold_{fold_idx + 1}")
        os.makedirs(fold_folder, exist_ok=True)

        # Get training data only for model training and hyperparameter tuning
        X_train, X_val, X_val_counter, y_train, y_val = data_module.get_fold_data(fold)

        # Train model (validation data not used in training anymore)
        logger.info("Training XGBoost model...")
        model = train_xgb_model(
            X_train,
            y_train,
            X_val,
            y_val,
            params,
            param_space,
            n_trials=cfg.model.get("n_trials", 20),
            cv=cfg.model.get("cv", 5),
            scoring=cfg.model.get("scoring", "neg_log_loss"),
            early_stopping_rounds=cfg.model.get("early_stopping_rounds", 10),
        )

        logger.info("Validation metrics:")
        metrics = initialize_metrics(cfg.model.get("metrics", None))
        scores = calculate_metrics(model, X_val, y_val, metrics)
        # Save scores dictionary
        with open(join(fold_folder, "scores.json"), "w") as f:
            json.dump(scores, f, indent=4)

        # Save model
        model.save_model(join(fold_folder, "model.json"))

        # Calibrate predictions
        logger.info("Calibrating predictions...")
        fold_results = calibrate_predictions(
            model, X_train, X_val, X_val_counter, y_train, y_val, val_pids=fold[VAL_KEY]
        )
        all_fold_results.append(fold_results)

    # Combine and save results
    all_fold_results = pd.concat(all_fold_results)
    all_fold_results.to_csv(
        join(cfg.paths.trained_xgb, CALIBRATED_PREDICTIONS_FILE), index=False
    )


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_train(args.config_path)
