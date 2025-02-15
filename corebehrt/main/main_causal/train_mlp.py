import logging
import os
from datetime import datetime
from os.path import join

import lightning as pl
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from corebehrt.constants.data import (ABSPOS_COL, TIMESTAMP_COL, TRAIN_KEY,
                                      VAL_KEY)
from corebehrt.constants.paths import DATA_CFG, FOLDS_FILE, INDEX_DATES_FILE
from corebehrt.functional.causal.load import (
    load_encodings_and_pids_from_encoded_dir, load_exposure_from_predictions)
from corebehrt.functional.cohort_handling.outcomes import get_binary_outcomes
from corebehrt.functional.setup.args import get_args
from corebehrt.functional.trainer.setup import replace_steps_with_epochs
from corebehrt.functional.utils.time import get_abspos_from_origin_point
from corebehrt.main.helper.causal.train_mlp import (
    check_val_fold_pids, combine_encodings_and_exposures)
from corebehrt.modules.model.mlp import LitMLP
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.directory import DirectoryPreparer
from corebehrt.modules.setup.initializer import Initializer

CONFIG_PATH = "./corebehrt/configs/causal/train_mlp.yaml"


def main_train(config_path):
    cfg = load_config(config_path)

    # Setup directories
    DirectoryPreparer(cfg).setup_train_mlp()

    # Logger
    logger = logging.getLogger("train_mlp")

    logger.info("Load data")

    # Encodings and exposure
    encodings, pids = load_encodings_and_pids_from_encoded_dir(cfg.paths.encoded_data)
    exposure = load_exposure_from_predictions(cfg.paths.calibrated_predictions, pids)
    X = combine_encodings_and_exposures(encodings, exposure)
    # load index_dates, and cohort pids
    cohort_dir = cfg.paths.cohort
    index_dates = pd.read_csv(
        join(cohort_dir, INDEX_DATES_FILE), parse_dates=[TIMESTAMP_COL]
    )
    origin_point = load_config(
        join(cfg.paths.encoded_data, DATA_CFG)
    ).features.origin_point
    index_dates[ABSPOS_COL] = get_abspos_from_origin_point(
        index_dates[TIMESTAMP_COL], datetime(**origin_point)
    )
    folds = torch.load(join(cohort_dir, FOLDS_FILE))

    check_val_fold_pids(folds, pids)

    # Load outcomes
    outcomes = pd.read_csv(cfg.paths.outcomes)
    binary_outcomes = get_binary_outcomes(
        index_dates,
        outcomes,
        cfg.outcome.n_hours_start_follow_up,
        cfg.outcome.n_hours_end_follow_up,
    )
    binary_outcomes = binary_outcomes.loc[pids]
    y = torch.tensor(binary_outcomes.values, dtype=torch.float32)

    for i, fold in enumerate(folds):
        logger.info(f"Training fold {i+1} of {len(folds)}")
        val_fold_pids = fold[VAL_KEY]
        train_fold_pids = fold[TRAIN_KEY]
        val_fold_ids = [i for i, pid in enumerate(pids) if pid in val_fold_pids]
        train_fold_ids = [i for i, pid in enumerate(pids) if pid in train_fold_pids]

        X_val = X[val_fold_ids]
        X_train = X[train_fold_ids]
        y_val = y[val_fold_ids]
        y_train = y[train_fold_ids]

        cfg.scheduler = replace_steps_with_epochs(
            cfg.scheduler, cfg.trainer_args.train_loader_kwargs.batch_size, len(X_train)
        )
        model = LitMLP(
            input_dim=X_train.shape[1],
            hidden_dims=cfg.model.hidden_dims,
            output_dim=1,
            dropout_rate=cfg.model.get("dropout_rate", 0.1),
            lr=cfg.optimizer.lr,
            scheduler_cfg=cfg.scheduler,
        )

        # Compile the model
        if torch.cuda.is_available():
            model = torch.compile(model)
        logger.info("Initializing training components")

        initializer = Initializer(cfg)
        sampler, _ = initializer.initialize_sampler(y_train)

        train_dataset = SimpleDataset(X_train, y_train)
        train_loader = DataLoader(
            dataset=train_dataset,
            sampler=sampler,  # use your custom sampler
            shuffle=(sampler is None),  # if you provide a sampler, disable shuffling
            **cfg.trainer_args.train_loader_kwargs,
        )

        val_dataset = SimpleDataset(X_val, y_val)
        val_loader = DataLoader(
            dataset=val_dataset, **cfg.trainer_args.val_loader_kwargs
        )

        fold_folder = join(cfg.paths.trained_mlp, f"fold_{i+1}")
        os.makedirs(fold_folder, exist_ok=True)

        trainer = pl.Trainer(
            max_epochs=cfg.trainer_args.epochs,
            callbacks=[
                ModelCheckpoint(
                    monitor=cfg.trainer_args.monitor.metric,
                    mode=cfg.trainer_args.monitor.mode,
                    save_top_k=1,
                ),
            ],
            precision=16,
            default_root_dir=fold_folder,
            gradient_clip_val=cfg.trainer_args.gradient_clip.clip_value,
            log_every_n_steps=1,
        )

        trainer.fit(model, train_loader, val_loader)


class ModelCheckpoint(pl.Callback):
    def __init__(self, monitor="val_loss", mode="min", save_top_k=1):
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.best_score = None

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        current_score = trainer.callback_metrics[self.monitor]
        if (
            self.best_score is None
            or self.mode == "min"
            and current_score < self.best_score
        ):
            self.best_score = current_score


class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = (
            X.to(torch.float32)
            if isinstance(X, torch.Tensor)
            else torch.tensor(X, dtype=torch.float32)
        )
        self.y = (
            y.to(torch.float32)
            if isinstance(y, torch.Tensor)
            else torch.tensor(y, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Return (features, label)
        return self.X[idx], self.y[idx]



if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_train(args.config_path)
