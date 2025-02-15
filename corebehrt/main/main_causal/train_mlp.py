import logging
import os
from os.path import join

import lightning as pl
import torch
from torch.utils.data import DataLoader

from corebehrt.constants.data import TRAIN_KEY, VAL_KEY

from corebehrt.functional.setup.args import get_args
from corebehrt.functional.trainer.setup import replace_steps_with_epochs

from corebehrt.main.helper.causal.train_mlp import prepare_data
from corebehrt.modules.model.mlp import LitMLP
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.directory import DirectoryPreparer
from corebehrt.modules.setup.initializer import Initializer
from corebehrt.modules.trainer.dataset import SimpleDataset
from corebehrt.modules.trainer.checkpoint import ModelCheckpoint

CONFIG_PATH = "./corebehrt/configs/causal/train_mlp.yaml"


def main_train(config_path):
    cfg = load_config(config_path)

    # Setup directories
    DirectoryPreparer(cfg).setup_train_mlp()

    # Logger
    logger = logging.getLogger("train_mlp")

    X, y, pids, folds = prepare_data(cfg, logger)

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


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_train(args.config_path)
