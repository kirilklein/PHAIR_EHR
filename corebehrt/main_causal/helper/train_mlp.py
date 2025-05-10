import logging

import lightning as pl
import torch

from corebehrt import azure
from corebehrt.azure.util.pl_logger import AzureLogger
from corebehrt.functional.trainer.setup import replace_steps_with_epochs
from corebehrt.modules.model.mlp import LitMLP
from corebehrt.modules.trainer.checkpoint import ModelCheckpoint
from corebehrt.modules.trainer.utils import get_loss_weight

logger = logging.getLogger(__name__)


def setup_model(
    cfg: dict, num_features: int, num_train_samples: int, y_train: torch.Tensor = None
) -> pl.LightningModule:
    """Sets up the model with the given configuration and training data."""
    cfg.scheduler = replace_steps_with_epochs(
        cfg.scheduler,
        cfg.trainer_args.train_loader_kwargs.batch_size,
        num_train_samples,
    )

    if cfg.trainer_args.get("loss_weight_function") is None:
        loss_weight = None
    else:
        if y_train is None:
            raise ValueError(
                "y_train must be provided if loss_weight_function is defined"
            )
        loss_weight_val = float(get_loss_weight(cfg, y_train.tolist()))
        loss_weight = torch.tensor(loss_weight_val, dtype=torch.float32)
        logger.info("Loss weight: %.4f", loss_weight_val)
    model = LitMLP(
        input_dim=num_features,
        hidden_dims=cfg.model.hidden_dims,
        output_dim=1,
        dropout_rate=cfg.model.get("dropout_rate", 0.1),
        lr=cfg.optimizer.lr,
        scheduler_cfg=cfg.scheduler,
        loss_weight=loss_weight,
    )
    # Compile the model
    if torch.cuda.is_available() and cfg.trainer_args.get("compile", False):
        model = torch.compile(model)
    return model


def setup_trainer(root_folder: str, trainer_args: dict) -> pl.Trainer:
    """Setup the Lightning Trainer with logging."""
    callbacks = [
        ModelCheckpoint(
            monitor=trainer_args.monitor.metric,
            mode=trainer_args.monitor.mode,
            save_top_k=1,
        ),
    ]

    # Initialize logger only if MLflow is available
    logger = None
    if azure.is_mlflow_available():
        logger = AzureLogger(name="my_mlp_trainer")

    return pl.Trainer(
        max_epochs=trainer_args.epochs,
        callbacks=callbacks,
        precision=trainer_args.get("precision", 16),
        default_root_dir=root_folder,
        gradient_clip_val=trainer_args.get("gradient_clip", {}).get("clip_value", None),
        log_every_n_steps=10,
        val_check_interval=0.25,
        logger=logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
    )
