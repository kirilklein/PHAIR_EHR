import lightning as pl
import torch

from corebehrt.functional.trainer.setup import replace_steps_with_epochs
from corebehrt.modules.model.mlp import LitMLP
from corebehrt.modules.trainer.checkpoint import ModelCheckpoint
from corebehrt.azure.util.pl_logger import AzureLogger
from corebehrt import azure


def setup_model(
    cfg: dict, num_features: int, num_train_samples: int
) -> pl.LightningModule:
    """Sets up the model with the given configuration and training data."""
    cfg.scheduler = replace_steps_with_epochs(
        cfg.scheduler,
        cfg.trainer_args.train_loader_kwargs.batch_size,
        num_train_samples,
    )
    model = LitMLP(
        input_dim=num_features,
        hidden_dims=cfg.model.hidden_dims,
        output_dim=1,
        dropout_rate=cfg.model.get("dropout_rate", 0.1),
        lr=cfg.optimizer.lr,
        scheduler_cfg=cfg.scheduler,
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
        log_every_n_steps=1,
        logger=logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
    )
