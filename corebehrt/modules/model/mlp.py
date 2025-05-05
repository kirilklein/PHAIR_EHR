import lightning as pl
from typing import List
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW

from corebehrt.modules.setup.config import instantiate_class


class MLP(nn.Module):
    """
    Simple Multi-Layer Perceptron (MLP) for tabular data.

    Constructs a feedforward neural network with configurable hidden layers, ReLU activations,
    and optional dropout. The final layer is linear (no activation).

    Args:
        input_dim (int): Number of input features.
        hidden_dims (list of int): Sizes of hidden layers.
        output_dim (int): Number of output units.
        dropout_rate (float, optional): Dropout rate after each hidden layer (default: 0.0).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        for i in range(len(hidden_dims) - 1):
            layers.extend(
                [
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.ReLU() if i < len(hidden_dims) - 2 else nn.Identity(),
                    (
                        nn.Dropout(dropout_rate)
                        if i < len(hidden_dims) - 2
                        else nn.Identity()
                    ),
                ]
            )
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class LitMLP(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim=1,
        dropout_rate=0.0,
        lr=1e-3,  # accessed internally by the optimizer
        scheduler_cfg=None,
        loss_weight=None,
    ):
        """
        PyTorch Lightning module for training a Multi-Layer Perceptron (MLP) on tabular data.

        Supports binary classification with optional positive class weighting for imbalanced datasets.
        Handles training, validation, and optimizer/scheduler setup.

        Args:
            input_dim (int): Number of input features.
            hidden_dims (list of int): Hidden layer sizes.
            output_dim (int, optional): Output size (default: 1).
            dropout_rate (float, optional): Dropout rate (default: 0.0).
            lr (float, optional): Learning rate (default: 1e-3).
            scheduler_cfg (dict, optional): Scheduler config.
            loss_weight (float or torch.Tensor, optional): Positive class weight for BCEWithLogitsLoss.
        """
        super().__init__()
        self.scheduler_cfg = scheduler_cfg
        self.save_hyperparameters()

        self.model = MLP(input_dim, hidden_dims, output_dim, dropout_rate)
        if loss_weight is None:
            self.criterion = (
                nn.BCEWithLogitsLoss()
            )  # No reduction so far, so we can use the weights
        else:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=loss_weight)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze(-1)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze(-1)
        loss = self.criterion(logits, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "val_roc_auc",
            roc_auc_score(y.cpu().float().numpy(), logits.cpu().float().numpy()),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        # Instantiate scheduler from your config
        scheduler = instantiate_class(self.scheduler_cfg, optimizer=optimizer)

        # Return as a dict to ensure the scheduler is called per step or per epoch as needed.
        return [optimizer], [scheduler]
