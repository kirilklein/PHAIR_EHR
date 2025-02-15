import lightning as pl
import torch.nn as nn
from torch.optim import AdamW
from torcheval.metrics.functional.classification import binary_auroc

from corebehrt.modules.setup.config import instantiate_class


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.0):
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
        lr=1e-3,
        scheduler_cfg=None,
    ):
        super().__init__()
        self.scheduler_cfg = scheduler_cfg
        self.save_hyperparameters()

        self.model = MLP(input_dim, hidden_dims, output_dim, dropout_rate)

        self.criterion = nn.BCEWithLogitsLoss()  # for binary classification

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
            binary_auroc(logits, y),
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
