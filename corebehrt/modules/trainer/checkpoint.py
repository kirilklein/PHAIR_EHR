import lightning as pl
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