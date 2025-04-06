from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities.rank_zero import rank_zero_only
import corebehrt.azure as azure


class AzureLogger(Logger):
    """Minimal Lightning logger that logs metrics to Azure/MLflow."""

    def __init__(self, name: str = "AzureLogger", prefix: str = ""):
        super().__init__()
        self._name = name
        self._prefix = prefix

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        # can be any version string
        return "0"

    @property
    def experiment(self):
        # can return the azure module, or None
        return azure

    @rank_zero_only
    def log_hyperparams(self, params):
        if azure.is_mlflow_available():
            for k, v in params.items():
                azure.log_param(k, v)
        else:
            print("[azure_logger] log_hyperparams:", params)

    @rank_zero_only
    def log_metrics(self, metrics, step: int):
        if azure.is_mlflow_available():
            for k, v in metrics.items():
                azure.log_metric(k, v, step=step)
        else:
            print("[azure_logger] log_metrics:", metrics, "step:", step)

    def finalize(self, status):
        # (Optional) Called when the trainer is done.
        pass
