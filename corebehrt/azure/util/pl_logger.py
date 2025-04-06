from typing import Any, Dict, Optional, Union
from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities import rank_zero_only
from corebehrt import azure


class AzureLogger(Logger):
    """Minimal Lightning logger that logs metrics to Azure/MLflow."""

    def __init__(self, name: str = "lightning_logs"):
        super().__init__()
        self._name = name
        self._version = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> Union[int, str]:
        return self._version

    @rank_zero_only
    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        # Filter out None values and convert tensors/arrays to python types
        filtered_params = {}
        for k, v in params.items():
            if v is not None:
                if hasattr(v, "item"):  # Handle torch tensors
                    v = v.item()
                elif hasattr(v, "tolist"):  # Handle numpy arrays
                    v = v.tolist()
                filtered_params[k] = v

        for k, v in filtered_params.items():
            try:
                azure.log_param(k, v)
            except Exception as e:
                print(f"Warning: Could not log parameter {k}: {e}")

    @rank_zero_only
    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log metrics."""
        if azure.is_mlflow_available():
            metric_list = []
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    metric_list.append(azure.metric(k, v, step))
            if metric_list:
                azure.log_batch(metrics=metric_list)

    def save(self) -> None:
        """Nothing to save."""
        pass

    @rank_zero_only
    def finalize(self, status: str) -> None:
        """Nothing to finalize."""
        pass
