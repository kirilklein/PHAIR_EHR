from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class CausalPredictionData:
    logits_list: list
    metric_values: dict = None
    targets_list: list = None
    target_key: str = None


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: Optional[float] = None
    val_metrics: Optional[Dict] = None
    test_metrics: Optional[Dict] = None
    val_exposure_loss: Optional[float] = None
    val_outcome_losses: Optional[Dict] = None
    test_exposure_loss: Optional[float] = None
    test_outcome_losses: Optional[Dict] = None
