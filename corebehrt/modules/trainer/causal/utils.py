from dataclasses import dataclass
from typing import Dict


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
    val_loss: float
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    val_exposure_loss: float
    val_outcome_losses: Dict[str, float]
    val_consistency_losses: Dict[str, float]
    test_exposure_loss: float
    test_outcome_losses: Dict[str, float]
    test_consistency_losses: Dict[str, float]
