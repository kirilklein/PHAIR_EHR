import torch.nn.functional as F
from torch import nn
import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class FocalLossWithLogits(nn.Module):
    """
    Focal Loss for binary classification tasks with logits.

    It's designed to address class imbalance.
    See: https://arxiv.org/abs/1708.02002
    """

    def __init__(
        self,
        alpha: Optional[float] = None,  # same as no weighing
        gamma: float = 2.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets.float(), reduction="none"
        )
        p_t = torch.exp(
            -bce_loss
        )  # This exactly recovers the needed probability: probs * targets + (1 - probs) * (1 - targets)

        # Calculate alpha_t for each sample
        if self.alpha is not None:
            alpha_t = self.alpha * targets.float() + (1 - self.alpha) * (
                1 - targets.float()
            )  # alpha_t is the class weight for each sample
        else:
            alpha_t = 1.0

        # calculate weight
        difficulty_weight = (1 - p_t) ** self.gamma  # difficulty weight
        focal_loss = (
            alpha_t * difficulty_weight * bce_loss
        )  # focal loss is the weighted loss for each sample
        return focal_loss.mean()
