"""
Causal inference models for EHR data.

This module extends the base Corebehrt models to support causal inference tasks,
enabling simultaneous prediction of both exposure and outcome variables.
"""

import logging

import torch
import torch.nn as nn

from corebehrt.constants.causal.data import EXPOSURE_TARGET
from corebehrt.constants.data import ATTENTION_MASK, TARGET
from corebehrt.modules.model.causal.heads import CausalFineTuneHead
from corebehrt.modules.model.model import CorebehrtForFineTuning

logger = logging.getLogger(__name__)


class CorebehrtForCausalFineTuning(CorebehrtForFineTuning):
    """
    Fine-tuning model for causal inference on EHR sequences.

    This model extends CorebehrtForFineTuning to simultaneously predict both exposure and outcome
    variables. It uses two separate classification heads, each with its own BCEWithLogitsLoss,
    to predict the probability of exposure and outcome occurrence. The final loss is computed
    as the sum of both individual losses.

    The model is designed for causal inference tasks where we need to predict both the treatment
    (exposure) and the outcome of interest, allowing for joint learning of both predictions.
    The exposure status is appended to the BiGRU output vector for outcome prediction.

    Attributes:
        exposure_loss_fct (nn.BCEWithLogitsLoss): Loss function for exposure prediction
        outcome_loss_fct (nn.BCEWithLogitsLoss): Loss function for outcome prediction
        exposure_cls (CausalFineTuneHead): Classification head for exposure prediction
        outcome_cls (CausalFineTuneHead): Classification head for outcome prediction with exposure input
        counterfactual (bool): Whether to use counterfactual exposure values
    """

    def __init__(self, config):
        super().__init__(config)

        # Initialize loss functions for both targets
        if getattr(config, "pos_weight_outcomes", None):
            pos_weight_outcomes = torch.tensor(config.pos_weight_outcomes)
        else:
            pos_weight_outcomes = None
        if getattr(config, "pos_weight_exposures", None):
            pos_weight_exposures = torch.tensor(config.pos_weight_exposures)
        else:
            pos_weight_exposures = None

        self.exposure_loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight_exposures)
        self.outcome_loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight_outcomes)

        # Two separate classification heads
        self.exposure_cls = CausalFineTuneHead(
            hidden_size=config.hidden_size, with_exposure=False
        )
        # Outcome head needs extra dimension for exposure
        self.outcome_cls = CausalFineTuneHead(
            hidden_size=config.hidden_size, with_exposure=True
        )

    def forward(self, batch: dict, cf: bool = False):
        """
        Forward pass for causal fine-tuning.

        Args:
            batch (dict): must contain:
                - 'concept', 'segment', 'age', 'abspos', 'attention_mask'
                - 'exposure_target' and 'target' as labels

        Returns:
            BaseModelOutput: with exposure_logits, outcome_logits and optional losses if targets provided.
        """
        outputs = super().forward(batch)

        sequence_output = outputs[0]  # Last hidden state

        # Get exposure prediction at sequence level
        exposure_logits = self.exposure_cls(
            sequence_output, batch[ATTENTION_MASK]
        )  # shape: (batch_size, 1)
        outputs.exposure_logits = exposure_logits

        # Get exposure status (0/1) and convert to -1/1
        exposure_status = batch[EXPOSURE_TARGET]
        if cf:
            exposure_status = 1 - exposure_status
        exposure_status = 2 * exposure_status - 1  # Convert from 0/1 to -1/1

        # Get outcome prediction using sequence output with exposure
        outcome_logits = self.outcome_cls(
            sequence_output, batch[ATTENTION_MASK], exposure_status=exposure_status
        )
        outputs.outcome_logits = outcome_logits

        # Calculate losses if targets are provided
        if batch.get(EXPOSURE_TARGET) is not None and batch.get(TARGET) is not None:
            exposure_loss = self.get_exposure_loss(
                exposure_logits, batch[EXPOSURE_TARGET]
            )
            outcome_loss = self.get_outcome_loss(outcome_logits, batch[TARGET])
            outputs.loss = exposure_loss + outcome_loss  # Combined loss
        return outputs

    def get_exposure_loss(self, logits, labels):
        """Calculate binary cross-entropy loss for exposure prediction."""
        return self.exposure_loss_fct(logits.view(-1), labels.view(-1))

    def get_outcome_loss(self, logits, labels):
        """Calculate binary cross-entropy loss for outcome prediction."""
        return self.outcome_loss_fct(logits.view(-1), labels.view(-1))
