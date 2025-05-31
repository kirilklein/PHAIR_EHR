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
from corebehrt.modules.model.causal.heads import CausalBiGRU, CausalMLP
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

        self.shared_pool_cls = CausalBiGRU(hidden_size=config.hidden_size)

        # Separate normalizations for different input sizes
        self.exposure_norm = nn.LayerNorm(self.shared_pool_cls.pooled_size)  # Size: 64
        self.outcome_norm = nn.LayerNorm(
            self.shared_pool_cls.pooled_size + 1
        )  # Size: 65 (64 + 1 for exposure)

        # Separate MLP classifiers
        self.exposure_cls = CausalMLP(self.shared_pool_cls.pooled_size)
        self.outcome_cls = CausalMLP(
            self.shared_pool_cls.pooled_size + 1
        )  # +1 for exposure status

        self.register_buffer("exposure_weight", torch.tensor(1.0))
        self.register_buffer("outcome_weight", torch.tensor(1.0))

    def update_task_weights(self, exposure_weight, outcome_weight):
        self.exposure_weight.fill_(exposure_weight)
        self.outcome_weight.fill_(outcome_weight)

    def forward(self, batch: dict, cf: bool = False):
        """
        Forward pass for causal fine-tuning.
        """
        outputs = super().forward(batch)
        sequence_output = outputs[0]  # Last hidden state

        # Get shared pooled representation (without exposure)
        shared_embedding = self.shared_pool_cls(sequence_output, batch[ATTENTION_MASK])

        # Get exposure prediction from shared embedding (with normalization)
        exposure_input = self.exposure_norm(shared_embedding)
        exposure_logits = self.exposure_cls(exposure_input)
        outputs.exposure_logits = exposure_logits

        # Get exposure status (0/1) and convert to -1/1 as a new tensor
        exposure_status = batch[EXPOSURE_TARGET].float()  # Ensure it's float
        if cf:
            exposure_status = 1.0 - exposure_status
        exposure_status = 2.0 * exposure_status - 1.0  # Convert from 0/1 to -1/1

        # For outcome prediction: concatenate exposure status to shared embedding
        outcome_input = torch.cat(
            [shared_embedding, exposure_status.unsqueeze(-1)], dim=-1
        )
        # Apply normalization to the concatenated input (now with correct size)
        outcome_input = self.outcome_norm(outcome_input)
        outcome_logits = self.outcome_cls(outcome_input)
        outputs.outcome_logits = outcome_logits

        # Calculate losses if targets are provided
        if batch.get(EXPOSURE_TARGET) is not None and batch.get(TARGET) is not None:
            exposure_loss = self.get_exposure_loss(
                exposure_logits, batch[EXPOSURE_TARGET]
            )
            outcome_loss = self.get_outcome_loss(outcome_logits, batch[TARGET])

            # Store individual losses for tracking
            outputs.exposure_loss = exposure_loss
            outputs.outcome_loss = outcome_loss
            outputs.loss = exposure_loss + outcome_loss  # Combined loss
        return outputs

    def get_exposure_loss(self, logits, labels):
        """Calculate binary cross-entropy loss for exposure prediction."""
        return self.exposure_loss_fct(logits.view(-1), labels.view(-1))

    def get_outcome_loss(self, logits, labels):
        """Calculate binary cross-entropy loss for outcome prediction."""
        return self.outcome_loss_fct(logits.view(-1), labels.view(-1))
