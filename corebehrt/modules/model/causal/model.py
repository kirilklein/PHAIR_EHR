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
from corebehrt.modules.model.causal.heads import MLPHead, PatientRepresentationPooler
from corebehrt.modules.model.model import CorebehrtForFineTuning

logger = logging.getLogger(__name__)


class CorebehrtForCausalFineTuning(CorebehrtForFineTuning):
    """
    A unified causal fine-tuning model for predicting exposure and outcome.

    This class supports two architectures, controlled by `config.shared_representation`:
    1.  True (default): Uses a single, shared BiGRU pooler to create a patient
        representation for both tasks. This encourages learning a generalized
        and efficient representation.
    2.  False: Uses two independent BiGRU poolers, creating separate
        representations for the exposure and outcome tasks.

    The model always uses two separate MLP heads for the final predictions.
    """

    def __init__(self, config):
        super().__init__(config)

        # --- Architecture Configuration ---
        self.shared_representation = getattr(config, "shared_representation", True)
        bidirectional = getattr(config, "bidirectional", True)

        # --- 1. Pooling Layer(s) ---
        if self.shared_representation:
            # A single pooler for both tasks
            self.pooler = PatientRepresentationPooler(
                hidden_size=config.hidden_size, bidirectional=bidirectional
            )
        else:
            # Separate poolers for each task
            self.exposure_pooler = PatientRepresentationPooler(
                hidden_size=config.hidden_size, bidirectional=bidirectional
            )
            self.outcome_pooler = PatientRepresentationPooler(
                hidden_size=config.hidden_size, bidirectional=bidirectional
            )

        # --- 2. MLP Prediction Heads (Common to both architectures) ---
        self.exposure_head = MLPHead(input_size=config.hidden_size)
        self.outcome_head = MLPHead(
            input_size=config.hidden_size + 1
        )  # +1 for exposure status

        # --- 3. Loss Functions (Common to both architectures) ---
        self._setup_loss_functions(config)

    def _setup_loss_functions(self, config):
        """Helper method to initialize BCE loss functions with position weights."""
        pos_weight_outcomes = (
            torch.tensor(config.pos_weight_outcomes)
            if hasattr(config, "pos_weight_outcomes")
            else None
        )
        pos_weight_exposures = (
            torch.tensor(config.pos_weight_exposures)
            if hasattr(config, "pos_weight_exposures")
            else None
        )

        self.exposure_loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight_exposures)
        self.outcome_loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight_outcomes)

    def forward(self, batch: dict, cf: bool = False):
        """Forward pass for causal inference."""
        outputs = super().forward(batch)
        sequence_output = outputs[0]
        attention_mask = batch[ATTENTION_MASK]

        # --- Get Patient Representations ---
        if self.shared_representation:
            shared_repr = self.pooler(sequence_output, attention_mask)
            exposure_repr, outcome_repr = shared_repr, shared_repr
        else:
            exposure_repr = self.exposure_pooler(sequence_output, attention_mask)
            outcome_repr = self.outcome_pooler(sequence_output, attention_mask)

        # --- Exposure Prediction ---
        exposure_logits = self.exposure_head(exposure_repr)
        outputs.exposure_logits = exposure_logits

        # --- Outcome Prediction ---
        exposure_status = 2 * batch[EXPOSURE_TARGET] - 1  # Convert from 0/1 to -1/1
        if cf:
            exposure_status = -exposure_status  # Flip for counterfactual

        outcome_input = torch.cat((outcome_repr, exposure_status.unsqueeze(-1)), dim=-1)
        outcome_logits = self.outcome_head(outcome_input)
        outputs.outcome_logits = outcome_logits

        # --- Loss Calculation ---
        self._compute_losses(outputs, batch)

        return outputs

    def _compute_losses(self, outputs, batch):
        """Helper method to compute and assign losses if labels are present."""
        if EXPOSURE_TARGET in batch and TARGET in batch:
            exposure_loss = self.exposure_loss_fct(
                outputs.exposure_logits.view(-1), batch[EXPOSURE_TARGET].view(-1)
            )
            outcome_loss = self.outcome_loss_fct(
                outputs.outcome_logits.view(-1), batch[TARGET].view(-1)
            )

            outputs.exposure_loss = exposure_loss
            outputs.outcome_loss = outcome_loss
            outputs.loss = exposure_loss + outcome_loss
