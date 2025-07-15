"""
Causal inference models for EHR data.

This module extends the base Corebehrt models to support causal inference tasks,
enabling simultaneous prediction of both exposure and multiple outcome variables.
"""

import logging

import torch
import torch.nn as nn

from corebehrt.constants.causal.data import EXPOSURE_TARGET
from corebehrt.constants.data import ATTENTION_MASK
from corebehrt.modules.model.causal.heads import MLPHead, PatientRepresentationPooler
from corebehrt.modules.model.model import CorebehrtForFineTuning

logger = logging.getLogger(__name__)


class CorebehrtForCausalFineTuning(CorebehrtForFineTuning):
    """
    A unified causal fine-tuning model for predicting exposure and multiple outcomes.

    This class supports two architectures, controlled by `config.shared_representation`:
    1.  True (default): Uses a single, shared BiGRU pooler to create a patient
        representation for both tasks. This encourages learning a generalized
        and efficient representation.
    2.  False: Uses two independent BiGRU poolers, creating separate
        representations for the exposure and outcome tasks.

    The model uses separate MLP heads for exposure and each outcome prediction.
    """

    def __init__(self, config):
        super().__init__(config)

        # --- Architecture Configuration ---
        self.head_config = getattr(config, "head", {})
        self.shared_representation = self.head_config.get("shared_representation", True)
        self.bidirectional = self.head_config.get("bidirectional", True)

        # Get outcome names from config
        self.outcome_names = config.outcome_names

        self._setup_pooling_layers(config)
        self._setup_mlp_heads(config)
        self._setup_loss_functions(config)

    def _setup_pooling_layers(self, config):
        if self.shared_representation:
            logger.info("Using shared patient representation")
            # A single pooler for all tasks
            self.pooler = PatientRepresentationPooler(
                hidden_size=config.hidden_size, bidirectional=self.bidirectional
            )
        else:
            logger.info("Using separate patient representations")
            # Separate poolers for exposure and each outcome
            self.exposure_pooler = PatientRepresentationPooler(
                hidden_size=config.hidden_size, bidirectional=self.bidirectional
            )
            self.outcome_poolers = nn.ModuleDict()
            for outcome_name in self.outcome_names:
                self.outcome_poolers[outcome_name] = PatientRepresentationPooler(
                    hidden_size=config.hidden_size, bidirectional=self.bidirectional
                )

    def _setup_mlp_heads(self, config):
        self.exposure_head = MLPHead(input_size=config.hidden_size)

        # Create separate heads for each outcome
        self.outcome_heads = nn.ModuleDict()
        for outcome_name in self.outcome_names:
            self.outcome_heads[outcome_name] = MLPHead(
                input_size=config.hidden_size + 1  # +1 for exposure status
            )

    def _setup_loss_functions(self, config):
        """Helper method to initialize BCE loss functions with position weights."""
        # Setup exposure loss
        exposure_pos_weight = self._get_pos_weight_tensor(
            getattr(config, "pos_weight_exposures", None)
        )
        logger.info(f"pos_weight_exposures (loss): {exposure_pos_weight}")
        self.exposure_loss_fct = nn.BCEWithLogitsLoss(pos_weight=exposure_pos_weight)

        # Setup outcome losses
        self.outcome_loss_fcts = nn.ModuleDict()
        pos_weight_outcomes = getattr(config, "pos_weight_outcomes", {})

        for outcome_name in self.outcome_names:
            outcome_pos_weight = self._get_pos_weight_tensor(
                pos_weight_outcomes.get(outcome_name)
            )
            logger.info(f"pos_weight_{outcome_name} (loss): {outcome_pos_weight}")
            self.outcome_loss_fcts[outcome_name] = nn.BCEWithLogitsLoss(
                pos_weight=outcome_pos_weight
            )

    def _get_pos_weight_tensor(self, pos_weight_value):
        """Helper method to convert pos_weight value to tensor if not None."""
        if pos_weight_value is not None:
            return torch.tensor(pos_weight_value)
        return None

    def forward(self, batch: dict, cf: bool = False):
        """Forward pass for causal inference."""
        outputs = super().forward(batch)
        sequence_output = outputs[0]
        attention_mask = batch[ATTENTION_MASK]

        # --- Get Patient Representations ---
        if self.shared_representation:
            shared_repr = self.pooler(sequence_output, attention_mask)
            exposure_repr = shared_repr
            outcome_reprs = {
                outcome_name: shared_repr for outcome_name in self.outcome_names
            }
        else:
            exposure_repr = self.exposure_pooler(sequence_output, attention_mask)
            outcome_reprs = {
                outcome_name: self.outcome_poolers[outcome_name](
                    sequence_output, attention_mask
                )
                for outcome_name in self.outcome_names
            }

        # --- Exposure Prediction ---
        exposure_logits = self.exposure_head(exposure_repr)
        outputs.exposure_logits = exposure_logits

        # --- Multiple Outcome Predictions ---
        exposure_status = 2 * batch[EXPOSURE_TARGET] - 1  # Convert from 0/1 to -1/1
        if cf:
            exposure_status = -exposure_status  # Flip for counterfactual

        # Compute logits for each outcome using its specific representation
        outputs.outcome_logits = {}
        for outcome_name in self.outcome_names:
            outcome_input = torch.cat(
                (outcome_reprs[outcome_name], exposure_status.unsqueeze(-1)), dim=-1
            )  # [bs, hidden_size] vs [bs] -> [bs, hidden_size + 1]
            outputs.outcome_logits[outcome_name] = self.outcome_heads[outcome_name](
                outcome_input
            )

        # Only compute losses if we're training or if labels are available for evaluation
        if self.training or self._should_compute_losses(batch):
            self._compute_losses(outputs, batch)

        return outputs

    def _should_compute_losses(self, batch):
        """Check if we should compute losses based on available labels. If all outcome labels are present AND Exposure is present we can compute loss."""

        # Check if exposure label is present
        has_exposure_label = EXPOSURE_TARGET in batch

        # Check if any outcome labels are present
        has_outcome_labels = all(
            outcome_name in batch for outcome_name in self.outcome_names
        )

        return has_exposure_label and has_outcome_labels

    def _compute_losses(self, outputs, batch):
        """Helper method to compute and assign losses if labels are present."""
        total_loss = 0
        outputs.outcome_losses = {}

        # Only compute exposure loss if label is available
        if EXPOSURE_TARGET in batch:
            exposure_loss = self.exposure_loss_fct(
                outputs.exposure_logits.view(-1), batch[EXPOSURE_TARGET].view(-1)
            )
            outputs.exposure_loss = exposure_loss
            total_loss += exposure_loss

        # Only compute outcome losses for available labels
        for outcome_name in self.outcome_names:
            if outcome_name not in batch:
                continue
            predictions = outputs.outcome_logits[outcome_name].view(-1)
            targets = batch[outcome_name].view(-1)
            outcome_loss = self.outcome_loss_fcts[outcome_name](predictions, targets)

            outputs.outcome_losses[outcome_name] = outcome_loss
            total_loss += outcome_loss

        outputs.loss = total_loss
