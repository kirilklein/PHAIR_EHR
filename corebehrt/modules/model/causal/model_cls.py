"""
Causal inference models for EHR data.

This module extends the base Corebehrt models to support causal inference tasks,
enabling simultaneous prediction of both exposure and multiple outcome variables.
"""

import logging

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss

from corebehrt.constants.causal.data import EXPOSURE_TARGET
from corebehrt.constants.data import ATTENTION_MASK, CONCEPT_FEAT, PID_COL
from corebehrt.modules.model.causal.heads import MLPHead
from corebehrt.modules.model.causal.loss import FocalLossWithLogits
from corebehrt.modules.model.model import CorebehrtEncoder
from corebehrt.modules.trainer.utils import limit_dict_for_logging, pos_weight_to_alpha

logger = logging.getLogger(__name__)


class CorebehrtForCausalFineTuning(CorebehrtEncoder):
    """
    A simplified causal fine-tuning model using the last token representation.

    This model generates a patient representation by taking the hidden state of the
    last token from the CoreBEHRT encoder output. A bottleneck layer and
    separate MLP heads are then applied to this representation to predict exposure
    and multiple outcomes.
    """

    def __init__(self, config):
        super().__init__(config)

        # --- Architecture Configuration ---
        self.head_config = getattr(config, "head", {})
        self.loss_config = getattr(config, "loss", {})
        self.bottleneck_dim = self.head_config.get("bottleneck_dim", 128)
        self.l1_lambda = self.head_config.get("l1_lambda", 0.0)
        if self.l1_lambda > 0:
            logger.info(f"Applying L1 regularization with lambda={self.l1_lambda}")
        self.temperature = self.head_config.get("temperature", 1.0)

        # Get outcome names from config
        self.outcome_names = config.outcome_names
        # self.exposure_embedding_dim = self.head_config.get("exposure_embedding_dim", 1)
        # self.exposure_embedding = nn.Embedding(2, self.exposure_embedding_dim)

        self._setup_bottleneck_and_heads(config)
        self._setup_loss_functions(config)

        logger.info("Applying custom Kaiming weight initialization to causal heads...")
        self.encoder_bottleneck.apply(self._init_weights)
        self.exposure_head.apply(self._init_weights)
        self.outcome_heads.apply(self._init_weights)

    def _setup_bottleneck_and_heads(self, config):
        """Sets up the bottleneck layer and MLP heads for all tasks."""
        logger.info(
            f"Using last token representation with a bottleneck of dim {self.bottleneck_dim}"
        )

        # A single bottleneck applied to the last token representation
        self.encoder_bottleneck = nn.Sequential(
            nn.Linear(config.hidden_size, self.bottleneck_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # The input to all heads is the output of the bottleneck layer
        head_input_size = self.bottleneck_dim

        # MLP head for exposure prediction
        self.exposure_head = MLPHead(input_size=head_input_size)

        # Create separate MLP heads for each outcome
        self.outcome_heads = nn.ModuleDict()
        for outcome_name in self.outcome_names:
            # The outcome head takes the patient representation + the exposure status embedding
            self.outcome_heads[outcome_name] = MLPHead(
                input_size=head_input_size + 1  # self.exposure_embedding_dim
            )

    def _get_loss_fn(self, loss_name: str, loss_params: dict):
        """Returns the loss function instance based on the name."""
        if loss_name == "focal":
            if (pos_weight := loss_params.get("pos_weight", None)) is not None:
                alpha = pos_weight_to_alpha(pos_weight=pos_weight)
            else:
                alpha = None
            return FocalLossWithLogits(alpha=alpha, gamma=loss_params.get("gamma", 2.0))
        elif loss_name == "bce":
            return BCEWithLogitsLoss(pos_weight=loss_params.get("pos_weight"))
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")

    def _setup_loss_functions(self, config):
        """Helper method to initialize loss functions."""
        # Loss function choice and parameters
        loss_function_name = self.loss_config.get("name", "bce")
        loss_params = self.loss_config.get("params", {})

        # Setup exposure loss
        exposure_loss_params = loss_params.copy()
        exposure_pos_weight = self._get_pos_weight_tensor(
            getattr(config, "pos_weight_exposures", None)
        )
        exposure_loss_params["pos_weight"] = exposure_pos_weight
        if exposure_pos_weight is not None:
            logger.info(
                f"pos_weight_exposures (loss): {round(float(exposure_pos_weight), 3)}"
            )
        self.exposure_loss_fct = self._get_loss_fn(
            loss_function_name, exposure_loss_params
        )
        logger.info(f"Exposure loss function: {self.exposure_loss_fct}")

        # Setup outcome losses
        self.outcome_loss_fcts = nn.ModuleDict()
        pos_weight_outcomes = getattr(config, "pos_weight_outcomes", {})
        pos_weights_for_log = {}

        for outcome_name in self.outcome_names:
            outcome_loss_params = loss_params.copy()
            outcome_pos_weight = self._get_pos_weight_tensor(
                pos_weight_outcomes.get(outcome_name)
            )
            outcome_loss_params["pos_weight"] = outcome_pos_weight
            pos_weights_for_log[outcome_name] = (
                float(outcome_pos_weight) if outcome_pos_weight is not None else 0
            )

            self.outcome_loss_fcts[outcome_name] = self._get_loss_fn(
                loss_function_name, outcome_loss_params
            )
        if loss_function_name == "bce":
            logger.info(
                f"pos_weights_for_log: \n{limit_dict_for_logging(pos_weights_for_log)}"
            )

    def _get_pos_weight_tensor(self, pos_weight_value):
        """Helper method to convert pos_weight value to tensor if not None."""
        if pos_weight_value is not None:
            return torch.tensor(pos_weight_value, device=self.device)
        return None

    def forward(self, batch: dict, cf: bool = False, return_encodings: bool = False):
        """Forward pass for causal inference."""
        outputs = super().forward(batch)
        sequence_output = outputs[0]  # [batch_size, seq_len, hidden_size]
        attention_mask = batch[ATTENTION_MASK]  # [batch_size, seq_len]

        # --- Get Patient Representation from the Last Token ---
        # Get the length of each sequence by summing the attention mask
        sequence_lengths = torch.sum(attention_mask, dim=1)
        # Get the index of the last token for each sample in the batch
        last_token_indices = sequence_lengths - 1

        # Use advanced indexing to select the hidden state of the last token for each sample
        # This creates a tensor of shape [batch_size, hidden_size]
        last_token_repr = sequence_output[
            torch.arange(sequence_output.size(0)), last_token_indices
        ]

        # Apply bottleneck to get the final patient representation
        patient_repr = self.encoder_bottleneck(last_token_repr)
        outputs.bottleneck_repr = patient_repr  # Store for L1 loss calculation

        if return_encodings:
            outputs.patient_encodings = patient_repr
            outputs.token_encodings = sequence_output
            outputs.pids = batch[PID_COL]
            outputs.token_ids = batch[CONCEPT_FEAT]

        # --- Exposure Prediction ---
        exposure_logits = self.exposure_head(patient_repr)
        outputs.exposure_logits = exposure_logits

        # --- Multiple Outcome Predictions ---
        exposure_status = batch[EXPOSURE_TARGET]  # .to(torch.long)

        if cf:
            exposure_status = 2 * (1 - exposure_status) - 1  # Flip for counterfactual

        exposure_embedding = exposure_status  # self.exposure_embedding(exposure_status)

        # Compute logits for each outcome
        outputs.outcome_logits = {}
        for outcome_name in self.outcome_names:
            # Concatenate the patient representation with the exposure status embedding
            outcome_input = torch.cat(
                (patient_repr, exposure_embedding.unsqueeze(1)), dim=-1
            )
            outputs.outcome_logits[outcome_name] = self.outcome_heads[outcome_name](
                outcome_input
            )

        # Only compute losses if we're training or if labels are available for evaluation
        if self.training or self._should_compute_losses(batch):
            self._compute_losses(outputs, batch)

        return outputs

    def _should_compute_losses(self, batch):
        """Check if we should compute losses based on available labels."""
        has_exposure_label = EXPOSURE_TARGET in batch
        has_outcome_labels = all(
            outcome_name in batch for outcome_name in self.outcome_names
        )
        return has_exposure_label and has_outcome_labels

    def _compute_losses(self, outputs, batch):
        """Helper method to compute and assign losses if labels are present."""
        total_loss = 0
        outputs.outcome_losses = {}

        # Exposure loss
        if EXPOSURE_TARGET in batch:
            exposure_loss = self.exposure_loss_fct(
                outputs.exposure_logits.view(-1) / self.temperature,
                batch[EXPOSURE_TARGET].view(-1),
            )
            outputs.exposure_loss = exposure_loss
            total_loss += exposure_loss

        # Outcome losses
        for outcome_name in self.outcome_names:
            if outcome_name in batch:
                predictions = outputs.outcome_logits[outcome_name].view(-1)
                targets = batch[outcome_name].view(-1)
                outcome_loss = self.outcome_loss_fcts[outcome_name](
                    predictions / self.temperature, targets
                )
                outputs.outcome_losses[outcome_name] = outcome_loss
                total_loss += outcome_loss

        # Add L1 regularization on the bottleneck representation
        if self.l1_lambda > 0:
            l1_regularization = torch.norm(outputs.bottleneck_repr, p=1, dim=-1).mean()
            l1_loss = self.l1_lambda * l1_regularization
            outputs.l1_loss = l1_loss
            total_loss += l1_loss

        outputs.loss = total_loss

    @staticmethod
    def _init_weights(module: nn.Module):
        """Initializes weights of Linear layers with Kaiming Normal."""
        if isinstance(module, nn.Linear):
            # Kaiming Normal is a good choice for layers followed by ReLU/GELU
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
