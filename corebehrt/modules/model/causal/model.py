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
from corebehrt.modules.model.causal.heads import (
    MLPHead,
    PatientRepresentationPooler,
    SharedRepresentationPooler,
)
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
        self.head_config = getattr(config, "head", {})
        self.bidirectional = self.head_config.get("bidirectional", True)
        self.outcome_embedding_dim = self.head_config.get("outcome_embedding_dim", 32)
        self.outcome_names = config.outcome_names
        self.num_outcomes = len(self.outcome_names)

        # Create a mapping from outcome name to an integer index
        self.outcome_to_idx = {name: i for i, name in enumerate(self.outcome_names)}

        # ### 💡 NEW: Configuration for the split ###
        # The total size of the pooled representation to be split.
        self.pooled_hidden_size = config.hidden_size * 3
        self.split_sizes = [config.hidden_size, config.hidden_size, config.hidden_size]

        self._setup_pooling_layers(config)
        self._setup_mlp_heads()
        self._setup_loss_functions(config)

        # Orthogonality settings
        self.use_orthogonality = self.head_config.get("use_orthogonality", True)
        self.exposure_lambda = self.head_config.get("exposure_lambda", 1.0)
        self.ortho_lambda = self.head_config.get("ortho_lambda", 1.0)
        self.consistency_lambda = self.head_config.get("consistency_lambda", 1.0)

    def _setup_pooling_layers(self, config):
        # ### MODIFIED: Create an internal disentangling pooler ###
        logger.info(
            f"Using a single pooler with output size {self.pooled_hidden_size} for splitting."
        )
        # This is the main internal module that creates the large representation
        self.disentangling_pooler = PatientRepresentationPooler(
            input_size=config.hidden_size,
            output_size=self.pooled_hidden_size,
            bidirectional=self.bidirectional,
        )

        # This becomes the public-facing pooler for inference and encoding.
        # It wraps the disentangling pooler to return only the shared part.
        self.pooler = SharedRepresentationPooler(
            self.disentangling_pooler, self.split_sizes
        )

    def _setup_mlp_heads(self):
        # Exposure head takes Z_c + Z_e
        self.teacher_exposure_head = MLPHead(
            input_size=self.split_sizes[0] + self.split_sizes[1]
        )

        # 2. Student Head for Exposure (for inference, uses only z_c)
        self.student_exposure_head = MLPHead(
            input_size=self.split_sizes[0]  # Z_c only
        )

        # 1. Embedding layer for the outcomes
        self.outcome_embeddings = nn.Embedding(
            num_embeddings=self.num_outcomes, embedding_dim=self.outcome_embedding_dim
        )

        # 2. Teacher Head for all outcomes (uses full information: z_c + z_y)
        teacher_head_input_size = (
            self.split_sizes[0]
            + self.split_sizes[2]  # Z_c + Z_y
            + self.outcome_embedding_dim  # Outcome ID embedding
            + 1  # Exposure status
        )
        self.teacher_outcome_head = MLPHead(input_size=teacher_head_input_size)

        # 3. Student Head for all outcomes (for inference, uses only shared info: z_c)
        student_head_input_size = (
            self.split_sizes[0]  # Z_c only
            + self.outcome_embedding_dim  # Outcome ID embedding
            + 1  # Exposure status
        )
        self.student_outcome_head = MLPHead(input_size=student_head_input_size)

    def forward(self, batch: dict, cf: bool = False):
        outputs = super().forward(batch)
        sequence_output = outputs[0]
        attention_mask = batch[ATTENTION_MASK]

        # 1. --- Pool into a single large representation using the *internal* pooler ---
        pooled_repr = self.disentangling_pooler(sequence_output, attention_mask)

        # 2. --- Split the representation into three parts ---
        z_c, z_e, z_y = torch.split(pooled_repr, self.split_sizes, dim=-1)

        # EXPLICITLY set the canonical pooler_output to the shared representation
        outputs.pooler_output = z_c
        outputs.representations = {"shared": z_c, "exposure": z_e, "outcome": z_y}

        # 3. --- Exposure Prediction (Student-Teacher) ---
        # Teacher Prediction (uses z_c + z_e)
        teacher_exposure_input = torch.cat((z_c, z_e), dim=-1)
        outputs.teacher_exposure_logits = self.teacher_exposure_head(
            teacher_exposure_input
        )

        # Student Prediction (uses ONLY z_c.detach())
        student_exposure_input = z_c.detach()  # Stop gradients to encoder
        outputs.student_exposure_logits = self.student_exposure_head(
            student_exposure_input
        )

        # Use student logits for evaluation, teacher logits for training's BCE loss
        if self.training:
            outputs.exposure_logits = outputs.teacher_exposure_logits
        else:
            outputs.exposure_logits = outputs.student_exposure_logits

        # 4. --- Outcome Predictions (Student-Teacher) ---
        outputs.teacher_outcome_logits = {}
        outputs.student_outcome_logits = {}
        exposure_status = (2 * batch[EXPOSURE_TARGET] - 1).unsqueeze(-1)
        if cf:
            exposure_status = -exposure_status

        for outcome_name, outcome_idx in self.outcome_to_idx.items():
            # Get the embedding vector for the current outcome
            outcome_idx_tensor = torch.tensor([outcome_idx], device=z_c.device).expand(
                z_c.size(0)
            )
            outcome_emb = self.outcome_embeddings(outcome_idx_tensor)

            # Teacher Prediction (uses z_c + z_y)
            teacher_patient_repr = torch.cat((z_c, z_y), dim=-1)
            teacher_input = torch.cat(
                (teacher_patient_repr, outcome_emb, exposure_status), dim=-1
            )
            teacher_logits = self.teacher_outcome_head(teacher_input)
            outputs.teacher_outcome_logits[outcome_name] = teacher_logits

            # Student Prediction (uses ONLY z_c.detach())
            student_patient_repr = z_c.detach()  # Stop gradients to encoder
            student_input = torch.cat(
                (student_patient_repr, outcome_emb, exposure_status), dim=-1
            )
            student_logits = self.student_outcome_head(student_input)
            outputs.student_outcome_logits[outcome_name] = student_logits

        # Use student logits for evaluation, teacher logits for training's BCE loss
        if self.training:
            outputs.outcome_logits = outputs.teacher_outcome_logits
        else:
            outputs.outcome_logits = outputs.student_outcome_logits

        if self.training or self._should_compute_losses(batch):
            self._compute_losses(outputs, batch)  # Loss computation remains similar

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
        # Add consistency loss for student-teacher learning
        self.consistency_loss_fct = nn.MSELoss()

    def _compute_losses(self, outputs, batch):
        """Helper method to compute and assign losses if labels are present."""
        total_loss = 0
        outputs.outcome_losses = {}
        outputs.consistency_losses = {}

        # Only compute exposure loss if label is available
        if EXPOSURE_TARGET in batch:
            # Get predictions from both heads
            teacher_exposure_predictions = outputs.teacher_exposure_logits.view(-1)
            student_exposure_predictions = outputs.student_exposure_logits.view(-1)
            exposure_targets = batch[EXPOSURE_TARGET].view(-1)

            # a) Main Task Loss (from Teacher)
            # This loss drives the learning of the encoder and disentangling pooler.
            exposure_loss = self.exposure_loss_fct(
                teacher_exposure_predictions, exposure_targets
            )
            outputs.exposure_loss = exposure_loss
            total_loss += (
                self.exposure_lambda * exposure_loss
            )  # Use a lambda for weighting

            # b) Consistency Loss (for Student)
            # This loss only trains the student head, using the teacher as a fixed target.
            teacher_exposure_probs = torch.sigmoid(
                teacher_exposure_predictions.detach()
            )
            exposure_consistency_loss = self.exposure_loss_fct(
                student_exposure_predictions, teacher_exposure_probs
            )
            outputs.exposure_consistency_loss = exposure_consistency_loss
            total_loss += self.consistency_lambda * exposure_consistency_loss

        # Only compute outcome losses for available labels
        for outcome_name in self.outcome_names:
            if outcome_name not in batch:
                continue

            # BCE loss is calculated on the TEACHER's predictions
            student_predictions = outputs.student_outcome_logits[outcome_name].view(-1)
            teacher_predictions = outputs.teacher_outcome_logits[outcome_name].view(-1)

            # The "target" for the student is the teacher's probability distribution.
            # We use .detach() so the teacher is a fixed target.
            teacher_probs_as_target = torch.sigmoid(teacher_predictions.detach())
            targets = batch[outcome_name].view(-1)

            # We can reuse the same BCE loss function and pos_weight. This is crucial.
            # The pos_weight will correctly scale the loss even when comparing two models,
            # adjusting for the class imbalance of the underlying data.
            outcome_loss = self.outcome_loss_fcts[outcome_name](
                teacher_predictions, targets
            )
            outputs.outcome_losses[outcome_name] = outcome_loss
            total_loss += outcome_loss

            # Consistency loss between student and DETACHED teacher
            consistency_loss_fct = self.outcome_loss_fcts[outcome_name]
            consistency_loss = consistency_loss_fct(
                student_predictions, teacher_probs_as_target
            )

            outputs.consistency_losses[outcome_name] = consistency_loss
            total_loss += self.consistency_lambda * consistency_loss

        if self.use_orthogonality:
            ortho_loss = self._compute_orthogonality_loss(outputs.representations)
            outputs.orthogonality_loss = ortho_loss
            total_loss += self.ortho_lambda * ortho_loss

        outputs.loss = total_loss

    def _compute_orthogonality_loss(self, representations: dict) -> torch.Tensor:
        """
        Computes the orthogonality penalty between the shared and specific representations.
        Loss = (z_c * z_e)^2 + (z_c * z_y)^2
        """
        z_c = representations["shared"]
        z_e = representations["exposure"]
        z_y = representations["outcome"]

        # Normalize the representations to prevent the loss from growing with embedding size
        z_c = torch.nn.functional.normalize(z_c, p=2, dim=1)
        z_e = torch.nn.functional.normalize(z_e, p=2, dim=1)
        z_y = torch.nn.functional.normalize(z_y, p=2, dim=1)

        # Dot product between shared and exposure representations
        dot_product_e = torch.sum(z_c * z_e, dim=1)

        # Dot product between shared and outcome representations
        dot_product_y = torch.sum(z_c * z_y, dim=1)

        # Loss is the squared dot product, averaged over the batch
        loss_e = (dot_product_e**2).mean()
        loss_y = (dot_product_y**2).mean()

        return loss_e + loss_y

    def _get_pos_weight_tensor(self, pos_weight_value):
        """Helper method to convert pos_weight value to tensor if not None."""
        if pos_weight_value is not None:
            return torch.tensor(pos_weight_value)
        return None
