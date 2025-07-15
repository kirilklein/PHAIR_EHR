"""
Unit tests for causal inference models.

This module tests the CorebehrtForCausalFineTuning class to ensure
proper forward pass behavior and loss calculation for both exposure and multiple outcome predictions.
"""

import unittest

import torch
import torch.nn as nn
from transformers import ModernBertConfig

from corebehrt.constants.causal.data import EXPOSURE_TARGET
from corebehrt.constants.data import (
    ABSPOS_FEAT,
    AGE_FEAT,
    ATTENTION_MASK,
    CONCEPT_FEAT,
    SEGMENT_FEAT,
    TARGET,
)
from corebehrt.modules.model.causal.model import CorebehrtForCausalFineTuning


class TestCorebehrtForCausalFineTuning(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Set all required config attributes
        self.config = ModernBertConfig(
            hidden_size=64,
            num_attention_heads=2,
            num_hidden_layers=2,
            type_vocab_size=2,  # Required by ModernBert
            vocab_size=1000,  # Required by ModernBert
            intermediate_size=128,  # Required by ModernBert
            pad_token_id=0,
        )
        self.config.pos_weight = None
        self.config.outcome_names = ["outcome_1", "outcome_2", "outcome_3"]
        self.outcome_names = self.config.outcome_names
        self.batch_size = 8
        self.seq_len = 10
        self.hidden_size = 64

        self.batch = {
            CONCEPT_FEAT: torch.randint(
                0,
                1000,
                size=(self.batch_size, self.seq_len),
                device=self.device,
                dtype=torch.int,
            ),
            SEGMENT_FEAT: torch.randint(
                0,
                2,
                size=(self.batch_size, self.seq_len),
                device=self.device,
                dtype=torch.int,
            ),
            AGE_FEAT: torch.randint(
                0,
                100,
                size=(self.batch_size, self.seq_len),
                device=self.device,
                dtype=torch.int,
            ),
            ABSPOS_FEAT: torch.randint(
                0,
                100,
                size=(self.batch_size, self.seq_len),
                device=self.device,
                dtype=torch.int,
            ),
            ATTENTION_MASK: torch.ones(
                self.batch_size, self.seq_len, device=self.device
            ).bool(),
            EXPOSURE_TARGET: torch.randint(
                0, 2, size=(self.batch_size,), device=self.device, dtype=torch.float
            ),
        }

        # Add multiple outcome targets to the batch
        for outcome_name in self.config.outcome_names:
            self.batch[outcome_name] = torch.randint(
                0, 2, size=(self.batch_size,), device=self.device, dtype=torch.float
            )

    def test_initialization_shared_representation(self):
        """Test initialization with shared representation (default)."""
        model = CorebehrtForCausalFineTuning(self.config)
        self.assertIsInstance(model.exposure_loss_fct, nn.BCEWithLogitsLoss)

        # Check that outcome loss functions are created for each outcome
        self.assertIsInstance(model.outcome_loss_fcts, nn.ModuleDict)
        for outcome_name in self.outcome_names:
            self.assertIn(outcome_name, model.outcome_loss_fcts)
            self.assertIsInstance(
                model.outcome_loss_fcts[outcome_name], nn.BCEWithLogitsLoss
            )

        self.assertTrue(model.shared_representation)
        self.assertTrue(hasattr(model, "pooler"))
        self.assertTrue(hasattr(model, "exposure_head"))

        # Check that outcome heads are created for each outcome
        self.assertIsInstance(model.outcome_heads, nn.ModuleDict)
        for outcome_name in self.outcome_names:
            self.assertIn(outcome_name, model.outcome_heads)

    def test_initialization_separate_representation(self):
        """Test initialization with separate representations."""
        self.config.head = {"shared_representation": False}
        model = CorebehrtForCausalFineTuning(self.config)
        self.assertIsInstance(model.exposure_loss_fct, nn.BCEWithLogitsLoss)

        # Check that outcome loss functions are created for each outcome
        self.assertIsInstance(model.outcome_loss_fcts, nn.ModuleDict)
        for outcome_name in self.config.outcome_names:
            self.assertIn(outcome_name, model.outcome_loss_fcts)

        self.assertFalse(model.shared_representation)
        self.assertTrue(hasattr(model, "exposure_pooler"))

        # Check that separate outcome poolers are created
        self.assertIsInstance(model.outcome_poolers, nn.ModuleDict)
        for outcome_name in self.config.outcome_names:
            self.assertIn(outcome_name, model.outcome_poolers)

        self.assertTrue(hasattr(model, "exposure_head"))
        self.assertIsInstance(model.outcome_heads, nn.ModuleDict)

    def test_forward_with_labels(self):
        model = CorebehrtForCausalFineTuning(self.config).to(self.device)
        # Forward pass
        outputs = model(self.batch)
        self.assertTrue(hasattr(outputs, "exposure_logits"))
        self.assertTrue(hasattr(outputs, "outcome_logits"))
        self.assertTrue(hasattr(outputs, "loss"))
        self.assertTrue(hasattr(outputs, "exposure_loss"))
        self.assertTrue(hasattr(outputs, "outcome_losses"))

        # Check that outcome_logits is a dictionary with correct keys
        self.assertIsInstance(outputs.outcome_logits, dict)
        for outcome_name in self.config.outcome_names:
            self.assertIn(outcome_name, outputs.outcome_logits)
            self.assertIsInstance(outputs.outcome_logits[outcome_name], torch.Tensor)

        # Check that outcome_losses is a dictionary with correct keys
        self.assertIsInstance(outputs.outcome_losses, dict)
        for outcome_name in self.config.outcome_names:
            self.assertIn(outcome_name, outputs.outcome_losses)
            self.assertIsInstance(outputs.outcome_losses[outcome_name], torch.Tensor)

        self.assertIsInstance(outputs.loss, torch.Tensor)

    def test_forward_without_labels(self):
        model = CorebehrtForCausalFineTuning(self.config).to(self.device)
        batch_no_labels = {
            ATTENTION_MASK: torch.ones(
                self.batch_size, self.seq_len, device=self.device
            ).bool(),
            CONCEPT_FEAT: self.batch[CONCEPT_FEAT],
            SEGMENT_FEAT: self.batch[SEGMENT_FEAT],
            AGE_FEAT: self.batch[AGE_FEAT],
            ABSPOS_FEAT: self.batch[ABSPOS_FEAT],
            EXPOSURE_TARGET: self.batch[EXPOSURE_TARGET],
        }
        outputs = model(batch_no_labels)
        self.assertTrue(hasattr(outputs, "exposure_logits"))
        self.assertTrue(hasattr(outputs, "outcome_logits"))

        # Check that outcome_logits is a dictionary with correct keys
        self.assertIsInstance(outputs.outcome_logits, dict)
        for outcome_name in self.config.outcome_names:
            self.assertIn(outcome_name, outputs.outcome_logits)

    def test_counterfactual_mode(self):
        model = CorebehrtForCausalFineTuning(self.config).to(self.device)
        # Use fixed logits to check inversion
        batch = self.batch.copy()
        batch[self.outcome_names[0]] = torch.ones(
            self.batch_size, 1, device=self.device
        )
        batch[EXPOSURE_TARGET] = torch.zeros(self.batch_size, device=self.device)
        outputs = model(batch, cf=True)
        self.assertTrue(hasattr(outputs, "exposure_logits"))
        self.assertTrue(hasattr(outputs, "outcome_logits"))

        # Check that outcome_logits is a dictionary with correct keys
        self.assertIsInstance(outputs.outcome_logits, dict)
        for outcome_name in self.config.outcome_names:
            self.assertIn(outcome_name, outputs.outcome_logits)

    def test_separate_representations_forward(self):
        """Test forward pass with separate representations."""
        self.config.head = {"shared_representation": False}
        model = CorebehrtForCausalFineTuning(self.config).to(self.device)
        outputs = model(self.batch)
        self.assertTrue(hasattr(outputs, "exposure_logits"))
        self.assertTrue(hasattr(outputs, "outcome_logits"))
        self.assertTrue(hasattr(outputs, "loss"))

        # Check that outcome_logits is a dictionary with correct keys
        self.assertIsInstance(outputs.outcome_logits, dict)
        for outcome_name in self.config.outcome_names:
            self.assertIn(outcome_name, outputs.outcome_logits)


if __name__ == "__main__":
    unittest.main()
