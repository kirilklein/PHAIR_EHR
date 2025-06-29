"""
Unit tests for causal inference models.

This module tests the CorebehrtForCausalFineTuning class to ensure
proper forward pass behavior and loss calculation for both exposure and outcome prediction.
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
            TARGET: torch.randint(
                0, 2, size=(self.batch_size, 1), device=self.device, dtype=torch.float
            ),
            EXPOSURE_TARGET: torch.randint(
                0, 2, size=(self.batch_size, 1), device=self.device, dtype=torch.float
            ),
        }

    def test_initialization(self):
        model = CorebehrtForCausalFineTuning(self.config)
        self.assertIsInstance(model.exposure_loss_fct, nn.BCEWithLogitsLoss)
        self.assertIsInstance(model.outcome_loss_fct, nn.BCEWithLogitsLoss)
        self.assertFalse(model.exposure_cls.pool.with_exposure)
        self.assertTrue(model.outcome_cls.pool.with_exposure)

    def test_forward_with_labels(self):
        model = CorebehrtForCausalFineTuning(self.config).to(self.device)
        # Forward pass
        outputs = model(self.batch)
        self.assertTrue(hasattr(outputs, "exposure_logits"))
        self.assertTrue(hasattr(outputs, "outcome_logits"))
        self.assertTrue(hasattr(outputs, "loss"))
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
        self.assertFalse(hasattr(outputs, "loss"))

    def test_counterfactual_mode(self):
        model = CorebehrtForCausalFineTuning(self.config).to(self.device)
        # Use fixed logits to check inversion
        batch = self.batch.copy()
        batch[TARGET] = torch.ones(self.batch_size, 1, device=self.device)
        batch[EXPOSURE_TARGET] = torch.zeros(self.batch_size, 1, device=self.device)
        model(batch, cf=True)

    def test_loss_functions(self):
        model = CorebehrtForCausalFineTuning(self.config).to(self.device)
        logits = torch.tensor([0.2, -0.5, 0.1, 0.7], device=self.device).reshape(-1, 1)
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0], device=self.device).reshape(-1, 1)
        exposure_loss = model.get_exposure_loss(logits, labels)
        outcome_loss = model.get_outcome_loss(logits, labels)
        self.assertIsInstance(exposure_loss, torch.Tensor)
        self.assertEqual(exposure_loss.shape, torch.Size([]))
        self.assertIsInstance(outcome_loss, torch.Tensor)
        self.assertEqual(outcome_loss.shape, torch.Size([]))
        self.assertEqual(exposure_loss, outcome_loss)


if __name__ == "__main__":
    unittest.main()
