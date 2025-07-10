"""
Unit tests for causal head components.

This module tests the CausalBiGRU, CausalLastTokenPool, and CausalFineTuneHead classes to ensure
proper forward pass behavior and correct handling of exposure information.
"""

import unittest

import torch

from corebehrt.modules.model.causal.heads import (
    CausalBiGRU,
    CausalFineTuneHead,
    CausalLastTokenPool,
)


class TestCausalLastTokenPool(unittest.TestCase):
    """Test cases for the CausalLastTokenPool component."""

    def setUp(self):
        """Set up common test parameters."""
        self.hidden_size = 64
        self.batch_size = 8
        self.seq_len = 10
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_initialization(self):
        """Test that CausalLastTokenPool initializes with correct parameters."""
        # Test without exposure
        model = CausalLastTokenPool(self.hidden_size, with_exposure=False)
        self.assertEqual(model.hidden_size, self.hidden_size)
        self.assertEqual(model.classifier_input_size, self.hidden_size)
        self.assertFalse(model.with_exposure)

        # Test with exposure
        model = CausalLastTokenPool(self.hidden_size, with_exposure=True)
        self.assertEqual(model.classifier_input_size, self.hidden_size + 1)
        self.assertTrue(model.with_exposure)

        # Check classifier architecture
        self.assertIsInstance(model.classifier, torch.nn.Sequential)
        self.assertEqual(
            len(model.classifier), 5
        )  # Linear, LayerNorm, ReLU, Dropout, Linear

    def test_forward_without_exposure(self):
        """Test forward pass without exposure information."""
        model = CausalLastTokenPool(self.hidden_size, with_exposure=False).to(
            self.device
        )

        # Create dummy inputs
        hidden_states = torch.randn(
            self.batch_size, self.seq_len, self.hidden_size, device=self.device
        )

        # Create attention mask with at least one token per sequence
        attention_mask = torch.ones(
            self.batch_size, self.seq_len, device=self.device
        ).bool()
        # Make some sequences shorter
        attention_mask[0, 5:] = 0
        attention_mask[1, 8:] = 0

        # Test standard forward pass
        output = model(hidden_states, attention_mask)
        self.assertEqual(output.shape, (self.batch_size, 1))

        # Test with return_embedding=True
        embeddings = model(hidden_states, attention_mask, return_embedding=True)
        self.assertEqual(embeddings.shape, (self.batch_size, self.hidden_size))

        # Verify last_pooled_output was stored
        self.assertIsNotNone(model.last_pooled_output)
        self.assertEqual(
            model.last_pooled_output.shape, (self.batch_size, self.hidden_size)
        )

    def test_forward_with_exposure_batch_seq(self):
        """Test forward pass with sequence-level exposure information."""
        model = CausalLastTokenPool(self.hidden_size, with_exposure=True).to(
            self.device
        )

        # Create dummy inputs
        hidden_states = torch.randn(
            self.batch_size, self.seq_len, self.hidden_size, device=self.device
        )
        attention_mask = torch.ones(
            self.batch_size, self.seq_len, device=self.device
        ).bool()
        # Vary sequence lengths
        attention_mask[0, 5:] = 0
        attention_mask[2, 7:] = 0

        # Create sequence-level exposure status [batch_size, seq_len]
        exposure_status = torch.zeros(self.batch_size, self.seq_len, device=self.device)
        exposure_status[:, 0] = 1  # Set first token exposure to 1

        # Test with exposure_status
        output = model(hidden_states, attention_mask, exposure_status)
        self.assertEqual(output.shape, (self.batch_size, 1))

        # Test with return_embedding=True
        embeddings = model(
            hidden_states, attention_mask, exposure_status, return_embedding=True
        )
        self.assertEqual(embeddings.shape, (self.batch_size, self.hidden_size + 1))

        # Verify last_pooled_output includes exposure
        self.assertEqual(
            model.last_pooled_output.shape, (self.batch_size, self.hidden_size + 1)
        )

    def test_forward_with_exposure_batch_one(self):
        """Test forward pass with scalar exposure information per batch item."""
        model = CausalLastTokenPool(self.hidden_size, with_exposure=True).to(
            self.device
        )

        # Create dummy inputs
        hidden_states = torch.randn(
            self.batch_size, self.seq_len, self.hidden_size, device=self.device
        )
        attention_mask = torch.ones(
            self.batch_size, self.seq_len, device=self.device
        ).bool()

        # Create batch-level exposure status [batch_size, 1]
        exposure_status = torch.zeros(self.batch_size, 1, device=self.device)
        exposure_status[0] = 1
        exposure_status[3] = 1

        # Test with exposure_status
        output = model(hidden_states, attention_mask, exposure_status)
        self.assertEqual(output.shape, (self.batch_size, 1))

        # Verify last_pooled_output includes exposure
        self.assertEqual(
            model.last_pooled_output.shape, (self.batch_size, self.hidden_size + 1)
        )

    def test_forward_with_exposure_1d(self):
        """Test forward pass with 1D exposure information."""
        model = CausalLastTokenPool(self.hidden_size, with_exposure=True).to(
            self.device
        )

        # Create dummy inputs
        hidden_states = torch.randn(
            self.batch_size, self.seq_len, self.hidden_size, device=self.device
        )
        attention_mask = torch.ones(
            self.batch_size, self.seq_len, device=self.device
        ).bool()

        # Create 1D exposure status [batch_size]
        exposure_status = torch.zeros(self.batch_size, device=self.device)
        exposure_status[0] = 1
        exposure_status[3] = 1

        # Test with exposure_status
        output = model(hidden_states, attention_mask, exposure_status)
        self.assertEqual(output.shape, (self.batch_size, 1))

        # Verify last_pooled_output includes exposure
        self.assertEqual(
            model.last_pooled_output.shape, (self.batch_size, self.hidden_size + 1)
        )

    def test_variable_sequence_lengths(self):
        """Test handling of variable sequence lengths."""
        model = CausalLastTokenPool(self.hidden_size, with_exposure=False).to(
            self.device
        )

        # Create dummy inputs with widely varying sequence lengths
        hidden_states = torch.randn(
            self.batch_size, self.seq_len, self.hidden_size, device=self.device
        )
        attention_mask = torch.zeros(
            self.batch_size, self.seq_len, device=self.device
        ).bool()

        # Set varying sequence lengths with minimum length of 1
        for i in range(self.batch_size):
            length = max(1, i + 1)  # Ensure at least length 1
            attention_mask[i, :length] = 1

        # Test forward pass
        output = model(hidden_states, attention_mask)
        self.assertEqual(output.shape, (self.batch_size, 1))

        # Verify each sequence uses its actual last token
        lengths = attention_mask.sum(dim=1)
        for i in range(self.batch_size):
            min(lengths[i] - 1, self.seq_len - 1)
            # This is an indirect test - we can't easily verify the exact token used
            # but we can check that the forward pass completes successfully

    def test_edge_case_single_token_sequences(self):
        """Test with sequences that have only one valid token."""
        model = CausalLastTokenPool(self.hidden_size, with_exposure=False).to(
            self.device
        )

        # Create inputs where all sequences have only one valid token
        hidden_states = torch.randn(
            self.batch_size, self.seq_len, self.hidden_size, device=self.device
        )
        attention_mask = torch.zeros(
            self.batch_size, self.seq_len, device=self.device
        ).bool()
        attention_mask[:, 0] = 1  # Only first token is valid

        # Test forward pass
        output = model(hidden_states, attention_mask)
        self.assertEqual(output.shape, (self.batch_size, 1))

    def test_exposure_status_none_with_exposure_false(self):
        """Test that None exposure_status works correctly when with_exposure=False."""
        model = CausalLastTokenPool(self.hidden_size, with_exposure=False).to(
            self.device
        )

        hidden_states = torch.randn(
            self.batch_size, self.seq_len, self.hidden_size, device=self.device
        )
        attention_mask = torch.ones(
            self.batch_size, self.seq_len, device=self.device
        ).bool()

        # Test with None exposure_status (should work fine)
        output = model(hidden_states, attention_mask, exposure_status=None)
        self.assertEqual(output.shape, (self.batch_size, 1))

    def test_exposure_status_none_with_exposure_true(self):
        """Test that ValueError is raised when exposure_status is None but with_exposure=True."""
        model = CausalLastTokenPool(self.hidden_size, with_exposure=True).to(
            self.device
        )

        hidden_states = torch.randn(
            self.batch_size, self.seq_len, self.hidden_size, device=self.device
        )
        attention_mask = torch.ones(
            self.batch_size, self.seq_len, device=self.device
        ).bool()

        # Test that ValueError is raised when exposure_status=None but with_exposure=True
        with self.assertRaises(ValueError) as context:
            model(hidden_states, attention_mask, exposure_status=None)

        self.assertIn(
            "exposure_status cannot be None when with_exposure=True",
            str(context.exception),
        )


class TestCausalBiGRU(unittest.TestCase):
    """Test cases for the CausalBiGRU component."""

    def setUp(self):
        """Set up common test parameters."""
        self.hidden_size = 64
        self.batch_size = 8
        self.seq_len = 10
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_initialization(self):
        """Test that CausalBiGRU initializes with correct parameters."""
        # Test without exposure
        model = CausalBiGRU(self.hidden_size, with_exposure=False)
        self.assertEqual(model.hidden_size, self.hidden_size)
        self.assertEqual(model.rnn_hidden_size, self.hidden_size // 2)
        self.assertEqual(model.classifier_input_size, self.hidden_size)
        self.assertFalse(model.with_exposure)

        # Test with exposure
        model = CausalBiGRU(self.hidden_size, with_exposure=True)
        self.assertEqual(model.classifier_input_size, self.hidden_size + 1)
        self.assertTrue(model.with_exposure)

    def test_forward_without_exposure(self):
        """Test forward pass without exposure information."""
        model = CausalBiGRU(self.hidden_size, with_exposure=False).to(self.device)

        # Create dummy inputs
        hidden_states = torch.randn(
            self.batch_size, self.seq_len, self.hidden_size, device=self.device
        )

        # Important: Create attention mask with at least one token per sequence
        # This prevents zero-length sequences that would cause issues
        attention_mask = torch.ones(
            self.batch_size, self.seq_len, device=self.device
        ).bool()
        # Make some sequences shorter BUT ensure at least one token is present
        attention_mask[0, 5:] = 0
        attention_mask[1, 8:] = 0

        # Test standard forward pass
        output = model(hidden_states, attention_mask)
        self.assertEqual(output.shape, (self.batch_size, 1))

        # Test with return_embedding=True
        embeddings = model(hidden_states, attention_mask, return_embedding=True)
        self.assertEqual(embeddings.shape, (self.batch_size, self.hidden_size))

        # Verify last_pooled_output was stored
        self.assertIsNotNone(model.last_pooled_output)
        self.assertEqual(
            model.last_pooled_output.shape, (self.batch_size, self.hidden_size)
        )

    def test_forward_with_exposure_batch_seq(self):
        """Test forward pass with sequence-level exposure information."""
        model = CausalBiGRU(self.hidden_size, with_exposure=True).to(self.device)

        # Create dummy inputs
        hidden_states = torch.randn(
            self.batch_size, self.seq_len, self.hidden_size, device=self.device
        )
        attention_mask = torch.ones(
            self.batch_size, self.seq_len, device=self.device
        ).bool()
        # Vary sequence lengths but keep at least one token
        attention_mask[0, 5:] = 0
        attention_mask[2, 7:] = 0

        # Create sequence-level exposure status [batch_size, seq_len]
        exposure_status = torch.zeros(self.batch_size, self.seq_len, device=self.device)
        exposure_status[:, 0] = 1  # Set first token exposure to 1

        # Test with exposure_status
        output = model(hidden_states, attention_mask, exposure_status)
        self.assertEqual(output.shape, (self.batch_size, 1))

        # Test with return_embedding=True
        embeddings = model(
            hidden_states, attention_mask, exposure_status, return_embedding=True
        )
        self.assertEqual(embeddings.shape, (self.batch_size, self.hidden_size + 1))

        # Verify last_pooled_output includes exposure
        self.assertEqual(
            model.last_pooled_output.shape, (self.batch_size, self.hidden_size + 1)
        )

    def test_forward_with_exposure_batch_one(self):
        """Test forward pass with scalar exposure information per batch item."""
        model = CausalBiGRU(self.hidden_size, with_exposure=True).to(self.device)

        # Create dummy inputs
        hidden_states = torch.randn(
            self.batch_size, self.seq_len, self.hidden_size, device=self.device
        )
        attention_mask = torch.ones(
            self.batch_size, self.seq_len, device=self.device
        ).bool()

        # Create batch-level exposure status [batch_size, 1]
        exposure_status = torch.zeros(self.batch_size, 1, device=self.device)
        exposure_status[0] = 1
        exposure_status[3] = 1

        # Test with exposure_status
        output = model(hidden_states, attention_mask, exposure_status)
        self.assertEqual(output.shape, (self.batch_size, 1))

        # Verify last_pooled_output includes exposure
        self.assertEqual(
            model.last_pooled_output.shape, (self.batch_size, self.hidden_size + 1)
        )

    def test_variable_sequence_lengths(self):
        """Test handling of variable sequence lengths."""
        model = CausalBiGRU(self.hidden_size, with_exposure=False).to(self.device)

        # Create dummy inputs with widely varying sequence lengths
        hidden_states = torch.randn(
            self.batch_size, self.seq_len, self.hidden_size, device=self.device
        )
        attention_mask = torch.zeros(
            self.batch_size, self.seq_len, device=self.device
        ).bool()

        # Set varying sequence lengths with minimum length of 1
        for i in range(self.batch_size):
            length = max(1, i + 1)  # Ensure at least length 1
            attention_mask[i, :length] = 1

        # Test forward pass
        output = model(hidden_states, attention_mask)
        self.assertEqual(output.shape, (self.batch_size, 1))

    def test_exposure_status_none_with_exposure_true_bigru(self):
        """Test that ValueError is raised when exposure_status is None but with_exposure=True for BiGRU."""
        model = CausalBiGRU(self.hidden_size, with_exposure=True).to(self.device)

        hidden_states = torch.randn(
            self.batch_size, self.seq_len, self.hidden_size, device=self.device
        )
        attention_mask = torch.ones(
            self.batch_size, self.seq_len, device=self.device
        ).bool()

        # Test that ValueError is raised when exposure_status=None but with_exposure=True
        with self.assertRaises(ValueError) as context:
            model(hidden_states, attention_mask, exposure_status=None)

        self.assertIn(
            "exposure_status cannot be None when with_exposure=True",
            str(context.exception),
        )


class TestCausalFineTuneHead(unittest.TestCase):
    """Test cases for the CausalFineTuneHead component."""

    def setUp(self):
        """Set up common test parameters."""
        self.hidden_size = 64
        self.batch_size = 8
        self.seq_len = 10
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_initialization(self):
        """Test initialization of CausalFineTuneHead."""
        # Test without exposure
        head = CausalFineTuneHead(self.hidden_size, with_exposure=False)
        self.assertFalse(head.pool.with_exposure)

        # Test with exposure
        head = CausalFineTuneHead(self.hidden_size, with_exposure=True)
        self.assertTrue(head.pool.with_exposure)

    def test_initialization_pooling_types(self):
        """Test initialization with different pooling types."""
        # Test BiGRU pooling (default)
        head_bigru = CausalFineTuneHead(self.hidden_size, pooling_type="bigru")
        self.assertIsInstance(head_bigru.pool, CausalBiGRU)

        # Test last token pooling
        head_last_token = CausalFineTuneHead(
            self.hidden_size, pooling_type="last_token"
        )
        self.assertIsInstance(head_last_token.pool, CausalLastTokenPool)

        # Test invalid pooling type
        with self.assertRaises(ValueError):
            CausalFineTuneHead(self.hidden_size, pooling_type="invalid")

    def test_forward_without_exposure(self):
        """Test forward pass without exposure information."""
        head = CausalFineTuneHead(self.hidden_size, with_exposure=False).to(self.device)

        # Create dummy inputs
        hidden_states = torch.randn(
            self.batch_size, self.seq_len, self.hidden_size, device=self.device
        )

        # Important: Create attention mask with at least one token per sequence
        attention_mask = torch.ones(
            self.batch_size, self.seq_len, device=self.device
        ).bool()

        # Test standard forward pass with proper conversion of indices
        output = head(hidden_states, attention_mask)
        self.assertEqual(output.shape, (self.batch_size, 1))

        # Test with return_embedding=True
        embeddings = head(hidden_states, attention_mask, return_embedding=True)
        self.assertEqual(embeddings.shape, (self.batch_size, self.hidden_size))

    def test_forward_with_exposure(self):
        """Test forward pass with exposure information."""
        head = CausalFineTuneHead(self.hidden_size, with_exposure=True).to(self.device)

        # Create dummy inputs
        hidden_states = torch.randn(
            self.batch_size, self.seq_len, self.hidden_size, device=self.device
        )
        attention_mask = torch.ones(
            self.batch_size, self.seq_len, device=self.device
        ).bool()

        # Make sure exposure_status has the correct shape and device
        exposure_status = torch.rand(self.batch_size, 1, device=self.device)

        # Test standard forward pass
        output = head(hidden_states, attention_mask, exposure_status)
        self.assertEqual(output.shape, (self.batch_size, 1))

        # Test with return_embedding=True
        embeddings = head(
            hidden_states, attention_mask, exposure_status, return_embedding=True
        )
        self.assertEqual(embeddings.shape, (self.batch_size, self.hidden_size + 1))

    def test_forward_last_token_pooling(self):
        """Test forward pass with last token pooling."""
        head = CausalFineTuneHead(
            self.hidden_size, pooling_type="last_token", with_exposure=False
        ).to(self.device)

        # Create dummy inputs
        hidden_states = torch.randn(
            self.batch_size, self.seq_len, self.hidden_size, device=self.device
        )
        attention_mask = torch.ones(
            self.batch_size, self.seq_len, device=self.device
        ).bool()

        # Test standard forward pass
        output = head(hidden_states, attention_mask)
        self.assertEqual(output.shape, (self.batch_size, 1))

        # Test with return_embedding=True
        embeddings = head(hidden_states, attention_mask, return_embedding=True)
        self.assertEqual(embeddings.shape, (self.batch_size, self.hidden_size))

    def test_exposure_status_none_with_exposure_true_finetune_head(self):
        """Test that ValueError is raised when exposure_status is None but with_exposure=True for FineTuneHead."""
        head = CausalFineTuneHead(self.hidden_size, with_exposure=True).to(self.device)

        hidden_states = torch.randn(
            self.batch_size, self.seq_len, self.hidden_size, device=self.device
        )
        attention_mask = torch.ones(
            self.batch_size, self.seq_len, device=self.device
        ).bool()

        # Test that ValueError is raised when exposure_status=None but with_exposure=True
        with self.assertRaises(ValueError) as context:
            head(hidden_states, attention_mask, exposure_status=None)

        self.assertIn(
            "exposure_status cannot be None when with_exposure=True",
            str(context.exception),
        )


if __name__ == "__main__":
    unittest.main()
