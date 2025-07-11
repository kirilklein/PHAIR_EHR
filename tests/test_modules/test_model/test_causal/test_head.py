"""
Unit tests for causal head components.

This module tests the PatientRepresentationPooler and MLPHead classes to ensure
proper forward pass behavior and correct pooling/classification functionality.
"""

import unittest

import torch

from corebehrt.modules.model.causal.heads import (
    MLPHead,
    PatientRepresentationPooler,
)


class TestPatientRepresentationPooler(unittest.TestCase):
    """Test cases for the PatientRepresentationPooler component."""

    def setUp(self):
        """Set up common test parameters."""
        self.hidden_size = 64
        self.batch_size = 8
        self.seq_len = 10
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_initialization_bidirectional(self):
        """Test that PatientRepresentationPooler initializes correctly with bidirectional=True."""
        pooler = PatientRepresentationPooler(self.hidden_size, bidirectional=True)
        self.assertEqual(pooler.hidden_size, self.hidden_size)
        self.assertTrue(pooler.bidirectional)
        self.assertEqual(pooler.rnn_hidden_size, self.hidden_size // 2)
        self.assertIsInstance(pooler.rnn, torch.nn.GRU)
        self.assertTrue(pooler.rnn.bidirectional)

    def test_initialization_unidirectional(self):
        """Test that PatientRepresentationPooler initializes correctly with bidirectional=False."""
        pooler = PatientRepresentationPooler(self.hidden_size, bidirectional=False)
        self.assertEqual(pooler.hidden_size, self.hidden_size)
        self.assertFalse(pooler.bidirectional)
        self.assertEqual(pooler.rnn_hidden_size, self.hidden_size)
        self.assertFalse(pooler.rnn.bidirectional)

    def test_forward_bidirectional(self):
        """Test forward pass with bidirectional GRU."""
        pooler = PatientRepresentationPooler(self.hidden_size, bidirectional=True).to(
            self.device
        )

        # Create dummy inputs
        hidden_states = torch.randn(
            self.batch_size, self.seq_len, self.hidden_size, device=self.device
        )
        attention_mask = torch.ones(
            self.batch_size, self.seq_len, device=self.device
        ).bool()

        # Test forward pass
        output = pooler(hidden_states, attention_mask)
        self.assertEqual(output.shape, (self.batch_size, self.hidden_size))

    def test_forward_unidirectional(self):
        """Test forward pass with unidirectional GRU."""
        pooler = PatientRepresentationPooler(self.hidden_size, bidirectional=False).to(
            self.device
        )

        # Create dummy inputs
        hidden_states = torch.randn(
            self.batch_size, self.seq_len, self.hidden_size, device=self.device
        )
        attention_mask = torch.ones(
            self.batch_size, self.seq_len, device=self.device
        ).bool()

        # Test forward pass
        output = pooler(hidden_states, attention_mask)
        self.assertEqual(output.shape, (self.batch_size, self.hidden_size))

    def test_variable_sequence_lengths(self):
        """Test handling of variable sequence lengths."""
        pooler = PatientRepresentationPooler(self.hidden_size, bidirectional=True).to(
            self.device
        )

        # Create dummy inputs with varying sequence lengths
        hidden_states = torch.randn(
            self.batch_size, self.seq_len, self.hidden_size, device=self.device
        )
        attention_mask = torch.zeros(
            self.batch_size, self.seq_len, device=self.device
        ).bool()

        # Set varying sequence lengths with minimum length of 1
        for i in range(self.batch_size):
            length = max(1, i + 1)  # Ensure at least length 1
            if length <= self.seq_len:
                attention_mask[i, :length] = 1
            else:
                attention_mask[i, :] = 1

        # Test forward pass
        output = pooler(hidden_states, attention_mask)
        self.assertEqual(output.shape, (self.batch_size, self.hidden_size))

    def test_edge_case_single_token_sequences(self):
        """Test with sequences that have only one valid token."""
        pooler = PatientRepresentationPooler(self.hidden_size, bidirectional=True).to(
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
        output = pooler(hidden_states, attention_mask)
        self.assertEqual(output.shape, (self.batch_size, self.hidden_size))

    def test_different_hidden_sizes(self):
        """Test with different hidden sizes."""
        for hidden_size in [32, 64, 128, 256]:
            with self.subTest(hidden_size=hidden_size):
                pooler = PatientRepresentationPooler(
                    hidden_size, bidirectional=True
                ).to(self.device)

                hidden_states = torch.randn(
                    self.batch_size, self.seq_len, hidden_size, device=self.device
                )
                attention_mask = torch.ones(
                    self.batch_size, self.seq_len, device=self.device
                ).bool()

                output = pooler(hidden_states, attention_mask)
                self.assertEqual(output.shape, (self.batch_size, hidden_size))


class TestMLPHead(unittest.TestCase):
    """Test cases for the MLPHead component."""

    def setUp(self):
        """Set up common test parameters."""
        self.input_size = 64
        self.batch_size = 8
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_initialization_default(self):
        """Test that MLPHead initializes with default parameters."""
        head = MLPHead(self.input_size)
        self.assertIsInstance(head.norm, torch.nn.LayerNorm)
        self.assertIsInstance(head.classifier, torch.nn.Sequential)
        self.assertEqual(
            len(head.classifier), 5
        )  # Linear, LayerNorm, ReLU, Dropout, Linear

    def test_initialization_custom_params(self):
        """Test that MLPHead initializes with custom parameters."""
        hidden_size_ratio = 4
        dropout_prob = 0.2
        head = MLPHead(
            self.input_size,
            hidden_size_ratio=hidden_size_ratio,
            dropout_prob=dropout_prob,
        )

        # Check that the first linear layer has correct input/output dimensions
        first_linear = head.classifier[0]
        self.assertIsInstance(first_linear, torch.nn.Linear)
        self.assertEqual(first_linear.in_features, self.input_size)
        self.assertEqual(
            first_linear.out_features, self.input_size // hidden_size_ratio
        )

        # Check dropout probability
        dropout_layer = head.classifier[3]
        self.assertIsInstance(dropout_layer, torch.nn.Dropout)
        self.assertEqual(dropout_layer.p, dropout_prob)

    def test_forward_pass(self):
        """Test forward pass of MLPHead."""
        head = MLPHead(self.input_size).to(self.device)

        # Create dummy input
        x = torch.randn(self.batch_size, self.input_size, device=self.device)

        # Test forward pass
        output = head(x)
        self.assertEqual(output.shape, (self.batch_size, 1))

    def test_forward_pass_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        head = MLPHead(self.input_size).to(self.device)

        for batch_size in [1, 4, 16, 32]:
            with self.subTest(batch_size=batch_size):
                x = torch.randn(batch_size, self.input_size, device=self.device)
                output = head(x)
                self.assertEqual(output.shape, (batch_size, 1))

    def test_forward_pass_different_input_sizes(self):
        """Test forward pass with different input sizes."""
        for input_size in [32, 64, 128, 256]:
            with self.subTest(input_size=input_size):
                head = MLPHead(input_size).to(self.device)
                x = torch.randn(self.batch_size, input_size, device=self.device)
                output = head(x)
                self.assertEqual(output.shape, (self.batch_size, 1))

    def test_forward_pass_different_hidden_size_ratios(self):
        """Test forward pass with different hidden size ratios."""
        for ratio in [2, 4, 8]:
            with self.subTest(ratio=ratio):
                head = MLPHead(self.input_size, hidden_size_ratio=ratio).to(self.device)
                x = torch.randn(self.batch_size, self.input_size, device=self.device)
                output = head(x)
                self.assertEqual(output.shape, (self.batch_size, 1))

    def test_gradient_flow(self):
        """Test that gradients flow through the MLPHead."""
        head = MLPHead(self.input_size).to(self.device)
        x = torch.randn(
            self.batch_size, self.input_size, device=self.device, requires_grad=True
        )

        output = head(x)
        loss = output.sum()
        loss.backward()

        # Check that gradients are computed
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.allclose(x.grad, torch.zeros_like(x.grad)))

    def test_output_range(self):
        """Test that output is in reasonable range (not extreme values)."""
        head = MLPHead(self.input_size).to(self.device)
        x = torch.randn(self.batch_size, self.input_size, device=self.device)

        output = head(x)

        # Check that outputs are finite
        self.assertTrue(torch.isfinite(output).all())

        # Check that outputs are not extremely large (arbitrary threshold)
        self.assertTrue(torch.abs(output).max() < 100)


if __name__ == "__main__":
    unittest.main()
