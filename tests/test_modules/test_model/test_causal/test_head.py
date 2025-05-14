"""
Unit tests for causal head components.

This module tests the CausalBiGRU and CausalFineTuneHead classes to ensure
proper forward pass behavior and correct handling of exposure information.
"""

import unittest

import torch

from corebehrt.modules.model.causal.heads import (CausalBiGRU,
                                                  CausalFineTuneHead)


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
        hidden_states = torch.randn(self.batch_size, self.seq_len, self.hidden_size, device=self.device)
        
        # Important: Create attention mask with at least one token per sequence
        # This prevents zero-length sequences that would cause issues
        attention_mask = torch.ones(self.batch_size, self.seq_len, device=self.device).bool()
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
        self.assertEqual(model.last_pooled_output.shape, (self.batch_size, self.hidden_size))

    def test_forward_with_exposure_batch_seq(self):
        """Test forward pass with sequence-level exposure information."""
        model = CausalBiGRU(self.hidden_size, with_exposure=True).to(self.device)
        
        # Create dummy inputs
        hidden_states = torch.randn(self.batch_size, self.seq_len, self.hidden_size, device=self.device)
        attention_mask = torch.ones(self.batch_size, self.seq_len, device=self.device).bool()
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
        embeddings = model(hidden_states, attention_mask, exposure_status, return_embedding=True)
        self.assertEqual(embeddings.shape, (self.batch_size, self.hidden_size + 1))
        
        # Verify last_pooled_output includes exposure
        self.assertEqual(model.last_pooled_output.shape, (self.batch_size, self.hidden_size + 1))

    def test_forward_with_exposure_batch_one(self):
        """Test forward pass with scalar exposure information per batch item."""
        model = CausalBiGRU(self.hidden_size, with_exposure=True).to(self.device)
        
        # Create dummy inputs
        hidden_states = torch.randn(self.batch_size, self.seq_len, self.hidden_size, device=self.device)
        attention_mask = torch.ones(self.batch_size, self.seq_len, device=self.device).bool()
        
        # Create batch-level exposure status [batch_size, 1]
        exposure_status = torch.zeros(self.batch_size, 1, device=self.device)
        exposure_status[0] = 1
        exposure_status[3] = 1
        
        # Test with exposure_status
        output = model(hidden_states, attention_mask, exposure_status)
        self.assertEqual(output.shape, (self.batch_size, 1))
        
        # Verify last_pooled_output includes exposure
        self.assertEqual(model.last_pooled_output.shape, (self.batch_size, self.hidden_size + 1))

    def test_variable_sequence_lengths(self):
        """Test handling of variable sequence lengths."""
        model = CausalBiGRU(self.hidden_size, with_exposure=False).to(self.device)
        
        # Create dummy inputs with widely varying sequence lengths
        hidden_states = torch.randn(self.batch_size, self.seq_len, self.hidden_size, device=self.device)
        attention_mask = torch.zeros(self.batch_size, self.seq_len, device=self.device).bool()
        
        # Set varying sequence lengths with minimum length of 1
        for i in range(self.batch_size):
            length = max(1, i + 1)  # Ensure at least length 1
            attention_mask[i, :length] = 1
        
        # Test forward pass
        output = model(hidden_states, attention_mask)
        self.assertEqual(output.shape, (self.batch_size, 1))


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

    def test_forward_without_exposure(self):
        """Test forward pass without exposure information."""
        head = CausalFineTuneHead(self.hidden_size, with_exposure=False).to(self.device)
        
        # Create dummy inputs
        hidden_states = torch.randn(self.batch_size, self.seq_len, self.hidden_size, device=self.device)
        
        # Important: Create attention mask with at least one token per sequence
        attention_mask = torch.ones(self.batch_size, self.seq_len, device=self.device).bool()
        
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
        hidden_states = torch.randn(self.batch_size, self.seq_len, self.hidden_size, device=self.device)
        attention_mask = torch.ones(self.batch_size, self.seq_len, device=self.device).bool()
        
        # Make sure exposure_status has the correct shape and device
        exposure_status = torch.rand(self.batch_size, 1, device=self.device)
        
        # Test standard forward pass
        output = head(hidden_states, attention_mask, exposure_status)
        self.assertEqual(output.shape, (self.batch_size, 1))
        
        # Test with return_embedding=True
        embeddings = head(hidden_states, attention_mask, exposure_status, return_embedding=True)
        self.assertEqual(embeddings.shape, (self.batch_size, self.hidden_size + 1))


if __name__ == "__main__":
    unittest.main()