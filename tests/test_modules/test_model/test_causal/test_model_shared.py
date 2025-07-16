# test_causal_model.py
import unittest
import torch

# You need to import your model and its dependencies here
# Make sure the paths are correct based on your project structure
from corebehrt.constants.causal.data import EXPOSURE_TARGET
from corebehrt.modules.model.causal.model import CorebehrtForCausalFineTuning
from transformers import ModernBertConfig


class TestCausalModel(unittest.TestCase):
    def setUp(self):
        """Set up a mock config and a sample batch for testing."""
        # 1. Mock Configuration using a transformers-compatible config
        self.config = ModernBertConfig(
            # Using a smaller hidden_size for faster testing
            hidden_size=96,
            # Custom parameters for our model
            outcome_names=["outcome_a", "outcome_b", "outcome_c"],
            head={
                "use_orthogonality": True,
                "ortho_lambda": 1.0,
                "bidirectional": True,
                "outcome_embedding_dim": 32,
            },
            pos_weight_exposures=1.0,
            pos_weight_outcomes={"outcome_a": 1.0, "outcome_b": 2.0, "outcome_c": 1.5},
            # Add other required ModernBert parameters if needed, e.g., vocab_size
            vocab_size=30000,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=128,
            pad_token_id=0,
            type_vocab_size=2,
        )

        # 2. Instantiate the Model
        self.model = CorebehrtForCausalFineTuning(self.config)
        self.model.train()  # Set to training mode to ensure losses are computed

        # 3. Create a Mock Batch of Data
        self.batch_size = 4
        self.seq_len = 50
        self.batch = {
            "concept": torch.randint(0, 100, (self.batch_size, self.seq_len)),
            "segment": torch.randint(0, 2, (self.batch_size, self.seq_len)),
            "age": torch.rand((self.batch_size, self.seq_len)) * 100,
            "abspos": torch.rand((self.batch_size, self.seq_len)) * 100,
            "attention_mask": torch.ones(
                self.batch_size, self.seq_len, dtype=torch.long
            ),
            EXPOSURE_TARGET: torch.randint(0, 1, (self.batch_size,)).float(),
            "outcome_a": torch.randint(0, 2, (self.batch_size,)).float(),
            "outcome_b": torch.randint(0, 2, (self.batch_size,)).float(),
            "outcome_c": torch.randint(0, 2, (self.batch_size,)).float(),
        }

    def test_forward_pass_and_loss_computation(self):
        """Test that the forward pass runs and produces outputs of the correct shape."""
        # Run the forward pass
        outputs = self.model(self.batch)

        # --- Assertions ---
        # Check for total loss
        self.assertTrue(
            hasattr(outputs, "loss"), "Outputs should have a 'loss' attribute"
        )
        self.assertIsNotNone(outputs.loss, "Total loss should not be None")
        self.assertEqual(
            outputs.loss.shape, torch.Size([]), "Total loss must be a scalar"
        )

        # Check for orthogonality loss
        self.assertTrue(
            hasattr(outputs, "orthogonality_loss"),
            "Outputs should have 'orthogonality_loss'",
        )
        self.assertEqual(
            outputs.orthogonality_loss.shape,
            torch.Size([]),
            "Orthogonality loss must be a scalar",
        )

        # Check exposure logits
        self.assertTrue(
            hasattr(outputs, "exposure_logits"), "Outputs should have 'exposure_logits'"
        )
        self.assertEqual(
            outputs.exposure_logits.shape,
            (self.batch_size, 1),
            "Exposure logits shape is incorrect",
        )

        # Check outcome logits
        self.assertTrue(
            hasattr(outputs, "outcome_logits"), "Outputs should have 'outcome_logits'"
        )
        self.assertEqual(
            len(outputs.outcome_logits),
            len(self.config.outcome_names),
            "Should be one logit entry per outcome",
        )

        for outcome_name in self.config.outcome_names:
            self.assertIn(outcome_name, outputs.outcome_logits)
            self.assertEqual(
                outputs.outcome_logits[outcome_name].shape,
                (self.batch_size, 1),
                f"Logits shape for {outcome_name} is incorrect",
            )

        print(
            "\n✅ Test Passed: Forward pass successful, all output shapes are correct."
        )
        print(f"Total Loss: {outputs.loss.item():.4f}")
        print(f"Orthogonality Loss: {outputs.orthogonality_loss.item():.4f}")
