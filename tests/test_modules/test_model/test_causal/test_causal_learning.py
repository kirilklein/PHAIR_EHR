import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from transformers.modeling_outputs import BaseModelOutput

# Assume your model and its dependencies are in these locations
# You MUST adjust these import paths to match your project structure
from corebehrt.constants.causal.data import EXPOSURE_TARGET
from corebehrt.modules.model.causal.model import CorebehrtForCausalFineTuning


# To make the test self-contained, we create a mock of the base BEHRT model.
# This avoids needing the full transformer backbone and makes the test faster.
class MockCorebehrtForFineTuning(nn.Module):
    """A mock base model that returns a fake sequence output."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        # A simple linear layer to project input IDs to the hidden size
        self.embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )

    def forward(self, batch):
        # We only need the 'concept' key for this mock, which maps to input_ids
        input_ids = batch["concept"]
        sequence_output = self.embedding(input_ids)
        return BaseModelOutput(last_hidden_state=sequence_output)


# We dynamically replace the base class of your model with our mock for the test
CorebehrtForCausalFineTuning.__bases__ = (MockCorebehrtForFineTuning,)


def generate_causal_data(num_samples: int, seq_len: int, vocab_size: int) -> dict:
    """
    Generates a batch of synthetic data with a predefined causal structure.
    - Concept 10: Confounder (affects exposure and outcome_a)
    - Concept 20: Exposure-only feature
    - Concept 30: Outcome-only feature (for outcome_a)
    """
    # 1. Define latent features that drive the causal relationships
    # These are "hidden" patient properties we are trying to model
    latent_confounder = torch.randn(num_samples, 1)
    latent_exposure_feature = torch.randn(num_samples, 1)
    latent_outcome_feature = torch.randn(num_samples, 1)

    # 2. Define how these latent features determine exposure and outcomes
    # The confounder influences both exposure and outcome_a
    prob_exposure = torch.sigmoid(
        0.8 * latent_confounder + 1.2 * latent_exposure_feature - 0.5
    )
    prob_outcome_a = torch.sigmoid(
        0.9 * latent_confounder + 1.1 * latent_outcome_feature - 0.4
    )

    exposure_target = torch.bernoulli(prob_exposure).squeeze()
    outcome_a_target = torch.bernoulli(prob_outcome_a).squeeze()

    # Other outcomes can be random noise
    outcome_b_target = torch.randint(0, 2, (num_samples,)).float()
    outcome_c_target = torch.randint(0, 2, (num_samples,)).float()

    # 3. Create the patient sequences based on the latent features
    # Start with sequences full of padding tokens
    concepts = torch.zeros(num_samples, seq_len, dtype=torch.long)

    # Place special concept IDs in the sequence to represent the latent features
    # Samples with high confounder value get concept 10
    concepts[latent_confounder.squeeze() > 0.5, 5] = 10
    # Samples with high exposure-only feature value get concept 20
    concepts[latent_exposure_feature.squeeze() > 0.5, 10] = 20
    # Samples with high outcome-only feature value get concept 30
    concepts[latent_outcome_feature.squeeze() > 0.5, 15] = 30

    # Fill the rest with random concepts to add noise
    concepts[concepts == 0] = torch.randint(
        40, vocab_size, (int((concepts == 0).sum()),)
    )

    return {
        "concept": concepts,
        "attention_mask": torch.ones(num_samples, seq_len, dtype=torch.long),
        EXPOSURE_TARGET: exposure_target,
        "outcome_a": outcome_a_target,
        "outcome_b": outcome_b_target,
        "outcome_c": outcome_c_target,
    }


class TestCausalModelLearning(unittest.TestCase):
    def setUp(self):
        """Set up the configuration and model."""
        self.vocab_size = 100
        self.config = SimpleNamespace(
            hidden_size=96,
            vocab_size=self.vocab_size,
            pad_token_id=0,
            outcome_names=["outcome_a", "outcome_b", "outcome_c"],
            head={
                "use_orthogonality": True,
                "ortho_lambda": 1.0,
                "bidirectional": True,
            },
            pos_weight_exposures=1.0,
            pos_weight_outcomes={"outcome_a": 1.0, "outcome_b": 1.0, "outcome_c": 1.0},
        )
        self.model = CorebehrtForCausalFineTuning(self.config)

    def test_model_learns_orthogonality_on_causal_data(self):
        """
        A full test that generates causal data, trains the model for a few steps,
        and verifies that the orthogonality loss decreases and representations become orthogonal.
        """
        # 1. Generate a larger batch of causally structured data
        batch_size = 256
        seq_len = 50
        batch = generate_causal_data(batch_size, seq_len, self.vocab_size)

        # 2. Set up an optimizer for our mini-training loop
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

        print("\n--- Starting Training Loop ---")
        initial_ortho_loss = -1

        # 3. Mini-Training Loop
        for epoch in range(50):
            self.model.train()
            optimizer.zero_grad()

            outputs = self.model(batch)
            loss = outputs.loss

            self.assertIsNotNone(loss)
            loss.backward()
            optimizer.step()

            ortho_loss_val = outputs.orthogonality_loss.item()
            if epoch == 0:
                initial_ortho_loss = ortho_loss_val

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1:2d} | Total Loss: {loss.item():.4f} | Orthogonality Loss: {ortho_loss_val:.4f}"
                )

        print("--- Training Finished ---")

        # 4. Final Verification
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(batch)
            final_ortho_loss = outputs.orthogonality_loss.item()

            # Check that the orthogonality loss has decreased significantly
            self.assertLess(
                final_ortho_loss,
                initial_ortho_loss / 2,
                "Orthogonality loss should decrease after training.",
            )

            # Extract final representations
            z_c = F.normalize(outputs.representations["shared"])
            z_e = F.normalize(outputs.representations["exposure"])
            z_y = F.normalize(outputs.representations["outcome"])

            # Calculate final cosine similarity (should be near zero for orthogonality)
            cos_sim_ce = torch.mean(torch.sum(z_c * z_e, dim=1))
            cos_sim_cy = torch.mean(torch.sum(z_c * z_y, dim=1))

            print(f"\nInitial Ortho Loss: {initial_ortho_loss:.4f}")
            print(f"Final Ortho Loss  : {final_ortho_loss:.4f}")
            print(
                f"Final Cosine Similarity (Shared <-> Exposure): {cos_sim_ce.item():.4f}"
            )
            print(
                f"Final Cosine Similarity (Shared <-> Outcome) : {cos_sim_cy.item():.4f}"
            )

            # Assert that the representations are now nearly orthogonal
            self.assertAlmostEqual(
                cos_sim_ce.item(),
                0,
                places=2,
                msg="Shared and exposure reps should be orthogonal.",
            )
            self.assertAlmostEqual(
                cos_sim_cy.item(),
                0,
                places=2,
                msg="Shared and outcome reps should be orthogonal.",
            )

            print(
                "\n✅ Test Passed: Model successfully learned to create orthogonal representations on causal data."
            )


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
