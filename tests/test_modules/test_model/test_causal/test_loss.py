import unittest
import torch
import torch.nn.functional as F

from corebehrt.modules.model.causal.loss import FocalLossWithLogits


class TestFocalLossWithLogits(unittest.TestCase):
    """Unit tests for the FocalLossWithLogits class."""

    def setUp(self):
        """Set up basic tensors for testing."""
        self.logits = torch.randn(10, 1)
        self.targets = torch.randint(0, 2, (10, 1)).float()

    def test_prob_recovery_from_bce_loss(self):
        """
        Tests if p_t = exp(-bce_loss) correctly recovers the probability of the true class.

        This confirms that the underlying use of `binary_cross_entropy_with_logits`
        is correctly interpreted.
        """
        bce_loss_none = F.binary_cross_entropy_with_logits(
            self.logits, self.targets, reduction="none"
        )
        p_t_from_loss = torch.exp(-bce_loss_none)

        probs = torch.sigmoid(self.logits)
        p_t_direct = torch.where(self.targets == 1, probs, 1 - probs)

        self.assertTrue(torch.allclose(p_t_from_loss, p_t_direct, atol=1e-6))

    def test_focal_loss_calculation(self):
        """
        Test the focal loss output against a manually calculated value.
        """
        logits = torch.tensor([-2.0, 0.5, 1.5, 3.0])
        targets = torch.tensor([0.0, 0.0, 1.0, 1.0])
        alpha = 0.25
        gamma = 2.0

        loss_fn = FocalLossWithLogits(alpha=alpha, gamma=gamma)
        loss = loss_fn(logits, targets)

        # Manual calculation
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        expected_loss = (alpha_t * (1 - p_t) ** gamma * bce).mean()

        self.assertTrue(torch.allclose(loss, expected_loss))

    def test_gamma_zero_equals_weighted_bce(self):
        """
        When gamma=0, Focal Loss should be equivalent to weighted BCE.
        """
        alpha = 0.3
        loss_fn_gamma_zero = FocalLossWithLogits(alpha=alpha, gamma=0)
        focal_loss = loss_fn_gamma_zero(self.logits, self.targets)

        # Calculate weighted BCE manually
        bce_loss = F.binary_cross_entropy_with_logits(
            self.logits, self.targets, reduction="none"
        )
        alpha_t = alpha * self.targets + (1 - alpha) * (1 - self.targets)
        weighted_bce_loss = (alpha_t * bce_loss).mean()

        self.assertTrue(torch.allclose(focal_loss, weighted_bce_loss, atol=1e-6))

    def test_forward_pass_with_different_shapes(self):
        """
        Tests forward pass with different input shapes.
        """
        loss_fn = FocalLossWithLogits()

        # 1D input
        logits_1d = torch.randn(10)
        targets_1d = torch.randint(0, 2, (10,)).float()
        loss_1d = loss_fn(logits_1d, targets_1d)
        self.assertIsInstance(loss_1d, torch.Tensor)
        self.assertEqual(loss_1d.shape, torch.Size([]))

        # 2D input (batch, classes)
        logits_2d = torch.randn(10, 5)
        targets_2d = torch.randint(0, 2, (10, 5)).float()
        loss_2d = loss_fn(logits_2d, targets_2d)
        self.assertIsInstance(loss_2d, torch.Tensor)
        self.assertEqual(loss_2d.shape, torch.Size([]))


if __name__ == "__main__":
    unittest.main()
