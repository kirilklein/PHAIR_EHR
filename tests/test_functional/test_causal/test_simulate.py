import unittest

import torch
from numpy.testing import assert_allclose

from corebehrt.functional.causal.simulate import simulate_outcome_from_encodings


class TestSimulateOutcomeFromEncodings(unittest.TestCase):
    def setUp(self):
        # Use a simple encodings matrix and exposure vector.
        self.encodings = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.exposure = torch.tensor([0, 1, 0])
        self.params = {
            "exposure_coef": 1.0,
            "enc_coef": 1.0,
            "intercept": 0.0,
            "enc_sparsity": 0.7,
            "enc_scale": 0.1,
            "random_state": 42,
        }

    def test_output_shape(self):
        """Test that output arrays have the correct shape."""
        outcome, prob = simulate_outcome_from_encodings(
            self.encodings, self.exposure, **self.params
        )
        self.assertEqual(outcome.shape, (self.encodings.shape[0],))
        self.assertEqual(prob.shape, (self.encodings.shape[0],))

    def test_probability_bounds(self):
        """Test that probabilities are between 0 and 1."""
        _, prob = simulate_outcome_from_encodings(
            self.encodings, self.exposure, **self.params
        )
        self.assertTrue(torch.all(prob >= 0))
        self.assertTrue(torch.all(prob <= 1))

    def test_binary_outcome_values(self):
        """Test that binary outcomes are either 0 or 1."""
        outcome, _ = simulate_outcome_from_encodings(
            self.encodings, self.exposure, **self.params
        )
        self.assertTrue(torch.all((outcome == 0) | (outcome == 1)))

    def test_reproducibility_of_probability(self):
        """Test that probabilities are reproducible with fixed random_state."""
        _, prob1 = simulate_outcome_from_encodings(
            self.encodings, self.exposure, **self.params
        )
        _, prob2 = simulate_outcome_from_encodings(
            self.encodings, self.exposure, **self.params
        )
        assert_allclose(prob1, prob2)

    def test_extreme_intercept(self):
        """Test that extreme intercept values push probabilities near 0 or 1."""
        # High intercept: probabilities should be near 1.
        params_high = self.params.copy()
        params_high["intercept"] = 10.0
        _, prob_high = simulate_outcome_from_encodings(
            self.encodings, self.exposure, **params_high
        )
        self.assertTrue(torch.all(prob_high > 0.9))

        # Low intercept: probabilities should be near 0.
        params_low = self.params.copy()
        params_low["intercept"] = -10.0
        _, prob_low = simulate_outcome_from_encodings(
            self.encodings, self.exposure, **params_low
        )
        self.assertTrue(torch.all(prob_low < 0.1))


if __name__ == "__main__":
    unittest.main()
