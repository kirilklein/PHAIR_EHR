import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from corebehrt.functional.causal.simulate import (
    combine_counterfactuals,
    simulate_outcome_from_encodings,
)


class TestSimulateOutcomeFromEncodings(unittest.TestCase):
    def setUp(self):
        # Use a simple encodings matrix and exposure vector.
        self.encodings = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.exposure = np.array([0, 1, 0])
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
        self.assertTrue(np.all(prob >= 0))
        self.assertTrue(np.all(prob <= 1))

    def test_binary_outcome_values(self):
        """Test that binary outcomes are either 0 or 1."""
        outcome, _ = simulate_outcome_from_encodings(
            self.encodings, self.exposure, **self.params
        )
        self.assertTrue(np.all((outcome == 0) | (outcome == 1)))

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
        self.assertTrue(np.all(prob_high > 0.9))

        # Low intercept: probabilities should be near 0.
        params_low = self.params.copy()
        params_low["intercept"] = -10.0
        _, prob_low = simulate_outcome_from_encodings(
            self.encodings, self.exposure, **params_low
        )
        self.assertTrue(np.all(prob_low < 0.1))


class TestCombineCounterfactuals(unittest.TestCase):
    def test_combine_counterfactuals_mixed(self):
        """Test counterfactual combination with mixed exposure values."""
        exposure = np.array([0, 1, 0, 1])
        exposed_values = np.array([10, 20, 30, 40])
        control_values = np.array([100, 200, 300, 400])
        expected = np.where(exposure == 1, control_values, exposed_values)
        result = combine_counterfactuals(exposure, exposed_values, control_values)
        assert_array_equal(result, expected)

    def test_combine_counterfactuals_all_exposed(self):
        """Test counterfactual combination when all individuals are exposed."""
        exposure = np.array([1, 1, 1])
        exposed_values = np.array([5, 5, 5])
        control_values = np.array([50, 50, 50])
        expected = np.where(exposure == 1, control_values, exposed_values)
        result = combine_counterfactuals(exposure, exposed_values, control_values)
        assert_array_equal(result, expected)

    def test_combine_counterfactuals_all_control(self):
        """Test counterfactual combination when all individuals are in control."""
        exposure = np.array([0, 0, 0])
        exposed_values = np.array([5, 5, 5])
        control_values = np.array([50, 50, 50])
        expected = np.where(exposure == 1, control_values, exposed_values)
        result = combine_counterfactuals(exposure, exposed_values, control_values)
        assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
