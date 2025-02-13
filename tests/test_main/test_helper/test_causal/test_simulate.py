import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from corebehrt.constants.causal import CF_OUTCOMES, CF_PROBAS, OUTCOMES, PROBAS, TARGETS
from corebehrt.constants.data import PID_COL, TIMESTAMP_COL

# Import the simulate function and related constants.
from corebehrt.main.helper.causal.simulate import DATE_FUTURE, simulate


# Define a simple dummy logger.
class DummyLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)


class TestSimulateFunction(unittest.TestCase):
    @patch("corebehrt.main.helper.causal.simulate.combine_counterfactuals")
    @patch("corebehrt.main.helper.causal.simulate.simulate_outcome_from_encodings")
    def test_simulate(self, mock_simulate_outcome, mock_combine_cf):
        # Set up controlled returns for the three calls to simulate_outcome_from_encodings.
        # First call: actual outcomes and probabilities.
        actual_outcome = np.array([0, 1, 0])
        actual_proba = np.array([0.3, 0.7, 0.4])
        # Second call: simulation under full exposure.
        all_exposed_outcome = np.array([1, 1, 1])
        all_exposed_proba = np.array([0.8, 0.9, 0.85])
        # Third call: simulation under control.
        all_control_outcome = np.array([0, 0, 0])
        all_control_proba = np.array([0.2, 0.1, 0.15])

        # Arrange that each call to simulate_outcome_from_encodings returns the next tuple.
        mock_simulate_outcome.side_effect = [
            (actual_outcome, actual_proba),
            (all_exposed_outcome, all_exposed_proba),
            (all_control_outcome, all_control_proba),
        ]

        # Define a side effect for combine_counterfactuals that mimics its np.where behavior.
        def combine_side_effect(exposure, exposed_values, control_values):
            # According to the logic: if exposure==1, choose control_values; otherwise choose exposed_values.
            return np.where(exposure == 1, control_values, exposed_values)

        mock_combine_cf.side_effect = combine_side_effect

        # Create a dummy logger.
        logger = DummyLogger()
        # Create a dummy encodings array.
        encodings = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        # Build a dummy predictions DataFrame containing the required columns.
        predictions = pd.DataFrame(
            {PID_COL: ["p1", "p2", "p3"], TARGETS: np.array([0, 1, 0])}
        )
        # A minimal simulation configuration.
        simulate_cfg = {"exposure_coef": 1.0, "enc_coef": 1.0, "intercept": 0.0}

        # Call the simulate function.
        results_df, timestamp_df = simulate(
            logger, encodings, predictions, simulate_cfg
        )

        # Verify that simulate_outcome_from_encodings was called three times.
        self.assertEqual(mock_simulate_outcome.call_count, 3)
        # And that combine_counterfactuals was called twice.
        self.assertEqual(mock_combine_cf.call_count, 2)

        # Verify that the results DataFrame has the expected columns.
        expected_columns = {PID_COL, OUTCOMES, CF_OUTCOMES, PROBAS, CF_PROBAS}
        self.assertEqual(set(results_df.columns), expected_columns)
        # Check that the PID column matches the input.
        self.assertTrue((results_df[PID_COL] == predictions[PID_COL]).all())
        # Verify that the actual outcomes and probabilities are as expected.
        np.testing.assert_array_equal(results_df[OUTCOMES].to_numpy(), actual_outcome)
        np.testing.assert_array_equal(results_df[PROBAS].to_numpy(), actual_proba)

        # Manually compute expected counterfactual outcomes and probabilities:
        # For each row, if TARGETS (exposure) == 1, then cf value should come from control simulation,
        # otherwise from the exposed simulation.
        exposure = predictions[TARGETS].to_numpy()
        expected_cf_outcomes = np.where(
            exposure == 1, all_control_outcome, all_exposed_outcome
        )
        expected_cf_probas = np.where(
            exposure == 1, all_control_proba, all_exposed_proba
        )
        np.testing.assert_array_equal(
            results_df[CF_OUTCOMES].to_numpy(), expected_cf_outcomes
        )
        np.testing.assert_array_equal(
            results_df[CF_PROBAS].to_numpy(), expected_cf_probas
        )

        # Check that the timestamp DataFrame contains only those rows where actual outcome == 1.
        expected_timestamp_pids = predictions[PID_COL][actual_outcome == 1].to_numpy()
        np.testing.assert_array_equal(
            timestamp_df[PID_COL].to_numpy(), expected_timestamp_pids
        )
        # And that every timestamp is set to DATE_FUTURE.
        self.assertTrue((timestamp_df[TIMESTAMP_COL] == DATE_FUTURE).all())

        # Optionally, verify that the logger recorded the key log messages.
        self.assertIn("simulate actual outcome", logger.messages)
        self.assertIn("simulate under exposure", logger.messages)
        self.assertIn("simulate under control", logger.messages)
        self.assertIn("combine into cf outcome", logger.messages)


if __name__ == "__main__":
    unittest.main()
