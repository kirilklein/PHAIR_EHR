import unittest
from unittest.mock import patch

import numpy as np
import torch

from corebehrt.constants.causal import (
    EXPOSURE_COL,
    OUTCOMES,
    PROBAS,
    SIMULATED_OUTCOME_CONTROL,
    SIMULATED_OUTCOME_EXPOSED,
    SIMULATED_PROBAS_CONTROL,
    SIMULATED_PROBAS_EXPOSED,
)
from corebehrt.constants.data import PID_COL, TIMESTAMP_COL
from corebehrt.main.helper.causal.simulate import DATE_FUTURE, simulate


# A simple dummy logger that does nothing.
class DummyLogger:
    def info(self, msg):
        pass


# Dummy implementation for simulate_outcome_from_encodings.
def dummy_simulate_outcome_from_encodings(encodings, exposure, **simulate_cfg):
    # For an input tensor of ones, return ones as outcomes and 0.9 for probabilities.
    # For an input tensor of zeros, return zeros as outcomes and 0.1 for probabilities.
    if torch.all(exposure == 1):
        outcome = torch.ones_like(exposure)
        proba = torch.full_like(exposure, 0.9, dtype=torch.float)
    elif torch.all(exposure == 0):
        outcome = torch.zeros_like(exposure)
        proba = torch.full_like(exposure, 0.1, dtype=torch.float)
    else:
        # For mixed values, choose element-wise.
        outcome = exposure.clone()
        proba = torch.where(exposure == 1, torch.tensor(0.9), torch.tensor(0.1))
    return outcome, proba


# Dummy implementation for get_true_outcome.
def dummy_get_true_outcome(exposure, outcome_exposed, outcome_control):
    # Return outcome_exposed where exposure==1 and outcome_control otherwise.
    return torch.where(exposure == 1, outcome_exposed, outcome_control)


class TestSimulate(unittest.TestCase):

    @patch(
        "corebehrt.main.helper.causal.simulate.simulate_outcome_from_encodings",
        side_effect=dummy_simulate_outcome_from_encodings,
    )
    @patch(
        "corebehrt.main.helper.causal.simulate.get_true_outcome",
        side_effect=dummy_get_true_outcome,
    )
    def test_simulate(self, mock_get_true_outcome, mock_simulate_outcome):
        logger = DummyLogger()
        pids = ["p1", "p2", "p3"]
        # Dummy encodings (content not used in dummy function).
        encodings = torch.zeros((3, 10))
        # Create a binary exposure tensor.
        exposure = torch.tensor([1, 0, 1])
        simulate_cfg = {"intercept": 1.0, "exposure_coef": 0.0, "enc_coef": 0.0}

        results_df, timestamp_df = simulate(
            logger, pids, encodings, exposure, simulate_cfg
        )

        # Verify that results_df contains the expected columns.
        expected_columns = [
            PID_COL,
            SIMULATED_OUTCOME_EXPOSED,
            SIMULATED_OUTCOME_CONTROL,
            SIMULATED_PROBAS_EXPOSED,
            SIMULATED_PROBAS_CONTROL,
            EXPOSURE_COL,
            PROBAS,
            OUTCOMES,
        ]
        for col in expected_columns:
            self.assertIn(col, results_df.columns)

        # Check that simulated outcomes/probabilities match the dummy behavior.
        self.assertListEqual(results_df[SIMULATED_OUTCOME_EXPOSED].tolist(), [1, 1, 1])
        self.assertListEqual(results_df[SIMULATED_OUTCOME_CONTROL].tolist(), [0, 0, 0])
        self.assertListEqual(
            [round(val, 4) for val in results_df[SIMULATED_PROBAS_EXPOSED].tolist()],
            [0.9, 0.9, 0.9],
        )
        self.assertListEqual(
            [round(val, 4) for val in results_df[SIMULATED_PROBAS_CONTROL].tolist()],
            [0.1, 0.1, 0.1],
        )

        # Check the final probabilities: if exposure==1, then 0.9; else 0.1.
        expected_final_probas = torch.where(
            exposure == 1, torch.tensor(0.9), torch.tensor(0.1)
        )
        self.assertTrue(
            np.allclose(
                results_df[PROBAS].astype(float).values, expected_final_probas.numpy()
            )
        )

        # Check the final outcomes: with our dummy, they match the exposure.
        self.assertListEqual(results_df[OUTCOMES].tolist(), exposure.tolist())

        # Test the timestamp_df: it should contain only patient IDs where outcomes are positive.
        expected_pids_timestamp = [
            pids[i] for i, val in enumerate(exposure.tolist()) if val == 1
        ]
        self.assertListEqual(timestamp_df[PID_COL].tolist(), expected_pids_timestamp)
        # And the TIMESTAMP_COL should be set to DATE_FUTURE for all rows.
        self.assertTrue(all(ts == DATE_FUTURE for ts in timestamp_df[TIMESTAMP_COL]))


if __name__ == "__main__":
    unittest.main()
