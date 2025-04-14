import unittest

import numpy as np
import pandas as pd
import torch

from corebehrt.constants.causal.data import PROBAS, TARGETS
from corebehrt.constants.data import PID_COL
from corebehrt.main_causal.helper.calibrate import calibrate, split_data


class TestCalibrationFunctions(unittest.TestCase):
    def setUp(self):
        # Sample training data for calibration
        self.train_data = pd.DataFrame(
            {
                PROBAS: [0.1, 0.4, 0.6, 0.9],
                TARGETS: [0, 0, 1, 1],
            }
        )
        self.val_data = pd.DataFrame(
            {
                PROBAS: [0.2, 0.5, 0.7, 0.8],
                TARGETS: [0, 0, 1, 1],
            }
        )
        # Sample predictions dataframe for splitting
        self.predictions_df = pd.DataFrame(
            {
                PID_COL: ["pid1", "pid2", "pid3", "pid4"],
                PROBAS: [0.1, 0.4, 0.6, 0.9],
                TARGETS: [0, 0, 1, 1],
            }
        )

        # Use torch.Tensor for consistency with type hints, but also test list behavior.
        self.train_pids_tensor = torch.tensor([0, 1])  # Assuming numeric IDs if needed
        self.val_pids_tensor = torch.tensor([2, 3])
        # Also prepare string-based PIDs for the given predictions_df.
        self.train_pids_list = ["pid1", "pid2"]
        self.val_pids_list = ["pid3", "pid4"]

    def test_calibrate(self):
        predictions = calibrate(self.train_data, self.val_data)
        # Check if the returned object is an IsotonicRegression instance.
        self.assertIsInstance(predictions, np.ndarray)

    def test_split_data_with_list_pids(self):
        # Test split_data using string-based PID lists.
        train_data, val_data = split_data(
            self.predictions_df, self.train_pids_list, self.val_pids_list
        )
        self.assertEqual(len(train_data), 2)
        self.assertEqual(len(val_data), 2)
        self.assertTrue(all(pid in self.train_pids_list for pid in train_data[PID_COL]))
        self.assertTrue(all(pid in self.val_pids_list for pid in val_data[PID_COL]))
        # Ensure all expected columns are present.
        expected_columns = {PID_COL, PROBAS, TARGETS}
        self.assertEqual(set(train_data.columns), expected_columns)
        self.assertEqual(set(val_data.columns), expected_columns)

    def test_split_data_with_missing_pids(self):
        # Test when some PIDs are not present in the DataFrame.
        missing_train = ["nonexistent"]
        train_data, _ = split_data(
            self.predictions_df, missing_train, self.val_pids_list
        )
        self.assertEqual(len(train_data), 0, "Expected no rows for missing train PIDs.")

    def test_split_data_preserves_order(self):
        # Optionally test if the original order or indices are preserved.
        train_data, _ = split_data(
            self.predictions_df, self.train_pids_list, self.val_pids_list
        )
        expected_order = self.predictions_df[
            self.predictions_df[PID_COL].isin(self.train_pids_list)
        ].index.tolist()
        self.assertEqual(
            list(train_data.index),
            expected_order,
            "The order of rows in the train split should be preserved.",
        )


if __name__ == "__main__":
    unittest.main()
