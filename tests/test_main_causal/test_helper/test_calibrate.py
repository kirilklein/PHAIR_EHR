import unittest
import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from corebehrt.main_causal.helper.calibrate import (
    train_isotonic_regression,
    calibrate_probas,
    split_data,
)
from corebehrt.constants.causal.data import PROBAS, TARGETS
from corebehrt.constants.data import PID_COL


class TestCalibrationFunctions(unittest.TestCase):
    def setUp(self):
        # Sample training data for isotonic regression
        self.train_data = pd.DataFrame(
            {
                PROBAS: [0.1, 0.4, 0.6, 0.9],
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

    def test_train_isotonic_regression(self):
        calibrator = train_isotonic_regression(self.train_data)
        # Check if the returned object is an IsotonicRegression instance.
        self.assertIsInstance(calibrator, IsotonicRegression)

        # Verify that predictions are monotonic by testing a sorted array.
        test_inputs = np.array([0.0, 0.3, 0.5, 0.7, 1.0])
        predictions = calibrator.predict(test_inputs)
        self.assertTrue(
            np.all(np.diff(predictions) >= 0),
            "Predictions should be monotonic non-decreasing.",
        )

    def test_calibrate_probas_default_epsilon(self):
        # Create a simple calibrator that is trained on edge values.
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit([0.1, 0.9], [0, 1])

        # Test with values that might go out-of-bound after calibration.
        test_probas = pd.Series([0.0, 0.5, 1.0])
        calibrated = calibrate_probas(calibrator, test_probas)

        # Check output type and length.
        self.assertIsInstance(calibrated, np.ndarray)
        self.assertEqual(len(calibrated), 3)

        # Assert that values are clipped using default epsilon (1e-8).
        epsilon = 1e-8
        self.assertTrue(np.all(calibrated >= epsilon))
        self.assertTrue(np.all(calibrated <= 1 - epsilon))

    def test_calibrate_probas_custom_epsilon(self):
        # Test calibrate_probas with a custom epsilon.
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit([0.1, 0.9], [0, 1])
        custom_epsilon = 0.05
        test_probas = pd.Series([0.0, 0.5, 1.0])
        calibrated = calibrate_probas(calibrator, test_probas, epsilon=custom_epsilon)

        # Verify clipping with custom epsilon.
        self.assertTrue(np.all(calibrated >= custom_epsilon))
        self.assertTrue(np.all(calibrated <= 1 - custom_epsilon))

        # Optionally, check specific values if the calibrator yields deterministic outputs.
        # np.testing.assert_allclose(calibrated[0], custom_epsilon)

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

    def test_error_on_invalid_train_data(self):
        # Test error handling if required columns are missing.
        invalid_data = pd.DataFrame({"some_other_column": [1, 2, 3]})
        with self.assertRaises(KeyError):
            _ = train_isotonic_regression(invalid_data)


if __name__ == "__main__":
    unittest.main()
