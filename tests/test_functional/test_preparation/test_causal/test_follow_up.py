import unittest
import pandas as pd
import numpy as np
from corebehrt.functional.preparation.causal.follow_up import (
    prepare_follow_ups_adjusted,
    prepare_follow_ups_simple,
)
from corebehrt.constants.causal.data import (
    START_COL,
    END_COL,
    GROUP_COL,
    NON_COMPLIANCE_COL,
    DEATH_COL,
    START_TIME_COL,
    END_TIME_COL,
)
from corebehrt.constants.data import PID_COL, ABSPOS_COL, TIMESTAMP_COL


class TestPrepareFollowUpsAdjusted(unittest.TestCase):
    """Test cases for the prepare_follow_ups_adjusted function."""

    def test_prepare_follow_ups_adjusted_basic(self):
        """Test the prepare_follow_ups_adjusted function with complete data."""
        # 1. Setup test data
        follow_ups = pd.DataFrame(
            {
                PID_COL: [1, 2, 3, 4],
                START_COL: [10.0, 20.0, 30.0, 40.0],
                END_COL: [100.0, 200.0, 300.0, 400.0],
            }
        )

        non_compliance_abspos = {
            1: 80.0,  # Patient 1 stops compliance at 80
            2: 250.0,  # Patient 2 stops compliance at 250
            3: 150.0,  # Patient 3 stops compliance at 150
            4: 500.0,  # Patient 4 stops compliance at 500
        }

        deaths = {
            1: np.nan,  # Patient 1 doesn't die
            2: 180.0,  # Patient 2 dies at 180
            3: np.nan,  # Patient 3 doesn't die
            4: 350.0,  # Patient 4 dies at 350
        }

        group_dict = {
            1: 0,  # Patients 1 and 2 are in group 0
            2: 0,
            3: 1,  # Patients 3 and 4 are in group 1
            4: 1,
        }

        # 2. Execute function
        result = prepare_follow_ups_adjusted(
            follow_ups, non_compliance_abspos, deaths, group_dict
        )

        # 3. Expected results (individual patient minimums):
        # Patient 1: min(100, 80, inf) = 80
        # Patient 2: min(200, 250, 180) = 180
        # Patient 3: min(300, 150, inf) = 150
        # Patient 4: min(400, 500, 350) = 350

        expected_end_values = [80.0, 180.0, 150.0, 350.0]

        # 4. Assertions
        self.assertEqual(len(result), 4, f"Expected 4 rows, got {len(result)}")
        pd.testing.assert_series_equal(
            result[NON_COMPLIANCE_COL],
            pd.Series([80.0, 250.0, 150.0, 500.0], name=NON_COMPLIANCE_COL),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            result[DEATH_COL],
            pd.Series([np.nan, 180.0, np.nan, 350.0], name=DEATH_COL),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            result[GROUP_COL],
            pd.Series([0, 0, 1, 1], name=GROUP_COL),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            result[END_COL],
            pd.Series(expected_end_values, name=END_COL),
            check_names=False,
        )

    def test_prepare_follow_ups_adjusted_missing_data(self):
        """Test the function with missing data (NaN values)."""
        # 1. Setup test data with missing values
        follow_ups = pd.DataFrame(
            {PID_COL: [1, 2], START_COL: [10.0, 20.0], END_COL: [100.0, 200.0]}
        )

        non_compliance_abspos = {
            1: 80.0,  # Patient 1 has compliance data
            # Patient 2 missing from non_compliance_abspos
        }

        deaths = {
            1: np.nan,  # Patient 1 doesn't die
            # Patient 2 missing from deaths
        }

        group_dict = {1: 0, 2: 0}

        # 2. Execute function
        result = prepare_follow_ups_adjusted(
            follow_ups, non_compliance_abspos, deaths, group_dict
        )

        # 3. Expected results (individual patient minimums):
        # Patient 1: min(100, 80, inf) = 80
        # Patient 2: min(200, NaN, NaN) = 200

        # 4. Assertions
        self.assertTrue(
            pd.isna(result.loc[result[PID_COL] == 2, NON_COMPLIANCE_COL].iloc[0]),
            "Missing non-compliance should be NaN",
        )
        self.assertTrue(
            pd.isna(result.loc[result[PID_COL] == 2, DEATH_COL].iloc[0]),
            "Missing death should be NaN",
        )
        pd.testing.assert_series_equal(
            result[END_COL], pd.Series([80.0, 200.0], name=END_COL), check_names=False
        )

    def test_prepare_follow_ups_adjusted_empty_input(self):
        """Test the function with empty input data."""
        # 1. Setup empty test data
        follow_ups = pd.DataFrame(columns=[PID_COL, START_COL, END_COL])

        non_compliance_abspos = {}
        deaths = {}
        group_dict = {}

        # 2. Execute function
        result = prepare_follow_ups_adjusted(
            follow_ups, non_compliance_abspos, deaths, group_dict
        )

        # 3. Assertions
        self.assertEqual(len(result), 0, "Empty input should return empty result")
        self.assertTrue(
            NON_COMPLIANCE_COL in result.columns, "Should have non_compliance column"
        )
        self.assertTrue(DEATH_COL in result.columns, "Should have death column")
        self.assertTrue(GROUP_COL in result.columns, "Should have group column")

    def test_prepare_follow_ups_adjusted_single_group(self):
        """Test the function with all patients in a single group."""
        # 1. Setup test data
        follow_ups = pd.DataFrame(
            {PID_COL: [1, 2], START_COL: [10.0, 20.0], END_COL: [100.0, 200.0]}
        )

        non_compliance_abspos = {1: 80.0, 2: 150.0}
        deaths = {1: np.nan, 2: 120.0}
        group_dict = {1: 0, 2: 0}  # Both in same group

        # 2. Execute function
        result = prepare_follow_ups_adjusted(
            follow_ups, non_compliance_abspos, deaths, group_dict
        )

        # 3. Expected results (individual patient minimums):
        # Patient 1: min(100, 80, inf) = 80
        # Patient 2: min(200, 150, 120) = 120

        # 4. Assertions
        expected_end_values = [80.0, 120.0]
        pd.testing.assert_series_equal(
            result[END_COL],
            pd.Series(expected_end_values, name=END_COL),
            check_names=False,
        )


class TestPrepareFollowUpsSimple(unittest.TestCase):
    """Test cases for the prepare_follow_ups_simple function."""

    def test_prepare_follow_ups_simple_basic(self):
        """Test basic functionality with positive follow-up hours."""
        # 1. Setup test data
        index_dates = pd.DataFrame(
            {
                PID_COL: [1, 2, 3],
                TIMESTAMP_COL: [
                    pd.Timestamp("2023-01-01 12:00:00"),
                    pd.Timestamp("2023-01-02 12:00:00"),
                    pd.Timestamp("2023-01-03 12:00:00"),
                ],
                ABSPOS_COL: [100.0, 200.0, 300.0],
                "other_col": [
                    "a",
                    "b",
                    "c",
                ],  # Additional column to ensure it's preserved
            }
        )

        n_hours_start_follow_up = 24
        n_hours_end_follow_up = 168  # 1 week
        data_end = pd.Timestamp("2023-12-31 23:59:59")

        # 2. Execute function
        result = prepare_follow_ups_simple(
            index_dates, n_hours_start_follow_up, n_hours_end_follow_up, data_end
        )

        # 3. Assertions
        self.assertEqual(len(result), 3, "Should have same number of rows")
        self.assertNotIn(ABSPOS_COL, result.columns, "abspos column should be dropped")
        self.assertIn(START_COL, result.columns, "Should have start column")
        self.assertIn(END_COL, result.columns, "Should have end column")
        self.assertIn("other_col", result.columns, "Other columns should be preserved")

        # Check that start and end times are calculated correctly from timestamps
        self.assertTrue(all(result[START_COL] > 0), "Start times should be positive")
        self.assertTrue(
            all(result[END_COL] > result[START_COL]),
            "End times should be after start times",
        )

    def test_prepare_follow_ups_simple_zero_hours(self):
        """Test with zero follow-up hours."""
        # 1. Setup test data
        index_dates = pd.DataFrame(
            {
                PID_COL: [1, 2],
                TIMESTAMP_COL: [
                    pd.Timestamp("2023-01-01 12:00:00"),
                    pd.Timestamp("2023-01-02 12:00:00"),
                ],
                ABSPOS_COL: [100.0, 200.0],
            }
        )

        n_hours_start_follow_up = 0
        n_hours_end_follow_up = 0
        data_end = pd.Timestamp("2023-12-31 23:59:59")

        # 2. Execute function
        result = prepare_follow_ups_simple(
            index_dates, n_hours_start_follow_up, n_hours_end_follow_up, data_end
        )

        # 3. Assertions
        # Start and end should be the same when both are 0 hours
        pd.testing.assert_series_equal(
            result[START_COL],
            result[END_COL],
            check_names=False,
        )

    def test_prepare_follow_ups_simple_negative_hours(self):
        """Test with negative follow-up hours."""
        # 1. Setup test data
        index_dates = pd.DataFrame(
            {
                PID_COL: [1, 2],
                TIMESTAMP_COL: [
                    pd.Timestamp("2023-01-01 12:00:00"),
                    pd.Timestamp("2023-01-02 12:00:00"),
                ],
                ABSPOS_COL: [100.0, 200.0],
            }
        )

        n_hours_start_follow_up = -12
        n_hours_end_follow_up = -24
        data_end = pd.Timestamp("2023-12-31 23:59:59")

        # 2. Execute function
        result = prepare_follow_ups_simple(
            index_dates, n_hours_start_follow_up, n_hours_end_follow_up, data_end
        )

        # 3. Assertions
        # End should be before start when end hours are more negative
        self.assertTrue(
            all(result[END_COL] < result[START_COL]),
            "End times should be before start times with negative hours",
        )

    def test_prepare_follow_ups_simple_none_end_hours(self):
        """Test with None end hours to use data_end."""
        # 1. Setup test data
        index_dates = pd.DataFrame(
            {
                PID_COL: [1, 2],
                TIMESTAMP_COL: [
                    pd.Timestamp("2023-01-01 12:00:00"),
                    pd.Timestamp("2023-01-02 12:00:00"),
                ],
                ABSPOS_COL: [100.0, 200.0],
            }
        )

        n_hours_start_follow_up = 24
        n_hours_end_follow_up = None  # Should use data_end
        data_end = pd.Timestamp("2023-12-31 23:59:59")

        # 2. Execute function
        result = prepare_follow_ups_simple(
            index_dates, n_hours_start_follow_up, n_hours_end_follow_up, data_end
        )

        # 3. Assertions
        # All end times should be the same (data_end converted to hours since epoch)
        expected_end_hours = result[END_COL].iloc[0]  # Get the first value
        self.assertTrue(
            all(result[END_COL] == expected_end_hours),
            "All end times should be the same when using data_end",
        )
        self.assertTrue(
            all(result[END_COL] > result[START_COL]),
            "End times should be after start times",
        )

    def test_prepare_follow_ups_simple_empty_input(self):
        """Test with empty DataFrame."""
        # 1. Setup empty test data
        index_dates = pd.DataFrame(columns=[PID_COL, TIMESTAMP_COL, ABSPOS_COL])

        n_hours_start_follow_up = 24
        n_hours_end_follow_up = 168
        data_end = pd.Timestamp("2023-12-31 23:59:59")

        # 2. Execute function
        result = prepare_follow_ups_simple(
            index_dates, n_hours_start_follow_up, n_hours_end_follow_up, data_end
        )

        # 3. Assertions
        self.assertEqual(len(result), 0, "Empty input should return empty result")
        self.assertIn(START_COL, result.columns, "Should have start column")
        self.assertIn(END_COL, result.columns, "Should have end column")
        self.assertNotIn(ABSPOS_COL, result.columns, "abspos column should be dropped")

    def test_prepare_follow_ups_simple_single_row(self):
        """Test with single row input."""
        # 1. Setup test data
        index_dates = pd.DataFrame(
            {
                PID_COL: [1],
                TIMESTAMP_COL: [pd.Timestamp("2023-01-01 12:00:00")],
                ABSPOS_COL: [500.0],
            }
        )

        n_hours_start_follow_up = 48
        n_hours_end_follow_up = 720  # 30 days
        data_end = pd.Timestamp("2023-12-31 23:59:59")

        # 2. Execute function
        result = prepare_follow_ups_simple(
            index_dates, n_hours_start_follow_up, n_hours_end_follow_up, data_end
        )

        # 3. Assertions
        self.assertEqual(len(result), 1, "Should have one row")
        self.assertTrue(
            result.iloc[0][END_COL] > result.iloc[0][START_COL],
            "End time should be after start time",
        )

    def test_prepare_follow_ups_simple_preserves_other_columns(self):
        """Test that other columns are preserved."""
        # 1. Setup test data with multiple columns
        index_dates = pd.DataFrame(
            {
                PID_COL: [1, 2],
                TIMESTAMP_COL: [
                    pd.Timestamp("2023-01-01 12:00:00"),
                    pd.Timestamp("2023-01-02 12:00:00"),
                ],
                ABSPOS_COL: [100.0, 200.0],
                "other_col1": [25, 30],
                "other_col2": ["M", "F"],
            }
        )

        n_hours_start_follow_up = 24
        n_hours_end_follow_up = 168
        data_end = pd.Timestamp("2023-12-31 23:59:59")

        # 2. Execute function
        result = prepare_follow_ups_simple(
            index_dates, n_hours_start_follow_up, n_hours_end_follow_up, data_end
        )

        # 3. Assertions
        expected_columns = {
            PID_COL,
            START_COL,
            END_COL,
            START_TIME_COL,
            END_TIME_COL,
            TIMESTAMP_COL,
            "other_col1",
            "other_col2",
        }
        self.assertEqual(
            set(result.columns),
            expected_columns,
            "All columns should be preserved except abspos",
        )

        # Check that original data is preserved
        pd.testing.assert_series_equal(
            result["other_col1"],
            pd.Series([25, 30], name="other_col1"),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            result["other_col2"],
            pd.Series(["M", "F"], name="other_col2"),
            check_names=False,
        )

    def test_prepare_follow_ups_simple_float_hours(self):
        """Test with float values in hours."""
        # 1. Setup test data with timestamps
        index_dates = pd.DataFrame(
            {
                PID_COL: [1, 2],
                TIMESTAMP_COL: [
                    pd.Timestamp("2023-01-01 12:00:00"),
                    pd.Timestamp("2023-01-02 12:00:00"),
                ],
                ABSPOS_COL: [100.5, 200.75],
            }
        )

        n_hours_start_follow_up = 12.5
        n_hours_end_follow_up = 48.25
        data_end = pd.Timestamp("2023-12-31 23:59:59")

        # 2. Execute function
        result = prepare_follow_ups_simple(
            index_dates, n_hours_start_follow_up, n_hours_end_follow_up, data_end
        )

        # 3. Assertions
        # Check that the time differences are correct
        time_diff = result[END_COL] - result[START_COL]
        expected_diff = 48.25 - 12.5  # 35.75 hours
        self.assertTrue(
            all(abs(time_diff - expected_diff) < 0.001),
            "Time differences should match expected hours",
        )


if __name__ == "__main__":
    unittest.main()
