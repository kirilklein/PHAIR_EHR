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
)
from corebehrt.constants.data import PID_COL, ABSPOS_COL


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

        # 3. Expected results:
        # Patient 1: min(100, 80, inf) = 80, then min within group 0 = min(80, 180) = 80
        # Patient 2: min(200, 250, 180) = 180, then min within group 0 = min(80, 180) = 80
        # Patient 3: min(300, 150, inf) = 150, then min within group 1 = min(150, 350) = 150
        # Patient 4: min(400, 500, 350) = 350, then min within group 1 = min(150, 350) = 150

        expected_end_values = [80.0, 80.0, 150.0, 150.0]

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

        # 3. Expected results:
        # Patient 1: min(100, 80, inf) = 80, then min within group 0 = min(80, 200) = 80
        # Patient 2: min(200, NaN, NaN) = 200, then min within group 0 = min(80, 200) = 80

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
            result[END_COL], pd.Series([80.0, 80.0], name=END_COL), check_names=False
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

        # 3. Expected results:
        # Patient 1: min(100, 80, inf) = 80, then min within group 0 = min(80, 120) = 80
        # Patient 2: min(200, 150, 120) = 120, then min within group 0 = min(80, 120) = 80

        # 4. Assertions
        expected_end_values = [80.0, 80.0]
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
                "pid": [1, 2, 3],
                "abspos": [100.0, 200.0, 300.0],
                "other_col": [
                    "a",
                    "b",
                    "c",
                ],  # Additional column to ensure it's preserved
            }
        )

        n_hours_start_follow_up = 24
        n_hours_end_follow_up = 168  # 1 week

        # 2. Execute function
        result = prepare_follow_ups_simple(
            index_dates, n_hours_start_follow_up, n_hours_end_follow_up
        )

        # 3. Expected results:
        # Patient 1: start=100+24=124, end=100+168=268
        # Patient 2: start=200+24=224, end=200+168=368
        # Patient 3: start=300+24=324, end=300+168=468

        # 4. Assertions
        self.assertEqual(len(result), 3, "Should have same number of rows")
        self.assertNotIn(ABSPOS_COL, result.columns, "abspos column should be dropped")
        self.assertIn(START_COL, result.columns, "Should have start column")
        self.assertIn(END_COL, result.columns, "Should have end column")
        self.assertIn("other_col", result.columns, "Other columns should be preserved")

        expected_start = [124.0, 224.0, 324.0]
        expected_end = [268.0, 368.0, 468.0]

        pd.testing.assert_series_equal(
            result[START_COL],
            pd.Series(expected_start, name=START_COL),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            result[END_COL], pd.Series(expected_end, name=END_COL), check_names=False
        )

    def test_prepare_follow_ups_simple_zero_hours(self):
        """Test with zero follow-up hours."""
        # 1. Setup test data
        index_dates = pd.DataFrame({PID_COL: [1, 2], ABSPOS_COL: [100.0, 200.0]})

        n_hours_start_follow_up = 0
        n_hours_end_follow_up = 0

        # 2. Execute function
        result = prepare_follow_ups_simple(
            index_dates, n_hours_start_follow_up, n_hours_end_follow_up
        )

        # 3. Expected results:
        # Patient 1: start=100+0=100, end=100+0=100
        # Patient 2: start=200+0=200, end=200+0=200

        # 4. Assertions
        expected_start = [100.0, 200.0]
        expected_end = [100.0, 200.0]

        pd.testing.assert_series_equal(
            result[START_COL],
            pd.Series(expected_start, name=START_COL),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            result[END_COL], pd.Series(expected_end, name=END_COL), check_names=False
        )

    def test_prepare_follow_ups_simple_negative_hours(self):
        """Test with negative follow-up hours."""
        # 1. Setup test data
        index_dates = pd.DataFrame({PID_COL: [1, 2], ABSPOS_COL: [100.0, 200.0]})

        n_hours_start_follow_up = -12
        n_hours_end_follow_up = -24

        # 2. Execute function
        result = prepare_follow_ups_simple(
            index_dates, n_hours_start_follow_up, n_hours_end_follow_up
        )

        # 3. Expected results:
        # Patient 1: start=100+(-12)=88, end=100+(-24)=76
        # Patient 2: start=200+(-12)=188, end=200+(-24)=176

        # 4. Assertions
        expected_start = [88.0, 188.0]
        expected_end = [76.0, 176.0]

        pd.testing.assert_series_equal(
            result[START_COL],
            pd.Series(expected_start, name=START_COL),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            result[END_COL], pd.Series(expected_end, name=END_COL), check_names=False
        )

    def test_prepare_follow_ups_simple_empty_input(self):
        """Test with empty DataFrame."""
        # 1. Setup empty test data
        index_dates = pd.DataFrame(columns=[PID_COL, ABSPOS_COL])

        n_hours_start_follow_up = 24
        n_hours_end_follow_up = 168

        # 2. Execute function
        result = prepare_follow_ups_simple(
            index_dates, n_hours_start_follow_up, n_hours_end_follow_up
        )

        # 3. Assertions
        self.assertEqual(len(result), 0, "Empty input should return empty result")
        self.assertIn(START_COL, result.columns, "Should have start column")
        self.assertIn(END_COL, result.columns, "Should have end column")
        self.assertNotIn(ABSPOS_COL, result.columns, "abspos column should be dropped")

    def test_prepare_follow_ups_simple_single_row(self):
        """Test with single row input."""
        # 1. Setup test data
        index_dates = pd.DataFrame({PID_COL: [1], ABSPOS_COL: [500.0]})

        n_hours_start_follow_up = 48
        n_hours_end_follow_up = 720  # 30 days

        # 2. Execute function
        result = prepare_follow_ups_simple(
            index_dates, n_hours_start_follow_up, n_hours_end_follow_up
        )

        # 3. Expected results:
        # Patient 1: start=500+48=548, end=500+720=1220

        # 4. Assertions
        self.assertEqual(len(result), 1, "Should have one row")
        self.assertEqual(result.iloc[0][START_COL], 548.0, "Start time incorrect")
        self.assertEqual(result.iloc[0][END_COL], 1220.0, "End time incorrect")

    def test_prepare_follow_ups_simple_preserves_other_columns(self):
        """Test that other columns are preserved."""
        # 1. Setup test data with multiple columns
        index_dates = pd.DataFrame(
            {
                PID_COL: [1, 2],
                ABSPOS_COL: [100.0, 200.0],
                "other_col1": [25, 30],
                "other_col2": ["M", "F"],
            }
        )

        n_hours_start_follow_up = 24
        n_hours_end_follow_up = 168

        # 2. Execute function
        result = prepare_follow_ups_simple(
            index_dates, n_hours_start_follow_up, n_hours_end_follow_up
        )

        # 3. Assertions
        expected_columns = {PID_COL, START_COL, END_COL, "other_col1", "other_col2"}
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

    def test_prepare_follow_ups_simple_float_abspos(self):
        """Test with float values in abspos column."""
        # 1. Setup test data with float abspos
        index_dates = pd.DataFrame({PID_COL: [1, 2], ABSPOS_COL: [100.5, 200.75]})

        n_hours_start_follow_up = 12.5
        n_hours_end_follow_up = 48.25

        # 2. Execute function
        result = prepare_follow_ups_simple(
            index_dates, n_hours_start_follow_up, n_hours_end_follow_up
        )

        # 3. Expected results:
        # Patient 1: start=100.5+12.5=113.0, end=100.5+48.25=148.75
        # Patient 2: start=200.75+12.5=213.25, end=200.75+48.25=249.0

        # 4. Assertions
        expected_start = [113.0, 213.25]
        expected_end = [148.75, 249.0]

        pd.testing.assert_series_equal(
            result[START_COL],
            pd.Series(expected_start, name=START_COL),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            result[END_COL], pd.Series(expected_end, name=END_COL), check_names=False
        )


if __name__ == "__main__":
    unittest.main()
