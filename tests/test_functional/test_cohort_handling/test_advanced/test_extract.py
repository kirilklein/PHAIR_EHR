"""Comprehensive tests for the extract_numeric_values function."""

import unittest
import pandas as pd
from pandas import to_datetime

from corebehrt.constants.cohort import CRITERION_FLAG, FINAL_MASK, NUMERIC_VALUE
from corebehrt.constants.data import PID_COL, TIMESTAMP_COL
from corebehrt.functional.cohort_handling.advanced.extract import extract_numeric_values


class TestExtractNumericValues(unittest.TestCase):
    """Test suite for extract_numeric_values function."""

    def setUp(self):
        """Set up common test data fixtures."""
        # Basic dataframe with valid numeric values
        self.basic_df = pd.DataFrame(
            {
                PID_COL: [1, 1, 2, 2],
                TIMESTAMP_COL: to_datetime(
                    [
                        "2023-01-01",
                        "2023-02-01",
                        "2023-01-15",
                        "2023-02-15",
                    ]
                ),
                NUMERIC_VALUE: [5.0, 7.0, 8.0, 10.0],
                FINAL_MASK: [True, True, True, True],
            }
        )

        # Flag dataframe for two patients
        self.flag_df = pd.DataFrame({PID_COL: [1, 2], CRITERION_FLAG: [True, True]})

    # =================================================================
    # 1. Basic Functionality Tests
    # =================================================================

    def test_extract_values_within_range(self):
        """Test basic case with min/max thresholds, values in range."""
        result = extract_numeric_values(
            self.basic_df, self.flag_df, min_value=6.0, max_value=9.0
        )

        # Patient 1 should have value 7.0 (most recent in range)
        # Patient 2 should have value 8.0 (most recent in range)
        self.assertEqual(result.shape[0], 2)

        val1 = result.loc[result[PID_COL] == 1, NUMERIC_VALUE].iloc[0]
        val2 = result.loc[result[PID_COL] == 2, NUMERIC_VALUE].iloc[0]

        self.assertAlmostEqual(val1, 7.0)
        self.assertAlmostEqual(val2, 8.0)

        # Both should have criterion flag True
        flag1 = result.loc[result[PID_COL] == 1, CRITERION_FLAG].iloc[0]
        flag2 = result.loc[result[PID_COL] == 2, CRITERION_FLAG].iloc[0]

        self.assertTrue(flag1)
        self.assertTrue(flag2)

    def test_extract_most_recent_value(self):
        """Test that most recent value is selected when multiple events per patient."""
        # Create data with multiple events for patient 1
        df = pd.DataFrame(
            {
                PID_COL: [1, 1, 1],
                TIMESTAMP_COL: to_datetime(
                    [
                        "2023-01-01",
                        "2023-02-01",
                        "2023-03-01",
                    ]
                ),
                NUMERIC_VALUE: [5.0, 10.0, 7.5],
                FINAL_MASK: [True, True, True],
            }
        )

        flag_df = pd.DataFrame({PID_COL: [1], CRITERION_FLAG: [True]})

        result = extract_numeric_values(df, flag_df, min_value=5.0, max_value=10.0)

        # Should get the most recent value (7.5 from 2023-03-01)
        val = result.loc[result[PID_COL] == 1, NUMERIC_VALUE].iloc[0]
        self.assertAlmostEqual(val, 7.5)

    def test_no_thresholds(self):
        """Test that all values are accepted when no min/max specified."""
        result = extract_numeric_values(self.basic_df, self.flag_df)

        # Should get most recent values for each patient
        val1 = result.loc[result[PID_COL] == 1, NUMERIC_VALUE].iloc[0]
        val2 = result.loc[result[PID_COL] == 2, NUMERIC_VALUE].iloc[0]

        self.assertAlmostEqual(val1, 7.0)  # Most recent for patient 1
        self.assertAlmostEqual(val2, 10.0)  # Most recent for patient 2

        # Both flags should be True
        self.assertTrue(result[CRITERION_FLAG].all())

    # =================================================================
    # 2. Threshold Filtering Tests
    # =================================================================

    def test_min_value_only(self):
        """Test filtering with only minimum threshold."""
        result = extract_numeric_values(self.basic_df, self.flag_df, min_value=8.0)

        # Patient 1: most recent is 7.0 (< 8.0) - should be excluded
        # Patient 2: most recent is 10.0 (>= 8.0) - should be included
        flag1 = result.loc[result[PID_COL] == 1, CRITERION_FLAG].iloc[0]
        flag2 = result.loc[result[PID_COL] == 2, CRITERION_FLAG].iloc[0]

        self.assertFalse(flag1)
        self.assertTrue(flag2)

        # Patient 1 should have NaN value
        val1 = result.loc[result[PID_COL] == 1, NUMERIC_VALUE].iloc[0]
        self.assertTrue(pd.isna(val1))

        # Patient 2 should have value 10.0
        val2 = result.loc[result[PID_COL] == 2, NUMERIC_VALUE].iloc[0]
        self.assertAlmostEqual(val2, 10.0)

    def test_max_value_only(self):
        """Test filtering with only maximum threshold."""
        result = extract_numeric_values(self.basic_df, self.flag_df, max_value=7.5)

        # Patient 1: most recent is 7.0 (<= 7.5) - should be included
        # Patient 2: most recent is 10.0 (> 7.5) - should check if any value <= 7.5
        val1 = result.loc[result[PID_COL] == 1, NUMERIC_VALUE].iloc[0]
        flag1 = result.loc[result[PID_COL] == 1, CRITERION_FLAG].iloc[0]

        self.assertAlmostEqual(val1, 7.0)
        self.assertTrue(flag1)

        # Patient 2 has 8.0 most recently, but need to check the function behavior
        # According to function logic, it filters first then gets most recent
        # So patient 2 would have no value in range (both 8.0 and 10.0 > 7.5)
        flag2 = result.loc[result[PID_COL] == 2, CRITERION_FLAG].iloc[0]
        self.assertFalse(flag2)

    def test_both_thresholds(self):
        """Test filtering with both min and max thresholds."""
        result = extract_numeric_values(
            self.basic_df, self.flag_df, min_value=7.0, max_value=8.0
        )

        # Patient 1: value 7.0 is in range [7.0, 8.0]
        val1 = result.loc[result[PID_COL] == 1, NUMERIC_VALUE].iloc[0]
        flag1 = result.loc[result[PID_COL] == 1, CRITERION_FLAG].iloc[0]

        self.assertAlmostEqual(val1, 7.0)
        self.assertTrue(flag1)

        # Patient 2: value 8.0 is in range [7.0, 8.0]
        val2 = result.loc[result[PID_COL] == 2, NUMERIC_VALUE].iloc[0]
        flag2 = result.loc[result[PID_COL] == 2, CRITERION_FLAG].iloc[0]

        self.assertAlmostEqual(val2, 8.0)
        self.assertTrue(flag2)

    def test_values_below_threshold(self):
        """Test that values below min_value are excluded."""
        result = extract_numeric_values(self.basic_df, self.flag_df, min_value=15.0)

        # All values are below 15.0, so both patients should have False flags
        self.assertTrue(result[CRITERION_FLAG].eq(False).all())
        self.assertTrue(result[NUMERIC_VALUE].isna().all())

    def test_values_above_threshold(self):
        """Test that values above max_value are excluded."""
        result = extract_numeric_values(self.basic_df, self.flag_df, max_value=3.0)

        # All values are above 3.0, so both patients should have False flags
        self.assertTrue(result[CRITERION_FLAG].eq(False).all())
        self.assertTrue(result[NUMERIC_VALUE].isna().all())

    # =================================================================
    # 3. Edge Cases
    # =================================================================

    def test_empty_dataframe(self):
        """Test with no events in input DataFrame."""
        empty_df = pd.DataFrame(
            columns=[PID_COL, TIMESTAMP_COL, NUMERIC_VALUE, FINAL_MASK]
        )
        empty_df[FINAL_MASK] = empty_df[FINAL_MASK].astype(bool)

        result = extract_numeric_values(empty_df, self.flag_df)

        # Should return flag_df with NUMERIC_VALUE column set to None
        self.assertEqual(result.shape[0], 2)
        self.assertTrue(result[CRITERION_FLAG].eq(False).all())
        self.assertTrue(result[NUMERIC_VALUE].isna().all())

    def test_no_matching_values(self):
        """Test when events exist but none in threshold range."""
        result = extract_numeric_values(
            self.basic_df, self.flag_df, min_value=20.0, max_value=30.0
        )

        # No values in range [20.0, 30.0]
        self.assertEqual(result.shape[0], 2)
        self.assertTrue(result[CRITERION_FLAG].eq(False).all())
        self.assertTrue(result[NUMERIC_VALUE].isna().all())

    def test_missing_numeric_values(self):
        """Test when FINAL_MASK=True but NUMERIC_VALUE is NaN."""
        df = pd.DataFrame(
            {
                PID_COL: [1, 2],
                TIMESTAMP_COL: to_datetime(["2023-01-01", "2023-01-02"]),
                NUMERIC_VALUE: [None, None],
                FINAL_MASK: [True, True],
            }
        )

        result = extract_numeric_values(df, self.flag_df)

        # No valid numeric values, so flags should be False
        self.assertTrue(result[CRITERION_FLAG].eq(False).all())
        self.assertTrue(result[NUMERIC_VALUE].isna().all())

    def test_final_mask_false(self):
        """Test that events with FINAL_MASK=False are ignored."""
        df = pd.DataFrame(
            {
                PID_COL: [1, 1, 2, 2],
                TIMESTAMP_COL: to_datetime(
                    [
                        "2023-01-01",
                        "2023-02-01",
                        "2023-01-15",
                        "2023-02-15",
                    ]
                ),
                NUMERIC_VALUE: [5.0, 7.0, 8.0, 10.0],
                FINAL_MASK: [False, True, False, True],
            }
        )

        result = extract_numeric_values(df, self.flag_df)

        # Only events with FINAL_MASK=True should be considered
        # Patient 1: only 7.0 (from 2023-02-01)
        # Patient 2: only 10.0 (from 2023-02-15)
        val1 = result.loc[result[PID_COL] == 1, NUMERIC_VALUE].iloc[0]
        val2 = result.loc[result[PID_COL] == 2, NUMERIC_VALUE].iloc[0]

        self.assertAlmostEqual(val1, 7.0)
        self.assertAlmostEqual(val2, 10.0)

    # =================================================================
    # 4. extract_value Parameter Tests
    # =================================================================

    def test_extract_value_with_thresholds(self):
        """Test that raw values are extracted when extract_value=True, but flag respects thresholds."""
        df = pd.DataFrame(
            {
                PID_COL: [1, 1, 2],
                TIMESTAMP_COL: to_datetime(
                    [
                        "2023-01-01",
                        "2023-02-01",
                        "2023-01-15",
                    ]
                ),
                NUMERIC_VALUE: [5.0, 12.0, 8.0],
                FINAL_MASK: [True, True, True],
            }
        )

        result = extract_numeric_values(
            df, self.flag_df, min_value=6.0, max_value=9.0, extract_value=True
        )

        # Patient 1: most recent value is 12.0 (out of range [6, 9])
        # Flag should be False, but numeric_value should be 12.0
        flag1 = result.loc[result[PID_COL] == 1, CRITERION_FLAG].iloc[0]
        val1 = result.loc[result[PID_COL] == 1, NUMERIC_VALUE].iloc[0]

        self.assertFalse(flag1)
        self.assertAlmostEqual(val1, 12.0)

        # Patient 2: most recent value is 8.0 (in range [6, 9])
        # Flag should be True, and numeric_value should be 8.0
        flag2 = result.loc[result[PID_COL] == 2, CRITERION_FLAG].iloc[0]
        val2 = result.loc[result[PID_COL] == 2, NUMERIC_VALUE].iloc[0]

        self.assertTrue(flag2)
        self.assertAlmostEqual(val2, 8.0)

    def test_extract_value_without_thresholds(self):
        """Test that raw values are extracted when extract_value=True without thresholds."""
        result = extract_numeric_values(self.basic_df, self.flag_df, extract_value=True)

        # Should extract most recent values without filtering
        val1 = result.loc[result[PID_COL] == 1, NUMERIC_VALUE].iloc[0]
        val2 = result.loc[result[PID_COL] == 2, NUMERIC_VALUE].iloc[0]

        self.assertAlmostEqual(val1, 7.0)
        self.assertAlmostEqual(val2, 10.0)

        # Both flags should be True
        self.assertTrue(result[CRITERION_FLAG].all())

    # =================================================================
    # 5. Mixed Patient Scenarios
    # =================================================================

    def test_mixed_patients(self):
        """Test scenario where some patients have valid values, others don't."""
        df = pd.DataFrame(
            {
                PID_COL: [1, 1, 2, 2, 3],
                TIMESTAMP_COL: to_datetime(
                    [
                        "2023-01-01",
                        "2023-02-01",
                        "2023-01-15",
                        "2023-02-15",
                        "2023-01-20",
                    ]
                ),
                NUMERIC_VALUE: [5.0, 7.0, 15.0, 20.0, 8.0],
                FINAL_MASK: [True, True, True, True, True],
            }
        )

        flag_df = pd.DataFrame({PID_COL: [1, 2, 3], CRITERION_FLAG: [True, True, True]})

        result = extract_numeric_values(df, flag_df, min_value=6.0, max_value=10.0)

        # Patient 1: has value 7.0 in range
        flag1 = result.loc[result[PID_COL] == 1, CRITERION_FLAG].iloc[0]
        val1 = result.loc[result[PID_COL] == 1, NUMERIC_VALUE].iloc[0]
        self.assertTrue(flag1)
        self.assertAlmostEqual(val1, 7.0)

        # Patient 2: most recent values are out of range (15.0, 20.0)
        flag2 = result.loc[result[PID_COL] == 2, CRITERION_FLAG].iloc[0]
        self.assertFalse(flag2)
        self.assertTrue(
            pd.isna(result.loc[result[PID_COL] == 2, NUMERIC_VALUE].iloc[0])
        )

        # Patient 3: has value 8.0 in range
        flag3 = result.loc[result[PID_COL] == 3, CRITERION_FLAG].iloc[0]
        val3 = result.loc[result[PID_COL] == 3, NUMERIC_VALUE].iloc[0]
        self.assertTrue(flag3)
        self.assertAlmostEqual(val3, 8.0)

    def test_multiple_patients_different_timestamps(self):
        """Test that each patient gets their most recent value independently."""
        df = pd.DataFrame(
            {
                PID_COL: [1, 1, 1, 2, 2, 3],
                TIMESTAMP_COL: to_datetime(
                    [
                        "2023-01-01",
                        "2023-03-01",
                        "2023-02-01",
                        "2023-01-01",
                        "2023-04-01",
                        "2023-05-01",
                    ]
                ),
                NUMERIC_VALUE: [5.0, 8.0, 6.0, 7.0, 9.0, 10.0],
                FINAL_MASK: [True, True, True, True, True, True],
            }
        )

        flag_df = pd.DataFrame({PID_COL: [1, 2, 3], CRITERION_FLAG: [True, True, True]})

        result = extract_numeric_values(df, flag_df)

        # Patient 1: most recent is 2023-03-01 with value 8.0
        val1 = result.loc[result[PID_COL] == 1, NUMERIC_VALUE].iloc[0]
        self.assertAlmostEqual(val1, 8.0)

        # Patient 2: most recent is 2023-04-01 with value 9.0
        val2 = result.loc[result[PID_COL] == 2, NUMERIC_VALUE].iloc[0]
        self.assertAlmostEqual(val2, 9.0)

        # Patient 3: only one event at 2023-05-01 with value 10.0
        val3 = result.loc[result[PID_COL] == 3, NUMERIC_VALUE].iloc[0]
        self.assertAlmostEqual(val3, 10.0)


if __name__ == "__main__":
    unittest.main()
