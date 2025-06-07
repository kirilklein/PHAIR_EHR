import unittest
from datetime import datetime

import pandas as pd

from corebehrt.constants.data import PID_COL, TIMESTAMP_COL
from corebehrt.main_causal.helper.select_cohort_full import select_time_eligible_exposed


class TestSelectTimeEligibleExposed(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        self.time_windows = {
            "min_follow_up": {"days": 365},
            "min_lookback": {"days": 365 * 2},
            "data_start": {"year": 2010, "month": 1, "day": 1},
            "data_end": {"year": 2020, "month": 12, "day": 31},
        }

        # Create test index_dates DataFrame
        self.index_dates = pd.DataFrame(
            {
                PID_COL: [1, 2, 3, 4, 5],
                TIMESTAMP_COL: [
                    datetime(2015, 6, 15),  # Should pass both criteria
                    datetime(2011, 6, 15),  # Should fail lookback (too early)
                    datetime(2019, 6, 15),  # Should fail follow-up (too late)
                    datetime(2009, 6, 15),  # Should fail lookback (before data_start)
                    datetime(2013, 6, 15),  # Should pass both criteria
                ],
            }
        )

    def test_patients_meeting_all_criteria(self):
        """Test that patients meeting both follow-up and lookback criteria are returned"""
        result = select_time_eligible_exposed(self.index_dates, self.time_windows)
        expected = [1, 3, 5]  # Patients 1, 3, and 5 should meet all criteria
        self.assertCountEqual(result, expected)

    def test_insufficient_followup(self):
        """Test that patients with insufficient follow-up time are excluded"""
        # Create DataFrame with patient having index date too close to data_end
        late_index_dates = pd.DataFrame(
            {
                PID_COL: [1],
                TIMESTAMP_COL: [
                    datetime(2020, 6, 15)
                ],  # Less than 1 year before data_end
            }
        )

        result = select_time_eligible_exposed(late_index_dates, self.time_windows)
        self.assertEqual(len(result), 0)

    def test_insufficient_lookback(self):
        """Test that patients with insufficient lookback time are excluded"""
        # Create DataFrame with patient having index date too close to data_start
        early_index_dates = pd.DataFrame(
            {
                PID_COL: [1],
                TIMESTAMP_COL: [
                    datetime(2011, 6, 15)
                ],  # Less than 2 years after data_start
            }
        )

        result = select_time_eligible_exposed(early_index_dates, self.time_windows)
        self.assertEqual(len(result), 0)

    def test_empty_dataframe(self):
        """Test handling of empty input DataFrame"""
        empty_df = pd.DataFrame({PID_COL: [], TIMESTAMP_COL: []})
        result = select_time_eligible_exposed(empty_df, self.time_windows)
        self.assertEqual(len(result), 0)

    def test_duplicate_patients(self):
        """Test that duplicate patient IDs raise an error"""
        duplicate_index_dates = pd.DataFrame(
            {
                PID_COL: [1, 1, 2, 2],  # Duplicate patient IDs
                TIMESTAMP_COL: [
                    datetime(2015, 6, 15),
                    datetime(2015, 7, 15),  # Different timestamp for same patient
                    datetime(2015, 6, 15),
                    datetime(2015, 8, 15),  # Different timestamp for same patient
                ],
            }
        )

        # Should raise an error due to duplicate patient IDs
        with self.assertRaises((ValueError, KeyError, Exception)):
            select_time_eligible_exposed(duplicate_index_dates, self.time_windows)

    def test_edge_case_exact_boundaries(self):
        """Test patients with index dates exactly at the boundary conditions"""
        # Patient with exactly min_follow_up before data_end
        # Patient with exactly min_lookback after data_start
        boundary_index_dates = pd.DataFrame(
            {
                PID_COL: [1, 2],
                TIMESTAMP_COL: [
                    datetime(2019, 12, 31),  # Exactly 1 year before data_end
                    datetime(2012, 1, 1),  # Exactly 2 years after data_start
                ],
            }
        )

        result = select_time_eligible_exposed(boundary_index_dates, self.time_windows)
        expected = [1, 2]  # Both should pass (using <= and >= comparisons)
        self.assertCountEqual(result, expected)

    def test_different_time_windows(self):
        """Test with different time window configurations"""
        strict_time_windows = {
            "min_follow_up": {"days": 365 * 3},
            "min_lookback": {"days": 365 * 5},
            "data_start": {"year": 2010, "month": 1, "day": 1},
            "data_end": {"year": 2020, "month": 12, "day": 31},
        }

        result = select_time_eligible_exposed(self.index_dates, strict_time_windows)
        # With stricter criteria, fewer patients should qualify
        # Patient 1 (2015-06-15): needs lookback until 2010-06-15 ✓, follow-up until 2018-06-15 ✓
        # Patient 5 (2013-06-15): needs lookback until 2008-06-15 ✗, follow-up until 2016-06-15 ✓
        expected = [1]
        print(result)
        print(expected)
        self.assertCountEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
