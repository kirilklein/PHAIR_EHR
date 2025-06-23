import unittest
from datetime import datetime

import numpy as np
import pandas as pd

from corebehrt.constants.causal.data import CONTROL_PID_COL, EXPOSED_PID_COL
from corebehrt.constants.data import (
    BIRTHDATE_COL,
    DEATHDATE_COL,
    PID_COL,
    TIMESTAMP_COL,
)
from corebehrt.functional.cohort_handling.advanced.index_dates import (
    draw_index_dates_for_control_with_redraw,
    select_time_eligible_exposed,
)


class TestDrawIndexDatesForControl(unittest.TestCase):
    def setUp(self):
        """Set up test data for draw_index_dates_for_control tests"""
        # Set random seed for reproducible tests
        np.random.seed(42)

        # Create exposed index dates
        self.exposed_index_dates = pd.DataFrame(
            {
                PID_COL: ["exp_1", "exp_2", "exp_3"],
                TIMESTAMP_COL: [
                    datetime(2015, 1, 1),
                    datetime(2015, 6, 1),
                    datetime(2015, 12, 1),
                ],
            }
        )

        # Create control patient info without death dates (alive patients)
        self.patients_info_alive = pd.DataFrame(
            {
                PID_COL: ["ctrl_1", "ctrl_2", "ctrl_3"],
                DEATHDATE_COL: [pd.NaT, pd.NaT, pd.NaT],  # All alive
                BIRTHDATE_COL: [
                    datetime(1990, 1, 1),
                    datetime(1990, 1, 1),
                    datetime(1990, 1, 1),
                ],
            }
        )

        # Create control patient info with some death dates
        self.patients_info_mixed = pd.DataFrame(
            {
                PID_COL: ["ctrl_1", "ctrl_2", "ctrl_3", "ctrl_4"],
                DEATHDATE_COL: [
                    pd.NaT,  # Alive
                    datetime(2014, 6, 1),  # Died before all exposed index dates
                    datetime(2015, 8, 1),  # Died after some exposed index dates
                    pd.NaT,  # Alive
                ],
                BIRTHDATE_COL: [
                    datetime(1990, 1, 1),
                    datetime(1990, 1, 1),
                    datetime(1990, 1, 1),
                    datetime(2016, 1, 1),
                ],  # born after all exposed index dates
            }
        )

    def test_basic_functionality_all_alive(self):
        """Test basic functionality with all control patients alive"""
        control_pids = ["ctrl_1", "ctrl_2", "ctrl_3"]

        control_index_dates, exposure_matching = (
            draw_index_dates_for_control_with_redraw(
                control_pids, self.exposed_index_dates, self.patients_info_alive
            )
        )

        # Check return types and structure
        self.assertIsInstance(control_index_dates, pd.DataFrame)
        self.assertIsInstance(exposure_matching, pd.DataFrame)

        # Check that all control patients got index dates (since all alive)
        self.assertEqual(len(control_index_dates), 3)
        self.assertEqual(len(exposure_matching), 3)

        # Check column names
        self.assertIn(PID_COL, control_index_dates.columns)
        self.assertIn(TIMESTAMP_COL, control_index_dates.columns)
        self.assertIn(EXPOSED_PID_COL, exposure_matching.columns)
        self.assertIn(CONTROL_PID_COL, exposure_matching.columns)

        # Check that assigned dates are from exposed patients
        assigned_dates = set(control_index_dates[TIMESTAMP_COL])
        exposed_dates = set(self.exposed_index_dates[TIMESTAMP_COL])
        self.assertTrue(assigned_dates.issubset(exposed_dates))

    def test_death_date_validation(self):
        """Test that patients who die before assigned dates are handled correctly"""
        control_pids = ["ctrl_1", "ctrl_2", "ctrl_3", "ctrl_4"]

        control_index_dates, exposure_matching = (
            draw_index_dates_for_control_with_redraw(
                control_pids, self.exposed_index_dates, self.patients_info_mixed
            )
        )

        # ctrl_2 died on 2014-06-01, before all exposed dates, so should be excluded
        # ctrl_4 born on 2016-01-01, after all exposed dates, so should be excluded
        # Other patients should potentially be included
        self.assertLessEqual(len(control_index_dates), 3)
        self.assertEqual(len(control_index_dates), len(exposure_matching))

        # Check that no assigned dates are after death dates
        for _, row in control_index_dates.iterrows():
            patient_id = row[PID_COL]
            assigned_date = row[TIMESTAMP_COL]

            death_date = self.patients_info_mixed[
                self.patients_info_mixed[PID_COL] == patient_id
            ][DEATHDATE_COL].iloc[0]

            if pd.notna(death_date):
                self.assertLessEqual(assigned_date, death_date)

    def test_empty_control_pids(self):
        """Test handling of empty control patient list"""
        control_index_dates, exposure_matching = (
            draw_index_dates_for_control_with_redraw(
                [], self.exposed_index_dates, self.patients_info_alive
            )
        )

        self.assertEqual(len(control_index_dates), 0)
        self.assertEqual(len(exposure_matching), 0)
        self.assertIn(PID_COL, control_index_dates.columns)
        self.assertIn(TIMESTAMP_COL, control_index_dates.columns)

    def test_empty_exposed_dates(self):
        """Test handling of empty exposed index dates"""
        empty_exposed = pd.DataFrame({PID_COL: [], TIMESTAMP_COL: []})

        control_pids = ["ctrl_1", "ctrl_2"]

        # This should raise an error since there are no exposed dates to sample from
        with self.assertRaises((ValueError, IndexError)):
            draw_index_dates_for_control_with_redraw(
                control_pids, empty_exposed, self.patients_info_alive
            )

    def test_single_exposed_patient(self):
        """Test with only one exposed patient (edge case for sampling)"""
        single_exposed = pd.DataFrame(
            {PID_COL: ["exp_1"], TIMESTAMP_COL: [datetime(2015, 6, 1)]}
        )

        control_pids = ["ctrl_1", "ctrl_2", "ctrl_3"]

        control_index_dates, exposure_matching = (
            draw_index_dates_for_control_with_redraw(
                control_pids, single_exposed, self.patients_info_alive
            )
        )

        # All control patients should get the same index date
        unique_dates = control_index_dates[TIMESTAMP_COL].unique()
        self.assertEqual(len(unique_dates), 1)
        self.assertEqual(unique_dates[0], datetime(2015, 6, 1))

    def test_all_control_die_before_exposed_dates(self):
        """Test scenario where all control patients die before any exposed dates"""
        early_death_info = pd.DataFrame(
            {
                PID_COL: ["ctrl_1", "ctrl_2"],
                DEATHDATE_COL: [
                    datetime(2014, 1, 1),  # Dies before all exposed dates
                    datetime(2014, 6, 1),  # Dies before all exposed dates
                ],
                BIRTHDATE_COL: [datetime(1990, 1, 1), datetime(1990, 1, 1)],
            }
        )

        control_pids = ["ctrl_1", "ctrl_2"]

        control_index_dates, exposure_matching = (
            draw_index_dates_for_control_with_redraw(
                control_pids, self.exposed_index_dates, early_death_info
            )
        )

        # Should return empty DataFrames since no valid assignments possible
        self.assertEqual(len(control_index_dates), 0)
        self.assertEqual(len(exposure_matching), 0)

    def test_matching_info_consistency(self):
        """Test that matching information is consistent between return values"""
        control_pids = ["ctrl_1", "ctrl_2", "ctrl_3"]

        control_index_dates, exposure_matching = (
            draw_index_dates_for_control_with_redraw(
                control_pids, self.exposed_index_dates, self.patients_info_alive
            )
        )

        # Check that the control PIDs in both DataFrames match
        control_pids_from_dates = set(control_index_dates[PID_COL])
        control_pids_from_matching = set(exposure_matching[CONTROL_PID_COL])
        self.assertEqual(control_pids_from_dates, control_pids_from_matching)

        # Check that timestamps are consistent
        for _, match_row in exposure_matching.iterrows():
            ctrl_pid = match_row[CONTROL_PID_COL]
            expected_date = match_row[TIMESTAMP_COL]

            actual_date = control_index_dates[control_index_dates[PID_COL] == ctrl_pid][
                TIMESTAMP_COL
            ].iloc[0]

            self.assertEqual(expected_date, actual_date)

    def test_exposed_pids_are_valid(self):
        """Test that exposed PIDs in matching info are from the original exposed list"""
        control_pids = ["ctrl_1", "ctrl_2", "ctrl_3"]

        control_index_dates, exposure_matching = (
            draw_index_dates_for_control_with_redraw(
                control_pids, self.exposed_index_dates, self.patients_info_alive
            )
        )

        exposed_pids_in_matching = set(exposure_matching[EXPOSED_PID_COL])
        original_exposed_pids = set(self.exposed_index_dates[PID_COL])

        self.assertTrue(exposed_pids_in_matching.issubset(original_exposed_pids))


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
        self.assertCountEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
