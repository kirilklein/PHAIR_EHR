import unittest

import pandas as pd

from corebehrt.constants.data import ABSPOS_COL, PID_COL
from corebehrt.functional.cohort_handling.outcomes import (
    adjust_windows_for_compliance,
    get_binary_outcomes,
)


class TestGetBinaryOutcomes(unittest.TestCase):
    def setUp(self):
        """
        Prepare test data for get_binary_outcomes:
        - index_dates: each patient has a known 'abspos' (e.g., index date in hours).
        - outcomes: multiple rows for some patients, none for others, to test the function thoroughly.
        """
        # Example: 5 patients with index positions
        self.index_dates = pd.DataFrame(
            {
                PID_COL: [1, 2, 3, 4, 5],
                ABSPOS_COL: [100, 200, 300, 400, 500],  # The 'index' positions in hours
            }
        )

        # We'll define outcomes for some patients
        # p1 has an outcome *before* index and one *after*
        # p2 has an outcome *exactly at the boundary*, also one far after
        # p3 => no outcomes
        # p4 => multiple outcomes, some in-window, some out-of-window
        # p5 => no outcomes
        self.outcomes = pd.DataFrame(
            {
                PID_COL: [1, 1, 2, 2, 4, 4],
                ABSPOS_COL: [90, 110, 200, 1000, 390, 405],
            }
        )
        # Explanation:
        #  p1 => outcomes at abspos=90, 110
        #  p2 => outcomes at abspos=200 (exactly at index), 1000
        #  p3 => no outcomes
        #  p4 => outcomes at 390, 405
        #  p5 => not in outcomes => means no outcomes

    # ----------------------------------------------------------------------
    # 1) Test no end follow-up (n_hours_end_follow_up=None)
    # ----------------------------------------------------------------------
    def test_no_end_followup(self):
        """
        With n_hours_start_follow_up=0 and no end limit,
        any outcome at or after the index pos should flag the patient as True.
        """
        # p1 => index=100 => outcome=90 (before index, ignore), 110 (after index => True)
        # p2 => index=200 => outcome=200 (equal to index => included), 1000 => also included
        # p3 => no outcomes => remain False
        # p4 => index=400 => outcomes=390 (before index => ignore), 405 (after index => True)
        # p5 => no outcomes => False
        result = get_binary_outcomes(
            index_dates=self.index_dates,
            outcomes=self.outcomes,
            n_hours_start_follow_up=0,
            n_hours_end_follow_up=None,
        )

        # We expect a Series of length 5 (one boolean per patient)
        self.assertEqual(
            len(result), 5, "Should have 5 patients in the resulting Series."
        )

        # Check individual flags
        self.assertTrue(result[1], "p1 should be True (has outcome at 110 >= 100).")
        self.assertTrue(result[2], "p2 should be True (outcome at 200 >= 200).")
        self.assertFalse(result[3], "p3 has no outcomes => False.")
        self.assertTrue(result[4], "p4 should be True (outcome at 405 >= 400).")
        self.assertFalse(result[5], "p5 has no outcomes => False.")

    # ----------------------------------------------------------------------
    # 2) Test with a start and end window
    # ----------------------------------------------------------------------
    def test_start_and_end_window(self):
        """
        If we define a specific window [start_pos=0, end_pos=50 hours after index],
        only outcomes within that offset from index are considered.
        """
        # For each patient, an outcome must fall in index_abspos + [0..50]
        # p1 => index=100 => window [100..150] => outcomes=90(no), 110(yes) => True
        # p2 => index=200 => window [200..250] => outcomes=200(yes), 1000(no) => True
        # p3 => none => False
        # p4 => index=400 => window [400..450] => outcomes=390(no), 405(yes) => True
        # p5 => none => False
        result = get_binary_outcomes(
            index_dates=self.index_dates,
            outcomes=self.outcomes,
            n_hours_start_follow_up=0,
            n_hours_end_follow_up=50,
        )

        # Expect identical booleans to test_no_end_followup except that p2's outcome at 1000 doesn't matter
        # but p2 is still True from its 200 outcome
        self.assertTrue(result[1])
        self.assertTrue(result[2])
        self.assertFalse(result[3])
        self.assertTrue(result[4])
        self.assertFalse(result[5])

    # ----------------------------------------------------------------------
    # 3) Test a different start_offset
    # ----------------------------------------------------------------------
    def test_positive_start_offset(self):
        """
        If the follow-up starts AFTER the index date (say 5 hours),
        an outcome must be at abspos >= index_abspos + 5 to be counted.
        """
        # p1 => index=100 => window [105..∞]
        #       outcomes => 90 < 100 => no, 110 >= 105 => yes => True
        # p2 => index=200 => outcomes => 200 >=205? no => 1000 >=205 => yes => True
        # p3 => no outcomes => False
        # p4 => index=400 => outcomes => 390(no), 405(yes >=405 => yes) => True
        # p5 => no outcomes => False
        result = get_binary_outcomes(
            index_dates=self.index_dates,
            outcomes=self.outcomes,
            n_hours_start_follow_up=5,
            n_hours_end_follow_up=None,
        )

        # p1 => True, p2 => True, p3 => False, p4 => True, p5 => False
        self.assertTrue(result[1])
        self.assertTrue(result[2])
        self.assertFalse(result[3])
        self.assertTrue(result[4])
        self.assertFalse(result[5])

    # ----------------------------------------------------------------------
    # 4) Test scenario where end_pos excludes borderline outcome
    # ----------------------------------------------------------------------
    def test_exclude_borderline_outcome(self):
        """
        If end_pos is smaller than the outcome offset, that outcome won't count.
        Let's choose end_pos=0 => must be exactly at the index (rel_pos=0) to count.
        """
        # p1 => index=100 => outcomes=90(rel_pos=-10 => no), 110(rel_pos=10 => no) => no => False
        # p2 => index=200 => outcomes=200(rel_pos=0 => yes), 1000(rel_pos=800 => no) => True
        # p3 => no outcomes => False
        # p4 => index=400 => outcomes=390(rel_pos=-10 => no), 405(rel_pos=5 => no) => False
        # p5 => no outcomes => False
        result = get_binary_outcomes(
            index_dates=self.index_dates,
            outcomes=self.outcomes,
            n_hours_start_follow_up=0,
            n_hours_end_follow_up=0,
        )

        self.assertFalse(result[1], "p1 has no outcome exactly at index=100.")
        self.assertTrue(result[2], "p2 has outcome exactly at index=200.")
        self.assertFalse(result[3])
        self.assertFalse(result[4])
        self.assertFalse(result[5])

    # ----------------------------------------------------------------------
    # 5) Test no outcomes at all
    # ----------------------------------------------------------------------
    def test_no_outcomes_data(self):
        """If outcomes DataFrame is empty, every patient is False."""
        empty_outcomes = pd.DataFrame(columns=[PID_COL, "abspos"])
        result = get_binary_outcomes(
            index_dates=self.index_dates, outcomes=empty_outcomes
        )
        self.assertEqual(len(result), 5, "Should still have one entry per patient.")
        self.assertFalse(result.any(), "All should be False if no outcomes exist.")

    # ----------------------------------------------------------------------
    # 6) Test with extra patients in outcomes not in index_dates
    # ----------------------------------------------------------------------
    def test_extra_patients_in_outcomes(self):
        """
        If outcomes contains PIDs not in index_dates,
        they should be ignored and not appear in the final result.
        """
        extra = pd.DataFrame({PID_COL: [6, 7], "abspos": [100, 200]})
        new_outcomes = pd.concat([self.outcomes, extra], ignore_index=True)
        # No change expected for p1..p5 results because p6,p7 aren't in index_dates
        result = get_binary_outcomes(
            index_dates=self.index_dates, outcomes=new_outcomes
        )
        self.assertEqual(len(result), 5, "We only expect p1..p5 in the result.")
        self.assertIn(1, result.index)
        self.assertNotIn(6, result.index)

    # ----------------------------------------------------------------------
    # 7) Test multiple outcomes for same patient
    # ----------------------------------------------------------------------
    def test_multiple_outcomes_for_one_patient(self):
        """
        If a patient has multiple outcomes, as soon as one falls in range,
        that patient should be marked True.
        """
        # We'll create a scenario for p3 with multiple outcomes, some out-of-window, some in-window
        # We'll define start=0, end=10
        new_outcomes = self.outcomes.copy()
        # Add outcomes for p3 => index=300 => we want to test outcomes at 295, 305, 310
        # Let's put them in new_outcomes
        extra_rows = pd.DataFrame({PID_COL: [3, 3, 3], "abspos": [295, 305, 310]})
        new_outcomes = pd.concat([new_outcomes, extra_rows], ignore_index=True)

        # Now with end=10 => p3 => index=300 => valid window=[300..310]
        #  295 => rel_pos=-5 => out
        #  305 => rel_pos=5 => in
        #  310 => rel_pos=10 => in
        # => p3 => True
        result = get_binary_outcomes(
            index_dates=self.index_dates,
            outcomes=new_outcomes,
            n_hours_start_follow_up=0,
            n_hours_end_follow_up=10,
        )
        self.assertTrue(
            result[3],
            "p3 should be True because it has an outcome at abspos=305 or 310.",
        )


class TestAdjustWindowsForCompliance(unittest.TestCase):
    def setUp(self):
        """Prepare test data for adjust_windows_for_compliance"""
        # Create merged DataFrame with outcomes and index dates
        self.merged = pd.DataFrame(
            {
                PID_COL: [1, 1, 1, 2, 2, 2, 3, 3, 4, 4],
                ABSPOS_COL: [110, 115, 120, 205, 210, 215, 310, 320, 410, 420],
                "index_abspos": [100, 100, 100, 200, 200, 200, 300, 300, 400, 400],
                "rel_pos": [10, 15, 20, 5, 10, 15, 10, 20, 10, 20],
            }
        )

        # Create exposures DataFrame
        self.exposures = pd.DataFrame(
            {
                PID_COL: [1, 1, 2],
                ABSPOS_COL: [
                    105,
                    115,
                    205,
                ],  # Last exposures: subject 1 at 115, subject 2 at 205
            }
        )

        # Create index_date_matching DataFrame with multiple controls per exposed
        self.index_date_matching = pd.DataFrame(
            {
                "exposed_subject_id": [1, 1, 2, 2],
                "control_subject_id": [
                    3,
                    4,
                    5,
                    6,
                ],  # subjects 3,4 control for 1; subjects 5,6 control for 2
            }
        )

    def test_compliance_window_basic(self):
        """Test basic compliance window adjustment"""
        n_hours_compliance = 5
        result = adjust_windows_for_compliance(
            self.merged, self.exposures, self.index_date_matching, n_hours_compliance
        )

        # Expected logic:
        # Exposed subject 1: last exposure at 115, window ends at 120 (115+5)
        #   - outcomes at 110, 115, 120: all <= 120, so True, True, True
        # Exposed subject 2: last exposure at 205, window ends at 210 (205+5)
        #   - outcomes at 205, 210, 215: 205<=210 (True), 210<=210 (True), 215>210 (False)
        # Control subjects 3,4: inherit subject 1's window (end at 120)
        #   - outcomes at 310, 320: both > 120, so False, False
        #   - outcomes at 410, 420: both > 120, so False, False

        expected = pd.Series(
            [True, True, True, True, True, False, False, False, False, False]
        )
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_missing_required_inputs(self):
        """Test that appropriate errors are raised for missing inputs"""
        with self.assertRaises(ValueError):
            adjust_windows_for_compliance(
                self.merged, None, self.index_date_matching, 5
            )

        with self.assertRaises(ValueError):
            adjust_windows_for_compliance(self.merged, self.exposures, None, 5)

    def test_subjects_without_exposures(self):
        """Test handling of subjects without exposures"""
        # Add a subject with no exposures (subject 7)
        extended_merged = pd.concat(
            [
                self.merged,
                pd.DataFrame(
                    {
                        PID_COL: [7],
                        ABSPOS_COL: [510],
                        "index_abspos": [500],
                        "rel_pos": [10],
                    }
                ),
            ],
            ignore_index=True,
        )

        result = adjust_windows_for_compliance(
            extended_merged, self.exposures, self.index_date_matching, 5
        )

        # Subject 7 should be True as it has no exposures (no compliance restriction)
        self.assertTrue(result.iloc[-1])

    def test_multiple_controls_per_exposed(self):
        """Test that multiple controls get the same window as their exposed subject"""
        n_hours_compliance = 5
        result = adjust_windows_for_compliance(
            self.merged, self.exposures, self.index_date_matching, n_hours_compliance
        )

        # The test data has subjects 3,4 as controls in index_date_matching,
        # but their actual outcome times in merged are much later than the compliance windows
        # So they should all be False based on the time comparison

        # Verify the pattern matches our expectation:
        # Subjects 1,2 (exposed): based on their own exposures
        # Subjects 3,4 (controls): should inherit from their exposed subjects but fail due to timing
        expected_pattern = [
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
        ]

        for i, expected_val in enumerate(expected_pattern):
            self.assertEqual(
                result.iloc[i],
                expected_val,
                f"Index {i} (subject {self.merged.iloc[i][PID_COL]}) should be {expected_val}",
            )


if __name__ == "__main__":
    unittest.main()
