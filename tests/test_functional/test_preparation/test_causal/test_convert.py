import unittest

import pandas as pd

from corebehrt.constants.causal.data import END_COL, START_COL
from corebehrt.constants.data import ABSPOS_COL, PID_COL
from corebehrt.functional.preparation.causal.convert import abspos_to_binary_outcome


class TestAbsposToBinaryOutcome(unittest.TestCase):
    """Test cases for the abspos_to_binary_outcome function."""

    def test_abspos_to_binary_outcome_basic(self):
        """Test basic functionality with outcomes within follow-up periods."""
        # 1. Setup test data
        follow_ups = pd.DataFrame(
            {PID_COL: [1, 2, 3], START_COL: [100, 200, 300], END_COL: [400, 500, 600]}
        )

        outcomes = pd.DataFrame(
            {
                PID_COL: [1, 1, 2, 3],
                ABSPOS_COL: [150, 350, 250, 450],  # All within follow-up periods
            }
        )

        # 2. Execute function
        result = abspos_to_binary_outcome(follow_ups, outcomes)

        # 3. Expected results:
        # Patient 1: outcomes at 150, 350 (both within 100-400) → 1
        # Patient 2: outcome at 250 (within 200-500) → 1
        # Patient 3: outcome at 450 (within 300-600) → 1

        # 4. Assertions
        expected = pd.Series([1, 1, 1], index=[1, 2, 3], name="has_outcome", dtype=int)
        expected.index.name = PID_COL
        pd.testing.assert_series_equal(result, expected)

    def test_abspos_to_binary_outcome_outside_followup(self):
        """Test with outcomes outside follow-up periods."""
        # 1. Setup test data
        follow_ups = pd.DataFrame(
            {PID_COL: [1, 2, 3], START_COL: [100, 200, 300], END_COL: [400, 500, 600]}
        )

        outcomes = pd.DataFrame(
            {
                PID_COL: [1, 2, 3, 3],
                ABSPOS_COL: [
                    50,
                    550,
                    350,
                    650,
                ],  # 50, 550, 650 outside follow-up periods
            }
        )

        # 2. Execute function
        result = abspos_to_binary_outcome(follow_ups, outcomes)

        # 3. Expected results:
        # Patient 1: outcome at 50 (before start 100) → 0
        # Patient 2: outcome at 550 (after end 500) → 0
        # Patient 3: outcomes at 350 (within 300-600) and 650 (after end) → 1 (has one within)

        # 4. Assertions
        expected = pd.Series([0, 0, 1], index=[1, 2, 3], name="has_outcome", dtype=int)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_abspos_to_binary_outcome_no_outcomes(self):
        """Test with patients who have no outcomes."""
        # 1. Setup test data
        follow_ups = pd.DataFrame(
            {PID_COL: [1, 2, 3], START_COL: [100, 200, 300], END_COL: [400, 500, 600]}
        )

        outcomes = pd.DataFrame(
            {
                PID_COL: [1],  # Only patient 1 has outcomes
                ABSPOS_COL: [150],
            }
        )

        # 2. Execute function
        result = abspos_to_binary_outcome(follow_ups, outcomes)

        # 3. Expected results:
        # Patient 1: outcome at 150 (within 100-400) → 1
        # Patient 2: no outcomes → 0
        # Patient 3: no outcomes → 0

        # 4. Assertions
        expected = pd.Series([1, 0, 0], index=[1, 2, 3], name="has_outcome", dtype=int)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_abspos_to_binary_outcome_boundary_conditions(self):
        """Test boundary conditions (outcomes exactly at start/end times)."""
        # 1. Setup test data
        follow_ups = pd.DataFrame(
            {
                PID_COL: [1, 2, 3, 4],
                START_COL: [100, 200, 300, 400],
                END_COL: [400, 500, 600, 700],
            }
        )

        outcomes = pd.DataFrame(
            {
                PID_COL: [1, 2, 3, 4],
                ABSPOS_COL: [100, 500, 300, 700],  # Exactly at boundaries
            }
        )

        # 2. Execute function
        result = abspos_to_binary_outcome(follow_ups, outcomes)

        # 3. Expected results:
        # Patient 1: outcome at 100 (exactly at start) → 0 (not > start)
        # Patient 2: outcome at 500 (exactly at end) → 0 (not < end)
        # Patient 3: outcome at 300 (exactly at start) → 0 (not > start)
        # Patient 4: outcome at 700 (exactly at end) → 0 (not < end)

        # 4. Assertions
        expected = pd.Series(
            [1, 1, 1, 1], index=[1, 2, 3, 4], name="has_outcome", dtype=int
        )
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_abspos_to_binary_outcome_empty_outcomes(self):
        """Test with empty outcomes DataFrame."""
        # 1. Setup test data
        follow_ups = pd.DataFrame(
            {PID_COL: [1, 2, 3], START_COL: [100, 200, 300], END_COL: [400, 500, 600]}
        )

        outcomes = pd.DataFrame(columns=[PID_COL, ABSPOS_COL])  # Empty outcomes

        # 2. Execute function
        result = abspos_to_binary_outcome(follow_ups, outcomes)

        # 3. Expected results:
        # All patients should have 0 (no outcomes)

        # 4. Assertions
        expected = pd.Series([0, 0, 0], index=[1, 2, 3], name="has_outcome", dtype=int)
        expected.index.name = PID_COL
        pd.testing.assert_series_equal(result, expected)

    def test_abspos_to_binary_outcome_empty_followups(self):
        """Test with empty follow_ups DataFrame."""
        # 1. Setup test data
        follow_ups = pd.DataFrame(
            columns=[PID_COL, START_COL, END_COL]
        )  # Empty follow-ups

        outcomes = pd.DataFrame({PID_COL: [1, 2], ABSPOS_COL: [150, 250]})

        # 2. Execute function
        result = abspos_to_binary_outcome(follow_ups, outcomes)

        # 3. Expected results:
        # Empty result since no follow-up periods

        # 4. Assertions
        self.assertEqual(
            len(result), 0, "Should return empty series for empty follow-ups"
        )

    def test_abspos_to_binary_outcome_multiple_outcomes_same_patient(self):
        """Test with multiple outcomes for the same patient."""
        # 1. Setup test data
        follow_ups = pd.DataFrame(
            {PID_COL: [1, 2], START_COL: [100, 200], END_COL: [400, 500]}
        )

        outcomes = pd.DataFrame(
            {
                PID_COL: [1, 1, 1, 2],  # Patient 1 has 3 outcomes
                ABSPOS_COL: [
                    50,
                    150,
                    450,
                    250,
                ],  # 50 and 450 outside follow-up for patient 1
            }
        )

        # 2. Execute function
        result = abspos_to_binary_outcome(follow_ups, outcomes)

        # 3. Expected results:
        # Patient 1: outcomes at 50 (outside), 150 (inside), 450 (outside) → 1 (has one inside)
        # Patient 2: outcome at 250 (inside 200-500) → 1

        # 4. Assertions
        expected = pd.Series([1, 1], index=[1, 2], name="has_outcome", dtype=int)
        expected.index.name = PID_COL
        pd.testing.assert_series_equal(result, expected)

    def test_abspos_to_binary_outcome_float_values(self):
        """Test with float values in abspos."""
        # 1. Setup test data
        follow_ups = pd.DataFrame(
            {PID_COL: [1, 2], START_COL: [100.0, 200.0], END_COL: [400.0, 500.0]}
        )

        outcomes = pd.DataFrame(
            {
                PID_COL: [1, 2],
                ABSPOS_COL: [150.5, 250.75],  # Float values
            }
        )

        # 2. Execute function
        result = abspos_to_binary_outcome(follow_ups, outcomes)

        # 3. Expected results:
        # Patient 1: outcome at 150.5 (within 100.0-400.0) → 1
        # Patient 2: outcome at 250.75 (within 200.0-500.0) → 1

        # 4. Assertions
        expected = pd.Series([1, 1], index=[1, 2], name="has_outcome", dtype=int)
        expected.index.name = PID_COL
        pd.testing.assert_series_equal(result, expected)

    def test_abspos_to_binary_outcome_series_properties(self):
        """Test that the returned series has correct properties."""
        # 1. Setup test data
        follow_ups = pd.DataFrame(
            {PID_COL: [1, 2], START_COL: [100, 200], END_COL: [400, 500]}
        )

        outcomes = pd.DataFrame({PID_COL: [1], ABSPOS_COL: [150]})

        # 2. Execute function
        result = abspos_to_binary_outcome(follow_ups, outcomes)

        # 3. Assertions
        self.assertEqual(result.name, "has_outcome", "Series should have correct name")
        self.assertEqual(result.dtype, int, "Series should have int dtype")
        self.assertTrue(all(result.isin([0, 1])), "All values should be 0 or 1")
        self.assertEqual(
            set(result.index), {1, 2}, "Index should contain all patient IDs"
        )

    def test_additional_patients_in_outcomes(self):
        """Test with patients who have no outcomes."""
        # 1. Setup test data
        follow_ups = pd.DataFrame(
            {PID_COL: [1, 2, 3], START_COL: [100, 200, 300], END_COL: [400, 500, 600]}
        )

        outcomes = pd.DataFrame(
            {PID_COL: [1, 2, 3, 4], ABSPOS_COL: [150, 250, 350, 450]}
        )

        # 2. Execute function
        result = abspos_to_binary_outcome(follow_ups, outcomes)

        # 3. Expected results:
        # Patient 1: outcome at 150 (within 100-400) → 1
        # Patient 2: outcome at 250 (within 200-500) → 1
        # Patient 3: outcome at 350 (within 300-600) → 1
        # Patient 4: not in result

        # 4. Assertions
        expected = pd.Series([1, 1, 1], index=[1, 2, 3], name="has_outcome", dtype=int)
        expected.index.name = PID_COL
        pd.testing.assert_series_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
