import unittest
import pandas as pd
from corebehrt.functional.preparation.causal.utils import (
    get_non_compliance_abspos,
    assign_groups_to_followups,
)
from corebehrt.constants.data import PID_COL, ABSPOS_COL
from corebehrt.constants.causal.data import CONTROL_PID_COL, EXPOSED_PID_COL, GROUP_COL


class TestAssignGroupsToFollowups(unittest.TestCase):
    """
    Test suite for the assign_groups_to_followups function.
    """

    def test_standard_many_to_one_matching(self):
        """
        Tests the standard case with one exposed patient having multiple controls,
        plus an unmatched exposed patient.
        """
        # PIDs 1, 2 are controls for exposed patient 10.
        # PID 3 is a control for exposed patient 20.
        # PID 30 is an unmatched exposed patient.
        follow_ups = pd.DataFrame({PID_COL: [1, 2, 3, 10, 20, 30]})
        matching = pd.DataFrame(
            {CONTROL_PID_COL: [1, 2, 3], EXPOSED_PID_COL: [10, 10, 20]}
        )
        result = assign_groups_to_followups(follow_ups, matching)

        expected_groups = {
            1: 10,  # Control -> Exposed
            2: 10,  # Control -> Exposed
            3: 20,  # Control -> Exposed
            10: 10,  # Exposed -> Self
            20: 20,  # Exposed -> Self
            30: 30,  # Unmatched Exposed -> Self
        }

        result_map = result.set_index(PID_COL)[GROUP_COL].to_dict()
        self.assertDictEqual(result_map, expected_groups)
        self.assertTrue(pd.api.types.is_integer_dtype(result[GROUP_COL].dtype))

    def test_one_to_one_matching(self):
        """Tests a simple 1-to-1 matching scenario."""
        follow_ups = pd.DataFrame({PID_COL: [1, 10, 2, 20]})
        matching = pd.DataFrame({CONTROL_PID_COL: [1, 2], EXPOSED_PID_COL: [10, 20]})
        result = assign_groups_to_followups(follow_ups, matching)

        expected = {1: 10, 10: 10, 2: 20, 20: 20}
        result_map = result.set_index(PID_COL)[GROUP_COL].to_dict()
        self.assertDictEqual(result_map, expected)

    def test_empty_matching_file(self):
        """Tests that if no matching is provided, each patient is their own group."""
        follow_ups = pd.DataFrame({PID_COL: [10, 20, 30]})
        matching = pd.DataFrame(columns=[CONTROL_PID_COL, EXPOSED_PID_COL])

        result = assign_groups_to_followups(follow_ups, matching)

        # Each patient's group ID should be their own PID
        pd.testing.assert_series_equal(
            result[GROUP_COL].reset_index(drop=True),
            follow_ups[PID_COL].reset_index(drop=True),
            check_names=False,
        )

    def test_none_matching_file(self):
        """Tests that if the matching DataFrame is None, each patient is their own group."""
        follow_ups = pd.DataFrame({PID_COL: [10, 20, 30]})

        result = assign_groups_to_followups(follow_ups, index_date_matching=None)

        # Each patient's group ID should be their own PID
        pd.testing.assert_series_equal(
            result[GROUP_COL].reset_index(drop=True),
            follow_ups[PID_COL].reset_index(drop=True),
            check_names=False,
        )

    def test_cohort_with_no_matching_controls(self):
        """
        Tests a cohort of only exposed patients, where controls from the matching
        file are not present in the final cohort.
        """
        follow_ups = pd.DataFrame({PID_COL: [10, 20, 30]})
        # Matching file exists but its controls (1, 2) aren't in the follow_ups DataFrame
        matching = pd.DataFrame({CONTROL_PID_COL: [1, 2], EXPOSED_PID_COL: [10, 20]})
        result = assign_groups_to_followups(follow_ups, matching)

        # The mapping has no effect on the present PIDs; each is their own group.
        pd.testing.assert_series_equal(
            result[GROUP_COL].reset_index(drop=True),
            follow_ups[PID_COL].reset_index(drop=True),
            check_names=False,
            check_dtype=False,
        )

    def test_empty_follow_ups_input(self):
        """Tests that an empty cohort DataFrame results in an empty DataFrame."""
        follow_ups = pd.DataFrame({PID_COL: pd.Series(dtype=int)})
        matching = pd.DataFrame({CONTROL_PID_COL: [1, 2], EXPOSED_PID_COL: [10, 20]})

        result = assign_groups_to_followups(follow_ups, matching)

        self.assertTrue(result.empty)
        self.assertIn(GROUP_COL, result.columns)


class TestGetNonComplianceAbspos(unittest.TestCase):
    """Test cases for the get_non_compliance_abspos function."""

    def test_get_non_compliance_abspos_basic(self):
        """Test basic functionality with multiple patients and exposures."""
        # 1. Setup sample data
        exposures = pd.DataFrame(
            {PID_COL: [1, 1, 2, 2, 3], ABSPOS_COL: [100, 200, 150, 300, 250]}
        )
        n_hours_compliance = 24

        # 2. Execute function
        result = get_non_compliance_abspos(exposures, n_hours_compliance)

        # 3. Expected results:
        # Patient 1: max(100, 200) + 24 = 200 + 24 = 224
        # Patient 2: max(150, 300) + 24 = 300 + 24 = 324
        # Patient 3: max(250) + 24 = 250 + 24 = 274
        expected = {1: 224, 2: 324, 3: 274}

        # 4. Assertions
        self.assertEqual(result, expected)

    def test_get_non_compliance_abspos_single_exposure(self):
        """Test with patients having only one exposure each."""
        # 1. Setup sample data
        exposures = pd.DataFrame({PID_COL: [1, 2, 3], ABSPOS_COL: [100, 200, 300]})
        n_hours_compliance = 12

        # 2. Execute function
        result = get_non_compliance_abspos(exposures, n_hours_compliance)

        # 3. Expected results:
        # Patient 1: 100 + 12 = 112
        # Patient 2: 200 + 12 = 212
        # Patient 3: 300 + 12 = 312
        expected = {1: 112, 2: 212, 3: 312}

        # 4. Assertions
        self.assertEqual(result, expected)

    def test_get_non_compliance_abspos_zero_compliance(self):
        """Test with zero compliance hours."""
        # 1. Setup sample data
        exposures = pd.DataFrame({PID_COL: [1, 1, 2], ABSPOS_COL: [100, 200, 150]})
        n_hours_compliance = 0

        # 2. Execute function
        result = get_non_compliance_abspos(exposures, n_hours_compliance)

        # 3. Expected results:
        # Patient 1: max(100, 200) + 0 = 200
        # Patient 2: max(150) + 0 = 150
        expected = {1: 200, 2: 150}

        # 4. Assertions
        self.assertEqual(result, expected)

    def test_get_non_compliance_abspos_negative_compliance(self):
        """Test with negative compliance hours."""
        # 1. Setup sample data
        exposures = pd.DataFrame({PID_COL: [1, 2], ABSPOS_COL: [100, 200]})
        n_hours_compliance = -10

        # 2. Execute function
        result = get_non_compliance_abspos(exposures, n_hours_compliance)

        # 3. Expected results:
        # Patient 1: 100 + (-10) = 90
        # Patient 2: 200 + (-10) = 190
        expected = {1: 90, 2: 190}

        # 4. Assertions
        self.assertEqual(result, expected)

    def test_get_non_compliance_abspos_float_values(self):
        """Test with float values in abspos and compliance hours."""
        # 1. Setup sample data
        exposures = pd.DataFrame(
            {PID_COL: [1, 1, 2], ABSPOS_COL: [100.5, 200.75, 150.25]}
        )
        n_hours_compliance = 12.5

        # 2. Execute function
        result = get_non_compliance_abspos(exposures, n_hours_compliance)

        # 3. Expected results:
        # Patient 1: max(100.5, 200.75) + 12.5 = 200.75 + 12.5 = 213.25
        # Patient 2: max(150.25) + 12.5 = 150.25 + 12.5 = 162.75
        expected = {1: 213.25, 2: 162.75}

        # 4. Assertions
        self.assertEqual(result, expected)

    def test_get_non_compliance_abspos_empty_input(self):
        """Test with empty DataFrame."""
        # 1. Setup empty data
        exposures = pd.DataFrame(columns=[PID_COL, ABSPOS_COL])
        n_hours_compliance = 24

        # 2. Execute function
        result = get_non_compliance_abspos(exposures, n_hours_compliance)

        # 3. Assertions
        self.assertEqual(result, {}, "Empty input should return empty dictionary")

    def test_get_non_compliance_abspos_single_patient(self):
        """Test with single patient having multiple exposures."""
        # 1. Setup sample data
        exposures = pd.DataFrame(
            {
                PID_COL: [1, 1, 1],
                ABSPOS_COL: [100, 300, 200],  # Multiple exposures for patient 1
            }
        )
        n_hours_compliance = 50

        # 2. Execute function
        result = get_non_compliance_abspos(exposures, n_hours_compliance)

        # 3. Expected results:
        # Patient 1: max(100, 300, 200) + 50 = 300 + 50 = 350
        expected = {1: 350}

        # 4. Assertions
        self.assertEqual(result, expected)

    def test_get_non_compliance_abspos_large_numbers(self):
        """Test with large abspos values."""
        # 1. Setup sample data
        exposures = pd.DataFrame({PID_COL: [1, 2], ABSPOS_COL: [1000000, 2000000]})
        n_hours_compliance = 1000

        # 2. Execute function
        result = get_non_compliance_abspos(exposures, n_hours_compliance)

        # 3. Expected results:
        # Patient 1: 1000000 + 1000 = 1001000
        # Patient 2: 2000000 + 1000 = 2001000
        expected = {1: 1001000, 2: 2001000}

        # 4. Assertions
        self.assertEqual(result, expected)

    def test_get_non_compliance_abspos_return_type(self):
        """Test that the function returns correct type and structure."""
        # 1. Setup sample data
        exposures = pd.DataFrame({PID_COL: [1, 2], ABSPOS_COL: [100, 200]})
        n_hours_compliance = 24

        # 2. Execute function
        result = get_non_compliance_abspos(exposures, n_hours_compliance)

        # 3. Assertions
        self.assertIsInstance(result, dict, "Should return a dictionary")
        self.assertTrue(
            all(isinstance(k, int) for k in result.keys()),
            "All keys should be integers",
        )
        self.assertTrue(
            all(isinstance(v, (int, float)) for v in result.values()),
            "All values should be numeric",
        )

    def test_get_non_compliance_abspos_duplicate_abspos(self):
        """Test with duplicate abspos values for the same patient."""
        # 1. Setup sample data
        exposures = pd.DataFrame(
            {
                PID_COL: [1, 1, 1],
                ABSPOS_COL: [100, 100, 200],  # Duplicate 100 for patient 1
            }
        )
        n_hours_compliance = 10

        # 2. Execute function
        result = get_non_compliance_abspos(exposures, n_hours_compliance)

        # 3. Expected results:
        # Patient 1: max(100, 100, 200) + 10 = 200 + 10 = 210
        expected = {1: 210}

        # 4. Assertions
        self.assertEqual(result, expected)  #

    def test_get_non_compliance_abspos_none_compliance(self):
        """Test with None compliance hours (should be treated as infinity)."""
        # 1. Setup sample data
        exposures = pd.DataFrame({PID_COL: [1, 2], ABSPOS_COL: [100, 200]})
        n_hours_compliance = None

        # 2. Execute function
        result = get_non_compliance_abspos(exposures, n_hours_compliance)

        # 3. Expected results:
        # Patient 1: 100 + inf = inf
        # Patient 2: 200 + inf = inf
        expected = {1: float("inf"), 2: float("inf")}

        # 4. Assertions
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
