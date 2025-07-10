import unittest
import pandas as pd
from corebehrt.functional.preparation.causal.utils import (
    get_group_dict,
    get_non_compliance_abspos,
)
from corebehrt.constants.causal.data import CONTROL_PID_COL, EXPOSED_PID_COL
from corebehrt.constants.data import PID_COL, ABSPOS_COL


class TestGetGroupDict(unittest.TestCase):
    """Test cases for the get_group_dict function."""

    def test_get_group_dict_basic(self):
        """Test basic functionality with multiple exposed subjects."""
        # 1. Setup sample data
        index_date_matching = pd.DataFrame(
            {CONTROL_PID_COL: [1, 2, 3, 5], EXPOSED_PID_COL: [10, 10, 20, 30]}
        )

        # 2. Execute function
        result_dict = get_group_dict(index_date_matching.copy())

        # 3. Expected results:
        # Group 0: exposed_subject_id=10, control_subject_ids=[1, 2]
        # Group 1: exposed_subject_id=20, control_subject_ids=[3]
        # Group 2: exposed_subject_id=30, control_subject_ids=[5]
        expected_data = {
            1: 0,  # Control for exposed 10
            2: 0,  # Control for exposed 10
            3: 1,  # Control for exposed 20
            5: 2,  # Control for exposed 30
            10: 0,  # Exposed 10
            20: 1,  # Exposed 20
            30: 2,  # Exposed 30
        }

        # 4. Assertions
        self.assertEqual(result_dict, expected_data)

    def test_get_group_dict_single_exposed(self):
        """Test with only one exposed subject."""
        # 1. Setup sample data
        index_date_matching = pd.DataFrame(
            {
                CONTROL_PID_COL: [1, 2, 3],
                EXPOSED_PID_COL: [10, 10, 10],  # All controls for same exposed
            }
        )

        # 2. Execute function
        result_dict = get_group_dict(index_date_matching.copy())

        # 3. Expected results:
        # Group 0: exposed_subject_id=10, control_subject_ids=[1, 2, 3]
        expected_data = {
            1: 0,  # Control for exposed 10
            2: 0,  # Control for exposed 10
            3: 0,  # Control for exposed 10
            10: 0,  # Exposed 10
        }

        # 4. Assertions
        self.assertEqual(result_dict, expected_data)

    def test_get_group_dict_one_to_one_matching(self):
        """Test with one-to-one matching (no multiple controls per exposed)."""
        # 1. Setup sample data
        index_date_matching = pd.DataFrame(
            {
                CONTROL_PID_COL: [1, 2, 3],
                EXPOSED_PID_COL: [10, 20, 30],  # Each exposed has one control
            }
        )

        # 2. Execute function
        result_dict = get_group_dict(index_date_matching.copy())

        # 3. Expected results:
        # Group 0: exposed_subject_id=10, control_subject_id=1
        # Group 1: exposed_subject_id=20, control_subject_id=2
        # Group 2: exposed_subject_id=30, control_subject_id=3
        expected_data = {
            1: 0,  # Control for exposed 10
            2: 1,  # Control for exposed 20
            3: 2,  # Control for exposed 30
            10: 0,  # Exposed 10
            20: 1,  # Exposed 20
            30: 2,  # Exposed 30
        }

        # 4. Assertions
        self.assertEqual(result_dict, expected_data)

    def test_get_group_dict_empty_input(self):
        """Test with empty DataFrame."""
        # 1. Setup empty data
        index_date_matching = pd.DataFrame(columns=[CONTROL_PID_COL, EXPOSED_PID_COL])

        # 2. Execute function
        result_dict = get_group_dict(index_date_matching.copy())

        # 3. Assertions
        self.assertEqual(result_dict, {}, "Empty input should return empty dictionary")

    def test_get_group_dict_single_row(self):
        """Test with single row input."""
        # 1. Setup sample data
        index_date_matching = pd.DataFrame(
            {CONTROL_PID_COL: [1], EXPOSED_PID_COL: [10]}
        )

        # 2. Execute function
        result_dict = get_group_dict(index_date_matching.copy())

        # 3. Expected results:
        # Group 0: exposed_subject_id=10, control_subject_id=1
        expected_data = {
            1: 0,  # Control for exposed 10
            10: 0,  # Exposed 10
        }

        # 4. Assertions
        self.assertEqual(result_dict, expected_data)

    def test_get_group_dict_duplicate_controls(self):
        """Test with duplicate control subjects for same exposed."""
        # 1. Setup sample data
        index_date_matching = pd.DataFrame(
            {
                CONTROL_PID_COL: [1, 1, 2],  # Control 1 appears twice
                EXPOSED_PID_COL: [10, 10, 20],
            }
        )

        # 2. Execute function
        result_dict = get_group_dict(index_date_matching.copy())

        # 3. Expected results:
        # Group 0: exposed_subject_id=10, control_subject_ids=[1] (duplicates removed)
        # Group 1: exposed_subject_id=20, control_subject_ids=[2]
        expected_data = {
            1: 0,  # Control for exposed 10 (duplicate removed)
            2: 1,  # Control for exposed 20
            10: 0,  # Exposed 10
            20: 1,  # Exposed 20
        }

        # 4. Assertions
        self.assertEqual(result_dict, expected_data)

    def test_get_group_dict_large_numbers(self):
        """Test with large subject ID numbers."""
        # 1. Setup sample data
        index_date_matching = pd.DataFrame(
            {
                CONTROL_PID_COL: [1000, 2000, 3000],
                EXPOSED_PID_COL: [10000, 20000, 30000],
            }
        )

        # 2. Execute function
        result_dict = get_group_dict(index_date_matching.copy())

        # 3. Expected results:
        # Group 0: exposed_subject_id=10000, control_subject_id=1000
        # Group 1: exposed_subject_id=20000, control_subject_id=2000
        # Group 2: exposed_subject_id=30000, control_subject_id=3000
        expected_data = {
            1000: 0,  # Control for exposed 10000
            2000: 1,  # Control for exposed 20000
            3000: 2,  # Control for exposed 30000
            10000: 0,  # Exposed 10000
            20000: 1,  # Exposed 20000
            30000: 2,  # Exposed 30000
        }

        # 4. Assertions
        self.assertEqual(result_dict, expected_data)

    def test_get_group_dict_group_numbering(self):
        """Test that group numbers are assigned correctly based on exposed subject order."""
        # 1. Setup sample data with specific exposed subject order
        index_date_matching = pd.DataFrame(
            {
                CONTROL_PID_COL: [1, 2, 3],
                EXPOSED_PID_COL: [30, 10, 20],  # Order: 30, 10, 20
            }
        )

        # 2. Execute function
        result_dict = get_group_dict(index_date_matching.copy())

        # 3. Expected results:
        # ngroup() assigns groups based on sorted unique exposed_subject_id values: 10, 20, 30
        # Group 0: exposed_subject_id=10
        # Group 1: exposed_subject_id=20
        # Group 2: exposed_subject_id=30
        expected_data = {
            1: 2,  # Control for exposed 30 (group 2)
            2: 0,  # Control for exposed 10 (group 0)
            3: 1,  # Control for exposed 20 (group 1)
            10: 0,  # Exposed 10 (group 0)
            20: 1,  # Exposed 20 (group 1)
            30: 2,  # Exposed 30 (group 2)
        }

        # 4. Assertions
        self.assertEqual(result_dict, expected_data)

    def test_get_group_dict_return_type(self):
        """Test that the function returns correct type and structure."""
        # 1. Setup sample data
        index_date_matching = pd.DataFrame(
            {CONTROL_PID_COL: [1, 2], EXPOSED_PID_COL: [10, 20]}
        )

        # 2. Execute function
        result_dict = get_group_dict(index_date_matching.copy())

        # 3. Assertions
        self.assertIsInstance(result_dict, dict, "Should return a dictionary")
        self.assertTrue(
            all(isinstance(k, int) for k in result_dict.keys()),
            "All keys should be integers",
        )
        self.assertTrue(
            all(isinstance(v, int) for v in result_dict.values()),
            "All values should be integers",
        )
        self.assertTrue(
            all(v >= 0 for v in result_dict.values()),
            "All group numbers should be non-negative",
        )


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
