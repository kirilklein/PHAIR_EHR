import unittest
import pandas as pd

from corebehrt.constants.cohort import (
    N_EXCLUDED_BY_EXPRESSION,
    INCLUSION,
    EXCLUSION,
)
from corebehrt.constants.data import AGE_COL, PID_COL
from corebehrt.modules.cohort_handling.advanced.apply import apply_criteria_with_stats


class TestApplyCriteria(unittest.TestCase):
    def setUp(self):
        """Create test data and configuration."""
        # Create test DataFrame with various patient scenarios
        self.df = pd.DataFrame(
            {
                PID_COL: [1, 2, 3, 4, 5, 6],
                AGE_COL: [55, 45, 65, 60, 70, 52],
                # Criteria for testing expressions
                "type2_diabetes": [True, True, True, False, True, True],
                "myocardial_infarction": [True, False, False, False, False, False],
                "stroke": [False, True, False, False, False, False],
                "glycemic_control": [False, False, True, False, False, True],
                "type1_diabetes": [False, False, False, True, False, False],
                "cancer": [False, False, True, False, False, False],
                "pregnancy": [False, False, False, False, True, False],
            }
        )

        # Updated config format using expressions
        self.config = {
            INCLUSION: "type2_diabetes & (myocardial_infarction | stroke | glycemic_control)",
            EXCLUSION: "type1_diabetes | cancer | pregnancy",
        }

    def test_inclusion_expression(self):
        """Test that inclusion expression works correctly."""
        included, stats = apply_criteria_with_stats(
            self.df, self.config[INCLUSION], self.config[EXCLUSION]
        )
        # Patient 1 has type2_diabetes AND myocardial_infarction
        self.assertIn(1, included[PID_COL].values)
        # Patient 4 doesn't have type2_diabetes
        self.assertNotIn(4, included[PID_COL].values)

        # Check stats
        self.assertGreater(stats[N_EXCLUDED_BY_EXPRESSION], 0)

    def test_exclusion_expression(self):
        """Test that exclusion expression works correctly."""
        included, stats = apply_criteria_with_stats(
            self.df, self.config[INCLUSION], self.config[EXCLUSION]
        )
        # Patient 3 has cancer
        self.assertNotIn(3, included[PID_COL].values)
        # Patient 5 has pregnancy
        self.assertNotIn(5, included[PID_COL].values)

        self.assertGreater(stats[N_EXCLUDED_BY_EXPRESSION], 0)

    def test_complex_expression(self):
        """Test complex expressions with NOT operator."""
        test_data = pd.DataFrame(
            {
                PID_COL: [1, 2],
                "condition_a": [True, True],
                "condition_b": [True, False],
                "condition_c": [False, True],
            }
        )

        expression = "condition_a & ~condition_b | condition_c"
        included, stats = apply_criteria_with_stats(
            test_data,
            expression,
            "condition_a & ~condition_a",  # Always False using existing column
        )

        # Patient 2 should be included (has condition_a and NOT condition_b)
        self.assertIn(2, included[PID_COL].values)
        # Patient 1 should be excluded (has both condition_a and condition_b)
        self.assertNotIn(1, included[PID_COL].values)

    def test_empty_expressions(self):
        """Test handling of empty or trivial expressions."""
        test_data = pd.DataFrame({PID_COL: [1, 2], "condition": [True, False]})

        # Test with trivial inclusion and exclusion using existing column
        included, stats = apply_criteria_with_stats(
            test_data,
            "condition | ~condition",  # Always True
            "condition & ~condition",  # Always False
        )

        self.assertEqual(len(included), 2)  # All patients should be included
        self.assertEqual(stats[N_EXCLUDED_BY_EXPRESSION], 0)


if __name__ == "__main__":
    unittest.main()
