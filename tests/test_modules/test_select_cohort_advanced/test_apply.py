import unittest

import pandas as pd

from corebehrt.constants.causal.data import (
    EXCLUDED_BY,
    EXCLUSION,
    FLOW,
    FLOW_AFTER_AGE,
    FLOW_AFTER_MINIMUM_ONE,
    FLOW_AFTER_STRICT,
    FLOW_FINAL,
    FLOW_INITIAL,
    INCLUDED,
    STRICT_INCLUSION,
    TOTAL,
)
from corebehrt.constants.data import AGE_COL, PID_COL
from corebehrt.modules.cohort_handling.advanced.criteria.apply import apply_criteria
from corebehrt.modules.cohort_handling.advanced.utils.definitions import (
    EXCLUSION_CRITERIA,
    INCLUSION_CRITERIA,
    MIN_AGE,
    MINIMUM_ONE,
    STRICT,
)


class TestApplyCriteria(unittest.TestCase):
    def setUp(self):
        """Create test data and config."""
        # Create test DataFrame with various patient scenarios
        self.df = pd.DataFrame(
            {
                PID_COL: [1, 2, 3, 4, 5, 6],
                AGE_COL: [55, 45, 65, 60, 70, 52],
                # Strict inclusion criteria
                "type2_diabetes": [True, True, True, False, True, True],
                # Minimum one criteria
                "myocardial_infarction": [True, False, False, False, False, False],
                "stroke": [False, True, False, False, False, False],
                "transient_ischemic_attack": [False, False, False, False, False, False],
                "glycemic_control": [False, False, True, False, False, True],
                # Exclusion criteria
                "type1_diabetes": [False, False, False, True, False, False],
                "cancer": [False, False, True, False, False, False],
                "pregnancy_and_birth": [False, False, False, False, True, False],
            }
        )

        # Test configuration
        self.config = {
            MIN_AGE: 50,
            INCLUSION_CRITERIA: {
                STRICT: ["type2_diabetes"],
                MINIMUM_ONE: [
                    "myocardial_infarction",
                    "stroke",
                    "transient_ischemic_attack",
                    "glycemic_control",
                ],
            },
            EXCLUSION_CRITERIA: ["type1_diabetes", "cancer", "pregnancy_and_birth"],
        }

    def test_age_criterion(self):
        """Test that patients below minimum age are excluded."""
        included, stats = apply_criteria(self.df, self.config)
        self.assertNotIn(2, included[PID_COL].values)  # Patient 2 is 45 years old
        self.assertEqual(stats[EXCLUDED_BY][AGE_COL], 1)

    def test_strict_inclusion(self):
        """Test that patients without required criteria are excluded."""
        included, stats = apply_criteria(self.df, self.config)
        self.assertNotIn(
            4, included[PID_COL].values
        )  # Patient 4 doesn't have type2_diabetes
        self.assertTrue(stats[EXCLUDED_BY][STRICT_INCLUSION]["type2_diabetes"] > 0)

    def test_minimum_one_criteria(self):
        """Test that patients with none of the minimum_one criteria are excluded."""
        # Patient 5 has type2_diabetes but none of the minimum_one criteria
        included, stats = apply_criteria(self.df, self.config)
        self.assertNotIn(5, included[PID_COL].values)
        self.assertTrue(stats[EXCLUDED_BY][MINIMUM_ONE] > 0)

    def test_exclusion_criteria(self):
        """Test that patients with any exclusion criteria are excluded."""
        included, stats = apply_criteria(self.df, self.config)
        # Patient 3 has cancer
        self.assertNotIn(3, included[PID_COL].values)
        # Patient 4 has type1_diabetes
        self.assertNotIn(4, included[PID_COL].values)
        # Patient 5 has pregnancy_and_birth
        self.assertNotIn(5, included[PID_COL].values)

        self.assertTrue(stats[EXCLUDED_BY][EXCLUSION]["cancer"] > 0)
        self.assertTrue(
            stats[EXCLUDED_BY][EXCLUSION]["type1_diabetes"] == 0
        )  # this patient does not have type2_diabetes
        self.assertTrue(
            stats[EXCLUDED_BY][EXCLUSION]["pregnancy_and_birth"] == 0
        )  # this patient does not have an additional inclusion criteria

    def test_included_patients(self):
        """Test that patients meeting all criteria are included."""
        included, _ = apply_criteria(self.df, self.config)
        # Patient 1 should be included:
        # - Age > 50
        # - Has type2_diabetes
        # - Has myocardial_infarction (minimum_one)
        # - No exclusion criteria
        self.assertIn(1, included[PID_COL].values)

        # Patient 6 should be included:
        # - Age > 50
        # - Has type2_diabetes
        # - Has glycemic_control (minimum_one)
        # - No exclusion criteria
        self.assertIn(6, included[PID_COL].values)

    def test_statistics_total(self):
        """
        Test that statistics are correctly calculated.
        Statistics should reflect the sequential application of criteria:
        1. First get inclusion cohort (age + strict + minimum_one)
        2. Then count exclusions only from that cohort
        """
        _, stats = apply_criteria(self.df, self.config)

        # Total patients at start
        self.assertEqual(stats[TOTAL], 6)

        # Verify inclusion criteria statistics
        self.assertEqual(stats[EXCLUDED_BY][AGE_COL], 1)  # Patient 2 (age 45)
        self.assertEqual(
            stats[EXCLUDED_BY][STRICT_INCLUSION], {"type2_diabetes": 1}
        )  # Patient 4 (no T2D)
        self.assertEqual(
            stats[EXCLUDED_BY][MINIMUM_ONE], 1
        )  # Patient 5 (no minimum_one criteria)

        # Only count exclusions from patients who passed inclusion criteria
        # Patient 3 has cancer and would be in inclusion cohort
        self.assertEqual(stats[EXCLUDED_BY][EXCLUSION]["cancer"], 1)
        # Patient 4 was already excluded by strict criteria, shouldn't count for T1D
        self.assertEqual(stats[EXCLUDED_BY][EXCLUSION]["type1_diabetes"], 0)
        # Patient 5 was already excluded by minimum_one, shouldn't count for pregnancy
        self.assertEqual(stats[EXCLUDED_BY][EXCLUSION]["pregnancy_and_birth"], 0)

        # Final included patients
        self.assertEqual(stats[INCLUDED], 2)  # Only patients 1 and 6 remain

    def test_patient_flow(self):
        """Test that patient flow counts are correctly tracked at each step."""
        _, stats = apply_criteria(self.df, self.config)

        self.assertEqual(stats[FLOW][FLOW_INITIAL], 6)  # All patients
        self.assertEqual(stats[FLOW][FLOW_AFTER_AGE], 5)  # Exclude patient 2 (age)
        self.assertEqual(
            stats[FLOW][FLOW_AFTER_STRICT], 4
        )  # Exclude patient 4 (no T2D)
        self.assertEqual(
            stats[FLOW][FLOW_AFTER_MINIMUM_ONE], 3
        )  # Exclude patient 5 (no min one)
        self.assertEqual(stats[FLOW][FLOW_FINAL], 2)  # Exclude patient 3 (cancer)

    def test_consort_numbers_match(self):
        """Test that flow numbers match exclusion counts."""
        _, stats = apply_criteria(self.df, self.config)

        # Verify that differences between steps match exclusion counts
        self.assertEqual(
            stats[FLOW][FLOW_INITIAL] - stats[FLOW][FLOW_AFTER_AGE],
            stats[EXCLUDED_BY][AGE_COL],
        )

        self.assertEqual(
            stats[FLOW][FLOW_AFTER_AGE] - stats[FLOW][FLOW_AFTER_STRICT],
            sum(stats[EXCLUDED_BY][STRICT_INCLUSION].values()),
        )

        self.assertEqual(
            stats[FLOW][FLOW_AFTER_STRICT] - stats[FLOW][FLOW_AFTER_MINIMUM_ONE],
            stats[EXCLUDED_BY][MINIMUM_ONE],
        )

        self.assertEqual(
            stats[FLOW][FLOW_AFTER_MINIMUM_ONE] - stats[FLOW][FLOW_FINAL],
            sum(stats[EXCLUDED_BY][EXCLUSION].values()),
        )


if __name__ == "__main__":
    unittest.main()
