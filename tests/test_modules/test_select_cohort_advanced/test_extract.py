import unittest

import pandas as pd

from corebehrt.constants.cohort import (
    CODE_ENTRY,
    CODE_GROUPS,
    CODE_PATTERNS,
    CRITERIA_DEFINITIONS,
    DAYS,
    DELAYS,
    EXCLUDE_CODES,
    MIN_AGE,
    OPERATOR,
    THRESHOLD,
    TIME_WINDOW_DAYS,
    USE_PATTERNS,
)
from corebehrt.constants.data import (
    BIRTH_CODE,
    CONCEPT_COL,
    PID_COL,
    TIMESTAMP_COL,
    VALUE_COL,
)
from corebehrt.modules.cohort_handling.advanced.extract import extract_patient_criteria


class TestCriteriaExtraction(unittest.TestCase):
    def setUp(self):
        self.config = {
            DELAYS: {DAYS: 14, CODE_GROUPS: ["D/", "RD/"]},
            MIN_AGE: 50,
            CRITERIA_DEFINITIONS: {
                "type2_diabetes": {CODE_ENTRY: ["^D/C11.*", "^M/A10BH.*"]},
                "stroke": {CODE_ENTRY: ["^D/I6[3-6].*", "^D/H341.*"]},
                "HbA1c": {
                    CODE_ENTRY: ["(?i).*hba1c.*"],
                    THRESHOLD: 7.0,
                    OPERATOR: ">=",
                },
                "type1_diabetes": {CODE_ENTRY: ["^D/C10.*"]},
                "cancer": {
                    CODE_ENTRY: ["^D/C[0-9][0-9].*"],
                    EXCLUDE_CODES: ["^D/C449.*"],
                    TIME_WINDOW_DAYS: 1826,
                },
                "pregnancy_and_birth": {CODE_ENTRY: ["^D/O[0-9][0-9].*"]},
            },
        }

        self.index_dates = pd.DataFrame(
            {
                PID_COL: [1, 2, 3, 4, 5, 6],
                TIMESTAMP_COL: pd.to_datetime(
                    [
                        "2023-06-01",
                        "2023-06-01",
                        "2023-06-01",
                        "2023-06-01",
                        "2023-06-01",
                        "2023-06-01",
                    ]
                ),
            }
        )

        self.df = pd.DataFrame(
            {
                PID_COL: [1, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 6],
                CONCEPT_COL: [
                    BIRTH_CODE,
                    "D/C11",
                    "D/I63",
                    "HbA1c",
                    BIRTH_CODE,
                    "D/C11",
                    BIRTH_CODE,
                    "D/C11",
                    "D/C50",
                    BIRTH_CODE,
                    "D/C10",
                    BIRTH_CODE,
                    "D/O20",
                    BIRTH_CODE,
                    "D/C11",
                    "D/I63",
                ],
                VALUE_COL: [
                    None,
                    None,
                    None,
                    7.5,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ],
                TIMESTAMP_COL: pd.to_datetime(
                    [
                        "1968-01-01",
                        "2023-05-15",
                        "2023-05-20",
                        "2023-05-28",
                        "1978-01-01",
                        "2023-05-15",
                        "1965-01-01",
                        "2023-05-15",
                        "2021-06-15",
                        "1965-01-01",
                        "2023-05-15",
                        "1968-01-01",
                        "2023-05-20",
                        "1968-01-01",
                        "2023-05-15",
                        "2023-06-10",
                    ]
                ),
            }
        )
        self.criteria_definitions = self.config.get(CRITERIA_DEFINITIONS)
        self.delays_config = self.config.get(DELAYS)
        self.code_patterns = self.config.get(CODE_PATTERNS)

    def test_patient_1_included(self):
        patients = extract_patient_criteria(
            self.df,
            self.index_dates,
            criteria_definitions=self.criteria_definitions,
            delays_config=self.delays_config,
            code_patterns=self.code_patterns,
        )
        patient = patients[1]
        self.assertTrue(patient.criteria_flags["type2_diabetes"])
        self.assertTrue(patient.criteria_flags["stroke"])
        self.assertTrue(patient.criteria_flags["HbA1c"])
        self.assertEqual(patient.values["HbA1c"], 7.5)

    def test_patient_2_too_young(self):
        patients = extract_patient_criteria(
            self.df,
            self.index_dates,
            criteria_definitions=self.criteria_definitions,
            delays_config=self.delays_config,
            code_patterns=self.code_patterns,
        )
        patient = patients[2]
        self.assertTrue(patient.criteria_flags["type2_diabetes"])
        self.assertFalse(patient.criteria_flags["stroke"])
        self.assertFalse(patient.criteria_flags["HbA1c"])
        self.assertLess(patient.age, self.config[MIN_AGE])

    def test_patient_3_recent_cancer(self):
        patients = extract_patient_criteria(
            self.df,
            self.index_dates,
            criteria_definitions=self.criteria_definitions,
            delays_config=self.delays_config,
            code_patterns=self.code_patterns,
        )
        patient = patients[3]
        self.assertTrue(patient.criteria_flags["cancer"])

    def test_patient_4_type1_diabetes(self):
        patients = extract_patient_criteria(
            self.df,
            self.index_dates,
            criteria_definitions=self.criteria_definitions,
            delays_config=self.delays_config,
            code_patterns=self.code_patterns,
        )
        patient = patients[4]
        self.assertTrue(patient.criteria_flags["type1_diabetes"])

    def test_patient_5_recent_pregnancy(self):
        patients = extract_patient_criteria(
            self.df,
            self.index_dates,
            criteria_definitions=self.criteria_definitions,
            delays_config=self.delays_config,
            code_patterns=self.code_patterns,
        )
        patient = patients[5]
        self.assertTrue(patient.criteria_flags["pregnancy_and_birth"])

    def test_patient_6_stroke_within_delay(self):
        patients = extract_patient_criteria(
            self.df,
            self.index_dates,
            criteria_definitions=self.criteria_definitions,
            delays_config=self.delays_config,
            code_patterns=self.code_patterns,
        )
        patient = patients[6]
        self.assertTrue(patient.criteria_flags["stroke"])


class TestPatternUsage(unittest.TestCase):
    def setUp(self):
        """Set up test data with various medication codes and patterns."""
        self.config = {
            CODE_PATTERNS: {
                "metformin": {CODE_ENTRY: ["^(?:M|RM)/A10BA.*"]},
                "dpp4_inhibitors": {
                    CODE_ENTRY: ["^(?:M|RM)/A10BH.*", "^(?:M|RM)/A10BD0[7-9].*"]
                },
                "sglt2_inhibitors": {CODE_ENTRY: ["^(?:M|RM)/A10BK.*"]},
            },
            CRITERIA_DEFINITIONS: {
                "type2_diabetes": {
                    USE_PATTERNS: ["metformin", "dpp4_inhibitors"],
                    CODE_ENTRY: [
                        "^D/C11.*"
                    ],  # Any of these OR any pattern match makes it true
                },
                "dpp4_use": {USE_PATTERNS: ["dpp4_inhibitors"]},
                "any_diabetes_med": {
                    USE_PATTERNS: ["metformin", "dpp4_inhibitors", "sglt2_inhibitors"]
                },
            },
        }

        # Create test events data
        self.df = pd.DataFrame(
            {
                PID_COL: [1, 1, 1, 2, 2, 3, 3],
                CONCEPT_COL: [
                    "M/A10BA01",  # metformin for patient 1
                    "D/C11",  # T2D diagnosis for patient 1
                    "M/A10BH01",  # DPP4 for patient 1
                    "M/A10BA01",  # metformin for patient 2
                    "M/A10BK01",  # SGLT2 for patient 2
                    "D/C11",  # T2D diagnosis only for patient 3
                    "M/A10BD08",  # Combination with DPP4 for patient 3
                ],
                TIMESTAMP_COL: pd.to_datetime(
                    [
                        "2023-05-15",
                        "2023-05-15",
                        "2023-05-15",
                        "2023-05-15",
                        "2023-05-15",
                        "2023-05-15",
                        "2023-05-15",
                    ]
                ),
            }
        )

        self.index_dates = pd.DataFrame(
            {PID_COL: [1, 2, 3], TIMESTAMP_COL: pd.to_datetime(["2023-06-01"] * 3)}
        )
        self.criteria_definitions = self.config.get(CRITERIA_DEFINITIONS)
        self.delays_config = self.config.get(DELAYS)
        self.code_patterns = self.config.get(CODE_PATTERNS)

    def test_pattern_combination(self):
        """Test criteria using multiple patterns."""
        patients = extract_patient_criteria(
            self.df,
            self.index_dates,
            criteria_definitions=self.criteria_definitions,
            delays_config=self.delays_config,
            code_patterns=self.code_patterns,
        )

        # Patient 1: Has metformin, DPP4, and T2D diagnosis (any would make it true)
        self.assertTrue(patients[1].criteria_flags["type2_diabetes"])
        self.assertTrue(patients[1].criteria_flags["dpp4_use"])
        self.assertTrue(patients[1].criteria_flags["any_diabetes_med"])

        # Patient 2: Has metformin and SGLT2
        self.assertTrue(
            patients[2].criteria_flags["type2_diabetes"]
        )  # True because has metformin
        self.assertFalse(patients[2].criteria_flags["dpp4_use"])
        self.assertTrue(patients[2].criteria_flags["any_diabetes_med"])

        # Patient 3: Has T2D diagnosis and DPP4 combination
        self.assertTrue(
            patients[3].criteria_flags["type2_diabetes"]
        )  # True from either diagnosis or DPP4
        self.assertTrue(patients[3].criteria_flags["dpp4_use"])

    def test_pattern_with_direct_codes(self):
        """Test criteria using both patterns and direct codes."""
        patients = extract_patient_criteria(
            self.df,
            self.index_dates,
            criteria_definitions=self.criteria_definitions,
            delays_config=self.delays_config,
            code_patterns=self.code_patterns,
        )

        # Patient 3: Has T2D diagnosis and DPP4 (either makes it true)
        self.assertTrue(patients[3].criteria_flags["type2_diabetes"])

        # Patient 2: Has metformin (makes type2_diabetes true)
        self.assertTrue(patients[2].criteria_flags["type2_diabetes"])

    def test_pattern_reuse(self):
        """Test that the same pattern can be reused in different criteria."""
        patients = extract_patient_criteria(
            self.df,
            self.index_dates,
            criteria_definitions=self.criteria_definitions,
            delays_config=self.delays_config,
            code_patterns=self.code_patterns,
        )

        # Patient 1: Check DPP4 pattern in both criteria
        self.assertTrue(patients[1].criteria_flags["type2_diabetes"])
        self.assertTrue(patients[1].criteria_flags["dpp4_use"])

        # Patient 2: No DPP4, should be false in dpp4_use but true in type2_diabetes (has metformin)
        self.assertFalse(patients[2].criteria_flags["dpp4_use"])
        self.assertTrue(patients[2].criteria_flags["type2_diabetes"])


if __name__ == "__main__":
    unittest.main()
