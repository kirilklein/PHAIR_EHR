import unittest

import pandas as pd
from pandas import to_datetime

from corebehrt.constants.cohort import (
    AGE_AT_INDEX_DATE,
    CODE_ENTRY,
    CODE_GROUPS,
    CRITERIA_DEFINITIONS,
    CRITERION_FLAG,
    DAYS,
    DELAYS,
    EXCLUDE_CODES,
    MIN_AGE,
    MIN_VALUE,
    NUMERIC_VALUE,
    NUMERIC_VALUE_SUFFIX,
    TIME_WINDOW_DAYS,
)
from corebehrt.constants.data import CONCEPT_COL, PID_COL, TIMESTAMP_COL, VALUE_COL
from corebehrt.modules.cohort_handling.advanced.extract import (
    CohortExtractor,
    CriteriaExtraction,
)


class TestExtraction(unittest.TestCase):
    def setUp(self):
        # Define a configuration using the new constant names.
        self.config = {
            DELAYS: {DAYS: 14, CODE_GROUPS: ["D/", "RD/"]},
            CRITERIA_DEFINITIONS: {
                "type2_diabetes": {  # Simple criterion: code-based
                    CODE_ENTRY: ["^D/C11.*", "^M/A10BH.*"]
                },
                "stroke": {  # Simple criterion: code-based
                    CODE_ENTRY: ["^D/I6[3-6].*", "^D/H341.*"]
                },
                "HbA1c": {  # Numeric criterion: uses a threshold via NUMERIC_VALUE; we require value ≥ 7.0.
                    CODE_ENTRY: ["(?i).*hba1c.*"],
                    NUMERIC_VALUE: {MIN_VALUE: 7.0},
                },
                "type1_diabetes": {CODE_ENTRY: ["^D/C10.*"]},
                "cancer": {  # Simple criterion with a time window
                    CODE_ENTRY: ["^D/C[0-9][0-9].*"],
                    EXCLUDE_CODES: ["^D/C449.*"],
                    TIME_WINDOW_DAYS: 1826,
                },
                "pregnancy_and_birth": {CODE_ENTRY: ["^D/O[0-9][0-9].*"]},
                # Example composite criterion could be defined via EXPRESSION,
                # but for these tests we limit to simple and age-based.
                "age_based": {  # Age-based: flag True if patient is at least MIN_AGE
                    MIN_AGE: 50
                },
            },
        }

        # Create index_dates DataFrame with 6 patients.
        self.index_dates = pd.DataFrame(
            {
                PID_COL: [1, 2, 3, 4, 5, 6],
                TIMESTAMP_COL: to_datetime(["2023-06-01"] * 6),
            }
        )

        # Create events DataFrame.
        # Add "DOB" events to allow age computation.
        self.df = pd.DataFrame(
            {
                PID_COL: [1, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 6],
                CONCEPT_COL: [
                    "DOB",  # patient 1: birth event
                    "D/C11",  # type2_diabetes
                    "D/I63",  # stroke
                    "HbA1c",  # HbA1c with numeric value
                    "DOB",  # patient 2
                    "D/C11",  # type2_diabetes
                    "DOB",  # patient 3
                    "D/C11",  # type2_diabetes
                    "D/C50",  # cancer (matches ^D/C[0-9][0-9].*)
                    "DOB",  # patient 4
                    "D/C10",  # type1_diabetes
                    "DOB",  # patient 5
                    "D/O20",  # pregnancy_and_birth
                    "DOB",  # patient 6
                    "D/C11",  # type2_diabetes
                    "D/I63",  # stroke, within delay
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
                TIMESTAMP_COL: to_datetime(
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

    def test_patient_1_included(self):
        # Patient 1 should:
        # - Have type2_diabetes flag True from "D/C11"
        # - Have stroke flag True from "D/I63"
        # - Have HbA1c flag True and a numeric value of 7.5, because the measurement is ≥ 7.0
        final_results = CohortExtractor(
            self.criteria_definitions, self.delays_config
        ).extract(self.df, self.index_dates)
        patient1 = final_results.loc[final_results[PID_COL] == 1].iloc[0]
        self.assertTrue(patient1["type2_diabetes"])
        self.assertTrue(patient1["stroke"])
        self.assertTrue(patient1["HbA1c"])
        self.assertEqual(patient1["HbA1c" + NUMERIC_VALUE_SUFFIX], 7.5)

    def test_patient_2_too_young(self):
        # For patient 2, DOB is "1978-01-01" and index_date "2023-06-01": age ~45.
        # Expect type2_diabetes flag True from "D/C11" but stroke and HbA1c flags False.
        final_results = CohortExtractor(
            self.criteria_definitions, self.delays_config
        ).extract(self.df, self.index_dates)
        patient2 = final_results.loc[final_results[PID_COL] == 2].iloc[0]
        self.assertTrue(patient2["type2_diabetes"])
        self.assertFalse(patient2["stroke"])
        self.assertFalse(patient2["HbA1c"])

    def test_patient_3_recent_cancer(self):
        # Patient 3 has event "D/C50" which should trigger the cancer criterion.
        final_results = CohortExtractor(
            self.criteria_definitions, self.delays_config
        ).extract(self.df, self.index_dates)
        patient3 = final_results.loc[final_results[PID_COL] == 3].iloc[0]
        self.assertTrue(patient3["cancer"])

    def test_patient_4_type1_diabetes(self):
        # Patient 4 should have type1_diabetes flagged True.
        final_results = CohortExtractor(
            self.criteria_definitions, self.delays_config
        ).extract(self.df, self.index_dates)
        patient4 = final_results.loc[final_results[PID_COL] == 4].iloc[0]
        self.assertTrue(patient4["type1_diabetes"])

    def test_patient_5_pregnancy_and_birth(self):
        # Patient 5 should have the pregnancy_and_birth flag set True.
        final_results = CohortExtractor(
            self.criteria_definitions, self.delays_config
        ).extract(self.df, self.index_dates)
        patient5 = final_results.loc[final_results[PID_COL] == 5].iloc[0]
        self.assertTrue(patient5["pregnancy_and_birth"])

    def test_patient_6_stroke_within_delay(self):
        # Patient 6: Has events "D/C11" and "D/I63". The stroke event should be flagged.
        final_results = CohortExtractor(
            self.criteria_definitions, self.delays_config
        ).extract(self.df, self.index_dates)
        patient6 = final_results.loc[final_results[PID_COL] == 6].iloc[0]
        self.assertTrue(patient6["stroke"])

    def test_age_calculation(self):
        # Validate that age_at_index_date is computed correctly.
        # For example, for patient 1: birth "1968-01-01" and index "2023-06-01" ~55 years.
        final_results = CohortExtractor(
            self.criteria_definitions, self.delays_config
        ).extract(self.df, self.index_dates)
        patient1 = final_results.loc[final_results[PID_COL] == 1].iloc[0]
        self.assertAlmostEqual(patient1[AGE_AT_INDEX_DATE], 55, delta=1)


class TestPatternUsage(unittest.TestCase):
    def setUp(self):
        # Define a configuration that uses code patterns and the USE_PATTERNS mechanism.
        self.config = {
            "code_patterns": {  # This part is used by pattern-based criteria.
                "metformin": {CODE_ENTRY: ["^(?:M|RM)/A10BA.*"]},
                "dpp4_inhibitors": {
                    CODE_ENTRY: ["^(?:M|RM)/A10BH.*", "^(?:M|RM)/A10BD0[7-9].*"]
                },
                "sglt2_inhibitors": {CODE_ENTRY: ["^(?:M|RM)/A10BK.*"]},
            },
            CRITERIA_DEFINITIONS: {
                "type2_diabetes": {
                    # For this test we simulate that direct codes are combined with patterns.
                    CODE_ENTRY: ["^D/C11.*"],
                    # In a more complete system you would integrate patterns via a separate mechanism.
                },
                "dpp4_use": {
                    CODE_ENTRY: [],  # Simulate usage of dpp4_inhibitors pattern.
                    # In the new code, pattern usage could be implemented by merging pattern results.
                },
                "any_diabetes_med": {
                    CODE_ENTRY: []  # Simulate union of multiple patterns.
                },
            },
            DELAYS: {DAYS: 14, CODE_GROUPS: ["D/"]},
        }

        # Create test events data.
        self.df = pd.DataFrame(
            {
                PID_COL: [1, 1, 1, 2, 2, 3, 3],
                CONCEPT_COL: [
                    "M/A10BA01",  # metformin for patient 1
                    "D/C11",  # T2D diagnosis for patient 1
                    "M/A10BH01",  # dpp4 inhibitor for patient 1
                    "M/A10BA01",  # metformin for patient 2
                    "M/A10BK01",  # sglt2 inhibitor for patient 2
                    "D/C11",  # T2D diagnosis for patient 3
                    "M/A10BD08",  # dpp4 inhibitor for patient 3
                ],
                TIMESTAMP_COL: to_datetime(
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
            {PID_COL: [1, 2, 3], TIMESTAMP_COL: to_datetime(["2023-06-01"] * 3)}
        )
        self.criteria_definitions = self.config.get(CRITERIA_DEFINITIONS)
        self.delays_config = self.config.get(DELAYS)
        # For these tests we skip actual pattern extraction and test the basic functionality.

    def test_pattern_combination(self):
        # Using the basic type2_diabetes criteria (direct codes), patient 1 and 3 should be flagged.
        final_results = CohortExtractor(
            self.criteria_definitions, self.delays_config
        ).extract(self.df, self.index_dates)
        patient1 = final_results.loc[final_results[PID_COL] == 1].iloc[0]
        patient3 = final_results.loc[final_results[PID_COL] == 3].iloc[0]
        self.assertTrue(patient1["type2_diabetes"])
        self.assertTrue(patient3["type2_diabetes"])

    def test_pattern_reuse(self):
        # Here we simulate the reuse of a pattern across multiple criteria.
        # For simplicity, we check that if patient 1 meets one criteria (type2_diabetes),
        # then related criteria (dpp4_use) which might share the same pattern in a full implementation
        # would also be flagged. For this test we assume that patient 1 has a dpp4 use via M/A10BH01.
        final_results = CohortExtractor(
            self.criteria_definitions, self.delays_config
        ).extract(self.df, self.index_dates)
        patient1 = final_results.loc[final_results[PID_COL] == 1].iloc[0]
        # Since our dummy config for dpp4_use has no codes, it will default to False,
        # but in an integrated system it would be combined with patterns.
        self.assertTrue(patient1["type2_diabetes"])
        # We simply check that the dpp4_use column is present.
        self.assertIn("dpp4_use", final_results.columns)


class TestVectorizedExtractionFunctions(unittest.TestCase):
    def setUp(self):
        self.events = pd.DataFrame(
            {
                PID_COL: [1, 1],
                TIMESTAMP_COL: to_datetime(["2023-05-15", "2023-05-16"]),
                CONCEPT_COL: ["D/TST01", "D/TST02"],
                VALUE_COL: [5.5, None],
            }
        )

        self.index_dates = pd.DataFrame(
            {PID_COL: [1], TIMESTAMP_COL: to_datetime(["2023-06-01"])}
        )

        self.delays_config = {DAYS: 14, CODE_GROUPS: ["D/"]}

    def test_vectorized_extraction_codes_non_numeric(self):
        crit_cfg = {CODE_ENTRY: ["^D/TST.*"]}
        criteria_definitions = {"test_criterion": crit_cfg}

        result = CriteriaExtraction.extract_codes(
            self.events, self.index_dates, crit_cfg, self.delays_config
        )

        self.assertEqual(result.shape[0], 1)
        self.assertTrue(result.iloc[0][CRITERION_FLAG])
        self.assertIsNone(result.iloc[0][NUMERIC_VALUE])

    def test_vectorized_extraction_codes_numeric_in_range(self):
        crit_cfg = {
            CODE_ENTRY: ["^D/TST.*"],
            NUMERIC_VALUE: {MIN_VALUE: 5},
        }

        result = CriteriaExtraction.extract_codes(
            self.events, self.index_dates, crit_cfg, self.delays_config
        )

        self.assertEqual(result.shape[0], 1)
        self.assertTrue(result.iloc[0][CRITERION_FLAG])
        self.assertAlmostEqual(result.iloc[0][NUMERIC_VALUE], 5.5)

    def test_vectorized_extraction_expression(self):
        initial_results = pd.DataFrame(
            {PID_COL: [1, 2], "TYPE2_DIABETES": [True, False], "STROKE": [False, True]}
        )

        expression = "TYPE2_DIABETES & ~STROKE"
        result = CriteriaExtraction.extract_expression(
            expression, initial_results
        )

        self.assertEqual(result.shape[0], 2)
        self.assertTrue(result.loc[result[PID_COL] == 1, CRITERION_FLAG].iloc[0])
        self.assertFalse(result.loc[result[PID_COL] == 2, CRITERION_FLAG].iloc[0])

    def test_vectorized_extraction_age(self):
        initial_results = pd.DataFrame(
            {PID_COL: [1, 2, 3], AGE_AT_INDEX_DATE: [55, 45, 60]}
        )

        result = CriteriaExtraction.extract_age(
            initial_results, min_age=50, max_age=59
        )

        self.assertEqual(result.shape[0], 3)
        self.assertTrue(result.loc[result[PID_COL] == 1, CRITERION_FLAG].iloc[0])
        self.assertFalse(result.loc[result[PID_COL] == 2, CRITERION_FLAG].iloc[0])
        self.assertFalse(result.loc[result[PID_COL] == 3, CRITERION_FLAG].iloc[0])


if __name__ == "__main__":
    unittest.main()
