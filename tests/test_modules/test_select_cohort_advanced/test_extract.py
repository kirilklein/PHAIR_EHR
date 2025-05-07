import unittest

import pandas as pd
from pandas import to_datetime

from corebehrt.constants.cohort import (
    AGE_AT_INDEX_DATE,
    CODE_ENTRY,
    EXPRESSION,
    EXCLUDE_CODES,
    MAX_AGE,
    MIN_AGE,
    MIN_VALUE,
    NUMERIC_VALUE,
    NUMERIC_VALUE_SUFFIX,
    START_DAYS,
    END_DAYS,
    UNIQUE_CRITERIA_LIST,
    MAX_COUNT,
    MIN_COUNT,
)
from corebehrt.constants.data import CONCEPT_COL, PID_COL, TIMESTAMP_COL, VALUE_COL
from corebehrt.modules.cohort_handling.advanced.extract import CohortExtractor


class TestExtraction(unittest.TestCase):
    def setUp(self):
        # Define a configuration using the new constant names and structure
        self.criteria_definitions = {
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
            "cancer": {  # Simple criterion with a time window, now using start_days and end_days
                CODE_ENTRY: ["^D/C[0-9][0-9].*"],
                EXCLUDE_CODES: ["^D/C449.*"],
                START_DAYS: -1826,  # 5 years before index date
                END_DAYS: 0,  # Up to index date
            },
            "pregnancy_and_birth": {CODE_ENTRY: ["^D/O[0-9][0-9].*"]},
            # Age-based criterion, replacing the old method
            "age_based": {MIN_AGE: 50},
            # Example of expression-based criterion
            "diabetes_and_stroke": {EXPRESSION: "type2_diabetes & stroke"},
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

    def test_patient_1_included(self):
        # Patient 1 should:
        # - Have type2_diabetes flag True from "D/C11"
        # - Have stroke flag True from "D/I63"
        # - Have HbA1c flag True and a numeric value of 7.5, because the measurement is ≥ 7.0
        final_results = CohortExtractor(self.criteria_definitions).extract(
            self.df, self.index_dates
        )
        patient1 = final_results.loc[final_results[PID_COL] == 1].iloc[0]
        self.assertTrue(patient1["type2_diabetes"])
        self.assertTrue(patient1["stroke"])
        self.assertTrue(patient1["HbA1c"])
        self.assertEqual(patient1["HbA1c" + NUMERIC_VALUE_SUFFIX], 7.5)

    def test_patient_2_too_young(self):
        # For patient 2, DOB is "1978-01-01" and index_date "2023-06-01": age ~45.
        # Expect type2_diabetes flag True from "D/C11" but age_based flag False
        final_results = CohortExtractor(self.criteria_definitions).extract(
            self.df, self.index_dates
        )
        patient2 = final_results.loc[final_results[PID_COL] == 2].iloc[0]
        self.assertTrue(patient2["type2_diabetes"])
        self.assertFalse(patient2["age_based"])  # Under 50 years old

    def test_patient_3_recent_cancer(self):
        # Patient 3 has event "D/C50" which should trigger the cancer criterion.
        final_results = CohortExtractor(self.criteria_definitions).extract(
            self.df, self.index_dates
        )
        patient3 = final_results.loc[final_results[PID_COL] == 3].iloc[0]
        self.assertTrue(patient3["cancer"])

    def test_patient_4_type1_diabetes(self):
        # Patient 4 should have type1_diabetes flagged True.
        final_results = CohortExtractor(self.criteria_definitions).extract(
            self.df, self.index_dates
        )
        patient4 = final_results.loc[final_results[PID_COL] == 4].iloc[0]
        self.assertTrue(patient4["type1_diabetes"])

    def test_patient_5_pregnancy_and_birth(self):
        # Patient 5 should have the pregnancy_and_birth flag set True.
        final_results = CohortExtractor(self.criteria_definitions).extract(
            self.df, self.index_dates
        )
        patient5 = final_results.loc[final_results[PID_COL] == 5].iloc[0]
        self.assertTrue(patient5["pregnancy_and_birth"])

    def test_patient_6_stroke_outside_time_window(self):
        # Patient 6: Has events "D/C11" and "D/I63". The stroke event occurs after index date
        # With default time window (no start_days/end_days specified), stroke should be flagged as False
        # since the event is after the index date
        criteria_definitions_with_time_window = self.criteria_definitions.copy()
        criteria_definitions_with_time_window["stroke"] = {
            CODE_ENTRY: ["^D/I6[3-6].*", "^D/H341.*"],
            START_DAYS: -30,  # 30 days before index date
            END_DAYS: 0,  # Up to index date
        }

        final_results = CohortExtractor(criteria_definitions_with_time_window).extract(
            self.df, self.index_dates
        )
        patient6 = final_results.loc[final_results[PID_COL] == 6].iloc[0]
        self.assertFalse(patient6["stroke"])

        # Now test with a time window that includes the future stroke event
        criteria_definitions_with_future = self.criteria_definitions.copy()
        criteria_definitions_with_future["stroke"] = {
            CODE_ENTRY: ["^D/I6[3-6].*", "^D/H341.*"],
            START_DAYS: 0,  # From index date
            END_DAYS: 15,  # Up to 15 days after index date
        }

        final_results = CohortExtractor(criteria_definitions_with_future).extract(
            self.df, self.index_dates
        )
        patient6 = final_results.loc[final_results[PID_COL] == 6].iloc[0]
        self.assertTrue(patient6["stroke"])

    def test_age_calculation(self):
        # Validate that age_at_index_date is computed correctly.
        # For example, for patient 1: birth "1968-01-01" and index "2023-06-01" ~55 years.
        final_results = CohortExtractor(self.criteria_definitions).extract(
            self.df, self.index_dates
        )
        patient1 = final_results.loc[final_results[PID_COL] == 1].iloc[0]
        self.assertAlmostEqual(patient1[AGE_AT_INDEX_DATE], 55, delta=1)

    def test_expression_criterion(self):
        # Test the expression-based criterion 'diabetes_and_stroke'
        final_results = CohortExtractor(self.criteria_definitions).extract(
            self.df, self.index_dates
        )

        # Patient 1 has both type2_diabetes and stroke
        patient1 = final_results.loc[final_results[PID_COL] == 1].iloc[0]
        self.assertTrue(patient1["diabetes_and_stroke"])

        # Patient 2 has only type2_diabetes, not stroke
        patient2 = final_results.loc[final_results[PID_COL] == 2].iloc[0]
        self.assertFalse(patient2["diabetes_and_stroke"])

    def test_unique_criteria_list(self):
        """Test count-based criteria using UNIQUE_CRITERIA_LIST."""
        # Add a count-based criterion to the existing criteria definitions
        self.criteria_definitions["max_two_conditions"] = {
            UNIQUE_CRITERIA_LIST: [
                "type2_diabetes",
                "stroke",
                "cancer",
                "pregnancy_and_birth",
            ],
            MAX_COUNT: 2,
        }

        final_results = CohortExtractor(self.criteria_definitions).extract(
            self.df, self.index_dates
        )

        # Patient 1: has type2_diabetes, stroke, and HbA1c -> should fail (3 conditions)
        patient1 = final_results.loc[final_results[PID_COL] == 1].iloc[0]
        self.assertFalse(patient1["max_two_conditions"])

        # Patient 2: has only type2_diabetes -> should pass (1 condition)
        patient2 = final_results.loc[final_results[PID_COL] == 2].iloc[0]
        self.assertTrue(patient2["max_two_conditions"])

        # Patient 3: has type2_diabetes and cancer -> should pass (2 conditions)
        patient3 = final_results.loc[final_results[PID_COL] == 3].iloc[0]
        self.assertTrue(patient3["max_two_conditions"])

        # Patient 4: has only type1_diabetes -> should pass (1 condition)
        patient4 = final_results.loc[final_results[PID_COL] == 4].iloc[0]
        self.assertTrue(patient4["max_two_conditions"])

    def test_unique_criteria_list_min_count(self):
        """Test count-based criteria with MIN_COUNT requirement."""
        # Add a count-based criterion that requires at least 2 conditions
        self.criteria_definitions["at_least_two_conditions"] = {
            UNIQUE_CRITERIA_LIST: [
                "type2_diabetes",
                "stroke",
                "cancer",
                "pregnancy_and_birth",
            ],
            MIN_COUNT: 2,
        }

        final_results = CohortExtractor(self.criteria_definitions).extract(
            self.df, self.index_dates
        )

        # Patient 1: has type2_diabetes, stroke, and HbA1c -> should pass (3 conditions)
        patient1 = final_results.loc[final_results[PID_COL] == 1].iloc[0]
        self.assertTrue(patient1["at_least_two_conditions"])

        # Patient 2: has type2_diabetes and HbA1c -> should pass (2 conditions)
        patient2 = final_results.loc[final_results[PID_COL] == 2].iloc[0]
        self.assertTrue(patient2["at_least_two_conditions"])

        # Patient 3: has type2_diabetes and cancer -> should pass (2 conditions)
        patient3 = final_results.loc[final_results[PID_COL] == 3].iloc[0]
        self.assertTrue(patient3["at_least_two_conditions"])

        # Patient 4: has only type1_diabetes -> should fail (1 condition)
        patient4 = final_results.loc[final_results[PID_COL] == 4].iloc[0]
        self.assertFalse(patient4["at_least_two_conditions"])


class TestPatternUsage(unittest.TestCase):
    def setUp(self):
        # In the new implementation, we don't need a separate patterns mechanism
        # as patterns can be directly defined as criteria
        self.criteria_definitions = {
            "metformin": {CODE_ENTRY: ["^(?:M|RM)/A10BA.*"]},
            "dpp4_inhibitors": {
                CODE_ENTRY: ["^(?:M|RM)/A10BH.*", "^(?:M|RM)/A10BD0[7-9].*"]
            },
            "sglt2_inhibitors": {CODE_ENTRY: ["^(?:M|RM)/A10BK.*"]},
            "type2_diabetes": {CODE_ENTRY: ["^D/C11.*"]},
            "any_diabetes_med": {
                EXPRESSION: "metformin | dpp4_inhibitors | sglt2_inhibitors"
            },
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
                VALUE_COL: [None] * 7,
            }
        )

        self.index_dates = pd.DataFrame(
            {PID_COL: [1, 2, 3], TIMESTAMP_COL: to_datetime(["2023-06-01"] * 3)}
        )

    def test_pattern_combination(self):
        # Using the basic type2_diabetes criteria (direct codes), patient 1 and 3 should be flagged.
        final_results = CohortExtractor(self.criteria_definitions).extract(
            self.df, self.index_dates
        )
        patient1 = final_results.loc[final_results[PID_COL] == 1].iloc[0]
        patient3 = final_results.loc[final_results[PID_COL] == 3].iloc[0]
        self.assertTrue(patient1["type2_diabetes"])
        self.assertTrue(patient3["type2_diabetes"])

    def test_pattern_expression(self):
        # Test if the expression properly combines different medication pattern criteria
        final_results = CohortExtractor(self.criteria_definitions).extract(
            self.df, self.index_dates
        )

        # Patient 1 has metformin and dpp4_inhibitor
        patient1 = final_results.loc[final_results[PID_COL] == 1].iloc[0]
        self.assertTrue(patient1["metformin"])
        self.assertTrue(patient1["dpp4_inhibitors"])
        self.assertTrue(patient1["any_diabetes_med"])

        # Patient 2 has metformin and sglt2_inhibitor
        patient2 = final_results.loc[final_results[PID_COL] == 2].iloc[0]
        self.assertTrue(patient2["metformin"])
        self.assertTrue(patient2["sglt2_inhibitors"])
        self.assertTrue(patient2["any_diabetes_med"])

        # Patient 3 has dpp4_inhibitor
        patient3 = final_results.loc[final_results[PID_COL] == 3].iloc[0]
        self.assertFalse(patient3["metformin"])
        self.assertTrue(patient3["dpp4_inhibitors"])
        self.assertTrue(patient3["any_diabetes_med"])


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

    def test_vectorized_extraction_codes_non_numeric(self):
        # Use the new CohortExtractor and criteria definition structure
        criteria_definitions = {"test_criterion": {CODE_ENTRY: ["^D/TST.*"]}}

        extractor = CohortExtractor(criteria_definitions)
        result = extractor.extract(self.events, self.index_dates)

        self.assertEqual(result.shape[0], 1)
        self.assertTrue(result.iloc[0]["test_criterion"])

    def test_vectorized_extraction_codes_numeric_in_range(self):
        # Test with numeric criteria
        criteria_definitions = {
            "test_criterion": {
                CODE_ENTRY: ["^D/TST.*"],
                NUMERIC_VALUE: {MIN_VALUE: 5},
            }
        }

        extractor = CohortExtractor(criteria_definitions)
        result = extractor.extract(self.events, self.index_dates)

        self.assertEqual(result.shape[0], 1)
        self.assertTrue(result.iloc[0]["test_criterion"])
        self.assertAlmostEqual(
            result.iloc[0]["test_criterion" + NUMERIC_VALUE_SUFFIX], 5.5
        )

    def test_vectorized_extraction_expression(self):
        # Set up a scenario with pre-defined criteria results
        self.events = pd.DataFrame(
            {
                PID_COL: [1, 1, 2, 2],
                TIMESTAMP_COL: to_datetime(
                    ["2023-05-15", "2023-05-16", "2023-05-15", "2023-05-16"]
                ),
                CONCEPT_COL: ["D/T2D", "D/STROKE", "D/T2D", "D/OTHER"],
                VALUE_COL: [None, None, None, None],
            }
        )

        # The issue is here - we need to ensure the index_dates includes both patients
        self.index_dates = pd.DataFrame(
            {
                PID_COL: [1, 2],  # Add both patients to index_dates
                TIMESTAMP_COL: to_datetime(["2023-06-01", "2023-06-01"]),
            }
        )

        criteria_definitions = {
            "diabetes": {CODE_ENTRY: ["D/T2D"]},
            "stroke": {CODE_ENTRY: ["D/STROKE"]},
            "composite": {EXPRESSION: "diabetes & ~stroke"},
        }

        extractor = CohortExtractor(criteria_definitions)
        result = extractor.extract(self.events, self.index_dates)

        # Now we can directly test without the conditional checks
        patient1 = result.loc[result[PID_COL] == 1].iloc[0]
        patient2 = result.loc[result[PID_COL] == 2].iloc[0]

        # Patient 1 has both diabetes and stroke, so composite should be False
        self.assertTrue(patient1["diabetes"])
        self.assertTrue(patient1["stroke"])
        self.assertFalse(patient1["composite"])

        # Patient 2 has diabetes but not stroke, so composite should be True
        self.assertTrue(patient2["diabetes"])
        self.assertFalse(patient2["stroke"])
        self.assertTrue(patient2["composite"])

    def test_vectorized_extraction_age(self):
        # Add DOB events to test age-based criteria
        events_with_dob = pd.DataFrame(
            {
                PID_COL: [1, 2, 3],
                TIMESTAMP_COL: to_datetime(["1968-01-01", "1978-01-01", "1958-01-01"]),
                CONCEPT_COL: ["DOB", "DOB", "DOB"],
                VALUE_COL: [None, None, None],
            }
        )

        index_dates = pd.DataFrame(
            {PID_COL: [1, 2, 3], TIMESTAMP_COL: to_datetime(["2023-06-01"] * 3)}
        )

        criteria_definitions = {"middle_aged": {MIN_AGE: 50, MAX_AGE: 59}}

        extractor = CohortExtractor(criteria_definitions)
        result = extractor.extract(events_with_dob, index_dates)

        # Patient 1 should be ~55, which is in range
        patient1 = result.loc[result[PID_COL] == 1].iloc[0]
        self.assertTrue(patient1["middle_aged"])

        # Patient 2 should be ~45, which is below range
        patient2 = result.loc[result[PID_COL] == 2].iloc[0]
        self.assertFalse(patient2["middle_aged"])

        # Patient 3 should be ~65, which is above range
        patient3 = result.loc[result[PID_COL] == 3].iloc[0]
        self.assertFalse(patient3["middle_aged"])

    def test_time_window_filtering(self):
        """Test that start_days and end_days properly filter events."""
        # Create events spanning different time periods
        events = pd.DataFrame(
            {
                PID_COL: [1, 1, 1],
                TIMESTAMP_COL: to_datetime(
                    [
                        "2023-01-01",  # 5 months before index
                        "2022-06-01",  # 12 months before index
                        "2021-06-01",  # 24 months before index
                    ]
                ),
                CONCEPT_COL: ["D/TST01", "D/TST01", "D/TST01"],
                VALUE_COL: [1.0, 2.0, 3.0],
            }
        )

        index_dates = pd.DataFrame(
            {
                PID_COL: [1],
                TIMESTAMP_COL: to_datetime(["2023-06-01"]),  # Index date
            }
        )

        # Test different time windows using start_days and end_days
        test_cases = [
            ([-180, 0], True),  # 6 months before to index - should include first event
            (
                [-365, 0],
                True,
            ),  # 1 year before to index - should include first two events
            ([-730, 0], True),  # 2 years before to index - should include all events
            ([-90, 0], False),  # 3 months before to index - should include no events
        ]

        for (start_days, end_days), expected_match in test_cases:
            criteria_definitions = {
                "test_criterion": {
                    CODE_ENTRY: ["^D/TST.*"],
                    START_DAYS: start_days,
                    END_DAYS: end_days,
                }
            }

            extractor = CohortExtractor(criteria_definitions)
            result = extractor.extract(events, index_dates)

            self.assertEqual(
                result.iloc[0]["test_criterion"],
                expected_match,
                f"Time window of {start_days} to {end_days} days should {'not ' if not expected_match else ''}match events",
            )

    def test_time_window_with_numeric_values(self):
        """Test that time_window_days works correctly with numeric criteria."""
        # Create events with numeric values at different times
        events = pd.DataFrame(
            {
                PID_COL: [1, 1, 1],
                TIMESTAMP_COL: to_datetime(
                    [
                        "2023-05-01",  # 1 month before index - value 7.5
                        "2023-01-01",  # 5 months before index - value 8.0
                        "2022-06-01",  # 12 months before index - value 8.5
                    ]
                ),
                CONCEPT_COL: ["HBA1C", "HBA1C", "HBA1C"],
                VALUE_COL: [7.5, 8.0, 8.5],
            }
        )

        index_dates = pd.DataFrame(
            {PID_COL: [1], TIMESTAMP_COL: to_datetime(["2023-06-01"])}
        )

        # Test cases with different time windows and expected values
        test_cases = [
            (
                [-90, 0],
                7.5,
            ),  # 3 months before to index - should get most recent value (7.5)
            (
                [-180, 0],
                7.5,
            ),  # 6 months before to index - should still get most recent value (7.5)
            (
                [-365, 0],
                7.5,
            ),  # 1 year before to index - should still get most recent value (7.5)
            (
                [-30, 0],
                None,
            ),  # 30 days before to index - no events, should find no values
        ]

        for (start_days, end_days), expected_value in test_cases:
            criteria_definitions = {
                "hba1c_criterion": {
                    CODE_ENTRY: ["HBA1C"],
                    NUMERIC_VALUE: {MIN_VALUE: 7.0},
                    START_DAYS: start_days,
                    END_DAYS: end_days,
                }
            }

            extractor = CohortExtractor(criteria_definitions)
            result = extractor.extract(events, index_dates)

            value_col = "hba1c_criterion" + NUMERIC_VALUE_SUFFIX

            if expected_value is None:
                self.assertFalse(
                    result.iloc[0]["hba1c_criterion"],
                    f"Time window of {start_days} to {end_days} days should not match any events",
                )
                # The value column might not exist if there are no matches
                if value_col in result.columns:
                    self.assertIsNone(
                        result.iloc[0][value_col],
                        f"Time window of {start_days} to {end_days} days should not have a numeric value",
                    )
            else:
                self.assertTrue(
                    result.iloc[0]["hba1c_criterion"],
                    f"Time window of {start_days} to {end_days} days should match events",
                )
                self.assertAlmostEqual(
                    result.iloc[0][value_col],
                    expected_value,
                    msg=f"Time window of {start_days} to {end_days} days should return value {expected_value}",
                )
