import datetime
import os
import tempfile
import unittest
from collections import defaultdict
from os.path import join

import pandas as pd

from corebehrt.constants.data import (
    ABSPOS_COL,
    CONCEPT_COL,
    PID_COL,
    TIMESTAMP_COL,
    VALUE_COL,
)
from corebehrt.functional.utils.time import get_hours_since_epoch
from corebehrt.modules.cohort_handling.outcomes import (
    OutcomeMaker,
)  # Update with your actual import path


class TestOutcomeMaker(unittest.TestCase):
    """Test outcome maker functionality"""

    def setUp(self):
        """Setup test data"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.outcomes_path = self.temp_dir.name
        # Create test concepts data
        self.concepts_plus = pd.DataFrame(
            {
                PID_COL: [1, 1, 1, 2, 2, 3, 3, 4, 4, 5],
                CONCEPT_COL: [
                    "D10.1",
                    "M112",
                    "D02.3",
                    "M112",
                    "D10.2",
                    "D10.5",
                    "M112",
                    "DOD",
                    "DI20",
                    "DOD",
                ],
                VALUE_COL: [
                    "pos",
                    "neg",
                    "pos",
                    "pos",
                    "neg",
                    "pos",
                    "neg",
                    "yes",
                    "yes",
                    "yes",
                ],
                TIMESTAMP_COL: [
                    datetime.datetime(2020, 1, 1, 10, 0),
                    datetime.datetime(2020, 1, 2, 10, 0),
                    datetime.datetime(2020, 1, 3, 10, 0),
                    datetime.datetime(2020, 1, 4, 10, 0),
                    datetime.datetime(2020, 1, 5, 10, 0),
                    datetime.datetime(2020, 1, 6, 10, 0),
                    datetime.datetime(2020, 1, 7, 10, 0),
                    datetime.datetime(2020, 1, 10, 10, 0),  # Patient 4: Death
                    datetime.datetime(
                        2020, 1, 9, 15, 0
                    ),  # Patient 4: Heart infarct (19 hours before death)
                    datetime.datetime(
                        2020, 1, 15, 10, 0
                    ),  # Patient 5: Death (with no prior heart infarct)
                ],
            }
        )

        # Add a patient with multiple events for testing more complex combinations
        combination_data = pd.DataFrame(
            {
                PID_COL: [6, 6, 6, 6, 7, 7],
                CONCEPT_COL: ["I63", "B01", "I63", "B01", "I63", "B01"],
                VALUE_COL: ["diag", "med", "diag", "med", "diag", "med"],
                TIMESTAMP_COL: [
                    datetime.datetime(2020, 2, 1, 10, 0),  # Patient 6: Stroke 1
                    datetime.datetime(
                        2020, 2, 1, 20, 0
                    ),  # Patient 6: Anticoagulant 1 (10 hours after stroke 1)
                    datetime.datetime(2020, 2, 10, 10, 0),  # Patient 6: Stroke 2
                    datetime.datetime(
                        2020, 2, 9, 10, 0
                    ),  # Patient 6: Anticoagulant 2 (24 hours before stroke 2)
                    datetime.datetime(2020, 2, 15, 10, 0),  # Patient 7: Stroke
                    datetime.datetime(
                        2020, 2, 20, 10, 0
                    ),  # Patient 7: Anticoagulant (120 hours after - outside window)
                ],
            }
        )

        self.concepts_plus = pd.concat(
            [self.concepts_plus, combination_data], ignore_index=True
        )

        # Add patients for exclusion testing
        exclusion_data = pd.DataFrame(
            {
                PID_COL: [8, 8, 9, 9],
                CONCEPT_COL: ["DI20", "DOD", "DI20", "DOD"],
                VALUE_COL: ["yes", "yes", "yes", "yes"],
                TIMESTAMP_COL: [
                    datetime.datetime(2020, 3, 1, 10, 0),  # Patient 8: MI
                    datetime.datetime(
                        2020, 3, 7, 10, 0
                    ),  # Patient 8: Death (6 days after MI)
                    datetime.datetime(2020, 3, 1, 10, 0),  # Patient 9: MI
                    datetime.datetime(
                        2020, 3, 9, 10, 0
                    ),  # Patient 9: Death (8 days after MI)
                ],
            }
        )
        self.concepts_plus = pd.concat(
            [self.concepts_plus, exclusion_data], ignore_index=True
        )

        # Patient set for testing
        self.patient_set = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_basic_outcome_creation(self):
        """Test basic outcome creation"""
        # Define test outcomes
        outcomes = {
            "TEST_OUTCOME": {
                "type": ["code"],
                "match": [["D10"]],
                "match_how": "startswith",
                "case_sensitive": True,
            }
        }

        # Create outcome maker
        outcome_maker = OutcomeMaker(outcomes)
        header_written = defaultdict(bool)

        # Get outcomes
        outcome_maker(
            self.concepts_plus,
            self.patient_set,
            self.outcomes_path,
            header_written,
        )

        # Check result
        output_file = os.path.join(self.outcomes_path, "TEST_OUTCOME.csv")
        self.assertTrue(os.path.exists(output_file))
        test_outcome = pd.read_csv(output_file, parse_dates=[TIMESTAMP_COL])

        # Should contain 3 rows (Patient 1, 2, and 3 have D10.x codes)
        self.assertEqual(len(test_outcome), 3)
        self.assertTrue(all(pid in test_outcome[PID_COL].values for pid in [1, 2, 3]))

        # Check ABSPOS calculation
        self.assertTrue(ABSPOS_COL in test_outcome.columns)
        # Verify that ABSPOS is calculated correctly for one entry
        first_timestamp = test_outcome.iloc[0][TIMESTAMP_COL]
        expected_hours = get_hours_since_epoch(first_timestamp)
        self.assertEqual(test_outcome.iloc[0][ABSPOS_COL], int(expected_hours))

    def test_exclude_and_case_sensitivity(self):
        """Test exclude feature and case sensitivity"""
        # Define test outcomes with exclude and case insensitivity
        outcomes = {
            "TEST_EXCLUDE": {
                "type": ["code"],
                "match": [["D1"]],
                "exclude": ["D10.5"],  # Exclude D10.5 specifically
                "match_how": "startswith",
                "case_sensitive": True,
            },
            "TEST_CASE_INSENSITIVE": {
                "type": ["code"],
                "match": [
                    ["m112"]
                ],  # lowercase, should match with case_sensitive=False
                "match_how": "startswith",
                "case_sensitive": False,
            },
        }

        # Create outcome maker
        outcome_maker = OutcomeMaker(outcomes)
        header_written = defaultdict(bool)

        # Get outcomes
        outcome_maker(
            self.concepts_plus,
            self.patient_set,
            self.outcomes_path,
            header_written,
        )

        # Check exclude result
        exclude_output_file = os.path.join(self.outcomes_path, "TEST_EXCLUDE.csv")
        self.assertTrue(os.path.exists(exclude_output_file))
        exclude_outcome = pd.read_csv(exclude_output_file)
        # Should contain 2 rows (Patient 1, 2 have D10.x codes, but not D10.5)
        self.assertEqual(len(exclude_outcome), 2)
        self.assertTrue(all(pid in exclude_outcome[PID_COL].values for pid in [1, 2]))

        # Check case insensitive result
        case_output_file = os.path.join(self.outcomes_path, "TEST_CASE_INSENSITIVE.csv")
        self.assertTrue(os.path.exists(case_output_file))
        case_outcome = pd.read_csv(case_output_file)
        # Should contain 3 rows (Patient 1, 2, 3 have M112 codes)
        self.assertEqual(len(case_outcome), 3)
        self.assertTrue(all(pid in case_outcome[PID_COL].values for pid in [1, 2, 3]))

    def test_negation(self):
        """Test negation feature"""
        # Define test outcomes with negation
        outcomes = {
            "NOT_M112": {
                "type": ["code"],
                "match": [["M112"]],
                "match_how": "startswith",
                "case_sensitive": True,
                "negation": True,
            }
        }

        # Create outcome maker
        outcome_maker = OutcomeMaker(outcomes)
        header_written = defaultdict(bool)

        # Get outcomes
        outcome_maker(
            self.concepts_plus,
            self.patient_set,
            self.outcomes_path,
            header_written,
        )

        # Check result
        output_file = os.path.join(self.outcomes_path, "NOT_M112.csv")
        self.assertTrue(os.path.exists(output_file))
        not_outcome = pd.read_csv(output_file)

        # Should contain rows for DOD, DI20, I63, and B01 codes (all non-M112 codes)
        expected_concepts = [
            "DOD",
            "DI20",
            "I63",
            "B01",
            "D10.1",
            "D02.3",
            "D10.2",
            "D10.5",
        ]
        # Count how many of these we expect
        expected_count = len(
            self.concepts_plus[self.concepts_plus[CONCEPT_COL].isin(expected_concepts)]
        )
        self.assertEqual(len(not_outcome), expected_count)

    def test_death_from_mi_combination(self):
        """Test combination outcome for death from myocardial infarction"""
        outcomes = {
            "DEATH_FROM_MI": {
                "combinations": {
                    "primary": {
                        "type": ["code"],
                        "match": [["DOD"]],
                        "match_how": "startswith",
                    },
                    "secondary": {
                        "type": ["code"],
                        "match": [["DI20"]],
                        "match_how": "startswith",
                    },
                    "window_hours_min": -24,  # Look for MI up to 24 hours before death
                    "window_hours_max": 0,  # up until death (not after)
                    "timestamp_source": "primary",  # Use death timestamp
                }
            }
        }

        # Create outcome maker
        outcome_maker = OutcomeMaker(outcomes)
        header_written = defaultdict(bool)

        # Get outcomes
        outcome_maker(
            self.concepts_plus,
            self.patient_set,
            self.outcomes_path,
            header_written,
        )

        # Check result
        output_file = os.path.join(self.outcomes_path, "DEATH_FROM_MI.csv")
        self.assertTrue(os.path.exists(output_file))
        death_mi_outcome = pd.read_csv(output_file, parse_dates=[TIMESTAMP_COL])

        # Should only include patient 4 who had DI20 before DOD within the window
        self.assertEqual(len(death_mi_outcome), 1)
        self.assertEqual(death_mi_outcome.iloc[0][PID_COL], 4)

        # Timestamp should be from the primary event (DOD)
        expected_timestamp = datetime.datetime(
            2020, 1, 10, 10, 0
        )  # DOD timestamp for patient 4
        self.assertEqual(death_mi_outcome.iloc[0][TIMESTAMP_COL], expected_timestamp)

    def test_stroke_with_anticoagulant_combination(self):
        """Test combination outcome for stroke with anticoagulant therapy"""
        # Define outcome for stroke with anticoagulant
        outcomes = {
            "STROKE_WITH_ANTICOAGULANT": {
                "combinations": {
                    "primary": {
                        "type": ["code"],
                        "match": [["I63"]],  # Stroke
                        "match_how": "startswith",
                    },
                    "secondary": {
                        "type": ["code"],
                        "match": [["B01"]],  # Anticoagulant
                        "match_how": "startswith",
                    },
                    "window_hours_min": -48,  # Look for anticoagulant up to 48 hours before
                    "window_hours_max": 48,  # or up to 48 hours after
                    "timestamp_source": "primary",  # Use stroke timestamp
                }
            }
        }

        # Create outcome maker
        outcome_maker = OutcomeMaker(outcomes)
        header_written = defaultdict(bool)

        # Get outcomes
        outcome_maker(
            self.concepts_plus,
            self.patient_set,
            self.outcomes_path,
            header_written,
        )

        # Add debug prints

        # Check result
        output_file = os.path.join(self.outcomes_path, "STROKE_WITH_ANTICOAGULANT.csv")
        self.assertTrue(os.path.exists(output_file))
        stroke_outcome = pd.read_csv(output_file, parse_dates=[TIMESTAMP_COL])

        # Should include both stroke events for patient 6
        # Patient 7's stroke and anticoagulant are outside the window
        self.assertEqual(len(stroke_outcome), 2)
        self.assertTrue(all(pid == 6 for pid in stroke_outcome[PID_COL].values))

        # Timestamps should match the two stroke events for patient 6
        expected_timestamps = [
            datetime.datetime(2020, 2, 1, 10, 0),  # Stroke 1
            datetime.datetime(2020, 2, 10, 10, 0),  # Stroke 2
        ]
        actual_timestamps = pd.to_datetime(stroke_outcome[TIMESTAMP_COL].values)
        self.assertTrue(all(ts in actual_timestamps for ts in expected_timestamps))

    def test_secondary_timestamp_source(self):
        """Test combination with secondary timestamp source"""
        # Define outcome using the secondary timestamp
        outcomes = {
            "MI_BEFORE_DEATH": {
                "combinations": {
                    "primary": {
                        "type": ["code"],
                        "match": [["DOD"]],
                        "match_how": "startswith",
                    },
                    "secondary": {
                        "type": ["code"],
                        "match": [["DI20"]],
                        "match_how": "startswith",
                    },
                    "window_hours_min": -24,  # Look back 24 hours from death
                    "window_hours_max": 0,  # Up until death (not after)
                    "timestamp_source": "secondary",  # Use heart infarct timestamp
                }
            }
        }

        # Create outcome maker
        outcome_maker = OutcomeMaker(outcomes)
        header_written = defaultdict(bool)

        # Get outcomes
        outcome_maker(
            self.concepts_plus,
            self.patient_set,
            self.outcomes_path,
            header_written,
        )

        # Check result
        output_file = os.path.join(self.outcomes_path, "MI_BEFORE_DEATH.csv")
        self.assertTrue(os.path.exists(output_file))
        mi_outcome = pd.read_csv(output_file, parse_dates=[TIMESTAMP_COL])

        # Should only include patient 4
        self.assertEqual(len(mi_outcome), 1)
        self.assertEqual(mi_outcome.iloc[0][PID_COL], 4)

        # Timestamp should be from the secondary event (DI20)
        expected_timestamp = datetime.datetime(
            2020, 1, 9, 15, 0
        )  # DI20 timestamp for patient 4
        self.assertEqual(mi_outcome.iloc[0][TIMESTAMP_COL], expected_timestamp)

    def test_empty_result(self):
        """Test handling of combinations that yield no results"""
        # Define outcome with impossible criteria
        outcomes = {
            "IMPOSSIBLE_COMBINATION": {
                "combinations": {
                    "primary": {
                        "type": ["code"],
                        "match": [["NONEXISTENT_CODE"]],
                        "match_how": "startswith",
                    },
                    "secondary": {
                        "type": ["code"],
                        "match": [["ANOTHER_NONEXISTENT"]],
                        "match_how": "startswith",
                    },
                    "window_hours_min": 24,
                    "window_hours_max": 24,
                }
            }
        }

        # Create outcome maker
        outcome_maker = OutcomeMaker(outcomes)
        header_written = defaultdict(bool)

        # Get outcomes
        outcome_maker(
            self.concepts_plus,
            self.patient_set,
            self.outcomes_path,
            header_written,
        )

        # Check result
        output_file = os.path.join(self.outcomes_path, "IMPOSSIBLE_COMBINATION.csv")
        # Should create a file even for empty results
        self.assertTrue(os.path.exists(output_file))
        empty_outcome = pd.read_csv(output_file)
        self.assertEqual(len(empty_outcome), 0)
        self.assertTrue(
            all(
                col in empty_outcome.columns
                for col in [PID_COL, TIMESTAMP_COL, ABSPOS_COL]
            )
        )

    def test_multiple_outcomes_together(self):
        """Test handling multiple outcome types in the same call"""
        # Define multiple outcome types
        outcomes = {
            "BASIC_OUTCOME": {
                "type": ["code"],
                "match": [["D10"]],
                "match_how": "startswith",
            },
            "COMBINATION_OUTCOME": {
                "combinations": {
                    "primary": {
                        "type": ["code"],
                        "match": [["DOD"]],
                        "match_how": "startswith",
                    },
                    "secondary": {
                        "type": ["code"],
                        "match": [["DI20"]],
                        "match_how": "startswith",
                    },
                    "window_hours_min": 24,
                    "window_hours_max": 24,
                }
            },
        }

        # Create outcome maker
        outcome_maker = OutcomeMaker(outcomes)
        header_written = defaultdict(bool)

        # Get outcomes
        outcome_maker(
            self.concepts_plus,
            self.patient_set,
            self.outcomes_path,
            header_written,
        )

        # Check result - all outcome types should produce files
        basic_file = os.path.join(self.outcomes_path, "BASIC_OUTCOME.csv")
        combo_file = os.path.join(self.outcomes_path, "COMBINATION_OUTCOME.csv")

        self.assertTrue(os.path.exists(basic_file))
        self.assertTrue(os.path.exists(combo_file))

        basic_outcome = pd.read_csv(basic_file)
        self.assertEqual(len(basic_outcome), 3)
        combo_outcome = pd.read_csv(combo_file)
        self.assertEqual(len(combo_outcome), 0)

    def test_batch_processing_and_header_writing(self):
        """Test that headers are written only once when processing multiple batches."""
        outcomes = {
            "BATCH_TEST": {
                "type": ["code"],
                "match": [["D10"]],
                "match_how": "startswith",
            }
        }
        outcome_maker = OutcomeMaker(outcomes)
        header_written = defaultdict(
            bool
        )  # This is no longer used by OutcomeMaker but required by the signature

        # First batch of data
        batch1_concepts = self.concepts_plus[self.concepts_plus[PID_COL].isin([1, 2])]
        patient_set1 = [1, 2]
        outcome_maker(batch1_concepts, patient_set1, self.outcomes_path, header_written)

        # Second batch of data, with the same OutcomeMaker instance
        batch2_concepts = self.concepts_plus[self.concepts_plus[PID_COL].isin([3])]
        patient_set2 = [3]
        outcome_maker(batch2_concepts, patient_set2, self.outcomes_path, header_written)

        # Check the output file
        output_file = join(self.outcomes_path, "BATCH_TEST.csv")
        self.assertTrue(os.path.exists(output_file))

        with open(output_file, "r") as f:
            content = f.read()
            # Check that header appears only once
            self.assertEqual(content.count("subject_id,time,abspos"), 1)

        # Check the content of the file
        batch_test_outcome = pd.read_csv(output_file)
        # Should contain 3 rows from both batches
        self.assertEqual(len(batch_test_outcome), 3)
        self.assertTrue(
            all(pid in batch_test_outcome[PID_COL].values for pid in [1, 2, 3])
        )

        # Now, create a new OutcomeMaker and write data. It should overwrite the file.
        new_outcome_maker = OutcomeMaker(outcomes)
        batch3_concepts = self.concepts_plus[self.concepts_plus[PID_COL].isin([3])]
        patient_set3 = [3]
        new_outcome_maker(
            batch3_concepts, patient_set3, self.outcomes_path, header_written
        )

        # Check that the file was overwritten
        overwritten_outcome = pd.read_csv(output_file)
        self.assertEqual(len(overwritten_outcome), 1)
        self.assertTrue(3 in overwritten_outcome[PID_COL].values)
        self.assertFalse(1 in overwritten_outcome[PID_COL].values)


if __name__ == "__main__":
    unittest.main()
