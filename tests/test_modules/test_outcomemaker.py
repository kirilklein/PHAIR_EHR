import datetime
import os
import tempfile
import unittest
from os.path import join
import json

import pandas as pd

from corebehrt.constants.data import (
    ABSPOS_COL,
    CONCEPT_COL,
    PID_COL,
    TIMESTAMP_COL,
    VALUE_COL,
)
from corebehrt.modules.cohort_handling.outcomes import OutcomeMaker


class TestOutcomeMakerEdgeCases(unittest.TestCase):
    """Additional edge case tests for OutcomeMaker"""

    def setUp(self):
        """Setup test data with edge cases"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.outcomes_path = self.temp_dir.name

        # Create comprehensive test data for edge cases
        self.edge_case_data = pd.DataFrame(
            {
                PID_COL: [1, 1, 1, 2, 2, 3, 3, 4, 5, 6, 6, 7, 7, 8, 9, 10, 10],
                CONCEPT_COL: [
                    "TEST",
                    "TEST",
                    "TEST",  # Patient 1: Same concept multiple times same day
                    "A01",
                    "A01",  # Patient 2: Duplicate entries
                    "B01",
                    "B01",  # Patient 3: Same timestamps
                    "",  # Patient 4: Empty concept
                    "TEST",  # Patient 5: Single event
                    "PRIMARY",
                    "SECONDARY",  # Patient 6: Valid combination
                    "PRIMARY",
                    "SECONDARY",  # Patient 7: Edge case timing
                    "SPECIAL_CHAR_!@#",  # Patient 8: Special characters
                    "UNICODE_ñ_ü_α",  # Patient 9: Unicode characters
                    "VERY_LONG_CONCEPT_NAME_THAT_EXCEEDS_NORMAL_LENGTH_EXPECTATIONS_AND_CONTINUES_FOR_A_WHILE_TO_TEST_HANDLING",  # Patient 10: Very long concept
                    "   WHITESPACE   ",  # Patient 10: Concept with whitespace
                ],
                VALUE_COL: [
                    "val1",
                    "val2",
                    "val1",  # Patient 1: Different values, one duplicate
                    "dup",
                    "dup",  # Patient 2: Exact duplicates
                    "same",
                    "same",  # Patient 3: Same values
                    "empty_concept",  # Patient 4: Empty concept
                    "single",  # Patient 5: Single event
                    "p",
                    "s",  # Patient 6: Valid combination
                    "p",
                    "s",  # Patient 7: Edge timing
                    "special",  # Patient 8: Special chars
                    "unicode_val",  # Patient 9: Unicode
                    "long_val",  # Patient 10: Long concept
                    "whitespace_val",  # Patient 10: Whitespace concept
                ],
                TIMESTAMP_COL: [
                    datetime.datetime(2020, 1, 1, 10, 0),
                    datetime.datetime(2020, 1, 1, 11, 0),
                    datetime.datetime(2020, 1, 1, 10, 0),  # Exact duplicate with first
                    datetime.datetime(2020, 1, 2, 10, 0),
                    datetime.datetime(2020, 1, 2, 10, 0),  # Exact duplicate
                    datetime.datetime(2020, 1, 3, 10, 0),
                    datetime.datetime(2020, 1, 3, 10, 0),  # Same timestamp
                    datetime.datetime(2020, 1, 4, 10, 0),
                    datetime.datetime(2020, 1, 5, 10, 0),
                    datetime.datetime(2020, 1, 6, 10, 0),
                    datetime.datetime(2020, 1, 6, 10, 1),  # 1 minute after (edge case)
                    datetime.datetime(2020, 1, 7, 10, 0),
                    datetime.datetime(2020, 1, 7, 10, 0),  # Exact same time
                    datetime.datetime(2020, 1, 8, 10, 0),
                    datetime.datetime(2020, 1, 9, 10, 0),
                    datetime.datetime(2020, 1, 10, 10, 0),
                    datetime.datetime(2020, 1, 10, 10, 0),  # Whitespace concept
                ],
            }
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_duplicate_entries_handling(self):
        """Test handling of exact duplicate entries"""
        outcomes = {
            "DUPLICATE_TEST": {
                "type": ["code"],
                "match": [["A01"]],
                "match_how": "exact",
            }
        }

        outcome_maker = OutcomeMaker(outcomes)
        outcome_maker(self.edge_case_data, self.outcomes_path)

        result = pd.read_csv(join(self.outcomes_path, "DUPLICATE_TEST.csv"))

        # Should include both duplicate entries for patient 2
        self.assertEqual(len(result), 2)
        self.assertTrue(all(result[PID_COL] == 2))

    def test_same_patient_multiple_events_same_day(self):
        """Test patient with multiple events on the same day"""
        outcomes = {
            "MULTIPLE_SAME_DAY": {
                "type": ["code"],
                "match": [["TEST"]],
                "match_how": "exact",
            }
        }

        outcome_maker = OutcomeMaker(outcomes)
        outcome_maker(self.edge_case_data, self.outcomes_path)

        result = pd.read_csv(join(self.outcomes_path, "MULTIPLE_SAME_DAY.csv"))

        # Patient 1 has 3 TEST events, Patient 5 has 1 TEST event
        self.assertEqual(len(result), 4)
        patient_1_events = result[result[PID_COL] == 1]
        self.assertEqual(len(patient_1_events), 3)

    def test_empty_and_whitespace_concepts(self):
        """Test handling of empty and whitespace-only concepts"""
        outcomes = {
            "EMPTY_CONCEPT": {
                "type": ["code"],
                "match": [[""]],  # Match empty string
                "match_how": "exact",
            },
            "WHITESPACE_CONCEPT": {
                "type": ["code"],
                "match": [["   WHITESPACE   "]],  # Match with whitespace
                "match_how": "exact",
            },
        }

        outcome_maker = OutcomeMaker(outcomes)
        outcome_maker(self.edge_case_data, self.outcomes_path)

        empty_result = pd.read_csv(join(self.outcomes_path, "EMPTY_CONCEPT.csv"))
        whitespace_result = pd.read_csv(
            join(self.outcomes_path, "WHITESPACE_CONCEPT.csv")
        )

        # Should find the empty concept for patient 4
        self.assertEqual(len(empty_result), 1)
        self.assertEqual(empty_result.iloc[0][PID_COL], 4)

        # Should find the whitespace concept for patient 10
        self.assertEqual(len(whitespace_result), 1)
        self.assertEqual(whitespace_result.iloc[0][PID_COL], 10)

    def test_special_characters_and_unicode(self):
        """Test handling of special characters and Unicode in concepts"""
        outcomes = {
            "SPECIAL_CHARS": {
                "type": ["code"],
                "match": [["SPECIAL_CHAR_!@#"]],
                "match_how": "exact",
            },
            "UNICODE_CHARS": {
                "type": ["code"],
                "match": [["UNICODE_ñ_ü_α"]],
                "match_how": "exact",
            },
        }

        outcome_maker = OutcomeMaker(outcomes)
        outcome_maker(self.edge_case_data, self.outcomes_path)

        special_result = pd.read_csv(join(self.outcomes_path, "SPECIAL_CHARS.csv"))
        unicode_result = pd.read_csv(join(self.outcomes_path, "UNICODE_CHARS.csv"))

        self.assertEqual(len(special_result), 1)
        self.assertEqual(special_result.iloc[0][PID_COL], 8)

        self.assertEqual(len(unicode_result), 1)
        self.assertEqual(unicode_result.iloc[0][PID_COL], 9)

    def test_very_long_concept_names(self):
        """Test handling of very long concept names"""
        outcomes = {
            "LONG_CONCEPT": {
                "type": ["code"],
                "match": [["VERY_LONG_CONCEPT_NAME"]],
                "match_how": "startswith",
            }
        }

        outcome_maker = OutcomeMaker(outcomes)
        outcome_maker(self.edge_case_data, self.outcomes_path)

        result = pd.read_csv(join(self.outcomes_path, "LONG_CONCEPT.csv"))
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0][PID_COL], 10)

    def test_combination_with_zero_window(self):
        """Test combination with zero time window (exact timing)"""
        outcomes = {
            "EXACT_TIMING": {
                "combinations": {
                    "primary": {"type": ["code"], "match": [["PRIMARY"]]},
                    "secondary": {"type": ["code"], "match": [["SECONDARY"]]},
                    "window_hours_min": 0,
                    "window_hours_max": 0,  # Exact same time only
                }
            }
        }

        outcome_maker = OutcomeMaker(outcomes)
        outcome_maker(self.edge_case_data, self.outcomes_path)

        result = pd.read_csv(join(self.outcomes_path, "EXACT_TIMING.csv"))

        # Only patient 7 has PRIMARY and SECONDARY at exact same time
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0][PID_COL], 7)

    def test_combination_with_minute_precision(self):
        """Test combination with very small time windows (minute precision)"""
        outcomes = {
            "MINUTE_PRECISION": {
                "combinations": {
                    "primary": {"type": ["code"], "match": [["PRIMARY"]]},
                    "secondary": {"type": ["code"], "match": [["SECONDARY"]]},
                    "window_hours_min": 0,
                    "window_hours_max": 0.017,  # ~1 minute
                }
            }
        }

        outcome_maker = OutcomeMaker(outcomes)
        outcome_maker(self.edge_case_data, self.outcomes_path)

        result = pd.read_csv(join(self.outcomes_path, "MINUTE_PRECISION.csv"))

        # Should include both patients 6 and 7 (6 has 1-minute gap, 7 has exact timing)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(pid in result[PID_COL].values for pid in [6, 7]))

    def test_large_patient_ids(self):
        """Test handling of large patient IDs"""
        large_id_data = pd.DataFrame(
            {
                PID_COL: [999999999, 1000000000, 9223372036854775807],  # Large integers
                CONCEPT_COL: ["TEST", "TEST", "TEST"],
                VALUE_COL: ["val", "val", "val"],
                TIMESTAMP_COL: [
                    datetime.datetime(2020, 1, 1, 10, 0),
                    datetime.datetime(2020, 1, 2, 10, 0),
                    datetime.datetime(2020, 1, 3, 10, 0),
                ],
            }
        )

        outcomes = {
            "LARGE_IDS": {"type": ["code"], "match": [["TEST"]], "match_how": "exact"}
        }

        outcome_maker = OutcomeMaker(outcomes)
        outcome_maker(large_id_data, self.outcomes_path)

        result = pd.read_csv(join(self.outcomes_path, "LARGE_IDS.csv"))
        self.assertEqual(len(result), 3)
        self.assertTrue(999999999 in result[PID_COL].values)
        self.assertTrue(9223372036854775807 in result[PID_COL].values)

    def test_extreme_timestamps(self):
        """Test handling of extreme timestamp values"""
        extreme_timestamp_data = pd.DataFrame(
            {
                PID_COL: [1, 2, 3],
                CONCEPT_COL: ["TEST", "TEST", "TEST"],
                VALUE_COL: ["val", "val", "val"],
                TIMESTAMP_COL: [
                    datetime.datetime(1900, 1, 1, 0, 0),  # Very old
                    datetime.datetime(2100, 12, 31, 23, 59),  # Far future
                    datetime.datetime(2020, 1, 1, 0, 0),  # Normal
                ],
            }
        )

        outcomes = {
            "EXTREME_TIMESTAMPS": {
                "type": ["code"],
                "match": [["TEST"]],
                "match_how": "exact",
            }
        }

        outcome_maker = OutcomeMaker(outcomes)
        outcome_maker(extreme_timestamp_data, self.outcomes_path)

        result = pd.read_csv(join(self.outcomes_path, "EXTREME_TIMESTAMPS.csv"))
        self.assertEqual(len(result), 3)

        # Check that ABSPOS is calculated correctly for extreme dates
        self.assertTrue(ABSPOS_COL in result.columns)
        self.assertTrue(all(pd.notna(result[ABSPOS_COL])))

    def test_mixed_data_types_in_values(self):
        """Test handling of different data types in VALUE_COL"""
        mixed_data = pd.DataFrame(
            {
                PID_COL: [1, 2, 3, 4, 5],
                CONCEPT_COL: ["TEST", "TEST", "TEST", "TEST", "TEST"],
                VALUE_COL: ["string", 123, 45.67, True, None],  # Mixed types
                TIMESTAMP_COL: [
                    datetime.datetime(2020, 1, i, 10, 0) for i in range(1, 6)
                ],
            }
        )

        outcomes = {
            "MIXED_VALUES": {
                "type": ["numeric_value"],  # Changed from "value" to actual column name
                "match": [["123"]],  # Match numeric as string
                "match_how": "exact",
            }
        }

        outcome_maker = OutcomeMaker(outcomes)
        outcome_maker(mixed_data, self.outcomes_path)

        result = pd.read_csv(join(self.outcomes_path, "MIXED_VALUES.csv"))
        # Should find the numeric value converted to string
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0][PID_COL], 2)

    def test_statistics_tracking_accuracy(self):
        """Test that statistics tracking is accurate across multiple batches"""
        outcomes = {
            "STATS_TEST": {"type": ["code"], "match": [["TEST"]], "match_how": "exact"}
        }

        outcome_maker = OutcomeMaker(outcomes)

        # Process first batch
        batch1 = self.edge_case_data[self.edge_case_data[PID_COL].isin([1, 5])]
        outcome_maker(batch1, self.outcomes_path)

        # Process second batch with overlapping patients
        batch2 = self.edge_case_data[self.edge_case_data[PID_COL].isin([1, 5])]
        outcome_maker(batch2, self.outcomes_path)

        stats = outcome_maker.get_stats()

        # Should have correct total outcomes (3 from patient 1, 1 from patient 5, doubled)
        self.assertEqual(stats["STATS_TEST"]["total_outcomes"], 8)
        # Should track unique subjects correctly (only 2 unique patients)
        self.assertEqual(stats["STATS_TEST"]["unique_subjects"], 2)

        # Check that JSON file is created and readable
        stats_file = join(self.outcomes_path, "outcome_statistics.json")
        self.assertTrue(os.path.exists(stats_file))

        with open(stats_file, "r") as f:
            saved_stats = json.load(f)

        self.assertEqual(saved_stats["STATS_TEST"]["total_outcomes"], 8)
        self.assertEqual(saved_stats["STATS_TEST"]["unique_subjects"], 2)

    def test_combination_with_self_reference_exclusion(self):
        """Test combination where primary and secondary refer to same concept (washout)"""
        washout_data = pd.DataFrame(
            {
                PID_COL: [1, 1, 1, 2, 2],
                CONCEPT_COL: ["EVENT", "EVENT", "EVENT", "EVENT", "EVENT"],
                VALUE_COL: ["val", "val", "val", "val", "val"],
                TIMESTAMP_COL: [
                    datetime.datetime(2020, 1, 1, 10, 0),  # Patient 1: Event 1
                    datetime.datetime(
                        2020, 1, 15, 10, 0
                    ),  # Patient 1: Event 2 (14 days later)
                    datetime.datetime(
                        2020, 1, 20, 10, 0
                    ),  # Patient 1: Event 3 (5 days after Event 2)
                    datetime.datetime(2020, 2, 1, 10, 0),  # Patient 2: Event 1
                    datetime.datetime(
                        2020, 2, 10, 10, 0
                    ),  # Patient 2: Event 2 (9 days later)
                ],
            }
        )

        outcomes = {
            "WASHOUT_30_DAYS": {
                "combinations": {
                    "primary": {"type": ["code"], "match": [["EVENT"]]},
                    "secondary": {"type": ["code"], "match": [["EVENT"]]},
                    "window_hours_min": -30 * 24,  # 30 days before
                    "window_hours_max": -1,  # Up to 1 hour before
                    "exclude": True,  # Exclude events with prior events in window
                }
            }
        }

        outcome_maker = OutcomeMaker(outcomes)
        outcome_maker(washout_data, self.outcomes_path)

        result = pd.read_csv(
            join(self.outcomes_path, "WASHOUT_30_DAYS.csv"), parse_dates=[TIMESTAMP_COL]
        )

        # Should include only first events for both patients
        # Patient 1's 2nd event (14 days after 1st) and 3rd event (19 days after 1st)
        # should be excluded as they're within the 30-day washout window
        print(f"\nDEBUG: Result has {len(result)} rows")
        print(f"Result content:\n{result}")
        print(f"Patient IDs: {result[PID_COL].tolist()}")
        print(f"Timestamps: {result[TIMESTAMP_COL].tolist()}")

        self.assertEqual(len(result), 2)

        # Check timestamps - should only have first events
        timestamps = pd.to_datetime(result[TIMESTAMP_COL])
        expected_timestamps = [
            datetime.datetime(2020, 1, 1, 10, 0),  # Patient 1: First event
            datetime.datetime(2020, 2, 1, 10, 0),  # Patient 2: First event
        ]

        # Convert expected timestamps to pandas Timestamps for comparison
        expected_timestamps_pd = [pd.Timestamp(ts) for ts in expected_timestamps]

        print(f"Expected timestamps: {expected_timestamps_pd}")
        for i, expected_ts in enumerate(expected_timestamps_pd):
            is_present = expected_ts in timestamps.values
            print(f"Expected timestamp {i} ({expected_ts}) present: {is_present}")
            self.assertTrue(
                is_present,
                f"Expected timestamp {expected_ts} not found in {timestamps.tolist()}",
            )

    def test_invalid_window_parameters(self):
        """Test handling of invalid window parameters"""
        outcomes = {
            "INVALID_WINDOW": {
                "combinations": {
                    "primary": {"type": ["code"], "match": [["PRIMARY"]]},
                    "secondary": {"type": ["code"], "match": [["SECONDARY"]]},
                    "window_hours_min": 24,  # Min > Max (invalid)
                    "window_hours_max": 0,
                }
            }
        }

        outcome_maker = OutcomeMaker(outcomes)
        # This should not crash but produce empty results
        outcome_maker(self.edge_case_data, self.outcomes_path)

        result = pd.read_csv(join(self.outcomes_path, "INVALID_WINDOW.csv"))
        self.assertEqual(len(result), 0)

    def test_memory_efficiency_large_dataset(self):
        """Test memory efficiency with a larger dataset"""
        # Create a larger dataset to test memory handling
        large_data = pd.DataFrame(
            {
                PID_COL: list(range(1, 1001)) * 10,  # 1000 patients, 10 events each
                CONCEPT_COL: ["TEST"] * 10000,
                VALUE_COL: ["val"] * 10000,
                TIMESTAMP_COL: [
                    datetime.datetime(2020, 1, 1, 10, 0) + datetime.timedelta(hours=i)
                    for i in range(10000)
                ],
            }
        )

        outcomes = {
            "LARGE_DATASET": {
                "type": ["code"],
                "match": [["TEST"]],
                "match_how": "exact",
            }
        }

        outcome_maker = OutcomeMaker(outcomes)
        # This should complete without memory errors
        outcome_maker(large_data, self.outcomes_path)

        result = pd.read_csv(join(self.outcomes_path, "LARGE_DATASET.csv"))
        self.assertEqual(len(result), 10000)

    def test_concurrent_access_file_writing(self):
        """Test behavior when multiple processes might access the same file"""
        outcomes = {
            "CONCURRENT_TEST": {
                "type": ["code"],
                "match": [["TEST"]],
                "match_how": "exact",
            }
        }

        # Simulate potential concurrent access by creating outcome maker instances
        outcome_maker1 = OutcomeMaker(outcomes)
        outcome_maker2 = OutcomeMaker(outcomes)

        # First instance writes
        outcome_maker1(self.edge_case_data, self.outcomes_path)

        # Second instance should overwrite (new instance behavior)
        outcome_maker2(self.edge_case_data, self.outcomes_path)

        result = pd.read_csv(join(self.outcomes_path, "CONCURRENT_TEST.csv"))
        # Should have results from second write
        self.assertTrue(len(result) > 0)

    def test_malformed_outcome_configuration(self):
        """Test handling of malformed outcome configurations"""
        # Test missing required fields
        malformed_outcomes = {
            "MISSING_TYPE": {
                "match": [["TEST"]],  # Missing "type"
                "match_how": "exact",
            },
            "MISSING_MATCH": {
                "type": ["code"],  # Missing "match"
                "match_how": "exact",
            },
        }

        outcome_maker = OutcomeMaker(malformed_outcomes)

        # Should handle gracefully and create empty files
        try:
            outcome_maker(self.edge_case_data, self.outcomes_path)
        except Exception as e:
            # If it throws an exception, it should be handled gracefully
            self.assertIsInstance(e, (KeyError, TypeError, ValueError))

    def test_combination_with_inclusion_flag(self):
        """Tests the `exclude: false` flag (default) for a combination."""
        outcomes = {
            "MI_WITH_DEATH_IN_7_DAYS": {
                "combinations": {
                    "primary": {
                        "type": ["code"],
                        "match": [["DI20"]],
                        "match_how": "exact",
                    },
                    "secondary": {
                        "type": ["code"],
                        "match": [["DOD"]],
                        "match_how": "exact",
                    },
                    "window_hours_min": 0,
                    "window_hours_max": 7 * 24,
                    # "exclude": False # This is the default, can be omitted
                }
            }
        }
        outcome_maker = OutcomeMaker(outcomes)
        outcome_maker(self.concepts_plus, self.outcomes_path)

        output_file = join(self.outcomes_path, "MI_WITH_DEATH_IN_7_DAYS.csv")
        self.assertTrue(os.path.exists(output_file))
        result_df = pd.read_csv(output_file)

        # Patient 8 had a death 6 days after MI -> Should be INCLUDED.
        # Patient 9 had a death 8 days after MI -> Should be EXCLUDED (outside window).
        # Patient 4 had a death <24h after MI -> Should be INCLUDED.
        self.assertEqual(len(result_df), 2)
        self.assertTrue(all(pid in result_df[PID_COL].values for pid in [4, 8]))
    
    def test_washout_period_combination(self):
        """Tests finding a primary event by excluding other instances of the same event."""
        # Patient 6 has a second stroke 9 days after the first one.
        # We want to find only the first stroke in a 30-day period.
        outcomes = {
            "FIRST_STROKE_IN_30_DAYS": {
                "combinations": {
                    "primary": {"type": ["code"], "match": [["I63"]]},
                    "secondary": {"type": ["code"], "match": [["I63"]]}, # Match against itself
                    "window_hours_min": -30 * 24, # Look for another stroke in the 30 days prior
                    "window_hours_max": -1,       # but not including the same day.
                    "exclude": True, # Keep the primary stroke if no other stroke is found before it.
                }
            }
        }
        outcome_maker = OutcomeMaker(outcomes)
        outcome_maker(self.concepts_plus, self.outcomes_path)

        output_file = join(self.outcomes_path, "FIRST_STROKE_IN_30_DAYS.csv")
        result_df = pd.read_csv(output_file, parse_dates=[TIMESTAMP_COL])

        # Patient 6's first stroke (Feb 1) should be found.
        # Patient 6's second stroke (Feb 10) should be EXCLUDED because another stroke occurred 9 days prior.
        # Patient 7's stroke should be found as it has no prior strokes.
        self.assertEqual(len(result_df), 2)
        
        # Check that we have the right events
        expected_timestamps = [
            datetime.datetime(2020, 2, 1, 10, 0), # Patient 6's first stroke
            datetime.datetime(2020, 2, 15, 10, 0), # Patient 7's stroke
        ]
        actual_timestamps = pd.to_datetime(result_df[TIMESTAMP_COL].values)
        self.assertTrue(all(ts in actual_timestamps for ts in expected_timestamps))



if __name__ == "__main__":
    unittest.main()
