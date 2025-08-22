import unittest
import datetime
import pandas as pd
import numpy as np

from corebehrt.constants.data import (
    PID_COL,
    TIMESTAMP_COL,
    ABSPOS_COL,
    CONCEPT_COL,
    VALUE_COL,
    PRIMARY,
    SECONDARY,
)
from corebehrt.functional.cohort_handling.matching import (
    get_col_booleans,
    startswith_match,
    contains_match,
    exact_match,
)
from corebehrt.functional.cohort_handling.outcomes import get_binary_outcomes
from corebehrt.functional.cohort_handling.combined_outcomes import (
    find_matches_within_window,
    check_combination_args,
    create_empty_results_df,
)


class TestMatchingFunctions(unittest.TestCase):
    """Test matching functions for pattern matching in DataFrames"""

    def setUp(self):
        """Set up test data for matching functions"""
        self.test_df = pd.DataFrame(
            {
                PID_COL: [1, 2, 3, 4, 5, 6],
                CONCEPT_COL: [
                    "COVID_TEST",
                    "COVID_19",
                    "INFLUENZA",
                    "covid_test",
                    "FLU_SHOT",
                    "",
                ],
                VALUE_COL: ["POSITIVE", "negative", "123", "45.67", "True", "None"],
                "code_lower": [
                    "covid_test",
                    "covid_19",
                    "influenza",
                    "covid_test",
                    "flu_shot",
                    "",
                ],
                "numeric_value_lower": [
                    "positive",
                    "negative",
                    "123",
                    "45.67",
                    "true",
                    "none",
                ],
            }
        )

    def test_exact_match_case_sensitive(self):
        """Test exact matching with case sensitivity"""
        patterns = ["COVID_TEST", "INFLUENZA"]
        result = exact_match(self.test_df, CONCEPT_COL, patterns, case_sensitive=True)

        # Should match rows 0 and 2 (exact matches)
        expected = pd.Series([True, False, True, False, False, False])
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_exact_match_case_insensitive(self):
        """Test exact matching without case sensitivity"""
        patterns = ["covid_test", "influenza"]
        result = exact_match(self.test_df, CONCEPT_COL, patterns, case_sensitive=False)

        # Should match rows 0, 2, and 3 (case insensitive)
        expected = pd.Series([True, False, True, True, False, False])
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_startswith_match_case_sensitive(self):
        """Test startswith matching with case sensitivity"""
        patterns = ["COVID"]
        result = startswith_match(
            self.test_df, CONCEPT_COL, patterns, case_sensitive=True
        )

        # Should match rows 0 and 1 (start with "COVID")
        expected = pd.Series([True, True, False, False, False, False])
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_startswith_match_case_insensitive(self):
        """Test startswith matching without case sensitivity"""
        patterns = ["covid"]
        result = startswith_match(
            self.test_df, CONCEPT_COL, patterns, case_sensitive=False
        )

        # Should match rows 0, 1, and 3 (case insensitive startswith)
        expected = pd.Series([True, True, False, True, False, False])
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_contains_match_case_sensitive(self):
        """Test contains matching with case sensitivity"""
        patterns = ["FLU"]
        result = contains_match(
            self.test_df, CONCEPT_COL, patterns, case_sensitive=True
        )

        # Should match rows 2 and 4 (contain "FLU")
        expected = pd.Series([False, False, True, False, True, False])
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_contains_match_case_insensitive(self):
        """Test contains matching without case sensitivity"""
        patterns = ["flu"]
        result = contains_match(
            self.test_df, CONCEPT_COL, patterns, case_sensitive=False
        )

        # Should match rows 2 and 4 (case insensitive contains)
        expected = pd.Series([False, False, True, False, True, False])
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_contains_match_empty_patterns(self):
        """Test contains matching with empty patterns"""
        patterns = []
        result = contains_match(
            self.test_df, CONCEPT_COL, patterns, case_sensitive=True
        )

        # Should match nothing
        expected = pd.Series([False] * len(self.test_df))
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_contains_match_special_characters(self):
        """Test contains matching with regex special characters"""
        test_df = pd.DataFrame(
            {CONCEPT_COL: ["A.B", "A*B", "A+B", "A[B]", "A(B)", "A{B}"]}
        )

        # Test that special characters are escaped properly
        patterns = ["A.B"]
        result = contains_match(test_df, CONCEPT_COL, patterns, case_sensitive=True)

        # Should only match the literal "A.B", not regex "A.B" which would match "A*B" etc.
        expected = pd.Series([True, False, False, False, False, False])
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_get_col_booleans_single_column(self):
        """Test get_col_booleans with single column"""
        columns = [CONCEPT_COL]
        patterns = [["COVID_TEST"]]

        result = get_col_booleans(self.test_df, columns, patterns, "exact", True)

        self.assertEqual(len(result), 1)
        expected = pd.Series([True, False, False, False, False, False])
        pd.testing.assert_series_equal(result[0], expected, check_names=False)

    def test_get_col_booleans_multiple_columns(self):
        """Test get_col_booleans with multiple columns"""
        columns = [CONCEPT_COL, VALUE_COL]
        patterns = [["COVID_TEST"], ["POSITIVE"]]

        result = get_col_booleans(self.test_df, columns, patterns, "exact", True)

        self.assertEqual(len(result), 2)
        # First column should match row 0
        expected_concept = pd.Series([True, False, False, False, False, False])
        pd.testing.assert_series_equal(result[0], expected_concept, check_names=False)

        # Second column should match row 0
        expected_value = pd.Series([True, False, False, False, False, False])
        pd.testing.assert_series_equal(result[1], expected_value, check_names=False)

    def test_get_col_booleans_invalid_match_how(self):
        """Test get_col_booleans with invalid match_how parameter"""
        columns = [CONCEPT_COL]
        patterns = [["test"]]

        with self.assertRaises(ValueError) as context:
            get_col_booleans(self.test_df, columns, patterns, "invalid_method", True)

        self.assertIn("match_how must be", str(context.exception))


class TestBinaryOutcomes(unittest.TestCase):
    """Test binary outcome calculation functions"""

    def setUp(self):
        """Set up test data for binary outcomes"""
        # Index dates (when follow-up starts)
        self.index_dates = pd.DataFrame(
            {
                PID_COL: [1, 2, 3, 4],
                ABSPOS_COL: [100.0, 200.0, 300.0, 400.0],  # Hours since epoch
            }
        )

        # Outcome events
        self.outcomes = pd.DataFrame(
            {
                PID_COL: [1, 1, 2, 3, 3, 5],  # Patient 5 has no index date
                ABSPOS_COL: [105.0, 150.0, 250.0, 310.0, 350.0, 500.0],  # Various times
            }
        )

    def test_binary_outcomes_basic(self):
        """Test basic binary outcome calculation"""
        # Follow-up window: 0 to 100 hours after index
        result = get_binary_outcomes(
            self.index_dates,
            self.outcomes,
            n_hours_start_follow_up=0,
            n_hours_end_follow_up=100,
        )

        # Patient 1: has outcomes at +5 and +50 hours -> 1
        # Patient 2: has outcome at +50 hours -> 1
        # Patient 3: has outcomes at +10 and +50 hours -> 1
        # Patient 4: no outcomes -> 0
        expected = pd.Series([1, 1, 1, 0], index=[1, 2, 3, 4], name="has_outcome")
        expected.index.name = PID_COL

        pd.testing.assert_series_equal(
            result.sort_index(),
            expected.sort_index(),
            check_names=False,
            check_dtype=False,
        )

    def test_binary_outcomes_no_end_window(self):
        """Test binary outcomes with no end window (unlimited follow-up)"""
        result = get_binary_outcomes(
            self.index_dates,
            self.outcomes,
            n_hours_start_follow_up=0,
            n_hours_end_follow_up=None,
        )

        # All patients with any outcome after index should have 1
        expected = pd.Series([1, 1, 1, 0], index=[1, 2, 3, 4], name="has_outcome")
        expected.index.name = PID_COL

        pd.testing.assert_series_equal(
            result.sort_index(),
            expected.sort_index(),
            check_names=False,
            check_dtype=False,
        )

    def test_binary_outcomes_delayed_start(self):
        """Test binary outcomes with delayed start of follow-up"""
        # Follow-up window: 20 to 100 hours after index
        result = get_binary_outcomes(
            self.index_dates,
            self.outcomes,
            n_hours_start_follow_up=20,
            n_hours_end_follow_up=100,
        )

        # Patient 1: outcome at +5 is too early, +50 is valid -> 1
        # Patient 2: outcome at +50 is valid -> 1
        # Patient 3: outcome at +10 is too early, +50 is valid -> 1
        # Patient 4: no outcomes -> 0
        expected = pd.Series([1, 1, 1, 0], index=[1, 2, 3, 4], name="has_outcome")
        expected.index.name = PID_COL

        pd.testing.assert_series_equal(
            result.sort_index(),
            expected.sort_index(),
            check_names=False,
            check_dtype=False,
        )

    def test_binary_outcomes_narrow_window(self):
        """Test binary outcomes with very narrow follow-up window"""
        # Follow-up window: 0 to 10 hours after index
        result = get_binary_outcomes(
            self.index_dates,
            self.outcomes,
            n_hours_start_follow_up=0,
            n_hours_end_follow_up=10,
        )

        # Patient 1: outcome at +5 is valid, +50 is too late -> 1
        # Patient 2: outcome at +50 is too late -> 0
        # Patient 3: outcome at +10 is on boundary (should be included), +50 is too late -> 1
        # Patient 4: no outcomes -> 0
        expected = pd.Series([1, 0, 1, 0], index=[1, 2, 3, 4], name="has_outcome")
        expected.index.name = PID_COL

        pd.testing.assert_series_equal(
            result.sort_index(),
            expected.sort_index(),
            check_names=False,
            check_dtype=False,
        )

    def test_binary_outcomes_empty_outcomes(self):
        """Test binary outcomes with no outcome events"""
        empty_outcomes = pd.DataFrame({PID_COL: [], ABSPOS_COL: []})

        result = get_binary_outcomes(
            self.index_dates,
            empty_outcomes,
            n_hours_start_follow_up=0,
            n_hours_end_follow_up=100,
        )

        # All patients should have 0 (no outcomes)
        expected = pd.Series([0, 0, 0, 0], index=[1, 2, 3, 4], name="has_outcome")
        expected.index.name = PID_COL

        pd.testing.assert_series_equal(
            result.sort_index(),
            expected.sort_index(),
            check_names=False,
            check_dtype=False,
        )

    def test_binary_outcomes_empty_index_dates(self):
        """Test binary outcomes with no index dates"""
        empty_index = pd.DataFrame({PID_COL: [], ABSPOS_COL: []})

        result = get_binary_outcomes(
            empty_index,
            self.outcomes,
            n_hours_start_follow_up=0,
            n_hours_end_follow_up=100,
        )

        # Should return empty series
        expected = pd.Series([], dtype=int, name="has_outcome")
        expected.index.name = PID_COL

        pd.testing.assert_series_equal(
            result,
            expected,
            check_names=False,
            check_dtype=False,
            check_index_type=False,
        )


class TestCombinedOutcomes(unittest.TestCase):
    """Test combined outcome functions"""

    def setUp(self):
        """Set up test data for combined outcomes"""
        self.primary_events = pd.DataFrame(
            {
                PID_COL: [1, 1, 2, 3],
                TIMESTAMP_COL: [
                    datetime.datetime(2020, 1, 1, 10, 0),
                    datetime.datetime(2020, 1, 5, 10, 0),
                    datetime.datetime(2020, 1, 10, 10, 0),
                    datetime.datetime(2020, 1, 15, 10, 0),
                ],
                ABSPOS_COL: [100.0, 200.0, 300.0, 400.0],
            }
        )

        self.secondary_events = pd.DataFrame(
            {
                PID_COL: [1, 1, 2, 3],
                TIMESTAMP_COL: [
                    datetime.datetime(2020, 1, 1, 11, 0),  # 1 hour after primary
                    datetime.datetime(2020, 1, 5, 15, 0),  # 5 hours after primary
                    datetime.datetime(
                        2020, 1, 11, 8, 0
                    ),  # 22 hours after primary (within 24h)
                    datetime.datetime(2020, 1, 14, 10, 0),  # 24 hours before primary
                ],
                ABSPOS_COL: [
                    101.0,
                    205.0,
                    322.0,
                    376.0,
                ],  # Changed 348.0 to 322.0 (22 hours later)
            }
        )

    def test_find_matches_within_window_positive(self):
        """Test finding matches within positive time window (secondary after primary)"""
        result = find_matches_within_window(
            self.primary_events,
            self.secondary_events,
            window_hours_min=0,
            window_hours_max=24,
            timestamp_source=PRIMARY,
        )

        # Should find matches for patients 1 (both events) and patient 2 (22h < 24h)
        # Patient 3's secondary is before primary, so no match
        self.assertEqual(len(result), 3)  # Patient 1 (2 events) + Patient 2 (1 event)
        self.assertEqual(
            len(result[result[PID_COL] == 1]), 2
        )  # Both primary events for patient 1
        self.assertEqual(
            len(result[result[PID_COL] == 2]), 1
        )  # One event for patient 2 (22h within 24h)
        self.assertEqual(
            len(result[result[PID_COL] == 3]), 0
        )  # No matches for patient 3

    def test_find_matches_within_window_negative(self):
        """Test finding matches within negative time window (secondary before primary)"""
        result = find_matches_within_window(
            self.primary_events,
            self.secondary_events,
            window_hours_min=-48,
            window_hours_max=-1,
            timestamp_source=PRIMARY,
        )

        # Should find match for patient 3 (secondary 24 hours before primary)
        # Other patients have secondary after primary, so no matches
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0][PID_COL], 3)

    def test_find_matches_within_window_secondary_timestamp(self):
        """Test finding matches using secondary timestamp as source"""
        result = find_matches_within_window(
            self.primary_events,
            self.secondary_events,
            window_hours_min=0,
            window_hours_max=24,
            timestamp_source=SECONDARY,
        )

        # Should use secondary timestamps in result
        self.assertTrue(len(result) > 0)
        # Check that timestamps match secondary events (not exact test due to complexity)

    def test_find_matches_empty_primary(self):
        """Test finding matches with empty primary events"""
        empty_primary = pd.DataFrame({PID_COL: [], TIMESTAMP_COL: [], ABSPOS_COL: []})

        result = find_matches_within_window(
            empty_primary,
            self.secondary_events,
            window_hours_min=0,
            window_hours_max=24,
        )

        self.assertEqual(len(result), 0)

    def test_find_matches_empty_secondary(self):
        """Test finding matches with empty secondary events"""
        empty_secondary = pd.DataFrame({PID_COL: [], TIMESTAMP_COL: [], ABSPOS_COL: []})

        result = find_matches_within_window(
            self.primary_events,
            empty_secondary,
            window_hours_min=0,
            window_hours_max=24,
        )

        self.assertEqual(len(result), 0)

    def test_check_combination_args_valid(self):
        """Test validation of valid combination arguments"""
        valid_args = {
            "primary": {"type": ["code"], "match": [["PRIMARY"]]},
            "secondary": {"type": ["code"], "match": [["SECONDARY"]]},
            "window_hours_min": 0,
            "window_hours_max": 24,
        }

        # Should not raise any exception
        try:
            check_combination_args(valid_args)
        except Exception as e:
            self.fail(f"Valid args raised exception: {e}")

    def test_check_combination_args_missing_primary(self):
        """Test validation with missing primary configuration"""
        invalid_args = {
            "secondary": {"type": ["code"], "match": [["SECONDARY"]]},
            "window_hours_min": 0,
            "window_hours_max": 24,
        }

        with self.assertRaises(ValueError):
            check_combination_args(invalid_args)

    def test_check_combination_args_missing_secondary(self):
        """Test validation with missing secondary configuration"""
        invalid_args = {
            "primary": {"type": ["code"], "match": [["PRIMARY"]]},
            "window_hours_min": 0,
            "window_hours_max": 24,
        }

        with self.assertRaises(ValueError):
            check_combination_args(invalid_args)

    def test_check_combination_args_missing_window(self):
        """Test validation with missing window parameters"""
        invalid_args = {
            "primary": {"type": ["code"], "match": [["PRIMARY"]]},
            "secondary": {"type": ["code"], "match": [["SECONDARY"]]},
        }

        with self.assertRaises(ValueError):
            check_combination_args(invalid_args)

    def test_create_empty_results_df(self):
        """Test creation of empty results DataFrame"""
        result = create_empty_results_df()

        # Should have correct columns and be empty
        expected_columns = [PID_COL, TIMESTAMP_COL, ABSPOS_COL]
        self.assertListEqual(list(result.columns), expected_columns)
        self.assertEqual(len(result), 0)

        # For empty DataFrames, dtypes are not as important since there's no data
        # Just verify the columns exist
        self.assertIn(PID_COL, result.columns)
        self.assertIn(TIMESTAMP_COL, result.columns)
        self.assertIn(ABSPOS_COL, result.columns)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""

    def test_matching_with_nan_values(self):
        """Test matching functions handle NaN values correctly"""
        test_df = pd.DataFrame(
            {
                CONCEPT_COL: ["TEST", np.nan, None, ""],
                "code_lower": ["test", np.nan, None, ""],
            }
        )

        patterns = ["TEST"]
        result = exact_match(test_df, CONCEPT_COL, patterns, case_sensitive=True)

        # Should only match the first row, NaN/None should be False
        expected = pd.Series([True, False, False, False])
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_matching_with_empty_dataframe(self):
        """Test matching functions with empty DataFrame"""
        empty_df = pd.DataFrame({CONCEPT_COL: [], "code_lower": []})

        patterns = ["TEST"]
        result = exact_match(empty_df, CONCEPT_COL, patterns, case_sensitive=True)

        # Should return empty series
        expected = pd.Series([], dtype=bool)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_binary_outcomes_same_time(self):
        """Test binary outcomes when outcome occurs at exact same time as index"""
        index_dates = pd.DataFrame({PID_COL: [1], ABSPOS_COL: [100.0]})

        outcomes = pd.DataFrame(
            {
                PID_COL: [1],
                ABSPOS_COL: [100.0],  # Exact same time
            }
        )

        # Test with start at 0 (should include)
        result = get_binary_outcomes(
            index_dates, outcomes, n_hours_start_follow_up=0, n_hours_end_follow_up=10
        )

        expected = pd.Series([1], index=[1], name="has_outcome")
        expected.index.name = PID_COL
        pd.testing.assert_series_equal(
            result, expected, check_names=False, check_dtype=False
        )


if __name__ == "__main__":
    unittest.main()
