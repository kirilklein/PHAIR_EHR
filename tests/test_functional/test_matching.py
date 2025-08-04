import unittest
import pandas as pd
from corebehrt.functional.cohort_handling.matching import (
    get_col_booleans,
    startswith_match,
    contains_match,
    exact_match,
)
from corebehrt.constants.data import CONCEPT_COL, VALUE_COL


class TestMatchingFunctions(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame for testing
        self.df = pd.DataFrame(
            {
                CONCEPT_COL: [
                    "COVID",
                    "COVID_TEST",
                    "POSITIVE",
                    "NEGATIVE",
                    "TEST_COVID",
                    "OTHER",
                ],
                VALUE_COL: [
                    "POSITIVE",
                    "NEGATIVE",
                    "POSITIVE",
                    "NEGATIVE",
                    "POSITIVE",
                    "NEGATIVE",
                ],
            }
        )
        self.df[f"{CONCEPT_COL}_lower"] = self.df[CONCEPT_COL].str.lower()

    def test_startswith_match_case_sensitive(self):
        # Test startswith_match with case-sensitive matching
        result = startswith_match(
            self.df, CONCEPT_COL, ["COVID", "TEST"], case_sensitive=True
        )
        expected = pd.Series([True, True, False, False, True, False], name=CONCEPT_COL)
        pd.testing.assert_series_equal(result, expected)

    def test_startswith_match_case_insensitive(self):
        # Test startswith_match with case-insensitive matching
        result = startswith_match(
            self.df, CONCEPT_COL, ["covid", "test"], case_sensitive=False
        )
        expected = pd.Series([True, True, False, False, True, False], name=CONCEPT_COL)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_contains_match_case_sensitive(self):
        # Test contains_match with case-sensitive matching
        result = contains_match(
            self.df, CONCEPT_COL, ["COVID", "TEST"], case_sensitive=True
        )

        expected = pd.Series([True, True, False, False, True, False], name=CONCEPT_COL)
        pd.testing.assert_series_equal(result, expected)

    def test_contains_match_case_insensitive(self):
        # Test contains_match with case-insensitive matching
        result = contains_match(
            self.df, CONCEPT_COL, ["covid", "test"], case_sensitive=False
        )
        expected = pd.Series([True, True, False, False, True, False], name=CONCEPT_COL)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_exact_match_case_sensitive(self):
        # Test exact_match with case-sensitive matching
        result = exact_match(
            self.df, CONCEPT_COL, ["COVID", "POSITIVE"], case_sensitive=True
        )
        expected = pd.Series([True, False, True, False, False, False], name=CONCEPT_COL)
        pd.testing.assert_series_equal(result, expected)

    def test_exact_match_case_insensitive(self):
        # Test exact_match with case-insensitive matching
        result = exact_match(
            self.df, CONCEPT_COL, ["covid", "positive"], case_sensitive=False
        )
        expected = pd.Series([True, False, True, False, False, False], name=CONCEPT_COL)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_exact_match_no_partial_matches(self):
        # Test that exact_match doesn't match partial strings
        result = exact_match(
            self.df,
            CONCEPT_COL,
            ["OVID"],
            case_sensitive=True,  # Should not match "COVID"
        )
        expected = pd.Series(
            [False, False, False, False, False, False], name=CONCEPT_COL
        )
        pd.testing.assert_series_equal(result, expected)

    def test_get_col_booleans_startswith(self):
        # Test get_col_booleans with startswith matching
        result = get_col_booleans(
            self.df,
            [CONCEPT_COL, VALUE_COL],
            [["COVID"], ["POSITIVE"]],
            match_how="startswith",
        )

        expected = [
            pd.Series([True, True, False, False, False, False], name=CONCEPT_COL),
            pd.Series([True, False, True, False, True, False], name=VALUE_COL),
        ]
        for res, exp in zip(result, expected):
            pd.testing.assert_series_equal(res, exp)

    def test_get_col_booleans_contains(self):
        # Test get_col_booleans with contains matching
        result = get_col_booleans(
            self.df,
            [CONCEPT_COL, VALUE_COL],
            [["OVID"], ["POSITIVE"]],
            match_how="contains",
        )
        expected = [
            pd.Series([True, True, False, False, True, False], name=CONCEPT_COL),
            pd.Series([True, False, True, False, True, False], name=VALUE_COL),
        ]
        for res, exp in zip(result, expected):
            pd.testing.assert_series_equal(res, exp)

    def test_get_col_booleans_exact(self):
        # Test get_col_booleans with exact matching
        result = get_col_booleans(
            self.df,
            [CONCEPT_COL, VALUE_COL],
            [["COVID", "OTHER"], ["POSITIVE"]],
            match_how="exact",
        )
        expected = [
            pd.Series([True, False, False, False, False, True], name=CONCEPT_COL),
            pd.Series([True, False, True, False, True, False], name=VALUE_COL),
        ]
        for res, exp in zip(result, expected):
            pd.testing.assert_series_equal(res, exp)

    def test_get_col_booleans_invalid_match_how(self):
        # Test get_col_booleans with an invalid match_how argument
        with self.assertRaises(ValueError) as context:
            get_col_booleans(self.df, [CONCEPT_COL], [["COVID"]], match_how="invalid")

        # Check that the error message mentions all valid options
        error_msg = str(context.exception)
        self.assertIn("invalid", error_msg)


if __name__ == "__main__":
    unittest.main()
