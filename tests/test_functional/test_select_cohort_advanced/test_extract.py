import unittest
import pandas as pd
from pandas import to_datetime
import re

# Import the constants used in the helper functions.
from corebehrt.constants.cohort import (
    AGE_AT_INDEX_DATE,
    CRITERION_FLAG,
    FINAL_MASK,
    INDEX_DATE,
    MAX_TIME,
    MIN_TIME,
    NUMERIC_VALUE,
    NUMERIC_VALUE_SUFFIX,
)
from corebehrt.constants.data import (
    CONCEPT_COL,
    PID_COL,
    TIMESTAMP_COL,
    BIRTH_CODE,
)

# Import helper functions from your extract module.
from corebehrt.functional.cohort_handling.advanced.extract import (
    compute_age_at_index_date,
    get_birth_date_for_each_patient,
    compute_code_masks,
    merge_index_dates,
    compute_time_mask_exclusive,
    rename_result,
    extract_numeric_values,
    extract_criteria_names_from_expression,
    _compile_regex,
)


class TestExtractHelpers(unittest.TestCase):
    def test_get_birth_date_for_each_patient(self):
        # Create sample events with DOB events.
        events = pd.DataFrame(
            {
                PID_COL: [1, 1, 2],
                CONCEPT_COL: [BIRTH_CODE, "D/TEST", BIRTH_CODE],
                TIMESTAMP_COL: to_datetime(["2000-01-01", "2023-01-01", "1990-05-01"]),
            }
        )
        birth_series = get_birth_date_for_each_patient(events)
        # Expect patient 1's earliest DOB is "2000-01-01" and patient 2's is "1990-05-01"
        self.assertEqual(birth_series.loc[1].strftime("%Y-%m-%d"), "2000-01-01")
        self.assertEqual(birth_series.loc[2].strftime("%Y-%m-%d"), "1990-05-01")

    def test_compute_age_at_index_date(self):
        # Create index_dates with index_date column.
        index_dates = pd.DataFrame(
            {
                PID_COL: [1, 2],
                # Provide index_date already as column "index_date"
                INDEX_DATE: to_datetime(["2023-06-01", "2023-06-01"]),
            }
        )
        # Create events with DOB events.
        events = pd.DataFrame(
            {
                PID_COL: [1, 2],
                CONCEPT_COL: [BIRTH_CODE, BIRTH_CODE],
                TIMESTAMP_COL: to_datetime(["1968-01-01", "1990-01-01"]),
            }
        )
        age_df = compute_age_at_index_date(index_dates, events)
        # For patient 1, age ~55; for patient 2, age ~33.
        age1 = age_df.loc[age_df[PID_COL] == 1, AGE_AT_INDEX_DATE].iloc[0]
        age2 = age_df.loc[age_df[PID_COL] == 2, AGE_AT_INDEX_DATE].iloc[0]
        self.assertAlmostEqual(age1, 55, delta=1)
        self.assertAlmostEqual(age2, 33, delta=1)

    def test_compile_regex(self):
        # Test the regex compilation function
        patterns = ("^A", "B$")
        compiled_regex = _compile_regex(patterns)
        self.assertIsInstance(compiled_regex, re.Pattern)
        # Check that it properly matches our patterns
        self.assertTrue(compiled_regex.search("ABC"))
        self.assertTrue(compiled_regex.search("XB"))
        self.assertFalse(compiled_regex.search("XYZ"))

    def test_compute_code_masks(self):
        # Create a sample DataFrame with codes.
        df = pd.DataFrame({PID_COL: [1, 2, 3], CONCEPT_COL: ["A123", "B123", "A999"]})
        # Allowed codes: starts with "A"; exclude codes: exactly "A123".
        mask = compute_code_masks(df, ["^A"], ["A123"])
        # Expected: For row 0, "A123" matches allowed but also excluded -> False;
        # row 1 "B123" doesn't match allowed -> False; row 2 "A999" matches allowed and not excluded -> True.
        expected = [False, False, True]
        self.assertListEqual(mask.tolist(), expected)

        # Test with empty allowed codes list
        mask = compute_code_masks(df, [], ["A123"])
        self.assertListEqual(mask.tolist(), [False, False, False])

        # Test with empty exclude codes list
        mask = compute_code_masks(df, ["^A"], [])
        self.assertListEqual(mask.tolist(), [True, False, True])

    def test_merge_index_dates(self):
        # Create a sample events DataFrame.
        events = pd.DataFrame(
            {
                PID_COL: [1, 1, 2],
                CONCEPT_COL: ["X", "Y", "Z"],
                TIMESTAMP_COL: to_datetime(["2023-05-01", "2023-05-02", "2023-05-03"]),
            }
        )
        # Create an index_dates DataFrame with TIMESTAMP_COL as the index date.
        index_dates = pd.DataFrame(
            {PID_COL: [1, 2], TIMESTAMP_COL: to_datetime(["2023-06-01", "2023-06-01"])}
        )
        merged_df = merge_index_dates(events, index_dates)
        # Check that the merged DataFrame has an "index_date" column.
        self.assertIn("index_date", merged_df.columns)
        # And that for patient 1, index_date is "2023-06-01"
        self.assertEqual(
            merged_df.loc[merged_df[PID_COL] == 1, "index_date"]
            .iloc[0]
            .strftime("%Y-%m-%d"),
            "2023-06-01",
        )

    def test_compute_time_mask_exclusive(self):
        # Create a DataFrame with MIN_TIME, MAX_TIME, and TIMESTAMP_COL.
        df = pd.DataFrame(
            {
                PID_COL: [1, 2, 3],
                MIN_TIME: to_datetime(["2023-05-01", "2023-05-01", "2023-05-01"]),
                MAX_TIME: to_datetime(["2023-06-01", "2023-06-01", "2023-04-01"]),
                TIMESTAMP_COL: to_datetime(["2023-05-15", "2023-07-01", "2023-04-15"]),
            }
        )
        mask = compute_time_mask_exclusive(df)
        # The first row has timestamp between min and max, second is after max, third is after min but max < min
        self.assertListEqual(mask.tolist(), [True, False, False])

    def test_rename_result(self):
        # Create a DataFrame with PID_COL, CRITERION_FLAG, and NUMERIC_VALUE.
        df = pd.DataFrame({PID_COL: [1], CRITERION_FLAG: [True], NUMERIC_VALUE: [7.5]})
        renamed = rename_result(df.copy(), "test_crit", has_numeric=True)
        # Expected columns: PID_COL, "test_crit", and "test_crit" + NUMERIC_VALUE_SUFFIX
        expected_cols = {PID_COL, "test_crit", "test_crit" + NUMERIC_VALUE_SUFFIX}
        self.assertEqual(set(renamed.columns), expected_cols)
        self.assertEqual(renamed.loc[0, "test_crit"], True)
        self.assertEqual(renamed.loc[0, "test_crit" + NUMERIC_VALUE_SUFFIX], 7.5)

    def test_extract_numeric_values(self):
        # Create a DataFrame with FINAL_MASK True and numeric_value values.
        df = pd.DataFrame(
            {
                PID_COL: [1, 1, 2],
                TIMESTAMP_COL: to_datetime(["2023-05-01", "2023-05-02", "2023-05-03"]),
                NUMERIC_VALUE: [5.0, 7.0, 8.0],
                FINAL_MASK: [True, True, True],
            }
        )
        # Create a flag dataframe (e.g., obtained from groupby) that indicates initial flags.
        flag_df = pd.DataFrame({PID_COL: [1, 2], CRITERION_FLAG: [True, True]})
        # Apply numeric extraction with a range: only values >= 6 and <= 8 are accepted.
        result = extract_numeric_values(df, flag_df, min_value=6, max_value=8)
        # For patient 1, only the event with numeric_value 7.0 is in range.
        # For patient 2, the event with 8.0 is in range.
        # Also, the CRITERION_FLAG should be updated to True only if a value exists.
        self.assertEqual(result.shape[0], 2)
        val1 = result.loc[result[PID_COL] == 1, NUMERIC_VALUE].iloc[0]
        val2 = result.loc[result[PID_COL] == 2, NUMERIC_VALUE].iloc[0]
        self.assertAlmostEqual(val1, 7.0)
        self.assertAlmostEqual(val2, 8.0)
        # Additionally, if a patient had no event in range, the flag would be False.
        # For simplicity here, both patients have valid events.

        # Test case for empty dataframe
        empty_df = pd.DataFrame(
            columns=[PID_COL, TIMESTAMP_COL, NUMERIC_VALUE, FINAL_MASK]
        )
        empty_df[FINAL_MASK] = False  # To ensure it's a boolean column
        result = extract_numeric_values(empty_df, flag_df)
        self.assertEqual(result.shape[0], 2)
        self.assertTrue(result[CRITERION_FLAG].eq(False).all())
        self.assertTrue(result[NUMERIC_VALUE].isna().all())

        # Test with no matching values in range
        result = extract_numeric_values(df, flag_df, min_value=10, max_value=20)
        self.assertEqual(result.shape[0], 2)
        self.assertTrue(result[CRITERION_FLAG].eq(False).all())
        self.assertTrue(result[NUMERIC_VALUE].isna().all())

    def test_extract_criteria_names_from_expression(self):
        # Test simple expressions
        expr1 = "A & B"
        criteria1 = extract_criteria_names_from_expression(expr1)
        self.assertEqual(criteria1, ("A", "B"))

        # Test complex expressions with parentheses and operators
        expr2 = "(A & B) | (C & ~D)"
        criteria2 = extract_criteria_names_from_expression(expr2)
        self.assertEqual(criteria2, ("A", "B", "C", "D"))

        # Test case sensitivity
        expr3 = "A & a"
        criteria3 = extract_criteria_names_from_expression(expr3)
        self.assertEqual(criteria3, ("A", "a"))

        # Test complex nesting
        expr4 = "((A & B) | C) & (~D | (E & F))"
        criteria4 = extract_criteria_names_from_expression(expr4)
        self.assertEqual(criteria4, ("A", "B", "C", "D", "E", "F"))

        # Test LRU cache
        # Call with the same expression should use cached result
        criteria1_again = extract_criteria_names_from_expression(expr1)
        self.assertEqual(criteria1, criteria1_again)
        # Verify they're the same object (due to caching)
        self.assertIs(criteria1, criteria1_again)


class TestComputeCodeMasks(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame with various codes
        self.df = pd.DataFrame(
            {
                PID_COL: [1, 1, 1, 2, 2, 3, 3, 3],
                CONCEPT_COL: [
                    "GENDER//Mand",  # Exact match
                    "D/C11",  # Regex match
                    "D/C11.1",  # Regex match
                    "D/I63",  # Regex match
                    "D/I64",  # Regex match
                    "D/C449",  # Exclusion code
                    "D/C449.1",  # Exclusion code
                    "OTHER",  # No match
                ],
                TIMESTAMP_COL: pd.to_datetime(["2023-01-01"] * 8),
            }
        )

    def test_exact_match(self):
        """Test exact code matching without regex."""
        codes = ["GENDER//Mand"]
        exclude_codes = []
        mask = compute_code_masks(self.df, codes, exclude_codes)

        # Only the first row should match
        self.assertTrue(mask.iloc[0])
        self.assertFalse(mask.iloc[1:].any())

    def test_regex_match(self):
        """Test regex pattern matching."""
        codes = ["^D/C11.*"]
        exclude_codes = []
        mask = compute_code_masks(self.df, codes, exclude_codes)

        # Rows 1 and 2 should match (D/C11 and D/C11.1)
        self.assertTrue(mask.iloc[1])
        self.assertTrue(mask.iloc[2])
        self.assertFalse(mask.iloc[0])
        self.assertFalse(mask.iloc[3:].any())

    def test_multiple_codes(self):
        """Test matching multiple code patterns."""
        codes = ["^D/C11.*", "^D/I6[3-4].*"]
        exclude_codes = []
        mask = compute_code_masks(self.df, codes, exclude_codes)

        # Rows 1, 2, 3, and 4 should match
        self.assertTrue(mask.iloc[1:5].all())
        self.assertFalse(mask.iloc[0])
        self.assertFalse(mask.iloc[5:].any())

    def test_exclusion_codes(self):
        """Test that exclusion codes properly filter out matches."""
        codes = ["^D/C.*"]  # Match all C codes
        exclude_codes = ["^D/C449.*"]  # Exclude C449 codes
        mask = compute_code_masks(self.df, codes, exclude_codes)

        # Rows 1 and 2 should match (D/C11 and D/C11.1)
        # Rows 5 and 6 should be excluded (D/C449 and D/C449.1)
        self.assertTrue(mask.iloc[1])
        self.assertTrue(mask.iloc[2])
        self.assertFalse(mask.iloc[5])
        self.assertFalse(mask.iloc[6])

    def test_empty_codes(self):
        """Test behavior with empty code lists."""
        # Empty allowed codes
        mask = compute_code_masks(self.df, [], [])
        self.assertFalse(mask.any())

        # Empty exclude codes
        codes = ["^D/C.*"]
        mask = compute_code_masks(self.df, codes, [])
        self.assertTrue(mask.iloc[1:3].all())  # C11 codes should match
        self.assertTrue(mask.iloc[5:7].all())  # C449 codes should match

    def test_case_sensitivity(self):
        """Test case sensitivity in matching."""
        # Create a DataFrame with mixed case codes
        mixed_case_df = pd.DataFrame(
            {
                PID_COL: [1, 1, 1],
                CONCEPT_COL: ["GENDER//Mand", "gender//mand", "GENDER//MAND"],
                TIMESTAMP_COL: pd.to_datetime(["2023-01-01"] * 3),
            }
        )

        # Test exact match
        codes = ["GENDER//Mand"]
        mask = compute_code_masks(mixed_case_df, codes, [])
        self.assertTrue(mask.iloc[0])
        self.assertFalse(mask.iloc[1:].any())

        # Test case-insensitive regex
        codes = ["(?i)gender//mand"]
        mask = compute_code_masks(mixed_case_df, codes, [])
        self.assertTrue(mask.all())

    def test_special_characters(self):
        """Test handling of special characters in codes."""
        special_df = pd.DataFrame(
            {
                PID_COL: [1, 1, 1],
                CONCEPT_COL: ["CODE//123", "CODE/123", "CODE.123"],
                TIMESTAMP_COL: pd.to_datetime(["2023-01-01"] * 3),
            }
        )

        # Test with special characters in pattern
        codes = ["CODE//123"]
        mask = compute_code_masks(special_df, codes, [])
        self.assertTrue(mask.iloc[0])
        self.assertFalse(mask.iloc[1:].any())

        # Test with regex special characters
        codes = ["CODE\\.123"]
        mask = compute_code_masks(special_df, codes, [])
        self.assertTrue(mask.iloc[2])
        self.assertFalse(mask.iloc[:2].any())

    def test_exact_match_with_double_slash(self):
        """Test exact matching of codes containing //."""
        # Create a DataFrame with various codes containing //
        double_slash_df = pd.DataFrame(
            {
                PID_COL: [1, 1, 1, 1, 1],
                CONCEPT_COL: [
                    "GENDER//Mand",  # Exact match
                    "GENDER//Mand/1",  # Should not match
                    "GENDER/Mand",  # Should not match
                    "GENDER//MAND",  # Case sensitive, should not match
                    "GENDER//Mand/2",  # Should not match
                ],
                TIMESTAMP_COL: pd.to_datetime(["2023-01-01"] * 5),
            }
        )

        # Test exact match with //
        codes = [
            "GENDER//Mand"  # matches 1, 2, -1
        ]  # this will match everything starting with GENDER//Mand
        exclude_codes = []
        mask = compute_code_masks(double_slash_df, codes, exclude_codes)

        # Only the first row should match exactly
        self.assertTrue(mask.iloc[:2].all(), "Exact match with // should work")
        self.assertTrue(mask.iloc[-1], "Also matches when starting with GENDER//Mand")
        self.assertFalse(mask.iloc[2:-1].any(), "No other rows should match")

    def test_multiple_exact_matches(self):
        double_slash_df = pd.DataFrame(
            {
                PID_COL: [1, 1, 1, 1, 1],
                CONCEPT_COL: [
                    "GENDER//Mand",  # Exact match
                    "GENDER//Mand/1",  # Should not match
                    "GENDER/Mand",  # Should not match
                    "GENDER//MAND",  # Case sensitive, should not match
                    "GENDER//Mand/2",  # Should not match
                ],
                TIMESTAMP_COL: pd.to_datetime(["2023-01-01"] * 5),
            }
        )
        # Test with multiple exact matches
        codes = ["GENDER//Mand", "GENDER//Mand/1"]
        mask = compute_code_masks(double_slash_df, codes, [])

        self.assertTrue(mask.iloc[:2].all(), "First two exact matches should work")
        self.assertFalse(mask.iloc[2:4].any(), "No other rows should match")
        self.assertTrue(mask.iloc[-1], "Last row is matching again")

    def test_exclusion(self):
        double_slash_df = pd.DataFrame(
            {
                PID_COL: [1, 1, 1, 1, 1],
                CONCEPT_COL: [
                    "GENDER//Mand",  # Exact match
                    "GENDER//Mand/1",  # Should not match
                    "GENDER/Mand",  # Should not match
                    "GENDER//MAND",  # Case sensitive, should not match
                    "GENDER//Mand/2",  # Should not match
                ],
                TIMESTAMP_COL: pd.to_datetime(["2023-01-01"] * 5),
            }
        )

        # Test with exclusion
        codes = ["GENDER//Mand"]
        exclude_codes = ["GENDER//Mand/1"]
        mask = compute_code_masks(double_slash_df, codes, exclude_codes)

        # Only first row should match, second row should be excluded
        self.assertTrue(mask.iloc[0], "Exact match should work")
        self.assertTrue(mask.iloc[-1], "Exact match should work")
        self.assertFalse(mask.iloc[1:-1].any(), "No other rows should match")

    def test_stroke_codes(self):
        """Test matching of stroke-related codes with the pattern ^(?:D|RD)/DI6[3-6].*"""
        # Create a DataFrame with various stroke-related codes
        stroke_df = pd.DataFrame(
            {
                PID_COL: [1, 1, 1, 1, 1, 1, 1, 1],
                CONCEPT_COL: [
                    "D/DI634",  # Valid D code
                    "RD/DI665",  # Valid RD code
                    "D/DI637",  # Valid D code
                    "D/DI662",  # Valid D code
                    "D/DI672",  # Invalid (outside range)
                    "RD/DI672",  # Invalid (outside range)
                    "D/DI62",  # Invalid (missing digit)
                    "OTHER",  # Invalid (completely different)
                ],
                TIMESTAMP_COL: pd.to_datetime(["2023-01-01"] * 8),
            }
        )

        # Test the stroke code pattern
        codes = ["^(?:D|RD)/DI6[3-6].*"]
        mask = compute_code_masks(stroke_df, codes, [])

        # First 4 rows should match (valid codes)
        self.assertTrue(mask.iloc[:4].all(), "Valid stroke codes should match")
        # Last 4 rows should not match (invalid codes)
        self.assertFalse(mask.iloc[4:].any(), "Invalid stroke codes should not match")

        # Test with exclusion
        exclude_codes = ["^D/DI634.*"]  # Exclude specific code
        mask = compute_code_masks(stroke_df, codes, exclude_codes)

        # First row should be excluded, but other valid codes should still match
        self.assertFalse(mask.iloc[0], "Excluded code should not match")
        self.assertTrue(mask.iloc[1:4].all(), "Other valid codes should still match")
        self.assertFalse(mask.iloc[4:].any(), "Invalid codes should still not match")

    def test_lab_value_patterns(self):
        """Test matching of complex lab value patterns with decimal points."""
        # Create a DataFrame with various lab value codes
        lab_df = pd.DataFrame(
            {
                PID_COL: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                CONCEPT_COL: [
                    "L/EGFR1.73",  # Valid - dot separator
                    "L/EGFR1,73",  # Valid - comma separator
                    "L/EGFR1.73/1",  # Valid - with additional info
                    "L/EGFR1,73/1",  # Valid - with additional info
                    "L/EGFR1.74",  # Invalid - wrong value
                    "L/EGFR1,74",  # Invalid - wrong value
                    "L/EGFR1.7",  # Invalid - missing digit
                    "L/EGFR1,7",  # Invalid - missing digit
                    "L/OTHER1.73",  # Invalid - wrong test
                    "OTHER",  # Invalid - completely different
                ],
                TIMESTAMP_COL: pd.to_datetime(["2023-01-01"] * 10),
            }
        )

        # Test the EGFR pattern
        codes = ["^L/.*EGFR.*1[.,]73.*"]
        mask = compute_code_masks(lab_df, codes, [])

        # First 4 rows should match (valid codes)
        self.assertTrue(mask.iloc[:4].all(), "Valid EGFR codes should match")
        # Last 6 rows should not match (invalid codes)
        self.assertFalse(mask.iloc[4:].any(), "Invalid EGFR codes should not match")

        # Test with exclusion
        exclude_codes = ["^L/.*EGFR.*1[.,]73/1.*"]  # Exclude codes with /1
        mask = compute_code_masks(lab_df, codes, exclude_codes)

        # First 2 rows should match, next 2 should be excluded
        self.assertTrue(mask.iloc[:2].all(), "Basic valid codes should still match")
        self.assertFalse(mask.iloc[2:4].any(), "Codes with /1 should be excluded")
        self.assertFalse(mask.iloc[4:].any(), "Invalid codes should still not match")

        # Test with multiple patterns
        codes = [
            "^L/.*EGFR.*1[.,]73.*",  # EGFR 1.73
            "^L/.*EGFR.*1[.,]74.*",  # EGFR 1.74
        ]
        mask = compute_code_masks(lab_df, codes, [])

        # First 6 rows should match (both 1.73 and 1.74)
        self.assertTrue(mask.iloc[:6].all(), "Both EGFR values should match")
        self.assertFalse(mask.iloc[6:].any(), "Other codes should not match")


if __name__ == "__main__":
    unittest.main()
