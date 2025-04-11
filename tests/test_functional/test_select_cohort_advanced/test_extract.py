import unittest
import pandas as pd
from pandas import to_datetime

# Import the constants used in the helper functions.
from corebehrt.constants.cohort import (
    AGE_AT_INDEX_DATE,
    CRITERION_FLAG,
    DELAY,
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
)

# Import helper functions from your extract module.
from corebehrt.functional.cohort_handling.advanced.extract import (
    compute_age_at_index_date,
    get_birth_date_for_each_patient,
    compute_delay_column,
    compute_time_window_columns,
    compute_code_masks,
    merge_index_dates,
    compute_time_mask,
    rename_result,
    extract_numeric_values,
)


class TestExtractHelpers(unittest.TestCase):
    def test_get_birth_date_for_each_patient(self):
        # Create sample events with DOB events.
        events = pd.DataFrame(
            {
                PID_COL: [1, 1, 2],
                CONCEPT_COL: ["DOB", "D/TEST", "DOB"],
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
                "index_date": to_datetime(["2023-06-01", "2023-06-01"]),
            }
        )
        # Create events with DOB events.
        events = pd.DataFrame(
            {
                PID_COL: [1, 2],
                CONCEPT_COL: ["DOB", "DOB"],
                TIMESTAMP_COL: to_datetime(["1968-01-01", "1990-01-01"]),
            }
        )
        age_df = compute_age_at_index_date(index_dates, events)
        # For patient 1, age ~55; for patient 2, age ~33.
        age1 = age_df.loc[age_df[PID_COL] == 1, AGE_AT_INDEX_DATE].iloc[0]
        age2 = age_df.loc[age_df[PID_COL] == 2, AGE_AT_INDEX_DATE].iloc[0]
        self.assertAlmostEqual(age1, 55, delta=1)
        self.assertAlmostEqual(age2, 33, delta=1)

    def test_compute_delay_column(self):
        # Create a simple DataFrame with concept codes.
        df = pd.DataFrame({PID_COL: [1, 2], CONCEPT_COL: ["D/ABC", "X/DEF"]})
        # Use code_groups that match "D/" and delay of 14.
        updated_df = compute_delay_column(df.copy(), ["D/"], 14)
        self.assertEqual(updated_df.loc[0, DELAY], 14)
        self.assertEqual(updated_df.loc[1, DELAY], 0)

    def test_compute_time_window_columns(self):
        # Create a DataFrame with an INDEX_DATE and DELAY columns.
        df = pd.DataFrame(
            {
                PID_COL: [1, 2],
                INDEX_DATE: to_datetime(["2023-06-01", "2023-06-01"]),
                DELAY: [14, 0],
            }
        )
        # Use a time window of 365 days.
        updated_df = compute_time_window_columns(df.copy(), 365)
        # For both rows: MIN_TIME = index_date - 365 days.
        expected_min = (to_datetime("2023-06-01") - pd.Timedelta(days=365)).strftime(
            "%Y-%m-%d"
        )
        self.assertEqual(updated_df.loc[0, MIN_TIME].strftime("%Y-%m-%d"), expected_min)
        self.assertEqual(updated_df.loc[1, MIN_TIME].strftime("%Y-%m-%d"), expected_min)
        # For row 0 (with DELAY 14), MAX_TIME = index_date + 14 days; for row 1 (DELAY 0), MAX_TIME = index_date.
        expected_max_row0 = (
            to_datetime("2023-06-01") + pd.Timedelta(days=14)
        ).strftime("%Y-%m-%d")
        expected_max_row1 = to_datetime("2023-06-01").strftime("%Y-%m-%d")
        self.assertEqual(
            updated_df.loc[0, MAX_TIME].strftime("%Y-%m-%d"), expected_max_row0
        )
        self.assertEqual(
            updated_df.loc[1, MAX_TIME].strftime("%Y-%m-%d"), expected_max_row1
        )

    def test_compute_code_masks(self):
        # Create a sample DataFrame with codes.
        df = pd.DataFrame({PID_COL: [1, 2, 3], CONCEPT_COL: ["A123", "B123", "A999"]})
        # Allowed codes: starts with "A"; exclude codes: exactly "A123".
        mask = compute_code_masks(df, ["^A"], ["A123"])
        # Expected: For row 0, "A123" matches allowed but also excluded -> False;
        # row 1 "B123" doesn't match allowed -> False; row 2 "A999" matches allowed and not excluded -> True.
        expected = [False, False, True]
        self.assertListEqual(mask.tolist(), expected)

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

    def test_compute_time_mask(self):
        # Create a DataFrame with MIN_TIME, MAX_TIME, and TIMESTAMP_COL.
        df = pd.DataFrame(
            {
                PID_COL: [1, 2],
                MIN_TIME: to_datetime(["2023-05-01", "2023-05-01"]),
                MAX_TIME: to_datetime(["2023-06-01", "2023-06-01"]),
                TIMESTAMP_COL: to_datetime(["2023-05-15", "2023-07-01"]),
            }
        )
        mask = compute_time_mask(df)
        self.assertListEqual(mask.tolist(), [True, False])

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


if __name__ == "__main__":
    unittest.main()
