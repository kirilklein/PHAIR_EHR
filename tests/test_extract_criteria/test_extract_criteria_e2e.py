"""
End-to-end test for extract_criteria with numeric values.
This test creates sample MEDS data and verifies that extract_value=True
produces the expected columns in the output CSV.
"""

import unittest
import pandas as pd
import tempfile
from pathlib import Path
from pandas import to_datetime

from corebehrt.constants.cohort import NUMERIC_VALUE_SUFFIX
from corebehrt.constants.data import PID_COL, TIMESTAMP_COL, CONCEPT_COL, VALUE_COL
from corebehrt.main_causal.helper.select_cohort_full import extract_criteria_from_shards


class TestExtractCriteriaEndToEnd(unittest.TestCase):
    """End-to-end test verifying extract_criteria produces numeric value columns."""

    def setUp(self):
        """Create sample MEDS data with lab test numeric values."""
        # Create sample MEDS data that looks like real data
        self.meds_data = pd.DataFrame(
            {
                PID_COL: [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3],
                TIMESTAMP_COL: to_datetime(
                    [
                        "2023-01-01",
                        "2023-01-15",
                        "2023-02-01",
                        "2023-02-10",  # Patient 1
                        "2023-01-05",
                        "2023-01-20",
                        "2023-02-05",  # Patient 2
                        "2023-01-10",
                        "2023-01-25",
                        "2023-02-15",
                        "2023-02-20",  # Patient 3
                    ]
                ),
                CONCEPT_COL: [
                    "L/HBA1C",
                    "L/HBA1C",
                    "D/E11.9",
                    "L/EGFR",  # Patient 1
                    "L/HBA1C",
                    "L/HBA1C",
                    "L/EGFR",  # Patient 2
                    "L/HBA1C",
                    "L/HBA1C",
                    "L/EGFR",
                    "M/A10BA02",  # Patient 3
                ],
                VALUE_COL: [
                    6.5,
                    7.2,
                    None,
                    58.0,  # Patient 1: HbA1c values, diagnosis, eGFR
                    8.1,
                    7.8,
                    55.0,  # Patient 2: HbA1c values, eGFR
                    9.2,
                    8.5,
                    45.0,
                    None,  # Patient 3: HbA1c values, eGFR, medication
                ],
            }
        )

        # Index dates for each patient
        self.index_dates = pd.DataFrame(
            {
                PID_COL: [1, 2, 3],
                TIMESTAMP_COL: to_datetime(
                    [
                        "2023-02-15",  # Patient 1
                        "2023-02-10",  # Patient 2
                        "2023-03-01",  # Patient 3
                    ]
                ),
            }
        )

    def test_extract_criteria_creates_numeric_value_columns(self):
        """Test that extract_criteria creates _numeric_value columns for extract_value=True."""

        # Create criteria config with extract_value=True
        criteria_config = {
            "hba1c_value": {
                "codes": ["^L/HBA1C.*"],
                "extract_value": True,
                "start_days": -365,
                "end_days": 0,
            },
            "egfr_value": {
                "codes": ["^L/EGFR.*"],
                "extract_value": True,
                "start_days": -365,
                "end_days": 0,
            },
            "diabetes_dx": {
                "codes": ["^D/E11.*"],
                "start_days": -365,
                "end_days": 0,
            },
        }

        # Create temporary directory and save MEDS data as parquet
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create MEDS directory structure
            meds_path = Path(tmpdir) / "meds"
            train_path = meds_path / "train"
            train_path.mkdir(parents=True)

            # Save MEDS data as parquet shard
            shard_path = train_path / "data-0.parquet"
            self.meds_data.to_parquet(shard_path, index=False)

            # Run extraction
            result_df = extract_criteria_from_shards(
                meds_path=str(meds_path),
                index_dates=self.index_dates,
                criteria_definitions_cfg=criteria_config,
                splits=["train"],
                pids=None,
            )

        # Verify output has expected columns
        expected_cols = [
            PID_COL,
            "hba1c_value",
            "hba1c_value" + NUMERIC_VALUE_SUFFIX,
            "egfr_value",
            "egfr_value" + NUMERIC_VALUE_SUFFIX,
            "diabetes_dx",
        ]

        print("\n" + "=" * 70)
        print("EXTRACTION RESULT")
        print("=" * 70)
        print(f"Columns: {result_df.columns.tolist()}")
        print(f"\nData:\n{result_df.to_string()}")

        for col in expected_cols:
            self.assertIn(
                col,
                result_df.columns,
                f"Expected column '{col}' not found in result. Available: {result_df.columns.tolist()}",
            )

        # Verify numeric value columns have actual values
        self.assertIn("hba1c_value_numeric_value", result_df.columns)
        self.assertIn("egfr_value_numeric_value", result_df.columns)

        # Check Patient 1: most recent HbA1c before index (2023-02-15) is 7.2 from 2023-02-01
        p1_row = result_df[result_df[PID_COL] == 1].iloc[0]
        self.assertTrue(p1_row["hba1c_value"], "Patient 1 should have HbA1c flag True")
        self.assertAlmostEqual(
            p1_row["hba1c_value_numeric_value"],
            7.2,
            places=1,
            msg="Patient 1 should have HbA1c value 7.2",
        )

        # Check Patient 2: most recent HbA1c before index (2023-02-10) is 7.8 from 2023-01-20
        p2_row = result_df[result_df[PID_COL] == 2].iloc[0]
        self.assertTrue(p2_row["hba1c_value"], "Patient 2 should have HbA1c flag True")
        self.assertAlmostEqual(
            p2_row["hba1c_value_numeric_value"],
            7.8,
            places=1,
            msg="Patient 2 should have HbA1c value 7.8",
        )

        # Check Patient 3: most recent HbA1c before index (2023-03-01) is 8.5 from 2023-01-25
        p3_row = result_df[result_df[PID_COL] == 3].iloc[0]
        self.assertTrue(p3_row["hba1c_value"], "Patient 3 should have HbA1c flag True")
        self.assertAlmostEqual(
            p3_row["hba1c_value_numeric_value"],
            8.5,
            places=1,
            msg="Patient 3 should have HbA1c value 8.5",
        )

        # Verify diabetes_dx does NOT have numeric value column (extract_value not specified)
        self.assertNotIn("diabetes_dx_numeric_value", result_df.columns)

        # Check eGFR values
        self.assertAlmostEqual(p1_row["egfr_value_numeric_value"], 58.0, places=1)
        self.assertAlmostEqual(p2_row["egfr_value_numeric_value"], 55.0, places=1)
        self.assertAlmostEqual(p3_row["egfr_value_numeric_value"], 45.0, places=1)

    def test_extract_value_with_threshold(self):
        """Test that thresholds work correctly with extract_value=True."""

        criteria_config = {
            "hba1c_high": {
                "codes": ["^L/HBA1C.*"],
                "extract_value": True,
                "min_value": 8.0,
                "start_days": -365,
                "end_days": 0,
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            meds_path = Path(tmpdir) / "meds"
            train_path = meds_path / "train"
            train_path.mkdir(parents=True)

            shard_path = train_path / "data-0.parquet"
            self.meds_data.to_parquet(shard_path, index=False)

            result_df = extract_criteria_from_shards(
                meds_path=str(meds_path),
                index_dates=self.index_dates,
                criteria_definitions_cfg=criteria_config,
                splits=["train"],
                pids=None,
            )

        print("\n" + "=" * 70)
        print("EXTRACTION WITH THRESHOLD")
        print("=" * 70)
        print(result_df.to_string())

        # Patient 1: most recent HbA1c is 7.2 < 8.0 → flag False, but value still present
        p1_row = result_df[result_df[PID_COL] == 1].iloc[0]
        self.assertFalse(p1_row["hba1c_high"])
        self.assertAlmostEqual(p1_row["hba1c_high_numeric_value"], 7.2, places=1)

        # Patient 2: most recent HbA1c is 7.8 < 8.0 → flag False, but value still present
        p2_row = result_df[result_df[PID_COL] == 2].iloc[0]
        self.assertFalse(p2_row["hba1c_high"])
        self.assertAlmostEqual(p2_row["hba1c_high_numeric_value"], 7.8, places=1)

        # Patient 3: most recent HbA1c is 8.5 >= 8.0 → flag True, value present
        p3_row = result_df[result_df[PID_COL] == 3].iloc[0]
        self.assertTrue(p3_row["hba1c_high"])
        self.assertAlmostEqual(p3_row["hba1c_high_numeric_value"], 8.5, places=1)

    def test_save_to_csv_preserves_numeric_values(self):
        """Test that saving to CSV and reloading preserves numeric value columns."""

        criteria_config = {
            "hba1c_value": {
                "codes": ["^L/HBA1C.*"],
                "extract_value": True,
                "start_days": -365,
                "end_days": 0,
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create MEDS data
            meds_path = Path(tmpdir) / "meds"
            train_path = meds_path / "train"
            train_path.mkdir(parents=True)

            shard_path = train_path / "data-0.parquet"
            self.meds_data.to_parquet(shard_path, index=False)

            # Extract criteria
            result_df = extract_criteria_from_shards(
                meds_path=str(meds_path),
                index_dates=self.index_dates,
                criteria_definitions_cfg=criteria_config,
                splits=["train"],
                pids=None,
            )

            # Save to CSV (like extract_criteria.py does)
            csv_path = Path(tmpdir) / "criteria_flags.csv"
            result_df.to_csv(csv_path, index=False)

            # Reload from CSV
            reloaded_df = pd.read_csv(csv_path)

        print("\n" + "=" * 70)
        print("RELOADED FROM CSV")
        print("=" * 70)
        print(f"Columns: {reloaded_df.columns.tolist()}")
        print(f"\nData types:")
        for col in reloaded_df.columns:
            print(f"  {col}: {reloaded_df[col].dtype}")
        print(f"\nData:\n{reloaded_df.to_string()}")

        # Verify numeric value column exists
        self.assertIn("hba1c_value_numeric_value", reloaded_df.columns)

        # Verify data type is numeric
        self.assertTrue(
            pd.api.types.is_numeric_dtype(reloaded_df["hba1c_value_numeric_value"]),
            f"hba1c_value_numeric_value should be numeric, got {reloaded_df['hba1c_value_numeric_value'].dtype}",
        )

        # Verify values are preserved
        p1_value = reloaded_df[reloaded_df[PID_COL] == 1][
            "hba1c_value_numeric_value"
        ].iloc[0]
        self.assertAlmostEqual(p1_value, 7.2, places=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
