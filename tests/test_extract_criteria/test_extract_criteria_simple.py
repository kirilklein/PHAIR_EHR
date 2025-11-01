"""
Simple, explicit test for extract_value functionality.
This test uses minimal data with clear expectations.
"""

import unittest
import pandas as pd
import tempfile
from pathlib import Path
from pandas import to_datetime

from corebehrt.constants.data import PID_COL, TIMESTAMP_COL, CONCEPT_COL, VALUE_COL
from corebehrt.main_causal.helper.select_cohort_full import extract_criteria_from_shards


class TestExtractValueSimple(unittest.TestCase):
    """Simple test with explicit input and expected output."""

    def test_simple_hba1c_extraction(self):
        """
        Test extract_value with a minimal, clear example.

        INPUT DATA:
        ===========
        Patient 1:
          - 2023-01-01: HbA1c = 6.0
          - 2023-01-15: HbA1c = 7.5
          - Index date: 2023-02-01

        Patient 2:
          - 2023-01-10: HbA1c = 8.5
          - Index date: 2023-02-01

        EXPECTED OUTPUT:
        ================
        Patient 1:
          - hba1c_value = True (has HbA1c measurement)
          - hba1c_value_numeric_value = 7.5 (most recent before index)

        Patient 2:
          - hba1c_value = True
          - hba1c_value_numeric_value = 8.5
        """

        # ============================================================
        # STEP 1: CREATE INPUT DATA
        # ============================================================
        print("\n" + "=" * 70)
        print("STEP 1: INPUT DATA")
        print("=" * 70)

        # MEDS data: medical events with lab test values
        meds_data = pd.DataFrame(
            {
                PID_COL: [1, 1, 2],
                TIMESTAMP_COL: to_datetime(
                    [
                        "2023-01-01",  # Patient 1, first HbA1c
                        "2023-01-15",  # Patient 1, second HbA1c (most recent)
                        "2023-01-10",  # Patient 2, only HbA1c
                    ]
                ),
                CONCEPT_COL: [
                    "L/HBA1C",  # Lab test: HbA1c
                    "L/HBA1C",
                    "L/HBA1C",
                ],
                VALUE_COL: [
                    6.0,  # Patient 1's first value
                    7.5,  # Patient 1's second value (THIS ONE should be extracted)
                    8.5,  # Patient 2's value (THIS ONE should be extracted)
                ],
            }
        )

        print("\nMEDS Data (input events):")
        print(meds_data.to_string())

        # Index dates: when to extract data "before"
        index_dates = pd.DataFrame(
            {
                PID_COL: [1, 2],
                TIMESTAMP_COL: to_datetime(
                    [
                        "2023-02-01",  # Patient 1's index date
                        "2023-02-01",  # Patient 2's index date
                    ]
                ),
            }
        )

        print("\nIndex Dates:")
        print(index_dates.to_string())

        # ============================================================
        # STEP 2: DEFINE CRITERIA
        # ============================================================
        print("\n" + "=" * 70)
        print("STEP 2: CRITERIA DEFINITION")
        print("=" * 70)

        criteria_config = {
            "hba1c_value": {
                "codes": ["^L/HBA1C.*"],
                "extract_value": True,  # ← This should create _numeric_value column
                "start_days": -365,  # Look back 365 days
                "end_days": 0,  # Up to (not including) index date
            },
        }

        print("\nCriteria: hba1c_value")
        print("  - extract_value: True")
        print(
            "  - Should create: hba1c_value (flag) + hba1c_value_numeric_value (value)"
        )

        # ============================================================
        # STEP 3: RUN EXTRACTION
        # ============================================================
        print("\n" + "=" * 70)
        print("STEP 3: RUN EXTRACTION")
        print("=" * 70)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save MEDS data as parquet
            meds_path = Path(tmpdir) / "meds" / "train"
            meds_path.mkdir(parents=True)
            meds_data.to_parquet(meds_path / "data-0.parquet", index=False)

            # Run extraction
            result = extract_criteria_from_shards(
                meds_path=str(meds_path.parent),
                index_dates=index_dates,
                criteria_definitions_cfg=criteria_config,
                splits=["train"],
                pids=None,
            )

        print("\nExtraction completed!")

        # ============================================================
        # STEP 4: CHECK OUTPUT
        # ============================================================
        print("\n" + "=" * 70)
        print("STEP 4: ACTUAL OUTPUT")
        print("=" * 70)

        print(f"\nColumns: {result.columns.tolist()}")
        print(f"\nResult DataFrame:")
        print(result.to_string())

        print(f"\nData types:")
        for col in result.columns:
            print(f"  {col}: {result[col].dtype}")

        # ============================================================
        # STEP 5: VERIFY EXPECTATIONS
        # ============================================================
        print("\n" + "=" * 70)
        print("STEP 5: VERIFY EXPECTATIONS")
        print("=" * 70)

        # Check columns exist
        print("\n✓ Checking columns...")
        self.assertIn(PID_COL, result.columns, "Should have subject_id column")
        self.assertIn(
            "hba1c_value", result.columns, "Should have hba1c_value flag column"
        )
        self.assertIn(
            "hba1c_value_numeric_value",
            result.columns,
            "Should have hba1c_value_numeric_value column",
        )
        print("  ✓ All expected columns present")

        # Check Patient 1
        print("\n✓ Checking Patient 1...")
        p1 = result[result[PID_COL] == 1].iloc[0]

        print(f"  Expected: hba1c_value = True")
        print(f"  Actual:   hba1c_value = {p1['hba1c_value']}")
        self.assertTrue(p1["hba1c_value"], "Patient 1 should have HbA1c flag True")

        print(f"  Expected: hba1c_value_numeric_value = 7.5 (most recent)")
        print(
            f"  Actual:   hba1c_value_numeric_value = {p1['hba1c_value_numeric_value']}"
        )
        self.assertAlmostEqual(
            p1["hba1c_value_numeric_value"],
            7.5,
            places=1,
            msg="Patient 1 should have most recent value 7.5",
        )
        print("  ✓ Patient 1 correct!")

        # Check Patient 2
        print("\n✓ Checking Patient 2...")
        p2 = result[result[PID_COL] == 2].iloc[0]

        print(f"  Expected: hba1c_value = True")
        print(f"  Actual:   hba1c_value = {p2['hba1c_value']}")
        self.assertTrue(p2["hba1c_value"], "Patient 2 should have HbA1c flag True")

        print(f"  Expected: hba1c_value_numeric_value = 8.5")
        print(
            f"  Actual:   hba1c_value_numeric_value = {p2['hba1c_value_numeric_value']}"
        )
        self.assertAlmostEqual(
            p2["hba1c_value_numeric_value"],
            8.5,
            places=1,
            msg="Patient 2 should have value 8.5",
        )
        print("  ✓ Patient 2 correct!")

        print("\n" + "=" * 70)
        print("✓✓✓ ALL TESTS PASSED! ✓✓✓")
        print("=" * 70)


if __name__ == "__main__":
    # Run with verbose output
    unittest.main(verbosity=2)
