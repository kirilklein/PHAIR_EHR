"""
Standalone test script to verify extract_value functionality.
Run this directly: python tests/test_extract_value_manual.py
"""

import unittest
import pandas as pd
from pandas import to_datetime

from corebehrt.constants.cohort import NUMERIC_VALUE_SUFFIX
from corebehrt.constants.data import CONCEPT_COL, PID_COL, TIMESTAMP_COL, VALUE_COL
from corebehrt.modules.cohort_handling.advanced.extract import CohortExtractor
from corebehrt.functional.cohort_handling.stats import StatConfig, get_stratified_stats


class TestExtractValueManual(unittest.TestCase):
    """Test extract_value functionality using unittest framework."""

    def test_extract_value_basic(self):
        """Test basic extract_value functionality."""
        print("\n" + "=" * 70)
        print("TEST 1: Basic extract_value=True functionality")
        print("=" * 70)

        # Create sample MEDS data with lab tests
        events = pd.DataFrame(
            {
                PID_COL: [1, 1, 1, 2, 2, 2, 3, 3],
                TIMESTAMP_COL: to_datetime(
                    [
                        "2023-01-01",
                        "2023-01-15",
                        "2023-02-01",
                        "2023-01-05",
                        "2023-01-20",
                        "2023-02-05",
                        "2023-01-10",
                        "2023-01-25",
                    ]
                ),
                CONCEPT_COL: [
                    "L/HBA1C",
                    "L/HBA1C",
                    "D/E11.9",  # Diagnosis code
                    "L/HBA1C",
                    "L/HBA1C",
                    "L/EGFR",
                    "L/HBA1C",
                    "L/EGFR",
                ],
                VALUE_COL: [6.5, 7.2, None, 8.1, 7.8, 55.0, 9.2, 45.0],
            }
        )

        # Index dates
        index_dates = pd.DataFrame(
            {
                PID_COL: [1, 2, 3],
                TIMESTAMP_COL: to_datetime(["2023-02-10", "2023-02-15", "2023-02-01"]),
            }
        )

        # Test criteria with extract_value=True
        criteria_config = {
            "hba1c_value": {
                "codes": ["^L/HBA1C.*"],
                "extract_value": True,
                "start_days": -365,
                "end_days": 0,
            }
        }

        print(f"\nInput events (n={len(events)}):")
        print(events.to_string())

        print(f"\nIndex dates:")
        print(index_dates.to_string())

        print(f"\nCriteria config:")
        print(criteria_config)

        extractor = CohortExtractor(criteria_config)
        result = extractor.extract(events, index_dates)

        print(f"\n\nExtracted criteria (n={len(result)}):")
        print(result.to_string())

        # Verify columns
        expected_cols = ["hba1c_value", "hba1c_value" + NUMERIC_VALUE_SUFFIX]
        print(f"\n\nColumn check:")
        for col in expected_cols:
            has_col = col in result.columns
            print(f"  {col}: {'✓ PRESENT' if has_col else '✗ MISSING'}")
            self.assertIn(
                col,
                result.columns,
                f"Expected column {col} not found. Available columns: {list(result.columns)}",
            )

        # Verify values
        print(f"\n\nValue verification:")
        for pid in [1, 2, 3]:
            row = result[result[PID_COL] == pid].iloc[0]
            flag = row["hba1c_value"]
            value = row["hba1c_value" + NUMERIC_VALUE_SUFFIX]
            print(f"  Patient {pid}:")
            print(f"    Flag: {flag}")
            print(f"    Value: {value}")

            self.assertFalse(
                pd.isna(value),
                f"Expected numeric value for patient {pid}, got NaN!",
            )

        print("\n✓ TEST 1 PASSED")

    def test_extract_value_with_threshold(self):
        """Test extract_value with thresholds."""
        print("\n" + "=" * 70)
        print("TEST 2: extract_value=True with thresholds")
        print("=" * 70)

        events = pd.DataFrame(
            {
                PID_COL: [1, 1, 2],
                TIMESTAMP_COL: to_datetime(["2023-01-01", "2023-01-15", "2023-01-05"]),
                CONCEPT_COL: ["L/HBA1C", "L/HBA1C", "L/HBA1C"],
                VALUE_COL: [6.5, 7.2, 9.5],
            }
        )

        index_dates = pd.DataFrame(
            {
                PID_COL: [1, 2],
                TIMESTAMP_COL: to_datetime(["2023-02-01", "2023-02-01"]),
            }
        )

        criteria_config = {
            "hba1c_high": {
                "codes": ["^L/HBA1C.*"],
                "extract_value": True,
                "min_value": 8.0,  # Threshold
                "start_days": -365,
                "end_days": 0,
            }
        }

        print(f"\nThreshold: >= 8.0")
        print(f"\nPatient 1 most recent value: 7.2 (BELOW threshold)")
        print(f"Patient 2 most recent value: 9.5 (ABOVE threshold)")

        extractor = CohortExtractor(criteria_config)
        result = extractor.extract(events, index_dates)

        print(f"\n\nExtracted criteria:")
        print(
            result[
                [PID_COL, "hba1c_high", "hba1c_high" + NUMERIC_VALUE_SUFFIX]
            ].to_string()
        )

        # Verify Patient 1: value below threshold
        p1 = result[result[PID_COL] == 1].iloc[0]
        print(f"\n\nPatient 1 verification:")
        print(f"  Flag should be FALSE (value < 8.0): {p1['hba1c_high']}")
        print(
            f"  Value should be 7.2 (raw value): {p1['hba1c_high' + NUMERIC_VALUE_SUFFIX]}"
        )

        self.assertFalse(
            p1["hba1c_high"],
            "Flag should be False for patient 1 (value < 8.0)",
        )
        self.assertAlmostEqual(
            p1["hba1c_high" + NUMERIC_VALUE_SUFFIX],
            7.2,
            places=2,
            msg="Value should be 7.2 for patient 1",
        )

        # Verify Patient 2: value above threshold
        p2 = result[result[PID_COL] == 2].iloc[0]
        print(f"\nPatient 2 verification:")
        print(f"  Flag should be TRUE (value >= 8.0): {p2['hba1c_high']}")
        print(
            f"  Value should be 9.5 (raw value): {p2['hba1c_high' + NUMERIC_VALUE_SUFFIX]}"
        )

        self.assertTrue(
            p2["hba1c_high"],
            "Flag should be True for patient 2 (value >= 8.0)",
        )
        self.assertAlmostEqual(
            p2["hba1c_high" + NUMERIC_VALUE_SUFFIX],
            9.5,
            places=2,
            msg="Value should be 9.5 for patient 2",
        )

        print("\n✓ TEST 2 PASSED")

    def test_stats_integration(self):
        """Test that stats calculation works with extracted numeric values."""
        print("\n" + "=" * 70)
        print("TEST 3: Stats calculation with numeric values")
        print("=" * 70)

        events = pd.DataFrame(
            {
                PID_COL: [1, 2, 3],
                TIMESTAMP_COL: to_datetime(["2023-01-01", "2023-01-05", "2023-01-10"]),
                CONCEPT_COL: ["L/HBA1C", "L/HBA1C", "L/HBA1C"],
                VALUE_COL: [7.0, 8.0, 9.0],
            }
        )

        index_dates = pd.DataFrame(
            {
                PID_COL: [1, 2, 3],
                TIMESTAMP_COL: to_datetime(["2023-02-01", "2023-02-01", "2023-02-01"]),
            }
        )

        criteria_config = {
            "hba1c_value": {
                "codes": ["^L/HBA1C.*"],
                "extract_value": True,
                "start_days": -365,
                "end_days": 0,
            },
            "hba1c_flag": {
                "codes": ["^L/HBA1C.*"],
                "start_days": -365,
                "end_days": 0,
            },
        }

        extractor = CohortExtractor(criteria_config)
        criteria_df = extractor.extract(events, index_dates)

        print(f"\nExtracted criteria:")
        print(criteria_df.to_string())

        # Compute stats
        config = StatConfig()
        stats = get_stratified_stats(criteria_df, config)

        binary_stats = stats["binary"]
        numeric_stats = stats["numeric"]

        print(f"\n\nBinary stats:")
        print(binary_stats.to_string())

        print(f"\n\nNumeric stats:")
        print(numeric_stats.to_string())

        # Verify hba1c_flag in binary
        has_binary = "hba1c_flag" in binary_stats["criterion"].values
        print(f"\n\nhba1c_flag in binary stats: {'✓' if has_binary else '✗'}")
        self.assertTrue(
            has_binary,
            "hba1c_flag should be in binary stats",
        )

        # Verify hba1c_value_numeric_value in numeric
        numeric_col = "hba1c_value" + NUMERIC_VALUE_SUFFIX
        has_numeric = numeric_col in numeric_stats["criterion"].values
        print(f"{numeric_col} in numeric stats: {'✓' if has_numeric else '✗'}")
        self.assertTrue(
            has_numeric,
            f"{numeric_col} should be in numeric stats",
        )

        # Check mean value (should be (7+8+9)/3 = 8.0)
        hba1c_numeric = numeric_stats[numeric_stats["criterion"] == numeric_col]
        overall_row = hba1c_numeric[hba1c_numeric["group"] == "Overall"]
        if not overall_row.empty:
            mean_val = overall_row["mean"].iloc[0]
            print(f"\nMean HbA1c value: {mean_val} (expected: 8.0)")
            self.assertAlmostEqual(
                mean_val,
                8.0,
                places=2,
                msg="Mean should be 8.0",
            )

        print("\n✓ TEST 3 PASSED")


if __name__ == "__main__":
    unittest.main(verbosity=2)
