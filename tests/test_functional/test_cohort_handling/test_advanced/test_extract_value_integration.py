"""Integration tests for extract_value functionality in criteria extraction."""

import unittest
import pandas as pd
from pandas import to_datetime

from corebehrt.constants.cohort import NUMERIC_VALUE, NUMERIC_VALUE_SUFFIX
from corebehrt.constants.data import CONCEPT_COL, PID_COL, TIMESTAMP_COL
from corebehrt.modules.cohort_handling.advanced.extract import CohortExtractor


class TestExtractValueIntegration(unittest.TestCase):
    """Integration tests for extract_value flag in criteria extraction."""

    def setUp(self):
        """Set up test data with medical events including lab values."""
        # Create sample MEDS data with lab tests
        self.events = pd.DataFrame(
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
                NUMERIC_VALUE: [6.5, 7.2, None, 8.1, 7.8, 55.0, 9.2, 45.0],
            }
        )

        # Index dates (reference time points)
        self.index_dates = pd.DataFrame(
            {
                PID_COL: [1, 2, 3],
                TIMESTAMP_COL: to_datetime(["2023-02-10", "2023-02-15", "2023-02-01"]),
            }
        )

    def test_extract_value_true_creates_numeric_column(self):
        """Test that extract_value=true creates both binary flag and numeric value columns."""
        criteria_config = {
            "hba1c_value": {
                "codes": ["^L/HBA1C.*"],
                "extract_value": True,
                "start_days": -365,
                "end_days": 0,
            }
        }

        extractor = CohortExtractor(criteria_config)
        result = extractor.extract(self.events, self.index_dates)

        # Check that result has the expected columns
        self.assertIn("hba1c_value", result.columns)
        self.assertIn("hba1c_value" + NUMERIC_VALUE_SUFFIX, result.columns)

        # Check that we have all patients
        self.assertEqual(len(result), 3)
        self.assertSetEqual(set(result[PID_COL]), {1, 2, 3})

        # Patient 1: should have binary flag True and value 7.2 (most recent before index)
        p1_row = result[result[PID_COL] == 1].iloc[0]
        self.assertTrue(p1_row["hba1c_value"])
        self.assertAlmostEqual(p1_row["hba1c_value" + NUMERIC_VALUE_SUFFIX], 7.2)

        # Patient 2: should have binary flag True and value 7.8 (most recent before index)
        p2_row = result[result[PID_COL] == 2].iloc[0]
        self.assertTrue(p2_row["hba1c_value"])
        self.assertAlmostEqual(p2_row["hba1c_value" + NUMERIC_VALUE_SUFFIX], 7.8)

        # Patient 3: should have binary flag True and value 9.2 (only one before index)
        p3_row = result[result[PID_COL] == 3].iloc[0]
        self.assertTrue(p3_row["hba1c_value"])
        self.assertAlmostEqual(p3_row["hba1c_value" + NUMERIC_VALUE_SUFFIX], 9.2)

    def test_extract_value_false_only_creates_binary(self):
        """Test that extract_value=false (default) only creates binary flag column."""
        criteria_config = {
            "hba1c_flag": {
                "codes": ["^L/HBA1C.*"],
                "start_days": -365,
                "end_days": 0,
            }
        }

        extractor = CohortExtractor(criteria_config)
        result = extractor.extract(self.events, self.index_dates)

        # Check that result has binary flag but NOT numeric value column
        self.assertIn("hba1c_flag", result.columns)
        self.assertNotIn("hba1c_flag" + NUMERIC_VALUE_SUFFIX, result.columns)

        # All patients should have the flag as True (they all have HbA1c measurements)
        self.assertTrue(result["hba1c_flag"].all())

    def test_extract_value_with_threshold_flag_respects_threshold(self):
        """Test that when extract_value=true with thresholds, flag respects threshold but value doesn't."""
        criteria_config = {
            "hba1c_high": {
                "codes": ["^L/HBA1C.*"],
                "extract_value": True,
                "min_value": 8.0,
                "start_days": -365,
                "end_days": 0,
            }
        }

        extractor = CohortExtractor(criteria_config)
        result = extractor.extract(self.events, self.index_dates)

        # Patient 1: most recent value is 7.2 (< 8.0)
        # Flag should be False, but numeric_value should still be 7.2
        p1_row = result[result[PID_COL] == 1].iloc[0]
        self.assertFalse(p1_row["hba1c_high"])
        self.assertAlmostEqual(p1_row["hba1c_high" + NUMERIC_VALUE_SUFFIX], 7.2)

        # Patient 2: most recent value is 7.8 (< 8.0)
        # Flag should be False, but numeric_value should still be 7.8
        p2_row = result[result[PID_COL] == 2].iloc[0]
        self.assertFalse(p2_row["hba1c_high"])
        self.assertAlmostEqual(p2_row["hba1c_high" + NUMERIC_VALUE_SUFFIX], 7.8)

        # Patient 3: most recent value is 9.2 (>= 8.0)
        # Flag should be True, and numeric_value should be 9.2
        p3_row = result[result[PID_COL] == 3].iloc[0]
        self.assertTrue(p3_row["hba1c_high"])
        self.assertAlmostEqual(p3_row["hba1c_high" + NUMERIC_VALUE_SUFFIX], 9.2)

    def test_multiple_criteria_with_different_extract_value_settings(self):
        """Test multiple criteria with mixed extract_value settings."""
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

        extractor = CohortExtractor(criteria_config)
        result = extractor.extract(self.events, self.index_dates)

        # Check columns exist
        self.assertIn("hba1c_value", result.columns)
        self.assertIn("hba1c_value" + NUMERIC_VALUE_SUFFIX, result.columns)
        self.assertIn("egfr_value", result.columns)
        self.assertIn("egfr_value" + NUMERIC_VALUE_SUFFIX, result.columns)
        self.assertIn("diabetes_dx", result.columns)
        self.assertNotIn("diabetes_dx" + NUMERIC_VALUE_SUFFIX, result.columns)

        # Patient 1: has HbA1c but no eGFR, has diabetes diagnosis
        p1_row = result[result[PID_COL] == 1].iloc[0]
        self.assertTrue(p1_row["hba1c_value"])
        self.assertAlmostEqual(p1_row["hba1c_value" + NUMERIC_VALUE_SUFFIX], 7.2)
        self.assertFalse(p1_row["egfr_value"])  # No eGFR measurement
        self.assertTrue(pd.isna(p1_row["egfr_value" + NUMERIC_VALUE_SUFFIX]))
        self.assertTrue(p1_row["diabetes_dx"])

        # Patient 2: has HbA1c and eGFR, no diabetes diagnosis
        p2_row = result[result[PID_COL] == 2].iloc[0]
        self.assertTrue(p2_row["hba1c_value"])
        self.assertAlmostEqual(p2_row["hba1c_value" + NUMERIC_VALUE_SUFFIX], 7.8)
        self.assertTrue(p2_row["egfr_value"])
        self.assertAlmostEqual(p2_row["egfr_value" + NUMERIC_VALUE_SUFFIX], 55.0)
        self.assertFalse(p2_row["diabetes_dx"])

        # Patient 3: has HbA1c and eGFR, no diabetes diagnosis
        p3_row = result[result[PID_COL] == 3].iloc[0]
        self.assertTrue(p3_row["hba1c_value"])
        self.assertAlmostEqual(p3_row["hba1c_value" + NUMERIC_VALUE_SUFFIX], 9.2)
        self.assertTrue(p3_row["egfr_value"])
        self.assertAlmostEqual(p3_row["egfr_value" + NUMERIC_VALUE_SUFFIX], 45.0)
        self.assertFalse(p3_row["diabetes_dx"])

    def test_stats_compatibility(self):
        """Test that extracted numeric values work correctly with stats calculation."""
        from corebehrt.functional.cohort_handling.stats import (
            StatConfig,
            get_stratified_stats,
        )

        # Extract criteria with numeric values
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
        criteria_df = extractor.extract(self.events, self.index_dates)

        # Compute stats
        config = StatConfig()
        stats = get_stratified_stats(criteria_df, config)

        # Check that hba1c_flag appears in BINARY stats
        binary_stats = stats["binary"]
        self.assertTrue(
            "hba1c_flag" in binary_stats["criterion"].values,
            "hba1c_flag should be in binary stats",
        )

        # Check that hba1c_value_numeric_value appears in NUMERIC stats
        numeric_stats = stats["numeric"]
        self.assertTrue(
            "hba1c_value" + NUMERIC_VALUE_SUFFIX in numeric_stats["criterion"].values,
            "hba1c_value_numeric_value should be in numeric stats",
        )

        # Verify numeric stats contain expected columns
        hba1c_numeric = numeric_stats[
            numeric_stats["criterion"] == "hba1c_value" + NUMERIC_VALUE_SUFFIX
        ]
        self.assertTrue(len(hba1c_numeric) > 0, "Should have numeric stats for HbA1c")

        # Check that mean is calculated (should be around (7.2 + 7.8 + 9.2) / 3 = 8.07)
        overall_row = hba1c_numeric[hba1c_numeric["group"] == "Overall"]
        if not overall_row.empty:
            mean_val = overall_row["mean"].iloc[0]
            self.assertAlmostEqual(mean_val, 8.066666, places=4)

    def test_no_matching_events(self):
        """Test behavior when no events match the criteria codes."""
        criteria_config = {
            "creatinine": {
                "codes": ["^L/CREATININE.*"],
                "extract_value": True,
                "start_days": -365,
                "end_days": 0,
            }
        }

        extractor = CohortExtractor(criteria_config)
        result = extractor.extract(self.events, self.index_dates)

        # All patients should have False flags and NaN values
        self.assertTrue(result["creatinine"].eq(False).all())
        self.assertTrue(result["creatinine" + NUMERIC_VALUE_SUFFIX].isna().all())

    def test_time_window_filtering(self):
        """Test that time windows are correctly applied when extract_value=true."""
        # Use a very short time window
        criteria_config = {
            "hba1c_recent": {
                "codes": ["^L/HBA1C.*"],
                "extract_value": True,
                "start_days": -10,  # Only 10 days before index
                "end_days": 0,
            }
        }

        extractor = CohortExtractor(criteria_config)
        result = extractor.extract(self.events, self.index_dates)

        # Patient 1 (index: 2023-02-10):
        # Events:
        # - 2023-01-01 L/HBA1C 6.5 (40 days before) - outside window
        # - 2023-01-15 L/HBA1C 7.2 (26 days before) - outside window
        # - 2023-02-01 D/E11.9 (diagnosis, not HbA1c!)
        #
        # Window with start_days=-10, end_days=0: [2023-01-31, 2023-02-10]
        # Patient 1 has NO HbA1c within this window
        p1_row = result[result[PID_COL] == 1].iloc[0]
        self.assertFalse(p1_row["hba1c_recent"])
        self.assertTrue(pd.isna(p1_row["hba1c_recent" + NUMERIC_VALUE_SUFFIX]))

        # Patient 2 (index: 2023-02-15):
        # Events:
        # - 2023-01-05 L/HBA1C 8.1 (41 days) - outside window
        # - 2023-01-20 L/HBA1C 7.8 (26 days) - outside window
        # - 2023-02-05 L/EGFR 55.0 (10 days) - inside window but it's EGFR, not HbA1c
        #
        # Window with start_days=-10, end_days=0: [2023-02-05, 2023-02-15]
        # Patient 2 has NO HbA1c within this window
        p2_row = result[result[PID_COL] == 2].iloc[0]
        self.assertFalse(p2_row["hba1c_recent"])
        self.assertTrue(pd.isna(p2_row["hba1c_recent" + NUMERIC_VALUE_SUFFIX]))


if __name__ == "__main__":
    unittest.main()
