import unittest
import pandas as pd
from corebehrt.functional.cohort_handling.advanced.filter import filter_by_compliant
from corebehrt.constants.data import PID_COL


class TestFilterByCompliant(unittest.TestCase):
    def setUp(self):
        # Create sample data for testing
        self.patients_info = pd.DataFrame(
            {PID_COL: [1, 2, 3, 4, 5], "age": [30, 40, 50, 60, 70]}
        )

        self.exposures = pd.DataFrame(
            {
                PID_COL: [1, 1, 2, 2, 2, 3, 4],
                "abspos": [
                    "2020-01-01",
                    "2020-02-01",
                    "2020-01-01",
                    "2020-02-01",
                    "2020-03-01",
                    "2020-01-01",
                    "2020-01-01",
                ],
            }
        )

    def test_min_exposures_2(self):
        """Test filtering with minimum 2 exposures"""
        result = filter_by_compliant(
            self.patients_info, self.exposures, min_exposures=2
        )

        # Should include:
        # - Patient 2 (3 exposures)
        # - Patient 5 (no exposures)
        print("===")
        print(result)
        expected_pids = [1, 2, 5]
        self.assertEqual(len(result), len(expected_pids))
        self.assertTrue(all(pid in expected_pids for pid in result[PID_COL]))

    def test_min_exposures_1(self):
        """Test filtering with minimum 1 exposure"""
        result = filter_by_compliant(
            self.patients_info, self.exposures, min_exposures=1
        )

        # Should include all patients except those with no exposures
        expected_pids = [1, 2, 3, 4, 5]
        self.assertEqual(len(result), len(expected_pids))
        self.assertTrue(all(pid in expected_pids for pid in result[PID_COL]))

    def test_min_exposures_3(self):
        """Test filtering with minimum 3 exposures"""
        result = filter_by_compliant(
            self.patients_info, self.exposures, min_exposures=3
        )

        # Should include:
        # - Patient 2 (3 exposures)
        # - Patient 5 (no exposures)
        expected_pids = [2, 5]
        self.assertEqual(len(result), len(expected_pids))
        self.assertTrue(all(pid in expected_pids for pid in result[PID_COL]))

    def test_empty_exposures(self):
        """Test with empty exposures dataframe"""
        empty_exposures = pd.DataFrame(columns=[PID_COL, "exposure_date"])
        result = filter_by_compliant(
            self.patients_info, empty_exposures, min_exposures=2
        )

        # Should include all patients as they have no exposures
        self.assertEqual(len(result), len(self.patients_info))
        self.assertTrue(
            all(pid in self.patients_info[PID_COL].values for pid in result[PID_COL])
        )


if __name__ == "__main__":
    unittest.main()
