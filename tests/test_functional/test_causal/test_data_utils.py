import unittest

import pandas as pd

from corebehrt.constants.data import PID_COL
from corebehrt.functional.causal.data_utils import align_df_with_pids


class TestDataUtils(unittest.TestCase):
    def setUp(self):
        # Create sample data for testing
        self.test_df = pd.DataFrame(
            {PID_COL: ["P1", "P2", "P3", "P4"], "value": [10, 20, 30, 40]}
        )

    def test_align_df_with_pids(self):
        # Test case 1: Normal case with existing PIDs
        pids = ["P1", "P3"]
        result = align_df_with_pids(self.test_df, pids)

        self.assertEqual(len(result), 2)
        self.assertTrue(all(pid in pids for pid in result[PID_COL]))
        self.assertEqual(result.iloc[0]["value"], 10)
        self.assertEqual(result.iloc[1]["value"], 30)

        # Test case 2: Empty PID list
        result_empty = align_df_with_pids(self.test_df, [])
        self.assertEqual(len(result_empty), 0)

        # Test case 3: PIDs that don't exist in DataFrame
        result_non_existing = align_df_with_pids(self.test_df, ["P5", "P6"])
        self.assertEqual(len(result_non_existing), 0)

        # Test case 4: Mixed existing and non-existing PIDs
        result_mixed = align_df_with_pids(self.test_df, ["P1", "P5"])
        self.assertEqual(len(result_mixed), 1)
        self.assertEqual(result_mixed.iloc[0][PID_COL], "P1")


if __name__ == "__main__":
    unittest.main()
