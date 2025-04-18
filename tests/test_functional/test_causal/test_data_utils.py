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
        self.assertListEqual(result[PID_COL].tolist(), pids)
        self.assertEqual(result.iloc[0]["value"], 10)
        self.assertEqual(result.iloc[1]["value"], 30)

        # Test case 2: Empty PID list
        result_empty = align_df_with_pids(self.test_df, [])
        self.assertEqual(len(result_empty), 0)

        # Test case 3: PIDs that don't exist in DataFrame should still be present with NaN values
        pids_non = ["P5", "P6"]
        result_non_existing = align_df_with_pids(self.test_df, pids_non)
        self.assertEqual(len(result_non_existing), len(pids_non))
        self.assertListEqual(result_non_existing[PID_COL].tolist(), pids_non)
        # All values should be NaN for non-existing PIDs
        self.assertTrue(result_non_existing["value"].isna().all())

        # Test case 4: Mixed existing and non-existing PIDs
        pids_mixed = ["P1", "P5"]
        result_mixed = align_df_with_pids(self.test_df, pids_mixed)
        self.assertEqual(len(result_mixed), len(pids_mixed))
        self.assertListEqual(result_mixed[PID_COL].tolist(), pids_mixed)
        # First existing PID retains value, second non-existing is NaN
        self.assertEqual(result_mixed.iloc[0]["value"], 10)
        self.assertTrue(pd.isna(result_mixed.iloc[1]["value"]))


if __name__ == "__main__":
    unittest.main()
