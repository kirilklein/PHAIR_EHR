import unittest

import pandas as pd

from corebehrt.constants.data import PID_COL
from corebehrt.functional.causal.data_utils import align_df_with_pids


class TestDataUtils(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame with unique patient IDs and a value column
        self.test_df = pd.DataFrame(
            {PID_COL: ["P1", "P2", "P3", "P4"], "value": [10, 20, 30, 40]}
        )

    def test_align_existing_pids_preserves_order(self):
        # Existing PIDs in arbitrary order should preserve that order
        pids = ["P3", "P1"]
        result = align_df_with_pids(self.test_df, pids)
        # Check length and order
        self.assertEqual(len(result), 2)
        self.assertListEqual(result[PID_COL].tolist(), pids)
        # Values correspond correctly
        self.assertEqual(result.iloc[0]["value"], 30)
        self.assertEqual(result.iloc[1]["value"], 10)

    def test_align_empty_pid_list_returns_empty(self):
        # Empty list should return an empty DataFrame
        result = align_df_with_pids(self.test_df, [])
        self.assertTrue(result.empty)

    def test_align_with_duplicates(self):
        # Duplicate PIDs should produce duplicate rows
        pids = ["P2", "P2", "P4"]
        result = align_df_with_pids(self.test_df, pids)
        self.assertEqual(len(result), 3)
        self.assertListEqual(result[PID_COL].tolist(), pids)
        # Check that duplicate rows have same values
        self.assertEqual(result.iloc[0]["value"], 20)
        self.assertEqual(result.iloc[1]["value"], 20)
        self.assertEqual(result.iloc[2]["value"], 40)

    def test_align_missing_pids_raises(self):
        # All missing PIDs should raise ValueError listing them
        missing = ["PX", "PY"]
        with self.assertRaises(ValueError) as cm:
            align_df_with_pids(self.test_df, missing)
        msg = str(cm.exception)
        self.assertIn("The following patient IDs are not in the DataFrame", msg)
        # Should list exactly the missing IDs
        for pid in missing:
            self.assertIn(pid, msg)

    def test_align_mixed_existing_and_missing_raises(self):
        # Mixed lists containing some valid and some missing should also raise
        mixed = ["P1", "P5", "P3"]
        with self.assertRaises(ValueError) as cm:
            align_df_with_pids(self.test_df, mixed)
        msg = str(cm.exception)
        # Only missing ones reported
        self.assertIn("P5", msg)
        self.assertNotIn("P1", msg)
        self.assertNotIn("P3", msg)

    def test_additional_columns_preserved(self):
        # Ensure DataFrame may contain extra columns and they are carried through
        df_extra = self.test_df.copy()
        df_extra["extra"] = [100, 200, 300, 400]
        pids = ["P4", "P2"]
        result = align_df_with_pids(df_extra, pids)
        # Check both columns are present and preserved
        self.assertIn("value", result.columns)
        self.assertIn("extra", result.columns)
        self.assertListEqual(result["extra"].tolist(), [400, 200])


if __name__ == "__main__":
    unittest.main()
