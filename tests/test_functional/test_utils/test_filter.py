import unittest

import pandas as pd
from corebehrt.functional.utils.filter import (
    filter_folds_by_pids,
    safe_control_pids,
    _align_indices_for_comparison,
    _convert_to_int,
    _convert_to_string,
)


class TestFilterFoldsByPids(unittest.TestCase):
    def test_empty_folds(self):
        """Test that an empty list of folds returns an empty list."""
        self.assertEqual(filter_folds_by_pids([], ["a", "b"]), [])

    def test_no_available_pids(self):
        """Test that none of the PIDs in folds are available."""
        folds = [{"train": ["1", "2"], "val": ["3"]}]
        available = ["a", "b"]
        expected = [{"train": [], "val": []}]
        self.assertEqual(filter_folds_by_pids(folds, available), expected)

    def test_all_available(self):
        """Test that when all PIDs are available, the folds remain unchanged."""
        folds = [{"train": ["1", "2"], "val": ["3", "4"]}]
        available = ["1", "2", "3", "4"]
        self.assertEqual(filter_folds_by_pids(folds, available), folds)

    def test_some_missing(self):
        """Test that only available PIDs are retained when some are missing."""
        folds = [{"train": ["1", "2", "5"], "val": ["3", "6"]}]
        available = ["1", "3", "5", "7"]
        expected = [{"train": ["1", "5"], "val": ["3"]}]
        self.assertEqual(filter_folds_by_pids(folds, available), expected)

    def test_multiple_folds(self):
        """Test the function with multiple folds having different available PIDs."""
        folds = [
            {"train": ["1", "2"], "val": ["3"]},
            {"train": ["4", "5"], "val": ["6", "7"]},
        ]
        available = ["1", "3", "4", "6", "8"]
        expected = [
            {"train": ["1"], "val": ["3"]},
            {"train": ["4"], "val": ["6"]},
        ]
        self.assertEqual(filter_folds_by_pids(folds, available), expected)


class TestSafeControlPids(unittest.TestCase):
    # Core functionality tests
    def test_simple_difference(self):
        """Test basic set difference with integers."""
        self.assertEqual(
            safe_control_pids([1, 2, 3], [2]),
            [1, 3],
        )

    def test_all_integers(self):
        """Test when both inputs are integers."""
        self.assertEqual(
            safe_control_pids([10, 20, 30, 40], [20, 40, 50]),
            [10, 30],
        )

    # Dtype alignment tests
    def test_dtype_alignment_str_vs_int(self):
        """Test alignment between string and integer types."""
        # exposed are ints, all_pids are strings
        self.assertEqual(
            safe_control_pids(["1", "2", "3"], [2, 4]),
            ["1", "3"],
        )
        # reverse: exposed are strings, all_pids are ints
        self.assertEqual(
            safe_control_pids([1, 2, 3], ["2", "4"]),
            [1, 3],
        )

    def test_mixed_types_object_alignment(self):
        """Test mixed types that require object alignment."""
        # all_pids object-like, exposed ints; alignment should still exclude "2"
        self.assertEqual(
            safe_control_pids(["1", 2, "3"], [2]),
            ["1", "3"],
        )

    def test_complex_type_fallback(self):
        """Test fallback to object dtype for incompatible types."""
        # This should work even with complex/datetime types
        from datetime import datetime

        dt1 = datetime(2024, 1, 1)
        self.assertEqual(
            safe_control_pids([1, "a", dt1], [dt1]),
            [1, "a"],
        )

    # Order and duplicates tests
    def test_order_preserved_and_duplicates_dropped_by_default(self):
        """Test that order is preserved and duplicates are dropped by default."""
        self.assertEqual(
            safe_control_pids(["a", "b", "a", "c"], ["b"]),
            ["a", "c"],
        )

    def test_keep_duplicates_when_requested(self):
        """Test keeping duplicates when drop_duplicates=False."""
        self.assertEqual(
            safe_control_pids(["a", "b", "a", "c"], ["b"], drop_duplicates=False),
            ["a", "a", "c"],
        )

    def test_preserve_order_false(self):
        """Test non-order-preserving mode."""
        # When preserve_order=False, we don't guarantee order; assert set equality
        out = safe_control_pids(["c", "a", "b"], ["a"], preserve_order=False)
        self.assertEqual(sorted(out), ["b", "c"])

    def test_duplicates_in_both_inputs(self):
        """Test handling duplicates in both all_pids and exposed_pids."""
        self.assertEqual(
            safe_control_pids([1, 2, 2, 3], [2, 2, 4]),
            [1, 3],
        )

    # Special pandas types tests
    def test_nullable_integers_and_missing(self):
        """Test nullable integer types with NA values."""
        s = pd.Series([1, pd.NA, 2], dtype="Int64")
        out = safe_control_pids(s, [2])
        # Expect [1, <NA>] in the same order
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0], 1)
        self.assertTrue(pd.isna(out[1]))

    def test_series_input_and_unique_exposed(self):
        """Test pandas Series as input."""
        all_s = pd.Series(["x", "y", "z", "y"])
        exp_s = pd.Series(["y"])
        self.assertEqual(
            safe_control_pids(all_s, exp_s),
            ["x", "z"],
        )

    # Edge cases
    def test_empty_inputs(self):
        """Test various empty input scenarios."""
        # Empty all_pids
        self.assertEqual(safe_control_pids([], [1, 2]), [])

        # Empty exposed_pids
        self.assertEqual(safe_control_pids(["u", "v"], []), ["u", "v"])

        # Both empty
        self.assertEqual(safe_control_pids([], []), [])

    def test_all_exposed(self):
        """Test when all items are in exposed list."""
        self.assertEqual(
            safe_control_pids(["u", "v"], ["u", "v"]),
            [],
        )

    def test_no_overlap(self):
        """Test when there's no overlap between inputs."""
        self.assertEqual(
            safe_control_pids([1, 2, 3], [4, 5, 6]),
            [1, 2, 3],
        )

    def test_single_item_inputs(self):
        """Test single-item inputs."""
        # Single item, excluded
        self.assertEqual(safe_control_pids([1], [1]), [])

        # Single item, not excluded
        self.assertEqual(safe_control_pids([1], [2]), [1])

    # Performance/stress tests
    def test_large_inputs(self):
        """Test with larger inputs to ensure performance."""
        large_all = list(range(1000))
        large_exposed = list(range(100, 200))
        result = safe_control_pids(large_all, large_exposed)

        # Should exclude 100-199, so length should be 1000 - 100 = 900
        self.assertEqual(len(result), 900)
        # First 100 should be 0-99
        self.assertEqual(result[:100], list(range(100)))
        # After gap, should continue from 200
        self.assertEqual(result[100:110], list(range(200, 210)))


class TestHelperFunctions(unittest.TestCase):
    """Test the helper functions separately."""

    def test_convert_to_int(self):
        """Test integer conversion."""
        # Pure integers
        int_idx = pd.Index([1, 2, 3])
        result = _convert_to_int(int_idx)
        self.assertTrue(all(isinstance(x, int) for x in result))
        self.assertEqual(list(result), [1, 2, 3])

        # String integers
        str_int_idx = pd.Index(["1", "2", "3"])
        result = _convert_to_int(str_int_idx)
        self.assertTrue(all(isinstance(x, int) for x in result))
        self.assertEqual(list(result), [1, 2, 3])

        # Should fail on non-numeric strings
        with self.assertRaises(ValueError):
            _convert_to_int(pd.Index(["a", "b", "c"]))

    def test_convert_to_string(self):
        """Test string conversion."""
        # Integers to strings
        int_idx = pd.Index([1, 2, 3])
        result = _convert_to_string(int_idx)
        self.assertTrue(all(isinstance(x, str) for x in result))
        self.assertEqual(list(result), ["1", "2", "3"])

        # Already strings
        str_idx = pd.Index(["a", "b", "c"])
        result = _convert_to_string(str_idx)
        self.assertTrue(all(isinstance(x, str) for x in result))
        self.assertEqual(list(result), ["a", "b", "c"])

    def test_align_indices_for_comparison(self):
        """Test index alignment logic with different scenarios."""

        # Case 1: Both can convert to int (should use int)
        str_int_idx = pd.Index(["1", "2", "3"])
        int_idx = pd.Index([2, 4])
        aligned_all, aligned_exp = _align_indices_for_comparison(str_int_idx, int_idx)

        # Should convert both to integers
        self.assertTrue(all(isinstance(x, int) for x in aligned_all))
        self.assertTrue(all(isinstance(x, int) for x in aligned_exp))
        self.assertEqual(list(aligned_all), [1, 2, 3])
        self.assertEqual(list(aligned_exp), [2, 4])

        # Case 2: Can't convert to int, should use strings
        str_idx = pd.Index(["a", "b", "c"])
        int_idx = pd.Index([1, 2])
        aligned_all, aligned_exp = _align_indices_for_comparison(str_idx, int_idx)

        # Should convert both to strings
        self.assertTrue(all(isinstance(x, str) for x in aligned_all))
        self.assertTrue(all(isinstance(x, str) for x in aligned_exp))
        self.assertEqual(list(aligned_all), ["a", "b", "c"])
        self.assertEqual(list(aligned_exp), ["1", "2"])

        # Case 3: Mixed types that can't convert to int
        mixed_idx = pd.Index(["1", 2, "3"])  # Mixed string/int
        int_idx = pd.Index([2])
        aligned_all, aligned_exp = _align_indices_for_comparison(mixed_idx, int_idx)

        # Should convert both to strings (since "1" can't convert to int cleanly)
        self.assertTrue(all(isinstance(x, int) for x in aligned_all))
        self.assertTrue(all(isinstance(x, int) for x in aligned_exp))
        self.assertEqual(list(aligned_all), [1, 2, 3])
        self.assertEqual(list(aligned_exp), [2])


if __name__ == "__main__":
    unittest.main()
