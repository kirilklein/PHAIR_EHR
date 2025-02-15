import unittest

from corebehrt.functional.utils.filter import filter_folds_by_pids


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


if __name__ == "__main__":
    unittest.main()
