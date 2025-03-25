import unittest

from corebehrt.constants.helper import RARE_STR
from corebehrt.functional.helpers.rare_code_mapping import (
    find_rare_codes,
    get_new_code,
    group_rare_codes,
    is_hierarchical,
    should_continue_aggregation,
)


class TestGroupRareCodes(unittest.TestCase):
    def test_group_rare_codes_main(self):
        """Test the main mapping from original codes to aggregated codes."""
        input_counts = {
            "A/1234": 2,  # hierarchical: should become "A/123"
            "A/123": 3,  # hierarchical: should become "A123"
            "B/456": 2,  # non-hierarchical: becomes "B/rare"
            "C/789": 10,  # unchanged (above threshold)
            "D/012": 1,  # non-hierarchical: becomes "D/rare"
            "E/345": 8,  # unchanged (above threshold)
            "F": 1,  # non-hierarchical: becomes "F/rare"
            "A/1": 3,  # hierarchical: should become "A/rare"
            "A123": 3,  # hierarchical: should become "A/rare"
        }
        hierarchical_pattern = r"^A"  # Only codes starting with A are hierarchical.
        rare_threshold = 5

        expected_mapping = {
            "A/1234": "A/123",  # "A/1234" is trimmed to "A/123"
            "A/123": "A/123",  # "A/123" becomes "A123" (separator dropped when detail is 3 chars)
            "B/456": "B/rare",  # non-hierarchical rare code
            "C/789": "C/789",  # unchanged
            "D/012": "D/rare",  # non-hierarchical rare code
            "E/345": "E/345",  # unchanged
            "F": "F/rare",  # non-hierarchical rare code
            "A/1": f"A/{RARE_STR}",  # hierarchical rare code with separator
            "A123": f"A/{RARE_STR}",  # hierarchical rare code without separator
        }
        mapping = group_rare_codes(input_counts, rare_threshold, hierarchical_pattern)
        self.assertEqual(mapping, expected_mapping)

    def test_is_hierarchical(self):
        """Test whether codes are identified as hierarchical."""
        self.assertTrue(is_hierarchical("A/1234", r"^A"))
        self.assertFalse(is_hierarchical("B/456", r"^A"))

    def test_get_new_code_hierarchical(self):
        """Test the new code for hierarchical codes."""
        # Hierarchical: detail part longer than 3 characters -> trim one char
        self.assertEqual(get_new_code("A/1234", True), "A/123")
        # Hierarchical: detail part exactly 3 characters -> drop separator
        self.assertEqual(get_new_code("A/123", True), "A/12")
        self.assertEqual(get_new_code("A/1", True), f"A/{RARE_STR}")

    def test_get_new_code_non_hierarchical(self):
        """Test the new code for non-hierarchical codes."""
        self.assertEqual(get_new_code("B/456", False), "B/rare")
        self.assertEqual(get_new_code("F", False), "F/rare")

    def test_find_rare_codes(self):
        """Test that rare codes are identified correctly."""
        counts = {"A/123": 3, "B/456": 2, "C/789": 10}
        rare = find_rare_codes(counts, 5)
        self.assertCountEqual(rare, ["A/123", "B/456"])

    def test_should_continue_aggregation(self):
        """Test aggregation stop conditions."""
        # No code below threshold -> stop
        counts = {"A/123": 6, "B/456": 7}
        self.assertFalse(should_continue_aggregation(counts, 5, 0))
        # At least one code below threshold -> continue if iteration count is less than 5
        counts = {"A/123": 4, "B/456": 7}
        self.assertTrue(should_continue_aggregation(counts, 5, 0))
        # If iteration count reaches 8 -> stop
        self.assertFalse(should_continue_aggregation(counts, 8, 8))

    def test_hierarchical_without_separator(self):
        """Test hierarchical codes without separator."""
        self.assertTrue(is_hierarchical("A123", r"^A"))
        self.assertEqual(get_new_code("A123", True), "A12")


if __name__ == "__main__":
    unittest.main()
