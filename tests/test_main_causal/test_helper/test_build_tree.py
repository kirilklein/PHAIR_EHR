import sys
import unittest
from io import StringIO

from corebehrt.main_causal.helper.build_tree import get_csv_file, print_tree_sample
from corebehrt.modules.tree.node import Node
from corebehrt.modules.tree.tree import TreeBuilder


class TestTreeFunctions(unittest.TestCase):
    def test_get_csv_file_diagnosis(self):
        """Test that the correct diagnosis CSV path is returned."""
        diagnosis_file = get_csv_file("diagnosis")
        self.assertIn("sks_dump_diag.csv", diagnosis_file)

    def test_get_csv_file_medication(self):
        """Test that the correct medication CSV path is returned."""
        medication_file = get_csv_file("medication")
        self.assertIn("sks_dump_med.csv", medication_file)

    def test_get_csv_file_invalid(self):
        """Test that an invalid tree type raises a ValueError."""
        with self.assertRaises(ValueError):
            get_csv_file("invalid_type")

    def test_tree_to_dict_at_level(self):
        """Test tree_to_dict_at_level with a simple dummy tree."""
        # Construct a dummy tree:
        #        root
        #       /    \
        #      A      B
        #     / \      \
        #    A1 A2     B1
        root = Node("root")
        root.add_child("A")
        root.add_child("B")
        node_A = root.children[0]
        node_B = root.children[1]
        node_A.add_child("A1")
        node_A.add_child("A2")
        node_B.add_child("B1")

        # Get a dictionary at level 1 (nodes A and B).
        tree_dict = TreeBuilder.tree_to_dict_at_level(root, 1)
        expected = {"A": ["A1", "A2"], "B": ["B1"]}
        self.assertEqual(tree_dict, expected)

    def test_print_tree_sample(self):
        """Test that print_tree_sample outputs expected content."""
        tree_dict = {
            "A": ["A1", "A2"],
            "B": ["B1"],
            "C": ["C1"],
            "D": ["D1"],
            "E": ["E1"],
            "F": ["F1"],
        }
        level = 1
        captured_output = StringIO()
        sys_stdout = sys.stdout
        try:
            sys.stdout = captured_output
            print_tree_sample(tree_dict, level, sample=2)
        finally:
            sys.stdout = sys_stdout
        output = captured_output.getvalue()
        self.assertIn("Outcome Dictionary at level 1", output)
        self.assertIn("Category: A", output)
        self.assertIn("Category: F", output)
        self.assertIn("Total number of codes at level 1: 6", output)


if __name__ == "__main__":
    unittest.main()
