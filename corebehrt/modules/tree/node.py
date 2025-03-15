import torch
from typing import List


class Node:
    def __init__(self, name: str, parent: "Node" = None):
        self.name: str = name
        self.parent: "Node" = parent
        self.children: List["Node"] = []

    def __repr__(self) -> str:
        return self.name

    def add_child(self, code: str) -> None:
        """
        Adds a child node with the given code.
        """
        child = Node(name=code, parent=self)
        self.children.append(child)

    def base_counts(self, counts: dict) -> None:
        """
        Sets the base count for the node and recursively for its children.
        """
        self.base_count: int = counts.get(self.name, 0) + 1
        for child in self.children:
            child.base_counts(counts)

    def sum_counts(self) -> int:
        """
        Recursively sums the counts for this node and its descendants.
        """
        self.sum_count: int = self.base_count + sum(
            child.sum_counts() for child in self.children
        )
        return self.sum_count

    def redist_counts(self) -> None:
        """
        Redistributes counts from this node to its children.
        """
        self.redist_count = getattr(self, "redist_count", self.sum_count)
        for child in self.children:
            child.redist_count = self.redist_count * (
                child.sum_count / (self.sum_count - self.base_count)
            )
            child.redist_counts()

    def extend_leaves(self, level: int) -> None:
        """
        Extends leaves to ensure that all branches reach a specified level.
        """
        if not self.children and level > 0:
            self.add_child(self.name)
        for child in self.children:
            child.extend_leaves(level - 1)

    def cutoff_at_level(self, cutoff_level: int, method: str = "flatten"):
        """
        Cuts off the tree at a specified level.
        For the 'flatten' method, truncated nodes are integrated at the parent's level.
        """
        if method == "flatten":
            if not self.children:
                return self

            if cutoff_level <= 0:
                new_children = [self] + [
                    child.cutoff_at_level(cutoff_level - 1) for child in self.children
                ]
                self.children = []
                return new_children
            else:
                self.children = [
                    child.cutoff_at_level(cutoff_level - 1) for child in self.children
                ]
                self.children = self._flatten(self.children)
                for child in self.children:
                    child.parent = self
                return self
        else:
            raise NotImplementedError("Collapse method not implemented")

    def get_tree_matrix(self) -> torch.Tensor:
        """
        Constructs a tensor representing the tree structure.
        """
        n_levels = self.get_max_level()
        n_leaves = len(self.get_level(n_levels))
        tree_matrix = torch.zeros((n_levels, n_leaves, n_leaves))
        for level in range(n_levels):
            nodes: List["Node"] = self.get_level(level + 1)
            acc = 0
            for i, node in enumerate(nodes):
                n_children = node.num_children_leaves()
                tree_matrix[level, i, acc : acc + n_children] = 1
                acc += n_children
        return tree_matrix

    def create_target_mapping(self, value: int = -100) -> dict:
        """
        Creates a mapping for each node that includes its parent's mapping,
        its index, and padding.
        """
        mapping: dict = {"root": []}
        max_level = self.get_max_level()
        for level in range(1, max_level + 1):
            nodes: List["Node"] = self.get_level(level)
            for i, node in enumerate(nodes):
                mapping[node.name] = (
                    mapping[node.parent.name][: level - 1]
                    + [i]
                    + [value] * (max_level - level)
                )
        del mapping["root"]
        return mapping

    def print_children(self, *attrs: str, spaces: int = 0) -> None:
        """
        Prints the node along with specified attributes, and recursively prints its children.
        """
        print(" " * spaces, self, [getattr(self, attr, "") for attr in attrs])
        for child in self.children:
            child.print_children(*attrs, spaces=spaces + 2)

    def get_level(self, level: int) -> List["Node"]:
        """
        Returns all nodes at a specified level.
        Level 0 is the current node.
        """
        if self.parent is None and level > self.get_max_level():
            raise IndexError(
                f"Level {level} is too high. Max level is {self.get_max_level()}"
            )
        if level == 0:
            return [self]
        return self._flatten([child.get_level(level - 1) for child in self.children])

    def get_max_level(self) -> int:
        """
        Returns the maximum depth (level) of the tree.
        """
        if not self.children:
            return 0
        return 1 + max(child.get_max_level() for child in self.children)

    def num_children_leaves(self) -> int:
        """
        Returns the number of leaf nodes under this node.
        """
        if not self.children:
            return 1
        return sum(child.num_children_leaves() for child in self.children)

    def get_leaf_counts(self) -> torch.Tensor:
        """
        Returns a tensor of the redistributed counts for all leaves.
        """
        return torch.tensor(
            [c.redist_count for c in self.get_level(self.get_max_level())]
        )

    def get_all_nodes(self) -> List["Node"]:
        """
        Returns a flat list of all nodes (excluding the root).
        """
        if not self.parent:  # Exclude the root node and category nodes
            return self._flatten([child.get_all_nodes() for child in self.children])
        return [self] + self._flatten(
            [child.get_all_nodes() for child in self.children]
        )

    @staticmethod
    def _flatten(data: List) -> List:
        """
        Recursively flattens a nested list.
        """

        def _flatten(data):
            for element in data:
                if isinstance(element, list):
                    yield from _flatten(element)
                else:
                    yield element

        return list(_flatten(data))
