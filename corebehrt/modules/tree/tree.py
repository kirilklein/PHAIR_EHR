import pandas as pd
from tqdm import tqdm
from typing import List
from corebehrt.modules.tree.node import Node


class TreeBuilder:
    def __init__(
        self,
        file,
        cutoff_level=6,
        extend_level=6,
    ):
        self.file = file
        self.cutoff_level = cutoff_level
        self.extend_level = extend_level

    def build(self):
        tree_codes = self.create_tree_codes()
        tree = self.create_tree(tree_codes)
        if self.cutoff_level is not None:
            tree.cutoff_at_level(self.cutoff_level)
        if self.extend_level is not None:
            tree.extend_leaves(self.extend_level)

        return tree

    def create_tree_codes(self):
        print(":Create tree codes")
        codes: list[tuple[int, str]] = []
        database: pd.DataFrame = pd.read_csv(self.file, sep=";", encoding="utf-8")
        database = self.determine_levels_and_codes(database)
        for _, row in tqdm(
            database.iterrows(), desc=f"Create tree codes from {self.file}"
        ):
            level: int = row.level
            code: str = row.code
            if pd.isna(row.code):
                continue
            # Needed to fix the levels for medication
            if "medication" in self.file and level in [3, 4, 5]:
                codes.append((level - 1, code))
            elif "medication" in self.file and level == 7:
                codes.append((level - 2, code))
            else:
                codes.append((level, code))
        return codes

    @staticmethod
    def determine_levels_and_codes(database: pd.DataFrame) -> pd.DataFrame:
        """Takes a DataFrame and returns a DataFrame with levels for each code. Also assigns proper codes for chapters and topics."""
        print("::Determine levels and codes")
        prev_code = ""
        level = -1
        for i, (code, text) in database.iterrows():
            if pd.isna(code):  # Only for diagnosis
                # Manually set nan codes for Chapter and Topic (as they have ranges)
                if text.startswith("Kap."):
                    code = "XX"  # Sets Chapter as level 2 (XX)
                else:
                    if pd.isna(
                        database.iloc[i + 1].Kode
                    ):  # Skip "subsub"-topics (double nans not started by chapter)
                        database.drop(i, inplace=True)
                        continue
                    code = "XXX"  # Sets Topic as level 3 (XXX)
            level += int(
                len(code) - len(prev_code)
            )  # Add distance between current and previous code to level
            prev_code = code  # Set current code as previous code
            database.loc[i, "level"] = level
            if code.startswith("XX"):  # Gets proper code (chapter/topic range)
                code = text.split()[-1]
            database.loc[i, "code"] = code
        database = database.astype({"level": "int32"})
        return database

    @staticmethod
    def create_tree(codes):
        root = Node("root")
        parent = root
        for i in tqdm(range(len(codes)), desc=":Create tree"):
            level, code = codes[i]
            next_level = codes[i + 1][0] if i < len(codes) - 1 else level
            dist = next_level - level

            if dist >= 1:
                for _ in range(dist):
                    parent.add_child(code)
                    parent = parent.children[-1]
            elif dist <= 0:
                parent.add_child(code)
                for _ in range(0, dist, -1):
                    parent = parent.parent
        return root

    @staticmethod
    def drop_empty_categories(database: pd.DataFrame) -> pd.DataFrame:
        """Takes a DataFrame and returns a DataFrame with empty chapters removed."""
        rows_to_drop = []

        # First pass: remove empty categories (level 2)
        for i in range(len(database) - 1):
            if database.iloc[i].level == 2 and database.iloc[i + 1].level <= 2:
                rows_to_drop.append(i)
        database = database.drop(rows_to_drop).reset_index(drop=True)

        # Second pass: remove empty chapters (level 1)
        rows_to_drop = []
        for i in range(len(database) - 1):
            if database.iloc[i].level == 1 and database.iloc[i + 1].level <= 1:
                rows_to_drop.append(i)
        database = database.drop(rows_to_drop).reset_index(drop=True)
        return database

    @staticmethod
    def tree_to_dict_at_level(root: Node, level: int) -> dict:
        """
        Converts the tree into a dictionary based on a specified level.
        Each key is the node name at the given level, and its value is a list of names
        for all descendant nodes (at any lower level).

        Args:
            root (Node): The root of the tree.
            level (int): The level at which to gather the keys (0-based level after root).

        Returns:
            dict: A dictionary mapping node names at the given level to a list of descendant node names.
        """
        # Get all nodes at the specified level.
        nodes_at_level: List[Node] = root.get_level(level)

        def get_descendants(node: Node) -> list:
            """Recursively collects all descendant nodes (excluding the node itself)."""
            descendants: List[Node] = []
            for child in node.children:
                descendants.append(child)
                descendants.extend(get_descendants(child))
            return descendants

        result: dict = {}
        for node in nodes_at_level:
            # Gather the names of all descendants of the current node.
            descendants: List[Node] = get_descendants(node)
            result[node.name] = [desc.name for desc in descendants]

        return result
