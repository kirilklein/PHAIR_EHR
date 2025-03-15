import pandas as pd
from tqdm import tqdm
from typing import List, Tuple
from corebehrt.modules.tree.node import Node


class TreeBuilder:
    def __init__(self, file: str, cutoff_level: int = 6, extend_level: int = 6):
        self.file = file
        self.cutoff_level = cutoff_level
        self.extend_level = extend_level

    def build(self) -> Node:
        """
        Builds the tree from the CSV file and applies the cutoff and extension operations.
        """
        tree_codes = self.create_tree_codes()
        tree = self.create_tree(tree_codes)
        if self.cutoff_level is not None:
            tree.cutoff_at_level(self.cutoff_level)
        if self.extend_level is not None:
            tree.extend_leaves(self.extend_level)
        return tree

    def create_tree_codes(self) -> List[Tuple[int, str]]:
        """
        Reads the CSV file, computes levels and codes, and returns a list of (level, code) tuples.
        """
        print(":Create tree codes")
        codes: List[Tuple[int, str]] = []
        df: pd.DataFrame = pd.read_csv(self.file, sep=";", encoding="utf-8")
        df = self.determine_levels_and_codes(df)
        for _, row in tqdm(df.iterrows(), desc=f"Create tree codes from {self.file}"):
            level: int = row.level
            code: str = row.code
            if pd.isna(code):
                continue
            # Adjust levels for medication files
            if "medication" in self.file:
                if level in [3, 4, 5]:
                    codes.append((level - 1, code))
                    continue
                elif level == 7:
                    codes.append((level - 2, code))
                    continue
            codes.append((level, code))
        return codes

    @staticmethod
    def determine_levels_and_codes(df: pd.DataFrame) -> pd.DataFrame:
        """
        Determines the hierarchical levels and fixes the codes for chapters and topics.
        """
        print("::Determine levels and codes")
        prev_code = ""
        level = -1
        for i, (code, text) in df.iterrows():
            if pd.isna(code):  # Only for diagnosis
                if text.startswith("Kap."):
                    code = "XX"  # Chapter (level 2)
                else:
                    # Skip "subsub"-topics (double NaNs not started by chapter)
                    if pd.isna(df.iloc[i + 1].Kode):
                        df.drop(i, inplace=True)
                        continue
                    code = "XXX"  # Topic (level 3)
            level += int(len(code) - len(prev_code))
            prev_code = code
            df.loc[i, "level"] = level
            if code.startswith("XX"):
                code = text.split()[-1]
            df.loc[i, "code"] = code
        df = df.astype({"level": "int32"})
        return df

    @staticmethod
    def create_tree(codes: List[Tuple[int, str]]) -> Node:
        """
        Creates a tree structure from a list of (level, code) tuples.
        """
        root = Node("root")
        parent = root
        for i in tqdm(range(len(codes)), desc=":Create tree"):
            level, code = codes[i]
            next_level = codes[i + 1][0] if i < len(codes) - 1 else level
            level_diff = next_level - level

            if level_diff >= 1:
                for _ in range(level_diff):
                    parent.add_child(code)
                    parent = parent.children[-1]
            elif level_diff <= 0:
                parent.add_child(code)
                for _ in range(0, level_diff, -1):
                    parent = parent.parent
        return root

    @staticmethod
    def drop_empty_categories(df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes empty chapters and categories from the DataFrame.
        First removes empty categories (level 2), then empty chapters (level 1).
        """
        rows_to_drop = [
            i
            for i in range(len(df) - 1)
            if df.iloc[i].level == 2 and df.iloc[i + 1].level <= 2
        ]
        df = df.drop(rows_to_drop).reset_index(drop=True)

        rows_to_drop = [
            i
            for i in range(len(df) - 1)
            if df.iloc[i].level == 1 and df.iloc[i + 1].level <= 1
        ]
        df = df.drop(rows_to_drop).reset_index(drop=True)
        return df

    @staticmethod
    def tree_to_dict_at_level(root: Node, level: int) -> dict:
        """
        Converts the tree into a dictionary for a specified level.
        Each key is a node's name at the given level and its value is a list of the names
        of all descendant nodes.
        """
        nodes_at_level: List[Node] = root.get_level(level)

        def get_descendants(node: Node) -> List[Node]:
            """Recursively collects all descendant nodes of a given node."""
            descendants: List[Node] = []
            for child in node.children:
                descendants.append(child)
                descendants.extend(get_descendants(child))
            return descendants

        return {
            node.name: [desc.name for desc in get_descendants(node)]
            for node in nodes_at_level
        }
