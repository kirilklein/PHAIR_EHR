import csv
import pickle
import re

# -- Node Class for Tree Structure --


class Node:
    def __init__(self, code, text, level):
        self.code = (
            code  # e.g. "DA00", or a derived code like "DA00-DB99" from a chapter row.
        )
        self.text = text  # e.g. "Cholera" or "Chap. I: Certain infectious..."
        self.level = level  # Explicit level in our hierarchy.
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def get_all_codes(self):
        """
        Recursively collects this node’s code and the codes of all its descendants.
        (Excludes the dummy root code.)
        """
        codes = [] if self.code == "root" else [self.code]
        for child in self.children:
            codes.extend(child.get_all_codes())
        return codes

    def __repr__(self):
        return f"Node(code={self.code}, text={self.text}, level={self.level})"


# -- Helpers for Chapter and Subchapter Detection --


def is_chapter(text):
    """
    Returns True if the text starts with a chapter keyword (e.g. "Chap.", "Ch.", "Chapter")
    and ends with a bracketed expression (e.g. "[DA00-DB99]").
    """
    pattern = r"^(Chap\.|Ch\.|Chapter)\s.*\[[^]]+\]\s*$"
    return re.match(pattern, text.strip()) is not None


def is_subchapter(text):
    """
    Returns True if the text ends with a bracketed expression but is not recognized as a chapter.
    """
    pattern = r"^.*\[[^]]+\]\s*$"
    return (not is_chapter(text)) and (re.match(pattern, text.strip()) is not None)


# -- Tree Building --


def build_tree(csv_file, delimiter=";"):
    """
    Reads the CSV file (assumed to have headers "Text" and "Code") and builds a tree.

    For rows with an empty "Code":
      - If the text is a chapter (matches is_chapter), assign level 1 and derive the code from the bracket.
      - If it is a subchapter (matches is_subchapter), assign level 2 and derive the code similarly.
      - Otherwise, default to parent's level + 1.

    For rows with a nonempty "Code", assign level = parent's level + 1.
    The stack-based approach ensures proper nesting.
    """
    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        rows = [row for row in reader]

    # Create a dummy root node at level 0.
    root = Node("root", "root", level=0)
    stack = [root]  # Stack to track current branch.

    for row in rows:
        text = row["Text"].strip()
        code = row["Code"].strip()

        # Determine explicit level if code is missing.
        explicit_level = None
        if code == "":
            if is_chapter(text):
                explicit_level = 1
                # Extract the code range from the bracket.
                m = re.search(r"\[([^]]+)\]", text)
                code = m.group(1) if m else "XX"
            elif is_subchapter(text):
                explicit_level = 2
                m = re.search(r"\[([^]]+)\]", text)
                code = m.group(1) if m else "XXX"
            else:
                # Fallback: assign as child of current node.
                explicit_level = stack[-1].level + 1

        # For rows with an explicit nonempty code, if not explicitly set, assign as child of current node.
        new_level = (
            explicit_level if explicit_level is not None else (stack[-1].level + 1)
        )

        # Adjust the stack: pop until the top of the stack has a level less than new_level.
        while stack and stack[-1].level >= new_level:
            stack.pop()
        parent = stack[-1]
        new_node = Node(code, text, new_level)
        parent.add_child(new_node)
        stack.append(new_node)

    return root


# -- Extraction of Dictionary at a Target Level --


def extract_dict_at_level(root, target_level):
    """
    Traverse the tree and for every node at the specified target level,
    create an entry in a dictionary mapping the node’s text to the list
    of all descendant codes (including its own code).
    """
    result = {}

    def dfs(node):
        if node.level == target_level and node.code != "root":
            # If multiple nodes share the same text, you could combine them here.
            result[node.text] = node.get_all_codes()
        for child in node.children:
            dfs(child)

    dfs(root)
    return result


# -- Debugging Helper: Print the Tree --


def print_tree(node, indent=0):
    print("  " * indent + f"{node.level}: {node.text} ({node.code})")
    for child in node.children:
        print_tree(child, indent + 1)


# -- Main Routine --


def main():
    csv_file = (
        "corebehrt/main_causal/helper/data/diag.csv"  # Adjust the path as needed.
    )

    # Build the tree.
    tree = build_tree(csv_file)

    # Extract the dictionary for the desired grouping level.
    # For example, with this scheme:
    #   Level 1: Chapters (e.g. "Chap. I: ...")
    #   Level 2: Subchapters (e.g. "Infectious intestinal diseases [DA00-DA09]")
    #   Level 3: Detailed diagnosis groups (e.g. "Cholera", "Typhoid and paratyphoid", "Other salmonella infections")
    target_level = 3
    outcome_dict = extract_dict_at_level(tree, target_level)

    print("\nOutcome Dictionary at level", target_level)
    for disease, codes in list(outcome_dict.items())[:5]:
        print("Disease:", disease)
        print("Codes:", codes)
        print("-" * 40)

    with open("corebehrt/main_causal/helper/data/diag_level_3.pkl", "wb") as f:
        pickle.dump(outcome_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
