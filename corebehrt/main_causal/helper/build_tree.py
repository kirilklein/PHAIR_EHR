"""
Builds a tree of diagnoses or medications at a specified level.
Saves the resulting tree as a pickle file.
"""

import argparse
import pickle
from typing import Any, Dict

from corebehrt.constants.causal import SKS_DUMP_DIR, SKS_TREES_DIR


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments containing level and type.
    """
    parser = argparse.ArgumentParser(
        description="Build diagnosis or medication tree at a specified level"
    )
    parser.add_argument(
        "--level",
        type=int,
        default=3,
        choices=[1, 2, 3, 4, 5, 6, 7, 8],
        help=(
            "Target level for grouping. "
            "1: Chapters, 2: Subchapters, 3: Detailed diagnosis, "
            "4: Detailed diagnosis groups, 5: Detailed diagnosis groups"
        ),
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["medication", "diagnosis"],
        default="diagnosis",
        help="Which tree to build (default: diagnosis)",
    )
    return parser.parse_args()


def get_csv_file(tree_type: str) -> str:
    """
    Get the CSV file path based on the type of tree.

    Args:
        tree_type (str): Type of the tree ('diagnosis' or 'medication').

    Returns:
        str: The CSV file path.

    Raises:
        ValueError: If the provided tree type is invalid.
    """
    if tree_type == "diagnosis":
        return f"{SKS_DUMP_DIR}/sks_dump_diag.csv"
    elif tree_type == "medication":
        return f"{SKS_DUMP_DIR}/sks_dump_med.csv"
    else:
        raise ValueError(f"Invalid type: {tree_type}")


def save_tree_dict(tree_dict: Dict[Any, Any], tree_type: str, level: int) -> None:
    """
    Save the tree dictionary as a pickle file.

    Args:
        tree_dict (dict): The dictionary representation of the tree.
        tree_type (str): The type of tree ('diagnosis' or 'medication').
        level (int): The target level used in building the tree.
    """
    output_path = f"{SKS_TREES_DIR}/{tree_type}_tree_{level}.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(tree_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved tree dictionary to {output_path}")


def print_tree_sample(tree_dict: Dict[Any, Any], level: int, sample: int = 5) -> None:
    """
    Print sample entries from the tree dictionary and summary information.

    Args:
        tree_dict (dict): The dictionary representation of the tree.
        level (int): The target level used in building the tree.
        sample (int): Number of entries to show from the start and end.
    """
    print("\nOutcome Dictionary at level", level)
    items = list(tree_dict.items())

    # Print first few entries
    for category, codes in items[:sample]:
        print("Category:", category)
        print("Codes:", codes)
        print("-" * 40)

    # Print last few entries if the dictionary is larger than the sample size
    if len(items) > sample:
        for category, codes in items[-sample:]:
            print("Category:", category)
            print("Codes:", codes)
            print("-" * 40)

    print(f"Total number of codes at level {level}: {len(tree_dict)}")
