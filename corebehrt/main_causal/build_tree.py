"""
Builds a tree of diagnoses or medications at a specified level.
Saves the tree as a pickle file.
"""

import argparse
import pickle

from corebehrt.modules.tree.tree import TreeBuilder
from corebehrt.constants.causal import SKS_DUMP_DIR, SKS_TREES_DIR


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Build diagnosis tree at specified level"
    )
    parser.add_argument(
        "--level",
        type=int,
        default=3,
        choices=[1, 2, 3, 4, 5],
        help="Target level for grouping (1: Chapters, 2: Subchapters, 3: Detailed diagnosis, 4: Detailed diagnosis groups, 5: Detailed diagnosis groups)",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["medication", "diagnosis"],
        help="Which tree to build (default: diagnosis)",
    )

    args = parser.parse_args()
    target_level = args.level

    if args.type == "diagnosis":
        csv_file = f"{SKS_DUMP_DIR}/sks_dump_diag.csv"  # Adjust the path as needed.
    elif args.type == "medication":
        csv_file = f"{SKS_DUMP_DIR}/sks_dump_med.csv"  # Adjust the path as needed.

    else:
        raise ValueError(f"Invalid type: {args.type}")

    # Build the tree.
    tree = TreeBuilder(file=csv_file).build()

    tree_dict = TreeBuilder.tree_to_dict_at_level(tree, target_level)

    with open(f"{SKS_TREES_DIR}/{args.type}_tree_{args.level}.pkl", "wb") as f:
        pickle.dump(tree_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("\nOutcome Dictionary at level", target_level)

    for category, codes in list(tree_dict.items())[:5]:
        print("Category:", category)
        print("Codes:", codes)
        print("-" * 40)

    for category, codes in list(tree_dict.items())[-5:]:
        print("Category:", category)
        print("Codes:", codes)
        print("-" * 40)

    print(f"Total number of codes at level {target_level}: {len(tree_dict)}")


if __name__ == "__main__":
    main()
