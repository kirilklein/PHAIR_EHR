"""
Builds a tree of diagnoses or medications at a specified level.
Saves the resulting tree as a pickle file.
"""

from corebehrt.main_causal.helper.build_tree import (
    get_csv_file,
    parse_arguments,
    print_tree_sample,
    save_tree_dict,
)
from corebehrt.modules.tree.tree import TreeBuilder


def main() -> None:
    """
    Main function to build the tree, convert it to a dictionary at the specified level,
    save the dictionary, and print sample output.
    """
    args = parse_arguments()
    csv_file = get_csv_file(args.type)

    # Build the tree
    tree = TreeBuilder(file=csv_file).build()
    tree_dict = TreeBuilder.tree_to_dict_at_level(tree, args.level)

    save_tree_dict(tree_dict, args.type, args.level)
    print_tree_sample(tree_dict, args.level)


if __name__ == "__main__":
    main()
