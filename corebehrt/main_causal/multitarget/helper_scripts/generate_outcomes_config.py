"""
Helper script to generate an outcomes config YAML file from a tree dictionary.
Takes a tree dictionary created by build_tree.py and generates a corresponding outcomes.yaml
with an outcome for each key in the dictionary.
Example command:

python corebehrt/main_causal/multitarget/helper_scripts/generate_outcomes_config.py \
    --input corebehrt/main_causal/helper_data/sks_trees/medication_tree_4.pkl \
    --output corebehrt/configs/causal/multitarget/outcomes/atc_4.yaml \
    --match_how startswith \
    --prepends M/ RM/
    --remove_prefix M

"""

import argparse
import os
import pickle
import yaml
from typing import Dict, Any, List


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate outcomes config from tree dictionary"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the tree dictionary pickle file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/causal/outcomes/generated_outcomes.yaml",
        help="Path to save the generated outcomes config",
    )
    parser.add_argument(
        "--prepends",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of strings to prepend to outcome names. Default: empty list",
    )
    parser.add_argument(
        "--remove_prefix",
        type=str,
        default=None,
        help="Optional string to remove from initial outcome names. Default: empty string",
    )
    parser.add_argument(
        "--match_how",
        type=str,
        default="startswith",
        choices=["startswith", "contains", "exact"],
        help="Match method to use for all outcomes. Default: startswith",
    )
    parser.add_argument(
        "--case_sensitive",
        action="store_true",
        default=False,
        help="Whether matching should be case sensitive. Default: False",
    )
    return parser.parse_args()


def load_tree_dict(file_path: str) -> Dict[str, Any]:
    """Load the tree dictionary from a pickle file."""
    with open(file_path, "rb") as f:
        return pickle.load(f)


def generate_outcomes_config(
    tree_dict: Dict[str, Any],
    prepends: List[str] = [],
    remove_prefix: str = "",
    match_how: str = "startswith",
    case_sensitive: bool = False,
) -> Dict[str, Any]:
    """
    Generate an outcomes config dictionary from the tree dictionary.

    Args:
        tree_dict: Dictionary from tree_builder with keys as outcome codes
        prepend: Optional string to prepend to outcome names
        match_how: Match method to use (startswith, contains, exact)
        case_sensitive: Whether matching should be case sensitive

    Returns:
        Dictionary with outcomes configuration
    """
    outcomes_config = {
        "logging": {"level": "INFO", "path": "./outputs/logs"},
        "paths": {
            "data": "./example_data/example_MEDS_data_w_labs",
            "outcomes": "./outputs/causal/outcomes",
            "features": "./outputs/features/",
        },
        "outcomes": {},
    }

    for code, _ in tree_dict.items():
        if remove_prefix:  # relevant for meds (on azure which do not have the M prefix)
            code = code.replace(remove_prefix, "", 1)
        outcome_name = code

        if prepends:
            codes = []
            for prepend in prepends:
                codes.append(f"{prepend}{code}")
        else:
            codes = [code]
        outcomes_config["outcomes"][outcome_name] = {
            "type": ["code"],
            "match": [codes],
            "match_how": match_how,
            "case_sensitive": case_sensitive,
        }

    return outcomes_config


def save_outcomes_config(config: Dict[str, Any], output_path: str) -> None:
    """Save the outcomes config to a YAML file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Outcomes config saved to {output_path}")
    print(f"Generated {len(config['outcomes'])} outcomes")


def main() -> None:
    """Main function to generate outcomes config from tree dictionary."""
    args = parse_arguments()
    tree_dict = load_tree_dict(args.input)
    outcomes_config = generate_outcomes_config(
        tree_dict,
        prepends=args.prepends,
        remove_prefix=args.remove_prefix,
        match_how=args.match_how,
        case_sensitive=args.case_sensitive,
    )
    save_outcomes_config(outcomes_config, args.output)

    # Print a sample of the generated config
    sample_count = min(3, len(outcomes_config["outcomes"]))
    print(f"\nSample of {sample_count} generated outcomes:")
    sample_items = list(outcomes_config["outcomes"].items())[:sample_count]
    sample_dict = {"outcomes": {k: v for k, v in sample_items}}
    print(yaml.dump(sample_dict, default_flow_style=False))


if __name__ == "__main__":
    main()
