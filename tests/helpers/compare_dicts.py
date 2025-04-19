#!/usr/bin/env python3
"""
compare_dicts.py

Compare two dictionary files for equality. Files can be either .pt (PyTorch) or .json format.
Exits with 0 if dictionaries match, or 1 if there are any differences (with detailed summary).

Usage:
    python -m corebehrt.main.helper.compare_dicts
        --actual path/to/first/dict.[pt|json]
        --expected path/to/second/dict.[pt|json]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

import torch


def load_dict(path: str) -> Dict[str, Any]:
    """
    Load a dictionary from either a .pt or .json file.

    Args:
        path (str): Path to the file to load

    Returns:
        dict: Loaded dictionary

    Raises:
        ValueError: If file extension not supported or loaded object is not a dict
        FileNotFoundError: If file doesn't exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix == ".pt":
        data = torch.load(path)
    elif path.suffix == ".json":
        with open(path, "r") as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}. Use .pt or .json")

    if not isinstance(data, dict):
        raise ValueError(f"Loaded object from {path!r} is not a dict")

    return data


def compare_dicts(actual: Dict, expected: Dict) -> Tuple[set, set, Dict]:
    """
    Compare two dictionaries and return their differences.

    Args:
        actual (dict): First dictionary
        expected (dict): Second dictionary

    Returns:
        Tuple[set, set, dict]: Contains:
            - missing: Keys in expected but not in actual
            - extra: Keys in actual but not in expected
            - diffs: Dictionary of differing values for common keys
    """
    actual_keys = set(actual.keys())
    expected_keys = set(expected.keys())

    missing = expected_keys - actual_keys
    extra = actual_keys - expected_keys
    common = actual_keys & expected_keys

    diffs = {k: (expected[k], actual[k]) for k in common if actual[k] != expected[k]}

    return missing, extra, diffs


def main():
    parser = argparse.ArgumentParser(
        description="Compare two dictionary files (.pt or .json) for equality"
    )
    parser.add_argument(
        "--actual", required=True, help="Path to first dictionary file (.pt or .json)"
    )
    parser.add_argument(
        "--expected",
        required=True,
        help="Path to second dictionary file (.pt or .json)",
    )
    args = parser.parse_args()

    try:
        actual = load_dict(args.actual)
        expected = load_dict(args.expected)
    except (ValueError, FileNotFoundError) as e:
        print(f"❌ Error loading files: {e}")
        sys.exit(1)

    if actual == expected:
        print("✅ Dictionaries match exactly")
        sys.exit(0)

    # Compute and display differences
    missing, extra, diffs = compare_dicts(actual, expected)

    if missing:
        print(f"❌ Keys missing in actual: {sorted(missing)}")
    if extra:
        print(f"❌ Extra keys in actual: {sorted(extra)}")
    if diffs:
        print("❌ Value mismatches:")
        for k, (exp, act) in diffs.items():
            print(f"   {k!r}: expected={exp!r}, actual={act!r}")

    # Summary
    total_diffs = len(missing) + len(extra) + len(diffs)
    print(f"\nSummary: Found {total_diffs} differences")
    print(f"- Missing keys: {len(missing)}")
    print(f"- Extra keys: {len(extra)}")
    print(f"- Value mismatches: {len(diffs)}")

    sys.exit(1)


if __name__ == "__main__":
    main()
