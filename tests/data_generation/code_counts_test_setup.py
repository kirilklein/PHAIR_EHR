#!/usr/bin/env python3
"""
Test Data Generator for Code Counting Pipeline

This script generates test fixtures for validating the code counting and rare code mapping pipeline.
It creates:
1. Sample parquet files with medical codes and their frequencies
2. Configuration files for the pipeline
3. Expected output files for validation

The generated data includes:
- Common codes (frequency > threshold)
- Rare codes (frequency < threshold)
- Hierarchical codes (starting with 'M/')
- Non-hierarchical codes (starting with 'L/')
- Special case: parent code with rare frequency but common total frequency when including children

Directory Structure Created:
./tmp/
├── example_MEDS_for_code_counts_test/  # Parquet data
├── counts/                             # Code frequency counts
├── mapping/                            # Rare code mapping
├── configs/                            # Pipeline configurations
└── expected/                           # Expected outputs for testing

Usage:
    python tests/data_generation/code_counts_test_setup.py
"""

import json
import os

import pandas as pd
import yaml

from corebehrt.constants.data import CONCEPT_COL

# Paths
PARQUET_DIR = "./tmp/example_MEDS_for_code_counts_test"
SPLIT_NAME = "train"
COUNTS_DIR = "./tmp/counts"
MAPPING_DIR = "./tmp/mapping"
CONFIG_DIR = "./tmp/configs"
EXPECTED_DIR = "./tmp/expected"

# Config dicts
counts_cfg = {
    "logging": {"level": "INFO", "path": "./outputs/logs"},
    "paths": {"data": PARQUET_DIR, "counts": COUNTS_DIR},
    "splits": [SPLIT_NAME],
}

rare_code_cfg = {
    "logging": {"level": "INFO", "path": "./outputs/logs"},
    "paths": {"code_counts": COUNTS_DIR, "mapping": MAPPING_DIR},
    "file": "code_counts.json",
    "threshold": 10,
    "hierarchical_pattern": "^(?:M)",
    "separator": "/",
}

INITIAL_COUNTS = {
    "M/10": 100,  # common
    "M/101": 1,  # rare → M/10
    "M/102": 1,  # rare → M/10
    "M/11": 1,  # rare parent, but children sum → common
    "M/12": 100,  # common
    "L/aa": 1,  # non‑hier rare
    "L/bb": 2,  # non‑hier rare
    "L/cc": 100,  # non‑hier common
    # Ten children under M/11, each count=9, so total=9*10+1 parent=91 → common
    **{f"M/11{d}": 9 for d in range(0, 10)},
}

EXPECTED_MAPPING = {
    # Hierarchical codes (M/*)
    "M/101": "M/10",  # rare → mapped to common parent
    "M/102": "M/10",  # rare → mapped to common parent
    "M/10": "M/10",  # common → unchanged
    "M/11": "M/11",  # common (due to children) → unchanged
    **{f"M/11{d}": "M/11" for d in range(0, 10)},
    "M/12": "M/12",  # common → unchanged
    # Non-hierarchical codes (L/*)
    "L/aa": "L/rare",  # rare → mapped to group
    "L/bb": "L/rare",  # rare → mapped to group
    "L/cc": "L/cc",  # common → unchanged
}


def generate_sample_parquet(parquet_dir: str, split_name: str) -> dict:
    """
    Generate sample parquet files containing medical codes with specific frequencies.

    Args:
        parquet_dir (str): Directory to store the parquet files
        split_name (str): Name of the data split (e.g., 'train')

    Generated Data Structure:
        - Common codes (M/10, M/12): 100 occurrences each
        - Rare codes (M/101, M/102): 1 occurrence each, map to M/10
        - Special case (M/11): Parent rare (1) but children sum to common (91)
        - Non-hierarchical codes:
            - L/aa, L/bb: rare (1-2 occurrences)
            - L/cc: common (100 occurrences)
    """
    # code -> number of occurrences

    # Flatten into records
    records = []
    for code, cnt in INITIAL_COUNTS.items():
        records.extend([code] * cnt)

    # Split into two shards
    half = len(records) // 2
    rec1, rec2 = records[:half], records[half:]

    # Write shards
    shard_dir = os.path.join(parquet_dir, split_name)
    os.makedirs(shard_dir, exist_ok=True)

    df1 = pd.DataFrame({CONCEPT_COL: rec1})
    path1 = os.path.join(shard_dir, "0.parquet")
    df1.to_parquet(path1)
    print(f"  • Wrote {len(df1)} rows to {path1}")

    df2 = pd.DataFrame({CONCEPT_COL: rec2})
    path2 = os.path.join(shard_dir, "1.parquet")
    df2.to_parquet(path2)
    print(f"  • Wrote {len(df2)} rows to {path2}")


def write_yaml_config(cfg: dict, path: str):
    """
    Write configuration dictionary to YAML file.

    Args:
        cfg (dict): Configuration dictionary
        path (str): Output path for YAML file
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    print(f"  • Wrote config YAML to {path}")


def write_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    print(f"  • Wrote JSON to {path}")


def main():
    print("Generating sample parquet …")
    generate_sample_parquet(PARQUET_DIR, SPLIT_NAME)

    # Write configs
    counts_cfg_path = os.path.join(CONFIG_DIR, "get_counts.yaml")
    map_cfg_path = os.path.join(CONFIG_DIR, "map_rare_codes.yaml")
    print("Writing config files …")
    write_yaml_config(counts_cfg, counts_cfg_path)
    write_yaml_config(rare_code_cfg, map_cfg_path)

    # Compute expected outputs
    print("Computing expected outputs …")

    # Write expected JSONs
    expected_counts_path = os.path.join(EXPECTED_DIR, "expected_counts.json")
    expected_mapping_path = os.path.join(EXPECTED_DIR, "expected_mapping.json")

    write_json(INITIAL_COUNTS, expected_counts_path)
    write_json(EXPECTED_MAPPING, expected_mapping_path)

    print("\nDone. Your test fixtures are under ./tmp/")


if __name__ == "__main__":
    main()
