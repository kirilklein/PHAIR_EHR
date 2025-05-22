#!/usr/bin/env python
"""
Compare cohorts between two directories and analyze patient overlap.

Usage:
    python compare_cohorts.py path/to/larger/cohort path/to/smaller/cohort

This script:
1. Loads PIDs from both cohort directories
2. Checks if the first cohort is at least as large as the second
"""

import argparse
import os
from typing import List, Set

import torch


def load_pids(directory: str) -> List[int]:
    """Load patient IDs from a directory."""
    pid_path = os.path.join(directory, "pids.pt")
    if not os.path.exists(pid_path):
        raise FileNotFoundError(f"PID file not found at {pid_path}")
    pids = torch.load(pid_path)
    if not isinstance(pids, list) or not all(isinstance(pid, int) for pid in pids):
        raise TypeError("PIDs file should contain a list of integers")
    return pids


def check_at_least_as_large_cohort(
    pids_larger: Set[int], pids_smaller: Set[int]
) -> bool:
    """Check if the first cohort is at least as large as the second cohort."""
    return len(pids_larger) >= len(pids_smaller)


def main():
    parser = argparse.ArgumentParser(description="Compare two cohorts")
    parser.add_argument("larger_dir", help="Directory with larger cohort")
    parser.add_argument("smaller_dir", help="Directory with smaller cohort")
    args = parser.parse_args()

    pids_larger = set(load_pids(args.larger_dir))
    pids_smaller = set(load_pids(args.smaller_dir))

    if check_at_least_as_large_cohort(pids_larger, pids_smaller):
        print("✅ First cohort is at least as large as the second cohort")
    else:
        raise ValueError(
            "❌ First cohort is not at least as large as the second cohort"
        )


if __name__ == "__main__":
    main()
