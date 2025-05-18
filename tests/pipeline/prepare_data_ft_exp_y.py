#!/usr/bin/env python
"""
Test script to validate consistency in processed data.

This script checks:
1. Whether the PIDs in folds.pt match the PIDs extracted from patients.pt
2. Whether the PIDs in index_dates match the PIDs in patients.pt
3. Whether all PIDs in folds are present in the patients list

Usage:
    python check_processed_data.py path/to/processed_data_dir
"""

import argparse
import os
import sys
from os.path import exists, join
from typing import Set

import pandas as pd
import torch

from corebehrt.constants.data import PID_COL, TRAIN_KEY, VAL_KEY
from corebehrt.constants.paths import FOLDS_FILE, INDEX_DATES_FILE
from corebehrt.modules.preparation.dataset import PatientDataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def get_patient_pids(processed_data_dir: str) -> Set[int]:
    """Extract PIDs from patients.pt file."""
    try:
        dataset = PatientDataset.load(processed_data_dir)
        return set(dataset.get_pids())
    except Exception as e:
        print(f"Error loading patient dataset: {e}")
        return set()


def get_index_dates_pids(processed_data_dir: str) -> Set[int]:
    """Extract PIDs from index_dates file."""
    index_dates_path = join(processed_data_dir, INDEX_DATES_FILE)
    if not exists(index_dates_path):
        print(f"Error: Index dates file not found at {index_dates_path}")
        return set()

    try:
        df = pd.read_csv(index_dates_path)
        if PID_COL not in df.columns:
            print(f"Error: {PID_COL} column not found in index dates file")
            return set()
        return set(df[PID_COL].astype(int).unique())
    except Exception as e:
        print(f"Error loading index dates: {e}")
        return set()


def get_fold_pids(processed_data_dir: str) -> Set[int]:
    """Extract all PIDs from folds.pt file."""
    folds_path = join(processed_data_dir, FOLDS_FILE)
    if not exists(folds_path):
        print(f"Error: Folds file not found at {folds_path}")
        return set()

    try:
        folds = torch.load(folds_path)
        all_pids = set()
        for fold in folds:
            all_pids.update(fold.get(TRAIN_KEY, []))
            all_pids.update(fold.get(VAL_KEY, []))
        return all_pids
    except Exception as e:
        print(f"Error loading folds: {e}")
        return set()


def check_pids_consistency(
    patient_pids: Set[int], fold_pids: Set[int], index_dates_pids: Set[int]
) -> bool:
    """Check if PIDs are consistent across different data sources."""
    all_consistent = True

    # Check if all fold PIDs are in patients
    missing_from_patients = fold_pids - patient_pids
    if missing_from_patients:
        print(
            f"Error: {len(missing_from_patients)} PIDs in folds are missing from patients"
        )
        print(f"  First few missing PIDs: {list(missing_from_patients)[:5]}")
        all_consistent = False

    # Check if all patients are in folds
    missing_from_folds = patient_pids - fold_pids
    if missing_from_folds:
        print(
            f"Warning: {len(missing_from_folds)} PIDs in patients are not in any fold"
        )
        print(f"  This may be intentional if test set is separate from folds")

    # Check if all patients have an index date
    missing_index_dates = patient_pids - index_dates_pids
    if missing_index_dates:
        print(
            f"Error: {len(missing_index_dates)} PIDs in patients do not have an index date"
        )
        print(f"  First few missing PIDs: {list(missing_index_dates)[:5]}")
        all_consistent = False
    # Check if all index_dates PIDs are in patients

    return all_consistent


def main(processed_data_dir: str) -> bool:
    """Main validation function."""
    print(f"Checking processed data in: {processed_data_dir}")

    # Get PIDs from different sources
    patient_pids = get_patient_pids(processed_data_dir)
    if not patient_pids:
        print("Error: Failed to load patient PIDs")
        return False
    print(f"Found {len(patient_pids)} PIDs in patients")

    fold_pids = get_fold_pids(processed_data_dir)
    if not fold_pids:
        print("Error: Failed to load fold PIDs")
        return False
    print(f"Found {len(fold_pids)} PIDs in folds")

    index_dates_pids = get_index_dates_pids(processed_data_dir)
    if not index_dates_pids:
        print("Error: Failed to load index dates PIDs")
        return False
    print(f"Found {len(index_dates_pids)} PIDs in index dates")

    # Check consistency
    is_consistent = check_pids_consistency(patient_pids, fold_pids, index_dates_pids)

    if is_consistent:
        print("✅ All checks passed! PIDs are consistent across data sources.")
        return True
    else:
        print("❌ Some checks failed. See errors above.")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate consistency in processed data"
    )
    parser.add_argument("data_dir", help="Path to processed data directory")
    args = parser.parse_args()

    success = main(args.data_dir)
    sys.exit(0 if success else 1)
