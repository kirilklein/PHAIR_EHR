#!/usr/bin/env python
"""
Test script to validate consistency in processed data with multiple outcomes.

This script checks:
1. Whether the PIDs in folds.pt match the PIDs extracted from patients.pt
2. Whether the PIDs in index_dates match the PIDs in patients.pt
3. Whether all PIDs in folds are present in the patients list
4. Whether patients have consistent outcome structures
5. Whether all expected outcomes are present

Usage:
    python prepare_data_ft_exp_y.py path/to/processed_data_dir [--expected_outcomes OUTCOME1 OUTCOME2 ...]
"""

import argparse
import os
import sys
from os.path import exists, join
from typing import Set, List, Dict

import pandas as pd
import torch

from corebehrt.constants.data import PID_COL, TRAIN_KEY, VAL_KEY
from corebehrt.constants.paths import FOLDS_FILE, INDEX_DATES_FILE
from corebehrt.modules.preparation.causal.dataset import CausalPatientDataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def get_patient_pids(processed_data_dir: str) -> Set[int]:
    """Extract PIDs from patients.pt file."""
    try:
        dataset = CausalPatientDataset.load(processed_data_dir)
        return set(dataset.get_pids())
    except Exception as e:
        print(f"Error loading patient dataset: {e}")
        return set()


def get_causal_dataset_info(processed_data_dir: str) -> Dict:
    """Extract detailed information from the causal patient dataset."""
    try:
        dataset = CausalPatientDataset.load(processed_data_dir)

        # Get basic info
        pids = set(dataset.get_pids())

        # Get outcome information
        if len(dataset.patients) > 0:
            outcome_names = dataset.get_outcome_names()
            outcomes_dict = dataset.get_outcomes()
            exposures = dataset.get_exposures()

            # Check for missing outcomes/exposures
            missing_outcomes = []
            missing_exposures = []

            for patient in dataset.patients:
                if patient.outcomes is None:
                    missing_outcomes.append(patient.pid)
                if patient.exposure is None:
                    missing_exposures.append(patient.pid)

            return {
                "pids": pids,
                "outcome_names": outcome_names,
                "outcomes_dict": outcomes_dict,
                "exposures": exposures,
                "missing_outcomes": missing_outcomes,
                "missing_exposures": missing_exposures,
                "total_patients": len(dataset.patients),
            }
        else:
            return {"pids": pids, "total_patients": 0}

    except Exception as e:
        print(f"Error loading causal dataset info: {e}")
        return {"pids": set(), "total_patients": 0}


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


def check_outcome_consistency(
    dataset_info: Dict, expected_outcomes: List[str] = None
) -> bool:
    """Check consistency of outcome data structure."""
    all_consistent = True

    if dataset_info["total_patients"] == 0:
        print("Error: No patients found in dataset")
        return False

    # Check for missing outcomes/exposures
    if dataset_info.get("missing_outcomes"):
        print(
            f"Error: {len(dataset_info['missing_outcomes'])} patients missing outcome data"
        )
        print(f"  First few PIDs: {dataset_info['missing_outcomes'][:5]}")
        all_consistent = False

    if dataset_info.get("missing_exposures"):
        print(
            f"Error: {len(dataset_info['missing_exposures'])} patients missing exposure data"
        )
        print(f"  First few PIDs: {dataset_info['missing_exposures'][:5]}")
        all_consistent = False

    # Check outcome names
    outcome_names = dataset_info.get("outcome_names", [])
    if not outcome_names:
        print("Error: No outcome names found")
        all_consistent = False
    else:
        print(f"Found outcome names: {outcome_names}")

        # Check expected outcomes if provided
        if expected_outcomes:
            missing_expected = set(expected_outcomes) - set(outcome_names)
            extra_outcomes = set(outcome_names) - set(expected_outcomes)

            if missing_expected:
                print(f"Error: Missing expected outcomes: {missing_expected}")
                all_consistent = False

            if extra_outcomes:
                print(f"Warning: Found unexpected outcomes: {extra_outcomes}")

    # Check outcome data consistency
    outcomes_dict = dataset_info.get("outcomes_dict", {})
    if outcomes_dict:
        total_patients = dataset_info["total_patients"]
        for outcome_name, outcome_values in outcomes_dict.items():
            if len(outcome_values) != total_patients:
                print(
                    f"Error: Outcome '{outcome_name}' has {len(outcome_values)} values but {total_patients} patients"
                )
                all_consistent = False

            # Check for valid binary values (0, 1, or NaN)
            unique_values = set(outcome_values)
            valid_values = {0, 1, 0.0, 1.0}
            invalid_values = unique_values - valid_values
            if invalid_values:
                # Filter out NaN values (which are valid for missing data)
                invalid_non_nan = [v for v in invalid_values if not pd.isna(v)]
                if invalid_non_nan:
                    print(
                        f"Warning: Outcome '{outcome_name}' has non-binary values: {invalid_non_nan}"
                    )

    # Check exposure data
    exposures = dataset_info.get("exposures", [])
    if exposures:
        total_patients = dataset_info["total_patients"]
        if len(exposures) != total_patients:
            print(
                f"Error: Found {len(exposures)} exposure values but {total_patients} patients"
            )
            all_consistent = False

        # Check for valid binary values
        unique_exposures = set(exposures)
        valid_values = {0, 1, 0.0, 1.0}
        invalid_exposures = unique_exposures - valid_values
        if invalid_exposures:
            invalid_non_nan = [v for v in invalid_exposures if not pd.isna(v)]
            if invalid_non_nan:
                print(f"Warning: Exposure has non-binary values: {invalid_non_nan}")

    return all_consistent


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

    return all_consistent


def main(processed_data_dir: str, expected_outcomes: List[str] = None) -> bool:
    """Main validation function."""
    print(f"Checking processed data in: {processed_data_dir}")
    print("=" * 60)

    # Get detailed dataset information
    dataset_info = get_causal_dataset_info(processed_data_dir)
    if not dataset_info["pids"]:
        print("Error: Failed to load causal patient dataset")
        return False

    patient_pids = dataset_info["pids"]
    print(f"Found {len(patient_pids)} PIDs in patients")
    print(f"Total patients in dataset: {dataset_info['total_patients']}")

    # Check outcome consistency
    print("\n" + "=" * 60)
    print("CHECKING OUTCOME CONSISTENCY")
    print("=" * 60)
    outcome_consistent = check_outcome_consistency(dataset_info, expected_outcomes)

    # Get PIDs from other sources
    fold_pids = get_fold_pids(processed_data_dir)
    if not fold_pids:
        print("Error: Failed to load fold PIDs")
        return False
    print(f"\nFound {len(fold_pids)} PIDs in folds")

    index_dates_pids = get_index_dates_pids(processed_data_dir)
    if not index_dates_pids:
        print("Error: Failed to load index dates PIDs")
        return False
    print(f"Found {len(index_dates_pids)} PIDs in index dates")

    # Check PID consistency
    print("\n" + "=" * 60)
    print("CHECKING PID CONSISTENCY")
    print("=" * 60)
    pids_consistent = check_pids_consistency(patient_pids, fold_pids, index_dates_pids)

    # Final result
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    overall_success = outcome_consistent and pids_consistent

    if outcome_consistent:
        print("✅ Outcome consistency checks passed!")
    else:
        print("❌ Outcome consistency checks failed!")

    if pids_consistent:
        print("✅ PID consistency checks passed!")
    else:
        print("❌ PID consistency checks failed!")

    if overall_success:
        print("\n🎉 All checks passed! Multi-outcome causal dataset is consistent.")
        return True
    else:
        print("\n💥 Some checks failed. See errors above.")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate consistency in processed multi-outcome causal data"
    )
    parser.add_argument("data_dir", help="Path to processed data directory")
    parser.add_argument(
        "--expected_outcomes",
        nargs="+",
        help="Expected outcome names to validate against",
        default=None,
    )
    args = parser.parse_args()

    success = main(args.data_dir, args.expected_outcomes)
    sys.exit(0 if success else 1)
