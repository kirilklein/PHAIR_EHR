#!/usr/bin/env python3

import argparse
import glob
import os
import sys
from typing import Optional, Dict, List

import pandas as pd
import torch

from corebehrt.constants.data import PID_COL


def test_ate_estimate(
    estimate_dir: str,
    data_dir: Optional[str] = None,
    outcome_names: Optional[List[str]] = None,
) -> bool:
    """
    Test whether ATE estimates contain true ATE within confidence interval for each outcome.

    Args:
        estimate_dir: Path to directory containing estimate_results.csv and .ate.txt
        data_dir: Path to directory containing data
        outcome_names: List of outcome names to test (if None, will auto-detect)

    Returns:
        bool: True if all estimates pass tests, False otherwise
    """

    # Load estimate results
    results_path = os.path.join(estimate_dir, "estimate_results.csv")
    if not os.path.exists(results_path):
        print(f"❌ Error: estimate_results.csv not found in {estimate_dir}")
        return False

    df = pd.read_csv(results_path)

    # Check required columns
    required_cols = ["effect", "CI95_lower", "CI95_upper", "outcome"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Error: Missing columns {missing_cols} in estimate_results.csv")
        print(f"Available columns: {list(df.columns)}")
        return False

    # Get unique outcomes from the results
    unique_outcomes = df["outcome"].unique()
    print(f"Found outcomes in results: {list(unique_outcomes)}")

    # Filter outcomes if specified
    if outcome_names:
        outcomes_to_test = [o for o in outcome_names if o in unique_outcomes]
        missing_outcomes = [o for o in outcome_names if o not in unique_outcomes]
        if missing_outcomes:
            print(
                f"⚠️ Warning: Requested outcomes not found in results: {missing_outcomes}"
            )
    else:
        outcomes_to_test = unique_outcomes

    if (
        isinstance(outcomes_to_test, list) and len(outcomes_to_test) == 0
    ) or isinstance(outcomes_to_test, type(None)):
        print("❌ Error: No outcomes to test")
        return False

    print(f"Testing outcomes: {list(outcomes_to_test)}")

    # Load true ITEs for all outcomes
    true_ites = load_true_ites_from_file(estimate_dir, data_dir, outcomes_to_test)
    if not true_ites:
        return False

    # Test each outcome
    overall_success = True
    outcome_results = {}

    for outcome in outcomes_to_test:
        print(f"\n{'=' * 60}")
        print(f"TESTING OUTCOME: {outcome}")
        print(f"{'=' * 60}")

        # Get estimates for this outcome
        outcome_df = df[df["outcome"] == outcome]

        if len(outcome_df) == 0:
            print(f"❌ No estimates found for outcome {outcome}")
            overall_success = False
            continue

        # Get true ATE for this outcome
        if outcome not in true_ites:
            print(f"❌ No true ITE data found for outcome {outcome}")
            overall_success = False
            continue

        true_ate = true_ites[outcome]
        print(f"True ATE for {outcome}: {true_ate:.4f}")

        # Test estimates for this outcome
        outcome_success = test_outcome_estimates(outcome_df, true_ate, outcome)
        outcome_results[outcome] = outcome_success

        if not outcome_success:
            overall_success = False

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for outcome, success in outcome_results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{outcome}: {status}")

    return overall_success


def test_outcome_estimates(
    df: pd.DataFrame, true_ate: float, outcome_name: str
) -> bool:
    """
    Test estimates for a single outcome.

    Args:
        df: DataFrame with estimates for this outcome
        true_ate: True ATE value for this outcome
        outcome_name: Name of the outcome being tested

    Returns:
        bool: True if all estimates for this outcome pass
    """
    all_passed = True

    for idx, row in df.iterrows():
        method = row["method"]
        if method in ["RD", "RR"]:
            continue

        effect = row["effect"]
        ci_lower = row["CI95_lower"]
        ci_upper = row["CI95_upper"]
        ci_width = ci_upper - ci_lower

        print(f"\nMethod: {method}")
        print(f"  Estimated effect: {effect:.4f}")
        print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}] (width: {ci_width:.4f})")

        # Use existing 95% CI and expand it for more lenient check
        ci_95_lower = ci_lower
        ci_95_upper = ci_upper

        # Expand the 95% CI by ~70% on each side to create a broader interval
        ci_width = ci_95_upper - ci_95_lower
        expansion_factor = 0.7

        ci_expanded_lower = ci_95_lower - (ci_width * expansion_factor)
        ci_expanded_upper = ci_95_upper + (ci_width * expansion_factor)

        # Check containment at different confidence levels
        within_95_ci = ci_95_lower <= true_ate <= ci_95_upper
        within_expanded_ci = ci_expanded_lower <= true_ate <= ci_expanded_upper

        if within_95_ci:
            print(
                f"  ✓ PASS: True ATE {true_ate:.4f} within 95% CI [{ci_95_lower:.4f}, {ci_95_upper:.4f}]"
            )
        elif within_expanded_ci:
            print(
                f"  ! WARNING: True ATE {true_ate:.4f} outside 95% CI [{ci_95_lower:.4f}, {ci_95_upper:.4f}] "
                f"but within expanded range [{ci_expanded_lower:.4f}, {ci_expanded_upper:.4f}]"
            )
            # Don't set all_passed = False for warnings
        else:
            print(
                f"  ❌ FAIL: True ATE {true_ate:.4f} outside expanded range "
                f"[{ci_expanded_lower:.4f}, {ci_expanded_upper:.4f}]"
            )
            all_passed = False

    return all_passed


def load_true_ites_from_file(
    estimate_dir: str, data_dir: str, outcome_names: List[str]
) -> Dict[str, float]:
    """
    Load true ITEs from .ite.csv file filtered by common support patients.

    Args:
        estimate_dir: Directory containing patients.pt file
        data_dir: Directory containing .ite.csv file
        outcome_names: List of outcome names to load

    Returns:
        Dictionary mapping outcome names to their true ATE values
    """
    try:
        # Load common support patient IDs
        patients_file = os.path.join(estimate_dir, "patients.pt")
        if not os.path.exists(patients_file):
            print(f"❌ Error: patients.pt not found in {estimate_dir}")
            return {}

        patient_ids = torch.load(patients_file)
        print(f"Loaded {len(patient_ids)} patient IDs from common support filtering")

        # Find .ite.csv file
        ite_file = os.path.join(data_dir, ".ite.csv")
        if not os.path.exists(ite_file):
            print(f"❌ Error: .ite.csv not found in {data_dir}")
            return {}

        # Load ITE data
        ite_df = pd.read_csv(ite_file)
        print(f"Loaded {len(ite_df)} individual treatment effects")
        print(f"Available columns: {list(ite_df.columns)}")

        # Check for patient ID column
        if PID_COL not in ite_df.columns:
            print(
                f"❌ Error: Could not find patient ID column '{PID_COL}' in {ite_file}"
            )
            return {}

        # Filter by common support patient IDs
        filtered_ite_df = ite_df[ite_df[PID_COL].isin(patient_ids)]
        print(
            f"Filtered to {len(filtered_ite_df)} patients after common support filtering"
        )

        if len(filtered_ite_df) == 0:
            print("❌ Error: No patients remained after filtering")
            return {}

        # Compute ATE for each outcome
        true_ates = {}
        for outcome_name in outcome_names:
            ite_col = f"ite_{outcome_name}"

            if ite_col not in filtered_ite_df.columns:
                print(f"⚠️ Warning: Column '{ite_col}' not found in ITE file")
                print(
                    f"Available ITE columns: {[col for col in filtered_ite_df.columns if col.startswith('ite_')]}"
                )
                continue

            true_ate = filtered_ite_df[ite_col].mean()
            true_ates[outcome_name] = true_ate
            print(
                f"Computed ATE for {outcome_name}: {true_ate:.4f} (from {len(filtered_ite_df)} patients)"
            )

        return true_ates

    except Exception as e:
        print(f"❌ Error loading true ITEs: {e}")
        return {}


def load_true_ate(data_dir: str) -> Optional[float]:
    """
    Load true ATE from .ate.txt file in the directory.

    Args:
        data_dir: Directory containing .ate.txt file

    Returns:
        True ATE value or None if not found/invalid
    """
    try:
        # Find .ate.txt file
        ate_files = glob.glob(os.path.join(data_dir, ".ate.txt"))

        if len(ate_files) == 0:
            print(f"❌ Error: No .ate.txt file found in {data_dir}")
            return None

        if len(ate_files) > 1:
            print(
                f"⚠️  Warning: Multiple .ate.txt files found, using first: {ate_files[0]}"
            )

        ate_file = ate_files[0]

        # Read and parse the file
        with open(ate_file, "r") as f:
            content = f.read().strip()

        # Extract ATE value from "ATE: ..." format
        if not content.startswith("ATE:"):
            print(f"❌ Error: Expected format 'ATE: ...' in {ate_file}, got: {content}")
            return None

        ate_str = content.replace("ATE:", "").strip()
        true_ate = float(ate_str)

        return true_ate

    except Exception as e:
        print(f"❌ Error loading true ATE: {e}")
        return None


def load_true_ate_from_ite(
    estimate_dir: str, data_dir: str, outcome_name: Optional[str] = None
) -> Optional[float]:
    """
    Load true ATE from .ite.csv file filtered by common support patients.

    Args:
        estimate_dir: Directory containing patients.pt file
        data_dir: Directory containing .ite.csv file

    Returns:
        True ATE value computed from filtered ITEs or None if not found/invalid
    """
    try:
        # Load common support patient IDs
        patients_file = os.path.join(estimate_dir, "patients.pt")
        if not os.path.exists(patients_file):
            print(f"❌ Error: patients.pt not found in {estimate_dir}")
            return None

        patient_ids = torch.load(patients_file)
        print(f"Loaded {len(patient_ids)} patient IDs from common support filtering")

        # Find .ite.csv file
        print(data_dir)
        print(os.listdir(data_dir))
        ite_file = os.path.join(data_dir, ".ite.csv")

        # Load ITE data
        ite_df = pd.read_csv(ite_file)
        print(f"Loaded {len(ite_df)} individual treatment effects")

        # Filter by common support patient IDs
        # Assuming the ITE file has a patient ID column (adjust column name as needed)
        if PID_COL in ite_df.columns:
            pid_col = PID_COL
        else:
            print(f"❌ Error: Could not find patient ID column in {ite_file}")
            print(f"Available columns: {list(ite_df.columns)}")
            return None

        filtered_ite_df = ite_df[ite_df[pid_col].isin(patient_ids)]
        print(
            f"Filtered to {len(filtered_ite_df)} patients after common support filtering"
        )

        if len(filtered_ite_df) == 0:
            print("❌ Error: No patients remained after filtering")
            return None

        # Compute ATE from filtered ITEs
        # Assuming the effect column is named 'effect' or 'ite' (adjust as needed)
        if "effect" in filtered_ite_df.columns:
            effect_col = "effect"
        elif f"ite_{outcome_name}" in filtered_ite_df.columns:
            effect_col = f"ite_{outcome_name}"
        elif "treatment_effect" in filtered_ite_df.columns:
            effect_col = "treatment_effect"
        else:
            print(f"❌ Error: Could not find effect column in {ite_file}")
            print(f"Available columns: {list(filtered_ite_df.columns)}")
            return None

        true_ate = filtered_ite_df[effect_col].mean()
        print(
            f"Computed ATE from {len(filtered_ite_df)} filtered patients: {true_ate:.4f}"
        )

        return true_ate

    except Exception as e:
        print(f"❌ Error loading true ATE from ITE: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Test ATE estimates against true values for multiple outcomes"
    )
    parser.add_argument(
        "estimate_dir",
        type=str,
        help="Path to directory containing estimate_results.csv and patients.pt",
    )
    parser.add_argument(
        "data_dir",
        type=str,
        help="Path to directory containing .ite.csv file",
    )
    parser.add_argument(
        "--outcome_names",
        type=str,
        nargs="+",
        help="Names of the outcomes to test (if not provided, will test all found outcomes)",
        default=None,
    )

    args = parser.parse_args()

    # Run the test
    success = test_ate_estimate(args.estimate_dir, args.data_dir, args.outcome_names)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
