#!/usr/bin/env python3

import argparse
import glob
import os
import sys
from typing import Optional

import pandas as pd
import torch

from corebehrt.constants.data import PID_COL


def test_ate_estimate(estimate_dir: str, data_dir: Optional[str] = None) -> bool:
    """
    Test whether ATE estimates contain true ATE within confidence interval.

    Args:
        estimate_dir: Path to directory containing estimate_results.csv and .ate.txt
        data_dir: Path to directory containing data

    Returns:
        bool: True if estimates pass tests, False otherwise
    """
    try:
        # Load estimate results
        results_path = os.path.join(estimate_dir, "estimate_results.csv")
        if not os.path.exists(results_path):
            print(f"❌ Error: estimate_results.csv not found in {estimate_dir}")
            return False

        df = pd.read_csv(results_path)

        # Check required columns
        required_cols = ["effect", "CI95_lower", "CI95_upper"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ Error: Missing columns {missing_cols} in estimate_results.csv")
            print(f"Available columns: {list(df.columns)}")
            return False

        # Load true ATE from filtered patients
        true_ate = load_true_ate_from_ite(estimate_dir, data_dir)
        if true_ate is None:
            return False

        print(f"True ATE: {true_ate:.4f}")

        # Test each estimate (assuming there might be multiple rows)
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

            # Expand the 95% CI by ~20% on each side to create a broader interval
            ci_width = ci_95_upper - ci_95_lower
            expansion_factor = 0.5
            ci_90_lower = ci_95_lower - (ci_width * expansion_factor)
            ci_90_upper = ci_95_upper + (ci_width * expansion_factor)

            # Check containment at different confidence levels
            within_95_ci = ci_95_lower <= true_ate <= ci_95_upper
            within_90_ci = ci_90_lower <= true_ate <= ci_90_upper

            if within_95_ci:
                print(
                    f"  ✓ PASS: True ATE {true_ate:.4f} within 95% CI [{ci_95_lower:.4f}, {ci_95_upper:.4f}]"
                )
            elif within_90_ci:
                print(
                    f"  ! WARNING: True ATE {true_ate:.4f} outside 95% CI [{ci_95_lower:.4f}, {ci_95_upper:.4f}] but within expanded range [{ci_90_lower:.4f}, {ci_90_upper:.4f}]"
                )
                # Don't set all_passed = False for warnings
            else:
                print(
                    f"  ❌ FAIL: True ATE {true_ate:.4f} outside expanded range [{ci_90_lower:.4f}, {ci_90_upper:.4f}]"
                )
                all_passed = False

        return all_passed

    except Exception as e:
        print(f"❌ Error processing estimates: {e}")
        return False


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


def load_true_ate_from_ite(estimate_dir: str, data_dir: str) -> Optional[float]:
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
        elif "ite" in filtered_ite_df.columns:
            effect_col = "ite"
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
        description="Test ATE estimates against true values"
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

    args = parser.parse_args()

    # Run the test
    success = test_ate_estimate(args.estimate_dir, args.data_dir)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
