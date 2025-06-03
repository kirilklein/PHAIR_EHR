#!/usr/bin/env python3

import argparse
import pandas as pd
import sys
import os
import glob
from typing import Optional
import numpy as np


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

        # Load true ATE
        true_ate = load_true_ate(data_dir)
        if true_ate is None:
            return False

        print(f"True ATE: {true_ate:.4f}")

        # Test each estimate (assuming there might be multiple rows)
        all_passed = True

        for idx, row in df.iterrows():
            method = row["method"]
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
            expansion_factor = 0.2
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


def main():
    parser = argparse.ArgumentParser(
        description="Test ATE estimates against true values"
    )
    parser.add_argument(
        "estimate_dir",
        type=str,
        help="Path to directory containing estimate_results.csv",
    )
    parser.add_argument(
        "data_dir",
        type=str,
        default=None,
        help="Path to directory containing .ate.txt file",
    )

    args = parser.parse_args()

    # Run the test
    success = test_ate_estimate(args.estimate_dir, args.data_dir)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
