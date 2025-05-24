#!/usr/bin/env python3

import argparse
import pandas as pd
import sys
from typing import Optional
import os


def test_roc_performance(
    ft_dir: str,
    min_roc: Optional[float] = None,
    max_roc: Optional[float] = None,
    metric_name: str = "roc_auc",
    file_start: str = "val_scores_mean_std",
) -> bool:
    """
    Test whether ROC AUC values in a CSV file are within specified bounds.

    Args:
        ft_dir: Path to directory containing performance metrics file
        min_roc: Minimum acceptable ROC AUC (default: None, no lower bound)
        max_roc: Maximum acceptable ROC AUC (default: None, no upper bound)
        metric_name: Name of the metric to look for in 'metric' column

    Returns:
        bool: True if all ROC values are within bounds, False otherwise
    """
    file_path = find_roc_auc_file(ft_dir, file_start)
    print(f"Found file: {file_path}")
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)

        # Check if required columns exist
        if "metric" not in df.columns:
            print(f"❌ Error: Column 'metric' not found in {file_path}")
            print(f"Available columns: {list(df.columns)}")
            return False

        if "mean" not in df.columns:
            print(f"❌ Error: Column 'mean' not found in {file_path}")
            print(f"Available columns: {list(df.columns)}")
            return False

        # Filter for the specific metric
        metric_rows = df[df["metric"] == metric_name]

        if len(metric_rows) == 0:
            print(f"❌ Error: Metric '{metric_name}' not found in metric column")
            print(f"Available metrics: {df['metric'].unique().tolist()}")
            return False

        # Get the mean values for this metric
        roc_value = float(metric_rows["mean"].iloc[0])
        # Test bounds
        passed = True

        if min_roc is not None:
            below_min = roc_value < min_roc
            if below_min:
                print(
                    f"❌FAIL: {metric_name} value {roc_value:.4f} below minimum {min_roc}"
                )
                passed = False

        if max_roc is not None:
            above_max = roc_value > max_roc
            if above_max:
                print(
                    f"❌FAIL: {metric_name} value {roc_value:.4f} above maximum {max_roc}"
                )
                passed = False

        if passed:
            print(f"✅ PASS: {metric_name} value {roc_value:.4f} within bounds")

        return passed

    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return False
    except Exception as e:
        print(f"Error loading or processing file: {e}")
        return False


def find_roc_auc_file(ft_dir: str, file_start: str) -> str:
    """Look for a file starting with val_scores_mean_std"""
    for file in os.listdir(ft_dir):
        if file.startswith(file_start):
            return os.path.join(ft_dir, file)


def main():
    parser = argparse.ArgumentParser(
        description="Test ROC AUC performance metrics from CSV file"
    )
    parser.add_argument(
        "path", type=str, help="Path to directory containing performance metrics"
    )
    parser.add_argument(
        "file_start", type=str, help="Start of file name", default="val_scores_mean_std"
    )
    parser.add_argument(
        "--min", type=float, default=0, help="Minimum acceptable value (default: 0)"
    )
    parser.add_argument(
        "--max", type=float, default=1, help="Maximum acceptable value (default: 1)"
    )
    parser.add_argument(
        "--metric-name",
        type=str,
        default="roc_auc",
        help="Name of metric to look for in 'metric' column (default: 'roc_auc')",
    )

    args = parser.parse_args()

    # Run the test
    success = test_roc_performance(
        args.path, args.min, args.max, args.metric_name, args.file_start
    )

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
