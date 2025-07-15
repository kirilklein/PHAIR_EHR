#!/usr/bin/env python3

"""
Test multi-target performance metrics from combined CSV file.

USAGE EXAMPLES:

1. Basic usage (auto-detect all targets, no bounds):
   python test_performance_multitarget.py outputs/causal/multitarget/models/simple

2. Test with bounds for specific targets:
   python test_performance_multitarget.py outputs/causal/multitarget/models/simple \
     --target-bounds "exposure:min:0.55,max:0.65" \
     --target-bounds "OUTCOME_2:min:0.65,max:0.85"

3. Test specific targets only:
   python test_performance_multitarget.py outputs/causal/multitarget/models/simple \
     --targets exposure OUTCOME_2

4. Test different metric (e.g., pr_auc instead of roc_auc):
   python test_performance_multitarget.py outputs/causal/multitarget/models/simple \
     --metric-name pr_auc \
     --target-bounds "exposure:min:0.35,max:0.45"

5. Comprehensive example with multiple bounds:
   python test_performance_multitarget.py outputs/causal/multitarget/models/simple \
     --metric-name roc_auc \
     --target-bounds "exposure:min:0.50" \
     --target-bounds "OUTCOME:min:0.35,max:0.60" \
     --target-bounds "OUTCOME_2:min:0.55" \
     --target-bounds "OUTCOME_3:min:0.40,max:0.70"

Expected scores file format:
  scores/scores_YYYYMMDD-HHMM.csv with columns: metric,outcome,mean,std
  
Example content:
  metric,outcome,mean,std
  pr_auc,OUTCOME,0.2203,0.0938
  pr_auc,exposure,0.4053,0.0138
  roc_auc,OUTCOME,0.3990,0.2418
  roc_auc,exposure,0.5491,0.0223
"""

import argparse
import pandas as pd
import sys
from typing import Dict, List
import os
import re
from datetime import datetime


def test_multitarget_performance(
    ft_dir: str,
    target_bounds: Dict[str, Dict[str, float]] = None,
    metric_name: str = "roc_auc",
    target_names: List[str] = None,
) -> bool:
    """
    Test whether performance metrics for all targets are within specified bounds.

    Args:
        ft_dir: Path to directory containing scores subdirectory
        target_bounds: Dict mapping target names to their min/max bounds
        metric_name: Name of the metric to look for (e.g., 'roc_auc')
        target_names: List of target names to test (if None, auto-detect)

    Returns:
        bool: True if all metrics are within bounds, False otherwise
    """
    scores_dir = os.path.join(ft_dir, "scores")
    if not os.path.exists(scores_dir):
        print(f"❌ Error: Scores directory not found: {scores_dir}")
        return False

    print(f"Looking for score files in: {scores_dir}")

    # Load the combined scores file
    try:
        scores_df = load_combined_scores_file(scores_dir)
    except Exception as e:
        print(f"❌ Error loading scores file: {e}")
        return False

    # Auto-detect target names if not provided
    if target_names is None:
        target_names = auto_detect_target_names_from_df(scores_df)

    if not target_names:
        print(f"❌ Error: No targets found in scores file")
        return False

    # Test all targets
    print(f"\n{'=' * 60}")
    print("TESTING TARGET PERFORMANCE")
    print(f"{'=' * 60}")
    print(f"Testing targets: {target_names}")

    all_passed = True

    for target_name in target_names:
        print(f"\n--- Testing {target_name} ---")
        current_bounds = target_bounds.get(target_name) if target_bounds else None

        target_passed = test_target_from_combined_df(
            scores_df, target_name, current_bounds, metric_name
        )
        all_passed = all_passed and target_passed

    # Overall summary
    print(f"\n{'=' * 60}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 60}")

    if all_passed:
        print("🎉 ALL TESTS PASSED! All targets meet performance requirements.")
    else:
        print("💥 SOME TESTS FAILED! Check individual target results above.")

    return all_passed


def load_combined_scores_file(scores_dir: str) -> pd.DataFrame:
    """Load the combined scores file."""
    # Look for files matching pattern scores_{date}.csv
    score_files = []
    for file in os.listdir(scores_dir):
        if file.startswith("scores_") and file.endswith(".csv"):
            score_files.append(file)

    if not score_files:
        raise FileNotFoundError(f"No scores files found in {scores_dir}")

    # If multiple files, get the latest one
    if len(score_files) > 1:
        latest_file = get_latest_scores_file(scores_dir, score_files)
    else:
        latest_file = score_files[0]

    file_path = os.path.join(scores_dir, latest_file)
    print(f"Loading scores from: {latest_file}")

    df = pd.read_csv(file_path)

    # Validate required columns
    required_cols = ["metric", "outcome", "mean"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns {missing_cols} in scores file")

    return df


def get_latest_scores_file(scores_dir: str, score_files: List[str]) -> str:
    """Get the latest scores file based on timestamp in filename."""
    latest_file = None
    latest_datetime = None

    for file in score_files:
        # Extract datetime pattern from filename (format: scores_YYYYmmdd-HHMM.csv)
        datetime_match = re.search(r"scores_(\d{8}-\d{4})\.csv", file)
        if datetime_match:
            datetime_str = datetime_match.group(1)
            try:
                file_datetime = datetime.strptime(datetime_str, "%Y%m%d-%H%M")
                if latest_datetime is None or file_datetime > latest_datetime:
                    latest_datetime = file_datetime
                    latest_file = file
            except ValueError:
                continue

    if latest_file is None:
        # Fallback to last file alphabetically if datetime parsing fails
        latest_file = sorted(score_files)[-1]

    return latest_file


def test_target_from_combined_df(
    scores_df: pd.DataFrame,
    target_name: str,
    bounds: Dict[str, float] = None,
    metric_name: str = "roc_auc",
) -> bool:
    """Test performance for a specific target from the combined dataframe."""
    try:
        # Filter for the specific target and metric
        target_metric_rows = scores_df[
            (scores_df["outcome"] == target_name) & (scores_df["metric"] == metric_name)
        ]

        if target_metric_rows.empty:
            available_metrics = (
                scores_df[scores_df["outcome"] == target_name]["metric"]
                .unique()
                .tolist()
            )
            available_targets = scores_df["outcome"].unique().tolist()

            if target_name not in available_targets:
                print(f"❌ Error: Target '{target_name}' not found in scores")
                print(f"Available targets: {available_targets}")
            else:
                print(f"❌ Error: Metric '{metric_name}' not found for {target_name}")
                print(f"Available metrics for {target_name}: {available_metrics}")
            return False

        metric_value = float(target_metric_rows["mean"].iloc[0])
        std_value = (
            float(target_metric_rows["std"].iloc[0])
            if "std" in target_metric_rows.columns
            else None
        )

        print(f"{target_name.upper()} {metric_name.upper()}:")
        if std_value is not None:
            print(f"  Value: {metric_value:.4f} ± {std_value:.4f}")
        else:
            print(f"  Value: {metric_value:.4f}")

        # Test bounds if provided
        if bounds is None:
            print(f"  ℹ️  No bounds specified for {target_name}")
            return True

        passed = True

        min_bound = bounds.get("min")
        max_bound = bounds.get("max")

        if min_bound is not None:
            if metric_value < min_bound:
                print(f"  ❌ FAIL: {metric_value:.4f} below minimum {min_bound}")
                passed = False
            else:
                print(f"  ✅ PASS: {metric_value:.4f} >= minimum {min_bound}")

        if max_bound is not None:
            if metric_value > max_bound:
                print(f"  ❌ FAIL: {metric_value:.4f} above maximum {max_bound}")
                passed = False
            else:
                print(f"  ✅ PASS: {metric_value:.4f} <= maximum {max_bound}")

        if min_bound is None and max_bound is None:
            print(f"  ℹ️  No bounds specified for {target_name}")

        return passed

    except Exception as e:
        print(f"❌ Error testing {target_name}: {e}")
        return False


def auto_detect_target_names_from_df(scores_df: pd.DataFrame) -> List[str]:
    """Auto-detect all target names from the combined scores dataframe."""
    all_targets = scores_df["outcome"].unique().tolist()
    return sorted(all_targets)


def parse_bounds_arg(bounds_str: str) -> Dict[str, float]:
    """Parse bounds string like 'min:0.6,max:0.9' into dict."""
    if not bounds_str:
        return {}

    bounds = {}
    for part in bounds_str.split(","):
        if ":" in part:
            key, value = part.split(":", 1)
            bounds[key.strip()] = float(value.strip())

    return bounds


def main():
    parser = argparse.ArgumentParser(
        description="Test multi-target performance metrics from combined CSV file"
    )
    parser.add_argument(
        "path", type=str, help="Path to directory containing 'scores' subdirectory"
    )
    parser.add_argument(
        "--metric-name",
        type=str,
        default="roc_auc",
        help="Name of metric to test (default: 'roc_auc')",
    )
    parser.add_argument(
        "--target-bounds",
        type=str,
        action="append",
        help="Bounds for specific target as 'TARGET_NAME:min:0.7,max:0.95'. Can be used multiple times.",
    )
    parser.add_argument(
        "--targets",
        type=str,
        nargs="+",
        help="Specific target names to test (if not provided, auto-detect from scores file)",
    )

    args = parser.parse_args()

    # Parse target bounds
    target_bounds = {}
    if args.target_bounds:
        for bound_spec in args.target_bounds:
            parts = bound_spec.split(":", 1)
            if len(parts) == 2:
                target_name = parts[0]
                bounds_str = parts[1]
                target_bounds[target_name] = parse_bounds_arg(bounds_str)

    print(f"Testing directory: {args.path}")
    print(f"Metric: {args.metric_name}")
    if target_bounds:
        print(f"Target bounds: {target_bounds}")

    # Run the test
    success = test_multitarget_performance(
        args.path, target_bounds, args.metric_name, args.targets
    )

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
