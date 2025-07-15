#!/usr/bin/env python3

import argparse
import pandas as pd
import sys
from typing import Dict, List
import os
import re
from datetime import datetime


def test_multitarget_performance(
    ft_dir: str,
    exposure_bounds: Dict[str, float] = None,
    outcome_bounds: Dict[str, Dict[str, float]] = None,
    metric_name: str = "roc_auc",
    outcome_names: List[str] = None,
) -> bool:
    """
    Test whether performance metrics for exposure and multiple outcomes are within specified bounds.

    Args:
        ft_dir: Path to directory containing scores subdirectory
        exposure_bounds: Dict with 'min' and 'max' bounds for exposure metric
        outcome_bounds: Dict mapping outcome names to their min/max bounds
        metric_name: Name of the metric to look for (e.g., 'roc_auc')
        outcome_names: List of outcome names to test (if None, auto-detect)

    Returns:
        bool: True if all metrics are within bounds, False otherwise
    """
    scores_dir = os.path.join(ft_dir, "scores")
    if not os.path.exists(scores_dir):
        print(f"❌ Error: Scores directory not found: {scores_dir}")
        return False
    
    print(f"Looking for score files in: {scores_dir}")
    
    all_passed = True
    
    # Test exposure metrics
    print(f"\n{'='*60}")
    print("TESTING EXPOSURE PERFORMANCE")
    print(f"{'='*60}")
    
    exposure_passed = test_target_from_file(
        scores_dir, "exposure", exposure_bounds, metric_name
    )
    all_passed = all_passed and exposure_passed

    # Auto-detect outcome names if not provided
    if outcome_names is None:
        outcome_names = auto_detect_outcome_names(scores_dir)
    
    if not outcome_names:
        print(f"❌ Error: No outcome files found in {scores_dir}")
        return False

    # Test outcome metrics
    print(f"\n{'='*60}")
    print("TESTING OUTCOME PERFORMANCE")
    print(f"{'='*60}")
    
    print(f"Testing outcomes: {outcome_names}")
    
    for outcome_name in outcome_names:
        print(f"\n--- Testing {outcome_name} ---")
        current_bounds = outcome_bounds.get(outcome_name) if outcome_bounds else None
        
        outcome_passed = test_target_from_file(
            scores_dir, outcome_name, current_bounds, metric_name
        )
        all_passed = all_passed and outcome_passed

    # Overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    
    if all_passed:
        print("🎉 ALL TESTS PASSED! All targets meet performance requirements.")
    else:
        print("💥 SOME TESTS FAILED! Check individual target results above.")
    
    return all_passed


def test_target_from_file(
    scores_dir: str,
    target_name: str, 
    bounds: Dict[str, float] = None, 
    metric_name: str = "roc_auc"
) -> bool:
    """Test performance for a specific target by loading its dedicated file."""
    try:
        file_path = find_target_scores_file(scores_dir, target_name)
        print(f"Found {target_name} file: {os.path.basename(file_path)}")
        
        df = pd.read_csv(file_path)
        
        # Check if required columns exist
        required_cols = ["metric", "mean"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ Error: Missing columns {missing_cols} in {target_name} file")
            return False

        # Look for the specific metric
        metric_rows = df[df["metric"] == metric_name]
        
        if metric_rows.empty:
            print(f"❌ Error: Metric '{metric_name}' not found in {target_name} file")
            print(f"Available metrics: {df['metric'].unique().tolist()}")
            return False
        
        metric_value = float(metric_rows["mean"].iloc[0])
        std_value = float(metric_rows["std"].iloc[0]) if "std" in metric_rows.columns else None
        
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


def find_target_scores_file(scores_dir: str, target_name: str) -> str:
    """Find the scores file for a specific target (exposure or outcome)."""
    file_pattern = f"val_{target_name}_scores_mean_std"
    
    matching_files = []
    for file in os.listdir(scores_dir):
        if file.startswith(file_pattern) and file.endswith('.csv'):
            matching_files.append(file)

    if not matching_files:
        raise FileNotFoundError(
            f"No CSV files found matching pattern '{file_pattern}*' in {scores_dir}"
        )

    if len(matching_files) == 1:
        return os.path.join(scores_dir, matching_files[0])

    # Parse datetime from filenames and find the latest
    latest_file = None
    latest_datetime = None

    for file in matching_files:
        # Extract datetime pattern from filename (assuming format like YYYYmmdd-HHMM)
        datetime_match = re.search(r"(\d{8}-\d{4})", file)
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
        # Fallback to last file if datetime parsing fails
        latest_file = sorted(matching_files)[-1]

    return os.path.join(scores_dir, latest_file)


def auto_detect_outcome_names(scores_dir: str) -> List[str]:
    """Auto-detect outcome names from score files in the directory."""
    outcome_names = []
    
    for file in os.listdir(scores_dir):
        if file.startswith("val_") and file.endswith(".csv") and "_scores_mean_std" in file:
            # Extract target name from pattern: val_{target}_scores_mean_std_*.csv
            match = re.match(r"val_([^_]+)_scores_mean_std", file)
            if match:
                target_name = match.group(1)
                if target_name != "exposure":  # Skip exposure, we handle that separately
                    outcome_names.append(target_name)
    
    return sorted(list(set(outcome_names)))  # Remove duplicates and sort


def parse_bounds_arg(bounds_str: str) -> Dict[str, float]:
    """Parse bounds string like 'min:0.6,max:0.9' into dict."""
    if not bounds_str:
        return {}
    
    bounds = {}
    for part in bounds_str.split(','):
        if ':' in part:
            key, value = part.split(':', 1)
            bounds[key.strip()] = float(value.strip())
    
    return bounds


def main():
    parser = argparse.ArgumentParser(
        description="Test multi-target performance metrics (exposure + outcomes) from separate CSV files"
    )
    parser.add_argument(
        "path", type=str, help="Path to directory containing 'scores' subdirectory"
    )
    parser.add_argument(
        "--metric-name",
        type=str,
        default="roc_auc",
        help="Name of metric to test (default: 'roc_auc')"
    )
    parser.add_argument(
        "--exposure-bounds",
        type=str,
        help="Bounds for exposure metric as 'min:0.6,max:0.9'"
    )
    parser.add_argument(
        "--outcome-bounds",
        type=str,
        action="append",
        help="Bounds for specific outcome as 'OUTCOME:min:0.7,max:0.95'. Can be used multiple times."
    )
    parser.add_argument(
        "--outcomes",
        type=str,
        nargs="+",
        help="Specific outcome names to test (if not provided, auto-detect from files)"
    )

    args = parser.parse_args()

    # Parse exposure bounds
    exposure_bounds = parse_bounds_arg(args.exposure_bounds) if args.exposure_bounds else None

    # Parse outcome bounds
    outcome_bounds = {}
    if args.outcome_bounds:
        for bound_spec in args.outcome_bounds:
            parts = bound_spec.split(':', 1)
            if len(parts) == 2:
                outcome_name = parts[0]
                bounds_str = parts[1]
                outcome_bounds[outcome_name] = parse_bounds_arg(bounds_str)

    print(f"Testing directory: {args.path}")
    print(f"Metric: {args.metric_name}")
    if exposure_bounds:
        print(f"Exposure bounds: {exposure_bounds}")
    if outcome_bounds:
        print(f"Outcome bounds: {outcome_bounds}")

    # Run the test
    success = test_multitarget_performance(
        args.path,
        exposure_bounds,
        outcome_bounds,
        args.metric_name,
        args.outcomes
    )

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
