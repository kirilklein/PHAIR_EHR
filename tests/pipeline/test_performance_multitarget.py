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
    
    # Load the combined scores file
    try:
        scores_df = load_combined_scores_file(scores_dir)
    except Exception as e:
        print(f"❌ Error loading scores file: {e}")
        return False
    
    all_passed = True
    
    # Test exposure metrics
    print(f"\n{'='*60}")
    print("TESTING EXPOSURE PERFORMANCE")
    print(f"{'='*60}")
    
    exposure_passed = test_target_from_combined_df(
        scores_df, "exposure", exposure_bounds, metric_name
    )
    all_passed = all_passed and exposure_passed

    # Auto-detect outcome names if not provided
    if outcome_names is None:
        outcome_names = auto_detect_outcome_names_from_df(scores_df)
    
    if not outcome_names:
        print(f"❌ Error: No outcomes found in scores file")
        return False

    # Test outcome metrics
    print(f"\n{'='*60}")
    print("TESTING OUTCOME PERFORMANCE")
    print(f"{'='*60}")
    
    print(f"Testing outcomes: {outcome_names}")
    
    for outcome_name in outcome_names:
        print(f"\n--- Testing {outcome_name} ---")
        current_bounds = outcome_bounds.get(outcome_name) if outcome_bounds else None
        
        outcome_passed = test_target_from_combined_df(
            scores_df, outcome_name, current_bounds, metric_name
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


def load_combined_scores_file(scores_dir: str) -> pd.DataFrame:
    """Load the combined scores file."""
    # Look for files matching pattern scores_{date}.csv
    score_files = []
    for file in os.listdir(scores_dir):
        if file.startswith("scores_") and file.endswith('.csv'):
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
    metric_name: str = "roc_auc"
) -> bool:
    """Test performance for a specific target from the combined dataframe."""
    try:
        # Filter for the specific target and metric
        target_metric_rows = scores_df[
            (scores_df["outcome"] == target_name) & 
            (scores_df["metric"] == metric_name)
        ]
        
        if target_metric_rows.empty:
            available_metrics = scores_df[scores_df["outcome"] == target_name]["metric"].unique().tolist()
            available_outcomes = scores_df["outcome"].unique().tolist()
            
            if target_name not in available_outcomes:
                print(f"❌ Error: Target '{target_name}' not found in scores")
                print(f"Available targets: {available_outcomes}")
            else:
                print(f"❌ Error: Metric '{metric_name}' not found for {target_name}")
                print(f"Available metrics for {target_name}: {available_metrics}")
            return False
        
        metric_value = float(target_metric_rows["mean"].iloc[0])
        std_value = float(target_metric_rows["std"].iloc[0]) if "std" in target_metric_rows.columns else None
        
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


def auto_detect_outcome_names_from_df(scores_df: pd.DataFrame) -> List[str]:
    """Auto-detect outcome names from the combined scores dataframe."""
    all_outcomes = scores_df["outcome"].unique().tolist()
    # Remove 'exposure' and return the rest as outcome names
    outcome_names = [outcome for outcome in all_outcomes if outcome != "exposure"]
    return sorted(outcome_names)


# Legacy functions kept for backward compatibility
def find_target_scores_file(scores_dir: str, target_name: str) -> str:
    """Legacy function - kept for backward compatibility."""
    raise NotImplementedError(
        "This function is deprecated. Use load_combined_scores_file instead."
    )


def auto_detect_outcome_names(scores_dir: str) -> List[str]:
    """Legacy function - kept for backward compatibility."""
    try:
        scores_df = load_combined_scores_file(scores_dir)
        return auto_detect_outcome_names_from_df(scores_df)
    except Exception:
        return []


def test_target_from_file(
    scores_dir: str,
    target_name: str, 
    bounds: Dict[str, float] = None, 
    metric_name: str = "roc_auc"
) -> bool:
    """Legacy function - kept for backward compatibility."""
    try:
        scores_df = load_combined_scores_file(scores_dir)
        return test_target_from_combined_df(scores_df, target_name, bounds, metric_name)
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


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
        description="Test multi-target performance metrics (exposure + outcomes) from combined CSV file"
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
        help="Specific outcome names to test (if not provided, auto-detect from scores file)"
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
