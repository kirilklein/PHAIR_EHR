#!/usr/bin/env python3
"""
Combine estimate results from two-stage experiment structure.

Usage:
    python -m experiments.causal_pipeline_resample.python_scripts.combine_results \
        --results_dir /path/to/results \
        --output combined_results.csv
"""

import argparse
import pandas as pd
from pathlib import Path
import sys


def combine_estimate_results(
    results_dir: str,
    fit_methods: list = None,
) -> pd.DataFrame:
    """
    Recursively find and combine all estimate_results.csv files.

    Args:
        results_dir: Root directory containing baseline/ and/or bert/ subdirs
        fit_methods: Optional list to filter (e.g., ['baseline', 'bert'])

    Returns:
        Combined dataframe with metadata columns
    """
    all_results = []
    results_path = Path(results_dir)

    print(f"Searching for estimate_results.csv files in: {results_path}")

    # Pattern: */run_*/*/reshuffles/k_*/estimate/*/estimate_results.csv
    pattern = "*/run_*/*/reshuffles/k_*/estimate/*/estimate_results.csv"
    csv_files = list(results_path.glob(pattern))

    print(f"Found {len(csv_files)} estimate result files")

    for csv_file in csv_files:
        parts = csv_file.parts

        # Extract metadata from path
        fit_method = parts[-2]  # "baseline" or "bert"
        reshuffle_run = parts[-4]  # "k_01"

        # Find run_XX
        run_indices = [i for i, p in enumerate(parts) if p.startswith("run_")]
        if not run_indices:
            print(f"Warning: Could not find run_XX in path: {csv_file}")
            continue

        simulation_run = parts[run_indices[0]]

        # Find experiment name (between run_XX and reshuffles)
        reshuffles_idx = parts.index("reshuffles")
        experiment = parts[reshuffles_idx - 1]

        # Filter by fit_method if specified
        if fit_methods and fit_method not in fit_methods:
            continue

        try:
            # Load CSV
            df = pd.read_csv(csv_file)

            # Add metadata columns
            df["simulation_run"] = simulation_run
            df["reshuffle_run"] = reshuffle_run
            df["fit_method"] = fit_method
            df["experiment"] = experiment
            df["file_path"] = str(csv_file.relative_to(results_path))

            all_results.append(df)
            print(
                f"  ✓ Loaded: {simulation_run}/{experiment}/{reshuffle_run}/{fit_method}"
            )

        except Exception as e:
            print(f"  ✗ Error loading {csv_file}: {e}")

    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        print(
            f"\nCombined {len(all_results)} files into dataframe with {len(combined_df)} rows"
        )
        return combined_df
    else:
        print("No results found!")
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(
        description="Combine estimate results from two-stage experiment structure"
    )
    parser.add_argument(
        "--results_dir",
        required=True,
        help="Root directory containing baseline/ and/or bert/ subdirectories",
    )
    parser.add_argument(
        "--output",
        default="combined_results.csv",
        help="Output CSV file path (default: combined_results.csv)",
    )
    parser.add_argument(
        "--fit_method",
        nargs="*",
        choices=["baseline", "bert"],
        help="Filter to specific fit methods (default: include all)",
    )
    parser.add_argument(
        "--print_summary",
        action="store_true",
        help="Print summary statistics before saving",
    )

    args = parser.parse_args()

    # Combine results
    combined_df = combine_estimate_results(
        args.results_dir, fit_methods=args.fit_method
    )

    if combined_df.empty:
        print("No results to save.")
        sys.exit(1)

    # Print summary if requested
    if args.print_summary:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"\nTotal rows: {len(combined_df)}")
        print(f"\nSimulation runs: {sorted(combined_df['simulation_run'].unique())}")
        print(f"Reshuffle runs: {sorted(combined_df['reshuffle_run'].unique())}")
        print(f"Fit methods: {sorted(combined_df['fit_method'].unique())}")
        print(f"Experiments: {sorted(combined_df['experiment'].unique())}")

        print("\nCounts by fit_method:")
        print(combined_df["fit_method"].value_counts())

        print("\nColumns:", list(combined_df.columns))

    # Save to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved combined results to: {output_path}")
    print(f"  Shape: {combined_df.shape}")


if __name__ == "__main__":
    main()
