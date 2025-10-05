# analyze_results.py
import argparse
from pathlib import Path

# Import the functions from your new submodules
from corebehrt.analysis_lib.data_loader import load_and_process_results
from corebehrt.analysis_lib.aggregation import (
    perform_bias_aggregation,
    perform_relative_bias_aggregation,
    perform_zscore_aggregation,
    perform_coverage_aggregation,
    perform_variance_aggregation,
)
from corebehrt.analysis_lib.plotting import create_plot_from_agg


def main():
    parser = argparse.ArgumentParser(
        description="Create simplified analysis plots for causal inference experiments."
    )
    parser.add_argument(
        "--results_dir", required=True, help="Directory containing run subdirectories."
    )
    parser.add_argument(
        "--output_dir",
        default="experiment_analysis_plots",
        help="Directory to save plots.",
    )
    parser.add_argument(
        "--outcomes",
        nargs="*",
        help="Specific outcomes to include in the analysis (default: all).",
    )
    parser.add_argument(
        "--max-subplots",
        type=int,
        default=None,
        help="Maximum number of subplots per figure (default: no limit, all in one figure).",
    )
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Load raw data
    raw_data = load_and_process_results(args.results_dir)

    # 2. Filter data based on the --outcomes argument
    if args.outcomes:
        print(f"Filtering results for specific outcomes: {', '.join(args.outcomes)}")
        initial_rows = len(raw_data)
        raw_data = raw_data[raw_data["outcome"].isin(args.outcomes)]
        print(f"Filtered data from {initial_rows} to {len(raw_data)} rows.")
        if raw_data.empty:
            print(
                "Warning: No data remains after filtering for specified outcomes. Exiting."
            )
            return

    # 3. Perform aggregations for each analysis type
    print("\n--- Performing Aggregations ---")
    agg_bias_data = perform_bias_aggregation(raw_data)
    agg_relative_bias_data = perform_relative_bias_aggregation(raw_data)
    agg_zscore_data = perform_zscore_aggregation(raw_data)
    agg_coverage_data = perform_coverage_aggregation(raw_data)
    agg_variance_data = perform_variance_aggregation(raw_data)

    # 4. Create plots for each metric
    print("\n--- Generating Plots ---")
    if args.max_subplots:
        print(f"Using maximum {args.max_subplots} subplots per figure")

    create_plot_from_agg(
        agg_bias_data,
        "bias",
        "Average Bias (± Std Dev)",
        "Average Bias",
        args.output_dir,
        "errorbar",
        max_subplots_per_figure=args.max_subplots,
    )
    create_plot_from_agg(
        agg_relative_bias_data,
        "relative_bias",
        "Average Relative Bias (± Std Dev)",
        "Relative Bias",
        args.output_dir,
        "errorbar",
        max_subplots_per_figure=args.max_subplots,
    )
    create_plot_from_agg(
        agg_zscore_data,
        "z_score",
        "Average Z-Score (± Std Dev)",
        "Standardized Bias (Z-Score)",
        args.output_dir,
        "errorbar",
        max_subplots_per_figure=args.max_subplots,
    )
    create_plot_from_agg(
        agg_coverage_data,
        "covered",
        "Coverage Probability",
        "95% CI Coverage",
        args.output_dir,
        "dot",
        max_subplots_per_figure=args.max_subplots,
    )
    create_plot_from_agg(
        agg_variance_data,
        "variance",
        "Average Empirical Variance",
        "Empirical Variance",
        args.output_dir,
        "line",
        max_subplots_per_figure=args.max_subplots,
    )

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
