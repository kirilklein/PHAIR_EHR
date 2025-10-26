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
        help="Maximum number of subplots per figure (default: no limit).",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=2,
        help="Minimum number of data points required to generate a plot (default: 2).",
    )
    parser.add_argument(
        "--estimator",
        nargs="+",  # Use '+' for one or more, as bash script will always provide at least one
        default=["baseline", "bert"],
        help="Which estimator(s) to analyze.",
    )
    args = parser.parse_args()

    # This logic correctly determines which estimators to run
    estimators_to_process = args.estimator

    print(f"\n--- Analysis configured for estimators: {estimators_to_process} ---")

    for estimator in estimators_to_process:
        print(f"\n{'=' * 60}")
        print(f"STARTING ANALYSIS FOR ESTIMATOR: {estimator.upper()}")
        print(f"{'=' * 60}")

        # 1. Load raw data for THIS estimator only.
        # We pass a list with a single item, e.g., ['baseline'], which works correctly.
        print(f"Loading data for '{estimator}'...")
        estimator_data = load_and_process_results(
            args.results_dir, estimators=[estimator]
        )

        if estimator_data.empty:
            print(f"Warning: No data found for estimator '{estimator}'. Skipping.")
            continue

        # 2. Filter data based on the --outcomes argument
        if args.outcomes:
            print(
                f"Filtering results for specific outcomes: {', '.join(args.outcomes)}"
            )
            initial_rows = len(estimator_data)
            estimator_data = estimator_data[
                estimator_data["outcome"].isin(args.outcomes)
            ]
            print(f"Filtered data from {initial_rows} to {len(estimator_data)} rows.")
            if estimator_data.empty:
                print(
                    "Warning: No data remains after filtering for specified outcomes. Skipping estimator."
                )
                continue

        # 3. Create estimator-specific output directory
        estimator_output_dir = Path(args.output_dir) / estimator
        estimator_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {estimator_output_dir}")

        # 4. Perform aggregations for this estimator
        print("\n--- Performing Aggregations ---")
        agg_bias_data = perform_bias_aggregation(estimator_data)
        agg_relative_bias_data = perform_relative_bias_aggregation(estimator_data)
        agg_zscore_data = perform_zscore_aggregation(estimator_data)
        agg_coverage_data = perform_coverage_aggregation(estimator_data)
        agg_variance_data = perform_variance_aggregation(estimator_data)

        # 5. Create plots for each metric
        print("\n--- Generating Plots ---")
        create_plot_from_agg(
            agg_bias_data,
            "bias",
            "Average Bias (± Std Dev)",
            "Average Bias",
            str(estimator_output_dir),
            "errorbar",
            max_subplots_per_figure=args.max_subplots,
            min_points=args.min_points,
        )
        create_plot_from_agg(
            agg_relative_bias_data,
            "relative_bias",
            "Average Relative Bias (± Std Dev)",
            "Relative Bias",
            str(estimator_output_dir),
            "errorbar",
            max_subplots_per_figure=args.max_subplots,
            min_points=args.min_points,
        )
        create_plot_from_agg(
            agg_zscore_data,
            "z_score",
            "Average Z-Score (± Std Dev)",
            "Standardized Bias (Z-Score)",
            str(estimator_output_dir),
            "errorbar",
            max_subplots_per_figure=args.max_subplots,
            min_points=args.min_points,
        )
        create_plot_from_agg(
            agg_coverage_data,
            "covered",
            "Coverage Probability",
            "95% CI Coverage",
            str(estimator_output_dir),
            "dot",
            max_subplots_per_figure=args.max_subplots,
            min_points=args.min_points,
        )
        create_plot_from_agg(
            agg_variance_data,
            "variance",
            "Average Empirical Variance",
            "Empirical Variance",
            str(estimator_output_dir),
            "line",
            max_subplots_per_figure=args.max_subplots,
            min_points=args.min_points,
        )
        print(f"\nCOMPLETED ANALYSIS FOR ESTIMATOR: {estimator.upper()}")

    print(f"\n{'=' * 60}")
    print("Analysis complete for all requested estimators!")
    print(f"Results saved in: {args.output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
