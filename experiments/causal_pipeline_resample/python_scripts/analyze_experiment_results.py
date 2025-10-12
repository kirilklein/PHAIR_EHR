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
        description="Create analysis plots for resampling causal inference experiments. "
        "This version keeps outcomes separate and aggregates only over runs."
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

        # 4. Get unique outcomes and process each separately
        unique_outcomes = estimator_data["outcome"].unique()
        print(f"\nFound {len(unique_outcomes)} unique outcomes to process")
        print(f"Outcomes: {', '.join(sorted(unique_outcomes))}")

        for outcome in sorted(unique_outcomes):
            print(f"\n{'*' * 50}")
            print(f"Processing outcome: {outcome}")
            print(f"{'*' * 50}")

            # Filter data for this outcome
            outcome_data = estimator_data[estimator_data["outcome"] == outcome]
            print(f"Data rows for {outcome}: {len(outcome_data)}")

            # Create outcome-specific output directory
            outcome_output_dir = estimator_output_dir / outcome
            outcome_output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Output directory for {outcome}: {outcome_output_dir}")

            # Perform aggregations for this outcome
            print(f"\n--- Performing Aggregations for {outcome} ---")
            agg_bias_data = perform_bias_aggregation(outcome_data)
            agg_relative_bias_data = perform_relative_bias_aggregation(outcome_data)
            agg_zscore_data = perform_zscore_aggregation(outcome_data)
            agg_coverage_data = perform_coverage_aggregation(outcome_data)
            agg_variance_data = perform_variance_aggregation(outcome_data)

            # 5. Create plots for each metric (outcome-specific)
            print(f"\n--- Generating Plots for {outcome} ---")
            create_plot_from_agg(
                agg_bias_data,
                "bias",
                f"Average Bias (± Std Dev) - {outcome}",
                "Average Bias",
                str(outcome_output_dir),
                "errorbar",
                max_subplots_per_figure=args.max_subplots,
                min_points=args.min_points,
            )
            create_plot_from_agg(
                agg_relative_bias_data,
                "relative_bias",
                f"Average Relative Bias (± Std Dev) - {outcome}",
                "Relative Bias",
                str(outcome_output_dir),
                "errorbar",
                max_subplots_per_figure=args.max_subplots,
                min_points=args.min_points,
            )
            create_plot_from_agg(
                agg_zscore_data,
                "z_score",
                f"Average Z-Score (± Std Dev) - {outcome}",
                "Standardized Bias (Z-Score)",
                str(outcome_output_dir),
                "errorbar",
                max_subplots_per_figure=args.max_subplots,
                min_points=args.min_points,
            )
            create_plot_from_agg(
                agg_coverage_data,
                "covered",
                f"Coverage Probability - {outcome}",
                "95% CI Coverage",
                str(outcome_output_dir),
                "dot",
                max_subplots_per_figure=args.max_subplots,
                min_points=args.min_points,
            )
            create_plot_from_agg(
                agg_variance_data,
                "variance",
                f"Average Empirical Variance - {outcome}",
                "Empirical Variance",
                str(outcome_output_dir),
                "line",
                max_subplots_per_figure=args.max_subplots,
                min_points=args.min_points,
            )
            print(f"\nCompleted plots for {outcome}")

        print(f"\nCOMPLETED ANALYSIS FOR ESTIMATOR: {estimator.upper()}")

    print(f"\n{'=' * 60}")
    print("Analysis complete for all requested estimators!")
    print(f"Results saved in: {args.output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
