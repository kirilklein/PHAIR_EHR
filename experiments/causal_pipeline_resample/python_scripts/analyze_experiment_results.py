import argparse
from pathlib import Path

# Import the functions from your new submodules
from corebehrt.analysis_lib.data_loader import load_and_process_results
from corebehrt.analysis_lib.aggregation import (
    perform_bias_aggregation_v2,
    perform_relative_bias_aggregation_v2,
    perform_zscore_aggregation_v2,
    perform_coverage_aggregation_v2,
    perform_variance_aggregation_v2,
)
from corebehrt.analysis_lib.plotting import create_method_comparison_plot


def main():
    parser = argparse.ArgumentParser(
        description="Create analysis plots for resampling causal inference experiments. "
        "Compares different methods (TMLE, IPW, TMLE_TH) across outcomes with subplots for (ce, cy, i) combinations."
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
        "--min-points",
        type=int,
        default=2,
        help="Minimum number of data points required to generate a plot (default: 2).",
    )
    parser.add_argument(
        "--max-subplots",
        type=int,
        default=None,
        help="Maximum number of subplots per figure (default: no limit).",
    )
    parser.add_argument(
        "--estimator",
        nargs="+",
        default=["baseline", "bert"],
        help="Which estimator(s) to analyze.",
    )
    args = parser.parse_args()

    # Load data for requested estimators (or all if not specified)
    estimators_to_load = args.estimator if args.estimator else None

    print(f"\n{'=' * 60}")
    print(f"STARTING UNIFIED ANALYSIS")
    print(f"{'=' * 60}")

    # 1. Load raw data for ALL requested estimators at once
    print(f"Loading data for estimators: {estimators_to_load or 'all available'}...")
    try:
        all_data = load_and_process_results(
            args.results_dir, estimators=estimators_to_load
        )
    except ValueError as e:
        print(f"Error: No data found. Details: {e}")
        return

    if all_data.empty:
        print(f"Error: No data found.")
        return

    # 2. Create composite method names (baseline_TMLE, bert_IPW, etc.)
    print(f"Creating composite method names...")
    all_data["method"] = all_data["estimator"] + "_" + all_data["method"]

    # 3. Filter data based on the --outcomes argument
    if args.outcomes:
        print(f"Filtering results for specific outcomes: {', '.join(args.outcomes)}")
        initial_rows = len(all_data)
        all_data = all_data[all_data["outcome"].isin(args.outcomes)]
        print(f"Filtered data from {initial_rows} to {len(all_data)} rows.")
        if all_data.empty:
            print("Warning: No data remains after filtering for specified outcomes.")
            return

    # 4. Create single output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path}")

    # 5. Get summary statistics
    unique_estimators = all_data["estimator"].unique()
    unique_outcomes = all_data["outcome"].unique()
    unique_methods = all_data["method"].unique()
    unique_params = all_data[["ce", "cy", "i"]].drop_duplicates()

    print(f"\nData Summary:")
    print(
        f"  - Estimators: {len(unique_estimators)} ({', '.join(sorted(unique_estimators))})"
    )
    print(
        f"  - Outcomes: {len(unique_outcomes)} ({', '.join(sorted(unique_outcomes))})"
    )
    print(
        f"  - Methods (composite): {len(unique_methods)} ({', '.join(sorted(unique_methods))})"
    )
    print(f"  - Parameter combinations (ce, cy, i): {len(unique_params)}")

    # 6. Perform aggregations across all data (grouped by method, ce, cy, i, outcome)
    print(f"\n--- Performing Aggregations ---")
    agg_bias_data = perform_bias_aggregation_v2(all_data)
    agg_relative_bias_data = perform_relative_bias_aggregation_v2(all_data)
    agg_zscore_data = perform_zscore_aggregation_v2(all_data)
    agg_coverage_data = perform_coverage_aggregation_v2(all_data)
    agg_variance_data = perform_variance_aggregation_v2(all_data)

    print(f"  - Bias aggregation: {len(agg_bias_data)} rows")
    print(f"  - Relative bias aggregation: {len(agg_relative_bias_data)} rows")
    print(f"  - Z-score aggregation: {len(agg_zscore_data)} rows")
    print(f"  - Coverage aggregation: {len(agg_coverage_data)} rows")
    print(f"  - Variance aggregation: {len(agg_variance_data)} rows")

    # 7. Create unified method comparison plots for each metric
    print(f"\n--- Generating Unified Method Comparison Plots ---")

    create_method_comparison_plot(
        agg_bias_data,
        metric_name="bias",
        y_label="Average Bias",
        title="Method Comparison: Bias Across Outcomes",
        output_dir=str(output_path),
        plot_type="errorbar",
        min_points=args.min_points,
        max_subplots=args.max_subplots,
    )

    create_method_comparison_plot(
        agg_relative_bias_data,
        metric_name="relative_bias",
        y_label="Relative Bias",
        title="Method Comparison: Relative Bias Across Outcomes",
        output_dir=str(output_path),
        plot_type="errorbar",
        min_points=args.min_points,
        max_subplots=args.max_subplots,
    )

    create_method_comparison_plot(
        agg_zscore_data,
        metric_name="z_score",
        y_label="Standardized Bias (Z-Score)",
        title="Method Comparison: Z-Score Across Outcomes",
        output_dir=str(output_path),
        plot_type="errorbar",
        min_points=args.min_points,
        max_subplots=args.max_subplots,
    )

    create_method_comparison_plot(
        agg_coverage_data,
        metric_name="coverage",
        y_label="95% CI Coverage",
        title="Method Comparison: Coverage Probability Across Outcomes",
        output_dir=str(output_path),
        plot_type="dot",
        min_points=args.min_points,
        max_subplots=args.max_subplots,
    )

    create_method_comparison_plot(
        agg_variance_data,
        metric_name="variance",
        y_label="Empirical Variance",
        title="Method Comparison: Variance Across Outcomes",
        output_dir=str(output_path),
        plot_type="line",
        min_points=args.min_points,
        max_subplots=args.max_subplots,
    )

    print(f"\n{'=' * 60}")
    print("Analysis complete!")
    print(f"Results saved in: {args.output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
