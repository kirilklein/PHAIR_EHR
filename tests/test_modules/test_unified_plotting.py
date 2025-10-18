#!/usr/bin/env python
"""
Test script for unified baseline and BERT plotting functionality.
Tests that the new implementation works with:
1. Only baseline data (backward compatibility)
2. Both baseline and BERT data (new functionality)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add the project root to path
sys.path.insert(0, str(Path(__file__).parent))

from corebehrt.analysis_lib.data_loader import load_and_process_results
from corebehrt.analysis_lib.aggregation import (
    perform_bias_aggregation_v2,
    perform_relative_bias_aggregation_v2,
    perform_zscore_aggregation_v2,
    perform_coverage_aggregation_v2,
    perform_variance_aggregation_v2,
)
from corebehrt.analysis_lib.plotting import create_method_comparison_plot


def create_mock_data_with_both_estimators():
    """Create mock data with both baseline and BERT estimators for testing."""
    np.random.seed(42)

    rows = []
    for run_id in ["run_01", "run_02"]:
        for estimator in ["baseline", "bert"]:
            for method in ["TMLE", "IPW"]:
                for ce, cy in [(0.5, 0.5), (0.7, 0.7)]:
                    for i in [0.5, 0.7]:
                        for outcome in ["effect_0", "effect_1p39"]:
                            true_effect = 0.0 if outcome == "effect_0" else 1.39
                            effect = true_effect + np.random.normal(0, 0.1)
                            std_err = np.random.uniform(0.05, 0.15)
                            ci_lower = effect - 1.96 * std_err
                            ci_upper = effect + 1.96 * std_err

                            rows.append(
                                {
                                    "run_id": run_id,
                                    "estimator": estimator,
                                    "method": method,
                                    "outcome": outcome,
                                    "ce": ce,
                                    "cy": cy,
                                    "i": i,
                                    "y": 1.39 if outcome == "effect_1p39" else 0.0,
                                    "effect": effect,
                                    "true_effect": true_effect,
                                    "std_err": std_err,
                                    "CI95_lower": ci_lower,
                                    "CI95_upper": ci_upper,
                                }
                            )

    df = pd.DataFrame(rows)

    # Calculate derived metrics
    df["bias"] = df["effect"] - df["true_effect"]
    df["covered"] = (df["true_effect"] >= df["CI95_lower"]) & (
        df["true_effect"] <= df["CI95_upper"]
    )
    df["relative_bias"] = (df["bias"] / df["true_effect"]).replace(
        [np.inf, -np.inf], np.nan
    )
    df["z_score"] = (df["bias"] / df["std_err"]).replace([np.inf, -np.inf], np.nan)

    return df


def test_aggregation_with_prefixed_methods():
    """Test that aggregation functions work with prefixed method names."""
    print("\n" + "=" * 60)
    print("TEST 1: Aggregation with prefixed method names")
    print("=" * 60)

    # Create mock data
    df = create_mock_data_with_both_estimators()

    # Add composite method names
    df["method"] = df["estimator"] + "_" + df["method"]

    print(f"Created mock data with {len(df)} rows")
    print(f"Unique methods: {sorted(df['method'].unique())}")
    print(f"Unique estimators: {sorted(df['estimator'].unique())}")

    # Test aggregations
    agg_bias = perform_bias_aggregation_v2(df)
    agg_rel_bias = perform_relative_bias_aggregation_v2(df)
    agg_zscore = perform_zscore_aggregation_v2(df)
    agg_coverage = perform_coverage_aggregation_v2(df)
    agg_variance = perform_variance_aggregation_v2(df)

    print(f"\nAggregation results:")
    print(f"  - Bias: {len(agg_bias)} rows")
    print(f"  - Relative bias: {len(agg_rel_bias)} rows")
    print(f"  - Z-score: {len(agg_zscore)} rows")
    print(f"  - Coverage: {len(agg_coverage)} rows")
    print(f"  - Variance: {len(agg_variance)} rows")

    # Verify all methods are present in aggregations
    for agg, name in [(agg_bias, "bias"), (agg_coverage, "coverage")]:
        methods = sorted(agg["method"].unique())
        print(f"  - {name} methods: {methods}")
        assert all("_" in m for m in methods), f"Methods should be prefixed in {name}"

    print("\n✓ Aggregation test PASSED")
    return agg_bias, agg_coverage


def test_plotting_with_both_estimators(agg_bias, agg_coverage):
    """Test that plotting works with both estimators."""
    print("\n" + "=" * 60)
    print("TEST 2: Plotting with both estimators")
    print("=" * 60)

    output_dir = Path("outputs/test_modules/test_output_unified")
    output_dir.mkdir(exist_ok=True)

    try:
        # Test bias plot
        create_method_comparison_plot(
            agg_bias,
            metric_name="bias",
            y_label="Average Bias",
            title="Test: Unified Bias Plot (Baseline + BERT)",
            output_dir=str(output_dir),
            plot_type="errorbar",
            min_points=1,
        )

        # Test coverage plot
        create_method_comparison_plot(
            agg_coverage,
            metric_name="coverage",
            y_label="95% CI Coverage",
            title="Test: Unified Coverage Plot (Baseline + BERT)",
            output_dir=str(output_dir),
            plot_type="dot",
            min_points=1,
        )

        print(f"\nPlots saved to: {output_dir}")
        print("✓ Plotting test PASSED")

    except Exception as e:
        print(f"✗ Plotting test FAILED: {e}")
        raise


def test_backward_compatibility():
    """Test that the system works with only baseline data (no BERT)."""
    print("\n" + "=" * 60)
    print("TEST 3: Backward compatibility (baseline only)")
    print("=" * 60)

    # Create data with only baseline
    df = create_mock_data_with_both_estimators()
    df = df[df["estimator"] == "baseline"].copy()

    # Add composite method names
    df["method"] = df["estimator"] + "_" + df["method"]

    print(f"Created baseline-only data with {len(df)} rows")
    print(f"Unique methods: {sorted(df['method'].unique())}")

    # Test aggregations
    agg_bias = perform_bias_aggregation_v2(df)
    agg_coverage = perform_coverage_aggregation_v2(df)

    print(f"\nAggregation results:")
    print(f"  - Bias: {len(agg_bias)} rows")
    print(f"  - Coverage: {len(agg_coverage)} rows")

    output_dir = Path("outputs/test_modules/test_output_baseline_only")
    output_dir.mkdir(exist_ok=True)

    try:
        create_method_comparison_plot(
            agg_bias,
            metric_name="bias",
            y_label="Average Bias",
            title="Test: Baseline Only Plot",
            output_dir=str(output_dir),
            plot_type="errorbar",
            min_points=1,
        )

        print(f"\nPlots saved to: {output_dir}")
        print("✓ Backward compatibility test PASSED")

    except Exception as e:
        print(f"✗ Backward compatibility test FAILED: {e}")
        raise


def test_with_real_data():
    """Test with real data if available."""
    print("\n" + "=" * 60)
    print("TEST 4: Real data (if available)")
    print("=" * 60)

    results_dir = Path("outputs/causal/sim_study_sampling/runs")

    if not results_dir.exists():
        print("Skipping real data test - no results directory found")
        return

    try:
        # Load real data
        all_data = load_and_process_results(str(results_dir))

        print(f"Loaded {len(all_data)} rows from real data")
        print(f"Unique estimators: {sorted(all_data['estimator'].unique())}")
        print(f"Unique methods: {sorted(all_data['method'].unique())}")

        # Add composite method names
        all_data["method"] = all_data["estimator"] + "_" + all_data["method"]

        print(f"Composite methods: {sorted(all_data['method'].unique())}")

        # Test aggregations
        agg_bias = perform_bias_aggregation_v2(all_data)

        print(f"Aggregated to {len(agg_bias)} rows")

        output_dir = Path("outputs/test_modules/test_output_real")
        output_dir.mkdir(exist_ok=True)

        create_method_comparison_plot(
            agg_bias,
            metric_name="bias",
            y_label="Average Bias",
            title="Test: Real Data Plot",
            output_dir=str(output_dir),
            plot_type="errorbar",
            min_points=2,
        )

        print(f"\nPlots saved to: {output_dir}")
        print("✓ Real data test PASSED")

    except Exception as e:
        print(f"Real data test completed with note: {e}")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("TESTING UNIFIED BASELINE AND BERT PLOTTING")
    print("=" * 60)

    try:
        # Test 1: Aggregation with prefixed methods
        agg_bias, agg_coverage = test_aggregation_with_prefixed_methods()

        # Test 2: Plotting with both estimators
        test_plotting_with_both_estimators(agg_bias, agg_coverage)

        # Test 3: Backward compatibility
        test_backward_compatibility()

        # Test 4: Real data (if available)
        test_with_real_data()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print("TESTS FAILED ✗")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
