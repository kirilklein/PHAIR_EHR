"""
This script tests, whether the estimate.py script works as intended.
We use generated data to achieve that.
This step should be run by the pipeline test estimate.yml after generating the data.
"""

import os

import pandas as pd

from corebehrt.constants.causal.paths import ESTIMATE_RESULTS_FILE
from corebehrt.constants.causal.data import EffectColumns, OUTCOME

ESTIMATE_RESULT_DIR = "./outputs/causal/generated/estimate_with_generated_data"
MARGIN = 0.02

# Methods that should be excluded from true effect comparison (unadjusted methods)
UNADJUSTED_METHODS = {"RD", "RR"}


def compare_estimate_result(margin, estimate_results_dir):
    """
    This function compares the estimated effect on generated data with the simulated effect.
    """
    # Read the results
    df = pd.read_csv(os.path.join(estimate_results_dir, ESTIMATE_RESULTS_FILE))

    # Filter out unadjusted methods for true effect comparison
    causal_methods_df = df[~df[EffectColumns.method].isin(UNADJUSTED_METHODS)].copy()
    unadjusted_methods_df = df[df[EffectColumns.method].isin(UNADJUSTED_METHODS)].copy()

    # Group by outcome for better organization
    outcomes = df[OUTCOME].unique() if OUTCOME in df.columns else ["default"]

    print("\n" + "=" * 80)
    print("CAUSAL EFFECT ESTIMATION RESULTS")
    print("=" * 80)

    all_passed = True
    failed_methods = []

    for outcome in outcomes:
        outcome_causal_df = (
            causal_methods_df[causal_methods_df[OUTCOME] == outcome]
            if OUTCOME in causal_methods_df.columns
            else causal_methods_df
        )
        outcome_unadjusted_df = (
            unadjusted_methods_df[unadjusted_methods_df[OUTCOME] == outcome]
            if OUTCOME in unadjusted_methods_df.columns
            else unadjusted_methods_df
        )

        print(f"\n📊 OUTCOME: {outcome.upper()}")
        print("-" * 50)

        # Check causal methods against true effects
        outcome_passed = True
        for _, row in outcome_causal_df.iterrows():
            diff = abs(row[EffectColumns.effect] - row[EffectColumns.true_effect])
            passed = diff <= abs(margin)
            status = "✓" if passed else "✗"

            print(
                f"{status} {row[EffectColumns.method]:>6}: {row[EffectColumns.effect]:7.4f} (true: {row[EffectColumns.true_effect]:7.4f}, diff: {diff:.4f})"
            )

            if not passed:
                outcome_passed = False
                all_passed = False
                failed_methods.append(f"{outcome}-{row[EffectColumns.method]}")

        # Show unadjusted methods (no true effect comparison)
        if not outcome_unadjusted_df.empty:
            print("\n  📈 Unadjusted estimates (no ground truth comparison):")
            for _, row in outcome_unadjusted_df.iterrows():
                print(
                    f"    {row[EffectColumns.method]:>6}: {row[EffectColumns.effect]:7.4f}"
                )

        print(
            f"\n  Outcome {outcome}: {'✅ PASSED' if outcome_passed else '❌ FAILED'}"
        )

    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 ALL METHODS PASSED - Effects within acceptable margin!")
        print(f"   Margin threshold: ±{margin:.4f}")
    else:
        print("❌ SOME METHODS FAILED")
        print(f"   Failed: {', '.join(failed_methods)}")
        print(f"   Margin threshold: ±{margin:.4f}")
    print("=" * 80)

    # Assert for test failure
    if failed_methods:
        failed_details = []
        for outcome in outcomes:
            outcome_df = (
                causal_methods_df[causal_methods_df[OUTCOME] == outcome]
                if OUTCOME in causal_methods_df.columns
                else causal_methods_df
            )
            for _, row in outcome_df.iterrows():
                diff = abs(row[EffectColumns.effect] - row[EffectColumns.true_effect])
                if diff > abs(margin):
                    failed_details.append(
                        f"{outcome}-{row[EffectColumns.method]}: estimated={row[EffectColumns.effect]:.4f}, "
                        f"true={row[EffectColumns.true_effect]:.4f}, diff={diff:.4f}"
                    )

        raise AssertionError(
            f"\n❌ {len(failed_methods)} method(s) exceeded the acceptable margin of ±{margin:.4f}:\n"
            + "\n".join(failed_details)
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--margin",
        type=float,
        default=MARGIN,
        help="Acceptable margin of error for comparing estimated vs true effect",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=ESTIMATE_RESULT_DIR,
        help="Path to the estimate result directory",
    )
    args = parser.parse_args()

    compare_estimate_result(args.margin, args.dir)
