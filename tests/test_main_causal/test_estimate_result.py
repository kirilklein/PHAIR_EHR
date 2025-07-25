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
CI_STRETCH_FACTOR = 1.0

# Methods that should be excluded from true effect comparison (unadjusted methods)
UNADJUSTED_METHODS = {"RD", "RR"}


def compare_estimate_result(
    ci_stretch_factor, estimate_results_dir, ipw_ci_stretch_factor=None
):
    """
    This function compares the estimated effect on generated data with the simulated effect.
    It checks if the true effect lies within the estimated (and possibly stretched) 95% CI.
    """
    if ipw_ci_stretch_factor is None:
        ipw_ci_stretch_factor = ci_stretch_factor

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
    failed_checks = []

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
            if row[EffectColumns.method] == "IPW":
                current_ci_stretch_factor = ipw_ci_stretch_factor
            else:
                current_ci_stretch_factor = ci_stretch_factor

            true_effect = row[EffectColumns.true_effect]
            lower_ci = row[EffectColumns.CI95_lower]
            upper_ci = row[EffectColumns.CI95_upper]

            # Stretch the CI
            center = (upper_ci + lower_ci) / 2
            half_width = (upper_ci - lower_ci) / 2
            stretched_half_width = half_width * current_ci_stretch_factor
            stretched_lower = center - stretched_half_width
            stretched_upper = center + stretched_half_width

            passed = (stretched_lower <= true_effect) and (
                true_effect <= stretched_upper
            )
            status = "✓" if passed else "✗"

            print(
                f"{status} {row[EffectColumns.method]:>6}: {row[EffectColumns.effect]:7.3f} ({lower_ci:.3f}, {upper_ci:.3f}) (true: {true_effect:7.3f}). "
                f"Stretched CI ({current_ci_stretch_factor:.1f}x): [{stretched_lower:.3f}, {stretched_upper:.3f}]"
            )

            if not passed:
                outcome_passed = False
                all_passed = False
                failed_checks.append(
                    {
                        "outcome": outcome,
                        "method": row[EffectColumns.method],
                        "true_effect": true_effect,
                        "stretched_lower": stretched_lower,
                        "stretched_upper": stretched_upper,
                    }
                )

        # Show unadjusted methods (no true effect comparison)
        if not outcome_unadjusted_df.empty:
            print("\n  📈 Unadjusted estimates (no ground truth comparison):")
            for _, row in outcome_unadjusted_df.iterrows():
                print(
                    f"    {row[EffectColumns.method]:>6}: {row[EffectColumns.effect]:7.3f}"
                )

        print(
            f"\n  Outcome {outcome}: {'✅ PASSED' if outcome_passed else '❌ FAILED'}"
        )

    print("\n" + "=" * 80)
    if all_passed:
        print(
            "🎉 ALL METHODS PASSED - True effects are within the (stretched) 95% CIs!"
        )
        print(f"   CI Stretch Factor: {ci_stretch_factor:.1f}x")
    else:
        print("❌ SOME METHODS FAILED")
        failed_method_names = [
            f"{check['outcome']}-{check['method']}" for check in failed_checks
        ]
        print(f"   Failed: {', '.join(failed_method_names)}")
        print(f"   CI Stretch Factor: {ci_stretch_factor:.1f}x")
    print("=" * 80)

    # Assert for test failure
    if failed_checks:
        failed_details = []
        for check in failed_checks:
            failed_details.append(
                f"{check['outcome']}-{check['method']}: true effect {check['true_effect']:.3f} is outside of "
                f"the stretched CI [{check['stretched_lower']:.3f}, {check['stretched_upper']:.3f}]"
            )

        raise AssertionError(
            f"\n❌ {len(failed_checks)} method(s) failed the CI check (stretch factor: {ci_stretch_factor:.1f}x/ ipw sf: {ipw_ci_stretch_factor:.1f}x):\n"
            + "\n".join(failed_details)
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ci_stretch_factor",
        type=float,
        default=CI_STRETCH_FACTOR,
        help="Factor to stretch the 95% CI. 1.0 means no stretch.",
    )
    parser.add_argument(
        "--ipw_ci_stretch_factor",
        type=float,
        default=None,
        help="Factor to stretch the 95% CI for IPW. 1.0 means no stretch.",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=ESTIMATE_RESULT_DIR,
        help="Path to the estimate result directory",
    )
    args = parser.parse_args()

    compare_estimate_result(
        args.ci_stretch_factor, args.dir, args.ipw_ci_stretch_factor
    )
