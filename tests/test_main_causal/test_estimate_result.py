"""
This script tests, whether the estimate.py script works as intended.
We use generated data to achieve that.
This step should be run by the pipeline test estimate.yml after generating the data.
"""

import pandas as pd

ESTIMATE_RESULT_PATH = (
    "./outputs/causal/generated/estimate_with_generated_data/estimate_results.csv"
)
MARGIN = 0.02


def compare_estimate_result(margin):
    """
    This function compares the estimated effect on generated data with the simulated effect.
    """
    # Read the results
    df = pd.read_csv(ESTIMATE_RESULT_PATH)

    # Check if estimated effects are within margin of true effect
    for _, row in df.iterrows():
        assert abs(row["effect"] - row["true_effect"]) <= abs(margin), (
            f"Estimated effect {row['effect']:.4f} for method {row['method']} "
            f"differs from true effect {row['true_effect']:.4f} by more than {abs(margin):.4f}"
        )
    print("All estimated effects are within the acceptable margin of error.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--margin",
        type=float,
        default=MARGIN,
        help="Acceptable margin of error for comparing estimated vs true effect",
    )
    args = parser.parse_args()

    compare_estimate_result(args.margin)
