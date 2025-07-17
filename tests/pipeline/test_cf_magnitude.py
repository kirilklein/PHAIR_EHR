import argparse
from pathlib import Path
from corebehrt.constants.causal.paths import COMBINED_CALIBRATED_PREDICTIONS_FILE
import pandas as pd


def get_outcome_names(df: pd.DataFrame) -> list[str]:
    """Get outcome names from dataframe columns."""
    outcome_names = []
    for col in df.columns:
        if col.startswith("probas_"):
            outcome_names.append(col.replace("probas_", ""))
    return outcome_names


def load_ate_from_file(ate_file_path: Path) -> dict[str, float]:
    """Load ATE values from a file."""
    ate_dict = {}
    with open(ate_file_path, "r") as f:
        for line in f:
            key, value = line.strip().split(": ")
            outcome_name = key.replace("ATE_", "")
            ate_dict[outcome_name] = float(value)
    return ate_dict


def test_cf_magnitude(
    calibration_dir: Path,
    data_split_dir: Path,
    top_n_percent: float = 10.0,
    ate_tolerance: float = 0.1,
):
    """
    Tests that the mean difference for samples with the highest absolute difference
    is close to the actual exposure effect.

    Args:
        calibration_dir: Path to the calibration directory.
        data_split_dir: Path to the data split directory.
        top_n_percent: The percentage of samples with the highest absolute difference to consider.
        ate_tolerance: The tolerance for comparing the mean difference to the ATE.
    """
    predictions_path = calibration_dir / COMBINED_CALIBRATED_PREDICTIONS_FILE
    ate_file_path = data_split_dir / ".ate.txt"

    assert predictions_path.exists(), f"Predictions file not found: {predictions_path}"
    assert ate_file_path.exists(), f"ATE file not found: {ate_file_path}"

    df = pd.read_csv(predictions_path)
    ate_dict = load_ate_from_file(ate_file_path)
    outcome_names = get_outcome_names(df)

    for outcome_name in outcome_names:
        probas_col = f"probas_{outcome_name}"
        cf_probas_col = f"cf_probas_{outcome_name}"

        abs_diff = (df[probas_col] - df[cf_probas_col]).abs()

        n_samples = int(len(df) * (top_n_percent / 100.0))
        if n_samples == 0:
            print(
                f"Not enough samples for outcome {outcome_name} to select top {top_n_percent}%. Skipping."
            )
            continue

        mean_of_top_abs_diffs = abs_diff.nlargest(n_samples).mean()

        ate_value = ate_dict.get(outcome_name)
        assert ate_value is not None, (
            f"ATE for outcome {outcome_name} not found in {ate_file_path}"
        )

        print(
            f"Outcome: {outcome_name}, "
            f"Mean of top {top_n_percent}% absolute differences: {mean_of_top_abs_diffs:.4f}, "
            f"ATE: {ate_value:.4f}"
        )

        assert abs(mean_of_top_abs_diffs - ate_value) <= ate_tolerance


def main():
    """Main function to run the test from the command line."""
    parser = argparse.ArgumentParser(
        description="Test counterfactual prediction magnitude for uncertain outcomes."
    )
    parser.add_argument(
        "calibration_dir", type=Path, help="Path to the calibration directory."
    )
    parser.add_argument(
        "data_split_dir", type=Path, help="Path to the data split directory."
    )
    parser.add_argument(
        "--top_n_percent",
        type=float,
        default=10.0,
        help="The percentage of samples with the highest absolute difference to consider (default: 10.0).",
    )
    parser.add_argument(
        "--ate_tolerance",
        type=float,
        default=0.1,
        help="The tolerance for comparing the mean difference to the ATE (default: 0.1).",
    )
    args = parser.parse_args()

    test_cf_magnitude(
        calibration_dir=args.calibration_dir,
        data_split_dir=args.data_split_dir,
        top_n_percent=args.top_n_percent,
        ate_tolerance=args.ate_tolerance,
    )


if __name__ == "__main__":
    main()
