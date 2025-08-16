import argparse
import torch
from os.path import join


def main(processed_data_dir: str, exposure_code: str = "EXPOSURE"):
    patients = torch.load(join(processed_data_dir, "patients.pt"))
    vocabulary = torch.load(join(processed_data_dir, "vocabulary.pt"))

    # Check exposure consistency
    exposure_token = vocabulary[exposure_code]
    exposure_mismatches = 0
    exposure_no_code = 0
    code_no_exposure = 0

    # Check outcome consistency for all outcome types
    outcome_stats = {}

    for p in patients:
        # Check exposure consistency
        has_exposure_code = exposure_token in p.concepts
        has_exposure_label = p.exposure == 1

        if has_exposure_code != has_exposure_label:
            exposure_mismatches += 1
            if has_exposure_label and not has_exposure_code:
                exposure_no_code += 1
            elif has_exposure_code and not has_exposure_label:
                code_no_exposure += 1

        # Check outcome consistency for each outcome type
        for outcome_name, outcome_value in p.outcomes.items():
            if outcome_name not in outcome_stats:
                outcome_stats[outcome_name] = {
                    "mismatches": 0,
                    "outcome_no_code": 0,
                    "code_no_outcome": 0,
                    "total_patients": 0,
                }

            outcome_stats[outcome_name]["total_patients"] += 1

            if outcome_name in vocabulary:
                outcome_token = vocabulary[outcome_name]
                has_outcome_code = outcome_token in p.concepts
                has_outcome_label = outcome_value == 1

                if has_outcome_code != has_outcome_label:
                    outcome_stats[outcome_name]["mismatches"] += 1
                    if has_outcome_label and not has_outcome_code:
                        outcome_stats[outcome_name]["outcome_no_code"] += 1
                    elif has_outcome_code and not has_outcome_label:
                        outcome_stats[outcome_name]["code_no_outcome"] += 1

    n_patients = len(patients)

    # Report exposure results
    if exposure_mismatches == 0:
        print(
            f"✓ Exposure ({exposure_code}): All patients w/ exposure labels have corresponding codes in sequence"
        )
    else:
        percentage_diff = exposure_mismatches / n_patients * 100
        print(
            f"✗ Exposure ({exposure_code}): {exposure_mismatches}/{n_patients} patients ({percentage_diff:.2f}%) have mismatched exposure labels and codes"
        )
        print(
            f"  - {exposure_no_code} patients w/ exposure label, wo/ code in sequence"
        )
        print(
            f"  - {code_no_exposure} patients w/ code in sequence, wo/ exposure label"
        )

    # Report outcome results
    overall_success = exposure_mismatches == 0
    for outcome_name, stats in outcome_stats.items():
        if stats["mismatches"] == 0:
            print(
                f"✓ Outcome ({outcome_name}): All patients w/ outcome labels have corresponding codes in sequence"
            )
        else:
            overall_success = False
            percentage_diff = stats["mismatches"] / stats["total_patients"] * 100
            print(
                f"✗ Outcome ({outcome_name}): {stats['mismatches']}/{stats['total_patients']} patients ({percentage_diff:.2f}%) have mismatched outcome labels and codes"
            )
            print(
                f"  - {stats['outcome_no_code']} patients w/ outcome label, wo/ code in sequence"
            )
            print(
                f"  - {stats['code_no_outcome']} patients w/ code in sequence, wo/ outcome label"
            )

    if overall_success:
        print("\n✓ Uncensored scenario validation: PASSED")
    else:
        print("\n✗ Uncensored scenario validation: FAILED")
        exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_data_dir", type=str, required=True)
    parser.add_argument("--exposure_code", type=str, required=False, default="EXPOSURE")
    args = parser.parse_args()
    main(args.processed_data_dir, args.exposure_code)
