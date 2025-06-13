import argparse
import torch
from os.path import join
import numpy as np


def main(processed_data_dir: str, trigger_code: str):
    patients = torch.load(join(processed_data_dir, "patients.pt"))
    vocabulary = torch.load(join(processed_data_dir, "vocabulary.pt"))

    triggers = []
    outcomes = []

    trigger_token = vocabulary[trigger_code]
    for p in patients:
        if trigger_token in p.concepts:
            triggers.append(1)
        else:
            triggers.append(0)
        outcomes.append(int(p.outcome))
    triggers = np.array(triggers)
    outcomes = np.array(outcomes)

    n_patients = len(triggers)
    n_match = np.sum(triggers == outcomes)
    n_mismatch = n_patients - n_match

    # Mismatch groups
    trigger_no_outcome = np.sum((triggers == 1) & (outcomes == 0))
    outcome_no_trigger = np.sum((triggers == 0) & (outcomes == 1))

    if n_mismatch == 0:
        print(
            f"✓ Uncensored scenario: Each patient with a positive outcome has an outcome in the sequence"
        )
    else:
        percentage_difference = n_mismatch / n_patients * 100
        print(
            f"✗ Some patients have a positive outcome but no trigger in the sequence for {n_mismatch}/{n_patients} patients ({percentage_difference:.2f}%)"
        )
        print(f"  - {trigger_no_outcome} patients have trigger but no outcome")
        print(f"  - {outcome_no_trigger} patients have outcome but no trigger")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_data_dir", type=str, required=True)
    parser.add_argument("--trigger_code", type=str, required=True)
    args = parser.parse_args()
    main(args.processed_data_dir, args.trigger_code)
