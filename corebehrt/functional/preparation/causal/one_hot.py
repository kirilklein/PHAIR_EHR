from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from collections import Counter
from corebehrt.constants.causal.data import EXPOSURE


def _generate_unique_feature_names(vocabulary: dict) -> List[str]:
    """Cleans and ensures uniqueness of feature names for model compatibility."""

    def clean_name(name: str) -> str:
        """Removes problematic characters from a single feature name."""
        cleaned = (
            str(name)
            .replace("[", "_")
            .replace("]", "_")
            .replace("<", "_")
            .replace(">", "_")
        )
        cleaned = "".join(c if c.isalnum() or c in ["_", "-"] else "_" for c in cleaned)
        if cleaned and cleaned[0].isdigit():
            cleaned = "feat_" + cleaned
        return cleaned

    cleaned_names = [clean_name(key) for key in vocabulary.keys()]

    # Ensure all names are unique by appending a counter if needed
    seen = set()
    unique_names = []
    for name in cleaned_names:
        original_name = name
        counter = 1
        while name in seen:
            name = f"{original_name}_{counter}"
            counter += 1
        seen.add(name)
        unique_names.append(name)

    return unique_names


def create_features_from_patients(
    patients: List, vocabulary: dict, multihot: bool = False, include_age: bool = True
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Creates a feature DataFrame and a dictionary of target variables."""
    print("Extracting concepts, ages, and targets from patients...")

    # Create feature matrix from patient concepts
    feature_counts = [Counter(patient.concepts) for patient in patients]
    if multihot:
        feature_matrix = np.array(
            [
                [count.get(code, 0) for code in vocabulary.values()]
                for count in feature_counts
            ]
        )
    else:
        feature_matrix = np.array(
            [
                [1 if count.get(code, 0) > 0 else 0 for code in vocabulary.values()]
                for count in feature_counts
            ]
        )

    unique_feature_names = _generate_unique_feature_names(vocabulary)
    feature_df = pd.DataFrame(feature_matrix, columns=unique_feature_names)

    # Add age features if requested
    if include_age:
        # Extract age statistics from each patient
        ages_list = []
        for patient in patients:
            if hasattr(patient, "ages") and patient.ages:
                ages_list.append(
                    {
                        "age_mean": np.mean(patient.ages),
                        "age_min": np.min(patient.ages),
                        "age_max": np.max(patient.ages),
                        "age_std": np.std(patient.ages) if len(patient.ages) > 1 else 0,
                        "age_range": np.max(patient.ages) - np.min(patient.ages),
                    }
                )
            else:
                # Default values if no age data
                ages_list.append(
                    {
                        "age_mean": 0,
                        "age_min": 0,
                        "age_max": 0,
                        "age_std": 0,
                        "age_range": 0,
                    }
                )

        age_df = pd.DataFrame(ages_list)
        feature_df = pd.concat([feature_df, age_df], axis=1)

    # Extract all targets (exposure + outcomes)
    targets_dict = {}
    targets_dict[EXPOSURE] = np.array([p.exposure or 0 for p in patients])

    if patients and patients[0].outcomes:
        outcome_names = list(patients[0].outcomes.keys())
        for name in outcome_names:
            targets_dict[name] = np.array(
                [p.outcomes.get(name, 0) if p.outcomes else 0 for p in patients]
            )

    # Add exposure as a feature for outcome prediction tasks
    feature_df[EXPOSURE] = targets_dict[EXPOSURE]
    max_print = 10
    for i, (name, values) in enumerate(targets_dict.items()):
        print(f"{name} distribution: {pd.Series(values).value_counts().to_dict()}")
        if i >= max_print:
            print("...")
            break

    return feature_df, targets_dict
