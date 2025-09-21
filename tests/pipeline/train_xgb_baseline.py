"""
Test the baseline classifier for EXPOSURE and OUTCOMES prediction using prepared
patient data with one-hot encoding. This refactored version improves code
organization and readability without changing the underlying logic or results.
"""

import argparse
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Tuple

# Add the project root to the Python path BEFORE importing corebehrt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from corebehrt.modules.preparation.causal.dataset import CausalPatientData

# --- Constants ---
EXPOSURE_KEY = "exposure"
AUC_KEY = "auc"
IMPORTANCE_KEY = "importance"
PREDICTIONS_KEY = "predictions"
CONCEPT_KEY = "concept"
IMPORTANCE_PCT_KEY = "importance_pct"


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


def create_multihot_features_from_patients(
    patients: List[CausalPatientData], vocabulary: dict, multihot: bool = False
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Creates a feature DataFrame and a dictionary of target variables."""
    print("Extracting concepts and targets from patients...")

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

    # Extract all targets (exposure + outcomes)
    targets_dict = {}
    targets_dict[EXPOSURE_KEY] = np.array([p.exposure or 0 for p in patients])

    if patients and patients[0].outcomes:
        outcome_names = list(patients[0].outcomes.keys())
        for name in outcome_names:
            targets_dict[name] = np.array(
                [p.outcomes.get(name, 0) if p.outcomes else 0 for p in patients]
            )

    # Add exposure as a feature for outcome prediction tasks
    feature_df[EXPOSURE_KEY] = targets_dict[EXPOSURE_KEY]

    print(f"Feature matrix shape: {feature_df.shape}")
    for name, values in targets_dict.items():
        print(f"{name} distribution: {pd.Series(values).value_counts().to_dict()}")

    return feature_df, targets_dict


def train_xgb_for_target(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    target_name: str,
) -> Dict[str, Any]:
    """Trains a single XGBoost model for a given target."""
    print(f"\n--- Training XGBoost for {target_name.upper()} ---")

    if y_train.sum() == 0:
        print(f"❌ No positive cases for {target_name} in training set!")
        return None
    if y_val.sum() == 0:
        print(f"⚠️ No positive cases for {target_name} in validation set!")

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = xgb.XGBClassifier(
        n_estimators=10,
        max_depth=4,
        learning_rate=0.25,
        subsample=0.8,
        colsample_bytree=1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)
    auc = roc_auc_score(y_val, y_pred_proba)

    print(f"XGBoost ROC-AUC for {target_name}: {auc:.4f}")
    print("Classification Report:\n", classification_report(y_val, y_pred))

    importance_df = pd.DataFrame(
        {
            CONCEPT_KEY: X_train.columns,
            IMPORTANCE_PCT_KEY: (model.feature_importances_ * 100).round(1),
        }
    ).sort_values(IMPORTANCE_PCT_KEY, ascending=False)

    print(f"Top 5 important concepts for {target_name}:\n{importance_df.head(5)}")

    return {
        "target": target_name,
        "model": model,
        AUC_KEY: auc,
        PREDICTIONS_KEY: y_pred_proba,
        IMPORTANCE_KEY: importance_df,
    }


def process_single_target(
    feature_df: pd.DataFrame,
    target_name: str,
    target_values: np.ndarray,
    min_roc_auc: float,
    expected_concept: str,
    target_bounds: dict,
) -> Tuple[Dict[str, Any], bool]:
    """Prepares data, trains, and evaluates a model for a single target."""
    print(f"\n{'=' * 60}\nPROCESSING TARGET: {target_name.upper()}\n{'=' * 60}")

    if np.unique(target_values).size < 2:
        print(f"❌ No variation in {target_name} labels!")
        return None, False

    # Prevent data leakage: for exposure prediction, remove exposure from features
    X_for_training = feature_df
    if target_name == EXPOSURE_KEY:
        X_for_training = feature_df.drop(columns=[EXPOSURE_KEY], errors="ignore")
        print(
            f"✅ Temporarily removed '{EXPOSURE_KEY}' from features for its own prediction."
        )

    X_train, X_val, y_train, y_val = train_test_split(
        X_for_training,
        target_values,
        test_size=0.2,
        random_state=42,
        stratify=target_values,
    )

    result = train_xgb_for_target(X_train, y_train, X_val, y_val, target_name)
    if result is None:
        return None, False

    # --- Perform checks ---
    passed = True
    auc = result[AUC_KEY]
    bounds = target_bounds.get(target_name, {})
    min_auc, max_auc = bounds.get("min", min_roc_auc), bounds.get("max", 1.0)

    if not (min_auc <= auc <= max_auc):
        print(
            f"❌ ROC-AUC {auc:.4f} is outside the allowed range [{min_auc}, {max_auc}] for {target_name}."
        )
        passed = False
    else:
        print(f"✅ ROC-AUC {auc:.4f} passes thresholds for {target_name}.")

    if expected_concept and target_name == EXPOSURE_KEY:
        most_important = result[IMPORTANCE_KEY].iloc[0][CONCEPT_KEY]
        if most_important != expected_concept:
            print(
                f"❌ Most important concept '{most_important}' != expected '{expected_concept}'."
            )
            passed = False
        else:
            print(f"✅ Most important concept matches expected: '{expected_concept}'.")

    return result, passed


def run_training_pipeline(
    feature_df: pd.DataFrame,
    targets_dict: Dict[str, np.ndarray],
    targets_to_train: List[str],
    min_roc_auc: float,
    expected_concept: str,
    target_bounds: dict,
) -> Tuple[Dict[str, Any], bool]:
    """Runs the main training loop over all specified targets."""
    all_results = {}
    all_tests_passed = True

    for target_name in targets_to_train:
        target_values = targets_dict[target_name]
        result, passed = process_single_target(
            feature_df,
            target_name,
            target_values,
            min_roc_auc,
            expected_concept,
            target_bounds,
        )
        all_results[target_name] = result
        if not passed:
            all_tests_passed = False

    return all_results, all_tests_passed


def summarize_results(results: Dict[str, Any], all_passed: bool):
    """Prints a final summary of the training outcomes."""
    print(f"\n{'=' * 60}\nFINAL SUMMARY\n{'=' * 60}")

    successful_results = {k: v for k, v in results.items() if v}
    for target, result in successful_results.items():
        most_important = result[IMPORTANCE_KEY].iloc[0][CONCEPT_KEY]
        print(
            f"{target}: ROC-AUC = {result[AUC_KEY]:.4f}, Most important = {most_important}"
        )

    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print("\n💥 SOME TESTS FAILED!")


def main(args: argparse.Namespace):
    """Main script orchestrator."""
    print("=" * 60)
    print("MULTI-TARGET BASELINE CLASSIFICATION - PREPARED DATA")
    print("=" * 60)

    patients = torch.load(os.path.join(args.data_path, "patients.pt"))
    vocabulary = torch.load(os.path.join(args.data_path, "vocabulary.pt"))

    feature_df, targets_dict = create_multihot_features_from_patients(
        patients, vocabulary, args.multihot
    )

    if len(feature_df) < 10:
        print("❌ Not enough patients for training!")
        return

    targets_to_train = args.targets or list(targets_dict.keys())
    print(f"Training targets: {targets_to_train}")

    results, all_passed = run_training_pipeline(
        feature_df,
        targets_dict,
        targets_to_train,
        args.min_roc_auc,
        args.expected_most_important_concept,
        args.target_bounds,
    )

    summarize_results(results, all_passed)


def parse_bounds_arg(value: str) -> Tuple[str, Dict[str, float]]:
    """Custom type function for parsing --target-bounds argument."""
    parts = value.split(":", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "Bounds must be in format 'TARGET:min:0.7,max:0.9'"
        )

    target_name, bounds_str = parts
    bounds = {}
    for part in bounds_str.split(","):
        key, val = part.split(":", 1)
        bounds[key.strip()] = float(val.strip())
    return target_name, bounds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Baseline classifier for exposure and outcome prediction."
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to prepared patient data."
    )
    parser.add_argument(
        "--min_roc_auc",
        type=float,
        default=0.5,
        help="Default minimum ROC-AUC threshold.",
    )
    parser.add_argument(
        "--expected_most_important_concept",
        type=str,
        default=None,
        help="Expected most important concept for exposure prediction.",
    )
    parser.add_argument(
        "--multihot",
        action="store_true",
        help="Use multi-hot (counts) instead of one-hot (binary) encoding.",
    )
    parser.add_argument(
        "--targets",
        type=str,
        nargs="+",
        help="Specific targets to train (default: all).",
    )
    parser.add_argument(
        "--target-bounds",
        type=parse_bounds_arg,
        action="append",
        default=[],
        help="Per-target bounds, e.g., 'outcome_A:min:0.7,max:0.95'",
    )

    args = parser.parse_args()

    # Convert list of tuples from argparse into a dictionary
    args.target_bounds = {target: bounds for target, bounds in args.target_bounds}

    main(args)
