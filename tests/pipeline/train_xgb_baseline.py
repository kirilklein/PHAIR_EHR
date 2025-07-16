"""
Test the baseline classifier for EXPOSURE and OUTCOMES prediction using prepared patient data with one-hot encoding.
Baseline classifier for multi-target prediction using prepared patient data with one-hot encoding.
"""

import argparse
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from corebehrt.modules.preparation.causal.dataset import CausalPatientData


def create_multihot_features_from_patients(
    patients: list[CausalPatientData], vocabulary: dict, multihot: bool = False
) -> tuple:
    """Create multi-hot encoded features from patient concept sequences."""
    print("Extracting concepts from patients...")

    # Extract all concepts and targets
    feature_matrix = np.zeros((len(patients), len(vocabulary)), dtype=int)
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

    # Extract all targets (exposure + outcomes)
    targets_dict = {}

    # Extract exposure target
    exposure_targets = []
    for patient in patients:
        if patient.exposure is not None:
            exposure_targets.append(patient.exposure)
        else:
            exposure_targets.append(0)  # Default to 0 if missing
    targets_dict["exposure"] = np.array(exposure_targets)

    # Extract outcome targets
    if patients[0].outcomes is not None:
        outcome_names = list(patients[0].outcomes.keys())
        for outcome_name in outcome_names:
            outcome_targets = []
            for patient in patients:
                if patient.outcomes is not None and outcome_name in patient.outcomes:
                    outcome_targets.append(patient.outcomes[outcome_name])
                else:
                    outcome_targets.append(0)  # Default to 0 if missing
            targets_dict[outcome_name] = np.array(outcome_targets)

    # Clean feature names for XGBoost
    def clean_feature_name(name):
        """Clean feature name by removing problematic characters."""
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

    clean_feature_names = [clean_feature_name(key) for key in vocabulary.keys()]

    seen = set()
    unique_names = []
    for name in clean_feature_names:
        original_name = name
        counter = 1
        while name in seen:
            name = f"{original_name}_{counter}"
            counter += 1
        seen.add(name)
        unique_names.append(name)

    # Convert to DataFrame for easier handling
    feature_df = pd.DataFrame(feature_matrix, columns=unique_names)

    # -------------------------------------------------------------------
    # >>>>>>> THE FIX: Add exposure as a feature for outcome prediction <<<<<<<
    # -------------------------------------------------------------------
    if "exposure" in targets_dict:
        feature_df["exposure"] = targets_dict["exposure"]
    # -------------------------------------------------------------------

    # Print contingency tables for each target
    for target_name, target_values in targets_dict.items():
        if target_name == "exposure" and "EXPOSURE" in vocabulary:
            exposure_idx = vocabulary["EXPOSURE"]
            contingency_table = compute_2x2_matrix(
                target_values, feature_matrix, exposure_idx
            )
            normalized_contingency_table = contingency_table / np.sum(contingency_table)
            contingency_df = pd.DataFrame(
                normalized_contingency_table,
                index=[f"EXPOSURE=0", f"EXPOSURE=1"],
                columns=[f"{target_name}=0", f"{target_name}=1"],
            )
            print("=" * 50)
            print(
                f"CONTINGENCY TABLE FOR {target_name.upper()} (rows: exposure, columns: {target_name})"
            )
            print("=" * 50)
            print(contingency_df)
            print("=" * 50)

    print(f"Feature matrix shape: {feature_df.shape}")
    for target_name, target_values in targets_dict.items():
        print(
            f"{target_name} distribution: {pd.Series(target_values).value_counts().to_dict()}"
        )

    return feature_df, targets_dict


def compute_2x2_matrix(targets, feature_matrix, exposure_idx):
    """
    Compute 2x2 contingency table from targets and multi-hot feature matrix.

    Args:
        targets: Binary target array/list (0/1)
        feature_matrix: Multi-hot feature matrix
        exposure_idx: Index of exposure feature in the feature matrix

    Returns:
        2x2 numpy array: [[unexposed_no_outcome, unexposed_outcome],
                         [exposed_no_outcome, exposed_outcome]]
    """
    exposure = feature_matrix[:, exposure_idx]  # Extract exposure column

    # Create 2x2 matrix
    matrix = np.array(
        [
            [
                np.sum((exposure == 0) & (targets == 0)),
                np.sum((exposure == 0) & (targets == 1)),
            ],  # Unexposed
            [
                np.sum((exposure == 1) & (targets == 0)),
                np.sum((exposure == 1) & (targets == 1)),
            ],  # Exposed
        ]
    )

    return matrix


def train_xgb_for_target(X_train, y_train, X_val, y_val, concept_names, target_name):
    """Train XGBoost model for a specific target."""

    print(f"\n--- Training XGBoost for {target_name.upper()} ---")

    # Check if we have any positive cases
    if y_train.sum() == 0:
        print(f"❌ No positive cases for {target_name} in training set!")
        return None

    if y_val.sum() == 0:
        print(f"⚠️  No positive cases for {target_name} in validation set!")

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    xgb_model = xgb.XGBClassifier(
        n_estimators=10,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
    )

    xgb_model.fit(X_train, y_train)
    y_pred_proba_xgb = xgb_model.predict_proba(X_val)[:, 1]
    y_pred_xgb = xgb_model.predict(X_val)

    xgb_auc = roc_auc_score(y_val, y_pred_proba_xgb)

    print(f"XGBoost ROC-AUC for {target_name}: {xgb_auc:.4f}")
    print("Classification Report:")
    print(classification_report(y_val, y_pred_xgb))

    # Feature importance
    xgb_importance = pd.DataFrame(
        {
            "concept": concept_names,
            "importance_pct": (xgb_model.feature_importances_ * 100).round(1),
        }
    ).sort_values("importance_pct", ascending=False)

    print(f"Top 5 important concepts for {target_name} (XGBoost):")
    print(xgb_importance.head(5))

    results = {
        "target": target_name,
        "model": xgb_model,
        "auc": xgb_auc,
        "predictions": y_pred_proba_xgb,
        "importance": xgb_importance,
    }

    return results


def plot_results_multitarget(y_val_dict, results_dict):
    """Plot ROC curves and feature importance for multiple targets."""

    n_targets = len(results_dict)
    _, axes = plt.subplots(1, min(n_targets + 1, 4), figsize=(15, 5))
    if n_targets == 1:
        axes = [axes]

    # ROC curves
    ax_roc = axes[0]
    for target_name, result in results_dict.items():
        if result is not None:
            y_val = y_val_dict[target_name]
            fpr, tpr, _ = roc_curve(y_val, result["predictions"])
            ax_roc.plot(fpr, tpr, label=f"{target_name} (AUC = {result['auc']:.3f})")

    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curves Comparison")
    ax_roc.legend()

    # Feature importance for each target (up to 3 additional plots)
    for idx, (target_name, result) in enumerate(list(results_dict.items())[:3]):
        if result is not None and idx + 1 < len(axes):
            ax = axes[idx + 1]
            top_features = result["importance"].head(10)
            ax.barh(range(len(top_features)), top_features["importance"])
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features["concept"])
            ax.set_xlabel("Importance")
            ax.set_title(f"Top 10 Features - {target_name}")

    plt.tight_layout()
    plt.show()


def main(
    data_path: str,
    min_roc_auc: float = 0.5,
    expected_most_important_concept: str = None,
    multihot: bool = False,
    targets: list = None,
    target_bounds: dict = None,
):
    """Main function to run the multi-target prepared data baseline experiment."""

    print("=" * 60)
    print("MULTI-TARGET BASELINE CLASSIFICATION - PREPARED DATA")
    print("=" * 60)

    # Load prepared patient data
    patients = torch.load(os.path.join(data_path, "patients.pt"))
    vocabulary = torch.load(os.path.join(data_path, "vocabulary.pt"))

    # Create multi-hot features
    print("\nCreating multi-hot encoded features...")
    feature_df, targets_dict = create_multihot_features_from_patients(
        patients, vocabulary, multihot
    )

    # Determine which targets to train
    available_targets = list(targets_dict.keys())
    if targets is None:
        targets_to_train = available_targets
    else:
        targets_to_train = [t for t in targets if t in available_targets]
        if not targets_to_train:
            print(
                f"❌ None of the specified targets {targets} found in available targets {available_targets}"
            )
            return None

    print(f"Training targets: {targets_to_train}")

    # Check if we have enough data for training
    if len(feature_df) < 10:
        print("❌ Not enough patients for training!")
        return None

    # Remove features with zero variance (all zeros)
    non_zero_features = feature_df.sum() > 0
    if not non_zero_features.all():
        print(
            f"Removing {(~non_zero_features).sum()} features with zero counts across all patients"
        )
        feature_df = feature_df.loc[:, non_zero_features]

    print(f"Final feature matrix shape: {feature_df.shape}")

    # Check for empty feature matrix
    if feature_df.shape[1] == 0:
        print("❌ No features remaining after filtering!")
        return None

    # Train models for each target
    all_results = {}
    all_passed = True

    for target_name in targets_to_train:
        print(f"\n{'=' * 60}")
        print(f"PROCESSING TARGET: {target_name.upper()}")
        print(f"{'=' * 60}")

        target_values = targets_dict[target_name]

        # Check if target has variation
        if target_values.sum() == 0 or target_values.sum() == len(target_values):
            print(f"❌ No variation in {target_name} labels!")
            all_results[target_name] = None
            continue

        print(f"Total patients: {len(feature_df)}")
        print(f"Features: {feature_df.shape[1]}")
        print(
            f"Positive class ({target_name}): {target_values.sum()} ({target_values.mean():.3f})"
        )

        # By default, use all features.
        X_for_training = feature_df

        # If the target is 'exposure', drop it from the feature set to prevent data leakage.
        if target_name == "exposure":
            if "exposure" in X_for_training.columns:
                X_for_training = feature_df.drop(columns=["exposure"])
                print(
                    "✅ Temporarily removed 'exposure' from features for its own prediction."
                )
        # -------------------------------------------------------------------------

        print(f"Total patients: {len(X_for_training)}")
        print(f"Features for this task: {X_for_training.shape[1]}")
        print(
            f"Positive class ({target_name}): {target_values.sum()} ({target_values.mean():.3f})"
        )

        # Train-test split
        X_train, X_val, y_train, y_val = train_test_split(
            X_for_training,
            target_values,
            test_size=0.2,
            random_state=42,
            stratify=target_values,
        )

        print(f"Train set: {len(X_train)} patients, {y_train.mean():.3f} positive rate")
        print(f"Val set: {len(X_val)} patients, {y_val.mean():.3f} positive rate")

        # Train model
        result = train_xgb_for_target(
            X_train, y_train, X_val, y_val, X_for_training.columns, target_name
        )
        all_results[target_name] = result

        if result is None:
            all_passed = False
            continue

        auc = result["auc"]
        most_important_concept_name = result["importance"].iloc[0]["concept"]

        # Test: ROC-AUC threshold
        current_bounds = target_bounds.get(target_name, {}) if target_bounds else {}
        min_auc_threshold = current_bounds.get("min", min_roc_auc)
        max_auc_threshold = current_bounds.get("max", 1.0)

        if auc < min_auc_threshold:
            print(
                f"❌ ROC-AUC {auc:.4f} is below threshold {min_auc_threshold} for {target_name}"
            )
            all_passed = False
        elif auc > max_auc_threshold:
            print(
                f"❌ ROC-AUC {auc:.4f} is above threshold {max_auc_threshold} for {target_name}"
            )
            all_passed = False
        else:
            print(f"✓ ROC-AUC {auc:.4f} passes thresholds for {target_name}")

        # Test: most important concept matches expected (only for exposure if specified)
        if expected_most_important_concept is not None and target_name == "exposure":
            if most_important_concept_name != expected_most_important_concept:
                print(
                    f"❌ Most important concept '{most_important_concept_name}' does not match expected '{expected_most_important_concept}' for {target_name}"
                )
                all_passed = False
            else:
                print(
                    f"✓ Most important concept matches expected: '{expected_most_important_concept}' for {target_name}"
                )

    # Summary
    print(f"\n{'=' * 60}")
    print("FINAL SUMMARY")
    print(f"{'=' * 60}")

    successful_results = {k: v for k, v in all_results.items() if v is not None}

    for target_name, result in successful_results.items():
        print(
            f"{target_name}: ROC-AUC = {result['auc']:.4f}, Most important = {result['importance'].iloc[0]['concept']}"
        )

    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("💥 SOME TESTS FAILED!")

    return {
        "results": successful_results,
        "all_passed": all_passed,
        "targets_trained": targets_to_train,
    }


def parse_bounds_arg(bounds_str: str) -> dict:
    """Parse bounds string like 'min:0.6,max:0.9' into dict."""
    if not bounds_str:
        return {}

    bounds = {}
    for part in bounds_str.split(","):
        if ":" in part:
            key, value = part.split(":", 1)
            bounds[key.strip()] = float(value.strip())

    return bounds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--min_roc_auc", type=float, default=0.5)
    parser.add_argument(
        "--expected_most_important_concept",
        type=str,
        required=False,
        default=None,
        help="Expected name of the most important concept for exposure (optional)",
    )
    parser.add_argument("--multihot", action="store_true", required=False)
    parser.add_argument(
        "--targets",
        type=str,
        nargs="+",
        help="Specific target names to train (if not provided, train all available targets)",
    )
    parser.add_argument(
        "--target-bounds",
        type=str,
        action="append",
        help="Bounds for specific target as 'TARGET_NAME:min:0.7,max:0.95'. Can be used multiple times.",
    )

    args = parser.parse_args()

    # Parse target bounds
    target_bounds = {}
    if args.target_bounds:
        for bound_spec in args.target_bounds:
            parts = bound_spec.split(":", 1)
            if len(parts) == 2:
                target_name = parts[0]
                bounds_str = parts[1]
                target_bounds[target_name] = parse_bounds_arg(bounds_str)

    main(
        args.data_path,
        args.min_roc_auc,
        args.expected_most_important_concept,
        args.multihot,
        args.targets,
        target_bounds,
    )
