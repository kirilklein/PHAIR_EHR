"""
Test the baseline classifier for EXPOSURE prediction using prepared patient data with one-hot encoding.
Baseline classifier for EXPOSURE prediction using prepared patient data with one-hot encoding.
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
    """Create multi-hot encoded features from patient concept sequences.

    Args:
        patients: list of CausalPatientData objects
        vocabulary: dictionary mapping concept codes to their names

    Returns:
        Tuple of (feature_matrix, targets)
    """
    print("Extracting concepts from patients...")

    # Extract all concepts and targets
    targets = []
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
    try:
        targets = [patient.exposure for patient in patients]
    except Exception:
        targets = [patient.outcome for patient in patients]

    # Clean feature names for XGBoost
    def clean_feature_name(name):
        """Clean feature name by removing problematic characters."""
        # Replace problematic characters with underscores
        cleaned = (
            str(name)
            .replace("[", "_")
            .replace("]", "_")
            .replace("<", "_")
            .replace(">", "_")
        )
        # Remove any other special characters that might cause issues
        cleaned = "".join(c if c.isalnum() or c in ["_", "-"] else "_" for c in cleaned)
        # Ensure it doesn't start with a number (some ML libraries don't like this)
        if cleaned and cleaned[0].isdigit():
            cleaned = "feat_" + cleaned
        return cleaned

    # Create clean feature names
    clean_feature_names = [clean_feature_name(key) for key in vocabulary.keys()]

    # Check for duplicate names after cleaning and make them unique
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

    targets = np.array(targets)

    contingency_table = compute_2x2_matrix(
        targets, feature_matrix, vocabulary["EXPOSURE"]
    )
    normalized_contingency_table = contingency_table / np.sum(contingency_table)

    # Make a labeled DataFrame for clarity
    exposure_code = (
        "EXPOSURE" if isinstance(vocabulary, dict) else str(vocabulary["EXPOSURE"])
    )
    contingency_df = pd.DataFrame(
        normalized_contingency_table,
        index=[f"{exposure_code}=0", f"{exposure_code}=1"],
        columns=["outcome=0", "outcome=1"],
    )

    print("=" * 50)
    print("CONTINGENCY TABLE (rows: exposure, columns: outcome)")
    print("=" * 50)
    print(contingency_df)
    print("=" * 50)

    print(f"Feature matrix shape: {feature_df.shape}")
    print(f"Target distribution: {pd.Series(targets).value_counts().to_dict()}")

    return feature_df, targets


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


def train_xgb(X_train, y_train, X_val, y_val, concept_names):
    """Train Random Forest and XGBoost models."""

    print("\n" + "=" * 50)
    print("MODEL TRAINING")
    print("=" * 50)

    # XGBoost
    print("\nTraining XGBoost...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

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

    print(f"XGBoost ROC-AUC: {xgb_auc:.4f}")
    print("Classification Report:")
    print(classification_report(y_val, y_pred_xgb))

    # Feature importance
    xgb_importance = pd.DataFrame(
        {"concept": concept_names, "importance": xgb_model.feature_importances_}
    ).sort_values("importance", ascending=False)

    print("Top 10 important concepts (XGBoost):")
    print(xgb_importance.head(10))

    results = {
        "model": xgb_model,
        "auc": xgb_auc,
        "predictions": y_pred_proba_xgb,
        "importance": xgb_importance,
    }

    return results


def plot_results(y_val, results):
    """Plot ROC curves and feature importance."""

    plt.figure(figsize=(15, 5))

    # ROC curves
    plt.subplot(1, 3, 1)
    for model_name, result in results.items():
        fpr, tpr, _ = roc_curve(y_val, result["predictions"])
        plt.plot(fpr, tpr, label=f"{model_name.upper()} (AUC = {result['auc']:.3f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves Comparison")
    plt.legend()

    xgb_top = results["xgb"]["importance"].head(15)
    plt.subplot(1, 3, 3)
    plt.barh(range(len(xgb_top)), xgb_top["importance"])
    plt.yticks(range(len(xgb_top)), xgb_top["concept"])
    plt.xlabel("Importance")
    plt.title("Top 15 Concepts - XGBoost")

    plt.tight_layout()
    plt.show()


def main(
    data_path: str,
    min_roc_auc: float,
    expected_most_important_concept: str = None,
    multihot: bool = False,
):
    """Main function to run the prepared data baseline experiment."""

    print("=" * 60)
    print("BASELINE EXPOSURE CLASSIFICATION - PREPARED DATA")
    print("=" * 60)

    # Load prepared patient data
    patients = torch.load(os.path.join(data_path, "patients.pt"))
    vocabulary = torch.load(os.path.join(data_path, "vocabulary.pt"))

    # Create multi-hot features
    print("\nCreating multi-hot encoded features...")
    feature_df, targets = create_multihot_features_from_patients(
        patients, vocabulary, multihot
    )

    # Check if we have enough data for training
    if len(feature_df) < 10:
        print("❌ Not enough patients for training!")
        return None

    if targets.sum() == 0 or targets.sum() == len(targets):
        print("❌ No variation in exposure labels!")
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

    # Train-test split
    print(f"\nSplitting data...")
    print(f"Total patients: {len(feature_df)}")
    print(f"Features: {feature_df.shape[1]}")
    print(f"Positive class (EXPOSURE): {targets.sum()} ({targets.mean():.3f})")

    X_train, X_val, y_train, y_val = train_test_split(
        feature_df, targets, test_size=0.2, random_state=42, stratify=targets
    )

    print(f"Train set: {len(X_train)} patients, {y_train.mean():.3f} positive rate")
    print(f"Val set: {len(X_val)} patients, {y_val.mean():.3f} positive rate")

    # Train models
    results = train_xgb(X_train, y_train, X_val, y_val, feature_df.columns)
    auc = results["auc"]
    most_important_concept_name = results["importance"].iloc[0]["concept"]
    n_important = results["importance"].shape[0]

    # Print essentials
    print(f"ROC-AUC: {auc:.4f}")
    print(f"Most important concept: {most_important_concept_name}")

    # Test: ROC-AUC threshold
    if auc < min_roc_auc:
        print(f"❌ ROC-AUC {auc:.4f} is below threshold {min_roc_auc}")
        return None
    else:
        print(f"✓ ROC-AUC passes threshold ({min_roc_auc})")

    # Test: most important concept matches expected
    if expected_most_important_concept is not None:
        if most_important_concept_name != expected_most_important_concept:
            print(
                f"❌ Most important concept '{most_important_concept_name}' does not match expected '{expected_most_important_concept}'"
            )
            return None
        else:
            print(
                f"✓ Most important concept matches expected: '{expected_most_important_concept}'"
            )

    return {
        "auc": auc,
        "most_important_concept": most_important_concept_name,
        "n_important": n_important,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--min_roc_auc", type=float, required=True)
    parser.add_argument(
        "--expected_most_important_concept",
        type=str,
        required=False,
        default=None,
        help="Expected name of the most important concept (optional)",
    )
    parser.add_argument("--multihot", action="store_true", required=False)
    args = parser.parse_args()
    main(
        args.data_path,
        args.min_roc_auc,
        args.expected_most_important_concept,
        args.multihot,
    )
