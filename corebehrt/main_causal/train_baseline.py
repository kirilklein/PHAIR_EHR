"""
Baseline training script for causal inference using CatBoost with one-hot/multi-hot encoding.

This script is an updated version that uses CatBoost, includes robust parameter handling
with defaults, and integrates Optuna for automated hyperparameter tuning. It includes age
features and supports cross-validation with the same directory structure and configuration system.
"""

import logging
import os
from os.path import join
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import optuna
from sklearn.metrics import classification_report, roc_auc_score
from catboost import CatBoostClassifier

from corebehrt.constants.paths import (
    FOLDS_FILE,
    OUTCOME_NAMES_FILE,
    PREPARED_ALL_PATIENTS,
    TEST_PIDS_FILE,
)
from corebehrt.functional.setup.args import get_args
from corebehrt.main.helper.finetune_cv import check_for_overlap
from corebehrt.modules.monitoring.causal.metric_aggregation import (
    compute_and_save_combined_scores_mean_std,
)
from corebehrt.modules.preparation.causal.dataset import CausalPatientDataset
from corebehrt.functional.io_operations.load import load_vocabulary
from corebehrt.modules.setup.config import Config, load_config
from corebehrt.modules.setup.directory import DirectoryPreparer

CONFIG_PATH = "./corebehrt/configs/causal/finetune/baseline.yaml"

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
            if hasattr(patient, 'ages') and patient.ages:
                ages_list.append({
                    'age_mean': np.mean(patient.ages),
                    'age_min': np.min(patient.ages),
                    'age_max': np.max(patient.ages),
                    'age_std': np.std(patient.ages) if len(patient.ages) > 1 else 0,
                    'age_range': np.max(patient.ages) - np.min(patient.ages),
                })
            else:
                # Default values if no age data
                ages_list.append({
                    'age_mean': 0,
                    'age_min': 0,
                    'age_max': 0,
                    'age_std': 0,
                    'age_range': 0,
                })
        
        age_df = pd.DataFrame(ages_list)
        feature_df = pd.concat([feature_df, age_df], axis=1)

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


def train_catboost_for_target(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    target_name: str,
    cfg: Config,
) -> Dict[str, Any]:
    """Trains a single CatBoost model for a given target, with optional hyperparameter tuning."""
    print(f"\n--- Training CatBoost for {target_name.upper()} ---")

    if y_train.sum() == 0:
        print(f"❌ No positive cases for {target_name} in training set! Skipping.")
        return None
    if y_val.sum() == 0:
        print(f"⚠️ No positive cases for {target_name} in validation set!")

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    
    # --- Parameter Handling ---
    # Good defaults for CatBoost
    DEFAULT_PARAMS = {
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bylevel': 0.8,
        'l2_leaf_reg': 3,
        'early_stopping_rounds': 50,
    }
    
    # Get params from config, fall back to defaults
    config_params = cfg.get("catboost", {})
    params = {**DEFAULT_PARAMS, **config_params}
    
    tuning_cfg = cfg.get("tuning", {})
    should_tune = tuning_cfg.get("tune_hyperparameters", False)
    n_trials = tuning_cfg.get("n_trials", 50)
    
    best_params = params

    # --- Hyperparameter Tuning (Optional) ---
    if should_tune:
        print(f"Running Optuna hyperparameter search for {n_trials} trials...")
        
        def objective(trial: optuna.Trial):
            trial_params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 4, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
            }
            
            model = CatBoostClassifier(
                n_estimators=params['n_estimators'],
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                verbose=0,
                **trial_params,
            )
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=params['early_stopping_rounds'],
                verbose=0,
            )
            
            preds = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, preds)
            return auc

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        print(f"Best trial AUC: {study.best_value:.4f}")
        print(f"Best parameters: {study.best_params}")
        best_params = {**params, **study.best_params}

    # --- Final Model Training ---
    print("Training final model with best parameters...")
    final_model = CatBoostClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="AUC", # Use AUC for eval metric during fit
        **best_params,
    )
    
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False, # Keeps logs clean
    )

    y_pred_proba = final_model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    auc = roc_auc_score(y_val, y_pred_proba)

    print(f"Final CatBoost ROC-AUC for {target_name}: {auc:.4f}")
    print("Classification Report:\n", classification_report(y_val, y_pred, zero_division=0))

    importance_df = pd.DataFrame(
        {
            CONCEPT_KEY: X_train.columns,
            IMPORTANCE_PCT_KEY: final_model.get_feature_importance(prettified=False),
        }
    ).sort_values(IMPORTANCE_PCT_KEY, ascending=False)
    importance_df[IMPORTANCE_PCT_KEY] = (importance_df[IMPORTANCE_PCT_KEY] / importance_df[IMPORTANCE_PCT_KEY].sum() * 100).round(2)


    print(f"Top 5 important features for {target_name}:\n{importance_df.head(5)}")

    return {
        "target": target_name,
        "model": final_model,
        AUC_KEY: auc,
        PREDICTIONS_KEY: y_pred_proba,
        IMPORTANCE_KEY: importance_df,
    }


def train_fold(
    cfg: Config,
    logger: logging.Logger,
    baseline_folder: str,
    train_data: CausalPatientDataset,
    val_data: CausalPatientDataset,
    fold: int,
    test_data: CausalPatientDataset = None,
) -> None:
    """Train baseline models on one fold"""
    
    fold_folder = join(baseline_folder, f"fold_{fold}")
    os.makedirs(fold_folder, exist_ok=True)
    
    logger.info("Saving pids")
    torch.save(train_data.get_pids(), join(fold_folder, "train_pids.pt"))
    torch.save(val_data.get_pids(), join(fold_folder, "val_pids.pt"))
    if test_data and len(test_data) > 0:
        torch.save(test_data.get_pids(), join(fold_folder, "test_pids.pt"))

    logger.info("Creating features from patient data")
    
    train_feature_df, train_targets_dict = create_features_from_patients(
        train_data.patients, 
        train_data.vocab, 
        multihot=cfg.get("multihot", False),
        include_age=cfg.get("include_age", True)
    )
    
    val_feature_df, val_targets_dict = create_features_from_patients(
        val_data.patients, 
        train_data.vocab, 
        multihot=cfg.get("multihot", False),
        include_age=cfg.get("include_age", True)
    )
    
    missing_cols = set(train_feature_df.columns) - set(val_feature_df.columns)
    for col in missing_cols:
        val_feature_df[col] = 0
    val_feature_df = val_feature_df[train_feature_df.columns]
    
    targets_to_train = cfg.get("targets", list(train_targets_dict.keys()))
    logger.info(f"Training targets: {targets_to_train}")
    
    results = {}
    
    for target_name in targets_to_train:
        if target_name not in train_targets_dict:
            logger.warning(f"Target {target_name} not found in training data")
            continue
            
        logger.info(f"Training model for {target_name}")
        
        y_train = train_targets_dict[target_name]
        y_val = val_targets_dict.get(target_name, np.zeros(len(val_data.patients)))
        
        X_train_full = train_feature_df
        X_val_full = val_feature_df
        if target_name == EXPOSURE_KEY:
            X_train = X_train_full.drop(columns=[EXPOSURE_KEY], errors="ignore")
            X_val = X_val_full.drop(columns=[EXPOSURE_KEY], errors="ignore")
            logger.info(f"Removed '{EXPOSURE_KEY}' from features for its own prediction")
        else:
            X_train = X_train_full
            X_val = X_val_full
        
        if np.unique(y_train).size < 2:
            logger.warning(f"No variation in {target_name} labels for fold {fold}")
            continue
            
        result = train_catboost_for_target(X_train, y_train, X_val, y_val, target_name, cfg)
        if result is not None:
            results[target_name] = result
            
            result_folder = join(fold_folder, target_name)
            os.makedirs(result_folder, exist_ok=True)
            
            predictions_df = pd.DataFrame({
                'pid': val_data.get_pids(),
                'true_label': y_val,
                'predicted_proba': result[PREDICTIONS_KEY],
                'predicted_label': (result[PREDICTIONS_KEY] > 0.5).astype(int)
            })
            predictions_df.to_csv(join(result_folder, "predictions_val.csv"), index=False)
            
            result[IMPORTANCE_KEY].to_csv(join(result_folder, "feature_importance.csv"), index=False)
            
            metrics_df = pd.DataFrame({'metric': ['roc_auc'], 'value': [result[AUC_KEY]]})
            metrics_df.to_csv(join(result_folder, "metrics_val.csv"), index=False)
    
    if test_data and len(test_data) > 0:
        logger.info("Evaluating on test set")
        
        test_feature_df, test_targets_dict = create_features_from_patients(
            test_data.patients, 
            train_data.vocab, 
            multihot=cfg.get("multihot", False),
            include_age=cfg.get("include_age", True)
        )
        
        missing_cols = set(train_feature_df.columns) - set(test_feature_df.columns)
        for col in missing_cols:
            test_feature_df[col] = 0
        test_feature_df = test_feature_df[train_feature_df.columns]
        
        for target_name, result in results.items():
            if target_name not in test_targets_dict:
                continue
                
            model = result['model']
            y_test = test_targets_dict[target_name]
            
            X_test_full = test_feature_df
            if target_name == EXPOSURE_KEY:
                X_test = X_test_full.drop(columns=[EXPOSURE_KEY], errors="ignore")
            else:
                X_test = X_test_full

            if np.unique(y_test).size < 2:
                logger.warning(f"Only one class in test set for {target_name}, skipping AUC.")
                test_auc = float('nan')
            else:
                test_pred_proba = model.predict_proba(X_test)[:, 1]
                test_auc = roc_auc_score(y_test, test_pred_proba)
            
            logger.info(f"Test ROC-AUC for {target_name}: {test_auc:.4f}")
            
            result_folder = join(fold_folder, target_name)
            
            test_predictions_df = pd.DataFrame({
                'pid': test_data.get_pids(),
                'true_label': y_test,
                'predicted_proba': test_pred_proba,
                'predicted_label': (test_pred_proba > 0.5).astype(int)
            })
            test_predictions_df.to_csv(join(result_folder, "predictions_test.csv"), index=False)
            
            test_metrics_df = pd.DataFrame({'metric': ['roc_auc'], 'value': [test_auc]})
            test_metrics_df.to_csv(join(result_folder, "metrics_test.csv"), index=False)


def cv_loop(
    cfg: Config,
    logger: logging.Logger,
    baseline_folder: str,
    data: CausalPatientDataset,
    folds: list,
    test_data: CausalPatientDataset,
) -> None:
    """Loop over predefined splits for baseline training"""
    for fold, fold_dict in enumerate(folds):
        fold += 1  # 1-indexed
        train_pids = fold_dict["train"]
        val_pids = fold_dict["val"]
        logger.info(f"Training baseline fold {fold}/{len(folds)}")

        train_data = data.filter_by_pids(train_pids)
        val_data = data.filter_by_pids(val_pids)

        train_fold(cfg, logger, baseline_folder, train_data, val_data, fold, test_data)


def handle_folds(cfg: Config, test_pids: list, logger: logging.Logger) -> list:
    """
    Load folds and check for overlap with test pids.
    Save folds to model directory.
    Return folds.
    """
    folds_path = join(cfg.paths.prepared_data, FOLDS_FILE)
    folds = torch.load(folds_path)
    check_for_overlap(folds, test_pids, logger)
    n_folds = len(folds)
    logger.info(f"Using {n_folds} predefined folds")
    torch.save(folds, join(cfg.paths.model, FOLDS_FILE))
    return folds


def main_baseline(config_path: str):
    """Main baseline training function"""
    cfg = load_config(config_path)

    # Setup directories
    DirectoryPreparer(cfg).setup_finetune()

    # Logger
    logger = logging.getLogger("train_baseline")

    loaded_data = torch.load(join(cfg.paths.prepared_data, PREPARED_ALL_PATIENTS))
    vocab = load_vocabulary(cfg.paths.prepared_data)
    data = CausalPatientDataset(loaded_data, vocab)
    test_data = CausalPatientDataset([], vocab)

    # Initialize test and train/val pid lists
    test_pids = []
    
    # If evaluation is desired, then:
    if cfg.get("evaluate", False):
        if os.path.exists(join(cfg.paths.prepared_data, TEST_PIDS_FILE)):
            test_pids = torch.load(join(cfg.paths.prepared_data, TEST_PIDS_FILE))
            test_data = data.filter_by_pids(test_pids)

    # Use folds from prepared data
    folds = handle_folds(cfg, test_pids, logger)
    all_pids_in_folds = {pid for fold in folds for split in fold.values() for pid in split}
    train_val_data = data.filter_by_pids(list(all_pids_in_folds))
    
    cv_loop(cfg, logger, cfg.paths.model, train_val_data, folds, test_data)

    outcome_names = data.get_outcome_names()

    # Save outcome names to the model directory
    torch.save(outcome_names, join(cfg.paths.model, OUTCOME_NAMES_FILE))
    logger.info(f"Saved outcome names: {outcome_names}")

    # Aggregate results across folds
    try:
        compute_and_save_combined_scores_mean_std(
            len(folds), cfg.paths.model, mode="val", outcome_names=['exposure'] + outcome_names
        )

        if len(test_data) > 0:
            compute_and_save_combined_scores_mean_std(
                len(folds), cfg.paths.model, mode="test", outcome_names=['exposure'] + outcome_names
            )
    except Exception as e:
        logger.warning(f"Could not aggregate scores: {e}")

    logger.info("Baseline training completed")


if __name__ == "__main__":
    # To avoid cluttered logs from Optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    args = get_args(CONFIG_PATH)
    main_baseline(args.config_path)