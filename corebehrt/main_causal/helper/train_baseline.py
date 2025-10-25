import logging
import os
import json
from dataclasses import dataclass
from datetime import datetime
from os.path import join
from typing import Any, Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd
import torch
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from corebehrt.constants.causal.data import (
    CF_PROBAS,
    EXPOSURE,
    EXPOSURE_COL,
    OUTCOME_COL,
    PROBAS,
    PS_COL,
    PROBAS_ROUND_DIGIT,
)
from corebehrt.constants.causal.paths import COMBINED_PREDICTIONS_FILE
from corebehrt.constants.data import PID_COL
from corebehrt.constants.paths import (
    FOLDS_FILE,
)
from corebehrt.functional.preparation.causal.one_hot import (
    create_features_from_patients,
)
from corebehrt.modules.preparation.causal.dataset import CausalPatientDataset
from corebehrt.modules.setup.config import Config


# Cache for GPU detection to avoid repeated logging
_CATBOOST_DEVICE_PARAMS_CACHE = None


def _get_catboost_device_params() -> Dict[str, Any]:
    """
    Detect GPU availability and return appropriate CatBoost parameters.
    Returns task_type and devices parameters for CatBoost.
    Logs only on first call (cached).
    """
    global _CATBOOST_DEVICE_PARAMS_CACHE

    if _CATBOOST_DEVICE_PARAMS_CACHE is None:
        if torch.cuda.is_available():
            logging.info("GPU detected. CatBoost will use GPU for training.")
            _CATBOOST_DEVICE_PARAMS_CACHE = {"task_type": "GPU", "devices": "0"}
        else:
            logging.info("No GPU detected. CatBoost will use CPU for training.")
            _CATBOOST_DEVICE_PARAMS_CACHE = {"task_type": "CPU"}

    return _CATBOOST_DEVICE_PARAMS_CACHE


def _prepare_catboost_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare CatBoost parameters by adding bootstrap_type if subsample is used.
    CatBoost's default 'Bayesian' bootstrap doesn't support subsample.
    When subsample is present, we use 'Bernoulli' bootstrap which is compatible.
    """
    params_copy = params.copy()
    if "subsample" in params_copy and "bootstrap_type" not in params_copy:
        params_copy["bootstrap_type"] = "Bernoulli"
    return params_copy


@dataclass
class FoldPredictionData:
    """Container for storing predictions from each fold."""

    fold_idx: int
    target_name: str
    pids: List[int]
    predictions: np.ndarray
    targets: np.ndarray
    cf_predictions: np.ndarray = None  # For counterfactual predictions (outcomes only)


def save_nested_cv_summary(
    all_results: List[pd.DataFrame], baseline_folder: str
) -> None:
    """Combines all target results and saves the final summary."""
    logger = logging.getLogger("save_summary")

    if not all_results:
        logger.warning("No nested CV results to save.")
        return

    # Combine all results into a single DataFrame
    all_results_df = pd.concat(all_results, ignore_index=True)
    all_results_df = all_results_df.sort_values(by="mean_auc", ascending=False)

    # Save the combined report
    scores_folder = join(baseline_folder, "scores")
    os.makedirs(scores_folder, exist_ok=True)
    final_report_path = join(
        scores_folder, f"scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    all_results_df.to_csv(final_report_path, index=False)

    logger.info("===== Nested CV Final Summary =====")
    logger.info(f"\n{all_results_df.to_string(index=False)}")
    logger.info(f"Final summary report saved to {final_report_path}")


def save_hyperparameters(
    target_hyperparams: Dict[str, Dict[str, Any]], baseline_folder: str
) -> None:
    """Saves the final hyperparameters for each target to a JSON file."""
    logger = logging.getLogger("save_hyperparams")

    hyperparams_folder = join(baseline_folder, "hyperparameters")
    os.makedirs(hyperparams_folder, exist_ok=True)

    hyperparams_path = join(
        hyperparams_folder,
        f"hyperparams_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )

    # Convert numpy types to Python native types for JSON serialization
    serializable_params = {}
    for target, params in target_hyperparams.items():
        serializable_params[target] = {}
        for key, value in params.items():
            if isinstance(value, (np.int64, np.int32)):
                serializable_params[target][key] = int(value)
            elif isinstance(value, (np.float64, np.float32)):
                serializable_params[target][key] = float(value)
            else:
                serializable_params[target][key] = value

    with open(hyperparams_path, "w") as f:
        json.dump(serializable_params, f, indent=2)

    logger.info(f"Hyperparameters saved to: {hyperparams_path}")
    logger.info("Final hyperparameters summary:")
    for target, params in serializable_params.items():
        logger.info(f"  {target}: {params}")


def save_combined_predictions(
    prediction_storage: List[FoldPredictionData],
    baseline_folder: str,
    outcome_names: List[str],
) -> None:
    """Combines predictions from all folds and saves in the same format as finetune_exp_y.py."""
    logger = logging.getLogger("save_predictions")

    if not prediction_storage:
        logger.warning("No fold predictions to save.")
        return

    # Group predictions by fold and organize by target
    fold_data = {}  # fold_idx -> {target_name: (pids, predictions, targets, cf_predictions)}

    for pred_data in prediction_storage:
        fold_idx = pred_data.fold_idx
        if fold_idx not in fold_data:
            fold_data[fold_idx] = {}

        fold_data[fold_idx][pred_data.target_name] = (
            pred_data.pids,
            pred_data.predictions,
            pred_data.targets,
            pred_data.cf_predictions,
        )

    # Combine all folds
    all_data = []

    for fold_idx in sorted(fold_data.keys()):
        fold_targets = fold_data[fold_idx]

        # Get PIDs from the first available target (should be consistent across targets)
        first_target = next(iter(fold_targets.keys()))
        pids, _, _, _ = fold_targets[first_target]

        # Create row data for each patient in this fold
        for i, pid in enumerate(pids):
            row_data = {PID_COL: pid}

            # Add exposure data (propensity score and target)
            if EXPOSURE in fold_targets:
                _, predictions, targets, _ = fold_targets[EXPOSURE]
                row_data[PS_COL] = predictions[i]
                row_data[EXPOSURE_COL] = int(targets[i])

            # Add outcome data
            for outcome_name in outcome_names:
                if outcome_name in fold_targets:
                    _, predictions, targets, cf_predictions = fold_targets[outcome_name]
                    row_data[f"{PROBAS}_{outcome_name}"] = predictions[i]
                    row_data[f"{OUTCOME_COL}_{outcome_name}"] = int(targets[i])
                    # Use proper counterfactual predictions
                    if cf_predictions is not None:
                        row_data[f"{CF_PROBAS}_{outcome_name}"] = cf_predictions[i]
                    else:
                        # Fallback for exposure (which doesn't have counterfactual)
                        row_data[f"{CF_PROBAS}_{outcome_name}"] = predictions[i]

            all_data.append(row_data)

    # Create DataFrame and save
    combined_df = pd.DataFrame(all_data)
    output_path = join(baseline_folder, COMBINED_PREDICTIONS_FILE)
    combined_df = combined_df.round(PROBAS_ROUND_DIGIT)
    combined_df.to_csv(output_path, index=False)

    logger.info(f"Combined predictions saved to: {output_path}")
    logger.info(f"Combined dataframe shape: {combined_df.shape}")
    logger.info(f"Columns: {list(combined_df.columns)}")


def run_hyperparameter_tuning(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    base_params: Dict[str, Any],
    config_params: Dict[str, Any],
    n_trials: int,
    scale_pos_weight: float,
) -> Dict[str, Any]:
    """
    INNER LOOP: Performs hyperparameter tuning using Optuna on a given train/val split.
    Only tunes parameters that are NOT explicitly set in the config.
    Returns the best set of parameters found.
    """
    logging.info(f"  Starting hyperparameter tuning with {n_trials} trials...")
    logging.info(
        f"  Inner train set size: {len(X_train)}, Inner val set size: {len(X_val)}"
    )
    logging.info(f"  Train class distribution: {np.bincount(y_train)}")
    logging.info(f"  Val class distribution: {np.bincount(y_val)}")
    logging.info(f"  Scale pos weight: {scale_pos_weight:.4f}")

    # Define default tuning ranges for each parameter
    TUNING_RANGES = {
        "learning_rate": ("float", 0.01, 0.3, True),  # (type, min, max, log_scale)
        "max_depth": ("int", 4, 10),  # (type, min, max)
        "subsample": ("float", 0.6, 1.0, False),
        "colsample_bylevel": ("float", 0.6, 1.0, False),
        "l2_leaf_reg": ("float", 1e-8, 10.0, True),
        "min_data_in_leaf": ("int", 1, 100),
    }

    # Determine which parameters to tune vs. fix
    # RULE: If parameter is explicitly in CONFIG → FIXED
    #       If parameter is NOT in config → TUNED (even if it has a default value)
    params_to_tune = {}
    fixed_params = {}

    for param_name, range_info in TUNING_RANGES.items():
        if param_name in config_params:
            # Parameter explicitly set in config → FIXED
            fixed_params[param_name] = config_params[param_name]
            logging.info(
                f"  Parameter '{param_name}' FIXED at {config_params[param_name]} (from config)"
            )
        else:
            # Parameter NOT in config → TUNED (ignore default values)
            params_to_tune[param_name] = range_info
            logging.info(
                f"  Parameter '{param_name}' will be TUNED in range {range_info[1:3]} (not in config)"
            )

    # Check for trivial cases
    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        logging.warning(
            "  Skipping tuning due to only one class present in inner train/val split."
        )
        return base_params

    def objective(trial: optuna.Trial):
        trial_params = {}

        # Only tune parameters not fixed by config
        for param_name, range_info in params_to_tune.items():
            if range_info[0] == "float":
                log_scale = range_info[3] if len(range_info) > 3 else False
                trial_params[param_name] = trial.suggest_float(
                    param_name, range_info[1], range_info[2], log=log_scale
                )
            elif range_info[0] == "int":
                trial_params[param_name] = trial.suggest_int(
                    param_name, range_info[1], range_info[2]
                )

        # Get GPU/CPU parameters
        device_params = _get_catboost_device_params()

        # Prepare trial params with proper bootstrap type if subsample is used
        prepared_trial_params = _prepare_catboost_params(trial_params)

        model = CatBoostClassifier(
            n_estimators=base_params["n_estimators"],
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbose=0,
            **device_params,
            **prepared_trial_params,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=base_params["early_stopping_rounds"],
            verbose=0,
        )

        preds = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds)

        # Log progress every 10 trials
        if trial.number % 10 == 0:
            logging.info(f"    Trial {trial.number + 1}/{n_trials}: AUC = {auc:.4f}")

        return auc

    if params_to_tune:
        logging.info(
            f"  Running Optuna optimization for {len(params_to_tune)} parameters..."
        )
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        logging.info(f"  Hyperparameter tuning completed!")
        logging.info(f"  Best trial AUC: {study.best_value:.4f}")
        logging.info(f"  Best tuned parameters: {study.best_params}")

        final_params = {**base_params, **study.best_params}
    else:
        logging.info("  No parameters to tune (all fixed by config)")
        final_params = base_params.copy()

    logging.info(f"  Final merged parameters: {final_params}")
    return final_params


def _setup_model_parameters(cfg: Config) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Loads and merges CatBoost and tuning parameters from the config."""
    # Only include defaults for parameters that are NEVER tuned
    NON_TUNABLE_DEFAULTS = {
        "n_estimators": 1000,
        "early_stopping_rounds": 50,
    }

    config_params = cfg.get("catboost", {})
    # Only use config params + non-tunable defaults
    base_params = {**NON_TUNABLE_DEFAULTS, **config_params}
    tuning_cfg = cfg.get("tuning", {})
    return base_params, tuning_cfg


def _prepare_data_for_modeling(
    target_name: str,
    train_data: CausalPatientDataset,
    test_data: CausalPatientDataset,
    cfg: Config,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Creates feature and target sets for training and testing."""
    X_train, y_train_all = create_features_from_patients(
        train_data.patients,
        train_data.vocab,
        cfg.get("multihot", False),
        cfg.get("include_age", True),
    )
    X_test, y_test_all = create_features_from_patients(
        test_data.patients,
        train_data.vocab,  # Use train vocab for test set
        cfg.get("multihot", False),
        cfg.get("include_age", True),
    )

    if target_name == EXPOSURE:
        X_train = X_train.drop(columns=[EXPOSURE], errors="ignore")
        X_test = X_test.drop(columns=[EXPOSURE], errors="ignore")

    # Ensure consistent columns
    missing_cols = set(X_train.columns) - set(X_test.columns)
    for col in missing_cols:
        X_test[col] = 0
    X_test = X_test[X_train.columns]

    y_train = y_train_all[target_name]
    y_test = y_test_all[target_name]

    return X_train, y_train, X_test, y_test


def _get_best_params_for_fold(
    outer_train_data: CausalPatientDataset,
    target_name: str,
    base_params: Dict,
    tuning_cfg: Dict,
    data: CausalPatientDataset,
    cfg: Config,
) -> Dict[str, Any]:
    """Performs inner-loop splitting and hyperparameter tuning."""
    logger = logging.getLogger("train_baseline")
    logger.info("  Starting hyperparameter tuning for this fold...")

    inner_val_size = tuning_cfg.get("inner_val_size", 0.2)
    n_trials = tuning_cfg.get("n_trials", 50)
    logger.info(f"  Inner validation size: {inner_val_size}")
    logger.info(f"  Number of tuning trials: {n_trials}")

    inner_train_pids, inner_val_pids = train_test_split(
        outer_train_data.get_pids(),
        test_size=inner_val_size,
        random_state=42,
    )

    logger.info(
        f"  Split outer train into inner train ({len(inner_train_pids)} patients) and inner val ({len(inner_val_pids)} patients)"
    )

    inner_train_data = data.filter_by_pids(inner_train_pids)
    inner_val_data = data.filter_by_pids(inner_val_pids)

    X_inner_train, y_inner_train, X_inner_val, y_inner_val = _prepare_data_for_modeling(
        target_name, inner_train_data, inner_val_data, cfg
    )

    scale_pos_weight = (y_inner_train == 0).sum() / max((y_inner_train == 1).sum(), 1)

    # Get config parameters for this tuning session
    config_params = cfg.get("catboost", {})

    tuned_params = run_hyperparameter_tuning(
        X_inner_train,
        y_inner_train,
        X_inner_val,
        y_inner_val,
        base_params,
        config_params,
        n_trials,
        scale_pos_weight,
    )

    logger.info("  Hyperparameter tuning completed for this fold")
    return tuned_params


def _generate_counterfactual_predictions(
    model: CatBoostClassifier,
    X_test: pd.DataFrame,
    target_name: str,
    logger: logging.Logger,
) -> np.ndarray:
    """Generate counterfactual predictions by flipping the exposure value."""
    if target_name == EXPOSURE:
        # No counterfactual for exposure itself
        return None

    if EXPOSURE not in X_test.columns:
        logger.warning(
            f"  No exposure column found for counterfactual prediction of {target_name}"
        )
        return None

    logger.info(f"  Generating counterfactual predictions for {target_name}...")

    # Create counterfactual dataset by flipping exposure
    X_cf = X_test.copy()
    X_cf[EXPOSURE] = 1 - X_cf[EXPOSURE]  # Flip exposure (0->1, 1->0)

    # Generate counterfactual predictions
    cf_predictions = model.predict_proba(X_cf)[:, 1]

    return cf_predictions


def _train_and_evaluate_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    best_params: Dict,
    logger: logging.Logger,
    test_pids: List[int],
    target_name: str,
    fold_idx: int,
    prediction_storage: List[FoldPredictionData],
) -> float:
    """Trains a final model and evaluates it on the holdout test set."""
    logger.info(f"  Training final model with outer train set size: {len(X_train)}")
    logger.info(f"  Outer test set size: {len(X_test)}")

    if np.unique(y_test).size < 2:
        logger.warning("  Only one class in outer test set. Skipping scoring.")
        return np.nan

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    logger.info(f"  Scale pos weight for final model: {scale_pos_weight:.4f}")

    # Get GPU/CPU parameters
    device_params = _get_catboost_device_params()

    # Prepare best params with proper bootstrap type if subsample is used
    prepared_best_params = _prepare_catboost_params(best_params)

    final_model = CatBoostClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        **device_params,
        **prepared_best_params,
    )

    logger.info("  Fitting final model...")
    final_model.fit(X_train, y_train, verbose=0)

    logger.info("  Generating predictions on test set...")
    y_pred_proba = final_model.predict_proba(X_test)[:, 1]
    unbiased_auc = roc_auc_score(y_test, y_pred_proba)

    logger.info(f"  Final model evaluation complete. AUC: {unbiased_auc:.4f}")

    # Generate counterfactual predictions for outcome targets
    cf_predictions = _generate_counterfactual_predictions(
        final_model, X_test, target_name, logger
    )

    # Store predictions for later combination
    # Convert to numpy array if it's a pandas Series, otherwise use as-is
    targets_array = y_test.values if hasattr(y_test, "values") else y_test

    prediction_storage.append(
        FoldPredictionData(
            fold_idx=fold_idx,
            target_name=target_name,
            pids=test_pids,
            predictions=y_pred_proba,
            targets=targets_array,
            cf_predictions=cf_predictions,
        )
    )

    return unbiased_auc


def _report_and_save_target_results(
    target_name: str, scores: List[float], logger: logging.Logger
) -> pd.DataFrame:
    """Calculates final metrics for a target and returns the results DataFrame."""
    scores = [
        s for s in scores if not np.isnan(s)
    ]  # Filter out NaNs from skipped folds
    mean_auc = np.mean(scores) if scores else 0.0
    std_auc = np.std(scores) if scores else 0.0

    logger.info(f"===== Final Nested CV Result for {target_name.upper()} =====")
    logger.info(f"  Unbiased ROC AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    logger.info(f"  Scores from all folds: {scores}")

    results_df = pd.DataFrame(
        {
            "target": [target_name],
            "mean_auc": [mean_auc],
            "std_auc": [std_auc],
        }
    )
    return results_df


def nested_cv_loop(
    cfg: Config,
    logger: logging.Logger,
    data: CausalPatientDataset,
    folds: list,
) -> Tuple[List[pd.DataFrame], List[FoldPredictionData], Dict[str, Dict[str, Any]]]:
    """
    Manages the nested cross-validation process using helper functions.
    Returns:
        - List of results DataFrames for each target
        - Collected predictions from all folds
        - Final hyperparameters used for each target
    """
    targets_to_train = cfg.get("targets", [EXPOSURE] + data.get_outcome_names())
    logger.info(f"Starting Nested Cross-Validation for targets: {targets_to_train}")

    base_params, tuning_cfg = _setup_model_parameters(cfg)
    should_tune = tuning_cfg.get("tune_hyperparameters", True)
    reuse_hyperparameters = tuning_cfg.get("reuse_hyperparameters", True)

    logger.info(f"Hyperparameter tuning enabled: {should_tune}")
    if should_tune:
        logger.info(f"Reuse hyperparameters across folds: {reuse_hyperparameters}")
        logger.info(f"Base parameters: {base_params}")
        logger.info(f"Tuning configuration: {tuning_cfg}")

    all_results = []
    prediction_storage = []  # Store all predictions
    target_hyperparams = {}  # Store final hyperparameters for each target

    for target_name in targets_to_train:
        logger.info(f"\n===== Processing Target: {target_name.upper()} =====")
        all_unbiased_scores = []
        best_params_for_target = None

        for i, fold_dict in enumerate(folds):
            logger.info(f"--- Outer Fold {i + 1}/{len(folds)} ---")
            logger.info(f"Train patients in this fold: {len(fold_dict['train'])}")
            logger.info(f"Test patients in this fold: {len(fold_dict['val'])}")

            outer_train_data = data.filter_by_pids(fold_dict["train"])
            outer_test_data = data.filter_by_pids(fold_dict["val"])
            test_pids = outer_test_data.get_pids()

            best_params = base_params
            if should_tune:
                if not reuse_hyperparameters or best_params_for_target is None:
                    logger.info("  Starting hyperparameter tuning for this fold...")
                    tuned_params = _get_best_params_for_fold(
                        outer_train_data,
                        target_name,
                        base_params,
                        tuning_cfg,
                        data,
                        cfg,
                    )
                    best_params = tuned_params
                    if reuse_hyperparameters:
                        best_params_for_target = tuned_params
                        logger.info(
                            "  Saved hyperparameters for reuse in subsequent folds"
                        )
                else:
                    logger.info("  Reusing hyperparameters from the first fold")
                    logger.info(f"  Reused parameters: {best_params_for_target}")
                    best_params = best_params_for_target
            else:
                logger.info("  Using base parameters (no tuning)")
                logger.info(f"  Parameters: {best_params}")

            X_outer_train, y_outer_train, X_outer_test, y_outer_test = (
                _prepare_data_for_modeling(
                    target_name, outer_train_data, outer_test_data, cfg
                )
            )

            unbiased_auc = _train_and_evaluate_fold(
                X_outer_train,
                y_outer_train,
                X_outer_test,
                y_outer_test,
                best_params,
                logger,
                test_pids,
                target_name,
                i,
                prediction_storage,
            )

            all_unbiased_scores.append(unbiased_auc)
            if not np.isnan(unbiased_auc):
                logger.info(
                    f"  UNBIASED AUC on outer test set (Fold {i + 1}): {unbiased_auc:.4f}"
                )

            logger.info(f"--- Completed Outer Fold {i + 1}/{len(folds)} ---\n")

        target_results = _report_and_save_target_results(
            target_name, all_unbiased_scores, logger
        )
        all_results.append(target_results)

        # Store final hyperparameters for this target
        if best_params_for_target is not None:
            target_hyperparams[target_name] = best_params_for_target
        else:
            target_hyperparams[target_name] = base_params

    return all_results, prediction_storage, target_hyperparams


def handle_folds(cfg: Config, logger: logging.Logger) -> list:
    """
    Load predefined folds, log and persist them into the model directory, and return.
    """
    folds_path = join(cfg.paths.prepared_data, FOLDS_FILE)
    folds = torch.load(folds_path)
    n_folds = len(folds)
    logger.info(f"Using {n_folds} predefined folds")
    torch.save(folds, join(cfg.paths.model, FOLDS_FILE))
    return folds
