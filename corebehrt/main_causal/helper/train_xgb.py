import logging
from typing import Any, Dict

import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

logger = logging.getLogger("train_xgb")


def setup_xgb_params(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Setup XGBoost parameters and hyperparameter search space."""
    # Base parameters
    if model_cfg.get("params") is None:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",  # For faster training
            "random_state": 42,
        }
    else:
        params = model_cfg["params"]

    # Hyperparameter search space
    if model_cfg.get("param_space") is None:
        param_space = {
            "max_depth": [3, 4, 5, 6],
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [100, 200, 300],
            "min_child_weight": [1, 3, 5],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
            "reg_alpha": [0, 0.1, 1],
            "reg_lambda": [0, 0.1, 1],
        }
    else:
        param_space = model_cfg["param_space"]

    return params, param_space


def train_xgb_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Dict[str, Any],
    param_space: Dict[str, Any],
    n_trials: int = 20,
    cv: int = 5,
    scoring: str = "neg_log_loss",
    early_stopping_rounds: int = 10,
) -> xgb.XGBClassifier:
    """Train XGBoost model with hyperparameter tuning using cross-validation.

    Args:
        X_train: Training features array
        y_train: Training labels array
        params: Base XGBoost parameters
        param_space: Hyperparameter search space
        n_trials: Number of random search trials
        logger: Optional logger instance

    Returns:
        Trained XGBoost classifier model
    """
    logger.info("Starting hyperparameter tuning...")

    # Create XGBoost classifier
    model = xgb.XGBClassifier(**params)

    # Setup RandomizedSearchCV
    search = RandomizedSearchCV(
        model,
        param_space,
        n_iter=n_trials,
        cv=cv,  # 5-fold CV on training data
        scoring=scoring,
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )

    # Fit using cross-validation on training data
    search.fit(X_train, y_train)

    logger.info(f"Best parameters found: {search.best_params_}")
    logger.info(f"Best CV score: {-search.best_score_:.4f} (log loss)")
    # Train final model on full training data with best parameters
    final_model = xgb.XGBClassifier(**{**params, **search.best_params_})
    final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    return final_model
