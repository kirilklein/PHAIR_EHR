import logging
from typing import Any, Dict, Literal, Tuple

import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV

from corebehrt.modules.setup.config import instantiate_function

logger = logging.getLogger("train_xgb")


def setup_xgb_params(
    model_cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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
        X_val: Validation features array
        y_val: Validation labels array
        params: Base XGBoost parameters
        param_space: Hyperparameter search space
        n_trials: Number of random search trials
        cv: Number of cross-validation folds
        scoring: Scoring metric for hyperparameter tuning
        early_stopping_rounds: Number of rounds to stop training if no improvement (Currently not used, need to switch to XGBoost's native early stopping)

    Returns:
        Trained XGBoost classifier model
    """
    logger.info("Starting hyperparameter tuning...")
    # Create a copy of params to avoid modifying the original
    scale_pos_weight_params = params.pop("scale_pos_weight", None)
    scale_pos_weight: float = get_scale_pos_weight(scale_pos_weight_params, y_train)

    model = xgb.XGBClassifier(**params)

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

    search.fit(X_train, y_train)

    logger.info(f"Best parameters found: {search.best_params_}")
    logger.info(f"Best CV score: {-search.best_score_:.4f} (log loss)")

    # Train final model on full training data with best parameters
    final_model = xgb.XGBClassifier(
        early_stopping_rounds=early_stopping_rounds,
        scale_pos_weight=scale_pos_weight,
        **{**params, **search.best_params_},
    )
    final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return final_model


def initialize_metrics(metrics: Dict[str, Any] = None) -> Dict[str, Any]:
    """Initialize metrics from config."""
    if metrics is None:
        metrics = {
            "roc_auc": roc_auc_score,
            "pr_auc": average_precision_score,
        }
    else:
        metrics = {
            name: instantiate_function(metric) for name, metric in metrics.items()
        }
    return metrics


def get_scale_pos_weight(scale_pos_weight_param: Any, y_train: np.ndarray) -> float:
    """Get the scale_pos_weight parameter for XGBoost. from params, if not provided, calculate it."""
    if scale_pos_weight_param is None:
        return 1.0
    if isinstance(scale_pos_weight_param, str):
        scale_pos_weight = calculate_scale_pos_weight(
            y_train, method=scale_pos_weight_param
        )
        logger.info(
            f"Calculated scale_pos_weight using {scale_pos_weight_param} method: {scale_pos_weight:.4f}"
        )
    elif isinstance(scale_pos_weight_param, (float, int)):
        scale_pos_weight = float(scale_pos_weight_param)
        logger.info(f"Using provided scale_pos_weight: {scale_pos_weight:.4f}")
    else:
        scale_pos_weight = 1.0
        logger.info("No scale_pos_weight specified, using default: 1.0")
    return scale_pos_weight


def calculate_scale_pos_weight(
    y_train: np.ndarray,
    method: Literal["simple", "log", "sqrt"] = "simple",
) -> float:
    """
    Calculate the `scale_pos_weight` parameter for XGBoost to address class imbalance.

    This function computes a weighting factor to balance positive and negative classes
    in binary classification tasks, which can help XGBoost handle imbalanced datasets.
    The calculation can be performed using different methods:

    - "simple": ratio of negative to positive samples (recommended for XGBoost)
    - "log": natural logarithm of the ratio
    - "sqrt": square root of the ratio

    Args:
        y_train (np.ndarray): Array of binary class labels (0 for negative, 1 for positive).
        method (str, optional): Method to compute the weight. One of {"simple", "log", "sqrt"}.
            Defaults to "simple".

    Returns:
        float: The computed scale_pos_weight value.

    Raises:
        ValueError: If an invalid method is specified.

    Example:
        >>> y = np.array([0, 0, 1, 0, 1])
        >>> calculate_scale_pos_weight(y, method="simple")
        1.5
    """
    # Count positive and negative samples
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)

    # Handle the case where there are no positive samples
    if pos_count == 0:
        logger.warning(
            "No positive samples found in training data. Using scale_pos_weight = 1.0"
        )
        return 1.0

    ratio = neg_count / pos_count

    if method == "simple":
        return ratio
    elif method == "log":
        return np.log(ratio) if ratio > 0 else 1
    elif method == "sqrt":
        return np.sqrt(ratio)
    else:
        raise ValueError(
            f"Invalid method: {method}. Choose from 'simple', 'log', or 'sqrt'"
        )


def calculate_metrics(
    model, X_val: np.ndarray, y_val: np.ndarray, metrics: Dict[str, Any] = None
) -> Dict[str, float]:
    """Calculate metrics on validation set."""
    y_val_pred_proba = model.predict_proba(X_val)[:, 1]
    logger.info("Validation metrics:")
    scores = {}
    for name, metric_fn in metrics.items():
        try:
            score = metric_fn(y_val, y_val_pred_proba)
            logger.info(f"  {name}: {score:.4f}")
            scores[name] = score
        except Exception as e:
            logger.warning(f"Could not compute {name}: {e}")
    return scores
