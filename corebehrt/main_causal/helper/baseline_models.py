"""
Model factory for the tabular baselines used as a reference for the transformer.

Two models are supported:
- `logistic`: L2-regularised logistic regression on standardised features.
- `catboost`: gradient boosting on the same features.

Both are fitted on the one-hot/multi-hot code matrix produced by
`create_features_from_patients`, so they share the nested CV machinery in
`helper/train_baseline.py`.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LOGISTIC = "logistic"
CATBOOST = "catboost"
SUPPORTED_MODELS = (LOGISTIC, CATBOOST)

# Parameters that are never tuned by Optuna, per model.
NON_TUNABLE_DEFAULTS = {
    LOGISTIC: {"max_iter": 1000},
    CATBOOST: {"n_estimators": 1000, "early_stopping_rounds": 50},
}

# (type, min, max[, log_scale]) per tunable parameter.
TUNING_RANGES = {
    LOGISTIC: {
        "C": ("float", 1e-4, 1e2, True),
    },
    CATBOOST: {
        "learning_rate": ("float", 0.01, 0.3, True),
        "max_depth": ("int", 4, 10),
        "subsample": ("float", 0.6, 1.0, False),
        "l2_leaf_reg": ("float", 1e-8, 10.0, True),
        "min_data_in_leaf": ("int", 1, 100),
        # colsample_bylevel is added below for CPU only (GPU does not support it).
    },
}

# Cache for GPU detection to avoid repeated logging
_CATBOOST_DEVICE_PARAMS_CACHE = None


def get_model_name(cfg) -> str:
    """Returns the baseline model to train, defaulting to logistic regression."""
    model_name = cfg.get("model", LOGISTIC)
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown baseline model '{model_name}'. Choose one of {SUPPORTED_MODELS}."
        )
    return model_name


def get_base_params(cfg, model_name: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns (base_params, config_params) for the selected model.

    `config_params` are the parameters explicitly set in the config; these are
    held FIXED during tuning. `base_params` additionally contains the
    non-tunable defaults.
    """
    config_params = dict(cfg.get(model_name, {}))
    base_params = {**NON_TUNABLE_DEFAULTS[model_name], **config_params}
    return base_params, config_params


def get_tuning_ranges(
    model_name: str, config_params: Dict[str, Any]
) -> Dict[str, tuple]:
    """Returns the tunable parameters and their ranges for the selected model."""
    ranges = dict(TUNING_RANGES[model_name])
    if model_name == CATBOOST:
        # colsample_bylevel is only supported on CPU for classification.
        if _effective_device_params(config_params)["task_type"] != "GPU":
            ranges["colsample_bylevel"] = ("float", 0.6, 1.0, False)
    return ranges


def build_model(
    model_name: str,
    params: Dict[str, Any],
    scale_pos_weight: float,
    random_seed: int,
) -> Any:
    """Builds an unfitted estimator with a `predict_proba` interface."""
    if model_name == LOGISTIC:
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                random_state=random_seed,
                **params,
            ),
        )

    device_params = _effective_device_params(params)
    catboost_params = _prepare_catboost_params(params, device_params)
    # early_stopping_rounds is passed to fit, not to the constructor.
    catboost_params.pop("early_stopping_rounds", None)
    return CatBoostClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=random_seed,
        verbose=0,
        **{**device_params, **catboost_params},
    )


def fit_model(
    model: Any,
    model_name: str,
    params: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[np.ndarray] = None,
) -> Any:
    """Fits the estimator, using early stopping on the validation set if supported."""
    if model_name == LOGISTIC:
        model.fit(X_train, y_train)
        return model

    if X_val is None:
        model.fit(X_train, y_train, verbose=0)
        return model

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=params.get("early_stopping_rounds"),
        verbose=0,
    )
    return model


def _effective_device_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """An explicit task_type/devices in the config wins over the detected device."""
    device_params = dict(_get_catboost_device_params())
    for key in ("task_type", "devices"):
        if key in params:
            device_params[key] = params[key]
    return device_params


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


def _prepare_catboost_params(
    params: Dict[str, Any], device_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Prepare CatBoost parameters by:
    1. Adding bootstrap_type if subsample is used (Bayesian bootstrap doesn't support subsample)
    2. Removing GPU-incompatible parameters when using GPU mode

    GPU limitations:
    - colsample_bylevel (RSM) is only supported in pairwise ranking modes, not classification
    """
    params_copy = params.copy()

    # Handle bootstrap type for subsample
    if "subsample" in params_copy and "bootstrap_type" not in params_copy:
        params_copy["bootstrap_type"] = "Bernoulli"

    # Remove GPU-incompatible parameters
    if device_params.get("task_type") == "GPU":
        gpu_incompatible_params = [
            "colsample_bylevel",
            "colsample_bynode",
            "colsample_bytree",
        ]
        for param in gpu_incompatible_params:
            if param in params_copy:
                logging.debug(f"Removing GPU-incompatible parameter: {param}")
                params_copy.pop(param)

    return params_copy
