from typing import Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from corebehrt.constants.causal.data import (
    CF_PROBAS,
    PROBAS,
    TARGETS,
    CALIBRATION_COLLAPSE_THRESHOLD,
)
from corebehrt.constants.data import PID_COL
from corebehrt.functional.causal.stats import (
    compute_calibration_metrics,
    compute_probas_stats,
)
from corebehrt.functional.trainer.calibrate import train_calibrator
from corebehrt.main_causal.helper.utils import safe_assign_calibrated_probas


def collect_predictions(
    model: pl.LightningModule,
    dataset,
    device: torch.device,
    batch_size: int = 512,  # Reasonable batch size to avoid memory issues
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect predictions using batches but without sampling."""
    model.eval()

    # Create DataLoader without sampling or shuffling
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, sampler=None, drop_last=False
    )

    preds_list, targets_list = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy().squeeze()
            targets = y.cpu().numpy().astype(int)

            preds_list.append(probs)
            targets_list.append(targets)

    # Concatenate all batches
    preds = np.concatenate(preds_list)
    targets = np.concatenate(targets_list)

    print(f"Unique target values: {np.unique(targets)}")
    print(f"Target distribution: {np.bincount(targets)}")

    # Ensure both arrays are 1D
    assert preds.ndim == 1, f"Predictions should be 1D, got shape {preds.shape}"
    assert targets.ndim == 1, f"Targets should be 1D, got shape {targets.shape}"

    return preds, targets


def calibrate_predictions(
    model: pl.LightningModule,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_cf_loader: DataLoader,
    val_pids: list,
    epsilon: float = 1e-8,
    collapse_threshold: float = CALIBRATION_COLLAPSE_THRESHOLD,
) -> pd.DataFrame:
    device = next(model.parameters()).device

    # Get raw datasets
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    val_cf_dataset = val_cf_loader.dataset

    # Collect predictions directly from datasets
    train_preds, train_targets = collect_predictions(model, train_dataset, device)
    if len(np.unique(train_targets)) < 2:
        raise ValueError(
            f"Training data must contain both classes for calibration. Found classes: {np.unique(train_targets)}"
        )

    # Compute pre-calibration metrics on validation set
    val_preds, val_targets = collect_predictions(model, val_dataset, device)
    pre_cal_metrics = compute_calibration_metrics(val_targets, val_preds)
    pre_cal_probas_stats = compute_probas_stats(val_preds, val_targets)

    print("\nPre-calibration metrics:")
    print(f"Brier Score: {pre_cal_metrics['brier_score']:.4f}")
    print(f"ECE: {pre_cal_metrics['ece']:.4f}")
    print("\nPre-calibration probas stats:")
    print(pre_cal_probas_stats)

    # Train calibrator and get calibrated predictions
    calibrator = train_calibrator(train_preds, train_targets)
    calibrated_val = calibrator.predict(val_preds)
    calibrated_val = safe_assign_calibrated_probas(
        calibrated_val, val_preds, epsilon, collapse_threshold
    )

    post_cal_probas_stats = compute_probas_stats(calibrated_val, val_targets)
    print("\nCalibrated Probas Statistics:")
    print(post_cal_probas_stats)

    # Compute post-calibration metrics
    post_cal_metrics = compute_calibration_metrics(val_targets, calibrated_val)
    print("\nPost-calibration metrics:")
    print(f"Brier Score: {post_cal_metrics['brier_score']:.4f}")
    print(f"ECE: {post_cal_metrics['ece']:.4f}")

    # Get counterfactual predictions
    val_cf_preds, _ = collect_predictions(model, val_cf_dataset, device)
    calibrated_cf = calibrator.predict(val_cf_preds)
    calibrated_cf = safe_assign_calibrated_probas(
        calibrated_cf, val_cf_preds, epsilon, collapse_threshold
    )

    val_df = pd.DataFrame(
        {
            PID_COL: val_pids,
            PROBAS: calibrated_val,
            CF_PROBAS: calibrated_cf,
            TARGETS: val_targets,
        }
    )

    return val_df
