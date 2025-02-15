from typing import Tuple

import torch
import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from corebehrt.constants.data import PID_COL
from corebehrt.constants.causal import PROBAS, TARGETS
from corebehrt.main.helper.causal.calibrate import (
    train_isotonic_regression,
    calibrate_probas,
)


def collect_predictions(
    model: pl.LightningModule, loader: DataLoader, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collects predictions and targets from a given DataLoader.

    Args:
        model: The trained model.
        loader: DataLoader to iterate over.
        device: Device to move data to.

    Returns:
        A tuple of (predictions, targets) as Tensors.
    """
    model.eval()
    preds_list, targets_list = [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            preds_list.append(probs.cpu())
            targets_list.append(y.cpu())
    preds = torch.cat(preds_list, dim=0)
    targets = torch.cat(targets_list, dim=0)
    return preds, targets


def train_calibrator_from_data(
    train_preds: torch.Tensor, train_targets: torch.Tensor
) -> IsotonicRegression:
    """
    Trains an isotonic regression calibrator using training predictions and targets.

    Args:
        train_preds: Raw predictions from the training set.
        train_targets: Ground truth targets for the training set.

    Returns:
        A trained IsotonicRegression calibrator.
    """
    train_df = pd.DataFrame(
        {
            PROBAS: train_preds.numpy().ravel(),
            TARGETS: train_targets.numpy().ravel(),
        }
    )
    calibrator = train_isotonic_regression(train_df)
    return calibrator


def apply_calibration_to_predictions(
    calibrator: IsotonicRegression, raw_preds: torch.Tensor, epsilon: float = 1e-8
) -> np.ndarray:
    """
    Applies the trained calibrator to raw predictions.

    Args:
        calibrator: A trained IsotonicRegression calibrator.
        raw_preds: Raw prediction tensor.
        epsilon: Small value for clipping calibrated probabilities.

    Returns:
        Calibrated predictions as a numpy array.
    """
    raw_preds_series = pd.Series(raw_preds.numpy().ravel())
    calibrated = calibrate_probas(calibrator, raw_preds_series, epsilon)
    return calibrated


def calibrate_predictions(
    model: pl.LightningModule,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_pids: torch.Tensor,
    epsilon: float = 1e-8,
) -> pd.DataFrame:
    """
    Calibrates validation predictions using an isotonic regression calibrator
    trained on training predictions, and saves the results to CSV.

    Args:
        model: The trained LightningModule.
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        fold_idx: Current fold index.
        fold_folder: Folder path to save CSV file.
        epsilon: Clipping epsilon for calibration.

    Returns:
        A DataFrame containing raw predictions, targets, calibrated predictions, and fold index.
    """
    device = next(model.parameters()).device

    # Collect training predictions and targets
    train_preds, train_targets = collect_predictions(model, train_loader, device)
    calibrator = train_calibrator_from_data(train_preds, train_targets)

    # Collect validation predictions and targets
    val_preds, val_targets = collect_predictions(model, val_loader, device)
    calibrated_val = apply_calibration_to_predictions(
        calibrator, val_preds, epsilon=epsilon
    )

    # Create a DataFrame for validation results
    val_df = pd.DataFrame(
        {
            PID_COL: val_pids,
            PROBAS: calibrated_val,
            TARGETS: val_targets,
        }
    )

    return val_df
