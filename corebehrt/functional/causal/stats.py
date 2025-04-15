import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Tuple
import pytorch_lightning as pl


def compute_treatment_outcome_table(
    df: pd.DataFrame, exposure_col: str, outcome_col: str
) -> pd.DataFrame:
    """
    Compute a 2x2 contingency table for binary exposure and outcome.

    Parameters:
    df (pd.DataFrame): The dataframe containing the data
    exposure_col (str): The name of the column containing the binary treatment indicator
    outcome_col (str): The name of the column containing the binary outcome indicator

    Returns:
    pd.DataFrame: A 2x2 contingency table with rows as treatment (0/1) and columns as outcome (0/1)
    """
    table = pd.crosstab(
        df[exposure_col], df[outcome_col], margins=True, margins_name="Total"
    )
    table.index = ["Untreated", "Treated", "Total"]
    table.columns = ["No Outcome", "Outcome", "Total"]
    return table


def compute_calibration_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10
) -> dict:
    """
    Compute calibration metrics including Brier score and ECE.

    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        n_bins: Number of bins for ECE calculation
    """
    # Brier Score
    brier = np.mean((y_true - y_pred) ** 2)

    # Expected Calibration Error
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0
    bin_stats = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find predictions in this bin
        in_bin = np.logical_and(y_pred > bin_lower, y_pred <= bin_upper)
        if any(in_bin):
            bin_pred = y_pred[in_bin]
            bin_true = y_true[in_bin]
            bin_conf = np.mean(bin_pred)
            bin_acc = np.mean(bin_true)
            bin_count = np.sum(in_bin)

            ece += (bin_count / len(y_pred)) * abs(bin_acc - bin_conf)
            bin_stats.append(
                {
                    "bin_lower": bin_lower,
                    "bin_upper": bin_upper,
                    "samples": bin_count,
                    "accuracy": bin_acc,
                    "confidence": bin_conf,
                }
            )

    return {"brier_score": brier, "ece": ece, "bin_stats": bin_stats}


def collect_predictions_without_sampler(
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
