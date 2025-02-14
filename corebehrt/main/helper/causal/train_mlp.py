import warnings
from typing import Dict, List

import torch

from corebehrt.constants.data import VAL_KEY


def combine_encodings_and_exposures(
    encodings: torch.Tensor, exposures: torch.Tensor
) -> torch.Tensor:
    """Combines input features with exposure values centered at 0.5.

    Args:
        x: Input feature matrix
        exposures: Exposure values to combine with features
    Returns:
        Combined matrix of features and centered exposures
    """
    return torch.cat([encodings, exposures.unsqueeze(1) - 0.5], dim=1)


def check_val_fold_pids(folds: List[Dict], pids: List[str]):
    """Validates that validation fold PIDs are a subset of provided PIDs.

    Args:
        folds: List of dictionaries containing validation fold PIDs
        pids: List of allowed patient IDs
    """
    pids = set(pids)
    for fold in folds:
        if not set(fold[VAL_KEY]).issubset(pids):
            warnings.warn(
                "Some validation fold PIDs are not present in the provided PIDs list. Perhaps Exclusion was performed during fine-tuning."
            )
