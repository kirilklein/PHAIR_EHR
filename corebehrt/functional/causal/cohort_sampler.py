"""
Cohort sampling utilities for resampling experiments.

This module provides functionality to randomly sample a subset of patients
from a full cohort for use in resampling-based causal inference experiments.
"""

import torch
import numpy as np


def sample_cohort(
    full_pids: torch.Tensor, sample_fraction: float, seed: int
) -> torch.Tensor:
    """
    Randomly sample a fraction of patient IDs without replacement.

    Args:
        full_pids: Tensor of all patient IDs in the full cohort
        sample_fraction: Fraction of patients to sample (0 < fraction <= 1)
        seed: Random seed for reproducibility

    Returns:
        Tensor of sampled patient IDs

    Raises:
        ValueError: If sample_fraction is not in valid range

    Example:
        >>> full_pids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> sampled = sample_cohort(full_pids, sample_fraction=0.5, seed=42)
        >>> len(sampled) == 5
        True
    """
    if not 0 < sample_fraction <= 1:
        raise ValueError(f"sample_fraction must be in (0, 1], got {sample_fraction}")

    # Set random seed for reproducibility
    rng = np.random.RandomState(seed)

    # Calculate number of patients to sample
    n_total = len(full_pids)
    n_sample = int(n_total * sample_fraction)

    if n_sample == 0:
        raise ValueError(
            f"sample_fraction {sample_fraction} results in 0 samples from {n_total} patients"
        )

    # Sample indices without replacement
    sampled_indices = rng.choice(n_total, size=n_sample, replace=False)

    # Return sampled PIDs
    return full_pids[sampled_indices]
