"""
Cohort sampling utilities for resampling experiments.

This module provides functionality to randomly sample a subset of patients
from a full cohort for use in resampling-based causal inference experiments.
"""

import torch
import numpy as np


def sample_cohort(
    full_pids: torch.Tensor,
    sample_fraction: float = None,
    sample_size: int = None,
    seed: int = 42,
    method: str = "resample",
) -> torch.Tensor:
    """
    Randomly sample patient IDs by fraction or absolute size.

    Args:
        full_pids: Tensor of all patient IDs in the full cohort
        sample_fraction: Fraction of patients to sample (0 < fraction <= 1).
                        Ignored if sample_size is provided.
        sample_size: Absolute number of patients to sample. Takes precedence over sample_fraction.
        seed: Random seed for reproducibility
        method: Sampling method - "resample" (without replacement) or "bootstrap" (with replacement)

    Returns:
        Tensor of sampled patient IDs

    Raises:
        ValueError: If neither or both parameters are provided, or values are invalid
    """
    if method not in ("resample", "bootstrap"):
        raise ValueError(f"method must be 'resample' or 'bootstrap', got '{method}'")

    # Validate inputs
    if sample_size is None and sample_fraction is None:
        raise ValueError("Either 'sample_size' or 'sample_fraction' must be provided")

    if sample_size is not None and sample_fraction is not None:
        raise ValueError("Cannot specify both 'sample_size' and 'sample_fraction'")

    # Set random seed for reproducibility
    rng = np.random.RandomState(seed)

    # Calculate number of patients to sample
    n_total = len(full_pids)
    replace = method == "bootstrap"

    if sample_size is not None:
        # Use absolute size
        if sample_size <= 0:
            raise ValueError(f"sample_size must be positive, got {sample_size}")
        if not replace and sample_size > n_total:
            raise ValueError(
                f"sample_size {sample_size} exceeds available patients {n_total}"
            )
        n_sample = sample_size
    else:
        # Use fraction
        if not 0 < sample_fraction <= 1:
            raise ValueError(
                f"sample_fraction must be in (0, 1], got {sample_fraction}"
            )
        n_sample = int(n_total * sample_fraction)
        if n_sample == 0:
            raise ValueError(
                f"sample_fraction {sample_fraction} results in 0 samples from {n_total} patients"
            )

    # Sample indices
    sampled_indices = rng.choice(n_total, size=n_sample, replace=replace)

    # Return sampled PIDs
    return full_pids[sampled_indices]
