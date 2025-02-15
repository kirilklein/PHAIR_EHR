from typing import Tuple

import torch


def simulate_outcome_from_encodings(
    encodings: torch.Tensor,
    exposure: torch.Tensor,
    exposure_coef: float,
    enc_coef: float,
    intercept: float,
    enc_sparsity: float = 0.7,
    enc_scale: float = 0.1,
    random_state: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simulate binary outcomes using patient encodings and exposure status with sparse feature coefficients.

    Args:
        encodings: Patient encodings matrix of shape (n_samples, n_features)
        exposure: Binary treatment/exposure status array of shape (n_samples,)
        exposure_coef: Coefficient controlling the effect of exposure on outcome
        enc_coef: Global scaling coefficient for the embedding features
        intercept: Baseline probability (on logit scale) for the outcome
        enc_sparsity: Proportion (0 to 1) of embedding features that will be zeroed out
        enc_scale: Standard deviation of the normal distribution used to generate feature weights

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - Binary outcomes tensor of shape (n_samples,)
            - Outcome probabilities tensor of shape (n_samples,)
    """
    # Set random seed for reproducibility
    torch.manual_seed(random_state)

    # Generate sparse feature coefficients
    n_enc = encodings.shape[1]
    W_enc = torch.normal(0, enc_scale, size=(n_enc,))
    zero_mask_enc = torch.rand(n_enc) < enc_sparsity
    W_enc[zero_mask_enc] = 0  # Set random subset of coefficients to zero

    # Calculate probability using feature combination
    s_enc = encodings @ W_enc
    probability = torch.sigmoid(exposure_coef * exposure + enc_coef * s_enc + intercept)
    binary_outcome = torch.bernoulli(probability)

    return binary_outcome, probability


def combine_counterfactuals(exposure, exposed_values, control_values):
    """Combines counterfactual values based on exposure status.

    For each individual, returns the opposite of their actual exposure:
    - If exposed (exposure=1), returns their control value
    - If not exposed (exposure=0), returns their exposed value

    Args:
        exposure (numpy.ndarray): Binary array indicating exposure status (1=exposed, 0=control)
        exposed_values (numpy.ndarray): Values under exposure condition
        control_values (numpy.ndarray): Values under control condition

    Returns:
        numpy.ndarray: Combined array where each element is the counterfactual value
            based on the opposite of the actual exposure status
    """
    return torch.where(exposure == 1, control_values, exposed_values)
