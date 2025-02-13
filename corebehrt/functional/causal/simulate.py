import numpy as np
from typing import Tuple
from scipy.stats import bernoulli
from scipy.special import expit as sigmoid


def simulate_outcome_from_encodings(
    encodings: np.ndarray,
    exposure: np.ndarray,
    exposure_coef: float,
    enc_coef: float,
    intercept: float,
    enc_sparsity: float = 0.7,
    enc_scale: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
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
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - Binary outcomes array of shape (n_samples,)
            - Outcome probabilities array of shape (n_samples,)
    """

    # Generate sparse feature coefficients
    n_enc = encodings.shape[1]
    rng = np.random.RandomState(42)  # Set fixed random state for reproducibility
    W_enc = rng.normal(
        0, enc_scale, size=n_enc
    )  # Small coefficients from normal distribution
    zero_mask_enc = rng.random(n_enc) < enc_sparsity

    W_enc[zero_mask_enc] = 0  # Set random subset of coefficients to zero

    # Calculate probability using feature combination
    s_enc = encodings @ W_enc
    probability = sigmoid(exposure_coef * exposure + enc_coef * s_enc + intercept)
    binary_outcome = bernoulli.rvs(probability)
    return binary_outcome, probability
