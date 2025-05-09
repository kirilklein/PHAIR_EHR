import numpy as np


def effective_sample_size(w: np.ndarray):
    """
    Compute the effective sample size as dervied in
    Given sample weights w, the effective sample size is defined as
    N_e = (sum(w)^2) / (sum(w^2))

    Shook‐Sa, Bonnie E., and Michael G. Hudgens.
    "Power and sample size for observational studies of point exposure effects." Biometrics 78.1 (2022): 388-398.
    """
    return np.sum(w) ** 2 / np.sum(w**2)
