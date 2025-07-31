import pandas as pd
from scipy.special import expit, logit


def apply_logit_bias(column: pd.Series, delta: float) -> pd.Series:
    """Transform column to logit space, add bias, transform back to probability space."""
    if delta == 0:
        return column
    clipped_column = column.clip(1e-6, 1 - 1e-6)
    logits = logit(clipped_column)
    biased_logits = logits + delta
    return expit(biased_logits)


def apply_sharpening_bias(column: pd.Series, alpha: float) -> pd.Series:
    """Sharpen or flatten probabilities using an alpha parameter."""
    if alpha == 1.0:
        return column
    p = column.clip(1e-6, 1 - 1e-6)
    p_a = p**alpha
    one_minus_p_a = (1 - p) ** alpha
    denominator = p_a + one_minus_p_a
    return p_a / denominator.where(denominator != 0, 1)
