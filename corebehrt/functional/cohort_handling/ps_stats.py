import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from typing import Dict, Tuple


def common_support_interval(ps: pd.Series, exposure: pd.Series) -> Dict[str, float]:
    """
    Compute the common-support interval for propensity scores and the fraction outside it.
    """
    if not exposure.isin([0, 1]).all():
        raise ValueError("Exposure must be binary (0/1).")
    p0 = ps[exposure == 0]
    p1 = ps[exposure == 1]
    min0, max0 = p0.min(), p0.max()
    min1, max1 = p1.min(), p1.max()
    low = max(min0, min1)
    high = min(max0, max1)
    pct_outside_control = ((p0 < low) | (p0 > high)).mean()
    pct_outside_treated = ((p1 < low) | (p1 > high)).mean()
    return {
        "cs_low": low,
        "cs_high": high,
        "pct_outside_control": pct_outside_control,
        "pct_outside_treated": pct_outside_treated,
    }


def overlap_coefficient(ps: pd.Series, exposure: pd.Series, n_bins: int = 100) -> float:
    """
    Compute the overlap coefficient (area of overlap) between treated and control PS distributions.
    """
    p0 = ps[exposure == 0]
    p1 = ps[exposure == 1]
    bins = np.linspace(0, 1, n_bins + 1)
    f0, _ = np.histogram(p0, bins=bins, density=True)
    f1, _ = np.histogram(p1, bins=bins, density=True)
    bin_width = bins[1] - bins[0]
    return np.sum(np.minimum(f0, f1)) * bin_width


def ks_statistic(ps: pd.Series, exposure: pd.Series) -> Tuple[float, float]:
    """
    Compute the Kolmogorov–Smirnov statistic and p-value for PS distributions.
    """
    p0 = ps[exposure == 0]
    p1 = ps[exposure == 1]
    stat, pval = ks_2samp(p0, p1)
    return stat, pval


def standardized_mean_difference(ps: pd.Series, exposure: pd.Series) -> float:
    """
    Compute the standardized difference in means of the propensity score.
    """
    p0 = ps[exposure == 0]
    p1 = ps[exposure == 1]
    m0, m1 = p0.mean(), p1.mean()
    v0, v1 = p0.var(ddof=1), p1.var(ddof=1)
    return (m1 - m0) / np.sqrt((v0 + v1) / 2)
