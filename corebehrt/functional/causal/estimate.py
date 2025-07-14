import numpy as np
import pandas as pd


def calculate_risk_difference(
    risk_exposed: float,
    risk_unexposed: float,
    n_exposed: int,
    n_unexposed: int,
) -> pd.DataFrame:
    """Calculate Risk Difference with 95% CI using delta method."""
    risk_difference = risk_exposed - risk_unexposed

    # Standard error using delta method
    variance_exposed = risk_exposed * (1 - risk_exposed) / n_exposed
    variance_unexposed = risk_unexposed * (1 - risk_unexposed) / n_unexposed
    se_rd = np.sqrt(variance_exposed + variance_unexposed)

    # 95% Confidence Interval
    ci_margin = 1.96 * se_rd
    ci_lower_rd = risk_difference - ci_margin
    ci_upper_rd = risk_difference + ci_margin

    return pd.DataFrame(
        {
            "method": ["RD"],
            "effect": [risk_difference],
            "std_err": [se_rd],
            "CI95_lower": [ci_lower_rd],
            "CI95_upper": [ci_upper_rd],
            "effect_1": [risk_exposed],
            "effct_0": [risk_unexposed],
            "n_beootstraps": [0],
        }
    )


def calculate_risk_ratio(
    risk_exposed: float,
    risk_unexposed: float,
    n_exposed: int,
    n_unexposed: int,
) -> pd.DataFrame:
    """Calculate Risk Ratio with 95% CI using log transformation."""
    # Handle division by zero
    if risk_unexposed == 0 or risk_exposed == 0:
        risk_ratio = np.inf if risk_exposed > 0 else np.nan
        se_log_rr = np.nan
        ci_lower_rr = np.nan
        ci_upper_rr = np.nan
    else:
        risk_ratio = risk_exposed / risk_unexposed

        # Standard error on log scale
        se_log_rr = np.sqrt(
            (1 - risk_exposed) / (risk_exposed * n_exposed)
            + (1 - risk_unexposed) / (risk_unexposed * n_unexposed)
        )

        # 95% CI on log scale, then exponentiate
        log_rr = np.log(risk_ratio)
        ci_margin_log = 1.96 * se_log_rr
        ci_lower_rr = np.exp(log_rr - ci_margin_log)
        ci_upper_rr = np.exp(log_rr + ci_margin_log)

    return pd.DataFrame(
        {
            "method": ["RR"],
            "effect": [risk_ratio],
            "std_err": [se_log_rr],
            "CI95_lower": [ci_lower_rr],
            "CI95_upper": [ci_upper_rr],
            "effect_1": [risk_exposed],
            "effect_0": [risk_unexposed],
            "n_bootstraps": [0],
        }
    )
