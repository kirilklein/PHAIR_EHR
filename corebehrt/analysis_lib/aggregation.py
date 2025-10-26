import pandas as pd


def perform_bias_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    """Performs two-step aggregation for absolute bias analysis."""
    methods_df = df[df["method"].isin(["TMLE", "IPW"])].copy()
    if methods_df.empty:
        return pd.DataFrame()
    mean_bias_per_run = (
        methods_df.groupby(["run_id", "method", "ce", "cy", "i", "y"])["bias"]
        .mean()
        .reset_index()
    )
    mean_bias_per_run["avg_confounding"] = (
        mean_bias_per_run["ce"] + mean_bias_per_run["cy"]
    ) / 2
    final_agg = (
        mean_bias_per_run.groupby(["method", "avg_confounding", "i"])["bias"]
        .agg(["mean", "std"])
        .reset_index()
    )
    return final_agg


def perform_relative_bias_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    """Performs two-step aggregation for relative bias, excluding true_effect=0 cases."""
    methods_df = df[df["method"].isin(["TMLE", "IPW"])].copy()
    methods_df.dropna(subset=["relative_bias"], inplace=True)
    if methods_df.empty:
        return pd.DataFrame()
    mean_rb_per_run = (
        methods_df.groupby(["run_id", "method", "ce", "cy", "i", "y"])["relative_bias"]
        .mean()
        .reset_index()
    )
    mean_rb_per_run["avg_confounding"] = (
        mean_rb_per_run["ce"] + mean_rb_per_run["cy"]
    ) / 2
    final_agg = (
        mean_rb_per_run.groupby(["method", "avg_confounding", "i"])["relative_bias"]
        .agg(["mean", "std"])
        .reset_index()
    )
    return final_agg


def perform_zscore_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    """Performs two-step aggregation for Z-score (Standardized Bias)."""
    methods_df = df[df["method"].isin(["TMLE", "IPW"])].copy()
    methods_df.dropna(subset=["z_score"], inplace=True)
    if methods_df.empty:
        return pd.DataFrame()
    mean_z_per_run = (
        methods_df.groupby(["run_id", "method", "ce", "cy", "i", "y"])["z_score"]
        .mean()
        .reset_index()
    )
    mean_z_per_run["avg_confounding"] = (
        mean_z_per_run["ce"] + mean_z_per_run["cy"]
    ) / 2
    final_agg = (
        mean_z_per_run.groupby(["method", "avg_confounding", "i"])["z_score"]
        .agg(["mean", "std"])
        .reset_index()
    )
    return final_agg


def perform_coverage_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    """Performs single-step aggregation for coverage analysis."""
    methods_df = df[df["method"].isin(["TMLE", "IPW"])].copy()
    if methods_df.empty:
        return pd.DataFrame()
    methods_df["avg_confounding"] = (methods_df["ce"] + methods_df["cy"]) / 2
    coverage_agg = (
        methods_df.groupby(["method", "avg_confounding", "i"])["covered"]
        .mean()
        .reset_index()
    )
    return coverage_agg


def perform_variance_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    """Performs aggregation for empirical standard deviation analysis."""
    methods_df = df[df["method"].isin(["TMLE", "IPW"])].copy()
    if methods_df.empty:
        return pd.DataFrame()
    std_per_outcome = (
        methods_df.groupby(["method", "ce", "cy", "i", "y", "outcome"])["effect"]
        .std(ddof=1)
        .reset_index()
    )
    std_per_outcome.rename(columns={"effect": "std_dev"}, inplace=True)
    std_per_outcome["avg_confounding"] = (
        std_per_outcome["ce"] + std_per_outcome["cy"]
    ) / 2
    final_std_agg = (
        std_per_outcome.groupby(["method", "avg_confounding", "i"])["std_dev"]
        .mean()
        .reset_index()
    )
    return final_std_agg


# ============================================================================
# V2 Aggregation Functions: Aggregate over runs, grouped by method + outcome
# ============================================================================


def perform_bias_aggregation_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates bias over runs, keeping outcomes separate.
    Groups by [method, ce, cy, i, outcome] and computes mean and std of bias.

    Returns DataFrame with columns: [method, ce, cy, i, outcome, mean, std]
    """
    # Accept any method that ends with known method names (supports baseline_ and bert_ prefixes)
    methods_df = df[
        df["method"].str.contains(r"(?:TMLE|IPW|TMLE_TH)$", regex=True, na=False)
    ].copy()
    if methods_df.empty:
        return pd.DataFrame()

    agg = (
        methods_df.groupby(["method", "ce", "cy", "i", "outcome"])["bias"]
        .agg(["mean", "std"])
        .reset_index()
    )
    return agg


def perform_relative_bias_aggregation_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates relative bias over runs, keeping outcomes separate.
    Groups by [method, ce, cy, i, outcome] and computes mean and std of relative_bias.
    Excludes cases where relative_bias is NaN.

    Returns DataFrame with columns: [method, ce, cy, i, outcome, mean, std]
    """
    # Accept any method that ends with known method names (supports baseline_ and bert_ prefixes)
    methods_df = df[
        df["method"].str.contains(r"(?:TMLE|IPW|TMLE_TH)$", regex=True, na=False)
    ].copy()
    methods_df.dropna(subset=["relative_bias"], inplace=True)
    if methods_df.empty:
        return pd.DataFrame()

    agg = (
        methods_df.groupby(["method", "ce", "cy", "i", "outcome"])["relative_bias"]
        .agg(["mean", "std"])
        .reset_index()
    )
    return agg


def perform_zscore_aggregation_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates z-score over runs, keeping outcomes separate.
    Groups by [method, ce, cy, i, outcome] and computes mean and std of z_score.
    Excludes cases where z_score is NaN.

    Returns DataFrame with columns: [method, ce, cy, i, outcome, mean, std]
    """
    # Accept any method that ends with known method names (supports baseline_ and bert_ prefixes)
    methods_df = df[
        df["method"].str.contains(r"(?:TMLE|IPW|TMLE_TH)$", regex=True, na=False)
    ].copy()
    methods_df.dropna(subset=["z_score"], inplace=True)
    if methods_df.empty:
        return pd.DataFrame()

    agg = (
        methods_df.groupby(["method", "ce", "cy", "i", "outcome"])["z_score"]
        .agg(["mean", "std"])
        .reset_index()
    )
    return agg


def perform_coverage_aggregation_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates coverage over runs, keeping outcomes separate.
    Groups by [method, ce, cy, i, outcome] and computes mean of covered.
    Note: No std for coverage, as it's a proportion.

    Returns DataFrame with columns: [method, ce, cy, i, outcome, mean]
    """
    # Accept any method that ends with known method names (supports baseline_ and bert_ prefixes)
    methods_df = df[
        df["method"].str.contains(r"(?:TMLE|IPW|TMLE_TH)$", regex=True, na=False)
    ].copy()
    if methods_df.empty:
        return pd.DataFrame()

    agg = (
        methods_df.groupby(["method", "ce", "cy", "i", "outcome"])["covered"]
        .mean()
        .reset_index()
        .rename(columns={"covered": "mean"})
    )
    return agg


def perform_variance_aggregation_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes empirical standard deviation of effects over runs for each outcome.
    Groups by [method, ce, cy, i, outcome] and computes standard deviation of effect estimates.

    Returns DataFrame with columns: [method, ce, cy, i, outcome, mean, std]
    where 'mean' contains the standard deviation and 'std' is set to 0.
    """
    # Accept any method that ends with known method names (supports baseline_ and bert_ prefixes)
    methods_df = df[
        df["method"].str.contains(r"(?:TMLE|IPW|TMLE_TH)$", regex=True, na=False)
    ].copy()
    if methods_df.empty:
        return pd.DataFrame()

    # Compute standard deviation of effects over runs for each (method, ce, cy, i, outcome) group
    std_agg = (
        methods_df.groupby(["method", "ce", "cy", "i", "outcome"])["effect"]
        .std(ddof=1)
        .reset_index()
        .rename(columns={"effect": "mean"})
    )

    # For standard deviation, std is not typically computed the same way, but we'll use 0 as placeholder
    # or we could compute the standard error of the standard deviation estimate
    std_agg["std"] = 0.0

    return std_agg


def perform_se_calibration_aggregation_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Standard Error Calibration ratio: empirical SE / mean estimated SE.

    A well-calibrated estimator should have a ratio near 1.0:
    - Ratio < 1.0: Standard errors are overestimated (too conservative)
    - Ratio > 1.0: Standard errors are underestimated (too optimistic)

    Groups by [method, ce, cy, i, outcome] and computes:
    - Empirical SE: std(effect) across runs
    - Mean estimated SE: mean(std_err) across runs
    - Ratio: empirical SE / mean estimated SE

    Returns DataFrame with columns: [method, ce, cy, i, outcome, mean, std]
    where 'mean' is the calibration ratio and 'std' is set to 0.
    """
    # Accept any method that ends with known method names (supports baseline_ and bert_ prefixes)
    methods_df = df[
        df["method"].str.contains(r"(?:TMLE|IPW|TMLE_TH)$", regex=True, na=False)
    ].copy()
    if methods_df.empty:
        return pd.DataFrame()

    # Group by method, parameters, and outcome
    grouped = methods_df.groupby(["method", "ce", "cy", "i", "outcome"])

    # Compute empirical SE (standard deviation of effects across runs)
    empirical_se = grouped["effect"].std(ddof=1)

    # Compute mean of the estimated standard errors
    mean_estimated_se = grouped["std_err"].mean()

    # Compute calibration ratio
    calibration_ratio = empirical_se / mean_estimated_se

    # Create result dataframe
    result = pd.DataFrame(
        {
            "mean": calibration_ratio,
            "std": 0.0,  # No std for a ratio metric
        }
    ).reset_index()

    return result
