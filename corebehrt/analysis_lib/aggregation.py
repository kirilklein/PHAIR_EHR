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
    """Performs aggregation for empirical variance analysis."""
    methods_df = df[df["method"].isin(["TMLE", "IPW"])].copy()
    if methods_df.empty:
        return pd.DataFrame()
    variance_per_outcome = (
        methods_df.groupby(["method", "ce", "cy", "i", "y", "outcome"])["effect"]
        .var(ddof=1)
        .reset_index()
    )
    variance_per_outcome.rename(columns={"effect": "variance"}, inplace=True)
    variance_per_outcome["avg_confounding"] = (
        variance_per_outcome["ce"] + variance_per_outcome["cy"]
    ) / 2
    final_variance_agg = (
        variance_per_outcome.groupby(["method", "avg_confounding", "i"])["variance"]
        .mean()
        .reset_index()
    )
    return final_variance_agg


# ============================================================================
# V2 Aggregation Functions: Aggregate over runs, grouped by method + outcome
# ============================================================================


def perform_bias_aggregation_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates bias over runs, keeping outcomes separate.
    Groups by [method, ce, cy, i, outcome] and computes mean and std of bias.

    Returns DataFrame with columns: [method, ce, cy, i, outcome, mean, std]
    """
    methods_df = df[df["method"].isin(["TMLE", "IPW", "TMLE_TH"])].copy()
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
    methods_df = df[df["method"].isin(["TMLE", "IPW", "TMLE_TH"])].copy()
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
    methods_df = df[df["method"].isin(["TMLE", "IPW", "TMLE_TH"])].copy()
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
    methods_df = df[df["method"].isin(["TMLE", "IPW", "TMLE_TH"])].copy()
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
    Computes empirical variance of effects over runs for each outcome.
    Groups by [method, ce, cy, i, outcome] and computes variance of effect estimates,
    then computes mean and std of this variance.

    Returns DataFrame with columns: [method, ce, cy, i, outcome, mean, std]
    """
    methods_df = df[df["method"].isin(["TMLE", "IPW", "TMLE_TH"])].copy()
    if methods_df.empty:
        return pd.DataFrame()

    # Compute variance of effects over runs for each (method, ce, cy, i, outcome) group
    variance_agg = (
        methods_df.groupby(["method", "ce", "cy", "i", "outcome"])["effect"]
        .var(ddof=1)
        .reset_index()
        .rename(columns={"effect": "mean"})
    )

    # For variance, std is not typically computed the same way, but we'll use 0 as placeholder
    # or we could compute the standard error of the variance estimate
    variance_agg["std"] = 0.0

    return variance_agg
