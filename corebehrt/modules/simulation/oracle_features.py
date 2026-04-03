"""Extract hand-crafted oracle features from pre-index patient histories."""

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

from corebehrt.constants.data import BIRTH_CODE, CONCEPT_COL, PID_COL, TIMESTAMP_COL
from corebehrt.modules.simulation.config_semisynthetic import FeatureConfig

logger = logging.getLogger("oracle_features")


def extract_oracle_features(
    history_df: pd.DataFrame,
    pids: np.ndarray,
    index_dates: pd.Series,
    feature_config: FeatureConfig,
) -> Tuple[pd.DataFrame, List[str]]:
    """Extract oracle features from pre-index patient histories.

    Args:
        history_df: MEDS DataFrame already filtered to pre-index events
        pids: array of patient IDs to extract features for
        index_dates: per-patient index dates (Series: PID -> Timestamp)
        feature_config: feature extraction configuration

    Returns:
        features_df: DataFrame with PID_COL as index, one column per feature
        feature_names: list of feature names present in the output
    """
    prefixes = feature_config.code_prefixes

    features = {}
    # Baseline risk features
    features["recent_event_count"] = _compute_recent_event_count(
        history_df, pids, index_dates, feature_config.recent_window_days
    )
    features["disease_burden"] = _compute_disease_burden(
        history_df, pids, index_dates, prefixes.diagnosis, feature_config.lookback_days
    )
    features["medication_count"] = _compute_medication_count(
        history_df, pids, index_dates, prefixes.medication, feature_config.lookback_days
    )
    features["utilization_intensity"] = _compute_utilization_intensity(
        history_df, pids, index_dates, feature_config.lookback_days
    )
    features["age"] = _compute_age(history_df, pids, index_dates)
    features["chronic_disease_count"] = _compute_chronic_disease_count(
        history_df, pids, prefixes.diagnosis
    )
    features["code_diversity"] = _compute_code_diversity(history_df, pids)

    # Longitudinal features
    features["event_recency"] = _compute_event_recency(history_df, pids, index_dates)
    features["recent_burst_ratio"] = _compute_recent_burst_ratio(
        history_df,
        pids,
        index_dates,
        feature_config.burst_window_days,
        feature_config.lookback_days,
    )
    features["sequence_motif_count"] = _compute_sequence_motif_count(
        history_df,
        pids,
        index_dates,
        prefixes.diagnosis,
        prefixes.medication,
        feature_config.motif_window_days,
    )

    features_df = pd.DataFrame(features, index=pids)
    features_df.index.name = PID_COL

    if feature_config.standardize:
        features_df = _standardize(features_df)

    return features_df, list(features_df.columns)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _filter_by_prefix_and_window(history_df, index_dates, prefix, window_days):
    """Filter events matching code prefix within lookback window per patient."""
    mask = history_df[CONCEPT_COL].str.startswith(prefix)
    filtered = history_df[mask].copy()
    if filtered.empty:
        logger.warning("No codes found with prefix '%s'", prefix)
        return filtered
    if window_days is not None:
        cutoff = index_dates.reindex(filtered[PID_COL]).values - pd.Timedelta(
            days=window_days
        )
        filtered = filtered[filtered[TIMESTAMP_COL].values >= cutoff]
    return filtered


def _count_per_patient(filtered_df, pids, count_col=None, unique=False):
    """Count (total or unique) events per patient, filling missing with 0."""
    if filtered_df.empty:
        return pd.Series(0, index=pids, dtype=int)
    grouped = filtered_df.groupby(PID_COL)
    if unique:
        col = count_col if count_col else CONCEPT_COL
        counts = grouped[col].nunique()
    else:
        counts = grouped.size()
    return counts.reindex(pids, fill_value=0)


# ---------------------------------------------------------------------------
# Baseline risk features (r_B)
# ---------------------------------------------------------------------------


def _compute_recent_event_count(history_df, pids, index_dates, recent_window_days):
    filtered = _filter_by_prefix_and_window(
        history_df, index_dates, "", recent_window_days
    )
    return _count_per_patient(filtered, pids)


def _compute_disease_burden(history_df, pids, index_dates, diag_prefix, lookback_days):
    filtered = _filter_by_prefix_and_window(
        history_df, index_dates, diag_prefix, lookback_days
    )
    return _count_per_patient(filtered, pids, unique=True)


def _compute_medication_count(history_df, pids, index_dates, med_prefix, lookback_days):
    filtered = _filter_by_prefix_and_window(
        history_df, index_dates, med_prefix, lookback_days
    )
    return _count_per_patient(filtered, pids, unique=True)


def _compute_utilization_intensity(history_df, pids, index_dates, lookback_days):
    filtered = _filter_by_prefix_and_window(history_df, index_dates, "", lookback_days)
    return _count_per_patient(filtered, pids)


def _compute_age(history_df, pids, index_dates):
    dob_events = history_df[history_df[CONCEPT_COL] == BIRTH_CODE]
    dob_per_patient = dob_events.groupby(PID_COL)[TIMESTAMP_COL].first()
    age_series = pd.Series(index=pids, dtype=float)
    for pid in pids:
        if pid in dob_per_patient.index:
            dob = dob_per_patient[pid]
            idx_date = index_dates[pid]
            age_series[pid] = (idx_date - dob).days / 365.25
    mean_age = age_series.mean()
    if np.isnan(mean_age):
        mean_age = 65.0
        logger.warning("No DOB events found, using default age %.0f", mean_age)
    age_series = age_series.fillna(mean_age)
    return age_series


def _compute_chronic_disease_count(history_df, pids, diag_prefix):
    diag_events = history_df[history_df[CONCEPT_COL].str.startswith(diag_prefix)]
    if diag_events.empty:
        logger.warning("No diagnosis codes found for chronic disease count")
        return pd.Series(0, index=pids, dtype=int)
    groups = diag_events.copy()
    groups["_diag_group"] = groups[CONCEPT_COL].str[:5]
    counts = groups.groupby(PID_COL)["_diag_group"].nunique()
    return counts.reindex(pids, fill_value=0)


def _compute_code_diversity(history_df, pids):
    return _count_per_patient(history_df, pids, unique=True)


# ---------------------------------------------------------------------------
# Longitudinal features (r_L)
# ---------------------------------------------------------------------------


def _compute_event_recency(history_df, pids, index_dates):
    last_event = history_df.groupby(PID_COL)[TIMESTAMP_COL].max()
    recency = pd.Series(index=pids, dtype=float)
    for pid in pids:
        if pid in last_event.index:
            recency[pid] = (index_dates[pid] - last_event[pid]).days
        else:
            recency[pid] = np.nan
    mean_recency = recency.mean()
    recency = recency.fillna(mean_recency)
    return recency


def _compute_recent_burst_ratio(
    history_df, pids, index_dates, burst_window_days, lookback_days
):
    burst_filtered = _filter_by_prefix_and_window(
        history_df, index_dates, "", burst_window_days
    )
    lookback_filtered = _filter_by_prefix_and_window(
        history_df, index_dates, "", lookback_days
    )
    burst_counts = _count_per_patient(burst_filtered, pids)
    lookback_counts = _count_per_patient(lookback_filtered, pids)
    return burst_counts / (lookback_counts + 1)


def _compute_sequence_motif_count(
    history_df, pids, _index_dates, diag_prefix, med_prefix, motif_window_days
):
    diag_events = history_df[history_df[CONCEPT_COL].str.startswith(diag_prefix)]
    med_events = history_df[history_df[CONCEPT_COL].str.startswith(med_prefix)]
    if diag_events.empty or med_events.empty:
        logger.warning("Missing diagnosis or medication codes for motif counting")
        return pd.Series(0, index=pids, dtype=int)

    motif_counts = {}
    window = pd.Timedelta(days=motif_window_days)
    for pid in pids:
        pid_diags = diag_events[diag_events[PID_COL] == pid][TIMESTAMP_COL].values
        pid_meds = med_events[med_events[PID_COL] == pid][TIMESTAMP_COL].values
        count = 0
        for diag_time in pid_diags:
            gaps = pid_meds - diag_time
            count += int(np.sum((gaps >= np.timedelta64(0)) & (gaps <= window)))
        motif_counts[pid] = count

    return pd.Series(motif_counts).reindex(pids, fill_value=0)


# ---------------------------------------------------------------------------
# Standardization
# ---------------------------------------------------------------------------


def _standardize(features_df):
    means = features_df.mean()
    stds = features_df.std(ddof=0).replace(0, 1).fillna(1)
    return (features_df - means) / stds
