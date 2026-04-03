import unittest

import numpy as np
import pandas as pd

from corebehrt.constants.data import CONCEPT_COL, PID_COL, TIMESTAMP_COL
from corebehrt.modules.simulation.config_semisynthetic import FeatureConfig
from corebehrt.modules.simulation.oracle_features import (
    BASELINE_FEATURES,
    LONGITUDINAL_FEATURES,
    extract_oracle_features,
)


def _make_test_df(records):
    """Build a MEDS DataFrame from (subject_id, time_str, code) tuples."""
    rows = []
    for pid, time_str, code in records:
        rows.append(
            {PID_COL: pid, TIMESTAMP_COL: pd.Timestamp(time_str), CONCEPT_COL: code}
        )
    return pd.DataFrame(rows)


def _default_feature_config(**overrides):
    kwargs = dict(standardize=False)
    kwargs.update(overrides)
    return FeatureConfig(**kwargs)


class TestExtractOracleFeaturesBasic(unittest.TestCase):
    def test_extract_oracle_features_basic(self):
        records = [
            (1, "1990-01-01", "DOB"),
            (1, "2020-06-01", "D/11111"),
            (1, "2020-09-01", "M/22222"),
            (2, "1985-03-15", "DOB"),
            (2, "2020-05-01", "D/33333"),
            (3, "1975-07-20", "DOB"),
            (3, "2020-08-01", "P/44444"),
            (4, "2000-01-01", "DOB"),
            (4, "2020-07-01", "D/55555"),
            (4, "2020-08-15", "M/66666"),
            (5, "1995-05-05", "DOB"),
            (5, "2020-04-01", "D/77777"),
        ]
        history_df = _make_test_df(records)
        pids = np.array([1, 2, 3, 4, 5])
        index_dates = pd.Series({p: pd.Timestamp("2021-01-01") for p in pids})
        config = _default_feature_config()
        features_df, baseline_names, longitudinal_names = extract_oracle_features(
            history_df, pids, index_dates, config
        )
        self.assertEqual(features_df.shape[0], 5)
        self.assertEqual(
            features_df.shape[1], len(BASELINE_FEATURES) + len(LONGITUDINAL_FEATURES)
        )
        self.assertEqual(baseline_names, BASELINE_FEATURES)
        self.assertEqual(longitudinal_names, LONGITUDINAL_FEATURES)


class TestRecentEventCount(unittest.TestCase):
    def test_recent_event_count(self):
        # 3 events in last 90 days, 2 outside window
        index = pd.Timestamp("2021-01-01")
        records = [
            (1, "2020-01-01", "DOB"),  # outside 90-day window
            (1, "2020-05-01", "D/11111"),  # outside 90-day window
            (1, "2020-10-15", "D/22222"),  # within 90 days
            (1, "2020-11-01", "M/33333"),  # within 90 days
            (1, "2020-12-01", "P/44444"),  # within 90 days
        ]
        history_df = _make_test_df(records)
        pids = np.array([1])
        index_dates = pd.Series({1: index})
        config = _default_feature_config(recent_window_days=90)
        features_df, _, _ = extract_oracle_features(
            history_df, pids, index_dates, config
        )
        self.assertEqual(features_df.loc[1, "recent_event_count"], 3)


class TestDiseaseBurden(unittest.TestCase):
    def test_disease_burden(self):
        index = pd.Timestamp("2021-01-01")
        records = [
            (1, "2020-03-01", "D/11111"),
            (1, "2020-04-01", "D/22222"),
            (1, "2020-05-01", "D/33333"),
            (1, "2020-06-01", "D/44444"),
            (1, "2020-07-01", "D/55555"),
            (1, "2020-08-01", "M/66666"),  # medication, not a diagnosis
        ]
        history_df = _make_test_df(records)
        pids = np.array([1])
        index_dates = pd.Series({1: index})
        config = _default_feature_config()
        features_df, _, _ = extract_oracle_features(
            history_df, pids, index_dates, config
        )
        self.assertEqual(features_df.loc[1, "disease_burden"], 5)


class TestMedicationCount(unittest.TestCase):
    def test_medication_count(self):
        index = pd.Timestamp("2021-01-01")
        records = [
            (1, "2020-05-01", "M/11111"),
            (1, "2020-06-01", "M/22222"),
            (1, "2020-07-01", "M/33333"),
            (1, "2020-08-01", "D/44444"),  # diagnosis, not medication
        ]
        history_df = _make_test_df(records)
        pids = np.array([1])
        index_dates = pd.Series({1: index})
        config = _default_feature_config()
        features_df, _, _ = extract_oracle_features(
            history_df, pids, index_dates, config
        )
        self.assertEqual(features_df.loc[1, "medication_count"], 3)


class TestAgeComputation(unittest.TestCase):
    def test_age_computation(self):
        index = pd.Timestamp("2021-01-01")
        records = [
            (1, "1990-01-01", "DOB"),
            (1, "2020-06-01", "D/11111"),
        ]
        history_df = _make_test_df(records)
        pids = np.array([1])
        index_dates = pd.Series({1: index})
        config = _default_feature_config()
        features_df, _, _ = extract_oracle_features(
            history_df, pids, index_dates, config
        )
        expected_age = (index - pd.Timestamp("1990-01-01")).days / 365.25
        self.assertAlmostEqual(features_df.loc[1, "age"], expected_age, places=2)


class TestEventRecency(unittest.TestCase):
    def test_event_recency(self):
        index = pd.Timestamp("2021-01-01")
        records = [
            (1, "2020-06-01", "D/11111"),
            (1, "2020-12-22", "M/22222"),  # 10 days before index
        ]
        history_df = _make_test_df(records)
        pids = np.array([1])
        index_dates = pd.Series({1: index})
        config = _default_feature_config()
        features_df, _, _ = extract_oracle_features(
            history_df, pids, index_dates, config
        )
        self.assertEqual(features_df.loc[1, "event_recency"], 10)


class TestRecentBurstRatio(unittest.TestCase):
    def test_recent_burst_ratio(self):
        index = pd.Timestamp("2021-01-01")
        # 5 events in burst window (30 days), 20 total in lookback (365 days)
        records = []
        # 15 events outside burst window but inside lookback
        for i in range(15):
            records.append((1, f"2020-{3 + i % 9 + 1:02d}-01", f"D/{10000 + i}"))
        # 5 events inside burst window (last 30 days of 2020)
        for i in range(5):
            records.append((1, f"2020-12-{5 + i * 5:02d}", f"M/{20000 + i}"))
        history_df = _make_test_df(records)
        pids = np.array([1])
        index_dates = pd.Series({1: index})
        config = _default_feature_config(burst_window_days=30, lookback_days=365)
        features_df, _, _ = extract_oracle_features(
            history_df, pids, index_dates, config
        )
        # burst=5, lookback=20, ratio = 5 / (20 + 1)
        expected = 5.0 / 21.0
        self.assertAlmostEqual(
            features_df.loc[1, "recent_burst_ratio"], expected, places=5
        )


class TestNoMedicationCodes(unittest.TestCase):
    def test_no_medication_codes(self):
        index = pd.Timestamp("2021-01-01")
        records = [
            (1, "2020-06-01", "D/11111"),
            (2, "2020-07-01", "D/22222"),
        ]
        history_df = _make_test_df(records)
        pids = np.array([1, 2])
        index_dates = pd.Series({1: index, 2: index})
        config = _default_feature_config()
        features_df, _, _ = extract_oracle_features(
            history_df, pids, index_dates, config
        )
        self.assertEqual(features_df.loc[1, "medication_count"], 0)
        self.assertEqual(features_df.loc[2, "medication_count"], 0)


class TestStandardization(unittest.TestCase):
    def test_standardization(self):
        records = [
            (1, "1990-01-01", "DOB"),
            (1, "2020-06-01", "D/11111"),
            (1, "2020-09-01", "M/22222"),
            (2, "1985-03-15", "DOB"),
            (2, "2020-05-01", "D/33333"),
            (2, "2020-10-01", "M/44444"),
            (3, "1975-07-20", "DOB"),
            (3, "2020-08-01", "D/55555"),
            (3, "2020-11-01", "M/66666"),
        ]
        history_df = _make_test_df(records)
        pids = np.array([1, 2, 3])
        index_dates = pd.Series({p: pd.Timestamp("2021-01-01") for p in pids})
        config = FeatureConfig(standardize=True)
        features_df, _, _ = extract_oracle_features(
            history_df, pids, index_dates, config
        )
        for col in features_df.columns:
            col_std = features_df[col].std()
            # columns with zero variance stay at 0 after standardization
            if col_std > 1e-10:
                self.assertAlmostEqual(features_df[col].mean(), 0.0, places=10)


class TestSequenceMotifCount(unittest.TestCase):
    def test_sequence_motif_count(self):
        index = pd.Timestamp("2021-01-01")
        records = [
            (1, "2020-12-01", "D/11111"),  # diagnosis at day -31
            (
                1,
                "2020-12-20",
                "M/22222",
            ),  # medication 19 days later -> within 30-day window
        ]
        history_df = _make_test_df(records)
        pids = np.array([1])
        index_dates = pd.Series({1: index})
        config = _default_feature_config(motif_window_days=30)
        features_df, _, _ = extract_oracle_features(
            history_df, pids, index_dates, config
        )
        self.assertEqual(features_df.loc[1, "sequence_motif_count"], 1)


if __name__ == "__main__":
    unittest.main()
