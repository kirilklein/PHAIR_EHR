"""Tests for the semi-synthetic causal simulator."""

import unittest

import numpy as np
import pandas as pd

from corebehrt.constants.data import CONCEPT_COL, PID_COL, TIMESTAMP_COL
from corebehrt.modules.simulation.config_semisynthetic import (
    FeatureConfig,
    OutcomeModelConfig,
    PathsConfig,
    SemiSyntheticOutcomeConfig,
    SemiSyntheticSimulationConfig,
    TreatmentEffectConfig,
)
from corebehrt.modules.simulation.semisynthetic_simulator import (
    ASSIGNED_INDEX_DATE_COL,
    SemiSyntheticCausalSimulator,
)


def _make_test_shard(n_patients=20, n_exposed=8, seed=42):
    """Create a synthetic MEDS shard with known structure."""
    rng = np.random.RandomState(seed)
    base_date = pd.Timestamp("2018-01-01")
    index_date = pd.Timestamp("2020-01-01")

    rows = []
    for i in range(n_patients):
        pid = i + 1
        # DOB event
        dob = base_date - pd.Timedelta(days=rng.randint(10000, 25000))
        rows.append(
            {
                PID_COL: pid,
                TIMESTAMP_COL: dob,
                CONCEPT_COL: "DOB",
                ASSIGNED_INDEX_DATE_COL: index_date,
            }
        )

        # Generate several diagnosis codes (for min_num_codes to pass)
        n_events = rng.randint(6, 15)
        for _ in range(n_events):
            days_before = rng.randint(1, 700)
            code = f"D/{rng.choice(['A01', 'B02', 'C03', 'D04', 'E05', 'F06'])}"
            rows.append(
                {
                    PID_COL: pid,
                    TIMESTAMP_COL: index_date - pd.Timedelta(days=days_before),
                    CONCEPT_COL: code,
                    ASSIGNED_INDEX_DATE_COL: index_date,
                }
            )

        # Add some medication codes
        for _ in range(rng.randint(2, 5)):
            days_before = rng.randint(1, 400)
            code = f"M/{rng.choice(['X01', 'X02', 'X03'])}"
            rows.append(
                {
                    PID_COL: pid,
                    TIMESTAMP_COL: index_date - pd.Timedelta(days=days_before),
                    CONCEPT_COL: code,
                    ASSIGNED_INDEX_DATE_COL: index_date,
                }
            )

        # Exposure event for first n_exposed patients
        if i < n_exposed:
            rows.append(
                {
                    PID_COL: pid,
                    TIMESTAMP_COL: index_date,
                    CONCEPT_COL: "EXPOSURE",
                    ASSIGNED_INDEX_DATE_COL: index_date,
                }
            )

    df = pd.DataFrame(rows)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    df[ASSIGNED_INDEX_DATE_COL] = pd.to_datetime(df[ASSIGNED_INDEX_DATE_COL])
    return df


def _make_config(
    tmpdir,
    delta=1.0,
    mode="constant",
    seed=42,
    min_num_codes=3,
    noise_scale=0.0,
):
    """Create a minimal SemiSyntheticSimulationConfig for testing."""
    paths = PathsConfig(data=".", splits=["test"], outcomes=tmpdir)
    outcome = SemiSyntheticOutcomeConfig(
        outcome_model=OutcomeModelConfig(
            beta_0=-1.0,
            baseline_coefficients={"disease_burden": 0.3},
            longitudinal_coefficients={},
            noise_scale=noise_scale,
        ),
        treatment_effect=TreatmentEffectConfig(mode=mode, delta=delta),
    )
    return SemiSyntheticSimulationConfig(
        paths=paths,
        features=FeatureConfig(standardize=True),
        outcomes={"OUTCOME_test": outcome},
        seed=seed,
        min_num_codes=min_num_codes,
    )


class TestSemiSyntheticSimulatorOutputFormat(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmpdir = tempfile.mkdtemp()
        self.config = _make_config(self.tmpdir)
        self.simulator = SemiSyntheticCausalSimulator(self.config)
        self.shard = _make_test_shard()
        self.results = self.simulator.simulate_dataset(self.shard)

    def test_simulate_dataset_output_format(self):
        self.assertIn("counterfactuals", self.results)
        self.assertIn("ite", self.results)

        cf_df = self.results["counterfactuals"]
        self.assertIn(PID_COL, cf_df.columns)
        self.assertIn("exposure", cf_df.columns)
        self.assertIn("outcome_OUTCOME_test", cf_df.columns)
        self.assertIn("Y1_OUTCOME_test", cf_df.columns)
        self.assertIn("Y0_OUTCOME_test", cf_df.columns)
        self.assertIn("P1_OUTCOME_test", cf_df.columns)
        self.assertIn("P0_OUTCOME_test", cf_df.columns)

        ite_df = self.results["ite"]
        self.assertIn(PID_COL, ite_df.columns)
        self.assertIn("ite_OUTCOME_test", ite_df.columns)

    def test_probabilities_in_range(self):
        cf_df = self.results["counterfactuals"]
        p0 = cf_df["P0_OUTCOME_test"].values
        p1 = cf_df["P1_OUTCOME_test"].values
        self.assertTrue(np.all(p0 >= 0) and np.all(p0 <= 1))
        self.assertTrue(np.all(p1 >= 0) and np.all(p1 <= 1))

    def test_outcome_consistency(self):
        """Y_obs = A * Y1 + (1-A) * Y0 for all patients."""
        cf_df = self.results["counterfactuals"]
        a = cf_df["exposure"].values
        y1 = cf_df["Y1_OUTCOME_test"].values
        y0 = cf_df["Y0_OUTCOME_test"].values
        y_obs = cf_df["outcome_OUTCOME_test"].values
        expected = a * y1 + (1 - a) * y0
        np.testing.assert_array_equal(y_obs, expected)


class TestSemiSyntheticSimulatorReproducibility(unittest.TestCase):
    def test_reproducibility(self):
        import tempfile

        tmpdir = tempfile.mkdtemp()
        config = _make_config(tmpdir, seed=123)
        shard = _make_test_shard()

        sim1 = SemiSyntheticCausalSimulator(config)
        results1 = sim1.simulate_dataset(shard)

        # Re-create from scratch with same seed
        config2 = _make_config(tmpdir, seed=123)
        sim2 = SemiSyntheticCausalSimulator(config2)
        results2 = sim2.simulate_dataset(shard)

        cf1 = results1["counterfactuals"]
        cf2 = results2["counterfactuals"]
        np.testing.assert_array_equal(
            cf1["outcome_OUTCOME_test"].values,
            cf2["outcome_OUTCOME_test"].values,
        )
        np.testing.assert_array_almost_equal(
            cf1["P0_OUTCOME_test"].values,
            cf2["P0_OUTCOME_test"].values,
        )


class TestSemiSyntheticTreatmentEffect(unittest.TestCase):
    def test_constant_treatment_effect(self):
        """With constant positive delta, ITE should be positive for most patients."""
        import tempfile

        tmpdir = tempfile.mkdtemp()
        config = _make_config(tmpdir, delta=2.0)
        simulator = SemiSyntheticCausalSimulator(config)
        results = simulator.simulate_dataset(_make_test_shard())

        ite = results["ite"]["ite_OUTCOME_test"].values
        # Most ITEs should be positive with delta=2.0
        self.assertGreater(np.mean(ite > 0), 0.5)

    def test_null_treatment_effect(self):
        """With delta=0, mean ITE should be close to 0."""
        import tempfile

        tmpdir = tempfile.mkdtemp()
        config = _make_config(tmpdir, delta=0.0)
        simulator = SemiSyntheticCausalSimulator(config)
        results = simulator.simulate_dataset(_make_test_shard())

        ite = results["ite"]["ite_OUTCOME_test"].values
        self.assertAlmostEqual(np.mean(ite), 0.0, places=5)


class TestSemiSyntheticExposureFromData(unittest.TestCase):
    def test_exposure_from_data(self):
        """Exposed patients should be exactly those with EXPOSURE events."""
        import tempfile

        tmpdir = tempfile.mkdtemp()
        n_exposed = 8
        shard = _make_test_shard(n_patients=20, n_exposed=n_exposed)
        config = _make_config(tmpdir)
        simulator = SemiSyntheticCausalSimulator(config)
        results = simulator.simulate_dataset(shard)

        cf_df = results["counterfactuals"]
        exposed_pids_from_sim = set(cf_df.loc[cf_df["exposure"] == 1, PID_COL].values)

        # In the test shard, patients 1..n_exposed have EXPOSURE events
        expected_exposed = set(range(1, n_exposed + 1))
        # The sim may have dropped some patients (min_num_codes), so check
        # that the exposed set is a subset of the expected
        remaining_pids = set(cf_df[PID_COL].values)
        expected_in_remaining = expected_exposed & remaining_pids
        self.assertEqual(exposed_pids_from_sim, expected_in_remaining)


if __name__ == "__main__":
    unittest.main()
