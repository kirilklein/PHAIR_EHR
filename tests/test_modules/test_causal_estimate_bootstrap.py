"""Tests for patient-bootstrap persistence in causal estimation."""

import logging
import unittest

import numpy as np
import pandas as pd

from corebehrt.constants.causal.data import (
    EXPOSURE_COL,
    OUTCOME,
    PROBAS,
    PROBAS_CONTROL,
    PROBAS_EXPOSED,
    PS_COL,
    EffectColumns,
)
from corebehrt.constants.data import PID_COL
from corebehrt.functional.estimate.benchmarks import append_true_effect
from corebehrt.modules.causal.estimate import EffectEstimator
from corebehrt.modules.setup.config import Config


class TestEffectEstimatorBootstrap(unittest.TestCase):
    def test_saves_draws_but_keeps_observed_point_estimate(self):
        rng = np.random.default_rng(42)
        exposure = np.repeat([0, 1], 30)
        outcome = rng.binomial(1, 0.3 + 0.2 * exposure)
        frame = pd.DataFrame(
            {
                EXPOSURE_COL: exposure,
                OUTCOME: outcome,
                PS_COL: np.where(exposure, 0.55, 0.45),
                PROBAS: 0.3 + 0.2 * exposure,
                PROBAS_CONTROL: 0.3,
                PROBAS_EXPOSED: 0.5,
            }
        )
        estimator = EffectEstimator.__new__(EffectEstimator)
        estimator.estimator_cfg = Config({"methods": ["IPW"]})
        estimator.effect_type = "ATE"
        estimator.clip_percentile = 0.99
        estimator.n_bootstrap = 5
        estimator.save_bootstrap_samples = True
        estimator.use_observed_point_estimate = True
        estimator.bootstrap_records = []
        estimator.logger = logging.getLogger("test_bootstrap")

        result = estimator._estimate_effects(frame, "OUTCOME")

        self.assertEqual(len(estimator.bootstrap_records), 5)
        self.assertEqual(
            {row["bootstrap_id"] for row in estimator.bootstrap_records},
            set(range(1, 6)),
        )
        observed = outcome[exposure == 1].mean() - outcome[exposure == 0].mean()
        self.assertAlmostEqual(result.iloc[0][EffectColumns.effect], observed)
        self.assertGreater(result.iloc[0][EffectColumns.std_err], 0)

    def test_true_risk_ratio_uses_counterfactual_probabilities(self):
        counterfactuals = pd.DataFrame(
            {
                PID_COL: [1, 2],
                "Y1_OUTCOME": [0, 1],
                "Y0_OUTCOME": [0, 0],
                "P1_OUTCOME": [0.4, 0.6],
                "P0_OUTCOME": [0.2, 0.3],
                EXPOSURE_COL: [0, 1],
            }
        )
        result = append_true_effect(
            pd.DataFrame({EffectColumns.method: ["IPW"]}),
            ite_df=None,
            outcome_name="OUTCOME",
            analysis_pids=np.array([1, 2]),
            effect_type="RR",
            counterfactual_df=counterfactuals,
        )

        self.assertAlmostEqual(result.iloc[0][EffectColumns.true_effect], 2.0)


if __name__ == "__main__":
    unittest.main()
