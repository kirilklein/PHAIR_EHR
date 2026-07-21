"""Tests for nested semi-synthetic study aggregation."""

import unittest

import numpy as np
import pandas as pd

from corebehrt.constants.causal.data import EffectColumns as E
from experiments.semisynthetic_simulation.python_scripts.study_summary import (
    aggregate_replicates,
    summarize_performance,
)


def _estimate_rows(run_id, effects, true_effect):
    return pd.DataFrame(
        {
            "model": "bert",
            "run_id": run_id,
            "inner_id": ["k_01", "k_02"],
            E.method: "IPW",
            "outcome": "OUTCOME",
            E.effect: effects,
            E.effect_1: np.asarray(effects) + 0.2,
            E.effect_0: 0.2,
            E.true_effect: true_effect,
        }
    )


def _bootstrap_rows(run_id, samples):
    rows = []
    for inner_id, effects in zip(["k_01", "k_02"], samples):
        for bootstrap_id, effect in enumerate(effects, 1):
            rows.append(
                {
                    "model": "bert",
                    "run_id": run_id,
                    "inner_id": inner_id,
                    E.method: "IPW",
                    "outcome": "OUTCOME",
                    "effect_type": "ATE",
                    "bootstrap_id": bootstrap_id,
                    E.effect: effect,
                    E.effect_1: effect + 0.2,
                    E.effect_0: 0.2,
                }
            )
    return pd.DataFrame(rows)


class TestNestedStudySummary(unittest.TestCase):
    def test_aggregates_k_times_b_with_run_specific_truth(self):
        results = pd.concat(
            [
                _estimate_rows("run_01", [0.10, 0.12], 0.11),
                _estimate_rows("run_02", [0.20, 0.22], 0.19),
            ],
            ignore_index=True,
        )
        bootstraps = pd.concat(
            [
                _bootstrap_rows("run_01", [[0.08, 0.12], [0.10, 0.14]]),
                _bootstrap_rows("run_02", [[0.18, 0.22], [0.20, 0.24]]),
            ],
            ignore_index=True,
        )

        replicates = aggregate_replicates(results, bootstraps)
        self.assertEqual(list(replicates["n_bootstrap"]), [4, 4])
        self.assertAlmostEqual(replicates.iloc[0][E.effect], 0.11)
        self.assertAlmostEqual(
            replicates.iloc[0][E.std_err], np.std([0.08, 0.12, 0.10, 0.14], ddof=1)
        )

        summary = summarize_performance(replicates).iloc[0]
        self.assertEqual(summary["n"], 2)
        self.assertAlmostEqual(summary["bias"], 0.01)
        self.assertAlmostEqual(summary[E.true_effect], 0.15)

    def test_rejects_unequal_bootstrap_counts(self):
        results = _estimate_rows("run_01", [0.10, 0.12], 0.11)
        bootstraps = _bootstrap_rows("run_01", [[0.08], [0.10, 0.14]])

        with self.assertRaisesRegex(ValueError, "Unequal bootstrap counts"):
            aggregate_replicates(results, bootstraps)

    def test_risk_ratio_uses_log_scale_interval(self):
        results = _estimate_rows("run_01", [2.0, 2.0], 2.0)
        results[E.effect_1] = [0.4, 0.4]
        results[E.effect_0] = [0.2, 0.2]
        bootstraps = _bootstrap_rows("run_01", [[2.0, 2.1], [1.9, 2.0]])
        bootstraps["effect_type"] = "RR"
        bootstraps[E.effect_1] = [0.39, 0.41, 0.38, 0.40]
        bootstraps[E.effect_0] = [0.20, 0.20, 0.19, 0.20]

        replicate = aggregate_replicates(results, bootstraps).iloc[0]
        self.assertAlmostEqual(replicate[E.effect], 2.0)
        self.assertGreater(replicate["std_err_log"], 0)
        self.assertGreater(replicate[E.CI95_lower], 0)


if __name__ == "__main__":
    unittest.main()
