"""Inner tuning split must keep bootstrap duplicates of inner-train patients."""

import sys
import types
import unittest

try:
    import optuna  # noqa: F401
except ImportError:
    optuna_stub = types.ModuleType("optuna")
    optuna_stub.Trial = object
    sys.modules["optuna"] = optuna_stub

from corebehrt.main_causal.helper.train_baseline import split_inner_data
from corebehrt.modules.preparation.causal.dataset import (
    CausalPatientData,
    CausalPatientDataset,
)


def _patient(pid):
    return CausalPatientData(
        pid=pid, concepts=[1], abspos=[0.0], segments=[0], ages=[1.0]
    )


class TestSplitInnerData(unittest.TestCase):
    def test_duplicates_survive_and_halves_are_disjoint(self):
        data = CausalPatientDataset([_patient(i) for i in range(10)])
        bootstrapped = [0, 0, 1, 2, 2, 2, 3, 4, 5, 6, 7, 8, 9, 9]
        outer_train = data.resample_by_pids(bootstrapped)

        inner_train, inner_val = split_inner_data(outer_train, data, 0.3, 0)

        train_pids = inner_train.get_pids()
        val_pids = inner_val.get_pids()
        self.assertTrue(set(train_pids).isdisjoint(val_pids))
        self.assertEqual(len(val_pids), len(set(val_pids)))
        for pid in set(train_pids):
            self.assertEqual(train_pids.count(pid), bootstrapped.count(pid))
        self.assertEqual(set(train_pids) | set(val_pids), set(bootstrapped))
