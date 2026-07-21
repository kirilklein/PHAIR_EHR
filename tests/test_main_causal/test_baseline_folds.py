"""Tests for baseline refit fold reshuffling."""

import logging
import os
import sys
import tempfile
import types
import unittest

import torch

try:
    import optuna  # noqa: F401
except ImportError:
    optuna_stub = types.ModuleType("optuna")
    optuna_stub.Trial = object
    sys.modules["optuna"] = optuna_stub

from corebehrt.constants.data import TRAIN_KEY, VAL_KEY
from corebehrt.constants.paths import FOLDS_FILE
from corebehrt.main_causal.helper.train_baseline import handle_folds
from corebehrt.modules.setup.config import Config


class TestBaselineFoldReshuffling(unittest.TestCase):
    def test_reshuffle_uses_configured_seed_and_all_patients(self):
        prepared_dir = tempfile.mkdtemp()
        model_dir = tempfile.mkdtemp()
        pids = list(range(20))
        original = [
            {TRAIN_KEY: pids[10:], VAL_KEY: pids[:10]},
            {TRAIN_KEY: pids[:10], VAL_KEY: pids[10:]},
        ]
        torch.save(original, os.path.join(prepared_dir, FOLDS_FILE))
        cfg = Config(
            {
                "paths": {"prepared_data": prepared_dir, "model": model_dir},
                "data": {"reshuffle": True, "reshuffle_seed": 123},
            }
        )

        reshuffled = handle_folds(cfg, logging.getLogger("test_baseline_folds"))

        self.assertNotEqual(reshuffled, original)
        for fold in reshuffled:
            self.assertEqual(set(fold[TRAIN_KEY]) | set(fold[VAL_KEY]), set(pids))
            self.assertTrue(
                set(fold[TRAIN_KEY]).isdisjoint(set(fold[VAL_KEY]))
            )


if __name__ == "__main__":
    unittest.main()
