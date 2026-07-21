"""Tests for baseline bootstrap-refit folds."""

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


class TestBaselineBootstrapFolds(unittest.TestCase):
    def test_bootstrap_resamples_training_and_keeps_validation_complete(self):
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
                "data": {
                    "reshuffle": False,
                    "bootstrap_seed": 123,
                },
                "bootstrap": True,
            }
        )

        folds = handle_folds(cfg, logging.getLogger("test_baseline_folds"))

        self.assertNotEqual(folds, original)
        validation_pids = []
        duplicate_training_found = False
        for fold, original_fold in zip(folds, original):
            validation_pids.extend(fold[VAL_KEY])
            self.assertEqual(fold[VAL_KEY], original_fold[VAL_KEY])
            self.assertEqual(len(fold[VAL_KEY]), len(set(fold[VAL_KEY])))
            self.assertTrue(set(fold[TRAIN_KEY]).isdisjoint(set(fold[VAL_KEY])))
            duplicate_training_found |= len(fold[TRAIN_KEY]) > len(set(fold[TRAIN_KEY]))
        self.assertEqual(set(validation_pids), set(pids))
        self.assertEqual(len(validation_pids), len(pids))
        self.assertTrue(duplicate_training_found)


if __name__ == "__main__":
    unittest.main()
