"""Tests for BERT bootstrap-refit fold construction."""

import logging
import os
import sys
import tempfile
import types
import unittest

import torch

sys.modules.setdefault("umap", types.ModuleType("umap"))

from corebehrt.constants.data import TRAIN_KEY, VAL_KEY
from corebehrt.constants.paths import FOLDS_FILE
from corebehrt.main_causal.finetune_exp_y import handle_folds
from corebehrt.modules.setup.config import Config


class TestFinetuneBootstrapFolds(unittest.TestCase):
    def test_training_is_bootstrapped_and_validation_covers_cohort(self):
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

        folds = handle_folds(
            cfg, [], pids, logging.getLogger("test_finetune_bootstrap_folds")
        )

        validation_pids = [pid for fold in folds for pid in fold[VAL_KEY]]
        self.assertEqual(
            [fold[VAL_KEY] for fold in folds],
            [fold[VAL_KEY] for fold in original],
        )
        self.assertEqual(set(validation_pids), set(pids))
        self.assertEqual(len(validation_pids), len(pids))
        self.assertTrue(
            any(len(fold[TRAIN_KEY]) > len(set(fold[TRAIN_KEY])) for fold in folds)
        )


if __name__ == "__main__":
    unittest.main()
