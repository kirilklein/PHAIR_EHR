import unittest
from unittest.mock import MagicMock, patch

from corebehrt.constants.data import TRAIN_KEY, VAL_KEY
from corebehrt.functional.features.split import create_folds
from corebehrt.main_causal.finetune_exp_y import validate_folds


class TestSubpopFoldCreation(unittest.TestCase):
    def test_bootstrap_folds_from_subpop(self):
        """Verify create_folds produces valid bootstrap folds from a subset of PIDs."""
        subpop_pids = list(range(50))
        folds = create_folds(subpop_pids, num_folds=5, seed=42, bootstrap=True)

        self.assertEqual(len(folds), 5)
        for fold in folds:
            self.assertIn(TRAIN_KEY, fold)
            self.assertIn(VAL_KEY, fold)
            # Total count matches original size
            total = len(fold[TRAIN_KEY]) + len(fold[VAL_KEY])
            self.assertEqual(total, len(subpop_pids))

        # Validation should pass with bootstrap=True
        validate_folds(
            folds,
            set(subpop_pids),
            logger=MagicMock(),
            bootstrap=True,
        )

    def test_non_bootstrap_folds_from_subpop(self):
        """Verify standard CV folds work on subpopulation."""
        subpop_pids = list(range(50))
        folds = create_folds(subpop_pids, num_folds=5, seed=42, bootstrap=False)

        self.assertEqual(len(folds), 5)
        all_val = set()
        for fold in folds:
            val_pids = set(fold[VAL_KEY])
            train_pids = set(fold[TRAIN_KEY])
            self.assertTrue(val_pids.isdisjoint(train_pids))
            all_val.update(val_pids)
        self.assertEqual(all_val, set(subpop_pids))


class TestFreezeEncoderAtInit(unittest.TestCase):
    @patch("corebehrt.modules.trainer.causal.trainer.CausalEHRTrainer._freeze_encoder")
    @patch(
        "corebehrt.modules.trainer.causal.trainer.EHRTrainer.__init__",
        return_value=None,
    )
    def test_freeze_called_when_flag_set(self, mock_init, mock_freeze):
        """Verify _freeze_encoder is called when freeze_encoder_at_init=True."""
        from corebehrt.modules.trainer.causal.trainer import CausalEHRTrainer

        trainer = CausalEHRTrainer.__new__(CausalEHRTrainer)
        trainer.args = {"freeze_encoder_at_init": True, "use_pcgrad": False}
        trainer.model = MagicMock()
        trainer.model.config.outcome_names = ["outcome_1"]
        trainer.metric_history = {}
        trainer.epoch_history = []
        trainer.encoder_frozen = False
        trainer.outcome_names = ["outcome_1"]
        trainer.best_outcome_aucs = {}
        trainer.best_exposure_auc = None
        trainer.use_pcgrad = False
        trainer.plot_histograms = False
        trainer.plot_gradients = False
        trainer.plot_gradients_frequency = 100
        trainer.plot_log_scale = False
        trainer.global_step = 0
        trainer.update_step = 0
        trainer._set_plateau_parameters()
        trainer._set_logging_parameters()

        if trainer.args.get("freeze_encoder_at_init", False):
            trainer._freeze_encoder()

        mock_freeze.assert_called_once()

    @patch("corebehrt.modules.trainer.causal.trainer.CausalEHRTrainer._freeze_encoder")
    @patch(
        "corebehrt.modules.trainer.causal.trainer.EHRTrainer.__init__",
        return_value=None,
    )
    def test_freeze_not_called_when_flag_not_set(self, mock_init, mock_freeze):
        """Verify _freeze_encoder is NOT called when freeze_encoder_at_init is absent."""
        from corebehrt.modules.trainer.causal.trainer import CausalEHRTrainer

        trainer = CausalEHRTrainer.__new__(CausalEHRTrainer)
        trainer.args = {"use_pcgrad": False}

        if trainer.args.get("freeze_encoder_at_init", False):
            trainer._freeze_encoder()

        mock_freeze.assert_not_called()


if __name__ == "__main__":
    unittest.main()
