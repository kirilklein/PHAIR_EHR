import unittest

import numpy as np
import pandas as pd

from corebehrt.main_causal.helper import baseline_models


def make_separable_data(n_samples: int = 200):
    rng = np.random.RandomState(0)
    features = pd.DataFrame(
        {
            "code_a": rng.randint(0, 2, n_samples),
            "code_b": rng.randint(0, 2, n_samples),
            "age": rng.randint(40, 80, n_samples),
        }
    )
    targets = (features["code_a"] == 1).astype(int).values
    return features, targets


class TestBaselineModels(unittest.TestCase):
    def test_default_model_is_logistic(self):
        self.assertEqual(baseline_models.get_model_name({}), baseline_models.LOGISTIC)

    def test_unknown_model_raises(self):
        with self.assertRaises(ValueError):
            baseline_models.get_model_name({"model": "randomforest"})

    def test_base_params_merge_config_over_defaults(self):
        cfg = {"logistic": {"max_iter": 50}}
        base_params, config_params = baseline_models.get_base_params(
            cfg, baseline_models.LOGISTIC
        )
        self.assertEqual(base_params["max_iter"], 50)
        self.assertEqual(config_params, {"max_iter": 50})

    def test_tuning_ranges_are_model_specific(self):
        logistic_ranges = baseline_models.get_tuning_ranges(
            baseline_models.LOGISTIC, {}
        )
        catboost_ranges = baseline_models.get_tuning_ranges(
            baseline_models.CATBOOST, {}
        )
        self.assertEqual(list(logistic_ranges), ["C"])
        self.assertIn("learning_rate", catboost_ranges)
        self.assertNotIn("C", catboost_ranges)

    def test_configured_gpu_excludes_gpu_incompatible_parameters(self):
        """colsample_bylevel is unsupported on GPU, also when GPU comes from the config."""
        ranges = baseline_models.get_tuning_ranges(
            baseline_models.CATBOOST, {"task_type": "GPU", "devices": "0"}
        )
        self.assertNotIn("colsample_bylevel", ranges)

        params = {"n_estimators": 10, "task_type": "GPU", "colsample_bylevel": 0.8}
        prepared = baseline_models._prepare_catboost_params(
            params, baseline_models._effective_device_params(params)
        )
        self.assertNotIn("colsample_bylevel", prepared)

    def test_logistic_fits_and_predicts(self):
        features, targets = make_separable_data()
        params, _ = baseline_models.get_base_params({}, baseline_models.LOGISTIC)
        model = baseline_models.build_model(
            baseline_models.LOGISTIC, params, scale_pos_weight=1.0, random_seed=42
        )
        baseline_models.fit_model(
            model, baseline_models.LOGISTIC, params, features, targets
        )

        probas = model.predict_proba(features)[:, 1]
        self.assertEqual(probas.shape, targets.shape)
        self.assertTrue(((probas >= 0) & (probas <= 1)).all())
        # code_a fully determines the target, so the fit should separate the classes.
        self.assertGreater(probas[targets == 1].mean(), probas[targets == 0].mean())

    def test_catboost_fits_with_early_stopping_on_validation_set(self):
        features, targets = make_separable_data()
        params = {"n_estimators": 10, "early_stopping_rounds": 5}
        model = baseline_models.build_model(
            baseline_models.CATBOOST, params, scale_pos_weight=1.0, random_seed=42
        )
        baseline_models.fit_model(
            model,
            baseline_models.CATBOOST,
            params,
            features,
            targets,
            features,
            targets,
        )

        probas = model.predict_proba(features)[:, 1]
        self.assertEqual(probas.shape, targets.shape)
        self.assertGreater(probas[targets == 1].mean(), probas[targets == 0].mean())


if __name__ == "__main__":
    unittest.main()
