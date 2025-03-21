import unittest

import pandas as pd

from corebehrt.functional.causal.counterfactuals import expand_counterfactuals


class TestExpandCounterfactuals(unittest.TestCase):
    def test_exposure_1(self):
        """For an exposed individual, outcome under exposure equals cf_probas,
        and outcome under control equals 1 - cf_probas."""
        data = {"exposure": [1], "cf_probas": [0.8]}
        df = pd.DataFrame(data)
        result = expand_counterfactuals(
            df, "exposure", "cf_probas", "outcome_control", "outcome_exposed"
        )
        # For exposure == 1:
        # outcome_exposed should be cf_probas (0.8) and outcome_control should be 1 - cf_probas (0.2)
        self.assertAlmostEqual(result.loc[0, "outcome_exposed"], 0.8)
        self.assertAlmostEqual(result.loc[0, "outcome_control"], 0.2)

    def test_exposure_0(self):
        """For a non-exposed individual, outcome under control equals cf_probas,
        and outcome under exposure equals 1 - cf_probas."""
        data = {"exposure": [0], "cf_probas": [0.3]}
        df = pd.DataFrame(data)
        result = expand_counterfactuals(
            df, "exposure", "cf_probas", "outcome_control", "outcome_exposed"
        )
        # For exposure == 0:
        # outcome_control should be cf_probas (0.3) and outcome_exposed should be 1 - cf_probas (0.7)
        self.assertAlmostEqual(result.loc[0, "outcome_control"], 0.3)
        self.assertAlmostEqual(result.loc[0, "outcome_exposed"], 0.7)

    def test_mixed_exposures(self):
        """Test a mixed scenario with multiple rows and verify the calculated outcomes."""
        data = {"exposure": [1, 0, 1, 0], "cf_probas": [0.6, 0.2, 0.9, 0.4]}
        df = pd.DataFrame(data)
        result = expand_counterfactuals(
            df, "exposure", "cf_probas", "outcome_control", "outcome_exposed"
        )
        # For rows with exposure == 1:
        # outcome_exposed = cf_probas, outcome_control = 1 - cf_probas.
        # For rows with exposure == 0:
        # outcome_control = cf_probas, outcome_exposed = 1 - cf_probas.
        expected_outcome_exposed = [
            0.6,  # row 0: exposure 1 -> 0.6
            0.8,  # row 1: exposure 0 -> 1 - 0.2 = 0.8
            0.9,  # row 2: exposure 1 -> 0.9
            0.6,  # row 3: exposure 0 -> 1 - 0.4 = 0.6
        ]
        expected_outcome_control = [
            0.4,  # row 0: exposure 1 -> 1 - 0.6 = 0.4
            0.2,  # row 1: exposure 0 -> 0.2
            0.1,  # row 2: exposure 1 -> 1 - 0.9 = 0.1
            0.4,  # row 3: exposure 0 -> 0.4
        ]
        for i in range(len(df)):
            self.assertAlmostEqual(
                result.loc[i, "outcome_exposed"], expected_outcome_exposed[i]
            )
            self.assertAlmostEqual(
                result.loc[i, "outcome_control"], expected_outcome_control[i]
            )

    def test_original_dataframe_not_modified(self):
        """Ensure that the original DataFrame is not modified by the function."""
        data = {"exposure": [1, 0], "cf_probas": [0.75, 0.25]}
        df = pd.DataFrame(data)
        df_copy = df.copy()
        _ = expand_counterfactuals(
            df, "exposure", "cf_probas", "outcome_control", "outcome_exposed"
        )
        pd.testing.assert_frame_equal(df, df_copy)
