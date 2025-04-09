import unittest

import pandas as pd

from corebehrt.functional.causal.counterfactuals import expand_counterfactuals
from corebehrt.constants.causal.data import (
    PROBAS,
    EXPOSURE_COL,
    CF_PROBAS,
)


class TestExpandCounterfactuals(unittest.TestCase):
    def test_exposure_1(self):
        """For an exposed individual, outcome under exposure equals cf_probas,
        and outcome under control equals 1 - cf_probas."""
        data = {EXPOSURE_COL: [1], CF_PROBAS: [0.3], PROBAS: [0.2]}
        df = pd.DataFrame(data)
        result = expand_counterfactuals(
            df, EXPOSURE_COL, PROBAS, CF_PROBAS, "outcome_control", "outcome_exposed"
        )
        self.assertAlmostEqual(result.loc[0, "outcome_exposed"], 0.2)
        self.assertAlmostEqual(result.loc[0, "outcome_control"], 0.3)

    def test_exposure_0(self):
        """For a non-exposed individual, outcome under control equals cf_probas,"""
        data = {EXPOSURE_COL: [0], CF_PROBAS: [0.5], PROBAS: [0.3]}
        df = pd.DataFrame(data)
        result = expand_counterfactuals(
            df, EXPOSURE_COL, PROBAS, CF_PROBAS, "outcome_control", "outcome_exposed"
        )
        self.assertAlmostEqual(result.loc[0, "outcome_control"], 0.3)
        self.assertAlmostEqual(result.loc[0, "outcome_exposed"], 0.5)

    def test_mixed_exposures(self):
        """Test a mixed scenario with multiple rows and verify the calculated outcomes."""
        data = {
            EXPOSURE_COL: [1, 0, 1, 0],
            CF_PROBAS: [0.6, 0.2, 0.9, 0.4],
            PROBAS: [0.2, 0.8, 0.1, 0.6],
        }
        df = pd.DataFrame(data)
        result = expand_counterfactuals(
            df, EXPOSURE_COL, PROBAS, CF_PROBAS, "outcome_control", "outcome_exposed"
        )
        expected_outcome_exposed = [
            0.2,  # row 0
            0.2,  # row 1
            0.1,  # row 2
            0.4,  # row 3
        ]
        expected_outcome_control = [
            0.6,  # row 0
            0.8,  # row 1
            0.9,  # row 2
            0.6,  # row 3
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
        data = {EXPOSURE_COL: [1, 0], CF_PROBAS: [0.75, 0.25], PROBAS: [0.2, 0.8]}
        df = pd.DataFrame(data)
        df_copy = df.copy()
        _ = expand_counterfactuals(
            df, "exposure", "probas", "cf_probas", "outcome_control", "outcome_exposed"
        )
        pd.testing.assert_frame_equal(df, df_copy)
