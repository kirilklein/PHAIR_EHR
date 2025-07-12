import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np

from corebehrt.functional.preparation.causal.outcomes import (
    get_binary_exposure,
    get_binary_outcome,
)
from corebehrt.constants.data import PID_COL


class TestGetBinaryExposure(unittest.TestCase):
    def setUp(self):
        """Set up test data for binary exposure tests."""
        self.exposures = pd.DataFrame(
            {PID_COL: [1, 1, 2, 3], "abspos": [100, 200, 150, 300]}
        )

        self.index_dates = pd.DataFrame({PID_COL: [1, 2, 3], "abspos": [50, 100, 250]})

        self.data_end = pd.Timestamp("2023-12-31")

    @patch("corebehrt.functional.preparation.causal.outcomes.prepare_follow_ups_simple")
    @patch("corebehrt.functional.preparation.causal.outcomes.abspos_to_binary_outcome")
    def test_get_binary_exposure_basic(
        self, mock_abspos_to_binary, mock_prepare_follow_ups
    ):
        """Test basic functionality of get_binary_exposure."""
        # Mock return values
        mock_follow_ups = pd.DataFrame(
            {
                PID_COL: [1, 2, 3],
                "start_abspos": [25, 75, 225],
                "end_abspos": [75, 125, 275],
            }
        )
        mock_prepare_follow_ups.return_value = mock_follow_ups

        expected_binary = pd.Series([1, 0, 1], index=[1, 2, 3])
        mock_abspos_to_binary.return_value = expected_binary

        # Call function
        result = get_binary_exposure(
            self.exposures,
            self.index_dates,
            n_hours_start_follow_up=-25,
            n_hours_end_follow_up=25,
            data_end=self.data_end,
        )

        # Verify mocks were called correctly
        mock_prepare_follow_ups.assert_called_once_with(
            self.index_dates, -25, 25, self.data_end
        )
        mock_abspos_to_binary.assert_called_once_with(mock_follow_ups, self.exposures)

        # Verify result
        pd.testing.assert_series_equal(result, expected_binary)

    @patch("corebehrt.functional.preparation.causal.outcomes.prepare_follow_ups_simple")
    @patch("corebehrt.functional.preparation.causal.outcomes.abspos_to_binary_outcome")
    def test_get_binary_exposure_empty_exposures(
        self, mock_abspos_to_binary, mock_prepare_follow_ups
    ):
        """Test get_binary_exposure with empty exposures DataFrame."""
        empty_exposures = pd.DataFrame({PID_COL: [], "abspos": []})

        mock_follow_ups = pd.DataFrame(
            {
                PID_COL: [1, 2, 3],
                "start_abspos": [25, 75, 225],
                "end_abspos": [75, 125, 275],
            }
        )
        mock_prepare_follow_ups.return_value = mock_follow_ups

        expected_binary = pd.Series([0, 0, 0], index=[1, 2, 3])
        mock_abspos_to_binary.return_value = expected_binary

        result = get_binary_exposure(
            empty_exposures,
            self.index_dates,
            n_hours_start_follow_up=0,
            n_hours_end_follow_up=100,
            data_end=self.data_end,
        )

        pd.testing.assert_series_equal(result, expected_binary)

    @patch("corebehrt.functional.preparation.causal.outcomes.prepare_follow_ups_simple")
    @patch("corebehrt.functional.preparation.causal.outcomes.abspos_to_binary_outcome")
    def test_get_binary_exposure_zero_follow_up_window(
        self, mock_abspos_to_binary, mock_prepare_follow_ups
    ):
        """Test get_binary_exposure with zero follow-up window."""
        mock_follow_ups = pd.DataFrame(
            {
                PID_COL: [1, 2, 3],
                "start_abspos": [50, 100, 250],
                "end_abspos": [50, 100, 250],
            }
        )
        mock_prepare_follow_ups.return_value = mock_follow_ups

        expected_binary = pd.Series([0, 0, 0], index=[1, 2, 3])
        mock_abspos_to_binary.return_value = expected_binary

        result = get_binary_exposure(
            self.exposures,
            self.index_dates,
            n_hours_start_follow_up=0,
            n_hours_end_follow_up=0,
            data_end=self.data_end,
        )

        mock_prepare_follow_ups.assert_called_once_with(
            self.index_dates, 0, 0, self.data_end
        )
        pd.testing.assert_series_equal(result, expected_binary)


class TestGetBinaryOutcome(unittest.TestCase):
    def setUp(self):
        """Set up test data for binary outcome tests."""
        self.index_dates = pd.DataFrame(
            {PID_COL: [1, 2, 3, 4], "abspos": [100, 200, 300, 400]}
        )

        self.outcomes = pd.DataFrame({PID_COL: [1, 2, 4], "abspos": [150, 250, 500]})

        self.index_date_matching = pd.DataFrame(
            {"control_subject_id": [2, 4], "exposed_subject_id": [1, 3]}
        )

        self.deaths = pd.Series([np.nan, 280, np.nan, 450], index=[1, 2, 3, 4])

        self.exposures = pd.DataFrame(
            {PID_COL: [1, 1, 3, 3], "abspos": [80, 120, 250, 350]}
        )

        self.data_end = pd.Timestamp("2023-12-31")

    @patch(
        "corebehrt.functional.preparation.causal.outcomes.filter_df_by_unique_values"
    )
    @patch("corebehrt.functional.preparation.causal.outcomes.get_non_compliance_abspos")
    @patch("corebehrt.functional.preparation.causal.outcomes.prepare_follow_ups_simple")
    @patch(
        "corebehrt.functional.preparation.causal.outcomes.prepare_follow_ups_adjusted"
    )
    @patch("corebehrt.functional.preparation.causal.outcomes.abspos_to_binary_outcome")
    def test_get_binary_outcome_basic(
        self,
        mock_abspos_to_binary,
        mock_prepare_adjusted,
        mock_prepare_simple,
        mock_get_non_compliance,
        mock_filter_df,
    ):
        """Test basic functionality of get_binary_outcome."""
        # Mock return values
        mock_filter_df.return_value = self.index_date_matching
        mock_non_compliance = pd.Series([200, np.nan, 400, np.nan], index=[1, 2, 3, 4])
        mock_get_non_compliance.return_value = mock_non_compliance

        mock_simple_follow_ups = pd.DataFrame(
            {
                PID_COL: [1, 2, 3, 4],
                "start_abspos": [100, 200, 300, 400],
                "end_abspos": [1000, 1100, 1200, 1300],
            }
        )
        mock_prepare_simple.return_value = mock_simple_follow_ups

        mock_adjusted_follow_ups = pd.DataFrame(
            {
                PID_COL: [1, 2, 3, 4],
                "start_abspos": [100, 200, 300, 400],
                "end_abspos": [200, 280, 400, 450],  # Adjusted for compliance/deaths
            }
        )
        mock_prepare_adjusted.return_value = mock_adjusted_follow_ups

        expected_binary = pd.Series([1, 1, 0, 0], index=[1, 2, 3, 4])
        mock_abspos_to_binary.return_value = expected_binary

        # Call function
        binary_outcomes, follow_ups = get_binary_outcome(
            self.index_dates,
            self.outcomes,
            n_hours_start_follow_up=0,
            n_hours_end_follow_up=900,
            n_hours_compliance=100,
            index_date_matching=self.index_date_matching,
            deaths=self.deaths,
            exposures=self.exposures,
            data_end=self.data_end,
        )

        # Verify mocks were called correctly
        mock_filter_df.assert_called_once()
        mock_get_non_compliance.assert_called_once_with(self.exposures, 100)
        mock_prepare_simple.assert_called_once_with(
            self.index_dates, 0, 900, self.data_end
        )
        mock_prepare_adjusted.assert_called_once_with(
            mock_simple_follow_ups, mock_non_compliance, self.deaths
        )
        mock_abspos_to_binary.assert_called_once_with(
            mock_adjusted_follow_ups, self.outcomes
        )

        # Verify results
        pd.testing.assert_series_equal(binary_outcomes, expected_binary)
        pd.testing.assert_frame_equal(follow_ups, mock_adjusted_follow_ups)

    @patch(
        "corebehrt.functional.preparation.causal.outcomes.filter_df_by_unique_values"
    )
    @patch("corebehrt.functional.preparation.causal.outcomes.get_non_compliance_abspos")
    @patch("corebehrt.functional.preparation.causal.outcomes.prepare_follow_ups_simple")
    @patch(
        "corebehrt.functional.preparation.causal.outcomes.prepare_follow_ups_adjusted"
    )
    @patch("corebehrt.functional.preparation.causal.outcomes.abspos_to_binary_outcome")
    def test_get_binary_outcome_empty_outcomes(
        self,
        mock_abspos_to_binary,
        mock_prepare_adjusted,
        mock_prepare_simple,
        mock_get_non_compliance,
        mock_filter_df,
    ):
        """Test get_binary_outcome with empty outcomes DataFrame."""
        empty_outcomes = pd.DataFrame({PID_COL: [], "abspos": []})

        # Mock return values
        mock_filter_df.return_value = self.index_date_matching
        mock_get_non_compliance.return_value = pd.Series([], dtype=float)

        mock_simple_follow_ups = pd.DataFrame(
            {
                PID_COL: [1, 2, 3, 4],
                "start_abspos": [100, 200, 300, 400],
                "end_abspos": [1000, 1100, 1200, 1300],
            }
        )
        mock_prepare_simple.return_value = mock_simple_follow_ups

        mock_adjusted_follow_ups = pd.DataFrame(
            {
                PID_COL: [1, 2, 3, 4],
                "start_abspos": [100, 200, 300, 400],
                "end_abspos": [200, 280, 400, 450],
            }
        )
        mock_prepare_adjusted.return_value = mock_adjusted_follow_ups

        expected_binary = pd.Series([0, 0, 0, 0], index=[1, 2, 3, 4])
        mock_abspos_to_binary.return_value = expected_binary

        binary_outcomes, follow_ups = get_binary_outcome(
            self.index_dates,
            empty_outcomes,
            n_hours_start_follow_up=0,
            n_hours_end_follow_up=900,
            n_hours_compliance=100,
            index_date_matching=self.index_date_matching,
            deaths=self.deaths,
            exposures=self.exposures,
            data_end=self.data_end,
        )

        pd.testing.assert_series_equal(binary_outcomes, expected_binary)
        pd.testing.assert_frame_equal(follow_ups, mock_adjusted_follow_ups)

    @patch(
        "corebehrt.functional.preparation.causal.outcomes.filter_df_by_unique_values"
    )
    @patch("corebehrt.functional.preparation.causal.outcomes.get_non_compliance_abspos")
    @patch("corebehrt.functional.preparation.causal.outcomes.prepare_follow_ups_simple")
    @patch(
        "corebehrt.functional.preparation.causal.outcomes.prepare_follow_ups_adjusted"
    )
    @patch("corebehrt.functional.preparation.causal.outcomes.abspos_to_binary_outcome")
    def test_get_binary_outcome_infinite_compliance(
        self,
        mock_abspos_to_binary,
        mock_prepare_adjusted,
        mock_prepare_simple,
        mock_get_non_compliance,
        mock_filter_df,
    ):
        """Test get_binary_outcome with infinite compliance hours."""
        # Mock return values
        mock_filter_df.return_value = self.index_date_matching
        mock_get_non_compliance.return_value = pd.Series(
            [np.inf, np.inf, np.inf, np.inf], index=[1, 2, 3, 4]
        )

        mock_simple_follow_ups = pd.DataFrame(
            {
                PID_COL: [1, 2, 3, 4],
                "start_abspos": [100, 200, 300, 400],
                "end_abspos": [np.inf, np.inf, np.inf, np.inf],
            }
        )
        mock_prepare_simple.return_value = mock_simple_follow_ups

        mock_adjusted_follow_ups = pd.DataFrame(
            {
                PID_COL: [1, 2, 3, 4],
                "start_abspos": [100, 200, 300, 400],
                "end_abspos": [np.inf, 280, np.inf, 450],  # Only deaths matter
            }
        )
        mock_prepare_adjusted.return_value = mock_adjusted_follow_ups

        expected_binary = pd.Series([1, 1, 0, 0], index=[1, 2, 3, 4])
        mock_abspos_to_binary.return_value = expected_binary

        binary_outcomes, follow_ups = get_binary_outcome(
            self.index_dates,
            self.outcomes,
            n_hours_start_follow_up=0,
            n_hours_end_follow_up=np.inf,
            n_hours_compliance=np.inf,
            index_date_matching=self.index_date_matching,
            deaths=self.deaths,
            exposures=self.exposures,
            data_end=self.data_end,
        )

        mock_get_non_compliance.assert_called_once_with(self.exposures, np.inf)
        pd.testing.assert_series_equal(binary_outcomes, expected_binary)
        pd.testing.assert_frame_equal(follow_ups, mock_adjusted_follow_ups)

    @patch(
        "corebehrt.functional.preparation.causal.outcomes.filter_df_by_unique_values"
    )
    @patch("corebehrt.functional.preparation.causal.outcomes.get_non_compliance_abspos")
    @patch("corebehrt.functional.preparation.causal.outcomes.prepare_follow_ups_simple")
    @patch(
        "corebehrt.functional.preparation.causal.outcomes.prepare_follow_ups_adjusted"
    )
    @patch("corebehrt.functional.preparation.causal.outcomes.abspos_to_binary_outcome")
    def test_get_binary_outcome_no_deaths(
        self,
        mock_abspos_to_binary,
        mock_prepare_adjusted,
        mock_prepare_simple,
        mock_get_non_compliance,
        mock_filter_df,
    ):
        """Test get_binary_outcome with no deaths."""
        deaths_no_deaths = pd.Series(
            [np.nan, np.nan, np.nan, np.nan], index=[1, 2, 3, 4]
        )

        # Mock return values
        mock_filter_df.return_value = self.index_date_matching
        mock_get_non_compliance.return_value = pd.Series(
            [200, np.nan, 400, np.nan], index=[1, 2, 3, 4]
        )

        mock_simple_follow_ups = pd.DataFrame(
            {
                PID_COL: [1, 2, 3, 4],
                "start_abspos": [100, 200, 300, 400],
                "end_abspos": [1000, 1100, 1200, 1300],
            }
        )
        mock_prepare_simple.return_value = mock_simple_follow_ups

        mock_adjusted_follow_ups = pd.DataFrame(
            {
                PID_COL: [1, 2, 3, 4],
                "start_abspos": [100, 200, 300, 400],
                "end_abspos": [200, 1100, 400, 1300],  # Only compliance matters
            }
        )
        mock_prepare_adjusted.return_value = mock_adjusted_follow_ups

        expected_binary = pd.Series([1, 1, 0, 0], index=[1, 2, 3, 4])
        mock_abspos_to_binary.return_value = expected_binary

        binary_outcomes, follow_ups = get_binary_outcome(
            self.index_dates,
            self.outcomes,
            n_hours_start_follow_up=0,
            n_hours_end_follow_up=900,
            n_hours_compliance=100,
            index_date_matching=self.index_date_matching,
            deaths=deaths_no_deaths,
            exposures=self.exposures,
            data_end=self.data_end,
        )

        mock_prepare_adjusted.assert_called_once_with(
            mock_simple_follow_ups,
            mock_get_non_compliance.return_value,
            deaths_no_deaths,
        )
        pd.testing.assert_series_equal(binary_outcomes, expected_binary)
        pd.testing.assert_frame_equal(follow_ups, mock_adjusted_follow_ups)


if __name__ == "__main__":
    unittest.main()
