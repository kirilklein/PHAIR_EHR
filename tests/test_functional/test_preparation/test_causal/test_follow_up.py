import unittest
import pandas as pd
import numpy as np
from corebehrt.functional.preparation.causal.follow_up import (
    prepare_follow_ups_adjusted,
    prepare_follow_ups_simple,
    get_combined_follow_ups,
)
from corebehrt.modules.preparation.causal.config import OutcomeConfig
from corebehrt.constants.causal.data import (
    START_COL,
    END_COL,
    NON_COMPLIANCE_COL,
    DEATH_COL,
    START_TIME_COL,
    END_TIME_COL,
    CONTROL_PID_COL,
    EXPOSED_PID_COL,
)
from corebehrt.constants.data import PID_COL, ABSPOS_COL, TIMESTAMP_COL
from corebehrt.functional.utils.time import get_hours_since_epoch


class TestGetCombinedFollowUps(unittest.TestCase):
    """Test cases for the get_combined_follow_ups function."""

    def setUp(self):
        self.index_dates = pd.DataFrame(
            {
                PID_COL: [1, 2, 3, 4, 5, 6],
                TIMESTAMP_COL: pd.to_datetime(
                    [
                        "2023-01-01",
                        "2023-01-01",
                        "2023-01-02",
                        "2023-01-02",
                        "2023-01-03",
                        "2023-01-03",
                    ]
                ),
            }
        )
        self.index_dates[ABSPOS_COL] = get_hours_since_epoch(
            self.index_dates[TIMESTAMP_COL]
        )
        pid_to_abspos = self.index_dates.set_index(PID_COL)[ABSPOS_COL]
        self.index_date_matching = pd.DataFrame(
            {
                CONTROL_PID_COL: [2, 4, 6],
                EXPOSED_PID_COL: [1, 3, 5],
            }
        )
        self.deaths = pd.Series(
            [np.nan, np.nan, 200, np.nan, 300, np.nan], index=[1, 2, 3, 4, 5, 6]
        )
        pids_with_death = self.deaths.dropna().index
        death_abspos = self.deaths.dropna()
        death_index_abspos = pids_with_death.map(pid_to_abspos)
        self.deaths.loc[pids_with_death] = death_abspos + death_index_abspos

        self.exposures = pd.DataFrame(
            {
                PID_COL: [1, 1, 3, 3, 5, 5],
                ABSPOS_COL: [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            }
        )
        self.exposures[ABSPOS_COL] = (
            self.exposures[PID_COL].map(pid_to_abspos) + self.exposures[ABSPOS_COL]
        )
        self.data_end = pd.Timestamp("2024-01-01")

    def test_basic_follow_up(self):
        """Test basic follow-up without group-wise adjustment."""
        cfg = OutcomeConfig(
            n_hours_start_follow_up=0,
            n_hours_end_follow_up=1000,
            n_hours_compliance=10,
            group_wise_follow_up=False,
            delay_death_hours=0,
        )
        result = get_combined_follow_ups(
            self.index_dates,
            self.index_date_matching,
            self.deaths,
            self.exposures,
            self.data_end,
            cfg,
        )
        self.assertEqual(len(result), 6)

        pid_to_abspos = self.index_dates.set_index(PID_COL)[ABSPOS_COL]

        # Patient 1: non-compliance at last_exposure(20) + 10 = 30 (relative). Absolute is abspos+30. End is min(abspos+1000, abspos+30)
        end_p1 = pid_to_abspos[1] + 20 + 10
        np.testing.assert_allclose(
            result.loc[result[PID_COL] == 1, END_COL].iloc[0], end_p1
        )

        # Patient 2: no exposure, so non-compliance is inf. end is abspos+1000
        end_p2 = pid_to_abspos[2] + 1000
        np.testing.assert_allclose(
            result.loc[result[PID_COL] == 2, END_COL].iloc[0], end_p2
        )

        # Patient 3: death at 200 (relative), non-compliance at 40+10=50 (relative). end is min(abspos+1000, abspos+200, abspos+50)
        end_p3 = pid_to_abspos[3] + 50
        np.testing.assert_allclose(
            result.loc[result[PID_COL] == 3, END_COL].iloc[0], end_p3
        )

        # Patient 4: no exposure, no death. end is abspos+1000
        end_p4 = pid_to_abspos[4] + 1000
        np.testing.assert_allclose(
            result.loc[result[PID_COL] == 4, END_COL].iloc[0], end_p4
        )

        # Patient 5: death at 300 (relative), non-compliance at 60+10=70 (relative). end is min(abspos+1000, abspos+300, abspos+70)
        end_p5 = pid_to_abspos[5] + 70
        np.testing.assert_allclose(
            result.loc[result[PID_COL] == 5, END_COL].iloc[0], end_p5
        )

        # Patient 6: no exposure, no death. end is abspos+1000
        end_p6 = pid_to_abspos[6] + 1000
        np.testing.assert_allclose(
            result.loc[result[PID_COL] == 6, END_COL].iloc[0], end_p6
        )

    def test_group_wise_follow_up(self):
        """Test with group-wise follow-up adjustment."""
        cfg = OutcomeConfig(
            n_hours_start_follow_up=0,
            n_hours_end_follow_up=1000,
            n_hours_compliance=10,
            group_wise_follow_up=True,
            delay_death_hours=0,
        )
        result = get_combined_follow_ups(
            self.index_dates,
            self.index_date_matching,
            self.deaths,
            self.exposures,
            self.data_end,
            cfg,
        )

        pid_to_abspos = self.index_dates.set_index(PID_COL)[ABSPOS_COL]

        end_p1 = pid_to_abspos[1] + 20 + 10
        end_p2 = pid_to_abspos[2] + 1000
        end_p3 = min(
            pid_to_abspos[3] + 1000, self.deaths.loc[3], pid_to_abspos[3] + 40 + 10
        )
        end_p4 = pid_to_abspos[4] + 1000
        end_p5 = min(
            pid_to_abspos[5] + 1000, self.deaths.loc[5], pid_to_abspos[5] + 60 + 10
        )
        end_p6 = pid_to_abspos[6] + 1000

        expected_group1_end = min(end_p1, end_p2)
        expected_group2_end = min(end_p3, end_p4)
        expected_group3_end = min(end_p5, end_p6)

        self.assertAlmostEqual(
            result.loc[result[PID_COL] == 1, END_COL].iloc[0], expected_group1_end
        )
        self.assertAlmostEqual(
            result.loc[result[PID_COL] == 2, END_COL].iloc[0], expected_group1_end
        )
        self.assertAlmostEqual(
            result.loc[result[PID_COL] == 3, END_COL].iloc[0], expected_group2_end
        )
        self.assertAlmostEqual(
            result.loc[result[PID_COL] == 4, END_COL].iloc[0], expected_group2_end
        )
        self.assertAlmostEqual(
            result.loc[result[PID_COL] == 5, END_COL].iloc[0], expected_group3_end
        )
        self.assertAlmostEqual(
            result.loc[result[PID_COL] == 6, END_COL].iloc[0], expected_group3_end
        )

    def test_group_wise_follow_up_no_matching_raises_error(self):
        """Test that group-wise follow-up raises an error if no matching is provided."""
        cfg = OutcomeConfig(
            n_hours_start_follow_up=0,
            n_hours_end_follow_up=1000,
            n_hours_compliance=10,
            group_wise_follow_up=True,
            delay_death_hours=0,
        )
        with self.assertRaises(ValueError):
            get_combined_follow_ups(
                self.index_dates, None, self.deaths, self.exposures, self.data_end, cfg
            )

    def test_no_censoring(self):
        """Test with no censoring events (no deaths, no non-compliance)."""
        cfg = OutcomeConfig(
            n_hours_start_follow_up=0,
            n_hours_end_follow_up=500,
            n_hours_compliance=np.inf,
            group_wise_follow_up=False,
            delay_death_hours=0,
        )
        deaths = pd.Series(np.nan, index=self.index_dates[PID_COL])
        exposures = pd.DataFrame({PID_COL: [], ABSPOS_COL: []})
        result = get_combined_follow_ups(
            self.index_dates,
            self.index_date_matching,
            deaths,
            exposures,
            self.data_end,
            cfg,
        )

        expected_ends = self.index_dates.set_index(PID_COL)[ABSPOS_COL] + 500
        pd.testing.assert_series_equal(
            result.set_index(PID_COL)[END_COL],
            expected_ends,
            check_names=False,
        )

    def test_delay_death_hours(self):
        """Test the delay_death_hours parameter."""
        cfg = OutcomeConfig(
            n_hours_start_follow_up=0,
            n_hours_end_follow_up=1000,
            n_hours_compliance=np.inf,
            group_wise_follow_up=False,
            delay_death_hours=50,
        )

        # death for patient 3 is 200 (relative), delayed is 250.
        # end is min(abspos+1000, abspos+250)
        result = get_combined_follow_ups(
            self.index_dates,
            self.index_date_matching,
            self.deaths,
            pd.DataFrame({PID_COL: [], ABSPOS_COL: []}),
            self.data_end,
            cfg,
        )
        self.assertAlmostEqual(
            result.loc[result[PID_COL] == 3, END_COL].iloc[0], self.deaths.loc[3] + 50
        )


class TestPrepareFollowUpsAdjusted(unittest.TestCase):
    """Test cases for the prepare_follow_ups_adjusted function."""

    def test_prepare_follow_ups_adjusted_basic(self):
        """Test the prepare_follow_ups_adjusted function with complete data."""
        # 1. Setup test data
        follow_ups = pd.DataFrame(
            {
                PID_COL: [1, 2, 3, 4],
                START_COL: [10.0, 20.0, 30.0, 40.0],
                END_COL: [100.0, 200.0, 300.0, 400.0],
            }
        )

        non_compliance_abspos = {
            1: 80.0,  # Patient 1 stops compliance at 80
            2: 250.0,  # Patient 2 stops compliance at 250
            3: 150.0,  # Patient 3 stops compliance at 150
            4: 500.0,  # Patient 4 stops compliance at 500
        }

        deaths = {
            1: np.nan,  # Patient 1 doesn't die
            2: 180.0,  # Patient 2 dies at 180
            3: np.nan,  # Patient 3 doesn't die
            4: 350.0,  # Patient 4 dies at 350
        }

        # 2. Execute function
        result = prepare_follow_ups_adjusted(follow_ups, non_compliance_abspos, deaths)

        # 3. Expected results (individual patient minimums):
        # Patient 1: min(100, 80, inf) = 80
        # Patient 2: min(200, 250, 180) = 180
        # Patient 3: min(300, 150, inf) = 150
        # Patient 4: min(400, 500, 350) = 350

        expected_end_values = [80.0, 180.0, 150.0, 350.0]

        # 4. Assertions
        self.assertEqual(len(result), 4, f"Expected 4 rows, got {len(result)}")
        pd.testing.assert_series_equal(
            result[NON_COMPLIANCE_COL],
            pd.Series([80.0, 250.0, 150.0, 500.0], name=NON_COMPLIANCE_COL),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            result[DEATH_COL],
            pd.Series([np.nan, 180.0, np.nan, 350.0], name=DEATH_COL),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            result[END_COL],
            pd.Series(expected_end_values, name=END_COL),
            check_names=False,
        )

    def test_prepare_follow_ups_adjusted_missing_data(self):
        """Test the function with missing data (NaN values)."""
        # 1. Setup test data with missing values
        follow_ups = pd.DataFrame(
            {PID_COL: [1, 2], START_COL: [10.0, 20.0], END_COL: [100.0, 200.0]}
        )

        non_compliance_abspos = {
            1: 80.0,  # Patient 1 has compliance data
            # Patient 2 missing from non_compliance_abspos
        }

        deaths = {
            1: np.nan,  # Patient 1 doesn't die
            # Patient 2 missing from deaths
        }

        # 2. Execute function
        result = prepare_follow_ups_adjusted(follow_ups, non_compliance_abspos, deaths)

        # 3. Expected results (individual patient minimums):
        # Patient 1: min(100, 80, inf) = 80
        # Patient 2: min(200, NaN, NaN) = 200

        # 4. Assertions
        self.assertTrue(
            pd.isna(result.loc[result[PID_COL] == 2, NON_COMPLIANCE_COL].iloc[0]),
            "Missing non-compliance should be NaN",
        )
        self.assertTrue(
            pd.isna(result.loc[result[PID_COL] == 2, DEATH_COL].iloc[0]),
            "Missing death should be NaN",
        )
        pd.testing.assert_series_equal(
            result[END_COL], pd.Series([80.0, 200.0], name=END_COL), check_names=False
        )

    def test_prepare_follow_ups_adjusted_empty_input(self):
        """Test the function with empty input data."""
        # 1. Setup empty test data
        follow_ups = pd.DataFrame(columns=[PID_COL, START_COL, END_COL])

        non_compliance_abspos = {}
        deaths = {}

        # 2. Execute function
        result = prepare_follow_ups_adjusted(follow_ups, non_compliance_abspos, deaths)

        # 3. Assertions
        self.assertEqual(len(result), 0, "Empty input should return empty result")
        self.assertTrue(
            NON_COMPLIANCE_COL in result.columns, "Should have non_compliance column"
        )
        self.assertTrue(DEATH_COL in result.columns, "Should have death column")

    def test_prepare_follow_ups_adjusted_single_group(self):
        """Test the function with individual patient minimums."""
        # 1. Setup test data
        follow_ups = pd.DataFrame(
            {PID_COL: [1, 2], START_COL: [10.0, 20.0], END_COL: [100.0, 200.0]}
        )

        non_compliance_abspos = {1: 80.0, 2: 150.0}
        deaths = {1: np.nan, 2: 120.0}

        # 2. Execute function
        result = prepare_follow_ups_adjusted(follow_ups, non_compliance_abspos, deaths)

        # 3. Expected results (individual patient minimums):
        # Patient 1: min(100, 80, inf) = 80
        # Patient 2: min(200, 150, 120) = 120

        # 4. Assertions
        expected_end_values = [80.0, 120.0]
        pd.testing.assert_series_equal(
            result[END_COL],
            pd.Series(expected_end_values, name=END_COL),
            check_names=False,
        )


class TestPrepareFollowUpsSimple(unittest.TestCase):
    """Test cases for the prepare_follow_ups_simple function."""

    def test_prepare_follow_ups_simple_basic(self):
        """Test basic functionality with positive follow-up hours."""
        # 1. Setup test data
        index_dates = pd.DataFrame(
            {
                PID_COL: [1, 2, 3],
                TIMESTAMP_COL: [
                    pd.Timestamp("2023-01-01 12:00:00"),
                    pd.Timestamp("2023-01-02 12:00:00"),
                    pd.Timestamp("2023-01-03 12:00:00"),
                ],
                ABSPOS_COL: [100.0, 200.0, 300.0],
                "other_col": [
                    "a",
                    "b",
                    "c",
                ],  # Additional column to ensure it's preserved
            }
        )

        n_hours_start_follow_up = 24
        n_hours_end_follow_up = 168  # 1 week
        data_end = pd.Timestamp("2023-12-31 23:59:59")

        # 2. Execute function
        result = prepare_follow_ups_simple(
            index_dates, n_hours_start_follow_up, n_hours_end_follow_up, data_end
        )

        # 3. Assertions
        self.assertEqual(len(result), 3, "Should have same number of rows")
        self.assertNotIn(ABSPOS_COL, result.columns, "abspos column should be dropped")
        self.assertIn(START_COL, result.columns, "Should have start column")
        self.assertIn(END_COL, result.columns, "Should have end column")
        self.assertIn("other_col", result.columns, "Other columns should be preserved")

        # Check that start and end times are calculated correctly from timestamps
        self.assertTrue(all(result[START_COL] > 0), "Start times should be positive")
        self.assertTrue(
            all(result[END_COL] > result[START_COL]),
            "End times should be after start times",
        )

    def test_prepare_follow_ups_simple_zero_hours(self):
        """Test with zero follow-up hours."""
        # 1. Setup test data
        index_dates = pd.DataFrame(
            {
                PID_COL: [1, 2],
                TIMESTAMP_COL: [
                    pd.Timestamp("2023-01-01 12:00:00"),
                    pd.Timestamp("2023-01-02 12:00:00"),
                ],
                ABSPOS_COL: [100.0, 200.0],
            }
        )

        n_hours_start_follow_up = 0
        n_hours_end_follow_up = 0
        data_end = pd.Timestamp("2023-12-31 23:59:59")

        # 2. Execute function
        result = prepare_follow_ups_simple(
            index_dates, n_hours_start_follow_up, n_hours_end_follow_up, data_end
        )

        # 3. Assertions
        # Start and end should be the same when both are 0 hours
        pd.testing.assert_series_equal(
            result[START_COL],
            result[END_COL],
            check_names=False,
        )

    def test_prepare_follow_ups_simple_negative_hours(self):
        """Test with negative follow-up hours."""
        # 1. Setup test data
        index_dates = pd.DataFrame(
            {
                PID_COL: [1, 2],
                TIMESTAMP_COL: [
                    pd.Timestamp("2023-01-01 12:00:00"),
                    pd.Timestamp("2023-01-02 12:00:00"),
                ],
                ABSPOS_COL: [100.0, 200.0],
            }
        )

        n_hours_start_follow_up = -12
        n_hours_end_follow_up = -24
        data_end = pd.Timestamp("2023-12-31 23:59:59")

        # 2. Execute function
        result = prepare_follow_ups_simple(
            index_dates, n_hours_start_follow_up, n_hours_end_follow_up, data_end
        )

        # 3. Assertions
        # End should be before start when end hours are more negative
        self.assertTrue(
            all(result[END_COL] < result[START_COL]),
            "End times should be before start times with negative hours",
        )

    def test_prepare_follow_ups_simple_none_end_hours(self):
        """Test with None end hours to use data_end."""
        # 1. Setup test data
        index_dates = pd.DataFrame(
            {
                PID_COL: [1, 2],
                TIMESTAMP_COL: [
                    pd.Timestamp("2023-01-01 12:00:00"),
                    pd.Timestamp("2023-01-02 12:00:00"),
                ],
                ABSPOS_COL: [100.0, 200.0],
            }
        )

        n_hours_start_follow_up = 24
        n_hours_end_follow_up = None  # Should use data_end
        data_end = pd.Timestamp("2023-12-31 23:59:59")

        # 2. Execute function
        result = prepare_follow_ups_simple(
            index_dates, n_hours_start_follow_up, n_hours_end_follow_up, data_end
        )

        # 3. Assertions
        # All end times should be the same (data_end converted to hours since epoch)
        expected_end_hours = result[END_COL].iloc[0]  # Get the first value
        self.assertTrue(
            all(result[END_COL] == expected_end_hours),
            "All end times should be the same when using data_end",
        )
        self.assertTrue(
            all(result[END_COL] > result[START_COL]),
            "End times should be after start times",
        )

    def test_prepare_follow_ups_simple_empty_input(self):
        """Test with empty DataFrame."""
        # 1. Setup empty test data
        index_dates = pd.DataFrame(columns=[PID_COL, TIMESTAMP_COL, ABSPOS_COL])

        n_hours_start_follow_up = 24
        n_hours_end_follow_up = 168
        data_end = pd.Timestamp("2023-12-31 23:59:59")

        # 2. Execute function
        result = prepare_follow_ups_simple(
            index_dates, n_hours_start_follow_up, n_hours_end_follow_up, data_end
        )

        # 3. Assertions
        self.assertEqual(len(result), 0, "Empty input should return empty result")
        self.assertIn(START_COL, result.columns, "Should have start column")
        self.assertIn(END_COL, result.columns, "Should have end column")
        self.assertNotIn(ABSPOS_COL, result.columns, "abspos column should be dropped")

    def test_prepare_follow_ups_simple_single_row(self):
        """Test with single row input."""
        # 1. Setup test data
        index_dates = pd.DataFrame(
            {
                PID_COL: [1],
                TIMESTAMP_COL: [pd.Timestamp("2023-01-01 12:00:00")],
                ABSPOS_COL: [500.0],
            }
        )

        n_hours_start_follow_up = 48
        n_hours_end_follow_up = 720  # 30 days
        data_end = pd.Timestamp("2023-12-31 23:59:59")

        # 2. Execute function
        result = prepare_follow_ups_simple(
            index_dates, n_hours_start_follow_up, n_hours_end_follow_up, data_end
        )

        # 3. Assertions
        self.assertEqual(len(result), 1, "Should have one row")
        self.assertTrue(
            result.iloc[0][END_COL] > result.iloc[0][START_COL],
            "End time should be after start time",
        )

    def test_prepare_follow_ups_simple_preserves_other_columns(self):
        """Test that other columns are preserved."""
        # 1. Setup test data with multiple columns
        index_dates = pd.DataFrame(
            {
                PID_COL: [1, 2],
                TIMESTAMP_COL: [
                    pd.Timestamp("2023-01-01 12:00:00"),
                    pd.Timestamp("2023-01-02 12:00:00"),
                ],
                ABSPOS_COL: [100.0, 200.0],
                "other_col1": [25, 30],
                "other_col2": ["M", "F"],
            }
        )

        n_hours_start_follow_up = 24
        n_hours_end_follow_up = 168
        data_end = pd.Timestamp("2023-12-31 23:59:59")

        # 2. Execute function
        result = prepare_follow_ups_simple(
            index_dates, n_hours_start_follow_up, n_hours_end_follow_up, data_end
        )

        # 3. Assertions
        expected_columns = {
            PID_COL,
            START_COL,
            END_COL,
            START_TIME_COL,
            END_TIME_COL,
            TIMESTAMP_COL,
            "other_col1",
            "other_col2",
        }
        self.assertEqual(
            set(result.columns),
            expected_columns,
            "All columns should be preserved except abspos",
        )

        # Check that original data is preserved
        pd.testing.assert_series_equal(
            result["other_col1"],
            pd.Series([25, 30], name="other_col1"),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            result["other_col2"],
            pd.Series(["M", "F"], name="other_col2"),
            check_names=False,
        )

    def test_prepare_follow_ups_simple_float_hours(self):
        """Test with float values in hours."""
        # 1. Setup test data with timestamps
        index_dates = pd.DataFrame(
            {
                PID_COL: [1, 2],
                TIMESTAMP_COL: [
                    pd.Timestamp("2023-01-01 12:00:00"),
                    pd.Timestamp("2023-01-02 12:00:00"),
                ],
                ABSPOS_COL: [100.5, 200.75],
            }
        )

        n_hours_start_follow_up = 12.5
        n_hours_end_follow_up = 48.25
        data_end = pd.Timestamp("2023-12-31 23:59:59")

        # 2. Execute function
        result = prepare_follow_ups_simple(
            index_dates, n_hours_start_follow_up, n_hours_end_follow_up, data_end
        )

        # 3. Assertions
        # Check that the time differences are correct
        time_diff = result[END_COL] - result[START_COL]
        expected_diff = 48.25 - 12.5  # 35.75 hours
        self.assertTrue(
            all(abs(time_diff - expected_diff) < 0.001),
            "Time differences should match expected hours",
        )


class TestGetCombinedFollowUpsRigorous(unittest.TestCase):
    """
    A rigorous test suite for get_combined_follow_ups, specifically targeting
    the group-wise follow-up logic and its edge cases.
    """

    def setUp(self):
        """Set up data designed to expose bugs in group-wise follow-up logic."""
        self.index_dates = pd.DataFrame(
            {
                PID_COL: [101, 102, 201, 202, 301],
                TIMESTAMP_COL: pd.to_datetime(
                    [
                        "2023-01-10",  # Patient 101 (Exposed): Early index date
                        "2023-05-20",  # Patient 102 (Control): Late index date, matched to 101
                        "2023-02-01",  # Patient 201 (Exposed)
                        "2023-02-15",  # Patient 202 (Control), matched to 201
                        "2023-03-01",  # Patient 301: Unmatched patient
                    ]
                ),
            }
        )
        self.index_dates[ABSPOS_COL] = get_hours_since_epoch(
            self.index_dates[TIMESTAMP_COL]
        )

        # This matching is crucial for the bug: 101 and 102 have very different index dates
        self.index_date_matching = pd.DataFrame(
            {
                CONTROL_PID_COL: [102, 202],
                EXPOSED_PID_COL: [101, 201],
            }
        )

        # --- Censoring Events ---
        pid_to_abspos = self.index_dates.set_index(PID_COL)[ABSPOS_COL]

        # Patient 101 (Exposed) is censored early by non-compliance
        # Patient 202 (Control) is censored early by death
        self.exposures = pd.DataFrame(
            {
                PID_COL: [101, 201],
                # Last exposure for 101 is 30 days after index
                ABSPOS_COL: [
                    pid_to_abspos[101] + 30 * 24,
                    pid_to_abspos[201] + 500 * 24,
                ],
            }
        )

        self.deaths = pd.Series(
            [np.nan, np.nan, np.nan, pid_to_abspos[202] + 45 * 24, np.nan],
            index=[101, 102, 201, 202, 301],
        )
        self.data_end = pd.Timestamp("2025-01-01")

    def test_group_wise_follow_up_avoids_negative_duration_bug(self):
        """
        THE KEY TEST: Ensures group-wise follow-up with disparate index dates
        does not produce negative follow-up durations. This test will FAIL
        with the old, buggy implementation.
        """
        cfg = OutcomeConfig(
            n_hours_start_follow_up=24,  # Start follow-up 1 day after index
            n_hours_end_follow_up=365 * 24,  # Max follow-up is 1 year
            n_hours_compliance=10 * 24,  # Non-compliance after 10 days of no exposure
            group_wise_follow_up=True,
            delay_death_hours=0,
        )

        result = get_combined_follow_ups(
            self.index_dates,
            self.index_date_matching,
            self.deaths,
            self.exposures,
            self.data_end,
            cfg,
        )

        # --- Primary Assertion: No negative follow-up times ---
        durations = result[END_COL] - result[START_COL]
        self.assertTrue(
            (durations >= 0).all(), "Follow-up duration should never be negative."
        )

        # --- Detailed Calculation and Assertions for Group 1 (101, 102) ---
        p101 = result[result[PID_COL] == 101].iloc[0]
        p102 = result[result[PID_COL] == 102].iloc[0]

        # Individual durations BEFORE group minimization:
        # P101: start=idx+1d. non-compliance=idx+30d+10d = idx+40d. Max_fu=idx+365d.
        #       Censored by non-compliance. Duration = 40d - 1d = 39 days.
        duration_101 = (30 + 10 - 1) * 24

        # The minimum duration for the group is 39 days (from P101).
        min_duration_group1 = duration_101

        # Both patients in the group should now have a follow-up of 39 days from THEIR OWN start time.
        self.assertAlmostEqual(p101[END_COL] - p101[START_COL], min_duration_group1)
        self.assertAlmostEqual(p102[END_COL] - p102[START_COL], min_duration_group1)

        # Explicitly check P102's end date
        expected_end_102 = p102[START_COL] + min_duration_group1
        self.assertAlmostEqual(p102[END_COL], expected_end_102)

    def test_censoring_before_followup_start_produces_zero_duration(self):
        """
        Tests that if a censoring event occurs before the follow-up starts,
        the duration is correctly set to zero (and not negative).
        """
        cfg = OutcomeConfig(
            n_hours_start_follow_up=60 * 24,  # Start follow-up 60 days after index
            n_hours_end_follow_up=365 * 24,
            n_hours_compliance=np.inf,
            group_wise_follow_up=True,
            delay_death_hours=0,
        )
        # P202 dies 45 days after index, but follow-up starts at 60 days.
        # This patient should have a follow-up duration of 0.

        result = get_combined_follow_ups(
            self.index_dates,
            self.index_date_matching,
            self.deaths,
            self.exposures,
            self.data_end,
            cfg,
        )

        p202 = result[result[PID_COL] == 202].iloc[0]
        duration_202 = p202[END_COL] - p202[START_COL]

        self.assertAlmostEqual(
            duration_202, 0, "Duration must be 0 if censoring is before start."
        )

        # Now check the matched patient, P201
        p201 = result[result[PID_COL] == 201].iloc[0]
        duration_201 = p201[END_COL] - p201[START_COL]

        # Since P202's duration is 0, P201's duration must also be 0 due to group minimization.
        self.assertAlmostEqual(
            duration_201, 0, "Matched patient's duration should also be 0."
        )

    def test_unmatched_patient_is_unaffected_by_grouping(self):
        """
        Tests that an unmatched patient's follow-up is not altered in group-wise mode.
        """
        cfg = OutcomeConfig(
            n_hours_start_follow_up=0,
            n_hours_end_follow_up=100 * 24,
            n_hours_compliance=np.inf,  # No non-compliance censoring
            group_wise_follow_up=True,
            delay_death_hours=0,
        )

        result = get_combined_follow_ups(
            self.index_dates,
            self.index_date_matching,
            self.deaths,
            pd.DataFrame({PID_COL: [], ABSPOS_COL: []}),  # No exposure censoring
            self.data_end,
            cfg,
        )

        p301 = result[result[PID_COL] == 301].iloc[0]
        duration_301 = p301[END_COL] - p301[START_COL]

        # Patient 301 has no censoring and is unmatched, so should have the max duration.
        self.assertAlmostEqual(duration_301, 100 * 24)


if __name__ == "__main__":
    unittest.main()
