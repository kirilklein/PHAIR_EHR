import unittest
from datetime import datetime
import numpy as np
import pandas as pd

from corebehrt.modules.cohort_handling.advanced.data.patient import (
    Patient,
    patients_to_dataframe,
)
from corebehrt.constants.data import PID_COL, AGE_COL
from corebehrt.constants.causal.data import INDEX_DATE


class TestPatientsToDataFrame(unittest.TestCase):
    def setUp(self):
        # Create test patients with various combinations of flags and values
        self.patients = {}

        # Patient 1: Has all flags and values
        p1 = Patient(1, datetime(2020, 1, 1))
        p1.age = 65
        p1.criteria_flags = {"diabetes": True, "hypertension": False}
        p1.values = {"bmi": 25.5, "hba1c": 7.2}
        self.patients[1] = p1

        # Patient 2: Has some flags but no values
        p2 = Patient(2, datetime(2020, 2, 1))
        p2.age = 55
        p2.criteria_flags = {"diabetes": False}
        self.patients[2] = p2

        # Patient 3: Has some values but no flags
        p3 = Patient(3, datetime(2020, 3, 1))
        p3.age = 45
        p3.values = {"bmi": 30.0}
        self.patients[3] = p3

    def test_basic_conversion(self):
        """Test basic conversion of patients to DataFrame"""
        df = patients_to_dataframe(self.patients)

        # Check that we have the correct number of rows
        self.assertEqual(len(df), 3)

        # Check that we have all expected columns
        expected_columns = [
            PID_COL,
            INDEX_DATE,
            AGE_COL,
            "diabetes",
            "hypertension",
            "value_bmi",
            "value_hba1c",
        ]
        self.assertListEqual(list(df.columns), expected_columns)

    def test_column_values(self):
        """Test that values are correctly populated in the DataFrame"""
        df = patients_to_dataframe(self.patients)

        # Check Patient 1's values
        p1_row = df[df[PID_COL] == 1].iloc[0]
        self.assertEqual(p1_row["diabetes"], True)
        self.assertEqual(p1_row["hypertension"], False)
        self.assertEqual(p1_row["value_bmi"], 25.5)
        self.assertEqual(p1_row["value_hba1c"], 7.2)
        self.assertEqual(p1_row[AGE_COL], 65)

        # Check Patient 2's values (missing values should be np.nan)
        p2_row = df[df[PID_COL] == 2].iloc[0]
        self.assertEqual(p2_row["diabetes"], False)
        self.assertEqual(p2_row["hypertension"], False)
        self.assertTrue(pd.isna(p2_row["value_bmi"]))
        self.assertTrue(pd.isna(p2_row["value_hba1c"]))
        self.assertEqual(p2_row[AGE_COL], 55)

        # Check Patient 3's values
        p3_row = df[df[PID_COL] == 3].iloc[0]
        self.assertEqual(p3_row["value_bmi"], 30.0)
        self.assertTrue(pd.isna(p3_row["value_hba1c"]))
        self.assertEqual(p3_row["diabetes"], False)  # Default value for missing flag

    def test_empty_patients_dict(self):
        """Test conversion of empty patients dictionary"""
        df = patients_to_dataframe({})
        self.assertEqual(len(df), 0)
        self.assertListEqual(list(df.columns), [PID_COL, INDEX_DATE, AGE_COL])

    def test_index_date_conversion(self):
        """Test that index dates are correctly preserved"""
        df = patients_to_dataframe(self.patients)
        expected_dates = {
            1: datetime(2020, 1, 1),
            2: datetime(2020, 2, 1),
            3: datetime(2020, 3, 1),
        }

        for pid, expected_date in expected_dates.items():
            actual_date = df[df[PID_COL] == pid][INDEX_DATE].iloc[0]
            self.assertEqual(actual_date, expected_date)

    def test_column_order(self):
        """Test that columns are ordered correctly"""
        df = patients_to_dataframe(self.patients)

        # Check that required columns come first in correct order
        self.assertEqual(df.columns[0], PID_COL)
        self.assertEqual(df.columns[1], INDEX_DATE)
        self.assertEqual(df.columns[2], AGE_COL)

        # Check that flag columns come before value columns
        flag_cols = [
            col
            for col in df.columns
            if not col.startswith("value_")
            and col not in [PID_COL, INDEX_DATE, AGE_COL]
        ]
        value_cols = [col for col in df.columns if col.startswith("value_")]

        # Check that flag columns are sorted
        self.assertListEqual(flag_cols, sorted(flag_cols))

        # Check that value columns are sorted
        self.assertListEqual(value_cols, sorted(value_cols))

        # Check that all flag columns come before value columns
        last_flag_idx = max(df.columns.get_loc(col) for col in flag_cols)
        first_value_idx = min(df.columns.get_loc(col) for col in value_cols)
        self.assertTrue(last_flag_idx < first_value_idx)
