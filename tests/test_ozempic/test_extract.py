import unittest

import pandas as pd

from corebehrt.constants.data import (
    BIRTH_CODE,
    CONCEPT_COL,
    PID_COL,
    TIMESTAMP_COL,
    VALUE_COL,
    AGE_COL
)
from corebehrt.ozempic.criteria.extract import extract_patient_criteria


class TestCriteriaExtraction(unittest.TestCase):
    def setUp(self):
        self.config = {
            "delays": {"days": 14, "code_groups": ["D/", "RD/"]},
            "min_age": 50,
            "criteria_definitions": {
                "type2_diabetes": {"codes": ["^D/C11.*", "^M/A10BH.*"]},
                "stroke": {"codes": ["^D/I6[3-6].*", "^D/H341.*"]},
                "HbA1c": {
                    "codes": ["(?i).*hba1c.*"],
                    "threshold": 7.0,
                    "operator": ">=",
                },
                "type1_diabetes": {"codes": ["^D/C10.*"]},
                "cancer": {
                    "codes": ["^D/C[0-9][0-9].*"],
                    "exclude_codes": ["^D/C449.*"],
                    "time_window_days": 1826,
                },
                "pregnancy_and_birth": {"codes": ["^D/O[0-9][0-9].*"]},
            },
        }

        self.index_dates = pd.DataFrame(
            {
                PID_COL: [1, 2, 3, 4, 5, 6],
                "index_date": pd.to_datetime(
                    [
                        "2023-06-01",
                        "2023-06-01",
                        "2023-06-01",
                        "2023-06-01",
                        "2023-06-01",
                        "2023-06-01",
                    ]
                ),
            }
        )

        self.df = pd.DataFrame(
            {
                PID_COL: [1, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 6],
                CONCEPT_COL: [
                    BIRTH_CODE,
                    "D/C11",
                    "D/I63",
                    "HbA1c",
                    BIRTH_CODE,
                    "D/C11",
                    BIRTH_CODE,
                    "D/C11",
                    "D/C50",
                    BIRTH_CODE,
                    "D/C10",
                    BIRTH_CODE,
                    "D/O20",
                    BIRTH_CODE,
                    "D/C11",
                    "D/I63",
                ],
                VALUE_COL: [
                    None,
                    None,
                    None,
                    7.5,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ],
                TIMESTAMP_COL: pd.to_datetime(
                    [
                        "1968-01-01",
                        "2023-05-15",
                        "2023-05-20",
                        "2023-05-28",
                        "1978-01-01",
                        "2023-05-15",
                        "1965-01-01",
                        "2023-05-15",
                        "2021-06-15",
                        "1965-01-01",
                        "2023-05-15",
                        "1968-01-01",
                        "2023-05-20",
                        "1968-01-01",
                        "2023-05-15",
                        "2023-06-10",
                    ]
                ),
            }
        )

    def test_patient_1_included(self):
        patients = extract_patient_criteria(self.df, self.index_dates, self.config)
        patient = patients[1]
        self.assertTrue(patient.criteria_flags["type2_diabetes"])
        self.assertTrue(patient.criteria_flags["stroke"])
        self.assertTrue(patient.criteria_flags["HbA1c"])
        self.assertEqual(patient.values["HbA1c"], 7.5)

    def test_patient_2_too_young(self):
        patients = extract_patient_criteria(self.df, self.index_dates, self.config)
        patient = patients[2]
        self.assertTrue(patient.criteria_flags["type2_diabetes"])
        self.assertFalse(patient.criteria_flags["stroke"])
        self.assertFalse(patient.criteria_flags["HbA1c"])
        self.assertLess(patient.age, self.config["min_age"])

    def test_patient_3_recent_cancer(self):
        patients = extract_patient_criteria(self.df, self.index_dates, self.config)
        patient = patients[3]
        self.assertTrue(patient.criteria_flags["cancer"])

    def test_patient_4_type1_diabetes(self):
        patients = extract_patient_criteria(self.df, self.index_dates, self.config)
        patient = patients[4]
        self.assertTrue(patient.criteria_flags["type1_diabetes"])

    def test_patient_5_recent_pregnancy(self):
        patients = extract_patient_criteria(self.df, self.index_dates, self.config)
        patient = patients[5]
        self.assertTrue(patient.criteria_flags["pregnancy_and_birth"])

    def test_patient_6_stroke_within_delay(self):
        patients = extract_patient_criteria(self.df, self.index_dates, self.config)
        patient = patients[6]
        self.assertTrue(patient.criteria_flags["stroke"])


if __name__ == "__main__":
    unittest.main()
