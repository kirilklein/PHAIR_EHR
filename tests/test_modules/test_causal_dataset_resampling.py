"""Tests for multiplicity-preserving patient resampling."""

import unittest

from corebehrt.modules.preparation.causal.dataset import (
    CausalPatientData,
    CausalPatientDataset,
)


class TestCausalPatientResampling(unittest.TestCase):
    def test_resample_preserves_order_and_duplicates(self):
        patients = [
            CausalPatientData(pid=1, concepts=[], abspos=[], segments=[], ages=[]),
            CausalPatientData(pid=2, concepts=[], abspos=[], segments=[], ages=[]),
            CausalPatientData(pid=3, concepts=[], abspos=[], segments=[], ages=[]),
        ]
        dataset = CausalPatientDataset(patients)

        resampled = dataset.resample_by_pids([2, 1, 2, 3, 2])

        self.assertEqual(resampled.get_pids(), [2, 1, 2, 3, 2])

    def test_resample_rejects_unknown_patient(self):
        dataset = CausalPatientDataset(
            [CausalPatientData(pid=1, concepts=[], abspos=[], segments=[], ages=[])]
        )

        with self.assertRaises(KeyError):
            dataset.resample_by_pids([1, 2])


if __name__ == "__main__":
    unittest.main()
