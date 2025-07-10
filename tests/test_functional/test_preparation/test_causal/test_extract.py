import unittest

import numpy as np

from corebehrt.functional.preparation.causal.extract import extract_death
from corebehrt.modules.preparation.causal.dataset import CausalPatientData


class TestExtractDeath(unittest.TestCase):
    """Test cases for the extract_death function."""

    def test_extract_death_simple(self):
        """Test extracting death when death token is present."""
        # 1. Setup test data
        patient = CausalPatientData(
            pid=1,
            concepts=[1, 2, 3, 4, 5],
            abspos=[1, 2, 3, 4, 5],
            segments=[1, 2, 3, 4, 5],
            ages=[1, 2, 3, 4, 5],
        )
        death_token = 5

        # 2. Execute function
        result = extract_death(patient, death_token)

        # 3. Assertions
        self.assertEqual(
            result, 5, "Should return the abspos value corresponding to death_token"
        )

    def test_extract_death_not_present(self):
        """Test extracting death when death token is not present."""
        # 1. Setup test data
        patient = CausalPatientData(
            pid=1,
            concepts=[1, 2, 3, 4, 5],
            abspos=[1, 2, 3, 4, 5],
            segments=[1, 2, 3, 4, 5],
            ages=[1, 2, 3, 4, 5],
        )
        death_token = 6  # Not in concepts

        # 2. Execute function
        result = extract_death(patient, death_token)

        # 3. Assertions
        self.assertTrue(
            np.isnan(result), "Should return np.nan when death_token not found"
        )

    def test_extract_death_multiple_occurrences(self):
        """Test extracting death when death token appears multiple times."""
        # 1. Setup test data
        patient = CausalPatientData(
            pid=1,
            concepts=[1, 5, 2, 5, 3],  # Death token 5 appears twice
            abspos=[10, 20, 30, 40, 50],
            segments=[1, 2, 3, 4, 5],
            ages=[1, 2, 3, 4, 5],
        )
        death_token = 5

        # 2. Execute function
        result = extract_death(patient, death_token)

        # 3. Assertions
        # Should return the abspos of the first occurrence (index 1)
        self.assertEqual(
            result, 20, "Should return abspos of first occurrence of death_token"
        )

    def test_extract_death_first_position(self):
        """Test extracting death when death token is at the beginning."""
        # 1. Setup test data
        patient = CausalPatientData(
            pid=1,
            concepts=[5, 1, 2, 3, 4],  # Death token at index 0
            abspos=[100, 200, 300, 400, 500],
            segments=[1, 2, 3, 4, 5],
            ages=[1, 2, 3, 4, 5],
        )
        death_token = 5

        # 2. Execute function
        result = extract_death(patient, death_token)

        # 3. Assertions
        self.assertEqual(
            result, 100, "Should return abspos of death_token at first position"
        )

    def test_extract_death_last_position(self):
        """Test extracting death when death token is at the end."""
        # 1. Setup test data
        patient = CausalPatientData(
            pid=1,
            concepts=[1, 2, 3, 4, 5],  # Death token at last index
            abspos=[100, 200, 300, 400, 500],
            segments=[1, 2, 3, 4, 5],
            ages=[1, 2, 3, 4, 5],
        )
        death_token = 5

        # 2. Execute function
        result = extract_death(patient, death_token)

        # 3. Assertions
        self.assertEqual(
            result, 500, "Should return abspos of death_token at last position"
        )

    def test_extract_death_empty_concepts(self):
        """Test extracting death from patient with empty concepts list."""
        # 1. Setup test data
        patient = CausalPatientData(
            pid=1,
            concepts=[],  # Empty concepts
            abspos=[],
            segments=[],
            ages=[],
        )
        death_token = 5

        # 2. Execute function
        result = extract_death(patient, death_token)

        # 3. Assertions
        self.assertTrue(
            np.isnan(result), "Should return np.nan for empty concepts list"
        )

    def test_extract_death_zero_token(self):
        """Test extracting death with token value 0."""
        # 1. Setup test data
        patient = CausalPatientData(
            pid=1,
            concepts=[0, 1, 2, 3, 4],  # Token 0 at index 0
            abspos=[50, 100, 150, 200, 250],
            segments=[1, 2, 3, 4, 5],
            ages=[1, 2, 3, 4, 5],
        )
        death_token = 0

        # 2. Execute function
        result = extract_death(patient, death_token)

        # 3. Assertions
        self.assertEqual(result, 50, "Should handle death_token value of 0")

    def test_extract_death_negative_token(self):
        """Test extracting death with negative token value."""
        # 1. Setup test data
        patient = CausalPatientData(
            pid=1,
            concepts=[-1, 1, 2, 3, 4],  # Negative token at index 0
            abspos=[75, 100, 150, 200, 250],
            segments=[1, 2, 3, 4, 5],
            ages=[1, 2, 3, 4, 5],
        )
        death_token = -1

        # 2. Execute function
        result = extract_death(patient, death_token)

        # 3. Assertions
        self.assertEqual(result, 75, "Should handle negative death_token values")

    def test_extract_death_float_abspos(self):
        """Test extracting death with float abspos values."""
        # 1. Setup test data
        patient = CausalPatientData(
            pid=1,
            concepts=[1, 2, 3, 4, 5],
            abspos=[1.5, 2.5, 3.5, 4.5, 5.5],  # Float values
            segments=[1, 2, 3, 4, 5],
            ages=[1, 2, 3, 4, 5],
        )
        death_token = 5

        # 2. Execute function
        result = extract_death(patient, death_token)

        # 3. Assertions
        self.assertEqual(result, 5.5, "Should handle float abspos values")

    def test_extract_death_return_type(self):
        """Test that the function returns correct types."""
        # 1. Setup test data for both cases
        patient_with_death = CausalPatientData(
            pid=1,
            concepts=[1, 2, 3, 4, 5],
            abspos=[1, 2, 3, 4, 5],
            segments=[1, 2, 3, 4, 5],
            ages=[1, 2, 3, 4, 5],
        )

        patient_without_death = CausalPatientData(
            pid=1,
            concepts=[1, 2, 3, 4, 5],
            abspos=[1, 2, 3, 4, 5],
            segments=[1, 2, 3, 4, 5],
            ages=[1, 2, 3, 4, 5],
        )

        # 2. Execute function
        result_with_death = extract_death(patient_with_death, 5)
        result_without_death = extract_death(patient_without_death, 6)

        # 3. Assertions
        self.assertIsInstance(
            result_with_death,
            (int, float),
            "Should return numeric type when death found",
        )
        self.assertTrue(
            np.isnan(result_without_death), "Should return np.nan when death not found"
        )


if __name__ == "__main__":
    unittest.main()
