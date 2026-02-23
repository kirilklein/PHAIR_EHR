"""Unit tests for cohort sampling utilities."""

import unittest

import torch

from corebehrt.functional.causal.cohort_sampler import sample_cohort


class TestSampleCohort(unittest.TestCase):

    def setUp(self):
        self.full_pids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    def test_default_method_is_resample(self):
        """Default method should be resample (without replacement)."""
        sampled = sample_cohort(self.full_pids, sample_size=5, seed=42)
        self.assertEqual(len(sampled), 5)
        # No duplicates
        self.assertEqual(len(set(sampled.tolist())), 5)

    def test_resample_no_duplicates(self):
        """Resample method should produce no duplicate PIDs."""
        sampled = sample_cohort(
            self.full_pids, sample_size=10, seed=42, method="resample"
        )
        self.assertEqual(len(set(sampled.tolist())), 10)

    def test_resample_rejects_oversize(self):
        """Resample method should reject sample_size > n_total."""
        with self.assertRaises(ValueError):
            sample_cohort(
                self.full_pids, sample_size=20, seed=42, method="resample"
            )

    def test_bootstrap_allows_duplicates(self):
        """Bootstrap method should sample with replacement (duplicates possible)."""
        # Use a large sample to make duplicates very likely
        sampled = sample_cohort(
            self.full_pids, sample_size=100, seed=42, method="bootstrap"
        )
        self.assertEqual(len(sampled), 100)
        # With 100 draws from 10 items, duplicates are virtually guaranteed
        self.assertLess(len(set(sampled.tolist())), 100)

    def test_bootstrap_accepts_oversize(self):
        """Bootstrap method should accept sample_size > n_total."""
        sampled = sample_cohort(
            self.full_pids, sample_size=20, seed=42, method="bootstrap"
        )
        self.assertEqual(len(sampled), 20)

    def test_invalid_method_raises(self):
        """Invalid method should raise ValueError."""
        with self.assertRaises(ValueError):
            sample_cohort(
                self.full_pids, sample_size=5, seed=42, method="invalid"
            )

    def test_bootstrap_with_fraction(self):
        """Bootstrap with fraction should work."""
        sampled = sample_cohort(
            self.full_pids, sample_fraction=0.5, seed=42, method="bootstrap"
        )
        self.assertEqual(len(sampled), 5)

    def test_resample_reproducibility(self):
        """Same seed should produce same results."""
        s1 = sample_cohort(self.full_pids, sample_size=5, seed=123)
        s2 = sample_cohort(self.full_pids, sample_size=5, seed=123)
        self.assertTrue(torch.equal(s1, s2))


if __name__ == "__main__":
    unittest.main()
