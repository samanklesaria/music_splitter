"""Tests for augmentation pipeline."""

import numpy as np
import pytest

from src.data.augmentation import AugmentationPipeline, power_set_subsets, make_training_pair


class TestAugmentationPipeline:
    def test_random_gain(self):
        pipe = AugmentationPipeline(gain_range_db=(-3.0, 3.0))
        rng = np.random.default_rng(0)
        x = np.ones(1000, dtype=np.float32)
        out = pipe.random_gain(x, rng)
        # Should be scaled but not identical
        assert out.shape == (1000,)
        assert not np.allclose(out, x)

    def test_augment_stem_preserves_length(self):
        pipe = AugmentationPipeline(
            enable_pitch_shift=False,  # skip slow operations for test speed
            enable_time_stretch=False,
            enable_gain=True,
            enable_rir=False,
        )
        rng = np.random.default_rng(0)
        x = np.random.randn(4000).astype(np.float32)
        out = pipe.augment_stem(x, sr=44100, rng=rng)
        assert out.shape == x.shape

    def test_augment_stems_independent(self):
        pipe = AugmentationPipeline(
            enable_pitch_shift=False,
            enable_time_stretch=False,
            enable_gain=True,
        )
        rng = np.random.default_rng(0)
        stems = np.ones((3, 1000), dtype=np.float32)
        out = pipe.augment_stems(stems, sr=44100, rng=rng)
        # Each stem should get different gain
        assert not np.allclose(out[0], out[1])


class TestPowerSet:
    def test_subset_count(self):
        stems = np.zeros((4, 100))
        subsets = power_set_subsets(stems, min_size=2)
        # C(4,2) + C(4,3) + C(4,4) = 6 + 4 + 1 = 11
        assert len(subsets) == 11

    def test_min_size(self):
        stems = np.zeros((3, 100))
        subsets = power_set_subsets(stems, min_size=3)
        # Only the full set
        assert len(subsets) == 1

    def test_indices_correct(self):
        stems = np.arange(12).reshape(3, 4).astype(np.float32)
        subsets = power_set_subsets(stems, min_size=2)
        for indices, subset in subsets:
            np.testing.assert_array_equal(subset, stems[indices])


class TestMakeTrainingPair:
    def test_output_shapes(self):
        rng = np.random.default_rng(0)
        stems = np.random.randn(4, 1000).astype(np.float32)
        mixture, selected = make_training_pair(stems, rng, min_stems=2)
        assert mixture.shape == (1000,)
        assert selected.shape[1] == 1000
        assert 2 <= selected.shape[0] <= 4

    def test_mixture_is_sum(self):
        rng = np.random.default_rng(0)
        stems = np.random.randn(4, 1000).astype(np.float32)
        mixture, selected = make_training_pair(stems, rng)
        np.testing.assert_allclose(mixture, selected.sum(axis=0), atol=1e-6)
