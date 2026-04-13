"""Tests for loss functions."""

import jax
import jax.numpy as jnp
import pytest

from src.losses.sisdr import si_sdr, neg_si_sdr
from src.losses.stft import multi_resolution_stft_loss, stft_loss
from src.losses.pit import pit_loss
from src.losses.composite import composite_loss


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


class TestSISDR:
    def test_identical_signals(self):
        x = jnp.sin(jnp.linspace(0, 10 * jnp.pi, 1000))
        val = si_sdr(x, x)
        # SI-SDR of identical signals should be very high
        assert val > 30.0

    def test_scaled_signal(self):
        x = jnp.sin(jnp.linspace(0, 10 * jnp.pi, 1000))
        val = si_sdr(2.0 * x, x)
        # Scale-invariant, so should still be very high
        assert val > 30.0

    def test_orthogonal_signals(self):
        x = jnp.sin(jnp.linspace(0, 10 * jnp.pi, 1000))
        y = jnp.cos(jnp.linspace(0, 10 * jnp.pi, 1000))
        val = si_sdr(x, y)
        # Near-orthogonal signals → low SI-SDR
        assert val < 5.0

    def test_neg_si_sdr(self):
        x = jnp.sin(jnp.linspace(0, 10 * jnp.pi, 1000))
        assert jnp.isclose(neg_si_sdr(x, x), -si_sdr(x, x))


class TestSTFTLoss:
    def test_identical_is_zero(self):
        x = jnp.sin(jnp.linspace(0, 10 * jnp.pi, 2000))
        loss = stft_loss(x, x, fft_size=256, hop_size=64, win_size=256)
        assert loss < 0.01

    def test_different_is_positive(self, key):
        x = jax.random.normal(key, (2000,))
        y = jax.random.normal(jax.random.PRNGKey(1), (2000,))
        loss = stft_loss(x, y, fft_size=256, hop_size=64, win_size=256)
        assert loss > 0.1

    def test_multi_resolution(self, key):
        x = jax.random.normal(key, (4000,))
        y = jax.random.normal(jax.random.PRNGKey(1), (4000,))
        loss = multi_resolution_stft_loss(
            x, y,
            fft_sizes=(256, 512),
            hop_sizes=(64, 128),
            win_sizes=(256, 512),
        )
        assert loss > 0.0


class TestPIT:
    def test_finds_correct_permutation(self):
        # Target: [[1,1,...], [2,2,...]]
        # Estimate: [[2,2,...], [1,1,...]] (reversed)
        t1 = jnp.ones(100)
        t2 = 2.0 * jnp.ones(100)
        targets = jnp.stack([t1, t2])
        estimates = jnp.stack([t2, t1])  # swapped

        def l1_loss(est, tgt):
            return jnp.mean(jnp.abs(est - tgt))

        loss, perm = pit_loss(estimates, targets, l1_loss)
        # PIT should find the swap and give zero loss
        assert loss < 0.01
        assert list(perm) == [1, 0]


class TestCompositeLoss:
    def test_runs(self, key):
        k1, k2 = jax.random.split(key)
        estimates = jax.random.normal(k1, (2, 2000))
        targets = jax.random.normal(k2, (2, 2000))
        loss = composite_loss(estimates, targets, use_pit=True)
        assert jnp.isfinite(loss)

    def test_pit_vs_no_pit(self, key):
        k1, k2 = jax.random.split(key)
        estimates = jax.random.normal(k1, (2, 2000))
        targets = jax.random.normal(k2, (2, 2000))
        loss_pit = composite_loss(estimates, targets, use_pit=True)
        loss_no_pit = composite_loss(estimates, targets, use_pit=False)
        # PIT loss should be <= non-PIT (it optimizes over permutations)
        assert loss_pit <= loss_no_pit + 1e-5
