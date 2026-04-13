"""Composite loss combining SI-SDR and multi-resolution STFT loss with PIT."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from src.losses.pit import pit_loss
from src.losses.sisdr import neg_si_sdr
from src.losses.stft import multi_resolution_stft_loss


def _pairwise_loss(
    estimate: Float[Array, "T"],
    target: Float[Array, "T"],
    stft_weight: float = 0.5,
) -> Float[Array, ""]:
    """Combined SI-SDR + STFT loss for a single (estimate, target) pair."""
    sisdr = neg_si_sdr(estimate, target)
    stft = multi_resolution_stft_loss(estimate, target)
    return sisdr + stft_weight * stft


def composite_loss(
    estimates: Float[Array, "N T"],
    targets: Float[Array, "N T"],
    use_pit: bool = True,
    stft_weight: float = 0.5,
) -> Float[Array, ""]:
    """Composite training loss with optional PIT.

    Args:
        estimates: (N, T) estimated stems from model.
        targets: (N, T) ground-truth stems.
        use_pit: Whether to use permutation-invariant training.
        stft_weight: Weight for STFT loss relative to SI-SDR.

    Returns:
        Scalar loss value.
    """

    def loss_fn(est: Float[Array, "T"], tgt: Float[Array, "T"]) -> Float[Array, ""]:
        return _pairwise_loss(est, tgt, stft_weight)

    if use_pit:
        total_loss, _ = pit_loss(estimates, targets, loss_fn)
    else:
        losses = jax.vmap(loss_fn)(estimates, targets)
        total_loss = jnp.sum(losses)

    return total_loss / estimates.shape[0]
