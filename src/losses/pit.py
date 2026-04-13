"""Permutation Invariant Training (PIT) loss wrapper.

When voice-role labels are unavailable, PIT finds the optimal assignment
between estimated and target stems by trying all permutations and selecting
the one with minimum total loss.
"""

from __future__ import annotations

import itertools
from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def pit_loss(
    estimates: Float[Array, "N T"],
    targets: Float[Array, "N T"],
    loss_fn: Callable[
        [Float[Array, "T"], Float[Array, "T"]], Float[Array, ""]
    ],
) -> tuple[Float[Array, ""], Array]:
    """Compute PIT loss over all permutations.

    Args:
        estimates: (N, T) estimated stems.
        targets: (N, T) target stems.
        loss_fn: Pairwise loss function (estimate, target) → scalar.

    Returns:
        (min_loss, best_perm) where best_perm is the optimal permutation indices.
    """
    n = estimates.shape[0]
    perms = jnp.array(list(itertools.permutations(range(n))))  # (N!, N)

    def perm_loss(perm: Array) -> Float[Array, ""]:
        # Compute total loss for this permutation
        permuted_estimates = estimates[perm]
        losses = jax.vmap(loss_fn)(permuted_estimates, targets)
        return jnp.sum(losses)

    all_losses = jax.vmap(perm_loss)(perms)  # (N!,)
    best_idx = jnp.argmin(all_losses)
    return all_losses[best_idx], perms[best_idx]
