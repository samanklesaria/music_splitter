"""Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).

Primary evaluation metric for source separation. Higher is better.
"""

import jax.numpy as jnp
from jaxtyping import Array, Float


def si_sdr(
    estimate: Float[Array, "T"],
    target: Float[Array, "T"],
    eps: float = 1e-8,
) -> Float[Array, ""]:
    """Compute SI-SDR between estimate and target.

    Returns scalar SI-SDR in dB (higher is better).
    """
    # Zero-mean
    estimate = estimate - jnp.mean(estimate)
    target = target - jnp.mean(target)

    # s_target = <e, t> / <t, t> * t
    dot = jnp.sum(estimate * target)
    s_target = (dot / (jnp.sum(target**2) + eps)) * target

    # e_noise = estimate - s_target
    e_noise = estimate - s_target

    si_sdr_val = 10.0 * jnp.log10(
        jnp.sum(s_target**2) / (jnp.sum(e_noise**2) + eps) + eps
    )
    return si_sdr_val


def neg_si_sdr(
    estimate: Float[Array, "T"],
    target: Float[Array, "T"],
    eps: float = 1e-8,
) -> Float[Array, ""]:
    """Negative SI-SDR (for use as a minimization loss)."""
    return -si_sdr(estimate, target, eps)
