"""Multi-resolution STFT loss.

Combines spectral convergence and log-magnitude STFT loss at multiple
FFT sizes for frequency-domain supervision.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def _stft_magnitude(
    x: Float[Array, "T"], fft_size: int, hop_size: int, win_size: int
) -> Float[Array, "F K"]:
    """Compute STFT magnitude spectrogram."""
    window = jnp.hanning(win_size)
    # Pad signal
    pad_amount = fft_size // 2
    x_padded = jnp.pad(x, (pad_amount, pad_amount))

    # Frame the signal
    num_frames = (len(x_padded) - win_size) // hop_size + 1
    indices = jnp.arange(win_size)[None, :] + jnp.arange(num_frames)[:, None] * hop_size
    frames = x_padded[indices] * window[None, :]

    # FFT
    spec = jnp.fft.rfft(frames, n=fft_size, axis=-1)
    return jnp.abs(spec).T  # (F, K)


def spectral_convergence(
    est_mag: Float[Array, "F K"], ref_mag: Float[Array, "F K"]
) -> Float[Array, ""]:
    """Spectral convergence loss (Frobenius norm ratio)."""
    return jnp.linalg.norm(ref_mag - est_mag) / (jnp.linalg.norm(ref_mag) + 1e-8)


def log_stft_magnitude(
    est_mag: Float[Array, "F K"], ref_mag: Float[Array, "F K"]
) -> Float[Array, ""]:
    """Log STFT magnitude loss (L1 in log domain)."""
    return jnp.mean(jnp.abs(jnp.log(est_mag + 1e-8) - jnp.log(ref_mag + 1e-8)))


def stft_loss(
    estimate: Float[Array, "T"],
    target: Float[Array, "T"],
    fft_size: int = 1024,
    hop_size: int = 256,
    win_size: int = 1024,
) -> Float[Array, ""]:
    """Single-resolution STFT loss."""
    est_mag = _stft_magnitude(estimate, fft_size, hop_size, win_size)
    ref_mag = _stft_magnitude(target, fft_size, hop_size, win_size)
    sc = spectral_convergence(est_mag, ref_mag)
    lm = log_stft_magnitude(est_mag, ref_mag)
    return sc + lm


def multi_resolution_stft_loss(
    estimate: Float[Array, "T"],
    target: Float[Array, "T"],
    fft_sizes: tuple[int, ...] = (512, 1024, 2048),
    hop_sizes: tuple[int, ...] = (128, 256, 512),
    win_sizes: tuple[int, ...] = (512, 1024, 2048),
) -> Float[Array, ""]:
    """Multi-resolution STFT loss: average over multiple FFT configurations."""
    total = jnp.float32(0.0)
    for fft_size, hop_size, win_size in zip(fft_sizes, hop_sizes, win_sizes):
        total = total + stft_loss(estimate, target, fft_size, hop_size, win_size)
    return total / len(fft_sizes)
