"""SepReformer model for source separation in JAX/Equinox.

Architecture follows the SepReformer (NeurIPS 2024) with modifications from
SepACap (Lanzendörfer & Pinkl, 2024):
  - SNAKE activations for better harmonic signal modeling
  - Dual-path (intra-chunk + inter-chunk) transformer blocks
  - Learnable encoder/decoder (1-D convolution)
  - Early split into N independent speaker streams with weight-shared reconstruction
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from typeguard import typechecked


# --------------------------------------------------------------------------- #
# SNAKE activation
# --------------------------------------------------------------------------- #

@typechecked
class Snake(eqx.Module):
    """SNAKE activation: x + (1/a) * sin^2(a * x).

    Better than ReLU/GELU for periodic/harmonic signals.
    """

    alpha: Array

    def __init__(self, features: int, *, key: jax.random.PRNGKey):
        self.alpha = jnp.ones(features)

    def __call__(self, x: Float[Array, "... F"]) -> Float[Array, "... F"]:
        a = self.alpha
        return x + (1.0 / (a + 1e-6)) * jnp.sin(a * x) ** 2


# --------------------------------------------------------------------------- #
# Learnable Encoder / Decoder
# --------------------------------------------------------------------------- #


@typechecked
class Encoder(eqx.Module):
    """1-D convolutional encoder: waveform → latent representation."""

    conv: eqx.nn.Conv1d

    def __init__(
        self,
        kernel_size: int = 16,
        stride: int = 8,
        out_channels: int = 256,
        *,
        key: jax.random.PRNGKey,
    ):
        self.conv = eqx.nn.Conv1d(
            1, out_channels, kernel_size, stride=stride, key=key
        )

    def __call__(self, x: Float[Array, "T"]) -> Float[Array, "L C"]:
        # x: (T,) → (1, T) for Conv1d
        x = x[None, :]  # (1, T)
        h = self.conv(x)  # (C, L)
        h = jax.nn.gelu(h)  # (C, L)
        h = jnp.transpose(h)  # (L, C)
        return h

@typechecked
class Decoder(eqx.Module):
    """Transposed 1-D convolution decoder: latent → waveform."""

    conv_t: eqx.nn.ConvTranspose1d

    def __init__(
        self,
        kernel_size: int = 16,
        stride: int = 8,
        in_channels: int = 256,
        *,
        key: jax.random.PRNGKey,
    ):
        self.conv_t = eqx.nn.ConvTranspose1d(
            in_channels, 1, kernel_size, stride=stride, key=key
        )

    def __call__(self, h: Float[Array, "L C"]) -> Float[Array, "T"]:
        h = jnp.transpose(h)  # (C, L)
        out = self.conv_t(h)  # (1, T)
        return out[0]  # (T,)


# --------------------------------------------------------------------------- #
# Dual-path transformer block
# --------------------------------------------------------------------------- #

@typechecked
class FeedForward(eqx.Module):
    """Two-layer FFN with SNAKE activation."""

    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    snake: Snake

    def __init__(self, dim: int, ff_dim: int, *, key: jax.random.PRNGKey):
        k1, k2, k3 = jax.random.split(key, 3)
        self.linear1 = eqx.nn.Linear(dim, ff_dim, key=k1)
        self.linear2 = eqx.nn.Linear(ff_dim, dim, key=k2)
        self.snake = Snake(ff_dim, key=k3)

    def __call__(self, x: Float[Array, "F"]) -> Float[Array, "F"]:
        h = self.linear1(x)
        h = self.snake(h)
        return self.linear2(h)

@typechecked
class TransformerBlock(eqx.Module):
    """Pre-norm transformer block with RoPE, multi-head attention, FFN, and LayerScale."""

    norm1: eqx.nn.LayerNorm
    attn: eqx.nn.MultiheadAttention
    norm2: eqx.nn.LayerNorm
    ff: FeedForward
    rope: eqx.nn.RotaryPositionalEmbedding
    scale1: Array  # LayerScale for attention branch
    scale2: Array  # LayerScale for FFN branch

    def __init__(
        self, dim: int, num_heads: int, ff_dim: int, *, key: jax.random.PRNGKey
    ):
        k1, k2 = jax.random.split(key)
        self.norm1 = eqx.nn.LayerNorm(dim)
        self.attn = eqx.nn.MultiheadAttention(
            num_heads=num_heads, query_size=dim, key=k1
        )
        self.norm2 = eqx.nn.LayerNorm(dim)
        self.ff = FeedForward(dim, ff_dim, key=k2)
        self.rope = eqx.nn.RotaryPositionalEmbedding(dim // num_heads)
        self.scale1 = jnp.full(dim, 1e-4)
        self.scale2 = jnp.full(dim, 1e-4)

    def __call__(self, x: Float[Array, "S D"]) -> Float[Array, "S D"]:
        # Self-attention with pre-norm, RoPE, and LayerScale
        normed = jax.vmap(self.norm1)(x)

        def process_heads(q, k, v):
            q = jax.vmap(self.rope, in_axes=1, out_axes=1)(q)
            k = jax.vmap(self.rope, in_axes=1, out_axes=1)(k)
            return q, k, v

        attn_out = self.attn(normed, normed, normed, process_heads=process_heads)
        x = x + self.scale1 * attn_out
        # FFN with pre-norm and LayerScale
        normed = jax.vmap(self.norm2)(x)
        ff_out = jax.vmap(self.ff)(normed)
        x = x + self.scale2 * ff_out
        return x

@typechecked
class DualPathBlock(eqx.Module):
    """Dual-path processing: intra-chunk attention + inter-chunk attention.

    Input is reshaped from (L, C) → (num_chunks, chunk_size, C):
      - Intra-chunk: attention within each chunk (local patterns)
      - Inter-chunk: attention across chunks at each position (global patterns)
    """

    intra_block: TransformerBlock
    inter_block: TransformerBlock
    chunk_size: int = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ff_dim: int,
        chunk_size: int = 64,
        *,
        key: jax.random.PRNGKey,
    ):
        k1, k2 = jax.random.split(key)
        self.intra_block = TransformerBlock(dim, num_heads, ff_dim, key=k1)
        self.inter_block = TransformerBlock(dim, num_heads, ff_dim, key=k2)
        self.chunk_size = chunk_size

    def __call__(self, x: Float[Array, "L C"]) -> Float[Array, "L C"]:
        L, C = x.shape
        K = self.chunk_size

        # Pad to multiple of chunk_size
        pad_len = (K - L % K) % K
        if pad_len > 0:
            x = jnp.pad(x, ((0, pad_len), (0, 0)))

        L_padded = x.shape[0]
        num_chunks = L_padded // K

        # Reshape to (num_chunks, chunk_size, C)
        chunks = x.reshape(num_chunks, K, C)

        # Intra-chunk attention: attend within each chunk
        chunks = jax.vmap(self.intra_block)(chunks)  # (num_chunks, K, C)

        # Inter-chunk attention: transpose to (K, num_chunks, C), attend across chunks
        inter = jnp.transpose(chunks, (1, 0, 2))  # (K, num_chunks, C)
        inter = jax.vmap(self.inter_block)(inter)  # (K, num_chunks, C)
        chunks = jnp.transpose(inter, (1, 0, 2))  # (num_chunks, K, C)

        # Reshape back
        out = chunks.reshape(L_padded, C)
        return out[:L]  # remove padding


# --------------------------------------------------------------------------- #
# Early split layer
# --------------------------------------------------------------------------- #


class SplitLayer(eqx.Module):
    """Projects shared features into N independent speaker streams (early split).

    Two linear layers with GLU activation (as specified in §3 of arXiv:2406.05983):
      linear1: (C → 2C) with GLU → (C)   [gated feature refinement]
      linear2: (C → N*C)                  [expand to N streams]
    """

    linear1: eqx.nn.Linear  # C → 2C for GLU
    linear2: eqx.nn.Linear  # C → N*C
    num_stems: int = eqx.field(static=True)

    def __init__(self, dim: int, num_stems: int, *, key: jax.random.PRNGKey):
        k1, k2 = jax.random.split(key)
        self.linear1 = eqx.nn.Linear(dim, dim * 2, key=k1)
        self.linear2 = eqx.nn.Linear(dim, dim * num_stems, key=k2)
        self.num_stems = num_stems

    def __call__(self, x: Float[Array, "L C"]) -> Float[Array, "N L C"]:
        L, C = x.shape
        # First linear + GLU
        h = jax.vmap(self.linear1)(x)               # (L, 2*C)
        gate, val = jnp.split(h, 2, axis=-1)        # each (L, C)
        h = jax.nn.sigmoid(gate) * val              # (L, C)
        # Second linear expands to N stems
        h = jax.vmap(self.linear2)(h)               # (L, N*C)
        h = h.reshape(L, self.num_stems, C)         # (L, N, C)
        return jnp.transpose(h, (1, 0, 2))          # (N, L, C)


# --------------------------------------------------------------------------- #
# Full SepReformer model
# --------------------------------------------------------------------------- #


class SepReformer(eqx.Module):
    """SepReformer for multi-voice separation with early split.

    Architecture (§3, arXiv:2406.05983):
      1. Audio encoder: waveform → latent (L, C).
      2. Separation blocks (shared): num_sep_blocks dual-path blocks process
         the single mixture representation.
      3. Early split: a learned projection expands (L, C) → (N, L, C), creating
         N independent speaker streams before the reconstruction stage.
      4. Reconstruction blocks (weight-shared): the same num_rec_blocks dual-path
         blocks are applied independently to each of the N streams via vmap,
         so all N speakers share identical parameters but receive distinct inputs.
      5. Audio decoder (weight-shared): the same decoder maps each (L, C) → (T,).

    Args:
        num_stems: Number of output stems (e.g. 4 for lead + 3 harmony groups).
        dim: Latent dimension.
        num_heads: Attention heads per transformer block.
        ff_dim: Feed-forward hidden dimension.
        num_sep_blocks: Dual-path blocks in the shared separation encoder.
        num_rec_blocks: Dual-path blocks in the weight-shared reconstruction decoder.
        chunk_size: Chunk size for dual-path processing.
        encoder_kernel: Encoder/decoder convolution kernel size.
        encoder_stride: Encoder/decoder convolution stride.
    """

    encoder: Encoder
    sep_blocks: list[DualPathBlock]
    split: SplitLayer
    rec_blocks: list[DualPathBlock]
    decoder: Decoder
    num_stems: int = eqx.field(static=True)

    def __init__(
        self,
        num_stems: int = 4,
        dim: int = 256,
        num_heads: int = 8,
        ff_dim: int = 1024,
        num_sep_blocks: int = 2,
        num_rec_blocks: int = 2,
        chunk_size: int = 64,
        encoder_kernel: int = 16,
        encoder_stride: int = 8,
        *,
        key: jax.random.PRNGKey,
    ):
        num_blocks = num_sep_blocks + num_rec_blocks
        keys = jax.random.split(key, 3 + num_blocks)
        self.num_stems = num_stems

        self.encoder = Encoder(encoder_kernel, encoder_stride, dim, key=keys[0])
        self.decoder = Decoder(encoder_kernel, encoder_stride, dim, key=keys[1])
        self.split = SplitLayer(dim, num_stems, key=keys[2])

        self.sep_blocks = [
            DualPathBlock(dim, num_heads, ff_dim, chunk_size, key=keys[3 + i])
            for i in range(num_sep_blocks)
        ]
        self.rec_blocks = [
            DualPathBlock(dim, num_heads, ff_dim, chunk_size, key=keys[3 + num_sep_blocks + i])
            for i in range(num_rec_blocks)
        ]

    def __call__(
        self, x: Float[Array, "T"]
    ) -> Float[Array, "N T"]:
        """Separate a mono mixture into N stems.

        Args:
            x: Input waveform of shape (T,).

        Returns:
            Separated waveforms of shape (num_stems, T).
        """
        original_len = x.shape[0]

        # Encode
        h = self.encoder(x)  # (L, C)

        # Shared separation processing
        for block in self.sep_blocks:
            h = block(h)  # (L, C)

        # Early split: shared features → N independent speaker streams
        stems = self.split(h)  # (N, L, C)

        # Weight-shared reconstruction: same block applied to each stem independently
        for block in self.rec_blocks:
            stems = jax.vmap(block)(stems)  # (N, L, C)

        # Weight-shared decode: same decoder applied to each stem
        out = jax.vmap(self.decoder)(stems)  # (N, T')

        # Trim or pad to original length
        T_out = out.shape[1]
        if T_out > original_len:
            out = out[:, :original_len]
        elif T_out < original_len:
            out = jnp.pad(out, ((0, 0), (0, original_len - T_out)))

        return out  # (N, T)
