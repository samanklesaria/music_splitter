"""Tests for the SepReformer model."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from src.model.sepreformer import Snake, Encoder, Decoder, DualPathBlock, SepReformer


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


class TestSnake:
    def test_shape_preserved(self, key):
        snake = Snake(64, key=key)
        x = jax.random.normal(key, (10, 64))
        out = jax.vmap(snake)(x)
        assert out.shape == (10, 64)

    def test_not_identity(self, key):
        snake = Snake(8, key=key)
        x = jax.random.normal(key, (8,))
        out = snake(x)
        assert not jnp.allclose(x, out)


class TestEncoder:
    def test_output_shape(self, key):
        enc = Encoder(kernel_size=16, stride=8, out_channels=128, key=key)
        x = jnp.zeros(44100)  # 1 second at 44.1 kHz
        h = enc(x)
        assert h.ndim == 2
        assert h.shape[1] == 128
        # L ≈ ceil((T - kernel_size) / stride) + 1
        expected_L = (44100 - 16) // 8 + 1
        assert h.shape[0] == expected_L


class TestDecoder:
    def test_roundtrip_shape(self, key):
        k1, k2 = jax.random.split(key)
        enc = Encoder(kernel_size=16, stride=8, out_channels=64, key=k1)
        dec = Decoder(kernel_size=16, stride=8, in_channels=64, key=k2)
        x = jnp.zeros(1000)
        h = enc(x)
        y = dec(h)
        # Output length should be close to input (may differ by up to kernel_size)
        assert abs(y.shape[0] - 1000) < 16


class TestDualPathBlock:
    def test_shape_preserved(self, key):
        block = DualPathBlock(dim=64, num_heads=4, ff_dim=128, chunk_size=16, key=key)
        x = jax.random.normal(key, (100, 64))
        out = block(x)
        assert out.shape == (100, 64)

    def test_shape_not_multiple_of_chunk(self, key):
        block = DualPathBlock(dim=64, num_heads=4, ff_dim=128, chunk_size=16, key=key)
        x = jax.random.normal(key, (73, 64))  # not a multiple of 16
        out = block(x)
        assert out.shape == (73, 64)


class TestSepReformer:
    def test_output_shape(self, key):
        model = SepReformer(
            num_stems=4,
            dim=64,
            num_heads=4,
            ff_dim=128,
            num_blocks=2,
            chunk_size=16,
            encoder_kernel=16,
            encoder_stride=8,
            key=key,
        )
        x = jnp.zeros(4000)
        out = model(x)
        assert out.shape == (4, 4000)

    def test_gradient_flows(self, key):
        model = SepReformer(
            num_stems=2,
            dim=32,
            num_heads=2,
            ff_dim=64,
            num_blocks=1,
            chunk_size=8,
            key=key,
        )
        x = jax.random.normal(key, (500,))

        @eqx.filter_grad
        def loss_fn(model):
            out = model(x)
            return jnp.sum(out**2)

        grads = loss_fn(model)
        # Check at least some gradients are non-zero
        grad_leaves = jax.tree.leaves(grads)
        has_nonzero = any(jnp.any(g != 0) for g in grad_leaves if hasattr(g, '__len__'))
        assert has_nonzero
