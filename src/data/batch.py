"""Batch construction for training using the grain data pipeline."""

import jax.numpy as jnp
import numpy as np
import grain
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype

from src.data.augmentation import AugmentationPipeline, make_training_pair


def build_loader(
    datasets: list,
    batch_size: int = 8,
    num_stems: int = 6,
    sample_rate: int = 44100,
    segment_seconds: float = 4.0,
    augmentation: AugmentationPipeline | None = None,
    seed: int = 42,
) -> grain.MapDataset:
    """Build a grain data pipeline over one or more datasets.

    Each element yielded by the returned dataset is a
    ``(mixtures, stems)`` pair of JAX arrays with shapes
    ``(batch_size, T)`` and ``(batch_size, num_stems, T)``.

    The pipeline: concatenate sources → shuffle → random segment
    extraction → optional augmentation → batch.
    """
    segment_samples = int(segment_seconds * sample_rate)

    sources = [grain.MapDataset.source(ds) for ds in datasets]
    combined = (
        grain.MapDataset.concatenate(sources) if len(sources) > 1 else sources[0]
    )

    ds = combined.seed(seed).shuffle()

    @jaxtyped(typechecker=beartype)
    def extract_segment(
        item: tuple[Float[np.ndarray, "T"], Float[np.ndarray, "N T"]],
        rng: np.random.Generator,
    ) -> tuple[Float[np.ndarray, "S"], Float[np.ndarray, "K S"]]:
        mixture, stems = item
        total = mixture.shape[0]

        if stems.shape[0] < num_stems:
            padded = np.zeros((num_stems, stems.shape[1]), dtype=np.float32)
            padded[: stems.shape[0]] = stems
            stems = padded
        elif stems.shape[0] > num_stems:
            stems = stems[:num_stems]

        if total <= segment_samples:
            mix_out = np.zeros(segment_samples, dtype=np.float32)
            stem_out = np.zeros((num_stems, segment_samples), dtype=np.float32)
            mix_out[:total] = mixture
            stem_out[:, :total] = stems
            return mix_out, stem_out

        start = rng.integers(0, total - segment_samples)
        return (
            mixture[start : start + segment_samples],
            stems[:, start : start + segment_samples],
        )

    ds = ds.random_map(extract_segment)

    if augmentation is not None:
        @jaxtyped(typechecker=beartype)
        def augment(
            item: tuple[Float[np.ndarray, "S"], Float[np.ndarray, "N S"]],
            rng: np.random.Generator,
        ) -> tuple[Float[np.ndarray, "S"], Float[np.ndarray, "N S"]]:
            _, stems = item
            stems = augmentation.augment_stems(stems, sample_rate, rng)
            mixture, selected = make_training_pair(stems, rng, min_stems=2)
            k = selected.shape[0]
            if k < num_stems:
                padded = np.zeros((num_stems, selected.shape[1]), dtype=np.float32)
                padded[:k] = selected
                selected = padded
            return mixture, selected

        ds = ds.random_map(augment)

    @jaxtyped(typechecker=beartype)
    def batch_to_jax(items: list) -> tuple[Float[Array, "B S"], Float[Array, "B N S"]]:
        mixtures = np.stack([m for m, _ in items])
        stems = np.stack([s for _, s in items])
        return jnp.array(mixtures), jnp.array(stems)

    return ds.batch(batch_size, drop_remainder=True, batch_fn=batch_to_jax)
