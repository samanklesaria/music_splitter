"""Batch construction for training.

Handles loading segments from datasets, applying augmentation, and collating
into JAX-compatible batches.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from src.data.augmentation import AugmentationPipeline
from src.data.dagstuhl import DagstuhlChoirSet
from src.data.jacappella import JaCappellaDataset


@dataclass
class BatchLoader:
    """Simple epoch-based batch loader for training.

    Loads random segments from datasets, applies augmentation, and yields
    batches as JAX arrays. Not a JAX data pipeline — keeps data loading in
    NumPy for simplicity.
    """

    datasets: list[JaCappellaDataset | DagstuhlChoirSet]
    batch_size: int = 8
    num_stems: int = 4
    sample_rate: int = 44100
    segment_seconds: float = 4.0
    augmentation: AugmentationPipeline | None = None
    seed: int = 42

    @property
    def segment_samples(self) -> int:
        return int(self.segment_seconds * self.sample_rate)

    @property
    def total_songs(self) -> int:
        return sum(len(d) for d in self.datasets)

    def _get_item(
        self, dataset_idx: int, song_idx: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        ds = self.datasets[dataset_idx]
        mixture, stems = ds.get_segment(song_idx, rng)

        # Ensure consistent number of stems
        if stems.shape[0] < self.num_stems:
            padded = np.zeros(
                (self.num_stems, stems.shape[1]), dtype=np.float32
            )
            padded[: stems.shape[0]] = stems
            stems = padded
        elif stems.shape[0] > self.num_stems:
            stems = stems[: self.num_stems]

        if self.augmentation is not None:
            stems = self.augmentation.augment_stems(stems, self.sample_rate, rng)
            mixture = stems.sum(axis=0)

        return mixture, stems

    def epoch_batches(
        self, epoch: int
    ) -> list[tuple[jnp.ndarray, jnp.ndarray]]:
        """Generate all batches for one epoch.

        Returns list of (mixtures, stems) where:
            mixtures: (batch_size, segment_samples)
            stems: (batch_size, num_stems, segment_samples)
        """
        rng = np.random.default_rng(self.seed + epoch)

        # Build flat index of (dataset_idx, song_idx)
        indices = []
        for di, ds in enumerate(self.datasets):
            for si in range(len(ds)):
                indices.append((di, si))

        rng.shuffle(indices)

        batches = []
        for b_start in range(0, len(indices), self.batch_size):
            b_indices = indices[b_start : b_start + self.batch_size]
            if len(b_indices) < self.batch_size:
                break  # drop incomplete last batch

            mixtures = []
            stems_list = []
            for di, si in b_indices:
                mix, stems = self._get_item(di, si, rng)
                mixtures.append(mix)
                stems_list.append(stems)

            mixtures_np = np.stack(mixtures)
            stems_np = np.stack(stems_list)
            batches.append(
                (jnp.array(mixtures_np), jnp.array(stems_np))
            )

        return batches
