"""Data augmentation pipeline for vocal separation training.

Implements the power-set augmentation strategy from SepACap, plus standard
audio augmentations (pitch shift, time stretch, random gain, RIR convolution).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np


@dataclass
class AugmentationPipeline:
    """Augments isolated stems before mixing.

    All augmentations operate on individual stems *before* they are summed
    into a mixture, so each stem gets independent transformations.
    """

    pitch_shift_range: tuple[float, float] = (-2.0, 2.0)  # semitones
    time_stretch_range: tuple[float, float] = (0.9, 1.1)
    gain_range_db: tuple[float, float] = (-6.0, 6.0)
    enable_pitch_shift: bool = True
    enable_time_stretch: bool = True
    enable_gain: bool = True
    enable_rir: bool = False  # requires RIR impulse responses on disk
    rir_paths: list[str] = field(default_factory=list)
    _rir_cache: list[np.ndarray] = field(default_factory=list, init=False, repr=False)

    def random_gain(self, stem: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Apply random gain in dB."""
        lo, hi = self.gain_range_db
        gain_db = rng.uniform(lo, hi)
        return stem * (10.0 ** (gain_db / 20.0))

    def pitch_shift(
        self, stem: np.ndarray, sr: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Pitch-shift a stem by a random number of semitones."""
        import librosa

        lo, hi = self.pitch_shift_range
        n_steps = rng.uniform(lo, hi)
        return librosa.effects.pitch_shift(stem, sr=sr, n_steps=n_steps)

    def time_stretch(self, stem: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Time-stretch a stem by a random factor."""
        import librosa

        lo, hi = self.time_stretch_range
        rate = rng.uniform(lo, hi)
        return librosa.effects.time_stretch(stem, rate=rate)

    def apply_rir(self, stem: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Convolve with a random room impulse response."""
        if not self._rir_cache:
            if not self.rir_paths:
                return stem
            import soundfile as sf

            for p in self.rir_paths:
                rir, _ = sf.read(p, dtype="float32")
                if rir.ndim > 1:
                    rir = rir[:, 0]
                self._rir_cache.append(rir)

        rir = self._rir_cache[rng.integers(0, len(self._rir_cache))]
        convolved = np.convolve(stem, rir, mode="full")[: len(stem)]
        # Normalize to preserve energy
        if np.max(np.abs(convolved)) > 0:
            convolved *= np.sqrt(np.sum(stem**2) / (np.sum(convolved**2) + 1e-8))
        return convolved

    def augment_stem(
        self, stem: np.ndarray, sr: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Apply all enabled augmentations to a single stem."""
        if self.enable_gain:
            stem = self.random_gain(stem, rng)
        if self.enable_pitch_shift:
            stem = self.pitch_shift(stem, sr, rng)
        if self.enable_time_stretch:
            original_len = len(stem)
            stem = self.time_stretch(stem, rng)
            # Crop or pad to original length after stretch
            if len(stem) > original_len:
                stem = stem[:original_len]
            elif len(stem) < original_len:
                stem = np.pad(stem, (0, original_len - len(stem)))
        if self.enable_rir:
            stem = self.apply_rir(stem, rng)
        return stem.astype(np.float32)

    def augment_stems(
        self, stems: np.ndarray, sr: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Augment each stem independently. stems shape: (num_stems, T)."""
        out = np.empty_like(stems)
        for i in range(stems.shape[0]):
            out[i] = self.augment_stem(stems[i], sr, rng)
        return out


def power_set_subsets(
    stems: np.ndarray, min_size: int = 2
) -> list[tuple[list[int], np.ndarray]]:
    """Generate all subsets of stems with at least `min_size` members.

    Args:
        stems: (num_stems, T) array of isolated stems.
        min_size: Minimum subset size (default 2 — need at least 2 to separate).

    Returns:
        List of (indices, subset_stems) tuples where subset_stems is (k, T).
    """
    n = stems.shape[0]
    subsets = []
    for size in range(min_size, n + 1):
        for combo in itertools.combinations(range(n), size):
            indices = list(combo)
            subset = stems[indices]
            subsets.append((indices, subset))
    return subsets


def make_training_pair(
    stems: np.ndarray,
    rng: np.random.Generator,
    min_stems: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a single training pair by randomly selecting a subset of stems.

    Returns (mixture, selected_stems) where mixture is the sum of selected stems.
    """
    n = stems.shape[0]
    k = rng.integers(min_stems, n + 1)
    indices = rng.choice(n, size=k, replace=False)
    indices.sort()
    selected = stems[indices]
    mixture = selected.sum(axis=0)
    return mixture, selected
