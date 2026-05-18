"""Data augmentation pipeline for vocal separation training.

Implements the power-set augmentation strategy from SepACap, plus standard
audio augmentations (pitch shift, time stretch, random gain, RIR convolution).
"""

import itertools
from dataclasses import dataclass, field

import numpy as np
from jaxtyping import Float, jaxtyped
from beartype import beartype


@dataclass
class AugmentationPipeline:
    """Augments isolated stems before mixing.

    Parameters are sampled once per segment and applied identically to every
    stem, so the relative balance between voices is preserved.
    """

    pitch_shift_range: tuple[float, float] = (-2.0, 2.0)  # semitones
    time_stretch_range: tuple[float, float] = (0.9, 1.1)
    gain_range_db: tuple[float, float] = (-6.0, 6.0)
    enable_pitch_shift: bool = False
    enable_time_stretch: bool = False
    enable_gain: bool = False
    enable_rir: bool = False  # requires RIR impulse responses on disk
    rir_paths: list[str] = field(default_factory=list)
    _rir_cache: list[np.ndarray] = field(default_factory=list, init=False, repr=False)

    @jaxtyped(typechecker=beartype)
    def random_gain(
        self, stem: Float[np.ndarray, "T"], rng: np.random.Generator
    ) -> Float[np.ndarray, "T"]:
        """Apply random gain in dB."""
        lo, hi = self.gain_range_db
        gain_db = rng.uniform(lo, hi)
        return stem * (10.0 ** (gain_db / 20.0))

    @jaxtyped(typechecker=beartype)
    def pitch_shift(
        self, stem: Float[np.ndarray, "T"], sr: int, rng: np.random.Generator
    ) -> Float[np.ndarray, "T"]:
        """Pitch-shift a stem by a random number of semitones."""
        import librosa

        lo, hi = self.pitch_shift_range
        n_steps = rng.uniform(lo, hi)
        return librosa.effects.pitch_shift(stem, sr=sr, n_steps=n_steps)

    @jaxtyped(typechecker=beartype)
    def time_stretch(
        self, stem: Float[np.ndarray, "T"], rng: np.random.Generator
    ) -> Float[np.ndarray, "S"]:
        """Time-stretch a stem by a random factor."""
        import librosa

        lo, hi = self.time_stretch_range
        rate = rng.uniform(lo, hi)
        return librosa.effects.time_stretch(stem, rate=rate)

    @jaxtyped(typechecker=beartype)
    def apply_rir(
        self, stem: Float[np.ndarray, "T"], rng: np.random.Generator
    ) -> Float[np.ndarray, "T"]:
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

    @jaxtyped(typechecker=beartype)
    def augment_stems(
        self, stems: Float[np.ndarray, "N T"], sr: int, rng: np.random.Generator
    ) -> Float[np.ndarray, "N T"]:
        """Augment all stems with the same sampled transformation parameters."""
        gain_db = rng.uniform(*self.gain_range_db) if self.enable_gain else None
        n_steps = rng.uniform(*self.pitch_shift_range) if self.enable_pitch_shift else None
        rate = rng.uniform(*self.time_stretch_range) if self.enable_time_stretch else None

        rir = None
        if self.enable_rir:
            if not self._rir_cache:
                if self.rir_paths:
                    import soundfile as sf
                    for p in self.rir_paths:
                        r, _ = sf.read(p, dtype="float32")
                        self._rir_cache.append(r[:, 0] if r.ndim > 1 else r)
            if self._rir_cache:
                rir = self._rir_cache[rng.integers(0, len(self._rir_cache))]

        out = np.empty_like(stems)
        for i in range(stems.shape[0]):
            stem = stems[i]
            if gain_db is not None:
                stem = stem * (10.0 ** (gain_db / 20.0))
            if n_steps is not None:
                import librosa
                stem = librosa.effects.pitch_shift(stem, sr=sr, n_steps=n_steps)
            if rate is not None:
                import librosa
                original_len = len(stem)
                stem = librosa.effects.time_stretch(stem, rate=rate)
                if len(stem) > original_len:
                    stem = stem[:original_len]
                elif len(stem) < original_len:
                    stem = np.pad(stem, (0, original_len - len(stem)))
            if rir is not None:
                pre_energy = np.sum(stem ** 2)
                stem = np.convolve(stem, rir, mode="full")[: len(stem)]
                if np.max(np.abs(stem)) > 0:
                    stem *= np.sqrt(pre_energy / (np.sum(stem ** 2) + 1e-8))
            out[i] = stem.astype(np.float32)
        return out


@jaxtyped(typechecker=beartype)
def power_set_subsets(
    stems: Float[np.ndarray, "N T"], min_size: int = 1
) -> list[tuple[list[int], Float[np.ndarray, "k T"]]]:
    """Generate all subsets of stems with at least `min_size` members."""
    n = stems.shape[0]
    subsets = []
    for size in range(min_size, n + 1):
        for combo in itertools.combinations(range(n), size):
            indices = list(combo)
            subset = stems[indices]
            subsets.append((indices, subset))
    return subsets


@jaxtyped(typechecker=beartype)
def make_training_pair(
    stems: Float[np.ndarray, "N T"],
    rng: np.random.Generator,
    min_stems: int = 2,
) -> tuple[Float[np.ndarray, "T"], Float[np.ndarray, "k T"]]:
    """Create a single training pair by randomly selecting a subset of stems."""
    n = stems.shape[0]
    k = rng.integers(min_stems, n + 1)
    indices = rng.choice(n, size=k, replace=False)
    indices.sort()
    selected = stems[indices]
    mixture = selected.sum(axis=0)
    return mixture, selected
