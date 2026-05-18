"""JaCappella dataset loader.

The JaCappella corpus contains 35 a cappella songs with 6 isolated stems each:
lead, soprano, alto, tenor, bass, vocal_percussion.

Directory structure (expected after download):
    data/jacappella/
        song_001/
            lead.wav
            soprano.wav
            alto.wav
            tenor.wav
            bass.wav
            vocal_percussion.wav
            mixture.wav
        song_002/
        ...
"""

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from jaxtyping import Float, jaxtyped
from beartype import beartype

STEM_NAMES = ("lead_vocal", "soprano", "alto", "tenor", "bass", "vocal_percussion")

@dataclass
class JaCappellaDataset:
    """Grain-compatible RandomAccessDataSource for the JaCappella corpus.

    Each element is a (mixture, stems) pair where stems has shape (num_stems, T).
    """

    root: str | Path
    sample_rate: int = 44100
    split: str = "train"  # "train", "val", "test"
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)
    _songs: list[Path] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        all_songs = sorted(
            song
            for genre in self.root.iterdir()
            if genre.is_dir() and not genre.name.startswith(".")
            for song in genre.iterdir()
            if song.is_dir() and not song.name.startswith(".")
        )
        if not all_songs:
            raise FileNotFoundError(f"No song directories found in {self.root}")

        n = len(all_songs)
        n_train = math.ceil(n * self.split_ratios[0])
        n_val = math.ceil(n * self.split_ratios[1])

        if self.split == "train":
            self._songs = all_songs[:n_train]
        elif self.split == "val":
            self._songs = all_songs[n_train : n_train + n_val]
        else:
            self._songs = all_songs[n_train + n_val :]

    @jaxtyped(typechecker=beartype)
    def _load_wav(self, path: Path) -> Float[np.ndarray, "T"]:
        """Load a wav file, resample if needed, return mono float32."""
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        audio = audio[:, 0]  # take first channel if stereo
        if sr != self.sample_rate:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        return audio

    @jaxtyped(typechecker=beartype)
    def _load_stems(self, song_dir: Path) -> dict[str, Float[np.ndarray, "T"]]:
        """Load all available stems for a song."""
        stems = {}
        for name in STEM_NAMES:
            path = song_dir / f"{name}.wav"
            if path.exists():
                stems[name] = self._load_wav(path)
        return stems

    def _group_stems(
        self, stems: dict[str, Float[np.ndarray, "T"]], length: int
    ) -> Float[np.ndarray, "N T"]:
        out = np.zeros((6, length), dtype=np.float32)
        for i, name in enumerate(STEM_NAMES):
            if name in stems:
                s = stems[name][:length]
                out[i, : len(s)] = s
        return out

    def load_song(self, idx: int) -> tuple[Float[np.ndarray, "T"], Float[np.ndarray, "N T"]]:
        """Load full song."""
        song_dir = self._songs[idx]
        stems = self._load_stems(song_dir)
        if not stems:
            raise RuntimeError(f"No stems found in {song_dir}")

        max_len = max(len(s) for s in stems.values())
        grouped = self._group_stems(stems, max_len)
        mixture = grouped.sum(axis=0)
        return mixture, grouped

    def __len__(self) -> int:
        return len(self._songs)

    def __getitem__(self, idx: int) -> tuple[Float[np.ndarray, "T"], Float[np.ndarray, "N T"]]:
        return self.load_song(idx)
