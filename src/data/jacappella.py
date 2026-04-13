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

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

STEM_NAMES = ("lead", "soprano", "alto", "tenor", "bass", "vocal_percussion")

# For 4-stem grouping used in baseline training:
#   0: lead, 1: high harmony (soprano), 2: mid harmony (alto+tenor), 3: low harmony (bass)
# vocal_percussion is excluded from harmonic separation.
FOUR_STEM_GROUPS = {
    "lead": 0,
    "soprano": 1,
    "alto": 2,
    "tenor": 2,
    "bass": 3,
}


@dataclass
class JaCappellaDataset:
    """Loads and serves chunks from the JaCappella corpus."""

    root: str | Path
    sample_rate: int = 44100
    segment_seconds: float = 4.0
    num_stems: int = 4  # 4 = grouped, 6 = all individual stems
    split: str = "train"  # "train", "val", "test"
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)
    _songs: list[Path] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        all_songs = sorted(
            p for p in self.root.iterdir() if p.is_dir() and not p.name.startswith(".")
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

    @property
    def segment_samples(self) -> int:
        return int(self.segment_seconds * self.sample_rate)

    def _load_wav(self, path: Path) -> np.ndarray:
        """Load a wav file, resample if needed, return mono float32."""
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        audio = audio[:, 0]  # take first channel if stereo
        if sr != self.sample_rate:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        return audio

    def _load_stems(self, song_dir: Path) -> dict[str, np.ndarray]:
        """Load all available stems for a song."""
        stems = {}
        for name in STEM_NAMES:
            path = song_dir / f"{name}.wav"
            if path.exists():
                stems[name] = self._load_wav(path)
        return stems

    def _group_stems(self, stems: dict[str, np.ndarray], length: int) -> np.ndarray:
        """Group 6 stems into `num_stems` output channels.

        Returns array of shape (num_stems, length).
        """
        if self.num_stems == 6:
            out = np.zeros((6, length), dtype=np.float32)
            for i, name in enumerate(STEM_NAMES):
                if name in stems:
                    s = stems[name][:length]
                    out[i, : len(s)] = s
            return out

        # 4-stem grouping
        out = np.zeros((4, length), dtype=np.float32)
        for name, group_idx in FOUR_STEM_GROUPS.items():
            if name in stems:
                s = stems[name][:length]
                out[group_idx, : len(s)] += s
        return out

    def load_song(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Load full song. Returns (mixture, stems) where stems shape is (num_stems, T)."""
        song_dir = self._songs[idx]
        stems = self._load_stems(song_dir)
        if not stems:
            raise RuntimeError(f"No stems found in {song_dir}")

        max_len = max(len(s) for s in stems.values())
        grouped = self._group_stems(stems, max_len)
        mixture = grouped.sum(axis=0)
        return mixture, grouped

    def get_segment(
        self, idx: int, rng: np.random.Generator | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get a random segment from song `idx`.

        Returns (mixture_segment, stems_segment) where:
            mixture_segment: (segment_samples,)
            stems_segment: (num_stems, segment_samples)
        """
        mixture, stems = self.load_song(idx)
        total = mixture.shape[0]
        seg = self.segment_samples

        if total <= seg:
            # Pad if shorter
            mix_out = np.zeros(seg, dtype=np.float32)
            stem_out = np.zeros((self.num_stems, seg), dtype=np.float32)
            mix_out[:total] = mixture
            stem_out[:, :total] = stems
            return mix_out, stem_out

        if rng is None:
            rng = np.random.default_rng()
        start = rng.integers(0, total - seg)
        return mixture[start : start + seg], stems[:, start : start + seg]

    def __len__(self) -> int:
        return len(self._songs)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        return self.get_segment(idx)
