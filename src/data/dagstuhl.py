"""Dagstuhl ChoirSet (DCS) dataset loader.

The DCS contains SATB choral recordings with multiple microphone types per singer.
We use the headset microphone signals as the cleanest isolated stems.

Expected directory structure after download:
    data/dagstuhl_choirset/
        DCS_LI_QuartetA_Take01/
            DCS_LI_QuartetA_Take01_S1_HS.wav   # Soprano, headset mic
            DCS_LI_QuartetA_Take01_S2_HS.wav   # Alto, headset mic
            DCS_LI_QuartetA_Take01_S3_HS.wav   # Tenor, headset mic
            DCS_LI_QuartetA_Take01_S4_HS.wav   # Bass, headset mic
            ...
        DCS_LI_QuartetA_Take02/
        ...

Voice mapping: S1=Soprano, S2=Alto, S3=Tenor, S4=Bass
Mic types: HS=headset (preferred), LM=lavalier, CM=close mic
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

VOICE_PARTS = ("S1", "S2", "S3", "S4")  # Soprano, Alto, Tenor, Bass
PREFERRED_MIC = "HS"  # headset mic — cleanest signal


@dataclass
class DagstuhlChoirSet:
    """Loads and serves chunks from the Dagstuhl ChoirSet."""

    root: str | Path
    sample_rate: int = 44100
    segment_seconds: float = 4.0
    split: str = "train"
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)
    _takes: list[Path] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        all_takes = sorted(
            p
            for p in self.root.iterdir()
            if p.is_dir() and p.name.startswith("DCS_")
        )
        if not all_takes:
            raise FileNotFoundError(f"No DCS take directories found in {self.root}")

        n = len(all_takes)
        n_train = math.ceil(n * self.split_ratios[0])
        n_val = math.ceil(n * self.split_ratios[1])

        if self.split == "train":
            self._takes = all_takes[:n_train]
        elif self.split == "val":
            self._takes = all_takes[n_train : n_train + n_val]
        else:
            self._takes = all_takes[n_train + n_val :]

    @property
    def segment_samples(self) -> int:
        return int(self.segment_seconds * self.sample_rate)

    def _find_stem_file(self, take_dir: Path, voice: str) -> Path | None:
        """Find the headset mic file for a given voice part."""
        pattern = f"*_{voice}_{PREFERRED_MIC}.wav"
        matches = list(take_dir.glob(pattern))
        if matches:
            return matches[0]
        # Fallback: try any mic type
        pattern = f"*_{voice}_*.wav"
        matches = list(take_dir.glob(pattern))
        return matches[0] if matches else None

    def _load_wav(self, path: Path) -> np.ndarray:
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        audio = audio[:, 0]
        if sr != self.sample_rate:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        return audio

    def load_take(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Load a full take. Returns (mixture, stems) where stems is (4, T)."""
        take_dir = self._takes[idx]
        stems = []
        max_len = 0

        for voice in VOICE_PARTS:
            path = self._find_stem_file(take_dir, voice)
            if path is not None:
                audio = self._load_wav(path)
                stems.append(audio)
                max_len = max(max_len, len(audio))
            else:
                stems.append(None)

        if max_len == 0:
            raise RuntimeError(f"No stems found in {take_dir}")

        out = np.zeros((4, max_len), dtype=np.float32)
        for i, s in enumerate(stems):
            if s is not None:
                out[i, : len(s)] = s

        mixture = out.sum(axis=0)
        return mixture, out

    def get_segment(
        self, idx: int, rng: np.random.Generator | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get a random segment from take `idx`.

        Returns (mixture_segment, stems_segment).
        """
        mixture, stems = self.load_take(idx)
        total = mixture.shape[0]
        seg = self.segment_samples

        if total <= seg:
            mix_out = np.zeros(seg, dtype=np.float32)
            stem_out = np.zeros((4, seg), dtype=np.float32)
            mix_out[:total] = mixture
            stem_out[:, :total] = stems
            return mix_out, stem_out

        if rng is None:
            rng = np.random.default_rng()
        start = rng.integers(0, total - seg)
        return mixture[start : start + seg], stems[:, start : start + seg]

    def __len__(self) -> int:
        return len(self._takes)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        return self.get_segment(idx)
