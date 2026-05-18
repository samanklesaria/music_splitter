"""Dagstuhl ChoirSet (DCS) dataset loader.

Files live flat in:
    data/dagstuhl_choirset/DagstuhlChoirSet/audio_wav_22050_mono/

Naming: DCS_{session}_{piece}_{take}_{voice_id}_{mic}.wav
  voice_id prefix → SATB index: S→0, A→1, T→2, B→3
  mic preference: HSM > DYN > LRX

Each dataset item is a (mixture, stems) pair: stems shape (4, T).
For takes with multiple singers per voice part (FullChoir), stems are summed
within each voice category.
"""

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from jaxtyping import Float, jaxtyped
from beartype import beartype

_FILE_RE = re.compile(r"^(DCS_\S+_\S+_\S+)_([SATB]\d+)_(HSM|DYN|LRX)\.wav$")
_MIC_RANK = {"HSM": 0, "DYN": 1, "LRX": 2}
_VOICE_IDX = {"S": 0, "A": 1, "T": 2, "B": 3}


def _parse_audio_dir(root: Path) -> Path:
    candidate = root / "DagstuhlChoirSet" / "audio_wav_22050_mono"
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(f"Expected audio dir not found: {candidate}")


def _discover_takes(audio_dir: Path) -> list[dict[str, list[tuple[int, Path]]]]:
    """Return a list of per-take voice maps.

    Each entry maps voice category letter → list of (mic_rank, path) for all
    singers in that category.  Only takes with all 4 SATB voices are kept.
    """
    # takes[take_key][voice_letter] = {singer_id: (best_rank, path)}
    from collections import defaultdict

    raw: dict[str, dict[str, dict[str, tuple[int, Path]]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    for f in sorted(audio_dir.iterdir()):
        m = _FILE_RE.match(f.name)
        if not m:
            continue
        take_key, voice_id, mic = m.group(1), m.group(2), m.group(3)
        voice_letter = voice_id[0]
        rank = _MIC_RANK[mic]
        existing = raw[take_key][voice_letter].get(voice_id)
        if existing is None or rank < existing[0]:
            raw[take_key][voice_letter][voice_id] = (rank, f)

    takes = []
    for take_key in sorted(raw):
        voices = raw[take_key]
        if set(voices.keys()) != {"S", "A", "T", "B"}:
            continue
        # flatten: voice_letter → list of (rank, path)
        entry = {letter: list(singers.values()) for letter, singers in voices.items()}
        takes.append(entry)

    return takes


@dataclass
class DagstuhlChoirSet:
    """Grain-compatible RandomAccessDataSource for the Dagstuhl ChoirSet.

    Each element is a (mixture, stems) pair where stems has shape (4, T).
    For FullChoir takes, stems within each SATB category are summed.
    """

    root: str | Path
    sample_rate: int = 22050
    split: str = "train"
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)
    _takes: list[dict] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        audio_dir = _parse_audio_dir(self.root)
        all_takes = _discover_takes(audio_dir)

        if not all_takes:
            raise FileNotFoundError(f"No complete SATB takes found under {self.root}")

        n = len(all_takes)
        n_train = math.ceil(n * self.split_ratios[0])
        n_val = math.ceil(n * self.split_ratios[1])

        if self.split == "train":
            self._takes = all_takes[:n_train]
        elif self.split == "val":
            self._takes = all_takes[n_train : n_train + n_val]
        else:
            self._takes = all_takes[n_train + n_val :]

    @jaxtyped(typechecker=beartype)
    def _load_wav(self, path: Path) -> Float[np.ndarray, "T"]:
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        audio = audio[:, 0]
        if sr != self.sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        return audio

    @jaxtyped(typechecker=beartype)
    def load_take(
        self, idx: int
    ) -> tuple[Float[np.ndarray, "T"], Float[np.ndarray, "4 T"]]:
        voice_map = self._takes[idx]
        stems_raw: list[Float[np.ndarray, "T"] | None] = [None] * 4

        for letter, singer_files in voice_map.items():
            stem_idx = _VOICE_IDX[letter]
            section: Float[np.ndarray, "T"] | None = None
            for _rank, path in singer_files:
                wav = self._load_wav(path)
                section = wav if section is None else section[: len(wav)] + wav[: len(section)]
            stems_raw[stem_idx] = section

        max_len = max(len(s) for s in stems_raw if s is not None)
        out = np.zeros((4, max_len), dtype=np.float32)
        for i, s in enumerate(stems_raw):
            if s is not None:
                out[i, : len(s)] = s

        mixture = out.sum(axis=0)
        return mixture, out

    def __len__(self) -> int:
        return len(self._takes)

    @jaxtyped(typechecker=beartype)
    def __getitem__(
        self, idx: int
    ) -> tuple[Float[np.ndarray, "T"], Float[np.ndarray, "4 T"]]:
        return self.load_take(idx)
