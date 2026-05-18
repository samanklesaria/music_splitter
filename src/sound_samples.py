"""Log a batch of augmented audio samples to TensorBoard for inspection."""

from pathlib import Path

import numpy as np
from tensorboardX import SummaryWriter
import fire

from data.augmentation import AugmentationPipeline
from data.batch import build_loader
from data.dagstuhl import DagstuhlChoirSet
from data.jacappella import JaCappellaDataset


def log_samples(
    data_root: str = "/space/samanklesaria/data",
    batch_size: int = 4,
    num_stems: int = 6,
    segment_seconds: float = 4.0,
    sample_rate: int = 44100,
    seed: int = 0,
) -> None:
    data_root = Path(data_root)
    named_datasets = []

    jacappella_path = data_root / "jacappella"
    if jacappella_path.exists():
        named_datasets.append(("jacappella", JaCappellaDataset(jacappella_path, split="train", sample_rate=sample_rate)))

    dcs_path = data_root / "dagstuhl_choirset"
    if dcs_path.exists():
        named_datasets.append(("dagstuhl", DagstuhlChoirSet(dcs_path, split="train", sample_rate=sample_rate)))

    if not named_datasets:
        raise RuntimeError(f"No datasets found in {data_root}.")

    writer = SummaryWriter("sound_samples")

    for dataset_name, dataset in named_datasets:
        loader = build_loader(
            datasets=[dataset],
            batch_size=batch_size,
            num_stems=num_stems,
            segment_seconds=segment_seconds,
            sample_rate=sample_rate,
            augmentation=AugmentationPipeline(),
            seed=seed,
        )

        mixture, stems = next(iter(loader))
        mixture = np.array(mixture)  # (B, T)
        stems = np.array(stems)      # (B, N, T)

        for b in range(mixture.shape[0]):
            peak = np.max(np.abs(mixture[b]))
            scale = 0.99 / peak if peak > 0 else 1.0
            writer.add_audio(
                f"{dataset_name}/sample_{b}/mixture",
                mixture[b] * scale,
                sample_rate=sample_rate,
            )
            for n in range(stems.shape[1]):
                writer.add_audio(
                    f"{dataset_name}/sample_{b}/stem_{n}",
                    stems[b, n] * scale,
                    sample_rate=sample_rate,
                )

    writer.close()


if __name__ == "__main__":
    fire.Fire(log_samples)
