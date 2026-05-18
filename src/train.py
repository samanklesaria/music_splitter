"""Training entrypoint for SepReformer vocal separation."""

from __future__ import annotations

import time
from pathlib import Path
from datetime import datetime

import numpy as np
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import optax
from typing import Optional
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype
from tensorboardX import SummaryWriter
import fire
import grain

from data.augmentation import AugmentationPipeline
from data.batch import build_loader
from data.dagstuhl import DagstuhlChoirSet
from data.jacappella import JaCappellaDataset
from losses.composite import composite_loss
from losses.sisdr import si_sdr
from model.sepreformer import SepReformer


@jaxtyped(typechecker=beartype)
def make_step(
    model: SepReformer,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    mixture: Float[Array, "B T"],
    targets: Float[Array, "B N T"],
    use_pit: bool = True,
) -> tuple[SepReformer, optax.OptState, Float[Array, ""]]:
    """Single training step."""

    @eqx.filter_value_and_grad
    def loss_fn(model: SepReformer) -> Float[Array, ""]:
        # vmap over batch dimension
        @jaxtyped(typechecker=beartype)
        def single_loss(
            mix: Float[Array, "T"], tgt: Float[Array, "N T"]
        ) -> Float[Array, ""]:
            estimates = model(mix)
            return composite_loss(estimates, tgt, use_pit=use_pit)

        losses = jax.vmap(single_loss)(mixture, targets)
        return jnp.mean(losses)

    loss, grads = loss_fn(model)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


@eqx.filter_jit
def jit_step(
    model: SepReformer,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    mixture: Float[Array, "B T"],
    targets: Float[Array, "B N T"],
) -> tuple[SepReformer, optax.OptState, Float[Array, ""]]:
    return make_step(model, opt_state, optimizer, mixture, targets, use_pit=True)


def evaluate(
    model: SepReformer,
    val_ds: grain.MapDataset,
) -> dict[str, float]:
    """Compute average SI-SDRi on validation set."""
    total_sisdr = 0.0
    total_sisdr_mix = 0.0
    count = 0

    for mixture, targets in val_ds:
        B = mixture.shape[0]
        for b in range(B):
            estimates = model(mixture[b])
            for n in range(estimates.shape[0]):
                est_sisdr = float(si_sdr(estimates[n], targets[b, n]))
                mix_sisdr = float(si_sdr(mixture[b], targets[b, n]))
                total_sisdr += est_sisdr
                total_sisdr_mix += mix_sisdr
                count += 1

    avg_sisdr = total_sisdr / max(count, 1)
    avg_sisdr_mix = total_sisdr_mix / max(count, 1)
    return {
        "si_sdr": avg_sisdr,
        "si_sdri": avg_sisdr - avg_sisdr_mix,
    }


def log_audio_samples(
    model: SepReformer,
    val_ds: grain.MapDataset,
    writer: SummaryWriter,
    global_step: int,
    sample_rate: int,
) -> None:
    mixture, _ = next(iter(val_ds))
    mix_np = np.array(mixture[0])        # (T,)
    est_np = np.array(model(mixture[0])) # (N, T)

    peak = np.max(np.abs(mix_np))
    scale = 0.99 / peak if peak > 0 else 1.0

    writer.add_audio("val/audio/mixture", mix_np * scale, global_step, sample_rate=sample_rate)
    for n in range(est_np.shape[0]):
        writer.add_audio(f"val/audio/estimate_{n}", est_np[n] * scale, global_step, sample_rate=sample_rate)


def train(
    data_root: str = "/space/samanklesaria/data",
    num_epochs: int = 200,
    batch_size: int = 1,
    lr: float = 1e-4,
    num_stems: int = 6,
    dim: int = 128, # 256,
    num_heads: int = 8,
    ff_dim: int = 512, # 1024,
    num_sep_blocks: int = 2,
    num_rec_blocks: int = 2,
    chunk_size: int = 64,
    segment_seconds: float = 2.0,
    sample_rate: int = 44100,
    run_name: Optional[str] = None,
    load_from: Optional[str] = None,
    use_augmentation: bool = True,
    seed: int = 42,
) -> None:
    if not run_name:
        run_name = datetime.today().strftime('%m-%d_%H_%M_%S')
    data_root = Path(data_root)
    key = jax.random.PRNGKey(seed)

    # --- Datasets ---
    datasets_train: list[JaCappellaDataset | DagstuhlChoirSet] = []
    datasets_val: list[JaCappellaDataset | DagstuhlChoirSet] = []

    jacappella_path = data_root / "jacappella"
    if jacappella_path.exists():
        datasets_train.append(
            JaCappellaDataset(jacappella_path, num_stems=num_stems, split="train", sample_rate=sample_rate)
        )
        datasets_val.append(
            JaCappellaDataset(jacappella_path, num_stems=num_stems, split="val", sample_rate=sample_rate)
        )
        print(f"JaCappella: {len(datasets_train[-1])} train, {len(datasets_val[-1])} val")

    dcs_path = data_root / "dagstuhl_choirset"
    if dcs_path.exists():
        datasets_train.append(DagstuhlChoirSet(dcs_path, split="train", sample_rate=sample_rate))
        datasets_val.append(DagstuhlChoirSet(dcs_path, split="val", sample_rate=sample_rate))
        print(f"DCS: {len(datasets_train[-1])} train, {len(datasets_val[-1])} val")

    if not datasets_train:
        raise RuntimeError(
            f"No datasets found in {data_root}. "
            "Run scripts/download_all.sh first."
        )

    total_train_songs = sum(len(d) for d in datasets_train)

    # --- Augmentation ---
    augmentation = None
    if use_augmentation:
        augmentation = AugmentationPipeline()

    train_ds = build_loader(
        datasets=datasets_train,
        batch_size=batch_size,
        segment_seconds=segment_seconds,
        sample_rate=sample_rate,
        augmentation=augmentation,
        seed=seed,
    )
    val_ds = build_loader(
        datasets=datasets_val,
        batch_size=batch_size,
        segment_seconds=segment_seconds,
        sample_rate=sample_rate,
        augmentation=None,
        seed=seed,
    )

    # --- Model ---
    key, model_key = jax.random.split(key)
    model = SepReformer(
        num_stems=num_stems,
        dim=dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_sep_blocks=num_sep_blocks,
        num_rec_blocks=num_rec_blocks,
        chunk_size=chunk_size,
        key=model_key,
    )

    if load_from is not None:
        model = eqx.tree_deserialise_leaves(load_from, model)
        print(f"Loaded checkpoint from {load_from}")

    num_params = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
    print(f"Model parameters: {num_params:,}")

    # --- Optimizer ---
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=lr * 0.01,
        peak_value=lr,
        warmup_steps=500,
        decay_steps=num_epochs * total_train_songs // batch_size,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule, weight_decay=1e-2),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # --- FSDP sharding over batch ---
    mesh = Mesh(np.array(jax.devices()[:2]), ('data',))
    replicated = NamedSharding(mesh, PartitionSpec())
    batch_sharded = NamedSharding(mesh, PartitionSpec('data'))
    model = jax.device_put(model, replicated)
    opt_state = jax.device_put(opt_state, replicated)

    # --- Logging ---
    writer = SummaryWriter(Path("runs") / run_name)
    checkpoint_path = Path("checkpoints") / run_name
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # --- Training loop ---
    global_step = 0
    best_sisdr_i = float("-inf")

    for epoch in range(num_epochs):
        t0 = time.time()
        epoch_loss = 0.0
        num_batches = 0

        for mixture, targets in train_ds:
            mixture = jax.device_put(jnp.array(mixture), batch_sharded)
            targets = jax.device_put(jnp.array(targets), batch_sharded)
            model, opt_state, loss = jit_step(
                model, opt_state, optimizer, mixture, targets
            )
            epoch_loss += float(loss)
            global_step += 1
            num_batches += 1

            if global_step % 50 == 0:
                writer.add_scalar("train/loss", float(loss), global_step)

        avg_loss = epoch_loss / max(num_batches, 1)
        elapsed = time.time() - t0

        # --- Validation ---
        if (epoch + 1) % 5 == 0:
            val_metrics = evaluate(model, val_ds)
            writer.add_scalar("val/si_sdr", val_metrics["si_sdr"], global_step)
            writer.add_scalar("val/si_sdri", val_metrics["si_sdri"], global_step)
            log_audio_samples(model, val_ds, writer, global_step, sample_rate)

            print(
                f"Epoch {epoch + 1:3d} | loss={avg_loss:.4f} | "
                f"SI-SDRi={val_metrics['si_sdri']:.2f} dB | "
                f"{elapsed:.1f}s"
            )

            # Save best model
            if val_metrics["si_sdri"] > best_sisdr_i:
                best_sisdr_i = val_metrics["si_sdri"]
                eqx.tree_serialise_leaves(
                    str(checkpoint_path / "best_model.eqx"), model
                )
                print(f"  → New best SI-SDRi: {best_sisdr_i:.2f} dB")
        else:
            print(f"Epoch {epoch + 1:3d} | loss={avg_loss:.4f} | {elapsed:.1f}s")

        # Periodic checkpoint
        if (epoch + 1) % 20 == 0:
            eqx.tree_serialise_leaves(
                str(checkpoint_path / f"model_epoch{epoch + 1:03d}.eqx"), model
            )

    writer.close()
    print(f"Training complete. Best SI-SDRi: {best_sisdr_i:.2f} dB")

if __name__ == "__main__":
    fire.Fire(train)
