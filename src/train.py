"""Training entrypoint for SepReformer vocal separation."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from tensorboardX import SummaryWriter

from data.augmentation import AugmentationPipeline
from data.batch import BatchLoader
from data.dagstuhl import DagstuhlChoirSet
from data.jacappella import JaCappellaDataset
from losses.composite import composite_loss
from losses.sisdr import si_sdr
from model.sepreformer import SepReformer


def make_step(
    model: SepReformer,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    mixture: jnp.ndarray,
    targets: jnp.ndarray,
    use_pit: bool = True,
) -> tuple[SepReformer, optax.OptState, jnp.ndarray]:
    """Single training step.

    Args:
        model: Current model.
        opt_state: Optimizer state.
        optimizer: Optax optimizer.
        mixture: (B, T) batch of mixtures.
        targets: (B, N, T) batch of target stems.
        use_pit: Whether to use permutation invariant training.

    Returns:
        (updated_model, updated_opt_state, loss_value)
    """

    @eqx.filter_value_and_grad
    def loss_fn(model: SepReformer) -> jnp.ndarray:
        # vmap over batch dimension
        def single_loss(mix: jnp.ndarray, tgt: jnp.ndarray) -> jnp.ndarray:
            estimates = model(mix)  # (N, T)
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
    mixture: jnp.ndarray,
    targets: jnp.ndarray,
) -> tuple[SepReformer, optax.OptState, jnp.ndarray]:
    return make_step(model, opt_state, optimizer, mixture, targets, use_pit=True)


def evaluate(
    model: SepReformer,
    loader: BatchLoader,
) -> dict[str, float]:
    """Compute average SI-SDRi on validation set."""
    batches = loader.epoch_batches(epoch=0)
    total_sisdr = 0.0
    total_sisdr_mix = 0.0
    count = 0

    for mixture, targets in batches:
        B = mixture.shape[0]
        for b in range(B):
            estimates = model(mixture[b])  # (N, T)
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


def train(
    data_root: str = "data",
    num_epochs: int = 200,
    batch_size: int = 4,
    lr: float = 1e-4,
    num_stems: int = 4,
    dim: int = 256,
    num_heads: int = 8,
    ff_dim: int = 1024,
    num_blocks: int = 4,
    chunk_size: int = 64,
    segment_seconds: float = 4.0,
    log_dir: str = "runs",
    checkpoint_dir: str = "checkpoints",
    use_augmentation: bool = True,
    seed: int = 42,
) -> None:
    """Main training loop."""
    data_root = Path(data_root)
    key = jax.random.PRNGKey(seed)

    # --- Datasets ---
    datasets_train: list[JaCappellaDataset | DagstuhlChoirSet] = []
    datasets_val: list[JaCappellaDataset | DagstuhlChoirSet] = []

    jacappella_path = data_root / "jacappella"
    if jacappella_path.exists():
        datasets_train.append(
            JaCappellaDataset(
                jacappella_path,
                num_stems=num_stems,
                split="train",
                segment_seconds=segment_seconds,
            )
        )
        datasets_val.append(
            JaCappellaDataset(
                jacappella_path,
                num_stems=num_stems,
                split="val",
                segment_seconds=segment_seconds,
            )
        )
        print(f"JaCappella: {len(datasets_train[-1])} train, {len(datasets_val[-1])} val")

    dcs_path = data_root / "dagstuhl_choirset"
    if dcs_path.exists():
        datasets_train.append(
            DagstuhlChoirSet(
                dcs_path, split="train", segment_seconds=segment_seconds
            )
        )
        datasets_val.append(
            DagstuhlChoirSet(
                dcs_path, split="val", segment_seconds=segment_seconds
            )
        )
        print(f"DCS: {len(datasets_train[-1])} train, {len(datasets_val[-1])} val")

    if not datasets_train:
        raise RuntimeError(
            f"No datasets found in {data_root}. "
            "Run scripts/download_all.sh first."
        )

    # --- Augmentation ---
    augmentation = None
    if use_augmentation:
        augmentation = AugmentationPipeline(
            enable_pitch_shift=True,
            enable_time_stretch=True,
            enable_gain=True,
            enable_rir=False,
        )

    train_loader = BatchLoader(
        datasets=datasets_train,
        batch_size=batch_size,
        num_stems=num_stems,
        segment_seconds=segment_seconds,
        augmentation=augmentation,
    )
    val_loader = BatchLoader(
        datasets=datasets_val,
        batch_size=batch_size,
        num_stems=num_stems,
        segment_seconds=segment_seconds,
        augmentation=None,
    )

    # --- Model ---
    key, model_key = jax.random.split(key)
    model = SepReformer(
        num_stems=num_stems,
        dim=dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_blocks=num_blocks,
        chunk_size=chunk_size,
        key=model_key,
    )

    num_params = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
    print(f"Model parameters: {num_params:,}")

    # --- Optimizer ---
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=lr * 0.01,
        peak_value=lr,
        warmup_steps=500,
        decay_steps=num_epochs * train_loader.total_songs // batch_size,
    )
    optimizer = optax.adamw(schedule, weight_decay=1e-2)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # --- Logging ---
    writer = SummaryWriter(log_dir)
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # --- Training loop ---
    global_step = 0
    best_sisdr_i = float("-inf")

    for epoch in range(num_epochs):
        t0 = time.time()
        batches = train_loader.epoch_batches(epoch)
        epoch_loss = 0.0

        for mixture, targets in batches:
            model, opt_state, loss = jit_step(
                model, opt_state, optimizer, mixture, targets
            )
            epoch_loss += float(loss)
            global_step += 1

            if global_step % 50 == 0:
                writer.add_scalar("train/loss", float(loss), global_step)

        avg_loss = epoch_loss / max(len(batches), 1)
        elapsed = time.time() - t0

        # --- Validation ---
        if (epoch + 1) % 5 == 0:
            val_metrics = evaluate(model, val_loader)
            writer.add_scalar("val/si_sdr", val_metrics["si_sdr"], global_step)
            writer.add_scalar("val/si_sdri", val_metrics["si_sdri"], global_step)

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SepReformer for vocal separation")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-stems", type=int, default=4)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ff-dim", type=int, default=1024)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--segment-seconds", type=float, default=4.0)
    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        data_root=args.data_root,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_stems=args.num_stems,
        dim=args.dim,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        num_blocks=args.num_blocks,
        chunk_size=args.chunk_size,
        segment_seconds=args.segment_seconds,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        use_augmentation=not args.no_augmentation,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
