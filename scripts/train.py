"""Train PlanT on collected CARLA frames.

Loads a model config (architecture) and a train config (loop hyperparameters),
builds an episode level train/val split, and runs the standard supervised loop:

    pred = model(batch)
    loss, logs = model.compute_loss(pred, batch)

The combined L1 waypoint loss plus weighted auxiliary cross entropy is computed
inside the model, so this script only handles batching, optimization, logging,
and checkpointing.
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from plant.data.collate import collate_dynamic, split_episodes, to_device
from plant.data.dataset import Dataset
from plant.model.plant import PlanT

logger = logging.getLogger(__name__)


def build_optimizer(model: nn.Module, lr: float, weight_decay: float, betas):
    """AdamW with minGPT style parameter groups.

    Weight decay is applied only to nn.Linear weights. Everything else (biases,
    LayerNorm and Embedding weights, GRU recurrent weights, the learned CLS
    token) is excluded.

    Note: nn.MultiheadAttention inside nn.TransformerEncoder stores its input
    projection as a raw Parameter (in_proj_weight), so it falls into the no decay
    group here. The original PlanT uses a custom GPT, so an exact match is not
    possible; this small deviation is accepted.
    """
    decay = set()
    for module in model.modules():
        if isinstance(module, nn.Linear):
            decay.add(id(module.weight))

    decay_params, no_decay_params = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in decay:
            decay_params.append(p)
        else:
            no_decay_params.append(p)

    groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return AdamW(groups, lr=lr, betas=tuple(betas))


@dataclass
class LoopCtx:
    """Constant objects and knobs shared across train and eval epochs.

    Bundling these keeps run_epoch's call sites short; the per call arguments
    (which loader, train vs eval, step counter) stay explicit.
    """

    optimizer: torch.optim.Optimizer | None = None
    scaler: torch.amp.GradScaler | None = None
    grad_clip: float = 0.0
    use_amp: bool = False
    writer: SummaryWriter | None = None
    log_interval: int = 50


def run_epoch(
    model,
    loader,
    device,
    ctx: LoopCtx,
    *,
    train: bool,
    limit_batches=None,
    global_step=0,
    desc="",
):
    """Run one train or eval epoch. Returns (mean_logs, global_step).

    During training the per step scalars are logged under train_step/* indexed by
    global_step; the per epoch train/* and val/* means (logged by the caller) share
    the epoch axis so train and val curves line up.
    """
    model.train(train)
    totals = {"loss": 0.0, "loss_wp": 0.0, "loss_aux": 0.0}
    n_batches = 0

    bar = tqdm(loader, desc=desc, leave=False)
    for batch in bar:
        if limit_batches is not None and n_batches >= limit_batches:
            break
        batch = to_device(batch, device)

        with torch.set_grad_enabled(train):
            with torch.autocast("cuda", enabled=ctx.use_amp):
                pred = model(batch)
                loss, logs = model.compute_loss(pred, batch)

            if train:
                ctx.optimizer.zero_grad(set_to_none=True)
                if ctx.use_amp:
                    ctx.scaler.scale(loss).backward()
                    if ctx.grad_clip > 0:
                        ctx.scaler.unscale_(ctx.optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), ctx.grad_clip)
                    ctx.scaler.step(ctx.optimizer)
                    ctx.scaler.update()
                else:
                    loss.backward()
                    if ctx.grad_clip > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), ctx.grad_clip)
                    ctx.optimizer.step()

        for k in totals:
            totals[k] += float(logs[k])
        n_batches += 1
        bar.set_postfix(
            loss=float(logs["loss"]),
            wp=float(logs["loss_wp"]),
            aux=float(logs["loss_aux"]),
        )

        if train and ctx.writer is not None and global_step % ctx.log_interval == 0:
            for k in totals:
                ctx.writer.add_scalar(f"train_step/{k}", float(logs[k]), global_step)
            ctx.writer.add_scalar(
                "train_step/lr", ctx.optimizer.param_groups[0]["lr"], global_step
            )
        if train:
            global_step += 1

    n_batches = max(1, n_batches)
    mean_logs = {k: v / n_batches for k, v in totals.items()}
    return mean_logs, global_step


def main():
    parser = argparse.ArgumentParser(description="Train PlanT")
    parser.add_argument("--model-config", default="configs/plant.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--data", default="data/frames.db")
    parser.add_argument(
        "--run-name",
        default=None,
        help="subdirectory under log_dir/ckpt_dir to keep runs separate",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)
    with open(args.train_config) as f:
        train_cfg = yaml.safe_load(f)

    torch.manual_seed(train_cfg["seed"])
    np.random.seed(train_cfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # Data.
    ds = Dataset(
        args.data,
        n_predictions=model_cfg["n_predictions"],
        n_obstacles=model_cfg["n_obstacles"],
        n_route_segments=model_cfg["n_route_segments"],
    )
    train_set, val_set, _ = split_episodes(ds, train_cfg["val_episodes"])
    logger.info("Samples: train=%d val=%d", len(train_set), len(val_set))

    common = dict(
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
        collate_fn=collate_dynamic,
        pin_memory=(device == "cuda"),
    )
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_set, shuffle=False, **common)

    # Model, optimizer, scheduler.
    model = PlanT(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %.1fM", n_params / 1e6)

    optimizer = build_optimizer(
        model,
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
        betas=train_cfg["betas"],
    )
    decay_epoch = train_cfg["lr_decay_epoch"]
    scheduler = MultiStepLR(
        optimizer,
        milestones=[decay_epoch, decay_epoch + 10],
        gamma=train_cfg["lr_decay_gamma"],
    )

    use_amp = bool(train_cfg["amp"]) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Logging and checkpoints. A run name (if given) keeps separate runs from
    # overwriting each other's checkpoints and overlaying each other's curves.
    log_dir = Path(train_cfg["log_dir"])
    ckpt_dir = Path(train_cfg["ckpt_dir"])
    if args.run_name:
        log_dir = log_dir / args.run_name
        ckpt_dir = ckpt_dir / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)

    ctx = LoopCtx(
        optimizer=optimizer,
        scaler=scaler,
        grad_clip=train_cfg["grad_clip"],
        use_amp=use_amp,
        writer=writer,
        log_interval=train_cfg["log_interval"],
    )

    best_val = float("inf")
    global_step = 0

    for epoch in range(train_cfg["epochs"]):
        train_logs, global_step = run_epoch(
            model,
            train_loader,
            device,
            ctx,
            train=True,
            limit_batches=train_cfg["limit_train_batches"],
            global_step=global_step,
            desc=f"epoch {epoch} train",
        )
        scheduler.step()

        val_logs, _ = run_epoch(
            model,
            val_loader,
            device,
            ctx,
            train=False,
            limit_batches=train_cfg["limit_val_batches"],
            desc=f"epoch {epoch} val",
        )
        # Per epoch means on a shared epoch axis so train and val line up.
        for k, v in train_logs.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        for k, v in val_logs.items():
            writer.add_scalar(f"val/{k}", v, epoch)

        logger.info(
            "epoch %d: train_loss=%.4f (wp=%.4f aux=%.4f) "
            "val_loss=%.4f (wp=%.4f aux=%.4f)",
            epoch,
            train_logs["loss"],
            train_logs["loss_wp"],
            train_logs["loss_aux"],
            val_logs["loss"],
            val_logs["loss_wp"],
            val_logs["loss_aux"],
        )

        ckpt = {
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "model_config": model_cfg,
            "train_config": train_cfg,
            "val_loss": val_logs["loss"],
        }
        torch.save(ckpt, ckpt_dir / "last.ckpt")
        if val_logs["loss"] < best_val:
            best_val = val_logs["loss"]
            torch.save(ckpt, ckpt_dir / "best.ckpt")

    writer.close()
    logger.info("Done. Best val loss: %.4f", best_val)


if __name__ == "__main__":
    main()
