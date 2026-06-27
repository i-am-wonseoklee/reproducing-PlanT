"""Offline evaluation of a trained PlanT checkpoint.

Loads a checkpoint, runs the model on the val split, computes ADE/FDE and loss
decomposition, then renders BEV previews with ground-truth (white) and predicted
(orange) waypoints side-by-side, writing them as BLOBs to a SQLite output DB.

Usage:
    python3 scripts/evaluate.py
    python3 scripts/evaluate.py --data data/frames.db \\
        --checkpoint checkpoints/best.ckpt \\
        --out data/eval.db
    python3 scripts/evaluate.py --episode episode_0001
"""

import argparse
import json
import logging
import sqlite3
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from plant.data.collate import collate_dynamic, split_episodes, to_device
from plant.data.dataset import Dataset
from plant.model.plant import PlanT
from plant.utils.visualizer import render_bev_features

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_eval(model: PlanT, loader: DataLoader, device: str) -> dict:
    """Compute loss and displacement errors over the given loader.

    Returns a dict with keys:
        loss, loss_wp, loss_aux  -- mean batch losses
        ade                      -- Average Displacement Error (m), mean over steps
        fde                      -- Final Displacement Error (m), last step
        step_errors              -- list of per-step L2 errors (m)
    """
    model.eval()
    totals = {"loss": 0.0, "loss_wp": 0.0, "loss_aux": 0.0}
    step_errors = None
    n_batches = 0
    n_samples = 0

    for batch in tqdm(loader, desc="evaluating", leave=False):
        batch = to_device(batch, device)
        pred = model(batch)
        _, logs = model.compute_loss(pred, batch)

        for k in totals:
            totals[k] += float(logs[k])

        l2 = torch.norm(
            pred["waypoints"] - batch["feature_waypoints"], dim=-1
        )  # (B, P)
        if step_errors is None:
            step_errors = torch.zeros(l2.shape[1], device=device)
        step_errors += l2.sum(dim=0)
        n_samples += l2.shape[0]
        n_batches += 1

    n_batches = max(1, n_batches)
    n_samples = max(1, n_samples)
    means = {k: v / n_batches for k, v in totals.items()}
    step_errors = (step_errors / n_samples).cpu().tolist()
    means["ade"] = sum(step_errors) / len(step_errors)
    means["fde"] = step_errors[-1]
    means["step_errors"] = step_errors
    return means


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


@torch.no_grad()
def render_episode(
    model: PlanT,
    ds: Dataset,
    episode: str,
    device: str,
    out_conn: sqlite3.Connection,
) -> int:
    """Run inference on one episode and write BEV images to out_conn.

    Returns the number of frames rendered.
    """
    model.eval()
    by_episode = ds.indices_by_episode()
    if episode not in by_episode:
        raise ValueError(f"episode {episode!r} not in dataset")
    ep_indices = by_episode[episode]

    rows = []
    for flat_idx in tqdm(ep_indices, desc=f"  {episode}", leave=False):
        sample = ds[flat_idx]
        ep, pos = ds._index[flat_idx]
        frame = ds._episode_frames[ep][pos]

        batch = collate_dynamic([sample])
        batch = to_device(batch, device)
        pred = model(batch)
        pred_wp = pred["waypoints"][0].cpu().numpy()  # (P, 2)

        png = render_bev_features(sample, ep, frame.tick, pred_wp=pred_wp)
        rows.append((ep, frame.tick, png))

    out_conn.execute("DELETE FROM previews WHERE episode = ?", (episode,))
    out_conn.executemany("INSERT INTO previews VALUES (?, ?, ?)", rows)
    out_conn.commit()
    return len(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/frames.db", help="path to frames DB")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best.ckpt",
        help="path to trained checkpoint",
    )
    parser.add_argument(
        "--out",
        default="data/eval.db",
        help="output SQLite DB for metrics and BEV images",
    )
    parser.add_argument(
        "--val-episodes",
        type=int,
        default=None,
        help="val episode count (default: from checkpoint train_config)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--episode",
        default=None,
        help="render only this episode; default renders all val episodes",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    db_path = Path(args.data)
    ckpt_path = Path(args.checkpoint)
    out_path = Path(args.out)

    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # Load checkpoint; model config is embedded so no separate --model-config needed.
    logger.info("Loading checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = ckpt["model_config"]
    train_cfg = ckpt.get("train_config", {})

    val_episodes = args.val_episodes or train_cfg.get("val_episodes", 2)

    model = PlanT(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model: %.1fM params, epoch %d, checkpoint val_loss=%.4f",
        n_params / 1e6,
        ckpt.get("epoch", -1),
        ckpt.get("val_loss", float("nan")),
    )

    # Dataset and val split.
    ds = Dataset(
        db_path,
        n_predictions=model_cfg["n_predictions"],
        n_obstacles=model_cfg["n_obstacles"],
        n_route_segments=model_cfg["n_route_segments"],
    )
    _, val_set, val_eps = split_episodes(ds, val_episodes)
    logger.info("Val episodes: %s  (%d samples)", val_eps, len(val_set))

    # Evaluate on the val split.
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_dynamic,
        pin_memory=(device == "cuda"),
    )
    metrics = run_eval(model, val_loader, device)
    logger.info(
        "val: loss=%.4f (wp=%.4f aux=%.4f)  ADE=%.4f m  FDE=%.4f m",
        metrics["loss"],
        metrics["loss_wp"],
        metrics["loss_aux"],
        metrics["ade"],
        metrics["fde"],
    )
    for i, e in enumerate(metrics["step_errors"], 1):
        logger.info("  step %d L2 = %.4f m", i, e)

    # Output DB.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(out_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            loss        REAL,
            loss_wp     REAL,
            loss_aux    REAL,
            ade         REAL,
            fde         REAL,
            step_errors TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS previews (
            episode TEXT    NOT NULL,
            tick    INTEGER NOT NULL,
            preview BLOB    NOT NULL
        )
    """)
    conn.execute("DELETE FROM metrics")
    conn.execute(
        "INSERT INTO metrics VALUES (?, ?, ?, ?, ?, ?)",
        (
            metrics["loss"],
            metrics["loss_wp"],
            metrics["loss_aux"],
            metrics["ade"],
            metrics["fde"],
            json.dumps(metrics["step_errors"]),
        ),
    )
    conn.commit()

    # Render BEV images.
    episodes_to_render = [args.episode] if args.episode else val_eps
    total_frames = 0
    for ep in episodes_to_render:
        n = render_episode(model, ds, ep, device, conn)
        logger.info("Rendered %d frames for %s", n, ep)
        total_frames += n

    conn.close()
    logger.info("Done. %d frames written to %s", total_frames, out_path)


if __name__ == "__main__":
    main()
