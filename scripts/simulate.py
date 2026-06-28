"""Closed-loop simulation: drive CARLA with a trained PlanT checkpoint.

The ego is steered by the model while NPCs run on autopilot. No CARLA 3D
rendering is used; the output is a BEV video (model inputs + predicted
waypoints, the same view as evaluate.py) written to an MP4.

Usage:
    python3 scripts/simulate.py
    python3 scripts/simulate.py --checkpoint checkpoints/best.ckpt \\
        --config configs/simulate.yaml \\
        --out data/simulation.mp4 \\
        --town Town03 --ticks 1000
"""

import argparse
import logging
from dataclasses import fields
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from tqdm import tqdm

from plant.carla.agent import AgentConfig, PlanTAgent
from plant.carla.controller import ControllerConfig
from plant.utils.visualizer import render_bev_features

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/simulate.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.ckpt")
    parser.add_argument("--out", default="data/simulation.mp4")
    parser.add_argument("--town", default=None, help="override config town")
    parser.add_argument(
        "--ticks", type=int, default=None, help="override config max_ticks"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.town is not None:
        cfg["town"] = args.town
    if args.ticks is not None:
        cfg["max_ticks"] = args.ticks

    agent_keys = {f.name for f in fields(AgentConfig)}
    agent_config = AgentConfig(**{k: v for k, v in cfg.items() if k in agent_keys})

    ctrl_cfg = cfg.get("controller", {})
    ctrl_keys = {f.name for f in fields(ControllerConfig)}
    controller_config = ControllerConfig(
        **{k: v for k, v in ctrl_cfg.items() if k in ctrl_keys}
    )

    max_ticks = cfg["max_ticks"]
    render_every = cfg.get("render_every", 1)
    video_fps = cfg.get("video_fps", cfg["fps"])
    bev_radius = cfg.get("bev_radius", 60.0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    agent = PlanTAgent(agent_config, str(ckpt_path), device, controller_config)
    writer = imageio.get_writer(
        out_path,
        fps=max(1, video_fps // render_every),
        codec="libx264",
        quality=5,
        macro_block_size=16,
    )

    n_frames = 0
    try:
        agent.setup()
        logger.info("Driving for up to %d ticks ...", max_ticks)
        for tick in tqdm(range(max_ticks), desc="simulating"):
            sample, pred_wp, _ = agent.run_step()
            if tick % render_every == 0:
                frame = render_bev_features(
                    sample,
                    "sim",
                    tick,
                    pred_wp=pred_wp,
                    bev_radius=bev_radius,
                    as_array=True,
                )
                writer.append_data(np.ascontiguousarray(frame))
                n_frames += 1
            if agent.collision:
                logger.warning("Stopping early: collision at tick %d", tick)
                break
    finally:
        writer.close()
        agent.cleanup()

    logger.info("Done. Wrote %d frames to %s", n_frames, out_path)


if __name__ == "__main__":
    main()
