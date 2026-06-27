"""Collect driving data from CARLA using autopilot."""

import argparse
import logging
import multiprocessing as mp
from dataclasses import fields
from pathlib import Path

import numpy as np
import yaml

from plant.carla.collector import Collector, CollectorConfig
from plant.data.storage import Storage

logger = logging.getLogger(__name__)


def _run_episode(config: CollectorConfig, episode_id: int, num_ticks: int) -> None:
    """Entry point for per-episode subprocess.

    Isolated so that a CARLA C++ abort (TimeoutException thrown from the TM
    background thread, uncatchable in Python) kills only this subprocess and
    not the outer collection loop.
    """
    # Fork inherits the parent's numpy RNG state; reseed from OS entropy so
    # each episode gets independent random choices (town, spawn point, etc.).
    np.random.seed()
    collector = Collector(config)
    collector.run(episode_id=episode_id, num_ticks=num_ticks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/collector.yaml")
    parser.add_argument("--output", default="data/frames.db")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--ticks", type=int, default=2000)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    valid_keys = {f.name for f in fields(CollectorConfig)}
    config = CollectorConfig(
        output_path=args.output,
        **{k: v for k, v in cfg.items() if k in valid_keys},
    )

    db_path = Path(config.output_path)
    start_episode = 0
    if db_path.exists():
        with Storage(db_path) as storage:
            start_episode = len(storage.episodes())

    i = 0
    while i < args.episodes:
        episode_id = start_episode + i
        logger.info("Episode %d", episode_id)
        p = mp.Process(
            target=_run_episode,
            args=(config, episode_id, args.ticks),
            daemon=True,
        )
        p.start()
        p.join()
        if p.exitcode != 0:
            logger.warning(
                "Episode %d crashed (exit code %d), retrying.", episode_id, p.exitcode
            )
            continue
        i += 1


if __name__ == "__main__":
    main()
