"""Collect driving data from CARLA using autopilot."""

import argparse
from dataclasses import fields
from pathlib import Path

import yaml

from plant.carla.collector import Collector, CollectorConfig
from plant.data.storage import Storage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/collect.yaml")
    parser.add_argument("--output", default="data/frames.db")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--ticks", type=int, default=2000)
    args = parser.parse_args()

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

    collector = Collector(config)
    for i in range(args.episodes):
        episode_id = start_episode + i
        print(f"Episode {episode_id}")
        collector.run(episode_id=episode_id, num_ticks=args.ticks)


if __name__ == "__main__":
    main()
