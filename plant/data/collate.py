"""Collate utilities shared between train and evaluate scripts."""

import numpy as np
import torch
from torch.utils.data import Subset

from plant.data.dataset import Dataset

_FIXED_KEYS = (
    "feature_route_segments",
    "feature_traffic_light",
    "feature_target_point",
    "feature_waypoints",
)
_OBSTACLE_KEYS = (
    "feature_obstacles",
    "mask_obstacles",
    "feature_future_obstacles",
    "mask_future_obstacles",
)


def collate_dynamic(samples: list[dict]) -> dict:
    """Stack samples and trim obstacle tensors to the batch maximum.

    The Dataset zero pads obstacles to a fixed cap (n_obstacles). Most frames use
    far fewer slots, so we trim every obstacle aligned tensor to the largest valid
    count in this batch. The model reads the obstacle count from the tensor shape.
    """
    counts = [int(s["mask_obstacles"].sum()) for s in samples]
    n = max(1, max(counts))

    batch = {}
    for key in _OBSTACLE_KEYS:
        stacked = np.stack([s[key][:n] for s in samples], axis=0)
        batch[key] = torch.from_numpy(stacked)
    for key in _FIXED_KEYS:
        stacked = np.stack([s[key] for s in samples], axis=0)
        batch[key] = torch.from_numpy(stacked)
    return batch


def to_device(batch: dict, device: str) -> dict:
    """Move every tensor in the batch to the target device."""
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def split_episodes(ds: Dataset, val_episodes: int) -> tuple[Subset, Subset, list[str]]:
    """Hold out the last val_episodes episodes as validation.

    Returns (train_set, val_set, val_episode_names).
    Splitting by episode keeps consecutive, correlated frames from leaking
    across the train/val boundary.
    """
    by_episode = ds.indices_by_episode()
    episodes = sorted(by_episode)
    assert len(episodes) > val_episodes, (
        f"need more than val_episodes={val_episodes} episodes, got {len(episodes)}"
    )
    train_eps = episodes[:-val_episodes]
    val_eps = episodes[-val_episodes:]

    train_idx = [i for ep in train_eps for i in by_episode[ep]]
    val_idx = [i for ep in val_eps for i in by_episode[ep]]
    return Subset(ds, train_idx), Subset(ds, val_idx), val_eps
