"""Dataset: frames.db to PlanT 6D feature vectors."""

from pathlib import Path

import numpy as np
from torch.utils.data import Dataset as TorchDataset

from plant.data.features import build_input_features
from plant.data.storage import Storage
from plant.utils.geometry import Pose, wrap_to_pi


class Dataset(TorchDataset):
    """Convert collected CARLA frames into PlanT model inputs.

    Each sample is a dict of numpy arrays (not tensors) so that a custom
    collate_fn in train.py can apply dynamic padding across the batch.

    Output keys (all ego frame):
        feature_obstacles        (n_obstacles, 6) f32: (speed, x, y, yaw, w, h)
        mask_obstacles           (n_obstacles,) bool: True = valid slot
        feature_route_segments   (n_route_segments, 6) f32:
                                 (seg_idx, cx, cy, yaw, lane_w, seg_len)
        feature_traffic_light    (1,) f32: 1.0 if Red, else 0.0
        feature_target_point     (2,) f32: (x, y)
        feature_waypoints        (n_predictions, 2) f32: future ego positions
        feature_future_obstacles (n_obstacles, 6) f32: t+1 obstacle attributes
        mask_future_obstacles    (n_obstacles,) bool: True = matched at t and t+1
    """

    def __init__(
        self,
        db_path: str | Path,
        n_predictions: int = 4,
        n_obstacles: int = 100,
        n_route_segments: int = 2,
    ):
        self.n_predictions = n_predictions
        self.n_obstacles = n_obstacles
        self.n_route_segments = n_route_segments

        self._storage = Storage(db_path)

        # Build flat index: (episode, position) pairs, excluding the last
        # n_predictions ticks of each episode (no future frames available).
        self._index: list[tuple[str, int]] = []
        self._episode_frames: dict[str, list] = {}

        for ep in self._storage.episodes():
            frames = self._storage.get_by_episode(ep)
            self._episode_frames[ep] = frames
            usable = len(frames) - n_predictions
            for pos in range(max(0, usable)):
                self._index.append((ep, pos))

    def __len__(self) -> int:
        return len(self._index)

    def indices_by_episode(self) -> dict[str, list[int]]:
        """Map each episode id to the flat sample indices it contributes.

        Lets callers build episode level splits without reaching into internals.
        """
        out: dict[str, list[int]] = {}
        for i, (ep, _) in enumerate(self._index):
            out.setdefault(ep, []).append(i)
        return out

    def __getitem__(self, idx: int) -> dict:
        episode, pos = self._index[idx]
        frames = self._episode_frames[episode]

        f_t = frames[pos]
        future_frames = frames[pos + 1 : pos + 1 + self.n_predictions]

        ego = f_t.ego
        ego_xyz = np.array([ego["x"], ego["y"], ego["z"]])
        ego_rpy = np.array([ego["roll"], ego["pitch"], ego["yaw"]])
        inv_ego = Pose.from_xyz_rpy(ego_xyz, ego_rpy).inv()
        ego_yaw = ego["yaw"]

        return {
            **build_input_features(
                ego,
                f_t.npcs,
                f_t.traffic_light,
                f_t.waypoints,
                self.n_obstacles,
                self.n_route_segments,
            ),
            "feature_waypoints": self._build_waypoints(future_frames, inv_ego),
            **self._build_future_obstacles(f_t.npcs, future_frames, inv_ego, ego_yaw),
        }

    # ------------------------------------------------------------------
    # Per-field builders (training-only targets; inference inputs live in
    # plant/data/features.py and are shared with the closed-loop agent)
    # ------------------------------------------------------------------

    def _build_waypoints(self, future_frames: list, inv_ego: Pose) -> np.ndarray:
        feat = np.zeros((self.n_predictions, 2), dtype=np.float32)
        for i, ff in enumerate(future_frames[: self.n_predictions]):
            world_pos = np.array([ff.ego["x"], ff.ego["y"], 0.0])
            ego_pos = inv_ego @ world_pos
            feat[i] = ego_pos[:2]
        return feat

    def _build_future_obstacles(
        self, npcs: list, future_frames: list, inv_ego: Pose, ego_yaw: float
    ) -> dict:
        feat = np.zeros((self.n_obstacles, 6), dtype=np.float32)
        mask = np.zeros(self.n_obstacles, dtype=bool)

        if not future_frames:
            return {"feature_future_obstacles": feat, "mask_future_obstacles": mask}

        next_npcs = future_frames[0].npcs
        next_by_id = {npc["actor_id"]: npc for npc in next_npcs}

        for i, npc in enumerate(npcs[: self.n_obstacles]):
            npc_next = next_by_id.get(npc["actor_id"])
            if npc_next is None:
                continue
            world_xy = np.array([npc_next["x"], npc_next["y"], 0.0])
            ego_xy = inv_ego @ world_xy
            yaw = wrap_to_pi(npc_next["yaw"] - ego_yaw)
            feat[i] = [
                npc_next["speed"],
                ego_xy[0],
                ego_xy[1],
                yaw,
                npc_next["w"],
                npc_next["h"],
            ]
            mask[i] = True

        return {"feature_future_obstacles": feat, "mask_future_obstacles": mask}
