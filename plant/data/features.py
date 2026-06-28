"""Scene to PlanT input features.

The four builders here turn a single frame's raw ego-frame scene (the dicts that
the collector writes to the DB) into the model's inference inputs. They are shared
between the training Dataset (plant/data/dataset.py) and the closed-loop agent
(plant/carla/agent.py) so that live inputs are tokenised exactly like training
inputs. The training-only targets (future waypoints, future obstacles) live in
Dataset, not here.
"""

import numpy as np

from plant.utils.geometry import Pose, wrap_to_pi
from plant.utils.route import waypoints_to_route_segments


def build_obstacles(
    npcs: list, inv_ego: Pose, ego_yaw: float, n_obstacles: int
) -> dict:
    """Obstacle tokens (speed, x, y, yaw, w, h) in ego frame, zero-padded.

    Returns {"feature_obstacles": (n_obstacles, 6) f32,
             "mask_obstacles": (n_obstacles,) bool}.
    """
    feat = np.zeros((n_obstacles, 6), dtype=np.float32)
    mask = np.zeros(n_obstacles, dtype=bool)

    for i, npc in enumerate(npcs[:n_obstacles]):
        world_xy = np.array([npc["x"], npc["y"], 0.0])
        ego_xy = inv_ego @ world_xy
        yaw = wrap_to_pi(npc["yaw"] - ego_yaw)
        feat[i] = [npc["speed"], ego_xy[0], ego_xy[1], yaw, npc["w"], npc["h"]]
        mask[i] = True

    return {"feature_obstacles": feat, "mask_obstacles": mask}


def build_route_segments(
    waypoints: list, inv_ego: Pose, n_route_segments: int
) -> np.ndarray:
    """Route OBB tokens (seg_idx, cx, cy, yaw, lane_w, seg_len) in ego frame."""
    if not waypoints:
        return np.zeros((n_route_segments, 6), dtype=np.float32)

    world_xy = np.array([[wp["x"], wp["y"], 0.0] for wp in waypoints])
    ego_xy = np.stack([inv_ego @ p for p in world_xy])[:, :2]
    road_widths = np.array([wp["road_width"] for wp in waypoints], dtype=float)

    return waypoints_to_route_segments(ego_xy, road_widths, n_route_segments)


def build_traffic_light(traffic_light: list) -> np.ndarray:
    """Red-light flag: (1,) f32, 1.0 if any active light is Red."""
    is_red = any(tl.get("state") == "Red" for tl in traffic_light)
    return np.array([1.0 if is_red else 0.0], dtype=np.float32)


def build_target_point(waypoints: list, inv_ego: Pose) -> np.ndarray:
    """Far point proxy for the sparse GPS goal p_target.

    The route's last waypoint is the farthest goal point we have, so it is
    always beyond the predicted waypoints. Empty route falls back to origin.
    """
    if not waypoints:
        return np.zeros(2, dtype=np.float32)
    wp = waypoints[-1]
    ego_xy = inv_ego @ np.array([wp["x"], wp["y"], 0.0])
    return ego_xy[:2].astype(np.float32)


def build_input_features(
    ego: dict,
    npcs: list,
    traffic_light: list,
    waypoints: list,
    n_obstacles: int,
    n_route_segments: int,
) -> dict:
    """Build all model inference inputs for one frame from raw scene dicts.

    Args:
        ego: ego state dict with x, y, z, roll, pitch, yaw.
        npcs: obstacle dicts (speed, x, y, yaw, w, h) in world frame.
        traffic_light: active traffic light dicts with a "state" key.
        waypoints: dense route waypoint dicts (x, y, road_width) in world frame.
        n_obstacles: obstacle token cap.
        n_route_segments: number of route tokens.

    Returns the five forward() input keys: feature_obstacles, mask_obstacles,
    feature_route_segments, feature_traffic_light, feature_target_point.
    """
    ego_xyz = np.array([ego["x"], ego["y"], ego["z"]])
    ego_rpy = np.array([ego["roll"], ego["pitch"], ego["yaw"]])
    inv_ego = Pose.from_xyz_rpy(ego_xyz, ego_rpy).inv()
    ego_yaw = ego["yaw"]

    return {
        **build_obstacles(npcs, inv_ego, ego_yaw, n_obstacles),
        "feature_route_segments": build_route_segments(
            waypoints, inv_ego, n_route_segments
        ),
        "feature_traffic_light": build_traffic_light(traffic_light),
        "feature_target_point": build_target_point(waypoints, inv_ego),
    }
