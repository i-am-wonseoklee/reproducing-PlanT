"""Closed-loop driving agent: ego driven by a trained PlanT model.

Mirrors the CARLA lifecycle of plant/carla/collector.py (synchronous world, NPC
traffic on autopilot, careful teardown), but the ego is steered by the model
instead of the Traffic Manager. Each run_step() reads the scene, tokenises it the
same way the training Dataset does (plant/data/features.py), runs the model, turns
the predicted waypoints into a control with plant/carla/controller.py, applies it,
and ticks the world.
"""

import logging
import math
from dataclasses import dataclass

import carla
import numpy as np
import torch

from plant.carla.controller import ControllerConfig, WaypointController
from plant.data.features import build_input_features
from plant.model.plant import PlanT

logger = logging.getLogger(__name__)

_TL_STATE_STR = {
    carla.TrafficLightState.Red: "Red",
    carla.TrafficLightState.Green: "Green",
    carla.TrafficLightState.Yellow: "Yellow",
}


@dataclass
class AgentConfig:
    host: str = "localhost"
    port: int = 2000
    carla_timeout: float = 60.0
    world_warmup_ticks: int = 60
    fps: int = 20
    town: str = "Town01"
    n_npcs: int = 50
    n_waypoints: int = 100
    vehicle_filter_radius: float = 30.0
    min_npc_spawn_distance: float = 10.0


class PlanTAgent:
    """One trained model driving one ego vehicle in a synchronous CARLA world."""

    def __init__(
        self,
        config: AgentConfig,
        checkpoint_path: str,
        device: str,
        controller_config: ControllerConfig | None = None,
    ):
        self.config = config
        self.device = device

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.model_cfg = ckpt["model_config"]
        self.model = PlanT(self.model_cfg).to(device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            "Loaded model: %.1fM params, epoch %d",
            n_params / 1e6,
            ckpt.get("epoch", -1),
        )

        self.controller = WaypointController(controller_config or ControllerConfig())

        self.client = carla.Client(config.host, config.port)
        self.client.set_timeout(config.carla_timeout)

        self.world: carla.World | None = None
        self.ego: carla.Actor | None = None
        self.npcs: list[carla.Actor] = []
        self.collision_sensor: carla.Actor | None = None
        self.collision = False

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self):
        town = self.config.town

        # Unregister any TM-controlled actors left over from a previous crashed run.
        try:
            prev_world = self.client.get_world()
            for actor in prev_world.get_actors().filter("vehicle.*"):
                actor.set_autopilot(False)
        except Exception:
            pass

        logger.info("Loading world (%s) ...", town)
        self.world = self.client.load_world(town)

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / self.config.fps
        self.world.apply_settings(settings)

        tm = self.client.get_trafficmanager()
        tm.set_synchronous_mode(True)

        self._spawn_ego()
        self._spawn_collision_sensor()
        self._spawn_npcs()

        logger.info("Warming up (%d ticks) ...", self.config.world_warmup_ticks)
        for _ in range(self.config.world_warmup_ticks):
            self.world.tick()

    def _vehicle_blueprints(self) -> list:
        return list(self.world.get_blueprint_library().filter("vehicle.*.*"))

    def _spawn_ego(self):
        bp = np.random.choice(self._vehicle_blueprints())
        start_sp = np.random.choice(self.world.get_map().get_spawn_points())
        self.ego = self.world.spawn_actor(bp, start_sp)
        # No autopilot: the model drives the ego.

    def _spawn_collision_sensor(self):
        bp = self.world.get_blueprint_library().find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(
            bp, carla.Transform(), attach_to=self.ego
        )
        self.collision_sensor.listen(self._on_collision)

    def _on_collision(self, event):
        self.collision = True
        logger.warning("Collision with %s", event.other_actor.type_id)

    def _spawn_npcs(self):
        ego_loc = self.ego.get_transform().location
        all_sps = [
            sp
            for sp in self.world.get_map().get_spawn_points()
            if ego_loc.distance(sp.location) > self.config.min_npc_spawn_distance
        ]
        blueprints = self._vehicle_blueprints()
        tm = self.client.get_trafficmanager()
        n = min(self.config.n_npcs, len(all_sps))
        for sp in np.random.choice(all_sps, size=n, replace=False):
            npc = self.world.try_spawn_actor(np.random.choice(blueprints), sp)
            if npc is not None:
                npc.set_autopilot(True, tm.get_port())
                self.npcs.append(npc)
        self.world.tick()

    # ------------------------------------------------------------------
    # Scene reading (collector frame format, world frame)
    # ------------------------------------------------------------------

    def _read_ego(self) -> dict:
        transform = self.ego.get_transform()
        velocity = self.ego.get_velocity()
        loc = transform.location
        rot = transform.rotation
        return {
            "x": loc.x,
            "y": loc.y,
            "z": loc.z,
            "roll": np.radians(rot.roll),
            "pitch": np.radians(rot.pitch),
            "yaw": np.radians(rot.yaw),
            "vx": velocity.x,
            "vy": velocity.y,
        }

    def _read_obstacles(self) -> list:
        ego_loc = self.ego.get_transform().location
        result = []
        for actor in self.world.get_actors().filter("vehicle.*"):
            if actor.id == self.ego.id:
                continue
            try:
                t = actor.get_transform()
                dist = ego_loc.distance(t.location)
                if dist > self.config.vehicle_filter_radius:
                    continue
                loc = t.location
                rot = t.rotation
                vel = actor.get_velocity()
                extent = actor.bounding_box.extent
            except Exception:
                continue
            result.append(
                {
                    "actor_id": actor.id,
                    "x": loc.x,
                    "y": loc.y,
                    "yaw": np.radians(rot.yaw),
                    "speed": np.sqrt(vel.x**2 + vel.y**2),
                    "w": 2.0 * extent.y,
                    "h": 2.0 * extent.x,
                    "type": actor.type_id,
                }
            )
        result.sort(key=lambda o: (o["x"] - ego_loc.x) ** 2 + (o["y"] - ego_loc.y) ** 2)
        return result

    def _read_route(self) -> list:
        ego_transform = self.ego.get_transform()
        wmap = self.world.get_map()
        current = wmap.get_waypoint(
            ego_transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if current is None:
            return []
        ego_yaw = math.radians(ego_transform.rotation.yaw)
        wp_yaw = math.radians(current.transform.rotation.yaw)
        diff = abs(math.atan2(math.sin(wp_yaw - ego_yaw), math.cos(wp_yaw - ego_yaw)))
        if diff > math.pi / 2:
            return []
        out = []
        for _ in range(self.config.n_waypoints):
            next_wps = current.next(1.0)
            if not next_wps:
                break
            current = next_wps[0]
            out.append(
                {
                    "x": current.transform.location.x,
                    "y": current.transform.location.y,
                    "yaw": np.radians(current.transform.rotation.yaw),
                    "road_width": current.lane_width,
                }
            )
        return out

    def _read_traffic_lights(self) -> list:
        if not self.ego.is_at_traffic_light():
            return []
        tl = self.ego.get_traffic_light()
        tl_loc = tl.get_transform().location
        ego_loc = self.ego.get_transform().location
        return [
            {
                "state": _TL_STATE_STR.get(
                    self.ego.get_traffic_light_state(), "Unknown"
                ),
                "distance": ego_loc.distance(tl_loc),
                "x": tl_loc.x,
                "y": tl_loc.y,
            }
        ]

    # ------------------------------------------------------------------
    # Inference + control
    # ------------------------------------------------------------------

    def _to_batch(self, sample: dict) -> dict:
        """Build a 1-element batch of the five forward() inputs.

        Trims obstacle tensors to the valid count, matching collate_dynamic.
        """
        n = max(1, int(sample["mask_obstacles"].sum()))
        batch = {
            "feature_obstacles": sample["feature_obstacles"][:n],
            "mask_obstacles": sample["mask_obstacles"][:n],
            "feature_route_segments": sample["feature_route_segments"],
            "feature_traffic_light": sample["feature_traffic_light"],
            "feature_target_point": sample["feature_target_point"],
        }
        return {
            k: torch.from_numpy(np.asarray(v)).unsqueeze(0).to(self.device)
            for k, v in batch.items()
        }

    @torch.no_grad()
    def run_step(self) -> tuple[dict, np.ndarray, carla.VehicleControl]:
        """One closed-loop step: observe, infer, control, tick.

        Returns (sample, pred_wp, control) for rendering. sample carries an empty
        feature_waypoints so the BEV renderer draws no GT markers.
        """
        ego = self._read_ego()
        obstacles = self._read_obstacles()
        route = self._read_route()
        traffic_lights = self._read_traffic_lights()

        sample = build_input_features(
            ego,
            obstacles,
            traffic_lights,
            route,
            self.model_cfg["n_obstacles"],
            self.model_cfg["n_route_segments"],
        )
        sample["feature_waypoints"] = np.zeros((0, 2), dtype=np.float32)

        pred = self.model(self._to_batch(sample))
        pred_wp = pred["waypoints"][0].cpu().numpy()  # (P, 2)

        speed = math.hypot(ego["vx"], ego["vy"])
        control = self.controller.run_step(pred_wp, speed)
        self.ego.apply_control(control)

        self.world.tick()
        return sample, pred_wp, control

    # ------------------------------------------------------------------
    # Cleanup (order mirrors Collector._cleanup to avoid CARLA C++ aborts)
    # ------------------------------------------------------------------

    def cleanup(self):
        if self.world is None:
            return

        if self.collision_sensor is not None:
            try:
                self.collision_sensor.stop()
                self.collision_sensor.destroy()
            except Exception:
                pass
            self.collision_sensor = None

        for actor in self.npcs:
            try:
                actor.set_autopilot(False)
            except Exception:
                pass

        try:
            self.world.tick()
        except Exception:
            pass

        for actor in [*self.npcs, *([self.ego] if self.ego else [])]:
            try:
                actor.destroy()
            except Exception:
                pass
        self.npcs.clear()
        self.ego = None

        try:
            tm = self.client.get_trafficmanager()
            tm.set_synchronous_mode(False)
        except Exception:
            pass

        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)
