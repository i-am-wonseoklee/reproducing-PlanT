"""Data collection agent: ego and NPCs driven by CARLA Traffic Manager autopilot."""

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import carla
import numpy as np

from plant.data.storage import Frame, Storage
from plant.utils.visualizer import render_bev

logger = logging.getLogger(__name__)

_TL_STATE_STR = {
    carla.TrafficLightState.Red: "Red",
    carla.TrafficLightState.Green: "Green",
    carla.TrafficLightState.Yellow: "Yellow",
}


@dataclass
class CollectorConfig:
    host: str = "localhost"
    port: int = 2000
    carla_timeout: float = 60.0
    world_warmup_ticks: int = 60
    fps: int = 20
    collect_tick: int = 10
    towns: list[str] | None = None
    n_npcs: int = 50
    n_waypoints: int = 50
    vehicle_filter_radius: float = 30.0
    min_npc_spawn_distance: float = 10.0
    output_path: str = "data/frames.db"


class Collector:
    """One run() call = one episode. Call repeatedly with incremented episode_id."""

    def __init__(self, config: CollectorConfig = CollectorConfig()):
        self.config = config
        self.client = carla.Client(config.host, config.port)
        self.client.set_timeout(config.carla_timeout)

        self.world: carla.World | None = None
        self.ego: carla.Actor | None = None
        self.npcs: list[carla.Actor] = []

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_world(self, town: str):
        # Unregister any TM-controlled actors left over from a previous crashed run.
        # load_world() destroys them server-side, but the TM C++ thread in our process
        # still holds references and crashes when it tries to control a destroyed actor.
        try:
            prev_world = self.client.get_world()
            for actor in prev_world.get_actors().filter("vehicle.*"):
                actor.set_autopilot(False)
        except Exception:
            pass

        self.world = self.client.load_world(town)

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / self.config.fps
        self.world.apply_settings(settings)

        traffic_manager = self.client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)

    def _vehicle_blueprints(self) -> list:
        return list(self.world.get_blueprint_library().filter("vehicle.*.*"))

    def _spawn_ego(self):
        bp = np.random.choice(self._vehicle_blueprints())
        start_sp = np.random.choice(self.world.get_map().get_spawn_points())
        self.ego = self.world.spawn_actor(bp, start_sp)

        tm = self.client.get_trafficmanager()
        tm.auto_lane_change(self.ego, False)
        tm.random_left_lanechange_percentage(self.ego, 0.0)
        tm.random_right_lanechange_percentage(self.ego, 0.0)
        self.ego.set_autopilot(True, tm.get_port())

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
    # Data collection
    # ------------------------------------------------------------------

    def _waypoints_ahead(self, n: int) -> list | None:
        """Returns None if the road snap is on a crossing road (yaw mismatch > 90°)."""
        ego_transform = self.ego.get_transform()
        wmap = self.world.get_map()
        current = wmap.get_waypoint(
            ego_transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if current is None:
            return None
        ego_yaw = math.radians(ego_transform.rotation.yaw)
        wp_yaw = math.radians(current.transform.rotation.yaw)
        diff = abs(math.atan2(math.sin(wp_yaw - ego_yaw), math.cos(wp_yaw - ego_yaw)))
        if diff > math.pi / 2:
            return None
        result = []
        for _ in range(n):
            next_wps = current.next(1.0)
            if not next_wps:
                break
            current = next_wps[0]
            result.append(current)
        return result

    def _collect_ego(self) -> dict:
        transform = self.ego.get_transform()
        velocity = self.ego.get_velocity()
        control = self.ego.get_control()

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
            "throttle": control.throttle,
            "steer": control.steer,
            "brake": control.brake,
        }

    def _collect_obstacles(self) -> list:
        ego_loc = self.ego.get_transform().location

        result = []
        for actor in self.world.get_actors().filter("vehicle.*"):
            if actor.id == self.ego.id:
                continue
            try:
                actor_transform = actor.get_transform()
                dist = ego_loc.distance(actor_transform.location)
                if dist > self.config.vehicle_filter_radius:
                    continue
                loc = actor_transform.location
                rot = actor_transform.rotation
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

    def _collect_traffic_lights(self) -> list:
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

    def _collect_step(self, episode: str, tick: int) -> Frame | None:
        wps = self._waypoints_ahead(self.config.n_waypoints)
        if wps is None:
            return None
        route_waypoints = [
            {
                "x": wp.transform.location.x,
                "y": wp.transform.location.y,
                "yaw": np.radians(wp.transform.rotation.yaw),
                "road_width": wp.lane_width,
            }
            for wp in wps
        ]
        ego = self._collect_ego()
        obstacles = self._collect_obstacles()
        traffic_lights = self._collect_traffic_lights()
        bev_radius = self.config.vehicle_filter_radius + 10.0
        return Frame(
            episode=episode,
            tick=tick,
            ego=ego,
            npcs=obstacles,
            traffic_light=traffic_lights,
            waypoints=route_waypoints,
            preview=render_bev(
                ego, obstacles, route_waypoints, traffic_lights, radius=bev_radius
            ),
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, episode_id: int = 0, num_ticks: int = 2000):
        episode = f"episode_{episode_id:04d}"
        towns = self.config.towns or ["Town01"]
        town = np.random.choice(towns)

        # Reconnect each episode: the TM keeps a C++ background thread alive for the
        # lifetime of carla.Client. Reusing the same client across episodes accumulates
        # TM thread state and eventually causes an uncatchable TimeoutException abort.
        self.client = carla.Client(self.config.host, self.config.port)
        self.client.set_timeout(self.config.carla_timeout)

        try:
            logger.info("[%s] setting up world (%s) ...", episode, town)
            self._setup_world(town)
            self._spawn_ego()
            self._spawn_npcs()

            with Storage(Path(self.config.output_path)) as storage:
                warmup = self.config.world_warmup_ticks
                logger.info("[%s] warming up (%d ticks) ...", episode, warmup)
                for _ in range(warmup):
                    self.world.tick()

                save_tick = 0
                frames = []
                for tick in range(num_ticks):
                    self.world.tick()
                    if tick % self.config.collect_tick == 0:
                        frame = self._collect_step(episode, save_tick)
                        if frame is not None:
                            frames.append(frame)
                            save_tick += 1
                            logger.debug(
                                "[%s] tick %d/%d  frames %d",
                                episode,
                                tick + 1,
                                num_ticks,
                                save_tick,
                            )
                logger.info("[%s] collected %d frames", episode, save_tick)
                storage.write_batch(frames)

        finally:
            self._cleanup()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup(self):
        if self.world is None:
            return

        # Unregister all actors from TM before destroying. Calling destroy() while the
        # TM C++ thread is mid-cycle on the same actor causes an uncatchable C++ abort.
        all_actors = [*self.npcs, *([self.ego] if self.ego else [])]
        for actor in all_actors:
            try:
                actor.set_autopilot(False)
            except Exception:
                pass

        # One tick so TM processes the unregistrations before we pull the actors out.
        try:
            self.world.tick()
        except Exception:
            pass

        for actor in all_actors:
            try:
                actor.destroy()
            except Exception:
                pass
        self.npcs.clear()
        self.ego = None

        # TM sync mode must be disabled before world sync mode. Leaving TM in sync
        # mode causes a deadlock on the next episode's load_world() tick, which
        # triggers a TimeoutException on the TM's C++ thread — uncatchable from Python.
        try:
            tm = self.client.get_trafficmanager()
            tm.set_synchronous_mode(False)
        except Exception:
            pass

        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)
