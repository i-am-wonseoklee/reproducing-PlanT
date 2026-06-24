"""Data collection agent using CARLA autopilot."""

import logging
from dataclasses import dataclass
from pathlib import Path

import carla
import numpy as np

from plant.carla.global_route_planner import GlobalRoutePlanner
from plant.data.storage import Frame, Storage
from plant.utils.visualizer import render_bev

logger = logging.getLogger(__name__)


@dataclass
class CollectorConfig:
    """Configuration for the CARLA data collector."""

    host: str = "localhost"
    port: int = 2000
    carla_timeout: float = 60.0
    world_warmup_ticks: int = 60
    fps: int = 20
    collect_tick: int = 10
    towns: list[str] = None
    n_npcs: int = 50
    n_waypoints: int = 50
    vehicle_filter_radius: float = 30.0
    min_ego_dest_distance: float = 100.0
    min_npc_spawn_distance: float = 10.0
    output_path: str = "data/frames.db"


class Collector:
    """Drives a CARLA autopilot ego and writes Frame snapshots to a SQLite database.

    Each call to run() covers one episode: world setup, warmup, data collection,
    and cleanup. Multiple episodes can be collected by calling run() repeatedly
    with different episode_id values.
    """

    def __init__(self, config: CollectorConfig = CollectorConfig()):
        self.config = config
        self.client = carla.Client(config.host, config.port)
        self.client.set_timeout(config.carla_timeout)

        self.world: carla.World = None
        self.ego: carla.Actor = None
        self.npcs: list[carla.Actor] = []
        self.route: list = []
        self._route_idx: int = 0

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
        spawn_points = self.world.get_map().get_spawn_points()
        start_sp = np.random.choice(spawn_points)

        min_dest_dist = self.config.min_ego_dest_distance
        dest_candidates = [
            sp
            for sp in spawn_points
            if start_sp.location.distance(sp.location) > min_dest_dist
        ]
        dest_sp = np.random.choice(dest_candidates if dest_candidates else spawn_points)

        self.ego = self.world.spawn_actor(bp, start_sp)
        self.ego.set_autopilot(True)

        grp = GlobalRoutePlanner(self.world.get_map(), sampling_resolution=1.0)
        self.route = grp.trace_route(start_sp.location, dest_sp.location)
        self._route_idx = 0

        if self.route:
            tm = self.client.get_trafficmanager()
            locs = [wp.transform.location for wp, _ in self.route]
            tm.set_path(self.ego, locs)

    def _spawn_npcs(self):
        ego_loc = self.ego.get_transform().location
        all_sps = [
            sp
            for sp in self.world.get_map().get_spawn_points()
            if ego_loc.distance(sp.location) > self.config.min_npc_spawn_distance
        ]
        blueprints = self._vehicle_blueprints()
        n = min(self.config.n_npcs, len(all_sps))
        for sp in np.random.choice(all_sps, size=n, replace=False):
            npc = self.world.try_spawn_actor(np.random.choice(blueprints), sp)
            if npc is not None:
                npc.set_autopilot(True)
                self.npcs.append(npc)
        self.world.tick()

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def _collect_ego(self) -> dict:
        """Return ego vehicle state as a flat dict (world-frame coordinates)."""
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
        """Return NPC vehicles within vehicle_filter_radius, sorted by distance."""
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

    def _select_branch(self, candidates: list) -> "carla.Waypoint":
        """교차로 분기 선택: global route 앞 구간과 가장 가까운 branch를 반환."""
        if not self.route or self._route_idx >= len(self.route):
            return candidates[0]
        look_to = min(len(self.route), self._route_idx + 30)
        route_locs = [
            self.route[i][0].transform.location
            for i in range(self._route_idx + 1, look_to)
        ]
        if not route_locs:
            return candidates[0]

        def min_dist(wp):
            loc = wp.transform.location
            return min(loc.distance(rl) for rl in route_locs)

        return min(candidates, key=min_dist)

    def _collect_route_waypoints(self) -> list:
        """Ego의 현재 도로 위치에서 next()로 waypoints를 동적 계산한다.

        self.route 좌표를 직접 쓰지 않으므로, TM이 GRP와 다른 교차로를
        선택해도 waypoints가 엉뚱한 도로로 점프하지 않는다.
        """
        ego_transform = self.ego.get_transform()
        wmap = self.world.get_map()

        current = wmap.get_waypoint(
            ego_transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if current is None:
            return []

        # Global route 커서를 현재 위치 기준으로 전진 (교차로 분기 힌트용).
        if self.route and self._route_idx < len(self.route):
            ego_loc = ego_transform.location
            n = self.config.n_waypoints
            search_end = min(len(self.route), self._route_idx + n + 20)
            local_closest = min(
                range(self._route_idx, search_end),
                key=lambda i: ego_loc.distance(self.route[i][0].transform.location),
            )
            self._route_idx = local_closest

        result = []
        for _ in range(self.config.n_waypoints):
            next_wps = current.next(1.0)
            if not next_wps:
                break
            current = (
                next_wps[0] if len(next_wps) == 1 else self._select_branch(next_wps)
            )
            loc = current.transform.location
            rot = current.transform.rotation
            result.append(
                {
                    "x": loc.x,
                    "y": loc.y,
                    "yaw": np.radians(rot.yaw),
                    "road_width": current.lane_width,
                }
            )
        return result

    def _collect_traffic_lights(self) -> list:
        """Return active traffic light state; empty list if ego is not at a light."""
        if not self.ego.is_at_traffic_light():
            return []

        tl = self.ego.get_traffic_light()
        state_str = {
            carla.TrafficLightState.Red: "Red",
            carla.TrafficLightState.Green: "Green",
            carla.TrafficLightState.Yellow: "Yellow",
        }.get(tl.get_state(), "Unknown")

        tl_loc = tl.get_transform().location
        distance = self.ego.get_transform().location.distance(tl_loc)
        return [
            {"state": state_str, "distance": distance, "x": tl_loc.x, "y": tl_loc.y}
        ]

    def _collect_step(self, episode: str, tick: int) -> Frame:
        ego = self._collect_ego()
        obstacles = self._collect_obstacles()
        route_waypoints = self._collect_route_waypoints()
        traffic_lights = self._collect_traffic_lights()
        return Frame(
            episode=episode,
            tick=tick,
            ego=ego,
            npcs=obstacles,
            traffic_light=traffic_lights,
            waypoints=route_waypoints,
            preview=render_bev(
                ego,
                obstacles,
                route_waypoints,
                traffic_lights,
                radius=self.config.vehicle_filter_radius + 5.0,
            ),
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, episode_id: int = 0, num_ticks: int = 2000):
        """Run one collection episode and write frames to the configured output_path."""
        episode = f"episode_{episode_id:04d}"
        towns = self.config.towns or ["Town01"]
        town = np.random.choice(towns)
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
                        frames.append(self._collect_step(episode, save_tick))
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
        all_actors = [*self.npcs, *([self.ego] if self.ego is not None else [])]
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

        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)
