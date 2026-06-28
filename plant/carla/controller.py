"""Waypoint to vehicle control: Transfuser-style PID + pure-pursuit.

The model predicts ego-frame waypoints; CARLA needs a throttle/steer/brake
command. This module turns the predicted trajectory into a carla.VehicleControl
the way the original PlanT/Transfuser closed-loop agent does: a longitudinal PID
on speed and a lateral PID steering toward an aim point.
"""

import math
from collections import deque
from dataclasses import dataclass, field

import carla
import numpy as np

# Waypoints are spaced 0.5 s apart (collector saves every 10th tick at 20 FPS),
# so the distance between consecutive waypoints over 0.5 s gives speed in m/s.
_WAYPOINT_DT = 0.5


class PIDController:
    """Discrete PID with a fixed-size error window for the I and D terms."""

    def __init__(self, k_p: float, k_i: float, k_d: float, n: int):
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self._window: deque[float] = deque(maxlen=max(1, n))

    def step(self, error: float) -> float:
        self._window.append(float(error))
        integral = float(np.mean(self._window))
        derivative = (
            self._window[-1] - self._window[-2] if len(self._window) > 1 else 0.0
        )
        return self.k_p * error + self.k_i * integral + self.k_d * derivative


@dataclass
class ControllerConfig:
    turn_k_p: float = 1.25
    turn_k_i: float = 0.75
    turn_k_d: float = 0.3
    turn_n: int = 20
    speed_k_p: float = 5.0
    speed_k_i: float = 0.5
    speed_k_d: float = 1.0
    speed_n: int = 20
    max_throttle: float = 0.75
    brake_speed: float = 0.4  # below this desired speed, brake [m/s]
    brake_ratio: float = 1.1  # brake if current/desired speed exceeds this
    clip_delta: float = 0.25  # cap on the speed error fed to the throttle PID


@dataclass
class WaypointController:
    """Convert predicted ego-frame waypoints to a carla.VehicleControl."""

    config: ControllerConfig = field(default_factory=ControllerConfig)

    def __post_init__(self):
        c = self.config
        self.turn_controller = PIDController(
            c.turn_k_p, c.turn_k_i, c.turn_k_d, c.turn_n
        )
        self.speed_controller = PIDController(
            c.speed_k_p, c.speed_k_i, c.speed_k_d, c.speed_n
        )

    def run_step(self, waypoints: np.ndarray, speed: float) -> carla.VehicleControl:
        """Compute control from predicted waypoints and current speed.

        Args:
            waypoints: (P, 2) ego-frame predicted positions, +x forward, +y right
                (CARLA is left-handed: x forward, y right).
            speed: current ego speed [m/s].
        """
        c = self.config
        wp = np.asarray(waypoints, dtype=float)

        # Longitudinal: desired speed from the first 0.5 s waypoint spacing.
        desired_speed = float(np.linalg.norm(wp[1] - wp[0])) / _WAYPOINT_DT
        brake = desired_speed < c.brake_speed or (
            desired_speed > 0 and speed / desired_speed > c.brake_ratio
        )
        delta = np.clip(desired_speed - speed, 0.0, c.clip_delta)
        throttle = float(
            np.clip(self.speed_controller.step(delta), 0.0, c.max_throttle)
        )
        if brake:
            throttle = 0.0

        # Lateral: pure-pursuit toward the 1.0 s aim point. In the ego frame +y is
        # to the right (CARLA is left-handed) and CARLA steer > 0 also turns right,
        # so a right target (positive angle) maps directly to positive steer.
        aim = wp[1]
        angle = math.atan2(aim[1], aim[0]) / (math.pi / 2)  # normalize to ~[-1, 1]
        steer = float(np.clip(self.turn_controller.step(angle), -1.0, 1.0))

        return carla.VehicleControl(throttle=throttle, steer=steer, brake=float(brake))
