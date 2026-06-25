"""Geometry utilities."""

import numpy as np
from scipy.spatial.transform import Rotation


class Pose:
    """Rigid-body pose represented as a 4x4 homogeneous transformation matrix."""

    def __init__(self, mat: np.ndarray):
        assert mat.shape == (4, 4)
        self.mat = mat

    @classmethod
    def from_rot_trans(cls, rot: np.ndarray, trans: np.ndarray) -> "Pose":
        """Construct from a 3x3 rotation matrix and a 3D translation vector."""
        mat = np.eye(4)
        mat[:3, :3] = rot
        mat[:3, 3] = trans
        return cls(mat)

    @classmethod
    def from_xyz_rpy(cls, xyz: np.ndarray, rpy: np.ndarray) -> "Pose":
        """Construct from translation xyz and roll-pitch-yaw angles (radians)."""
        roll, pitch, yaw = rpy
        rot = Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_matrix()
        return cls.from_rot_trans(rot, xyz)

    @property
    def rot(self) -> np.ndarray:
        """3x3 rotation matrix."""
        return self.mat[:3, :3]

    @property
    def trans(self) -> np.ndarray:
        """3D translation vector."""
        return self.mat[:3, 3]

    def inv(self) -> "Pose":
        """Return the inverse pose."""
        return Pose.from_rot_trans(self.rot.T, -(self.rot.T @ self.trans))

    def to_xyz_rpy(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (xyz, rpy) where rpy is [roll, pitch, yaw] in radians."""
        yaw, pitch, roll = Rotation.from_matrix(self.rot).as_euler("ZYX")
        return self.trans, np.array([roll, pitch, yaw])

    def __matmul__(self, other: "Pose | np.ndarray") -> "Pose | np.ndarray":
        """Compose two poses, or transform a point array of shape (..., 3)."""
        if isinstance(other, Pose):
            return Pose(self.mat @ other.mat)
        return other @ self.rot.T + self.trans


def wrap_to_pi(angle: float | np.ndarray) -> float | np.ndarray:
    """Normalize angle(s) to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def rotated_corners(
    x: float, y: float, w: float, h: float, yaw: float
) -> list[tuple[float, float]]:
    """Return the 4 corners of an axis-aligned rectangle rotated by yaw.

    w is the width (lateral extent) and h is the length (longitudinal extent).
    yaw is in radians, measured counter-clockwise from the x-axis.
    """
    c, s = np.cos(yaw), np.sin(yaw)
    corners = []
    for dx, dy in [(-h / 2, -w / 2), (h / 2, -w / 2), (h / 2, w / 2), (-h / 2, w / 2)]:
        corners.append((x + dx * c - dy * s, y + dx * s + dy * c))
    return corners
