"""Route segments: dense waypoints to oriented bounding-box features."""

import math
from itertools import islice, pairwise

import numpy as np
from shapely.geometry import LineString


def _densify(points: np.ndarray, max_len: float) -> np.ndarray:
    """Insert points so no consecutive pair is farther apart than max_len.

    Each edge longer than max_len is split into the fewest equal sub-edges
    that all stay within max_len. This keeps the segments contiguous (no
    gaps) instead of clipping a long edge to a shorter floating box.
    """
    out = [points[0]]
    for p0, p1 in pairwise(points):
        n_sub = max(1, math.ceil(math.dist(p0, p1) / max_len))
        for k in range(1, n_sub + 1):
            out.append(p0 + (p1 - p0) * (k / n_sub))
    return np.array(out)


def _segment_obb(
    seg_idx: int,
    u0: np.ndarray,
    u1: np.ndarray,
    xy_ego: np.ndarray,
    road_widths: np.ndarray,
) -> list:
    """Build one OBB feature row (seg_idx, cx, cy, yaw, lane_w, seg_len).

    The caller guarantees ||u0 - u1|| <= max_seg_len (via densify), so seg_len
    is the true distance. Lane width is near-constant along a lane, so it is
    taken from the original route point nearest the segment center.
    """
    cx, cy = (u0 + u1) * 0.5
    dx, dy = u1 - u0
    yaw = math.atan2(dy, dx)  # already in [-pi, pi]
    seg_len = math.hypot(dx, dy)

    nearest = int(np.argmin(np.sum((xy_ego - [cx, cy]) ** 2, axis=1)))
    lane_w = float(road_widths[nearest])

    return [float(seg_idx), cx, cy, yaw, lane_w, seg_len]


def waypoints_to_route_segments(
    xy_ego: np.ndarray,
    road_widths: np.ndarray,
    n_segments: int = 2,
    rdp_epsilon: float = 0.5,
    max_seg_len: float = 10.0,
) -> np.ndarray:
    """Compress a dense ego-frame route into n_segments OBB feature vectors.

    Args:
        xy_ego: (N, 2) dense route points in ego frame.
        road_widths: (N,) lane width at each point.
        n_segments: number of output segments (paper default N_s=2).
        rdp_epsilon: RDP tolerance in metres (paper default 0.5).
        max_seg_len: maximum segment length in metres (paper default L_max=10).

    Returns:
        (n_segments, 6) array; each row is (seg_idx, cx, cy, yaw, lane_w, seg_len).
        Rows beyond the available geometry are zero-padded.
    """
    xy_ego = np.asarray(xy_ego, dtype=float)
    road_widths = np.asarray(road_widths, dtype=float)

    out = np.zeros((n_segments, 6), dtype=np.float32)

    if len(xy_ego) < 2:
        return out

    # RDP simplification keeps the route's turns with few points; densify then
    # splits any over-long edge so every segment stays within max_seg_len.
    line = LineString(xy_ego)
    simplified = np.array(line.simplify(rdp_epsilon).coords)
    key_points = _densify(simplified, max_seg_len)

    pairs = islice(enumerate(pairwise(key_points)), n_segments)
    for i, (u0, u1) in pairs:
        out[i] = _segment_obb(i, u0, u1, xy_ego, road_widths)

    return out
