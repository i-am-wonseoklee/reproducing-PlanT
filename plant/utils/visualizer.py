"""Visualization utilities."""

import io

import matplotlib

matplotlib.use("Agg")
# isort: split
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from plant.utils.geometry import rotated_corners

_TL_COLOR = {"Red": "red", "Green": "limegreen", "Yellow": "yellow"}


def _finish(fig, as_array: bool):
    """Rasterize the figure to either PNG bytes or an (H, W, 3) uint8 array.

    PNG bytes (default) are what the DB stores; the array path feeds video
    writers without a PNG round-trip.
    """
    if as_array:
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        out = rgba[..., :3].copy()
        plt.close(fig)
        return out

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def render_bev(
    ego: dict,
    obstacles: list,
    route_waypoints: list,
    traffic_lights: list,
    radius: float = 35.0,
) -> bytes:
    """Render a bird's-eye-view frame for a collected dataset sample.

    Draws the ego vehicle (red semi-transparent OBB), nearby obstacle OBBs
    (blue semi-transparent), route waypoints (green line), and traffic light
    states (colored circles), all on a black background.
    Returns a PNG-encoded bytes object suitable for storing in the database.

    Args:
        ego: Ego state dict with keys x, y, yaw.
        obstacles: List of obstacle dicts with keys x, y, w, h, yaw.
        route_waypoints: List of waypoint dicts with keys x, y.
        traffic_lights: List of traffic light dicts with keys x, y, state.
        radius: Half-width of the rendered square viewport in metres.
    """
    ex, ey, eyaw = ego["x"], ego["y"], ego["yaw"]

    fig, ax = plt.subplots(figsize=(5, 5), facecolor="black")
    ax.set_facecolor("black")
    ax.set_xlim(ex - radius, ex + radius)
    ax.set_ylim(ey - radius, ey + radius)
    ax.set_aspect("equal")
    ax.set_axis_off()

    if route_waypoints:
        xs = [wp["x"] for wp in route_waypoints]
        ys = [wp["y"] for wp in route_waypoints]
        ax.plot(xs, ys, color="#00cc44", linewidth=1.5, marker=".", markersize=4)

    for obs in obstacles:
        corners = rotated_corners(obs["x"], obs["y"], obs["w"], obs["h"], obs["yaw"])
        ax.add_patch(
            mpatches.Polygon(
                corners,
                closed=True,
                facecolor=(0.2, 0.5, 1.0, 0.5),
                edgecolor=(0.4, 0.7, 1.0, 0.9),
                linewidth=1.0,
            )
        )

    for tl in traffic_lights:
        color = _TL_COLOR.get(tl["state"], "white")
        ax.plot(
            tl["x"],
            tl["y"],
            marker="o",
            markersize=10,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.5,
        )

    ego_corners = rotated_corners(ex, ey, w=2.0, h=4.5, yaw=eyaw)
    ax.add_patch(
        mpatches.Polygon(
            ego_corners,
            closed=True,
            facecolor=(1.0, 0.2, 0.2, 0.6),
            edgecolor=(1.0, 0.5, 0.5, 1.0),
            linewidth=1.0,
        )
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def render_bev_features(
    sample: dict,
    episode: str,
    tick: int,
    pred_wp: "np.ndarray | None" = None,
    bev_radius: float = 60.0,
    as_array: bool = False,
) -> "bytes | np.ndarray":
    """Render ego-frame BEV from Dataset output sample.

    Args:
        sample: dict returned by Dataset.__getitem__. An empty feature_waypoints
                (shape (0, 2)) draws no GT markers, which is what the closed-loop
                agent passes since there is no ground truth.
        episode: episode identifier string, shown as label.
        tick: frame tick, shown as label.
        pred_wp: (P, 2) numpy array of predicted waypoints, or None.
                 GT waypoints are drawn in white; predicted in orange.
        bev_radius: half-width of the rendered square viewport in metres.
        as_array: if True, return an (H, W, 3) uint8 RGB array (for video);
                  otherwise return PNG bytes (for DB storage).
    """
    fig, ax = plt.subplots(figsize=(5, 5), facecolor="black")
    ax.set_facecolor("black")
    ax.set_xlim(-bev_radius, bev_radius)
    ax.set_ylim(-bev_radius, bev_radius)
    ax.set_aspect("equal")
    ax.set_axis_off()

    tl_val = float(sample["feature_traffic_light"][0])
    tl_text = "Red" if tl_val > 0.5 else "Green"
    tl_color = "red" if tl_val > 0.5 else "limegreen"
    ax.text(
        0.02,
        0.98,
        f"TL: {tl_text}",
        transform=ax.transAxes,
        color=tl_color,
        fontsize=9,
        va="top",
        ha="left",
    )
    ax.text(
        0.02,
        0.92,
        f"{episode}  tick={tick}",
        transform=ax.transAxes,
        color="white",
        fontsize=7,
        va="top",
        ha="left",
    )
    has_gt = len(sample["feature_waypoints"]) > 0
    if pred_wp is not None:
        ax.text(
            0.98,
            0.98,
            "● GT  ● Pred" if has_gt else "● Pred",
            transform=ax.transAxes,
            color="white",
            fontsize=7,
            va="top",
            ha="right",
        )

    for seg in sample["feature_route_segments"]:
        _, cx, cy, yaw, lane_w, seg_len = seg
        if seg_len == 0:
            continue
        corners = rotated_corners(cx, cy, lane_w, seg_len, yaw)
        ax.add_patch(
            mpatches.Polygon(
                corners,
                closed=True,
                facecolor=(0.1, 0.7, 0.2, 0.35),
                edgecolor=(0.2, 1.0, 0.4, 0.9),
                linewidth=1.0,
            )
        )

    feat_obs = sample["feature_obstacles"]
    mask_obs = sample["mask_obstacles"]
    for i, valid in enumerate(mask_obs):
        if not valid:
            continue
        _, x, y, yaw, w, h = feat_obs[i]
        corners = rotated_corners(x, y, w, h, yaw)
        ax.add_patch(
            mpatches.Polygon(
                corners,
                closed=True,
                facecolor=(0.2, 0.5, 1.0, 0.5),
                edgecolor=(0.4, 0.7, 1.0, 0.9),
                linewidth=1.0,
            )
        )

    tp = sample["feature_target_point"]
    ax.plot(tp[0], tp[1], marker="*", markersize=12, color="yellow", zorder=5)

    gt_wp = sample["feature_waypoints"]
    sizes = np.linspace(10, 4, len(gt_wp))
    for i, (wx, wy) in enumerate(gt_wp):
        ax.plot(wx, wy, "o", color="white", markersize=sizes[i], zorder=6)

    if pred_wp is not None:
        sizes = np.linspace(10, 4, len(pred_wp))
        for i, (wx, wy) in enumerate(pred_wp):
            ax.plot(wx, wy, "o", color="orange", markersize=sizes[i], zorder=7)

    ego_corners = rotated_corners(0.0, 0.0, w=2.0, h=4.5, yaw=0.0)
    ax.add_patch(
        mpatches.Polygon(
            ego_corners,
            closed=True,
            facecolor=(1.0, 0.2, 0.2, 0.6),
            edgecolor=(1.0, 0.5, 0.5, 1.0),
            linewidth=1.5,
        )
    )

    return _finish(fig, as_array)
