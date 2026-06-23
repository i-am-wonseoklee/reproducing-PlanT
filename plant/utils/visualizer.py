"""Visualization utilities."""

import io

import matplotlib

matplotlib.use("Agg")
# isort: split
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from plant.utils.geometry import rotated_corners

_TL_COLOR = {"Red": "red", "Green": "limegreen", "Yellow": "yellow"}


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
