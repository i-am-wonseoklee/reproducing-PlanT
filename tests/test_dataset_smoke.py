"""Smoke test: extract features from one episode and render BEV images.

Reads frames.db, converts every frame in the chosen episode to PlanT feature
vectors via Dataset, renders each as an ego-frame BEV image, and writes
all images to data/test_dataset_smoke.db (previews table).

Run as a test (uses the first episode):
    pytest tests/test_dataset_smoke.py -s
Or as a script (optionally pass an episode id):
    python3 -m tests.test_dataset_smoke
    python3 -m tests.test_dataset_smoke episode_0003
"""

import argparse
import io
import sqlite3
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from plant.data.dataset import Dataset
from plant.data.storage import Storage
from plant.utils.geometry import rotated_corners

plt.switch_backend("Agg")

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "data" / "frames.db"
OUT_PATH = ROOT / "data" / "test_dataset_smoke.db"
BEV_RADIUS = 40.0


def render_bev_from_features(sample: dict, episode: str, tick: int) -> bytes:
    """Render an ego-frame BEV from Dataset output."""
    fig, ax = plt.subplots(figsize=(5, 5), facecolor="black")
    ax.set_facecolor("black")
    ax.set_xlim(-BEV_RADIUS, BEV_RADIUS)
    ax.set_ylim(-BEV_RADIUS, BEV_RADIUS)
    ax.set_aspect("equal")
    ax.set_axis_off()

    # Traffic light state label.
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

    # Route segments (green OBB).
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

    # Obstacles (blue OBB, valid slots only).
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

    # Target point (yellow star).
    tp = sample["feature_target_point"]
    ax.plot(tp[0], tp[1], marker="*", markersize=12, color="yellow", zorder=5)

    # Future waypoints (white dots, shrinking size).
    fw = sample["feature_waypoints"]
    sizes = np.linspace(10, 4, len(fw))
    for i, (wx, wy) in enumerate(fw):
        ax.plot(wx, wy, "o", color="white", markersize=sizes[i], zorder=5)

    # Ego vehicle (red OBB, always at origin heading +x).
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

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def smoke_episode(episode: str | None = None):
    """Render one episode's features to OUT_PATH. None selects the first episode."""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DB not found: {DB_PATH}. Run data collection first.")

    storage = Storage(DB_PATH)
    episodes = storage.episodes()
    storage.close()
    assert episodes, "no episodes in DB"

    if episode is None:
        episode = episodes[0]
    elif episode not in episodes:
        raise ValueError(f"episode {episode!r} not in DB; available: {episodes}")
    print(f"Smoke-testing episode: {episode}")

    ds = Dataset(DB_PATH)

    # Collect indices for the chosen episode.
    ep_indices = [i for i, (ep, _) in enumerate(ds._index) if ep == episode]
    print(f"  {len(ep_indices)} usable frames (episode length - n_predictions)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(OUT_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS previews (
            episode TEXT    NOT NULL,
            tick    INTEGER NOT NULL,
            preview BLOB    NOT NULL
        )
    """)
    conn.execute("DELETE FROM previews WHERE episode = ?", (episode,))
    conn.commit()

    rows = []
    for seq, flat_idx in enumerate(ep_indices):
        ep, pos = ds._index[flat_idx]
        sample = ds[flat_idx]
        frame = ds._episode_frames[ep][pos]
        png = render_bev_from_features(sample, ep, frame.tick)
        rows.append((ep, frame.tick, png))
        if (seq + 1) % 50 == 0:
            print(f"  rendered {seq + 1}/{len(ep_indices)}")

    conn.executemany("INSERT INTO previews VALUES (?, ?, ?)", rows)
    conn.commit()
    conn.close()

    print(f"Done. {len(rows)} frames written to {OUT_PATH}")

    # Quick shape assertions.
    sample = ds[ep_indices[0]]
    n_obs = ds.n_obstacles
    expected = {
        "feature_obstacles": (n_obs, 6),
        "mask_obstacles": (n_obs,),
        "feature_route_segments": (ds.n_route_segments, 6),
        "feature_traffic_light": (1,),
        "feature_target_point": (2,),
        "feature_waypoints": (ds.n_predictions, 2),
        "feature_future_obstacles": (n_obs, 6),
        "mask_future_obstacles": (n_obs,),
    }
    for key, shape in expected.items():
        assert sample[key].shape == shape, (key, sample[key].shape)
    print("All shape assertions passed.")


def test_dataset_smoke():
    """Pytest entry point: smoke-test the first episode."""
    smoke_episode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "episode",
        nargs="?",
        default=None,
        help="episode id to render (default: first episode in the DB)",
    )
    smoke_episode(parser.parse_args().episode)
