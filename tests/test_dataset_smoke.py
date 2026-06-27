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
import logging
import sqlite3
from pathlib import Path

from plant.data.dataset import Dataset
from plant.data.storage import Storage
from plant.utils.visualizer import render_bev_features

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "data" / "frames.db"
OUT_PATH = ROOT / "data" / "test_dataset_smoke.db"


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
    logger.info("Smoke-testing episode: %s", episode)

    ds = Dataset(DB_PATH)

    # Collect indices for the chosen episode.
    ep_indices = [i for i, (ep, _) in enumerate(ds._index) if ep == episode]
    logger.info("  %d usable frames (episode length - n_predictions)", len(ep_indices))

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
        png = render_bev_features(sample, ep, frame.tick)
        rows.append((ep, frame.tick, png))
        if (seq + 1) % 50 == 0:
            logger.info("  rendered %d/%d", seq + 1, len(ep_indices))

    conn.executemany("INSERT INTO previews VALUES (?, ?, ?)", rows)
    conn.commit()
    conn.close()

    logger.info("Done. %d frames written to %s", len(rows), OUT_PATH)

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
    logger.info("All shape assertions passed.")


def test_dataset_smoke():
    """Pytest entry point: smoke-test the first episode."""
    smoke_episode()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "episode",
        nargs="?",
        default=None,
        help="episode id to render (default: first episode in the DB)",
    )
    smoke_episode(parser.parse_args().episode)
