"""SQLite-backed storage for collected CARLA frames.

Column values example
---------------------
```
episode:             "episode_0001"
tick:                42
ego:                 {"x": 12.3, "y": -4.5, "z": 0.2,
                      "roll": 0.0, "pitch": 0.0, "yaw": 1.57,
                      "vx": 8.3, "vy": 0.1,
                      "throttle": 0.6, "steer": -0.05, "brake": 0.0}
npcs:                [{"actor_id": 42, "x": 18.1, "y": -4.3, "yaw": 1.60, "speed": 7.9,
                       "w": 2.1, "h": 4.8, "type": "vehicle.tesla.model3"},
                      {"actor_id": 87, "x": 4.8, "y": -6.7, "yaw": 1.55, "speed": 0.0,
                       "w": 2.3, "h": 5.2, "type": "vehicle.carlamotors.firetruck"}]
traffic_light:       [{"state": "Red", "distance": 18.4, "x": 112.3, "y": -5.1}]
waypoints:           [{"x": 14.2, "y": -4.4, "yaw": 1.57, "road_width": 3.5}, ...]
preview:             <PNG bytes>
```
"""

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Frame:
    """One timestep of collected driving data.

    episode:       episode identifier string (e.g. "episode_0001")
    tick:          sequential save index within the episode
    ego:           ego vehicle state (position, orientation, velocity, control)
    npcs:          nearby NPC vehicles within vehicle_filter_radius, sorted by distance
    traffic_light: active traffic light state list; empty when ego is not at a light
    waypoints:     route waypoints ahead of ego along the planned path
    preview:       optional bird's-eye-view PNG for debugging
    """

    episode: str
    tick: int
    ego: dict
    npcs: list
    traffic_light: list
    waypoints: list
    preview: bytes | None = None


# order must match _row_to_frame unpacking; BLOB excluded by default
_DATA_COLS = "episode, tick, ego, npcs, traffic_light, waypoints"
_ALL_COLS = _DATA_COLS + ", preview"


class Storage:
    """SQLite-backed append-only store for Frame objects.

    Supports context manager usage::

        with Storage("data/frames.db") as s:
            s.write_batch(frames)
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._init_schema()

    def _init_schema(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS frames (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                episode             TEXT    NOT NULL,
                tick                INTEGER NOT NULL,
                ego                 TEXT    NOT NULL,
                npcs                TEXT    NOT NULL,
                traffic_light       TEXT    NOT NULL,
                waypoints           TEXT    NOT NULL,
                preview             BLOB
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_episode ON frames (episode)")
        self._conn.commit()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self, frame: Frame):
        """Insert a single frame and commit."""
        self._conn.execute(
            f"INSERT INTO frames ({_ALL_COLS}) VALUES (?, ?, ?, ?, ?, ?, ?)",
            self._frame_to_row(frame),
        )
        self._conn.commit()

    def write_batch(self, frames: list[Frame]):
        """Insert multiple frames in a single transaction."""
        self._conn.executemany(
            f"INSERT INTO frames ({_ALL_COLS}) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [self._frame_to_row(f) for f in frames],
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM frames").fetchone()[0]

    def __getitem__(self, idx: int) -> Frame:
        row = self._conn.execute(
            f"SELECT {_DATA_COLS} FROM frames ORDER BY id LIMIT 1 OFFSET ?",
            (idx,),
        ).fetchone()
        if row is None:
            raise IndexError(f"index {idx} out of range (len={len(self)})")
        return self._row_to_frame(row)

    def get_by_episode(self, episode: str, include_image: bool = False) -> list[Frame]:
        """Return all frames for an episode, ordered by tick.

        preview is excluded unless include_image=True.
        """
        cols = _ALL_COLS if include_image else _DATA_COLS
        rows = self._conn.execute(
            f"SELECT {cols} FROM frames WHERE episode = ? ORDER BY tick",
            (episode,),
        ).fetchall()
        return [self._row_to_frame(r, has_image=include_image) for r in rows]

    def episodes(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT episode FROM frames ORDER BY episode"
        ).fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _frame_to_row(f: Frame) -> tuple:
        return (
            f.episode,
            f.tick,
            json.dumps(f.ego),
            json.dumps(f.npcs),
            json.dumps(f.traffic_light),
            json.dumps(f.waypoints),
            f.preview,
        )

    @staticmethod
    def _row_to_frame(row, has_image: bool = False) -> Frame:
        episode, tick, ego, npcs, traffic_light, waypoints = row[:6]
        return Frame(
            episode=episode,
            tick=tick,
            ego=json.loads(ego),
            npcs=json.loads(npcs),
            traffic_light=json.loads(traffic_light),
            waypoints=json.loads(waypoints),
            preview=row[6] if has_image else None,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
