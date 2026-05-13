"""KITTI raw sequence loader for the MPC tracking baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .coordinate_transforms import gps_to_local_xy, unwrap_yaw
from .oxts_parser import load_oxts_packets, packets_to_arrays
from .timestamp_utils import load_timestamps


@dataclass
class KITTITrajectory:
    """Ego-motion trajectory extracted from OXTS and expressed in local meters."""

    time_s: np.ndarray
    px: np.ndarray
    py: np.ndarray
    yaw: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    image_paths: list[Path]
    source: str

    def as_state_matrix(self) -> np.ndarray:
        """Return states ordered as [px, py, v, yaw]."""
        return np.column_stack([self.px, self.py, self.velocity, self.yaw])


class KITTILoader:
    """Load the OXTS stream, timestamps, and optional camera frame paths."""

    def __init__(self, data_root: Path, date: str, drive: str, max_frames: int | None = None) -> None:
        self.data_root = data_root
        self.date = date
        self.drive = drive
        self.max_frames = max_frames
        self.drive_dir = self.data_root / self.date / self.drive

    def exists(self) -> bool:
        """Return whether the expected drive directory contains OXTS packets."""
        return (self.drive_dir / "oxts" / "data").exists()

    def load(self) -> KITTITrajectory:
        """Load the configured drive and convert GPS to a local tangent plane."""
        if not self.exists():
            raise FileNotFoundError(f"KITTI drive not found at {self.drive_dir}")

        oxts_dir = self.drive_dir / "oxts" / "data"
        packets = load_oxts_packets(oxts_dir, self.max_frames)
        arrays = packets_to_arrays(packets)

        timestamp_path = self._resolve_timestamp_path()
        time_s = load_timestamps(timestamp_path)
        if self.max_frames is not None:
            time_s = time_s[: self.max_frames]
        time_s = time_s[: len(packets)]

        px, py = gps_to_local_xy(arrays["lat"], arrays["lon"])
        yaw = unwrap_yaw(arrays["yaw"])
        # KITTI OXTS reports body-frame forward/lateral velocity. The planar
        # speed is what enters the kinematic bicycle model.
        velocity = np.sqrt(arrays["vf"] ** 2 + arrays["vl"] ** 2)
        acceleration = np.gradient(velocity, time_s, edge_order=1) if len(time_s) > 1 else np.zeros_like(velocity)

        image_dir = self.drive_dir / "image_02" / "data"
        image_paths = sorted(image_dir.glob("*.png"))
        if self.max_frames is not None:
            image_paths = image_paths[: self.max_frames]

        return KITTITrajectory(
            time_s=time_s,
            px=px[: len(time_s)],
            py=py[: len(time_s)],
            yaw=yaw[: len(time_s)],
            velocity=velocity[: len(time_s)],
            acceleration=acceleration[: len(time_s)],
            image_paths=image_paths,
            source="kitti",
        )

    def _resolve_timestamp_path(self) -> Path:
        """Find KITTI timestamps in the common raw sequence locations."""
        candidates = [
            self.drive_dir / "oxts" / "timestamps.txt",
            self.drive_dir / "timestamps.txt",
            self.drive_dir / "image_02" / "timestamps.txt",
        ]
        for path in candidates:
            if path.exists():
                return path
        searched = "\n".join(str(path) for path in candidates)
        raise FileNotFoundError(f"No KITTI timestamp file found. Checked:\n{searched}")


def make_synthetic_trajectory(n_points: int = 180, dt: float = 0.1) -> KITTITrajectory:
    """Create a deterministic road-like path for smoke-testing without KITTI files."""
    time_s = np.arange(n_points, dtype=float) * dt
    px = 0.85 * time_s * 10.0
    py = 2.5 * np.sin(px / 24.0) + 0.25 * np.sin(px / 7.5)
    yaw = np.unwrap(np.arctan2(np.gradient(py), np.gradient(px)))
    velocity = np.maximum(0.1, np.gradient(px, time_s) / np.maximum(np.cos(yaw), 0.2))
    velocity = np.clip(velocity, 4.0, 12.0)
    acceleration = np.gradient(velocity, time_s, edge_order=1)
    return KITTITrajectory(time_s, px, py, yaw, velocity, acceleration, [], "synthetic")
