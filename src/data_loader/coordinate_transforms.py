"""Coordinate conversion utilities for KITTI GPS trajectories."""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_M = 6_378_137.0


def gps_to_local_xy(lat_deg: np.ndarray, lon_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert latitude and longitude to local Cartesian meters using equirectangular projection."""
    if len(lat_deg) == 0:
        return np.asarray([]), np.asarray([])

    lat = np.deg2rad(lat_deg.astype(float))
    lon = np.deg2rad(lon_deg.astype(float))
    lat0 = lat[0]
    lon0 = lon[0]

    x = EARTH_RADIUS_M * (lon - lon0) * np.cos(lat0)
    y = EARTH_RADIUS_M * (lat - lat0)
    return x, y


def unwrap_yaw(yaw_rad: np.ndarray) -> np.ndarray:
    """Return a continuous yaw profile in radians."""
    return np.unwrap(yaw_rad.astype(float))


def yaw_from_positions(px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Estimate yaw from planar positions when orientation is unavailable."""
    dx = np.gradient(px)
    dy = np.gradient(py)
    return unwrap_yaw(np.arctan2(dy, dx))
