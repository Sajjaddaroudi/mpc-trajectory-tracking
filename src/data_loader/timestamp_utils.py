"""Timestamp parsing and time-step utilities for KITTI raw data."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np


def load_timestamps(path: Path) -> np.ndarray:
    """Load KITTI timestamps and return elapsed seconds from the first frame."""
    stamps: list[datetime] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            text = line.strip()
            if not text:
                continue
            stamps.append(datetime.strptime(text[:26], "%Y-%m-%d %H:%M:%S.%f"))

    if not stamps:
        raise ValueError(f"No timestamps found in {path}")

    t0 = stamps[0]
    return np.asarray([(stamp - t0).total_seconds() for stamp in stamps], dtype=float)


def estimate_dt(times_s: np.ndarray, fallback_dt: float = 0.1) -> np.ndarray:
    """Estimate per-frame sample times with a stable fallback for degenerate gaps."""
    if len(times_s) < 2:
        return np.asarray([fallback_dt], dtype=float)

    dt = np.diff(times_s, prepend=times_s[0])
    first_valid = np.median(np.diff(times_s)) if len(times_s) > 2 else fallback_dt
    dt[0] = first_valid
    dt[dt <= 1e-4] = first_valid
    return dt.astype(float)
