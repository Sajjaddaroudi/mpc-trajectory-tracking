"""Reference trajectory construction for receding-horizon tracking.

KITTI gives us ego-motion rather than a planned lane-center trajectory. For
this baseline, the human-driven path is treated as the reference that the MPC
should reproduce. The smoothing here is intentionally light: enough to avoid
derivative noise in steering estimates, but not enough to erase the geometry of
the original drive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

from data_loader.kitti_loader import KITTITrajectory
from .steering_estimator import estimate_steering_from_curvature


@dataclass
class ReferenceTrajectory:
    """Smoothed KITTI path and feedforward quantities inferred from geometry."""

    states: np.ndarray
    steering: np.ndarray
    acceleration: np.ndarray
    time_s: np.ndarray
    source: str

    def horizon(self, start_index: int, horizon: int) -> np.ndarray:
        """Return the future reference window used by one MPC solve.

        Near the end of the sequence, the final state is repeated so the NLP
        dimensions stay fixed. This mirrors how a real tracking controller would
        hold the terminal reference once it runs out of preview.
        """
        end = start_index + horizon + 1
        if end <= len(self.states):
            return self.states[start_index:end]
        pad_count = end - len(self.states)
        tail = np.repeat(self.states[-1][None, :], pad_count, axis=0)
        return np.vstack([self.states[start_index:], tail])


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Apply a small Savitzky-Golay filter when enough samples are available."""
    if len(values) < 5 or window < 5:
        return values
    odd_window = min(window if window % 2 == 1 else window + 1, len(values) - (1 - len(values) % 2))
    if odd_window < 5:
        return values
    return savgol_filter(values, odd_window, polyorder=2, mode="interp")


def generate_reference_trajectory(
    trajectory: KITTITrajectory,
    wheelbase_m: float,
    smoothing_window: int,
    minimum_speed_mps: float,
) -> ReferenceTrajectory:
    """Construct x_ref = [px, py, v, psi] from one loaded KITTI trajectory."""
    px = _smooth(trajectory.px, smoothing_window)
    py = _smooth(trajectory.py, smoothing_window)
    yaw = np.unwrap(_smooth(trajectory.yaw, smoothing_window))
    velocity = np.maximum(_smooth(trajectory.velocity, smoothing_window), minimum_speed_mps)
    states = np.column_stack([px, py, velocity, yaw])
    steering = estimate_steering_from_curvature(px, py, wheelbase_m)
    acceleration = _smooth(trajectory.acceleration, smoothing_window)
    return ReferenceTrajectory(states, steering, acceleration, trajectory.time_s, trajectory.source)
