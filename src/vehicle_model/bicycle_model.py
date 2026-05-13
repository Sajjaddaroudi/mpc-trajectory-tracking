"""Kinematic bicycle model used by the classical MPC baseline.

The model is intentionally modest: no tire slip, no load transfer, and no
actuator lag. For this KITTI ego-motion experiment that is a feature rather
than a shortcut, because the first baseline should expose the tracking problem
without hiding it behind a high-parameter vehicle model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VehicleState:
    """Vehicle state x = [px, py, v, psi] in a local Cartesian frame."""

    px: float
    py: float
    v: float
    yaw: float

    def as_array(self) -> np.ndarray:
        """Return the state as a float NumPy array."""
        return np.asarray([self.px, self.py, self.v, self.yaw], dtype=float)


@dataclass(frozen=True)
class BicycleModel:
    """Forward-Euler discretization of the kinematic bicycle equations."""

    wheelbase_m: float

    def step(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        """Propagate one sample with u = [delta, a]."""
        px, py, v, psi = state
        delta, acceleration = control
        next_state = np.array(
            [
                px + v * np.cos(psi) * dt,
                py + v * np.sin(psi) * dt,
                v + acceleration * dt,
                psi + (v / self.wheelbase_m) * np.tan(delta) * dt,
            ],
            dtype=float,
        )
        # Store heading on the principal branch for cleaner error plots and
        # easier comparison with the wrapped heading residual in the optimizer.
        next_state[3] = np.arctan2(np.sin(next_state[3]), np.cos(next_state[3]))
        return next_state
