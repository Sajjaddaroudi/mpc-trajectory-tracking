"""Vehicle limits used as hard constraints in the MPC problem."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VehicleConstraints:
    """Speed and actuator bounds for the baseline passenger-car model."""

    min_velocity_mps: float
    max_velocity_mps: float
    max_steering_rad: float
    min_acceleration_mps2: float
    max_acceleration_mps2: float
