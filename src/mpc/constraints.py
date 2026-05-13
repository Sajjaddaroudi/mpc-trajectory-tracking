"""Small helpers for translating vehicle limits into optimizer bounds."""

from __future__ import annotations

from vehicle_model.vehicle_constraints import VehicleConstraints


def control_bounds(constraints: VehicleConstraints) -> tuple[list[float], list[float]]:
    """Return lower and upper bounds for u = [delta, a]."""
    return (
        [-constraints.max_steering_rad, constraints.min_acceleration_mps2],
        [constraints.max_steering_rad, constraints.max_acceleration_mps2],
    )
