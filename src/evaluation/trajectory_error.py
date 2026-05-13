"""Path-tracking error definitions used in the experiment report."""

from __future__ import annotations

import numpy as np


def position_errors(predicted: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return Euclidean position error ||p - p_ref|| at each step."""
    n = min(len(predicted), len(reference))
    return np.linalg.norm(predicted[:n, 0:2] - reference[:n, 0:2], axis=1)


def lateral_errors(predicted: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return signed cross-track errors in the reference path frame.

    Positive values indicate the vehicle lies to the left of the reference
    heading direction. Reporting this separately is useful because Euclidean
    position error mixes cross-track error with along-track timing error.
    """
    n = min(len(predicted), len(reference))
    position_error = predicted[:n, 0:2] - reference[:n, 0:2]
    ref_yaw = reference[:n, 3]
    left_normals = np.column_stack([-np.sin(ref_yaw), np.cos(ref_yaw)])
    return np.sum(position_error * left_normals, axis=1)


def heading_errors(predicted: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return wrapped heading error psi - psi_ref in radians."""
    n = min(len(predicted), len(reference))
    diff = predicted[:n, 3] - reference[:n, 3]
    return np.arctan2(np.sin(diff), np.cos(diff))


def velocity_errors(predicted: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return signed longitudinal velocity error in meters per second."""
    n = min(len(predicted), len(reference))
    return predicted[:n, 2] - reference[:n, 2]
