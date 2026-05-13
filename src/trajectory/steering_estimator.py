"""Geometry-based steering estimates for the human reference trajectory."""

from __future__ import annotations

import numpy as np


def estimate_curvature(px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Estimate signed planar curvature from sampled local-frame positions."""
    dx = np.gradient(px)
    dy = np.gradient(py)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denominator = np.maximum((dx * dx + dy * dy) ** 1.5, 1e-6)
    return (dx * ddy - dy * ddx) / denominator


def estimate_steering_from_curvature(px: np.ndarray, py: np.ndarray, wheelbase_m: float) -> np.ndarray:
    """Map path curvature to the bicycle steering angle delta = atan(L kappa)."""
    curvature = estimate_curvature(px, py)
    return np.arctan(wheelbase_m * curvature)
