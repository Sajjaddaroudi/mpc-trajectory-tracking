"""Control quality metrics."""

from __future__ import annotations

import numpy as np


def steering_smoothness(controls: np.ndarray) -> float:
    """Compute mean squared steering-rate proxy from adjacent commands."""
    if len(controls) < 2:
        return 0.0
    return float(np.mean(np.diff(controls[:, 0]) ** 2))


def control_energy(controls: np.ndarray) -> float:
    """Compute total quadratic control effort."""
    if len(controls) == 0:
        return 0.0
    return float(np.sum(controls[:, 0] ** 2 + controls[:, 1] ** 2))
