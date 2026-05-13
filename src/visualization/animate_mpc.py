"""Optional animation hook for MPC trajectory rollouts."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_mpc_animation(
    states: np.ndarray,
    reference: np.ndarray,
    predictions: list[np.ndarray],
    output_path: Path,
) -> None:
    """Placeholder for future video generation.

    The static publication plots are the default artifact for the baseline.
    This hook keeps the visualization API stable for later animation work.
    """
    _ = states, reference, predictions
    output_path.parent.mkdir(parents=True, exist_ok=True)
