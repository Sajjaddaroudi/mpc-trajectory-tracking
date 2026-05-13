"""Control command plotting utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_controls(controls: np.ndarray, output_path: Path, dpi: int) -> None:
    """Save steering and acceleration command profiles."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.0), sharex=True)
    axes[0].plot(controls[:, 0], color="#17becf", linewidth=1.8)
    axes[0].set_ylabel("Steering [rad]")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(controls[:, 1], color="#ff7f0e", linewidth=1.8)
    axes[1].set_xlabel("MPC step")
    axes[1].set_ylabel("Acceleration [m/s^2]")
    axes[1].grid(True, alpha=0.25)
    fig.suptitle("Optimized Control Commands")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
