"""Velocity tracking plotting utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_velocity(states: np.ndarray, reference: np.ndarray, output_path: Path, dpi: int) -> None:
    """Save velocity tracking profile."""
    n = min(len(states), len(reference))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.0, 3.2))
    plt.plot(reference[:n, 2], color="#1f77b4", linewidth=2.0, label="Reference")
    plt.plot(states[:n, 2], color="#d62728", linewidth=1.7, label="MPC")
    plt.xlabel("MPC step")
    plt.ylabel("Velocity [m/s]")
    plt.title("Velocity Tracking")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()
