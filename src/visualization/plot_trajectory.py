"""Trajectory plotting utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory(
    states: np.ndarray,
    reference: np.ndarray,
    predictions: list[np.ndarray],
    output_path: Path,
    dpi: int,
) -> None:
    """Save actual-vs-reference trajectory with sampled MPC horizons."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.0, 5.0))
    plt.plot(reference[:, 0], reference[:, 1], color="#1f77b4", linewidth=2.2, label="KITTI reference")
    plt.plot(states[:, 0], states[:, 1], color="#d62728", linewidth=1.9, label="MPC rollout")
    stride = max(1, len(predictions) // 12)
    for idx, prediction in enumerate(predictions[::stride]):
        label = "MPC prediction horizon" if idx == 0 else None
        plt.plot(prediction[:, 0], prediction[:, 1], color="#2ca02c", alpha=0.25, linewidth=1.0, label=label)
    plt.axis("equal")
    plt.xlabel("Local x [m]")
    plt.ylabel("Local y [m]")
    plt.title("Trajectory Tracking")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_tracking_error(errors: np.ndarray, output_path: Path, dpi: int) -> None:
    """Save tracking RMSE over time as position error."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.0, 3.2))
    plt.plot(errors, color="#9467bd", linewidth=1.8)
    plt.xlabel("MPC step")
    plt.ylabel("Position error [m]")
    plt.title("Tracking Error Over Time")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_lateral_error(errors: np.ndarray, output_path: Path, dpi: int) -> None:
    """Save signed lateral cross-track error over time."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.0, 3.2))
    plt.axhline(0.0, color="#222222", linewidth=0.8, alpha=0.5)
    plt.plot(errors, color="#8c564b", linewidth=1.8)
    plt.xlabel("MPC step")
    plt.ylabel("Lateral error [m]")
    plt.title("Signed Lateral Tracking Error")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_heading_error(errors: np.ndarray, output_path: Path, dpi: int) -> None:
    """Save wrapped heading error over time."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.0, 3.2))
    plt.axhline(0.0, color="#222222", linewidth=0.8, alpha=0.5)
    plt.plot(errors, color="#e377c2", linewidth=1.8)
    plt.xlabel("MPC step")
    plt.ylabel("Heading error [rad]")
    plt.title("Heading Tracking Error")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()
