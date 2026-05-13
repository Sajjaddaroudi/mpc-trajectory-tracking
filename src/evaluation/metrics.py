"""Experiment-level metrics for combined longitudinal-lateral tracking."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .control_smoothness import control_energy, steering_smoothness
from .trajectory_error import heading_errors, lateral_errors, position_errors, velocity_errors


@dataclass
class EvaluationSummary:
    """Scalar quantities reported after one closed-loop rollout."""

    trajectory_rmse_m: float
    lateral_rmse_m: float
    max_lateral_error_m: float
    mean_heading_error_rad: float
    heading_rmse_rad: float
    velocity_rmse_mps: float
    steering_smoothness: float
    control_energy: float
    solver_success_rate: float


def rmse(values: np.ndarray) -> float:
    """Return root-mean-square value with a safe empty-array convention."""
    if len(values) == 0:
        return 0.0
    return float(np.sqrt(np.mean(values**2)))


def evaluate_run(
    states: np.ndarray,
    reference_states: np.ndarray,
    controls: np.ndarray,
    success: list[bool],
) -> tuple[EvaluationSummary, dict[str, np.ndarray]]:
    """Compute scalar metrics and retain the time series used for plots."""
    pos_error = position_errors(states, reference_states)
    lateral_error = lateral_errors(states, reference_states)
    yaw_error = heading_errors(states, reference_states)
    v_error = velocity_errors(states, reference_states)
    summary = EvaluationSummary(
        trajectory_rmse_m=rmse(pos_error),
        lateral_rmse_m=rmse(lateral_error),
        max_lateral_error_m=float(np.max(np.abs(lateral_error))) if len(lateral_error) else 0.0,
        mean_heading_error_rad=float(np.mean(np.abs(yaw_error))) if len(yaw_error) else 0.0,
        heading_rmse_rad=rmse(yaw_error),
        velocity_rmse_mps=rmse(v_error),
        steering_smoothness=steering_smoothness(controls),
        control_energy=control_energy(controls),
        solver_success_rate=float(np.mean(success)) if success else 0.0,
    )
    series = {
        "position_error_m": pos_error,
        "lateral_error_m": lateral_error,
        "heading_error_rad": yaw_error,
        "velocity_error_mps": v_error,
    }
    return summary, series


def write_summary_report(path: Path, summary: EvaluationSummary, metadata: dict[str, str | int | float]) -> None:
    """Write a compact text report that can be committed with experiment runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["KITTI Classical MPC Experiment Summary", "=" * 40, ""]
    lines.append("Metadata")
    for key, value in metadata.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "Metrics"])
    for key, value in asdict(summary).items():
        lines.append(f"- {key}: {value:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
