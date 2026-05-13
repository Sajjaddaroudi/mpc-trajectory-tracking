"""Executable KITTI trajectory-tracking MPC experiment."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from data_loader.kitti_loader import KITTILoader, make_synthetic_trajectory
from evaluation.metrics import evaluate_run, write_summary_report
from mpc.cost_function import MPCWeights
from mpc.mpc_controller import MPCController, MPCResult
from trajectory.reference_generator import ReferenceTrajectory, generate_reference_trajectory
from vehicle_model.bicycle_model import BicycleModel
from vehicle_model.vehicle_constraints import VehicleConstraints
from visualization.plot_controls import plot_controls
from visualization.plot_trajectory import (
    plot_heading_error,
    plot_lateral_error,
    plot_tracking_error,
    plot_trajectory,
)
from visualization.plot_velocity import plot_velocity


def project_root() -> Path:
    """Return repository root when executed as `python src/main.py`."""
    return Path(__file__).resolve().parents[1]


def load_config(config_path: Path) -> dict:
    """Load experiment configuration from YAML."""
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_trajectory(config: dict, root: Path):
    """Load KITTI if present, otherwise optionally use a deterministic synthetic trajectory."""
    paths = config["paths"]
    data_cfg = config["data"]
    loader = KITTILoader(
        data_root=root / paths["data_root"],
        date=paths["date"],
        drive=paths["drive"],
        max_frames=data_cfg.get("max_frames"),
    )
    if loader.exists():
        return loader.load()
    if data_cfg.get("use_synthetic_if_missing", False):
        return make_synthetic_trajectory(
            n_points=int(data_cfg.get("synthetic_points", 180)),
            dt=float(config["mpc"]["dt"]),
        )
    raise FileNotFoundError(
        f"KITTI sequence not found at {loader.drive_dir}. See data/README_DATA.md for setup instructions."
    )


def camera_metadata(image_paths: list[Path]) -> dict[str, str | int]:
    """Read the first image with OpenCV to document camera availability."""
    if not image_paths:
        return {"camera_frames": 0, "first_image_shape": "not available"}
    image = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
    if image is None:
        return {"camera_frames": len(image_paths), "first_image_shape": "unreadable"}
    height, width = image.shape[:2]
    return {"camera_frames": len(image_paths), "first_image_shape": f"{height}x{width}"}


def build_controller(config: dict) -> MPCController:
    """Instantiate vehicle constraints, model, weights, and MPC controller."""
    vehicle_cfg = config["vehicle"]
    mpc_cfg = config["mpc"]
    model = BicycleModel(wheelbase_m=float(vehicle_cfg["wheelbase_m"]))
    constraints = VehicleConstraints(
        min_velocity_mps=float(vehicle_cfg["min_velocity_mps"]),
        max_velocity_mps=float(vehicle_cfg["max_velocity_mps"]),
        max_steering_rad=float(vehicle_cfg["max_steering_rad"]),
        min_acceleration_mps2=float(vehicle_cfg["min_acceleration_mps2"]),
        max_acceleration_mps2=float(vehicle_cfg["max_acceleration_mps2"]),
    )
    weights = MPCWeights(**{key: float(value) for key, value in mpc_cfg["weights"].items()})
    return MPCController(
        model=model,
        constraints=constraints,
        weights=weights,
        horizon=int(mpc_cfg["horizon"]),
        dt=float(mpc_cfg["dt"]),
        max_iterations=int(mpc_cfg["max_iterations"]),
        ipopt_print_level=int(mpc_cfg["ipopt_print_level"]),
        print_time=bool(mpc_cfg["print_time"]),
    )


def run_with_progress(controller: MPCController, reference: ReferenceTrajectory) -> MPCResult:
    """Run the MPC loop while exposing progress at experiment level."""
    n_steps = max(1, len(reference.states) - controller.horizon - 1)
    states = np.zeros((n_steps + 1, 4), dtype=float)
    controls = np.zeros((n_steps, 2), dtype=float)
    predictions: list[np.ndarray] = []
    success: list[bool] = []
    states[0] = reference.states[0]

    for k in tqdm(range(n_steps), desc="Running MPC", unit="step"):
        window = reference.horizon(k, controller.horizon)
        control, predicted, solved = controller.solve(states[k], window)
        control[0] = np.clip(
            control[0],
            -controller.constraints.max_steering_rad,
            controller.constraints.max_steering_rad,
        )
        control[1] = np.clip(
            control[1],
            controller.constraints.min_acceleration_mps2,
            controller.constraints.max_acceleration_mps2,
        )
        controls[k] = control
        states[k + 1] = controller.model.step(states[k], control, controller.dt)
        states[k + 1, 2] = np.clip(
            states[k + 1, 2],
            controller.constraints.min_velocity_mps,
            controller.constraints.max_velocity_mps,
        )
        predictions.append(predicted)
        success.append(solved)

    return MPCResult(states=states, controls=controls, predictions=predictions, success=success)


def save_outputs(
    config: dict,
    root: Path,
    reference: ReferenceTrajectory,
    result: MPCResult,
    metadata: dict[str, str | int | float],
) -> None:
    """Evaluate the run and save plots plus the text report."""
    output_root = root / config["paths"]["output_root"]
    plots_dir = output_root / "plots"
    logs_dir = output_root / "logs"
    dpi = int(config["visualization"]["dpi"])

    ref_aligned = reference.states[: len(result.states)]
    summary, series = evaluate_run(result.states, ref_aligned, result.controls, result.success)
    write_summary_report(logs_dir / "mpc_summary.txt", summary, metadata)

    plot_trajectory(result.states, ref_aligned, result.predictions, plots_dir / "trajectory_tracking.png", dpi)
    plot_controls(result.controls, plots_dir / "control_commands.png", dpi)
    plot_velocity(result.states, ref_aligned, plots_dir / "velocity_tracking.png", dpi)
    plot_tracking_error(series["position_error_m"], plots_dir / "tracking_error.png", dpi)
    plot_lateral_error(series["lateral_error_m"], plots_dir / "lateral_error.png", dpi)
    plot_heading_error(series["heading_error_rad"], plots_dir / "heading_error.png", dpi)


def main() -> None:
    """Run the complete KITTI MPC pipeline."""
    root = project_root()
    config = load_config(root / "configs" / "mpc_config.yaml")
    np.random.seed(int(config["project"]["random_seed"]))

    trajectory = load_trajectory(config, root)
    reference = generate_reference_trajectory(
        trajectory=trajectory,
        wheelbase_m=float(config["vehicle"]["wheelbase_m"]),
        smoothing_window=int(config["reference"]["smoothing_window"]),
        minimum_speed_mps=float(config["reference"]["minimum_speed_mps"]),
    )
    controller = build_controller(config)
    result = run_with_progress(controller, reference)

    metadata: dict[str, str | int | float] = {
        "project": config["project"]["name"],
        "trajectory_source": trajectory.source,
        "sequence": config["paths"]["drive"],
        "num_reference_states": len(reference.states),
        "mpc_horizon": int(config["mpc"]["horizon"]),
        "dt_s": float(config["mpc"]["dt"]),
    }
    metadata.update(camera_metadata(trajectory.image_paths))
    save_outputs(config, root, reference, result, metadata)

    print("Experiment complete. Results saved to outputs/plots/ and outputs/logs/mpc_summary.txt")


if __name__ == "__main__":
    main()
