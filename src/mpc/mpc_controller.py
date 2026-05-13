"""Closed-loop receding-horizon controller.

The optimizer is built once and then reused by updating parameters for the
current measured state and the next slice of the KITTI reference. Keeping the
symbolic problem fixed is important for repeatable timing and avoids rebuilding
IPOPT graphs inside the simulation loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cost_function import MPCWeights
from .optimizer import OptimizerBundle, build_optimizer
from trajectory.reference_generator import ReferenceTrajectory
from vehicle_model.bicycle_model import BicycleModel
from vehicle_model.vehicle_constraints import VehicleConstraints


@dataclass
class MPCResult:
    """Closed-loop trajectory, applied controls, and horizon predictions."""

    states: np.ndarray
    controls: np.ndarray
    predictions: list[np.ndarray]
    success: list[bool]


class MPCController:
    """Combined longitudinal-lateral nonlinear MPC controller."""

    def __init__(
        self,
        model: BicycleModel,
        constraints: VehicleConstraints,
        weights: MPCWeights,
        horizon: int,
        dt: float,
        max_iterations: int,
        ipopt_print_level: int,
        print_time: bool,
    ) -> None:
        self.model = model
        self.constraints = constraints
        self.horizon = horizon
        self.dt = dt
        self.bundle: OptimizerBundle = build_optimizer(
            horizon=horizon,
            dt=dt,
            wheelbase_m=model.wheelbase_m,
            weights=weights,
            constraints=constraints,
            max_iterations=max_iterations,
            ipopt_print_level=ipopt_print_level,
            print_time=print_time,
        )
        self._last_x: np.ndarray | None = None
        self._last_u: np.ndarray | None = None

    def solve(self, current_state: np.ndarray, reference_window: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
        """Solve one horizon and return the command applied to the vehicle.

        Warm starts use the previous predicted trajectory and a one-step-shifted
        control sequence. That small bookkeeping detail matters in practice:
        it makes consecutive solves more consistent and reduces unnecessary
        optimizer work once the controller has settled onto the path.
        """
        opti = self.bundle.opti
        opti.set_value(self.bundle.x0, current_state)
        opti.set_value(self.bundle.ref, reference_window.T)

        if self._last_x is not None:
            opti.set_initial(self.bundle.x, self._last_x)
        else:
            opti.set_initial(self.bundle.x, reference_window.T)
        if self._last_u is not None:
            opti.set_initial(self.bundle.u, self._last_u)
        else:
            opti.set_initial(self.bundle.u, np.zeros((2, self.horizon)))

        try:
            solution = opti.solve()
            predicted_states = np.asarray(solution.value(self.bundle.x)).T
            controls = np.asarray(solution.value(self.bundle.u)).T
            success = True
        except RuntimeError:
            # Keep the simulation moving so failure cases are visible in the
            # metrics rather than hidden by an exception. The debug values are
            # often still informative when IPOPT exits early.
            predicted_states = np.asarray(opti.debug.value(self.bundle.x)).T
            controls = np.asarray(opti.debug.value(self.bundle.u)).T
            if not np.all(np.isfinite(controls)):
                controls = np.zeros((self.horizon, 2), dtype=float)
            success = False

        self._last_x = predicted_states.T
        self._last_u = np.vstack([controls[1:], controls[-1:]]).T
        return controls[0], predicted_states, success

    def rollout(self, reference: ReferenceTrajectory) -> MPCResult:
        """Simulate the controller against a full reference trajectory."""
        n_steps = max(1, len(reference.states) - self.horizon - 1)
        states = np.zeros((n_steps + 1, 4), dtype=float)
        controls = np.zeros((n_steps, 2), dtype=float)
        predictions: list[np.ndarray] = []
        success: list[bool] = []
        states[0] = reference.states[0]

        for k in range(n_steps):
            window = reference.horizon(k, self.horizon)
            control, predicted, solved = self.solve(states[k], window)
            control[0] = np.clip(control[0], -self.constraints.max_steering_rad, self.constraints.max_steering_rad)
            control[1] = np.clip(
                control[1],
                self.constraints.min_acceleration_mps2,
                self.constraints.max_acceleration_mps2,
            )
            controls[k] = control
            states[k + 1] = self.model.step(states[k], control, self.dt)
            states[k + 1, 2] = np.clip(
                states[k + 1, 2],
                self.constraints.min_velocity_mps,
                self.constraints.max_velocity_mps,
            )
            predictions.append(predicted)
            success.append(solved)

        return MPCResult(states=states, controls=controls, predictions=predictions, success=success)
