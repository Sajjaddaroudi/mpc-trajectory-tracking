"""Nonlinear MPC transcription for the KITTI path-tracking experiment.

The formulation in this file is deliberately kept close to the notation used
in controls papers: X is the predicted state trajectory, U is the control
sequence, and the reference parameter carries the local-frame trajectory over
the current horizon. This makes the baseline easier to audit before adding
learned perception or dynamics models.
"""

from __future__ import annotations

from dataclasses import dataclass

import casadi as ca

from .cost_function import MPCWeights
from vehicle_model.vehicle_constraints import VehicleConstraints


@dataclass
class OptimizerBundle:
    """CasADi problem handles reused at every receding-horizon step."""

    opti: ca.Opti
    x: ca.MX
    u: ca.MX
    x0: ca.MX
    ref: ca.MX


def build_optimizer(
    horizon: int,
    dt: float,
    wheelbase_m: float,
    weights: MPCWeights,
    constraints: VehicleConstraints,
    max_iterations: int,
    ipopt_print_level: int,
    print_time: bool,
) -> OptimizerBundle:
    """Build the finite-horizon nonlinear program.

    State ordering is x = [px, py, v, psi]. Control ordering is
    u = [delta, a]. The model is the standard low-speed kinematic bicycle
    discretized with forward Euler integration. This is sufficient for a
    transparent KITTI ego-motion baseline; a dynamic bicycle model can be
    dropped in later without changing the experiment interface.
    """
    opti = ca.Opti()
    x = opti.variable(4, horizon + 1)
    u = opti.variable(2, horizon)
    x0 = opti.parameter(4)
    ref = opti.parameter(4, horizon + 1)

    opti.subject_to(x[:, 0] == x0)
    objective = 0

    for k in range(horizon):
        px_k, py_k, v_k, psi_k = x[0, k], x[1, k], x[2, k], x[3, k]
        delta_k, a_k = u[0, k], u[1, k]

        next_state = ca.vertcat(
            px_k + v_k * ca.cos(psi_k) * dt,
            py_k + v_k * ca.sin(psi_k) * dt,
            v_k + a_k * dt,
            psi_k + (v_k / wheelbase_m) * ca.tan(delta_k) * dt,
        )
        opti.subject_to(x[:, k + 1] == next_state)

        pos_error = x[0:2, k] - ref[0:2, k]
        psi_error = ca.atan2(ca.sin(x[3, k] - ref[3, k]), ca.cos(x[3, k] - ref[3, k]))
        objective += weights.position * ca.sumsqr(pos_error)
        objective += weights.velocity * (x[2, k] - ref[2, k]) ** 2
        objective += weights.yaw * psi_error**2
        objective += weights.steering * delta_k**2
        objective += weights.acceleration * a_k**2

        if k > 0:
            # Penalizing increments is a simple actuator-smoothness model. It
            # It also discourages high-frequency steering corrections between
            # adjacent shooting intervals.
            du = u[:, k] - u[:, k - 1]
            objective += weights.steering_rate * du[0] ** 2
            objective += weights.acceleration_rate * du[1] ** 2

    terminal_error = x[0:2, horizon] - ref[0:2, horizon]
    objective += weights.terminal_position * ca.sumsqr(terminal_error)

    opti.subject_to(opti.bounded(constraints.min_velocity_mps, x[2, :], constraints.max_velocity_mps))
    opti.subject_to(opti.bounded(-constraints.max_steering_rad, u[0, :], constraints.max_steering_rad))
    opti.subject_to(opti.bounded(constraints.min_acceleration_mps2, u[1, :], constraints.max_acceleration_mps2))
    opti.minimize(objective)

    options = {
        "ipopt.print_level": ipopt_print_level,
        "print_time": print_time,
        "ipopt.max_iter": max_iterations,
        "ipopt.sb": "yes",
    }
    opti.solver("ipopt", options)
    return OptimizerBundle(opti=opti, x=x, u=u, x0=x0, ref=ref)
