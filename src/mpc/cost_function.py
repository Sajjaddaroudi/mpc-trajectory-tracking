"""Weight container for the MPC quadratic objective."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MPCWeights:
    """Quadratic penalties in the finite-horizon tracking cost.

    The field names mirror the YAML config rather than mathematical symbols so
    experiment settings remain easy to read. In the objective, `position`
    corresponds to Q_p, `yaw` to Q_psi, and the rate fields to the usual
    actuator increment penalties.
    """

    position: float
    velocity: float
    yaw: float
    steering: float
    acceleration: float
    steering_rate: float
    acceleration_rate: float
    terminal_position: float
