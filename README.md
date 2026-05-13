# Classical Longitudinal-Lateral MPC for Autonomous Driving

A research-oriented implementation of classical Model Predictive Control (MPC) for autonomous vehicle trajectory tracking using the KITTI raw autonomous driving dataset.

This project implements a combined longitudinal and lateral MPC controller based on a kinematic bicycle model. The controller tracks a reference trajectory while generating smooth steering and acceleration commands under vehicle dynamics and actuator constraints.

The goal of this repository is to build a strong classical-control baseline before integrating learned perception, latent trajectory prediction, and learning-based control models.

---

# Project Motivation

A large amount of modern autonomous-driving research focuses heavily on deep learning and end-to-end policies. However, optimization-based control methods such as MPC remain important because they provide:

- physically feasible control
- smooth actuator behavior
- explicit handling of constraints
- interpretable optimization objectives
- stable trajectory tracking

In this project, the controller receives a reference trajectory derived from KITTI ego-motion data and solves a finite-horizon nonlinear optimization problem at every timestep to determine the optimal steering and acceleration commands.

This repository is intended as the control foundation for future work involving:

```text
camera frames
-> latent representations
-> learned future trajectory prediction
-> MPC trajectory tracking
```

---

# Vehicle Model

The controller uses a discrete-time kinematic bicycle model with state

$$
\mathbf{x}_k =
\begin{bmatrix}
p_{x,k} & p_{y,k} & v_k & \psi_k
\end{bmatrix}^{\mathsf{T}}
$$

and control input

$$
\mathbf{u}_k =
\begin{bmatrix}
\delta_k & a_k
\end{bmatrix}^{\mathsf{T}}
$$

The prediction dynamics are:

$$
\begin{aligned}
p_{x,k+1} &= p_{x,k} + v_k \cos(\psi_k)\Delta t \\
p_{y,k+1} &= p_{y,k} + v_k \sin(\psi_k)\Delta t \\
v_{k+1} &= v_k + a_k \Delta t \\
\psi_{k+1} &= \psi_k + \frac{v_k}{L}\tan(\delta_k)\Delta t
\end{aligned}
$$

where:

- $p_x, p_y$: vehicle position in the local Cartesian frame
- $\psi$: vehicle heading/yaw
- $v$: longitudinal velocity
- $a$: acceleration command
- $\delta$: steering command
- `L`: wheelbase
- $\Delta t$: sampling time

This formulation gives the controller true lateral authority: steering changes heading, and heading propagates future `px` and `py` motion.

---

# MPC Objective

At each timestep, the controller optimizes a finite-horizon cost function that penalizes:

- trajectory tracking error
- heading error
- velocity tracking error
- steering effort
- acceleration effort
- abrupt steering changes
- abrupt acceleration changes
- terminal position error

The optimization problem is:

$$
\begin{aligned}
\min_{\mathbf{X}, \mathbf{U}} \quad
&\sum_{k=0}^{N-1}
Q_p \lVert \mathbf{p}_k - \mathbf{p}_{k}^{\mathrm{ref}} \rVert_2^2
+ Q_v (v_k - v_{k}^{\mathrm{ref}})^2 \\
&+ Q_{\psi}\operatorname{wrap}(\psi_k - \psi_{k}^{\mathrm{ref}})^2
+ R_{\delta}\delta_k^2
+ R_a a_k^2 \\
&+ R_{\Delta\delta}(\delta_k - \delta_{k-1})^2
+ R_{\Delta a}(a_k - a_{k-1})^2 \\
&+ Q_f \lVert \mathbf{p}_N - \mathbf{p}_{N}^{\mathrm{ref}} \rVert_2^2
\end{aligned}
$$

subject to:

$$
\begin{aligned}
v_{\min} &\le v_k \le v_{\max} \\
|\delta_k| &\le \delta_{\max} \\
a_{\min} &\le a_k \le a_{\max}
\end{aligned}
$$

Only the first optimized control input is applied before the optimization is repeated at the next timestep. This is the standard receding-horizon MPC procedure.

The nonlinear optimization problem is implemented with CasADi and solved with IPOPT.

---

# Dataset

Experiments use the KITTI raw autonomous-driving dataset.

Current evaluations use:

```text
2011_09_26_drive_0011_sync
```

Expected data layout:

```text
data/KITTI/
+-- 2011_09_26/
    +-- 2011_09_26_calib/
    +-- 2011_09_26_drive_0011_sync/
        +-- image_02/
        |   +-- data/
        +-- oxts/
        |   +-- data/
        |   +-- timestamps.txt
        +-- velodyne_points/
```

The current classical MPC pipeline uses:

- `oxts/data/*.txt`
- `oxts/timestamps.txt`
- optionally `image_02/data/*.png` for camera availability checks and future vision work

The OXTS measurements provide vehicle position, orientation, velocity, and ego-motion information. These are converted into a local Cartesian trajectory for MPC tracking.

More detail is provided in [data/README_DATA.md](data/README_DATA.md).

---

# Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

The project uses:

- NumPy
- SciPy
- CasADi
- Matplotlib
- OpenCV
- PyYAML
- tqdm

---

# Running Experiments

From the repository root:

```bash
python src/main.py
```

On Windows, if `python` points to a different interpreter, use:

```bash
py src/main.py
```

The experiment configuration is stored in:

```text
configs/mpc_config.yaml
```

The main tunable MPC parameters are:

- `mpc.horizon`
- `mpc.weights.position`
- `mpc.weights.yaw`
- `mpc.weights.velocity`
- `mpc.weights.steering`
- `mpc.weights.acceleration`
- `mpc.weights.steering_rate`
- `mpc.weights.acceleration_rate`
- `mpc.weights.terminal_position`

---

# Results

The controller demonstrates stable trajectory tracking across both longitudinal and lateral motion. The generated steering and acceleration commands remain physically realistic while maintaining low tracking error throughout the trajectory.

The default configuration is manually tuned for `2011_09_26_drive_0011_sync` over the first 220 frames.

Reference run metrics:

```text
trajectory_rmse_m: 0.027688
lateral_rmse_m: 0.013462
max_lateral_error_m: 0.044503
heading_rmse_rad: 0.014823
velocity_rmse_mps: 0.238742
steering_smoothness: 0.000024
solver_success_rate: 1.000000
```

---

## Trajectory Tracking

The MPC trajectory closely follows the reference trajectory with small deviations during higher-curvature sections of the path.

![Trajectory Tracking](outputs/plots/trajectory_tracking.png)

The controller maintains accurate path following while also satisfying steering, acceleration, and velocity constraints.

---

## Velocity Tracking

The longitudinal controller tracks the reference velocity smoothly without high-frequency oscillatory behavior.

![Velocity Tracking](outputs/plots/velocity_tracking.png)

The current tuned configuration prioritizes lateral and heading tracking while keeping velocity tracking stable. A small initial velocity transient may remain because the first solve has no previous applied control history.

---

## Lateral Tracking Error

Most of the trajectory remains within a very small lateral tracking error range.

![Lateral Error](outputs/plots/lateral_error.png)

The signed lateral error is computed in the reference path frame by projecting position error onto the normal direction of the reference heading. This separates cross-track error from along-track timing error.

---

## Heading Error

Heading tracking remains stable throughout the trajectory without high-frequency oscillations.

![Heading Error](outputs/plots/heading_error.png)

The largest heading deviations occur mainly during higher-curvature segments of the reference path.

---

## Control Commands

The generated steering and acceleration commands remain smooth and physically plausible.

![Control Commands](outputs/plots/control_commands.png)

The steering controller avoids excessive oscillation while maintaining accurate trajectory tracking.

---

## Tracking Error

Overall tracking error remains bounded and stable during the rollout.

![Tracking Error](outputs/plots/tracking_error.png)

---

# Repository Structure

```text
src/
+-- data_loader/          KITTI OXTS parsing, timestamps, coordinate transforms
+-- vehicle_model/        kinematic bicycle model and vehicle constraints
+-- trajectory/           reference generation and steering estimation
+-- mpc/                  CasADi optimizer, objective, and controller
+-- evaluation/           lateral, heading, velocity, smoothness, and energy metrics
+-- visualization/        trajectory, control, velocity, and error plots
+-- future_extensions/    camera latent and latent-MPC placeholders
```

Generated experiment artifacts are saved to:

```text
outputs/
+-- plots/
+-- logs/
+-- videos/
```

The MPC architecture audit is saved at:

```text
results/mpc_architecture_audit.md
```

---

# Current Status

At the current stage, this project focuses entirely on classical control and assumes access to:

- accurate vehicle state estimation
- a known reference trajectory
- KITTI OXTS ego-motion data

The system does not yet include learned perception or trajectory prediction.

However, the current MPC implementation provides a strong and stable baseline that can later be extended using camera-based learned trajectory generation.

---

# Future Work

The next major direction of this project is integrating learned visual representations into the control pipeline.

The planned pipeline is:

```text
camera frames
-> encoder
-> latent representation
-> future trajectory prediction
-> MPC
-> steering + acceleration
```

Instead of relying on ground-truth future trajectories, the controller will eventually track trajectories predicted directly from visual information.

Additional future directions include:

- latent-based future trajectory prediction
- uncertainty-aware MPC
- camera-based road curvature estimation
- nonlinear MPC with richer state constraints
- dynamic bicycle models
- obstacle-aware MPC constraints
- learned trajectory refinement
- perception-control integration

---

# Research Direction

This repository is intended as a bridge between:

- classical control theory
- machine learning
- predictive perception
- autonomous-driving systems

The long-term objective is to combine learned scene understanding with optimization-based control for robust and physically feasible autonomous navigation.
