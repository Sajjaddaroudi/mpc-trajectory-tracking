# MPC Architecture Audit

## Executive Conclusion

Case A: This project already contains a true combined longitudinal-lateral MPC controller.

The implementation is not longitudinal-only. It optimizes both steering and acceleration, predicts local planar motion with a kinematic bicycle model, tracks a future reference trajectory containing x position, y position, yaw, and velocity, and penalizes XY tracking error, yaw error, velocity error, steering effort, acceleration effort, and control smoothness.

This audit also identified missing lateral-evaluation artifacts. Those gaps have been addressed by adding signed lateral error, lateral RMSE, max lateral error, heading RMSE, and dedicated lateral/heading error plots.

## 1. State Vector

Verified state vector:

```text
x = [px, py, v, yaw]
```

Implementation evidence:

- `src/vehicle_model/bicycle_model.py` defines `VehicleState` as `px`, `py`, `v`, `yaw`.
- `src/mpc/optimizer.py` creates `x = opti.variable(4, horizon + 1)`.
- `src/trajectory/reference_generator.py` builds reference states as `np.column_stack([px, py, velocity, yaw])`.

The state includes lateral-motion variables because it contains both local planar position components and heading. Lateral path-tracking is therefore represented in the optimizer state, not merely post-processed.

## 2. Control Vector

Verified control vector:

```text
u = [steering, acceleration]
```

Implementation evidence:

- `src/mpc/optimizer.py` creates `u = opti.variable(2, horizon)`.
- The optimizer unpacks `steering, acceleration = u[0, k], u[1, k]`.
- `src/vehicle_model/bicycle_model.py` applies the same control order in the rollout model.

The controller solves for steering angle and acceleration at every horizon step.

## 3. Vehicle Dynamics

Verified dynamics are equivalent to the standard kinematic bicycle model:

```text
px_{k+1}  = px_k + v_k cos(yaw_k) dt
py_{k+1}  = py_k + v_k sin(yaw_k) dt
v_{k+1}   = v_k + a_k dt
yaw_{k+1} = yaw_k + (v_k / L) tan(delta_k) dt
```

Implementation evidence:

- The same equations appear in `src/mpc/optimizer.py` for prediction constraints.
- The same equations appear in `src/vehicle_model/bicycle_model.py` for closed-loop rollout.

This means steering has a real lateral effect: steering changes yaw, and yaw changes the future x/y trajectory. The MPC is therefore capable of lateral path correction.

## 4. MPC Cost Function

Verified objective terms:

- x tracking error: yes, through `x[0:2, k] - ref[0:2, k]`.
- y tracking error: yes, through the same planar position term.
- heading/yaw error: yes, with wrapped yaw error.
- velocity error: yes.
- steering effort: yes.
- acceleration effort: yes.
- steering smoothness: yes, through `u[:, k] - u[:, k - 1]`.
- acceleration smoothness: yes.
- terminal position cost: yes.

The controller is not velocity-only. It has explicit path-tracking terms for both local x/y position and yaw.

## 5. Reference Trajectory

Verified reference state:

```text
x_ref = [px_ref, py_ref, v_ref, yaw_ref]
```

Implementation evidence:

- KITTI GPS is converted to local Cartesian `px` and `py`.
- OXTS yaw is unwrapped and included.
- OXTS velocity is included.
- `ReferenceTrajectory.horizon()` returns a future sequence of full four-state references to the optimizer.

The controller receives a full spatial trajectory, not only a velocity profile.

## 6. Outputs and Plots

Verified and updated outputs:

- XY trajectory plot: `outputs/plots/trajectory_tracking.png`.
- Steering-angle plot: `outputs/plots/control_commands.png`.
- Velocity tracking plot: `outputs/plots/velocity_tracking.png`.
- Euclidean tracking error plot: `outputs/plots/tracking_error.png`.
- Lateral error plot: `outputs/plots/lateral_error.png`.
- Heading error plot: `outputs/plots/heading_error.png`.

The added lateral and heading plots make lateral-control evaluation explicit and publication-ready.

## 7. Metrics

Verified and updated metrics:

- trajectory RMSE: already present.
- lateral RMSE: added.
- max lateral error: added.
- heading RMSE: added.
- mean absolute heading error: already present.
- velocity RMSE: already present.
- steering smoothness: already present.
- control energy: already present.
- solver success rate: already present.

Signed lateral error is computed by projecting position error onto the left-normal vector of the reference heading:

```text
e_lat = ([px, py] - [px_ref, py_ref]) dot [-sin(yaw_ref), cos(yaw_ref)]
```

This follows standard path-tracking convention and distinguishes cross-track error from along-track timing error.

## 8. Technical Assessment

The implementation satisfies the standard autonomous-driving kinematic bicycle MPC structure:

- full planar pose and speed state,
- steering plus acceleration controls,
- nonlinear bicycle dynamics,
- spatial reference tracking,
- heading tracking,
- longitudinal speed tracking,
- bounded steering, acceleration, and velocity,
- receding-horizon optimization with warm starts,
- lateral and longitudinal evaluation metrics.

The only identified deficiencies were in evaluation/reporting rather than the core MPC formulation. Those have been fixed.
