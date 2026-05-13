# KITTI Raw Dataset Setup

This repository is configured for the KITTI raw sequence:

- `2011_09_26_drive_0011_sync`
- `2011_09_26_calib`

Place the extracted KITTI files under `data/KITTI/` so the final layout is:

```text
data/KITTI/
+-- 2011_09_26/
    +-- 2011_09_26_calib/
    +-- 2011_09_26_drive_0011_sync/
        +-- image_02/
        +-- oxts/
        +-- velodyne_points/
```

Only the following components are required for the initial classical MPC baseline:

- `oxts/`: GPS/IMU packets used to recover pose, yaw, and velocity.
- `oxts/timestamps.txt`: frame timing used for numerical derivatives. The loader also accepts `timestamps.txt` at the drive root or `image_02/timestamps.txt` as fallbacks.
- `image_02/`: forward camera frames retained for visualization and future latent-vision work.

The `velodyne_points/` directory is not used by the current pipeline, but the folder is shown because it is part of the standard KITTI raw sequence layout and may be useful for future extensions.

The executable entry point is:

```bash
python src/main.py
```

If the KITTI files are not present, the default configuration runs a deterministic synthetic trajectory so the MPC implementation can be smoke-tested before downloading the dataset. Set `data.use_synthetic_if_missing: false` in `configs/mpc_config.yaml` to require the real dataset.
