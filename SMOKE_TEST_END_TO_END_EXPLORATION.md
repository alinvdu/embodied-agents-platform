# Smoke Test: End-to-End Real Exploration Stack

This runbook starts the real robot brain, offload ROS/Nav2 stack, and the Mode B smoke test.

Mode B means:

```text
robot brain asks offload Nav2 router for current pose
robot brain asks offload Nav2 router for compute_path_to_pose
robot brain follows the path locally with x.vel/theta.vel
robot brain records final pose through RGB-D odometry
```

Replace these placeholders:

```text
ROBOT_BRAIN_IP = robot brain network address
OFFLOAD_IP = ROS/Nav2 offload computer network address
```

Required data flow:

```text
robot brain Orbbec sidecar writes RGB/depth files
robot brain agent serves /rgb and /depth over HTTP
offload real_ros_bridge fetches /rgb and /depth
offload real_ros_bridge publishes /camera/head/* and /scan
offload rgbd_visual_odometry consumes /camera/head/*
offload rgbd_visual_odometry publishes /odom and odom -> base_link
Nav2 needs /map plus map -> odom -> base_link before planning can succeed
```

Do not start the moving smoke test until `/camera/head/camera_info`, `/odom`, and `odom -> base_link` work on the offload computer.

## Robot Brain

### Terminal RB-1: Robot Brain Agent

Keep the robot wheels raised for the first run.

```bash
cd /home/alin/Robot42

python -m xlerobot_playground.robot_brain_agent \
  --allow-motion-commands \
  --max-linear-m-s 0.03 \
  --max-angular-rad-s 0.10
```

### Terminal RB-2: Orbbec RGB-D Sidecar

```bash
cd /home/alin/Robot42

cmake -S tools/orbbec_rgb_test -B build/orbbec_rgb_test -DORBBEC_SDK_ROOT="$HOME/orbbec/sdk"
cmake --build build/orbbec_rgb_test

sudo ./build/orbbec_rgb_test/orbbec_rgb_test --list-profiles

sudo ./build/orbbec_rgb_test/orbbec_rgb_test \
  --frames 0 \
  --no-file-output \
  --enable-depth \
  --enable-depth-registration \
  --enable-imu \
  --imu-udp-host 127.0.0.1 \
  --imu-udp-port 8766 \
  --camera-http-enable \
  --camera-http-host 127.0.0.1 \
  --camera-http-port 8765 \
  --camera-http-path /camera/rgbd
```

Check the robot brain locally:

```bash
curl http://127.0.0.1:8765/health
curl -I http://127.0.0.1:8765/rgb
curl -I http://127.0.0.1:8765/depth
```

## Offload ROS/Nav2 Computer

### Terminal OFF-0: Generate SLAM and Nav2 Params

```bash
cd /home/alin/Robot42
source /opt/ros/humble/setup.bash

python -m xlerobot_playground.real_nav2_config \
  --base-nav2-params /opt/ros/humble/share/nav2_bringup/params/nav2_params.yaml \
  --output-dir /home/alin/Robot42/artifacts/nav2 \
  --scan-topic /scan \
  --map-frame map \
  --odom-frame odom \
  --base-frame base_link \
  --max-laser-range 6.0 \
  --max-linear-velocity 0.03 \
  --max-angular-velocity 0.10 \
  --local-costmap-width 2 \
  --local-costmap-height 2
```

This writes:

```text
/home/alin/Robot42/artifacts/nav2/xlerobot_slam_toolbox.yaml
/home/alin/Robot42/artifacts/nav2/xlerobot_nav2_params.yaml
```

### Terminal OFF-1: Real ROS Bridge

This process is required. It is what turns the robot brain HTTP camera files into ROS topics. Without it, `/camera/head/camera_info` will not exist, RGB-D visual odometry cannot run, `/odom` will not exist, and Nav2 will fail with `base_link` to `map` transform errors.

First verify the offload computer can fetch camera data from the robot brain:

```bash
curl --max-time 3 http://ROBOT_BRAIN_IP:8765/health
curl --max-time 3 http://ROBOT_BRAIN_IP:8765/rgbd --output /tmp/xlerobot_rgbd.bin
curl --max-time 3 http://ROBOT_BRAIN_IP:8765/rgb --output /tmp/xlerobot_rgb.ppm
curl --max-time 3 http://ROBOT_BRAIN_IP:8765/depth --output /tmp/xlerobot_depth.pgm
curl --max-time 3 http://ROBOT_BRAIN_IP:8765/imu
ls -lh /tmp/xlerobot_rgbd.bin /tmp/xlerobot_rgb.ppm /tmp/xlerobot_depth.pgm
```

Then start the bridge:

```bash
cd /home/alin/Robot42
source /opt/ros/humble/setup.bash

python -m xlerobot_playground.real_ros_bridge \
  --robot-brain-url http://ROBOT_BRAIN_IP:8765 \
  --publish-rate-hz 30 \
  --camera-x-m 0.24 \
  --camera-y-m 0.0 \
  --camera-z-m 0.35 \
  --camera-yaw-rad 0.0
```

In robot-brain mode the IMU path now comes from `ws://ROBOT_BRAIN_IP:8765/ws/imu`, not from the old `/imu` polling timer.

In another offload terminal, do not continue until these work:

```bash
ros2 topic echo /camera/head/camera_info --once
ros2 topic echo /camera/head/image_raw --once
ros2 topic echo /camera/head/depth/image_raw --once
ros2 topic echo /scan --once
ros2 topic echo /imu --once
```

### Terminal OFF-2: IMU Yaw Filter

```bash
cd /home/alin/Robot42
source /opt/ros/humble/setup.bash

python -m xlerobot_playground.imu_yaw_filter \
  --imu-topic /imu \
  --output-topic /imu/filtered_yaw \
  --input-frame-convention camera_optical \
  --yaw-source gyro_y \
  --bias-calibration-s 0.5 \
  --yaw-rate-lowpass-alpha 0.2
```

This matches the OrbbecViewer CSV path: integrate corrected `gyro_y`.
Keep the robot still for the first 0.5 seconds after startup so the yaw filter calibrates gyro bias cleanly.

In another offload terminal, do not continue until this works:

```bash
ros2 topic echo /imu/filtered_yaw --once
```

### Terminal OFF-3: RGB-D Visual Odometry

Start this only after the `real_ros_bridge` camera topics above are alive. This process creates `/odom` from RGB-D and publishes `odom -> base_link`.

```bash
cd /home/alin/Robot42
source /opt/ros/humble/setup.bash

python -m xlerobot_playground.rgbd_visual_odometry \
  --rgb-topic /camera/head/image_raw \
  --depth-topic /camera/head/depth/image_raw \
  --camera-info-topic /camera/head/camera_info \
  --imu-topic /imu/filtered_yaw \
  --odom-topic /odom \
  --imu-frame-convention base_link \
  --imu-bias-calibration-s 0.0 \
  --publish-rate-hz 30
```

This consumes the filtered yaw IMU topic, which is already bias-corrected and expressed in `base_link`.

In another offload terminal, do not continue until these work:

```bash
ros2 topic echo /odom --once
ros2 run tf2_ros tf2_echo odom base_link
```

### Terminal OFF-4: SLAM Toolbox Or Fake Map

For the current tiny smoke-test experiment, use the fake map path first. Skip SLAM Toolbox and let the Nav2 router publish a small all-free map. This is enough for:

```text
forward 5cm
rotate left 5deg
rotate right 5deg
```

The fake map only replaces `map -> odom` and `/map`. It still requires RGB-D visual odometry to publish `odom -> base_link`.

Do not run this SLAM Toolbox command for the fake-map smoke test:

```bash
cd /home/alin/Robot42
source /opt/ros/humble/setup.bash

ros2 launch slam_toolbox online_async_launch.py \
  use_sim_time:=false \
  slam_params_file:=/home/alin/Robot42/artifacts/nav2/xlerobot_slam_toolbox.yaml
```

Use SLAM Toolbox later when the real `/scan` and odometry path are stable.

### Terminal OFF-5: Nav2

```bash
cd /home/alin/Robot42
source /opt/ros/humble/setup.bash

ros2 launch nav2_bringup navigation_launch.py \
  use_sim_time:=false \
  use_composition:=False \
  params_file:=/home/alin/Robot42/artifacts/nav2/xlerobot_nav2_params.yaml
```

### Terminal OFF-6: Nav2 HTTP Router With Fake Map

```bash
cd /home/alin/Robot42
source /opt/ros/humble/setup.bash

python -m xlerobot_playground.ros_nav2_router \
  --host 0.0.0.0 \
  --port 8891 \
  --no-publish-clock \
  --no-publish-external-state-tf \
  --fake-free-map \
  --fake-map-size-m 2.0 \
  --fake-map-resolution-m 0.02
```

This publishes `/map` as a centered free 2m x 2m occupancy grid and publishes identity `map -> odom`. It does not publish fake `odom -> base_link`; RGB-D visual odometry still owns that transform.

Later, when using SLAM Toolbox instead of the fake map, remove these fake-map flags:

```text
--fake-free-map
--fake-map-size-m 2.0
--fake-map-resolution-m 0.02
```

## Verification

Run these on the offload computer:

```bash
ros2 topic echo /camera/head/camera_info --once
ros2 topic echo /scan --once
ros2 topic echo /odom --once
ros2 run tf2_ros tf2_echo odom base_link
ros2 run tf2_ros tf2_echo map odom
ros2 action list | grep compute_path_to_pose
```

Run these from the robot brain:

```bash
curl http://OFFLOAD_IP:8891/api/health
curl http://OFFLOAD_IP:8891/api/router/current_pose
```

Run this no-motion preflight from the robot brain before the real smoke test:

```bash
python -m xlerobot_playground.robot_brain_smoke_test \
  --router-url http://OFFLOAD_IP:8891 \
  --robot-brain-url http://127.0.0.1:8765 \
  --preflight-only
```

This checks the offload router, local robot brain agent, and RGB-D odometry pose path without sending wheel commands.

## Run The Smoke Tests From Robot Brain

### Terminal RB-3A: Motor Smoke Test

Run this first. It sends timed low-speed commands and verifies that the robot brain command path responds. It does not validate centimeter-level odometry accuracy.

```bash
cd /home/alin/Robot42

python -m xlerobot_playground.robot_brain_smoke_test \
  --robot-brain-url http://127.0.0.1:8765 \
  --motor-smoke-only \
  --forward-m 0.05 \
  --turn-deg 5 \
  --max-linear-m-s 0.03 \
  --max-angular-rad-s 0.10 \
  --robot-timeout-s 10 \
  --step-timeout-s 15
```

Expected success:

```json
{
  "ok": true,
  "mode": "motor_smoke_only_open_loop"
}
```

### Terminal RB-3B: Nav2 Participation Smoke Test

Run this after motor smoke works. This is the main "is Nav2 in the loop?" test: the robot brain asks the offload router/Nav2 for paths, follows them through the real motors, and reports the final pose error. By default, pose accuracy is diagnostic only, so the test can pass even if the real robot overshoots or turns imperfectly on soil.

```bash
cd /home/alin/Robot42

python -m xlerobot_playground.robot_brain_smoke_test \
  --router-url http://OFFLOAD_IP:8891 \
  --robot-brain-url http://127.0.0.1:8765 \
  --forward-m 0.05 \
  --turn-deg 5 \
  --max-linear-m-s 0.03 \
  --max-angular-rad-s 0.10 \
  --goal-distance-tolerance-m 0.025 \
  --yaw-tolerance-deg 2.5 \
  --pose-validation-mode diagnostic \
  --max-translation-error-m 0.05 \
  --max-yaw-error-deg 5 \
  --robot-timeout-s 10 \
  --step-timeout-s 15
```

Expected success:

```json
{
  "ok": true,
  "mode": "robot_brain_follows_nav2_path",
  "pose_validation_mode": "diagnostic"
}
```

The smoke test executes:

```text
forward 5cm
rotate left 5deg
rotate right 5deg
```

Each result includes:

```text
path_pose_count
path_start
path_end
reached_goal
timed_out
translation_error_m
yaw_error_deg
last_cmd_linear_m_s
last_cmd_angular_rad_s
```

In `diagnostic` mode, `timed_out: true` or a larger final pose error means "Nav2 was used, but real-world following/odometry needs tuning." It does not mean the robot command path failed.

### Optional: Strict Pose Accuracy Test

Use this later, after odometry and controller behavior look sane. This mode fails if the robot cannot reach the final pose tolerance before timeout.

```bash
python -m xlerobot_playground.robot_brain_smoke_test \
  --router-url http://OFFLOAD_IP:8891 \
  --robot-brain-url http://127.0.0.1:8765 \
  --forward-m 0.05 \
  --turn-deg 5 \
  --max-linear-m-s 0.03 \
  --max-angular-rad-s 0.10 \
  --goal-distance-tolerance-m 0.025 \
  --yaw-tolerance-deg 2.5 \
  --pose-validation-mode strict \
  --max-translation-error-m 0.05 \
  --max-yaw-error-deg 5 \
  --robot-timeout-s 10 \
  --step-timeout-s 15 \
  --debug-progress
```

## What To Watch

- `/depth` should be non-empty on the robot brain.
- `/scan` should not be all zero or all max range.
- `/odom` should be stable when stationary.
- `/odom` should move in the expected direction when the robot moves.
- The robot brain agent should stop after every smoke-test step.
- Nav2 should expose `compute_path_to_pose`.
- The final smoke-test JSON should report small translation and yaw errors.

## Current Missing Pieces

- Real validation of Orbbec depth and RGB-D odometry quality.
- Real measurement for `base_link -> head_camera_link`.
- No-Nav2 calibration script is still only a TODO.
- Nav2 parameters may need tuning after the first real run.
