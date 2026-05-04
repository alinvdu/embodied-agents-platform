# Real Robot Odometry Test Plan

This runbook validates the current real-robot odometry stack:

```text
Orbbec RGB-D + IMU sidecar
  -> robot_brain_agent in-memory /rgbd + /ws/imu
  -> real_ros_bridge publishes /camera/head/*, /scan, /imu, /cmd_vel forwarding
  -> imu_yaw_filter publishes /imu/filtered_yaw
  -> rgbd_visual_odometry publishes /odom and odom -> base_link
```

Current odometry split:

- RGB-D provides translation.
- Filtered gyro/IMU yaw provides rotation.
- Accelerometer double integration is not used for odometry position.

Run all motion tests with the robot on the floor in a clear area. Keep an emergency stop terminal ready.

## Parameters

On the offload machine:

```bash
export ROBOT_BRAIN_IP=192.168.1.133
export ROBOT_BRAIN_URL=http://${ROBOT_BRAIN_IP}:8765
export ROBOT42=/home/alin/Robot42
```

For a cautious first pass:

```bash
export MAX_LINEAR=0.03
export MAX_ANGULAR=0.10
```

At `0.03 m/s`, a 1 m forward test takes about 34 seconds.
At `0.10 rad/s`, a 90 degree rotation takes about 16 seconds.

## Terminal RB-1: Robot Brain Agent

Run this on the robot brain Mac before starting the Orbbec sidecar.

```bash
cd /Users/alin/Robot42
conda activate xlerobot
python -m pip install aiohttp

python -m xlerobot_playground.robot_brain_agent \
  --allow-motion-commands \
  --debug-motion \
  --robot-kind xlerobot_2wheels \
  --port1 /dev/tty.usbmodem5B140330101 \
  --port2 /dev/tty.usbmodem5B140332271 \
  --max-linear-m-s 0.03 \
  --max-angular-rad-s 0.10
```

Expected:

- Logs show the HTTP agent is ready on port `8765`.
- `/health` later reports `"motion_enabled": true`.

## Terminal RB-2: Orbbec RGB-D + IMU Sidecar

Run this on the robot brain Mac after RB-1 is running.

```bash
cd /Users/alin/Robot42

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
  --camera-http-path /camera/rgbd \
  --camera-http-timeout-ms 100 \
  --log-every 30 \
  --imu-log-every 200
```

If the default aligned depth profile is too heavy, use the listed Gemini 2 `Y16 640x400@30` depth source profile:

```bash
sudo ./build/orbbec_rgb_test/orbbec_rgb_test \
  --frames 0 \
  --no-file-output \
  --enable-depth \
  --enable-depth-registration \
  --enable-imu \
  --depth-width 640 \
  --depth-height 400 \
  --depth-fps 30 \
  --imu-udp-host 127.0.0.1 \
  --imu-udp-port 8766 \
  --camera-http-enable \
  --camera-http-host 127.0.0.1 \
  --camera-http-port 8765 \
  --camera-http-path /camera/rgbd \
  --camera-http-timeout-ms 100 \
  --log-every 30 \
  --imu-log-every 200
```

Expected:

- Logs show RGB-D HTTP publisher enabled.
- Logs show RGB frames and depth frames at roughly 30 FPS.
- No repeated `camera RGB-D HTTP publish failed` warnings.

## Terminal OFF-1: Robot Brain Health

Run on the offload machine.

```bash
cd "$ROBOT42"

curl --max-time 3 "${ROBOT_BRAIN_URL}/health" | python -m json.tool
curl --max-time 3 "${ROBOT_BRAIN_URL}/rgbd" --output /tmp/xlerobot_rgbd.bin
curl --max-time 3 "${ROBOT_BRAIN_URL}/rgb" --output /tmp/xlerobot_rgb.ppm
curl --max-time 3 "${ROBOT_BRAIN_URL}/depth" --output /tmp/xlerobot_depth.pgm
curl --max-time 3 "${ROBOT_BRAIN_URL}/imu" | python -m json.tool
ls -lh /tmp/xlerobot_rgbd.bin /tmp/xlerobot_rgb.ppm /tmp/xlerobot_depth.pgm
```

Expected:

- Health JSON has `ok: true`.
- Health JSON has `motion_enabled: true`.
- Health JSON has `rgbd.ready: true`.
- Health JSON has `imu.ready: true`.
- RGB and depth files are non-empty.

## Terminal OFF-2: ROS Bridge

Run on the offload machine.

```bash
cd "$ROBOT42"
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate

python -m xlerobot_playground.real_ros_bridge \
  --robot-brain-url "${ROBOT_BRAIN_URL}" \
  --publish-rate-hz 30 \
  --cmd-vel-timeout-s 0.5 \
  --max-linear-m-s 0.03 \
  --max-angular-rad-s 0.10 \
  --camera-x-m 0.0 \
  --camera-y-m 0.0 \
  --camera-z-m 0.35 \
  --camera-yaw-rad 0.0 \
  --no-laser-fill-no-return
```

Expected:

- Logs show `IMU websocket connected`.
- No repeated motion forwarding errors.

## Terminal OFF-3: ROS Topic Sanity

Run on another offload terminal.

```bash
cd "$ROBOT42"
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate

ros2 topic hz /camera/head/image_raw
```

Let it run for 5-10 seconds, then stop it with Ctrl-C. Expected: close to `30 Hz`.

```bash
ros2 topic hz /camera/head/depth/image_raw
```

Expected: close to `30 Hz`.

```bash
ros2 topic hz /imu
```

Expected: high rate. Around `200 Hz` is the target.

```bash
ros2 topic echo /camera/head/camera_info --once
ros2 topic echo /scan --once
```

Expected:

- `camera_info.width` and `camera_info.height` match the active depth/RGB-D stream used by the bridge.
- `/scan` has non-empty ranges.

## Terminal OFF-4: Filtered IMU Yaw

Keep the robot still during startup/bias calibration.

```bash
cd "$ROBOT42"
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate

python -m xlerobot_playground.imu_yaw_filter \
  --imu-topic /imu \
  --output-topic /imu/filtered_yaw \
  --input-frame-convention camera_optical \
  --yaw-source gyro_y \
  --bias-calibration-s 0.5 \
  --yaw-rate-lowpass-alpha 0.2
```

In another terminal:

```bash
source /opt/ros/humble/setup.bash
ros2 topic echo /imu/filtered_yaw --once
ros2 topic hz /imu/filtered_yaw
```

Expected:

- `/imu/filtered_yaw` exists.
- Orientation covariance is non-negative.
- Rate is high enough for yaw integration. Ideally it follows `/imu`.

## Terminal OFF-5: RGB-D Visual Odometry

```bash
cd "$ROBOT42"
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate

python -m xlerobot_playground.rgbd_visual_odometry \
  --rgb-topic /camera/head/image_raw \
  --depth-topic /camera/head/depth/image_raw \
  --camera-info-topic /camera/head/camera_info \
  --imu-topic /imu/filtered_yaw \
  --odom-topic /odom \
  --imu-frame-convention base_link \
  --imu-bias-calibration-s 0.0 \
  --publish-rate-hz 30 \
  --min-translation-update-m 0.01
```

In another terminal:

```bash
source /opt/ros/humble/setup.bash
ros2 topic hz /odom
ros2 topic echo /odom --once
ros2 run tf2_ros tf2_echo odom base_link
```

Expected:

- `/odom` exists and publishes near `30 Hz`.
- `tf2_echo odom base_link` works.
- While stationary, position and yaw should remain nearly stable.

## Safety Stop Command

Use this any time the robot should stop.

```bash
source /opt/ros/humble/setup.bash
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```

## Test 0: Stationary Drift

Purpose: verify `/odom` does not drift badly when the robot is not moving.

```bash
cd "$ROBOT42"
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate

mkdir -p artifacts/diagnostics

python -m xlerobot_playground.ros_rotation_diagnostic \
  --duration-s 60 \
  --sample-hz 10 \
  --imu-topic /imu/filtered_yaw \
  --imu-bias-calibration-s 0.0 \
  --target-source tf \
  --csv-out artifacts/diagnostics/stationary_yaw.csv \
  --json-out artifacts/diagnostics/stationary_yaw_summary.json

python -m json.tool artifacts/diagnostics/stationary_yaw_summary.json
```

Pass criteria:

- `tf.translation_drift_m` should be small, target below `0.03 m`.
- `tf.unwrapped_yaw_delta_deg` should be small, target below `3 deg`.
- If yaw drifts more than this while stationary, fix IMU yaw before moving.

## Test 1: 90 Degree Left Rotation

Purpose: validate authoritative IMU yaw through `/odom` and `odom -> base_link`.

This command publishes `/cmd_vel` until TF reports a 90 degree yaw change, or until the safety timeout.

```bash
cd "$ROBOT42"
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate

python -m xlerobot_playground.ros_rotation_diagnostic \
  --send-motion \
  --duration-s 30 \
  --sample-hz 20 \
  --angular-rad-s 0.10 \
  --target-yaw-deg 90 \
  --target-source tf \
  --imu-topic /imu/filtered_yaw \
  --imu-bias-calibration-s 0.0 \
  --csv-out artifacts/diagnostics/rotate_left_90.csv \
  --json-out artifacts/diagnostics/rotate_left_90_summary.json

python -m json.tool artifacts/diagnostics/rotate_left_90_summary.json
```

Expected:

- Robot rotates left in place and stops around 90 degrees.
- `stop_reason` is `target_tf_yaw_reached`.
- `tf.unwrapped_yaw_delta_deg` is near `+90`.
- Translation drift during rotation is small, target below `0.10 m`.

Pass criteria:

- Yaw error from 90 degrees is below `5 deg`.
- Robot does not translate significantly while rotating.

## Test 2: 90 Degree Right Rotation

Purpose: validate sign symmetry.

```bash
cd "$ROBOT42"
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate

python -m xlerobot_playground.ros_rotation_diagnostic \
  --send-motion \
  --duration-s 30 \
  --sample-hz 20 \
  --angular-rad-s -0.10 \
  --target-yaw-deg 90 \
  --target-source tf \
  --imu-topic /imu/filtered_yaw \
  --imu-bias-calibration-s 0.0 \
  --csv-out artifacts/diagnostics/rotate_right_90.csv \
  --json-out artifacts/diagnostics/rotate_right_90_summary.json

python -m json.tool artifacts/diagnostics/rotate_right_90_summary.json
```

Expected:

- Robot rotates right in place and stops around 90 degrees.
- `stop_reason` is `target_tf_yaw_reached`.
- `tf.unwrapped_yaw_delta_deg` is near `-90`.

Pass criteria:

- Absolute yaw error from 90 degrees is below `5 deg`.
- Direction is correct. If it rotates left, yaw sign is wrong.

## Test 3: Forward 0.25 m

Purpose: short, safer translation validation before 1 m.

```bash
cd "$ROBOT42"
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate

python -m xlerobot_playground.ros_forward_accel_diagnostic \
  --send-motion \
  --duration-s 15 \
  --sample-hz 30 \
  --linear-m-s 0.03 \
  --target-distance-m 0.25 \
  --target-source tf \
  --imu-topic /imu/filtered_yaw \
  --imu-frame-convention base_link \
  --accel-bias-calibration-s 0.0 \
  --max-imu-staleness-s 0.5 \
  --csv-out artifacts/diagnostics/forward_025m.csv \
  --json-out artifacts/diagnostics/forward_025m_summary.json

python -m json.tool artifacts/diagnostics/forward_025m_summary.json
```

Expected:

- Robot moves forward and stops around 0.25 m.
- `target_source` is `tf`.
- `stop_reason` is `target_tf_distance_reached`.

Pass criteria:

- TF/odom forward distance is within `0.05 m` of 0.25 m.
- Lateral drift is below `0.05 m`.
- Yaw drift is below `5 deg`.

## Test 4: Forward 1.0 m

Purpose: validate RGB-D translation over a useful distance.

Use only after the 0.25 m test passes.

```bash
cd "$ROBOT42"
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate

python -m xlerobot_playground.ros_forward_accel_diagnostic \
  --send-motion \
  --duration-s 45 \
  --sample-hz 30 \
  --linear-m-s 0.03 \
  --target-distance-m 1.0 \
  --target-source tf \
  --imu-topic /imu/filtered_yaw \
  --imu-frame-convention base_link \
  --accel-bias-calibration-s 0.0 \
  --max-imu-staleness-s 0.5 \
  --csv-out artifacts/diagnostics/forward_1m.csv \
  --json-out artifacts/diagnostics/forward_1m_summary.json

python -m json.tool artifacts/diagnostics/forward_1m_summary.json
```

Expected:

- Robot moves forward and stops around 1 m according to TF.
- RGB-D VO should produce continuous forward odom movement.
- Yaw should remain close to the starting yaw because IMU yaw is authoritative.

Pass criteria:

- Forward distance is within `0.10 m` of 1.0 m.
- Lateral drift is below `0.10 m`.
- Yaw drift is below `5 deg`.

If the robot physically travels much less or much more than `/odom`, the odometry scale or RGB-D alignment is wrong.

## Test 5: Square Path Consistency

Purpose: detect accumulated rotation or translation bias.

Run the sequence manually. Give the robot a clear 2 m x 2 m area.

```bash
cd "$ROBOT42"
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate

for i in 1 2 3 4; do
  echo "Square side $i: forward 0.5 m"
  python -m xlerobot_playground.ros_forward_accel_diagnostic \
    --send-motion \
    --duration-s 25 \
    --sample-hz 30 \
    --linear-m-s 0.03 \
    --target-distance-m 0.5 \
    --target-source tf \
    --imu-topic /imu/filtered_yaw \
    --imu-frame-convention base_link \
    --accel-bias-calibration-s 0.0 \
    --max-imu-staleness-s 0.5 \
    --csv-out "artifacts/diagnostics/square_forward_${i}.csv" \
    --json-out "artifacts/diagnostics/square_forward_${i}_summary.json"

  echo "Square corner $i: rotate left 90 deg"
  python -m xlerobot_playground.ros_rotation_diagnostic \
    --send-motion \
    --duration-s 30 \
    --sample-hz 20 \
    --angular-rad-s 0.10 \
    --target-yaw-deg 90 \
    --target-source tf \
    --imu-topic /imu/filtered_yaw \
    --imu-bias-calibration-s 0.0 \
    --csv-out "artifacts/diagnostics/square_rotate_${i}.csv" \
    --json-out "artifacts/diagnostics/square_rotate_${i}_summary.json"
done
```

After the sequence:

```bash
ros2 topic echo /odom --once
ros2 run tf2_ros tf2_echo odom base_link
```

Pass criteria:

- Final yaw should be close to the starting yaw.
- Final position should be roughly near the starting position, allowing real wheel slip and visual drift.
- If yaw is good but position is bad, focus on RGB-D registration, frame sync, feature quality, and camera intrinsics.

## Optional: Raw Open-Loop Commands

Use these only for quick motor direction checks. They do not stop based on odometry.

Forward for about 1 m at 0.03 m/s:

```bash
source /opt/ros/humble/setup.bash

timeout 34 ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.03, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"

ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```

Left 90 degrees at 0.10 rad/s:

```bash
source /opt/ros/humble/setup.bash

timeout 16 ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.10}}"

ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```

Right 90 degrees at 0.10 rad/s:

```bash
source /opt/ros/humble/setup.bash

timeout 16 ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: -0.10}}"

ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```

## Optional: Nav2 Smoke Test

Use this only after direct odometry tests pass and `/scan`, `/odom`, and `map -> odom -> base_link` are available.

```bash
cd "$ROBOT42"
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate

python -m xlerobot_playground.real_nav2_smoke_test \
  --robot-brain-url "${ROBOT_BRAIN_URL}" \
  --odom-topic /odom \
  --scan-topic /scan \
  --cmd-vel-topic /cmd_vel \
  --forward-m 0.10 \
  --turn-deg 10 \
  --max-translation-error-m 0.06 \
  --max-yaw-error-deg 6 \
  --nav-timeout-s 20
```

This is not the first odometry validation. It tests Nav2 plus odometry plus command forwarding.

## Data To Save

After each test run, keep:

```bash
ls -lh artifacts/diagnostics/*summary.json artifacts/diagnostics/*.csv
```

Useful summaries:

```bash
python -m json.tool artifacts/diagnostics/rotate_left_90_summary.json
python -m json.tool artifacts/diagnostics/rotate_right_90_summary.json
python -m json.tool artifacts/diagnostics/forward_025m_summary.json
python -m json.tool artifacts/diagnostics/forward_1m_summary.json
```

## Troubleshooting

If `/camera/head/*` is not 30 Hz:

- Check RB-2 sidecar logs for HTTP publish warnings.
- Check `curl ${ROBOT_BRAIN_URL}/health`; `rgbd.age_s` should stay low.
- Raw RGB-D over per-frame HTTP may be too heavy; the next transport improvement is a persistent socket or compression.

If rotation direction is wrong:

- Check `imu_yaw_filter --yaw-source`.
- For Gemini 2, current runbook uses `--yaw-source gyro_y`.
- Re-run the 90 degree tests before testing translation.

If yaw is stable but forward odometry is bad:

- Check RGB/depth resolution and registration.
- Check that `/camera/head/image_raw` and `/camera/head/depth/image_raw` are both live.
- Check that the scene has visible features; blank walls/floors are poor for ORB matching.
- Check camera intrinsics. Current camera info is synthetic from horizontal FOV.

If forward test stops early:

- Inspect `motion_blocked_reason` or `stop_reason` in the summary JSON.
- If IMU is stale, check `/imu`, `/imu/filtered_yaw`, and the websocket connection in `real_ros_bridge`.

If the robot does not move:

- Confirm robot brain health has `"motion_enabled": true`.
- Watch the robot brain terminal for `/cmd_vel` debug logs.
- Echo `/cmd_vel`:

```bash
source /opt/ros/humble/setup.bash
ros2 topic echo /cmd_vel --once
```
