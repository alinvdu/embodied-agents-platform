# Real XLeRobot Agentic Exploration Commands

This is the standalone command runbook for real exploration. It runs the same frontier/LLM exploration loop as the ManiSkill/ROS path, but with the real robot providing RGB-D, `/camera/head/points`, `/scan`, `/odom`, and motor execution through the robot brain bridge.

This full exploration path does **not** use the HTTP Nav2 router from the smoke tests. In this mode:

```text
exploration loop -> Nav2 navigate_to_pose -> /cmd_vel -> real_ros_bridge -> robot brain -> XLeRobot motors
```

So Nav2 does both path planning and local velocity control. The robot brain executes safe body velocity commands.

Replace these placeholders:

```text
ROBOT_BRAIN_IP   IP address of the Mac/robot brain running robot_brain_agent
OFFLOAD_IP       IP address of the ROS/Nav2 offload computer
YOUR_MODEL       LLM model name, only for LLM mode
YOUR_API_KEY     LLM API key, only for LLM mode
```

On the offload computer, it is convenient to export the robot brain address once:

```bash
export ROBOT_BRAIN_IP=192.168.1.133
```

On any machine where you open the UI from a browser, replace `OFFLOAD_IP` with the offload computer IP.

## What This Runs

```text
robot brain Orbbec sidecar
robot brain HTTP agent
offload real_ros_bridge image/depth/point-cloud publisher
offload RGB-D visual odometry
offload Nav2
offload real_agentic_exploration + web UI
```

The exploration session does:

```text
initial 360 degree scan
OctoMap projection into /projected_map for the first validation run
frontier detection
Nav2 path preview for frontier candidates
LLM or heuristic frontier choice
Nav2 navigate_to_pose for the selected frontier
repeat scan/frontier/decision/navigation
save the final map JSON
```

`/scan` is still published for diagnostics and Nav2 local costmap compatibility, but the default exploration map source in this runbook is now OctoMap's `/projected_map` through `--ros-navigation-map-source external`.

Default scan behavior:

```text
camera pan scan: 0 -> +180 capture, +180 -> 0 return, 0 -> -180 capture, -180 -> 0 return
robot spin scan: fallback only, selected with --ros-turn-scan-mode robot_spin
```

The default initial and arrival scans now rotate `head_motor_1` instead of rotating the robot base. This keeps `/odom` stable during scanning and lets the two outward pan sweeps form one 360 degree scan.

`real_agentic_exploration` now starts from a clean in-memory map by default even when `--persist-path` points to an existing JSON. Pass `--restore-persisted-state` only when you explicitly want to resume the backend/UI snapshot from that file.

## Robot Brain

Run these on the robot brain Mac. Keep both terminals running.

### Terminal RB-1: Robot Brain Agent

Use the Python environment that has LeRobot/XLeRobot installed.

```bash
cd /Users/alin/Robot42
conda activate xlerobot
python -m pip install aiohttp

python -m xlerobot_playground.robot_brain_agent \
  --allow-motion-commands \
  --debug-motion \
  --use-degrees \
  --robot-kind xlerobot_2wheels \
  --port1 /dev/tty.usbmodem5B140330101 \
  --port2 /dev/tty.usbmodem5B140332271 \
  --max-linear-m-s 0.1 \
  --max-angular-rad-s 0.50 \
  --base-angular-action-sign 1 \
  --camera-pan-action-key head_motor_1.pos \
  --camera-pan-action-units deg \
  --camera-pan-action-sign -1 \
  --camera-pan-settle-s 0.5 \
  --initial-camera-pan-deg 0 \
  --camera-pitch-action-key head_motor_2.pos \
  --camera-pitch-action-units deg \
  --camera-pitch-action-sign 1 \
  --camera-pitch-action-offset-deg -28 \
  --camera-pitch-settle-s 0.5 \
  --initial-camera-pitch-deg 0
```

`head_motor_1.pos` is the default horizontal head pan motor command. Keep `--allow-motion-commands` enabled here; camera-pan exploration scans use the same safe hardware command gate as wheel motion.

Keep `--base-angular-action-sign 1` if positive ROS `/cmd_vel.angular.z` turns the robot left/counter-clockwise in RViz/map coordinates. If a positive angular command physically turns the robot right, restart only `robot_brain_agent` with `--base-angular-action-sign -1`.

Keep `--use-degrees` enabled for camera-pan scans. The XLeRobot head motors use degree units only in degree mode; without it, `head_motor_1.pos` is interpreted in normalized `-100..100` units while the scan pipeline would believe the camera reached `-180..180` degrees.

Check the physical pan sign before trusting the 360 map. In ROS convention, positive pan/yaw is left/counter-clockwise viewed from above. If `+30 deg` turns the head right, restart `robot_brain_agent` with `--camera-pan-action-sign -1` so the motor command is inverted while the published camera pose remains correct.

`head_motor_2.pos` is the pitch motor. For mapping, logical `pitch_deg: 0` means the camera optical axis is parallel to the floor, not necessarily raw motor command `0`. On this robot, the physically level camera position currently reads about `head_motor_2.pos = -25`, so `--camera-pitch-action-offset-deg -25` makes logical pitch `0 deg` send raw motor `-25 deg` while ROS TF still publishes pitch `0`.

If you recalibrate and the level camera position changes, update only `--camera-pitch-action-offset-deg`. For example, if the camera is level at `head_motor_2.pos = -18`, use `--camera-pitch-action-offset-deg -18`.

### Terminal RB-2: Orbbec Sidecar

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
  --enable-point-cloud \
  --point-cloud-format xyz \
  --point-cloud-stride 2 \
  --point-cloud-max-points 200000 \
  --point-cloud-min-z-m 0.25 \
  --point-cloud-max-z-m 4.0 \
  --enable-imu \
  --imu-aggregate-mode any \
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

If the Orbbec default aligned depth profile is too heavy, use the listed Gemini 2 `Y16 640x400@30` depth source profile:

```bash
sudo ./build/orbbec_rgb_test/orbbec_rgb_test \
  --frames 0 \
  --no-file-output \
  --enable-depth \
  --enable-depth-registration \
  --enable-point-cloud \
  --point-cloud-format xyz \
  --point-cloud-stride 2 \
  --point-cloud-max-points 200000 \
  --point-cloud-min-z-m 0.25 \
  --point-cloud-max-z-m 4.0 \
  --enable-imu \
  --depth-width 640 \
  --depth-height 400 \
  --depth-fps 30 \
  --imu-aggregate-mode any \
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

Quick health checks from the robot brain:

```bash
curl http://127.0.0.1:8765/health
curl http://127.0.0.1:8765/camera/head/pose
curl http://127.0.0.1:8765/rgb --output /tmp/brain_rgb.ppm
curl http://127.0.0.1:8765/depth --output /tmp/brain_depth.pgm
curl http://127.0.0.1:8765/imu
python - <<'PY'
import asyncio
from aiohttp import ClientSession

async def main():
    async with ClientSession() as session:
        async with session.ws_connect("ws://127.0.0.1:8765/ws/imu") as ws:
            first = await ws.receive()
            print(first.data)

asyncio.run(main())
PY
```

The health response should include point-cloud stats after the first RGB-D frame. If `/camera/head/points` is empty on the offload computer, first confirm this sidecar command includes `--enable-point-cloud` and that the sidecar log reports nonzero point counts.

`/imu` is now an in-memory debug snapshot. The high-rate IMU path is `Orbbec callback -> UDP datagram -> robot_brain_agent memory -> /ws/imu websocket`. `latest_imu.json` is no longer used in the high-rate path.

## Offload Computer

Run these on the ROS/Nav2 offload computer. For the OctoMap first run, keep OC-1, OC-1A, and OC-6 running. Keep OC-2, OC-4, and OC-5 running when you want the full Nav2 exploration stack. OC-3 only needs to be run when generating/updating the Nav2 params.

### Terminal OC-1: Real ROS Bridge

```bash
cd /home/alin/Robot42
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate
python -m pip install aiohttp

python -m xlerobot_playground.real_ros_bridge \
  --robot-brain-url "http://${ROBOT_BRAIN_IP}:8765" \
  --publish-rate-hz 30 \
  --head-points-topic /camera/head/points \
  --head-points-mode settled \
  --head-points-settled-delay-s 0.20 \
  --scan-active-topic /xlerobot/scan_active \
  --no-head-points-update-map-while-base-moving \
  --cmd-vel-timeout-s 0.5 \
  --max-linear-m-s 0.1 \
  --max-angular-rad-s 0.50 \
  --camera-x-m 0.0 \
  --camera-y-m 0.0 \
  --camera-z-m 1.05 \
  --camera-yaw-rad 0.0 \
  --camera-pitch-topic /camera/head/pitch_rad \
  --camera-pan-topic /camera/head/pan_rad \
  --no-laser-fill-no-return \
  --allow-motion-commands
```

This publishes camera images, `/camera/head/points`, depth-derived `/scan`, `/imu`, camera pan/pitch topics, camera transforms, and forwards ROS `/cmd_vel` to the robot brain.

For OctoMap camera-pan scans, keep `--head-points-mode settled` and `--scan-active-topic /xlerobot/scan_active`. Before `Start Explore`, `/camera/head/points` stays live so OctoMap can bootstrap `/projected_map` and TF can provide the first `map -> base_link` pose. After `Start Explore`, the exploration runtime controls `/camera/head/points/update_map_enabled`: it opens PointCloud2 updates only during intentional scan windows and closes them again for waypoint preview/navigation. During scans, the bridge still suppresses `/camera/head/points` while the head is moving and only resumes after robot brain reports the target head pose as settled, so OctoMap integrates stable snapshots instead of smearing point clouds during pan motion.

Keep `--no-head-points-update-map-while-base-moving` for waypoint/Nav2 tests. This temporarily freezes OctoMap input while `/cmd_vel` is moving the base. It prevents base-motion TF/odometry lag from corrupting the already-built map; later live mapping can switch to `--head-points-update-map-while-base-moving` once base pose tracking is validated during motion.

The exploration runtime also pauses `/camera/head/points` for the full duration of each Nav2 waypoint goal through `/camera/head/points/update_map_enabled`. This pauses only the PointCloud2 feed into OctoMap; `/odom`, TF, `/cmd_vel`, images, and IMU stay live so Nav2 and the UI can still track robot motion.

`--camera-z-m 1.05` is the current effective camera height relative to `base_link`, validated in RViz by checking that the PointCloud2 floor remains flat against the ground grid at both `pitch_deg: 0` and `pitch_deg: 30`.

Keep `--no-laser-fill-no-return` for real Orbbec mapping. Missing/invalid depth should stay unknown; treating it as max-range free space creates false fan-shaped clear areas. The point-cloud occupancy mapper is intentionally conservative: it adds free space only along rays to valid points and does not clear through missing depth.

`/imu` is a raw `sensor_msgs/Imu` stream carrying both angular velocity and linear acceleration. In robot-brain mode it is now pushed over a persistent websocket, so `/imu` is no longer capped by the old poll timer.

### Camera Pitch Alignment Check

Before trusting point-cloud occupancy or OctoMap projection, confirm the physical camera pitch matches ROS TF.

In RViz:

- set `Fixed Frame` to `base_link`
- add `TF`
- add `PointCloud2` on `/camera/head/points`
- look from the side

When logical pitch is `0 deg`, floor points should lie roughly parallel to the RViz ground grid. If the floor cloud slopes upward or downward with distance, adjust `--camera-pitch-action-offset-deg` in the robot brain command and restart `robot_brain_agent`.

Useful commands:

```bash
curl -s -X POST "http://${ROBOT_BRAIN_IP}:8765/camera/head/pitch" \
  -H 'Content-Type: application/json' \
  -d '{"pitch_deg": 0, "settle_s": 0.5}' | python -m json.tool

ros2 topic echo /camera/head/pitch_rad --once
ros2 run tf2_ros tf2_echo base_link head_camera_link
```

With `--camera-pitch-action-offset-deg -25`, the first command publishes camera pitch `0 deg` but sends raw `head_motor_2.pos = -25`. This is expected.

Quick checks:

```bash
ros2 topic echo /camera/head/camera_info --once
ros2 topic echo /camera/head/points --once
ros2 topic hz /camera/head/points
ros2 topic echo /scan --once
ros2 topic echo /imu --once
ros2 topic echo /camera/head/pan_rad --once
ros2 topic hz /imu
curl "http://${ROBOT_BRAIN_IP}:8765/health"
```

Migration note:

- Remove any process that depends on `latest_imu.json`. The sidecar no longer writes it.
- Keep `/cmd_vel` and `/stop` on HTTP. IMU now comes from `ws://ROBOT_BRAIN_IP:8765/ws/imu`.
- If the brain and sidecar run on different hosts, change both `--imu-udp-host` and `--imu-udp-port` together.

Verification plan:

- On the robot brain, watch `orbbec_rgb_test` for `IMU callback rate ~= ... Hz`.
- On the robot brain, watch `robot_brain_agent` for `IMU rx rate~=...Hz` and `IMU ws send rate~=...Hz`.
- On the offload computer, watch `real_ros_bridge` for `IMU websocket connected` and `IMU stream rx~=...Hz publish~=...Hz`.
- On the offload computer, watch `real_ros_bridge` for point-cloud receive/publish logs, then confirm `ros2 topic hz /camera/head/points`.
- Run `ros2 topic hz /imu` and confirm the rate is no longer limited to the old tens-of-Hz poll ceiling.
- If the callback rate is still low, audit the current Orbbec aggregate mode `OB_FRAME_AGGREGATE_OUTPUT_ALL_TYPE_FRAME_REQUIRE` first. The transport bottleneck is removed, but that SDK aggregation setting can still quantize the upstream callback cadence.

### Terminal OC-1A: OctoMap From Orbbec Point Cloud

Use this terminal for the OctoMap first run. Keep `robot_brain_agent`, the Orbbec sidecar, and `real_ros_bridge` running first.

```bash
cd /home/alin/Robot42
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate

ros2 launch /home/alin/Robot42/launch/xlerobot_octomap.launch.py
```

The default `config/xlerobot_octomap.yaml` is set up for Nav2 waypoint validation:

```text
frame_id: map
base_frame_id: base_link
resolution: 0.08
```

Before starting OctoMap, TF must provide:

```text
map -> odom -> base_link -> head_camera_link
```

For the first validation pass, if visual odometry already publishes `odom -> base_link`, bootstrap the map frame with:

```bash
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 map odom
```

Expected topics:

```bash
ros2 topic list | grep -E 'octomap|projected|occupied'
ros2 topic echo /projected_map --once
ros2 topic echo /projected_map_updates --once
ros2 topic hz /projected_map
```

Expected output topics include:

```text
/occupied_cells_vis_array
/octomap_binary
/octomap_full
/octomap_point_cloud_centers
/projected_map
/projected_map_updates
```

In RViz:

- set `Fixed Frame` to `map` for Nav2 validation
- add `TF`
- add `PointCloud2` on `/camera/head/points`
- add `MarkerArray` on `/occupied_cells_vis_array`
- add `Map` on `/projected_map` with update topic `/projected_map_updates`

Do not tune OctoMap until the camera pitch alignment check above passes. If the floor cloud is tilted relative to the RViz grid, `/projected_map` will appear cut or will mark free/occupied cells in the wrong places.

Restarting OctoMap clears its in-memory map. RViz does not need to be restarted; toggle the `Map` and `MarkerArray` displays off/on if stale latched visuals remain.

### Terminal OC-2: IMU Yaw Filter

```bash
cd /home/alin/Robot42
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate

python -m xlerobot_playground.imu_yaw_filter \
  --imu-topic /imu \
  --output-topic /imu/filtered_yaw \
  --camera-pitch-topic /camera/head/pitch_rad \
  --input-frame-convention camera_optical \
  --yaw-source gyro_y \
  --compensate-camera-pitch \
  --bias-calibration-s 0.5 \
  --yaw-rate-lowpass-alpha 0.2
```

This matches the OrbbecViewer CSV path: integrate corrected `gyro_y`.
Because the IMU is on the pitching camera head, `--compensate-camera-pitch` divides the selected yaw-rate axis by `cos(camera_pitch)` using `/camera/head/pitch_rad`; this removes the large yaw underreport when the camera is pitched down for mapping.
Keep the robot still for the first 0.5 seconds after startup so the yaw filter calibrates gyro bias cleanly.

Quick checks:

```bash
ros2 topic echo /imu/filtered_yaw --once
```

### Terminal OC-3: RGB-D Visual Odometry

```bash
cd /home/alin/Robot42
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate

python -m xlerobot_playground.rgbd_visual_odometry \
  --rgb-topic /camera/head/image_raw \
  --depth-topic /camera/head/depth/image_raw \
  --camera-info-topic /camera/head/camera_info \
  --imu-topic /imu/filtered_yaw \
  --odom-topic /odom \
  --camera-pan-topic /camera/head/pan_rad \
  --scan-active-topic /xlerobot/scan_active \
  --nav-active-topic /xlerobot/nav_active \
  --scan-active-stale-timeout-s 10.0 \
  --freeze-orientation-during-scan \
  --freeze-odom-during-scan \
  --odom-requires-nav-active \
  --imu-frame-convention base_link \
  --odom-yaw-sign -1 \
  --imu-bias-calibration-s 0.0 \
  --publish-rate-hz 30 \
  --min-translation-update-m 0.001 \
  --jitter-threshold \
  --min-yaw-update-deg 0.15
```

This consumes RGB-D for translation and the filtered yaw IMU topic for authoritative yaw. Accelerometer double integration is not used for odometry position. With `--odom-requires-nav-active`, pose updates are accepted only while the exploration runtime publishes `/xlerobot/nav_active=true` around a Nav2 waypoint goal; when the robot is sitting, previewing, or scanning, odom keeps publishing the held pose and refreshes the VO keyframe without accumulating stationary drift. `--no-odom-requires-nav-active` is useful only for direct teleop/debug runs that do not publish `/xlerobot/nav_active`.

Keep `--freeze-odom-during-scan` enabled for stationary camera-pan mapping. `/xlerobot/scan_active=true` freezes odom yaw and RGB-D translation even if nav gating is disabled, while still refreshing the VO keyframe so scan-time camera motion is not accumulated into one post-scan jump. Head pan position alone no longer freezes odom. The older `--freeze-during-head-motion` option remains as a compatibility alias for yaw freezing, but the scan-active event is now the actual trigger. The exploration runtime publishes `/xlerobot/scan_active` before camera-pan or robot-spin scanning starts, refreshes it while scanning, and publishes `false` after the configured release delay. `--scan-active-stale-timeout-s 10.0` is a failsafe: if the odom node misses the final `false` and no scan-active refresh arrives for 10 seconds, it resumes scan freeze; `/xlerobot/nav_active` still controls whether normal motion updates are allowed.

Quick checks:

```bash
ros2 topic echo /odom --once
ros2 run tf2_ros tf2_echo odom base_link
```

### Terminal OC-4: Nav2 Params

Generate the conservative Nav2 params once:

```bash
cd /home/alin/Robot42
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate

python -m xlerobot_playground.real_nav2_config \
  --base-nav2-params /opt/ros/humble/share/nav2_bringup/params/nav2_params.yaml \
  --output-dir /home/alin/Robot42/artifacts/nav2 \
  --scan-topic /scan \
  --global-map-topic /projected_map \
  --map-frame map \
  --odom-frame odom \
  --base-frame base_link \
  --max-laser-range 4.0 \
  --max-linear-velocity 0.1 \
  --max-angular-velocity 0.40 \
  --min-linear-velocity-threshold 0.005 \
  --trans-stopped-velocity 0.02 \
  --follow-path-xy-goal-tolerance-m 0.08 \
  --local-costmap-width 2 \
  --local-costmap-height 2 \
  --inflation-radius-m 0.16 \
  --inflation-cost-scaling-factor 4.0 \
  --transform-tolerance-s 0.5 \
  --progress-required-movement-radius 0.04 \
  --progress-movement-time-allowance-s 14.0 \
  --xy-goal-tolerance-m 0.09 \
  --min-speed-theta 0.05 \
  --min-angular-velocity-threshold 0.05
```

The minimum velocity settings stop DWB from choosing tiny near-zero commands that the real base cannot execute. With `max_angular_velocity=0.18`, the old `min_speed_theta=0.0` allowed the first nonzero angular sample to be `0.00947 rad/s`, which shows up as `theta.vel=0.5428 deg/s` on the robot brain and can make Nav2 crawl at the goal. `FollowPath.xy_goal_tolerance` is kept below the `PositionGoalChecker` XY tolerance so DWB does not switch into near-goal behavior before Nav2 can mark the waypoint reached. The generated controller uses Nav2's `PositionGoalChecker`, so exploration waypoints complete on XY position and ignore final yaw; `RotateToGoal` is disabled for the same reason.

### Terminal OC-5: Nav2

```bash
cd /home/alin/Robot42
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate

ros2 launch nav2_bringup navigation_launch.py \
  use_sim_time:=false \
  use_composition:=False \
  params_file:=/home/alin/Robot42/artifacts/nav2/xlerobot_nav2_params.yaml
```

Quick checks:

```bash
ros2 action list
ros2 action list | grep compute_path_to_pose
ros2 action list | grep navigate_to_pose
```

### Terminal OC-6: Real Exploration UI And Loop

Start with heuristic policy first. This OctoMap first run tests the 360 degree camera-pan scan, `/projected_map` ingestion, frontier generation, and UI display without spending LLM calls or starting navigation.

Set the camera to the validated single pitch before starting the UI loop:

```bash
curl -s -X POST "http://${ROBOT_BRAIN_IP}:8765/camera/head/pitch" \
  -H 'Content-Type: application/json' \
  -d '{"pitch_deg": 30, "settle_s": 0.5}' | python -m json.tool
```

```bash
cd /home/alin/Robot42
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate

python -m xlerobot_playground.real_agentic_exploration \
  --persist-path /home/alin/Robot42/artifacts/real_xlerobot_exploration_map.json \
  --session real_house_v1 \
  --explorer-policy heuristic \
  --serve-review-ui \
  --review-host 0.0.0.0 \
  --review-port 8770 \
  --ros-navigation-map-source external \
  --ros-map-topic /projected_map \
  --ros-map-updates-topic /projected_map_updates \
  --ros-map-frame map \
  --ros-scan-topic /scan \
  --ros-point-cloud-topic /camera/head/points \
  --ros-scan-active-topic /xlerobot/scan_active \
  --ros-nav-active-topic /xlerobot/nav_active \
  --ros-scan-active-release-delay-s 3.0 \
  --ros-ready-timeout-s 30 \
  --ros-turn-scan-timeout-s 75 \
  --ros-turn-scan-mode camera_pan \
  --robot-brain-url "http://${ROBOT_BRAIN_IP}:8765" \
  --camera-pan-action-key head_motor_1.pos \
  --camera-pan-settle-s 1.2 \
  --camera-pan-step-deg 60 \
  --camera-pan-compute-s 1.5 \
  --ros-manual-spin-angular-speed-rad-s 0.30 \
  --max-decisions 8 \
  --ros-imu-topic /imu/filtered_yaw \
  --pause-for-operator-approval
```

Open the UI from your browser:

```text
http://OFFLOAD_IP:8770
```

Click `Start Explore` in the UI. The robot should keep its base still, keep pitch at `30 deg`, pan the head in 60 degree stops (`0 -> 60 -> 120 -> 180 -> 0 -> -60 -> -120`), wait `1.5s` for the motor at each stop, wait for a fresh `/camera/head/points` sample, then give OctoMap `2.0s` of compute time before the next pan move. It should then show `/projected_map` plus `/projected_map_updates` as the occupancy map in the UI. Because `--pause-for-operator-approval` is enabled, it should pause after the initial scan while keeping the live ROS session available for waypoint testing.

By default this real-exploration command waits for the UI start request before moving the robot or panning the head. Use `--no-wait-for-ui-start` only when you want the 360 degree camera-pan scan to begin immediately after the terminal command starts.

To test map coordinate accuracy without moving, click `Preview Path` in the Map Editing panel, then click a known free-space location in the map. The UI calls Nav2 `ComputePathToPose` and draws the planned path in orange. If the orange path is strange for a waypoint directly in front of the robot, debug Nav2 planning/costmaps before commanding motion.

To actually move, click `Waypoint`, then click the free-space location. The UI sends that map-frame pose to Nav2. Use this only after the initial scan has completed, the preview path looks sane, and the robot is physically clear to move.

For this first OctoMap validation run, keep `--ros-navigation-map-source external`, `--ros-map-topic /projected_map`, and `--ros-map-updates-topic /projected_map_updates`. Do not use `fused_point_cloud`; that path uses the custom Python point-cloud fusion instead of OctoMap. Also keep the extra projected-map snapshot fusion disabled, which is the default. Only add `--ros-fuse-external-projected-map-snapshots` if you intentionally want the exploration runtime to fuse multiple `/projected_map` snapshots itself.

Robot-spin fallback:

```bash
python -m xlerobot_playground.real_agentic_exploration ... \
  --ros-turn-scan-mode robot_spin
```

Use this only if the head pan motor path is unavailable. In fallback mode the 360 scan uses `/cmd_vel` and rotates the robot base.

Once heuristic exploration is sane, switch to LLM policy:

```bash
python -m xlerobot_playground.real_agentic_exploration \
  --persist-path /home/alin/Robot42/artifacts/real_xlerobot_exploration_map.json \
  --session real_house_v1 \
  --explorer-policy llm \
  --llm-provider openai \
  --llm-model YOUR_MODEL \
  --llm-api-key YOUR_API_KEY \
  --serve-review-ui \
  --review-host 0.0.0.0 \
  --review-port 8770 \
  --ros-navigation-map-source external \
  --ros-map-topic /projected_map \
  --ros-map-updates-topic /projected_map_updates \
  --ros-map-frame map \
  --ros-scan-topic /scan \
  --ros-point-cloud-topic /camera/head/points \
  --ros-scan-active-topic /xlerobot/scan_active \
  --ros-nav-active-topic /xlerobot/nav_active \
  --ros-scan-active-release-delay-s 3.0 \
  --ros-ready-timeout-s 30 \
  --ros-turn-scan-timeout-s 75 \
  --ros-turn-scan-mode camera_pan \
  --robot-brain-url "http://${ROBOT_BRAIN_IP}:8765" \
  --camera-pan-action-key head_motor_1.pos \
  --camera-pan-settle-s 0.25 \
  --camera-pan-step-deg 60 \
  --camera-pan-compute-s 0.8 \
  --ros-manual-spin-angular-speed-rad-s 0.30 \
  --max-decisions 8
```

## Preflight Checklist

Run these on the offload computer before starting exploration:

```bash
ros2 topic echo /camera/head/camera_info --once
ros2 topic echo /camera/head/image_raw --once
ros2 topic echo /camera/head/depth/image_raw --once
ros2 topic echo /camera/head/points --once
ros2 topic hz /camera/head/points
ros2 topic echo /camera/head/pan_rad --once
ros2 topic echo /projected_map --once
ros2 topic echo /projected_map_updates --once
ros2 topic hz /projected_map
ros2 topic echo /scan --once
ros2 run tf2_ros tf2_echo base_link head_camera_link
ros2 action list
```

Expected action servers include:

```text
/compute_path_to_pose
/navigate_to_pose
```

After `real_agentic_exploration` starts and the first scan begins, verify the OctoMap projected occupancy map:

```bash
ros2 topic echo /projected_map --once
ros2 topic echo /projected_map_updates --once
ros2 topic hz /projected_map
```

During the initial scan, the robot base should stay still while the head pans through the positive sweep first, returns to center, pans through the negative sweep second, and returns to center. The UI should move from an empty/not-started map to the OctoMap `/projected_map` occupancy view with candidate frontiers. `/scan` remains available for Nav2 local obstacle checks and debugging, but the UI map for this run comes directly from `/projected_map` plus `/projected_map_updates`, without extra runtime snapshot fusion by default.

If RViz shows the map rotating with the camera, or only the last camera direction appears to stick, first check the map frame:

```bash
ros2 topic echo /projected_map --once | grep -E 'frame_id|width|height|resolution'
```

For Nav2 waypoint tests, `/projected_map` must say `frame_id: map`. If it says `head_camera_link`, OctoMap is accumulating in the moving camera frame instead of the fixed world frame, so each pan angle overwrites the useful interpretation of the previous one. If it says `base_link`, the UI can display the stationary scan, but Nav2's global costmap will not consume it correctly as a global map. Restart OctoMap with `config/xlerobot_octomap.yaml` loaded and `frame_id: map`.

Optional RViz validation:

```bash
rviz2
```

For the OctoMap/Nav2 run in this document, set `Fixed Frame` to `map`, then add `PointCloud2` on `/camera/head/points`, `Map` on `/projected_map` with update topic `/projected_map_updates`, `Map` on `/global_costmap/costmap`, `Map` on `/local_costmap/costmap`, `MarkerArray` on `/occupied_cells_vis_array`, and `TF`. You should see the point cloud transform through `head_camera_link` during head pan sweeps, while `/projected_map` grows from OctoMap's accumulated 3D evidence and `/global_costmap/costmap` receives that projected map through Nav2's static layer.

If the map starts but the head does not pan, check the robot-brain head pose and motion gate:

```bash
curl "http://${ROBOT_BRAIN_IP}:8765/health"
curl "http://${ROBOT_BRAIN_IP}:8765/camera/head/pose"
curl -X POST "http://${ROBOT_BRAIN_IP}:8765/camera/head/pan" \
  -H 'Content-Type: application/json' \
  -d '{"pan_deg": 30, "action_key": "head_motor_1.pos", "settle_s": 0.5}'
curl -X POST "http://${ROBOT_BRAIN_IP}:8765/camera/head/pan" \
  -H 'Content-Type: application/json' \
  -d '{"pan_deg": 0, "action_key": "head_motor_1.pos", "settle_s": 0.5}'
curl -X POST "http://${ROBOT_BRAIN_IP}:8765/camera/head/pitch" \
  -H 'Content-Type: application/json' \
  -d '{"pitch_deg": 0, "settle_s": 0.5}'
curl -X POST "http://${ROBOT_BRAIN_IP}:8765/camera/head/pitch" \
  -H 'Content-Type: application/json' \
  -d '{"pitch_deg": 30, "settle_s": 0.5}'
```

The `robot_brain_agent` terminal should log motion/action errors if it rejects a pan or pitch command. In `robot_spin` fallback mode, the `real_ros_bridge` terminal should log motion forwarding errors if the robot brain rejects `/cmd_vel`.

## What You Should See

- Robot brain logs showing RGB-D/IMU receive rates and pan commands during 360 scans.
- `real_ros_bridge` publishing `/camera/head/*`, `/camera/head/points`, `/scan`, camera pan/pitch topics, and forwarding `/cmd_vel` during Nav2 navigation.
- RGB-D visual odometry publishing `/odom` and `odom -> base_link`.
- Nav2 accepting `compute_path_to_pose` and `navigate_to_pose`.
- `/projected_map` and `/projected_map_updates` receiving the OctoMap 2D projection and the UI using it as the occupancy map.
- UI showing:
  - current robot pose
  - partial occupancy map
  - frontier candidates
  - recent RGB keyframes
  - selected frontier and Nav2 path/result
- Final map saved to:

```text
/home/alin/Robot42/artifacts/real_xlerobot_exploration_map.json
```

## Current Limitations

- RGB-D visual odometry is experimental and may drift, especially if RGB-D alignment, intrinsics, or feature texture are poor.
- The default initial and arrival 360 scans use camera pan through robot brain `head_motor_1.pos`; robot base rotation is fallback only with `--ros-turn-scan-mode robot_spin`.
- Frontier navigation uses Nav2 `navigate_to_pose`; this is the existing ROS exploration execution path.
- Exact region naming and semantic waypoint quality depend on good RGB keyframes and LLM/VLM configuration.
- Saving the final map JSON works through `--persist-path`; saving directly into persistent robot memory is still a separate integration step.
