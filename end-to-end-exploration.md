# End-to-End Autonomous Exploration

This is the exploration-only counterpart to `end-to-end-action-inference.md`. It starts a fresh real-world
mapping run with wheel odometry, OctoMap, Nav2, automatic frontier exploration, the exploration review UI,
and RViz monitoring.

Each launch uses a timestamped session and working-snapshot file. Starting exploration clears only the new
process's live OctoMap; it does not load, delete, or overwrite existing approved maps under
`artifacts/memories/`. Approving the new map saves it under its own timestamped memory ID.

It intentionally does not start HomeTaskAgent, object detection, basket verification, SmolVLA, the React
agent UI, or the temporary relocalization map.

The robot starts exploring immediately when the final command is launched. Keep the physical e-stop within
reach and supervise the run; autonomous exploration is not a reason to leave the hardware unattended.

Start the robot at the location that should become map coordinate `(0, 0, 0)`.

## Mac Terminal RB-1: Robot Brain

```bash
cd /Users/alindumitru/embodied-agents-platform
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
  --initial-camera-pitch-deg 0 \
  --stream-wheel-state \
  --wheel-state-stream-rate-hz 100 \
  --no-stream-imu
```

## Mac Terminal RB-2: Orbbec

Build once if needed:

```bash
cd /Users/alindumitru/embodied-agents-platform

cmake -S tools/orbbec_rgb_test -B build/orbbec_rgb_test -DORBBEC_SDK_ROOT="$HOME/orbbec/sdk"
cmake --build build/orbbec_rgb_test
```

Run for every exploration session:

```bash
cd /Users/alindumitru/embodied-agents-platform

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
  --camera-http-enable \
  --camera-http-host 127.0.0.1 \
  --camera-http-port 8765 \
  --camera-http-path /camera/rgbd \
  --camera-http-timeout-ms 100 \
  --log-every 30
```

## Offload Terminal OC-0: Generate Nav2 Parameters Once

Run this once, and rerun it only after changing the Nav2 configuration.

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
  --robot-length-m 0.3913 \
  --robot-width-m 0.459 \
  --robot-footprint-front-m 0.3613 \
  --robot-footprint-rear-m 0.03 \
  --max-laser-range 4.0 \
  --max-linear-velocity 0.08 \
  --max-angular-velocity 0.30 \
  --min-linear-velocity-threshold 0.005 \
  --trans-stopped-velocity 0.02 \
  --follow-path-xy-goal-tolerance-m 0.08 \
  --local-costmap-width 2 \
  --local-costmap-height 2 \
  --inflation-radius-m 0.05 \
  --inflation-cost-scaling-factor 4.0 \
  --transform-tolerance-s 0.5 \
  --progress-required-movement-radius 0.04 \
  --progress-movement-time-allowance-s 14.0 \
  --xy-goal-tolerance-m 0.09 \
  --min-speed-theta 0.05 \
  --min-angular-velocity-threshold 0.05
```

## Offload Terminal OC-1: ROS Bridge

```bash
cd /home/alin/Robot42
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate
export ROBOT_BRAIN_IP=192.168.1.137
python -m pip install aiohttp

python -m xlerobot_playground.real_ros_bridge \
  --robot-brain-url "http://${ROBOT_BRAIN_IP}:8765" \
  --publish-rate-hz 30 \
  --motion-command-rate-hz 50 \
  --head-points-topic /camera/head/points \
  --head-points-mode settled \
  --head-points-settled-delay-s 0.20 \
  --scan-active-topic /xlerobot/scan_active \
  --no-head-points-update-map-while-base-moving \
  --cmd-vel-timeout-s 0.5 \
  --max-linear-m-s 0.1 \
  --max-angular-rad-s 0.50 \
  --camera-x-m 0.21 \
  --camera-y-m 0.0 \
  --camera-z-m 1.05 \
  --camera-yaw-rad 0.0 \
  --camera-pitch-topic /camera/head/pitch_rad \
  --camera-pan-topic /camera/head/pan_rad \
  --no-laser-fill-no-return \
  --allow-motion-commands \
  --no-publish-imu \
  --odom-source none
```

## Offload Terminal OC-2: Wheel Odometry

```bash
cd /home/alin/Robot42
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate
export ROBOT_BRAIN_IP=192.168.1.137

python -m xlerobot_playground.wheel_odometry \
  --robot-brain-url "http://${ROBOT_BRAIN_IP}:8765" \
  --wheel-state-path /wheel_state \
  --wheel-state-ws-path /ws/wheel_state \
  --wheel-state-transport websocket \
  --odom-topic /odom \
  --odom-reset-topic /xlerobot/odom/set_pose \
  --odom-frame odom \
  --base-frame base_link \
  --publish-rate-hz 100 \
  --http-timeout-s 2.0 \
  --encoder-ticks-per-revolution 4096 \
  --wheel-radius-m 0.0604 \
  --wheel-track-width-m 0.53 \
  --base-link-x-from-wheel-axle-m 0.0 \
  --base-link-y-from-wheel-axle-m 0.0 \
  --left-wheel-motor base_left_wheel \
  --right-wheel-motor base_right_wheel \
  --left-wheel-position-sign -1 \
  --right-wheel-position-sign 1
```

## Offload Terminal OC-3: Map Transform

```bash
cd /home/alin/Robot42
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate

ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 map odom
```

## Offload Terminal OC-4: Exploration OctoMap

This is the normal exploration OctoMap, not the temporary relocalization OctoMap.

```bash
cd /home/alin/Robot42
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate

ros2 launch /home/alin/Robot42/launch/xlerobot_octomap.launch.py
```

## Offload Terminal OC-5: Nav2

```bash
cd /home/alin/Robot42
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate

ros2 launch nav2_bringup navigation_launch.py \
  use_sim_time:=false \
  use_composition:=False \
  params_file:=/home/alin/Robot42/artifacts/nav2/xlerobot_nav2_params.yaml
```

## Offload Terminal OC-6: RViz Monitoring

Start RViz before launching exploration so the initial camera-pan scan is visible.

```bash
cd /home/alin/Robot42
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate

rviz2
```

Set `Fixed Frame` to `map`, then add:

```text
TF
Map /projected_map
Map /global_costmap/costmap
Map /local_costmap/costmap
MarkerArray /occupied_cells_vis_array
PointCloud2 /camera/head/points
LaserScan /scan
Odometry /odom
Path /plan
```

## Offload Terminal OC-7: Start Autonomous Exploration

The command creates a unique session and working-snapshot path, starts immediately, performs the initial
camera-pan scan, selects reachable frontiers with the heuristic policy, navigates with Nav2, scans again after
each arrival, and stops when exploration finishes or reaches 32 decisions.

```bash
cd /home/alin/Robot42
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate
export ROBOT_BRAIN_IP=192.168.1.137
export ROBOT42_EXPLORATION_SESSION="exploration_$(date +%Y%m%d_%H%M%S)"
export ROBOT42_EXPLORATION_SNAPSHOT="/home/alin/Robot42/artifacts/exploration_runs/${ROBOT42_EXPLORATION_SESSION}.json"
mkdir -p /home/alin/Robot42/artifacts/exploration_runs

curl -s -X POST "http://${ROBOT_BRAIN_IP}:8765/camera/head/pitch" \
  -H 'Content-Type: application/json' \
  -d '{"pitch_deg": 30, "settle_s": 0.5}' | python -m json.tool

python -m xlerobot_playground.real_agentic_exploration \
  --persist-path "$ROBOT42_EXPLORATION_SNAPSHOT" \
  --memory-root /home/alin/Robot42/artifacts/memories \
  --session "$ROBOT42_EXPLORATION_SESSION" \
  --no-restore-persisted-state \
  --explorer-policy heuristic \
  --serve-review-ui \
  --review-host 0.0.0.0 \
  --review-port 8770 \
  --no-wait-for-ui-start \
  --no-pause-for-operator-approval \
  --ros-navigation-map-source external \
  --ros-map-topic /projected_map \
  --ros-map-updates-topic /projected_map_updates \
  --ros-map-frame map \
  --ros-scan-topic /scan \
  --ros-point-cloud-topic /camera/head/points \
  --ros-scan-active-topic /xlerobot/scan_active \
  --ros-nav-active-topic /xlerobot/nav_active \
  --ros-local-rotation-active-topic /xlerobot/local_rotation_active \
  --no-relocalization \
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
  --ros-manual-spin-publish-hz 50 \
  --ros-manual-spin-direction-sign 1 \
  --ros-robot-length-m 0.3913 \
  --ros-robot-width-m 0.459 \
  --ros-base-link-x-from-wheel-axle-m 0.0 \
  --ros-base-link-y-from-wheel-axle-m 0.0 \
  --ros-camera-center-forward-m 0.21 \
  --ros-camera-center-lateral-m 0.0 \
  --no-ros-local-rotation-safety-enabled \
  --finish-coverage-threshold 0.96 \
  --max-decisions 32
```

Monitor the exploration review UI at:

```text
http://OFFLOAD_IP:8770
```

The process remains alive after exploration finishes so the map can be inspected. Use `Approve + Save Memory` in
the review UI when the map is ready to become long-term environment memory. Its timestamped ID keeps existing
approved memories untouched. Saved environments appear under:

```text
/home/alin/Robot42/artifacts/memories/<memory_id>/
```

Press `Ctrl-C` in OC-7 only after reviewing or saving the completed map.
