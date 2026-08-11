# End-to-End Action Inference

This setup uses wheel odometry, the saved environment republished on `/projected_map`, and only the temporary
relocalization OctoMap. Do not start the normal OctoMap, IMU filter, or RGB-D odometry.

## Mac Terminal RB-1: Robot Brain And SmolVLA

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
  --no-stream-imu \
  --camera right_wrist=opencv:1 \
  --camera-width 640 \
  --camera-height 480 \
  --camera-fps 30 \
  --vla-policy-path outputs/train/pretrained_vla_batch_16_30k_new_dataset \
  --vla-dataset-root datasets/small-juice-bottle-to-basket-right-arm \
  --vla-dataset-repo-id alindumitru/small-juice-bottle-to-basket-right-arm \
  --vla-device mps \
  --vla-duration-s 60 \
  --vla-action-steps 50 \
  --vla-max-joint-delta 50 \
  --vla-max-gripper-delta 50 \
  --vla-camera-max-age-s 1 \
  --vla-release-open-threshold 30 \
  --vla-release-closed-threshold 10 \
  --vla-release-transition-samples 3 \
  --vla-release-observed-open-samples 2 \
  --vla-release-observed-open-timeout-s 2 \
  --vla-release-settle-s 1 \
  --vla-release-capture-count 4 \
  --vla-release-capture-interval-s 0.25
```

## Mac Terminal RB-2: Orbbec

Build once if needed:

```bash
cd /Users/alindumitru/embodied-agents-platform

cmake -S tools/orbbec_rgb_test -B build/orbbec_rgb_test -DORBBEC_SDK_ROOT="$HOME/orbbec/sdk"
cmake --build build/orbbec_rgb_test
```

Run every time:

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
export ROBOT_BRAIN_IP=192.168.1.133
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
export ROBOT_BRAIN_IP=192.168.1.133

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

## Offload Terminal OC-3: Saved-Map Transform

```bash
cd /home/alin/Robot42
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate

ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 map odom
```

## Offload Terminal OC-4: Relocalization Transform

```bash
cd /home/alin/Robot42
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate

ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 odom relocalization_map
```

## Offload Terminal OC-5: Relocalization OctoMap

```bash
cd /home/alin/Robot42
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate

ros2 launch /home/alin/Robot42/launch/xlerobot_relocalization_octomap.launch.py \
  cloud_topic:=/camera/head/points
```

## Offload Terminal OC-6: Nav2

```bash
cd /home/alin/Robot42
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate

ros2 launch nav2_bringup navigation_launch.py \
  use_sim_time:=false \
  use_composition:=False \
  params_file:=/home/alin/Robot42/artifacts/nav2/xlerobot_nav2_params.yaml
```

## Offload Terminal OC-7: Loaded-Map Navigation Backend

```bash
cd /home/alin/Robot42
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate
export ROBOT_BRAIN_IP=192.168.1.133

curl -s -X POST "http://${ROBOT_BRAIN_IP}:8765/camera/head/pitch" \
  -H 'Content-Type: application/json' \
  -d '{"pitch_deg": 30, "settle_s": 0.5}' | python -m json.tool

python -m xlerobot_playground.real_agentic_exploration \
  --memory-root /home/alin/Robot42/artifacts/memories \
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
  --ros-local-rotation-active-topic /xlerobot/local_rotation_active \
  --relocalization true \
  --ros-relocalization-accept-confidence 0.85 \
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
  --ros-manual-spin-direction-sign 1 \
  --ros-robot-length-m 0.3913 \
  --ros-robot-width-m 0.459 \
  --ros-base-link-x-from-wheel-axle-m 0.0 \
  --ros-base-link-y-from-wheel-axle-m 0.0 \
  --ros-camera-center-forward-m 0.21 \
  --ros-camera-center-lateral-m 0.0 \
  --no-ros-local-rotation-safety-enabled \
  --max-decisions 8
```

## Offload Terminal OC-8: HomeTaskAgent

```bash
cd /home/alin/Robot42
source /home/alin/Robot42/.venv-maniskill/bin/activate

export ROBOT_BRAIN_IP=192.168.1.133
export OPENAI_API_KEY=""
export REPLICATE_API_TOKEN=""
export ROBOT42_AGENT_MODEL=gpt-5.6-terra
export ROBOT42_BASKET_VERIFIER_MODEL=gpt-5.6-terra

python examples/robot42_agent_backend.py \
  --host 0.0.0.0 \
  --port 8765 \
  --memory-root /home/alin/Robot42/artifacts/memories \
  --provider openai \
  --model "$ROBOT42_AGENT_MODEL" \
  --basket-verifier-provider openai \
  --basket-verifier-model "$ROBOT42_BASKET_VERIFIER_MODEL" \
  --basket-verification-manifest /home/alin/Robot42/config/basket_verification/small_cherry_juice_bottle_v0/reference_set.json \
  --basket-verification-minimum-confidence 0.8 \
  --object-detector-provider replicate_grounding_dino \
  --exploration-backend-url http://127.0.0.1:8770 \
  --robot-brain-url "http://${ROBOT_BRAIN_IP}:8765" \
  --vla-handoff \
  --vla-handoff-duration-s 60 \
  --backend-request-timeout-s 180 \
  --max-turns 32 \
  --no-dry-run
```

## Offload Terminal OC-9: React UI

Run `npm install` once if needed:

```bash
cd /home/alin/Robot42/frontend/robot42
npm install
```

```bash
cd /home/alin/Robot42/frontend/robot42
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

In `Configure Environment`:

1. Select the saved environment.
2. Click `Load`.
3. Click `Start Nav Session`.
4. Click `Relocalize`.

Then open the Agent screen and select the same environment memory.

## Offload Terminal OC-10: RViz

```bash
cd /home/alin/Robot42
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate

rviz2
```

Set `Fixed Frame` to `map`. Add:

```text
TF
Map /projected_map
Map /global_costmap/costmap
Map /local_costmap/costmap
Map /relocalization_projected_map
PointCloud2 /camera/head/points
LaserScan /scan
Odometry /odom
```

## Run The Task

In the Agent screen, send:

```text
Go to the kitchen, find the small cherry juice bottle, put it in the robot basket, return to the dock, and tell me when it is ready for handoff.
```

Accept the object-detection confirmation only when the displayed candidate is the correct
cherry juice bottle.

## Software Stop

From the offload PC:

```bash
curl -s -X POST http://127.0.0.1:8765/api/stop \
  -H 'Content-Type: application/json' \
  -d '{}' | python -m json.tool
```
