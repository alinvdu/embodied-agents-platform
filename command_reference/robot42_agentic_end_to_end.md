# Robot42 Agentic End To End

This guide is for running the agentic flow. If a long-term memory already exists, you do **not** need to start exploration.

## Normal Agent Flow

Use this when the environment has already been approved and saved under `artifacts/memories`.

### 1. Start Agent Backend

From the repo root:

For live agent navigation with OpenAI traces:

```bash
cd /home/alin/Robot42
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate

python examples/robot42_agent_backend.py \
  --memory-root ./artifacts/memories \
  --provider openai \
  --model gpt-5.5 \
  --agent-tool-output-mode compact \
  --exploration-backend-url http://127.0.0.1:8770 \
  --navigation-waypoint-horizon-m 2.0 \
  --navigation-auto-rotate-threshold-deg 45 \
  --agent-artifacts-root ./artifacts/agent_runs \
  --max-turns 80
```

For preview-only local testing, omit the provider:

```bash
python examples/robot42_agent_backend.py \
  --memory-root ./artifacts/memories
```

The preview-only command runs `mock/mock`: it resolves the region and shows the waypoint, but it does not call `navigate_to_waypoint`, does not move the robot, and will not show OpenAI Agent traces.

The agent backend automatically discovers environment folders under `artifacts/memories`.

Expected memory layout:

```text
artifacts/memories/<memory_id>/
  manifest.json
  environment_map.json
  home_memory.json
```

If there are multiple memories, the UI lets you select the active one. Without a manual selection, the backend auto-loads the newest available memory.

For one exact memory, you can bypass discovery:

```bash
python examples/robot42_agent_backend.py \
  --home-memory-path ./artifacts/memories/house_v1/home_memory.json
```

### 2. Start Robot42 UI

```bash
cd frontend/robot42
npm install
npm run dev
```

Open the Vite URL, usually:

```text
http://127.0.0.1:5173
```

In the Agent screen:
- check that the environment widget shows configured memory
- select another memory if needed
- send an agent command

Example commands:

```text
go to the kitchen
go to the office
scan the kitchen for a coke can
```

Current behavior:
- the agent reads regions and occupancy from `home_memory.json`
- memory lookup is prompt/context, not a tool call
- region navigation starts with `resolve_navigation_to_region`, which resolves `kitchen` against saved occupancy/free space and returns a final safe pose plus a short-horizon waypoint
- the exposed tools are `resolve_navigation_to_region`, `plan_region_exploration`, `execute_region_exploration_plan`, `navigate_to_waypoint`, `relocalize_here`, `rotate_by`, `rotate_towards_point`, `micro_adjust_to_pose`, `focus_detected_object`, `approach_detected_object`, and `grab_object`
- `navigate_to_waypoint` auto-rotates toward the waypoint before Nav2 when the bearing error is above `--navigation-auto-rotate-threshold-deg`
- if Nav2 fails, `navigate_to_waypoint` can use the direct local-motion fallback only when the saved occupancy map says the short straight-line corridor is footprint-clear
- `execute_region_exploration_plan` plans visual inspection stops for a named region, navigates to each stop, and rotates the robot through the planned 65-degree shot directions
- each executed visual-search shot saves an RGB frame under `artifacts/agent_runs/<run_id>/vision_report/`
- if `--object-detector-provider replicate_grounding_dino` is configured, each shot is sent to Replicate Grounding DINO with the requested `object_label`; the region search stops early when a match is found
- after a match, `focus_detected_object` reuses the selected bbox to recenter the object, `approach_detected_object` first tries to align the robot body perpendicular to the occupied support surface, relocalizes after that alignment, then uses the RGB bbox center for recentering plus RGB-D depth for safe forward steps until the object is around `0.35-0.45m`; a small default tolerance avoids chasing millimeter-scale final steps; `grab_object` returns a mocked VLA entrypoint result
- the React Agent screen shows those frames in the `What Robot Saw` panel

Region polygons are semantic labels, not navigation goals. The agent should not choose a target from the region shape directly. When it needs to move to a region, it calls the region navigation resolver, which searches known-free occupancy cells inside the named region, erodes free space by the robot footprint plus a small gap, and prefers centerline cells with higher clearance from occupied space. The resolver then walks along that same preview path to produce `next_waypoint`, defaulting to a `2.0 m` horizon.

Navigation flow:
- `resolve_navigation_to_region("kitchen")`
- agent reads `next_waypoint`
- `navigate_to_waypoint(waypoint_id, x, y, yaw)`
- `navigate_to_waypoint` may first call a backend local rotation if the waypoint is sideways/behind the robot
- wait for the Nav2 result from the exploration backend; if Nav2 fails and the waypoint is short/direct/clear, the same tool may fall back to bounded local motion
- if Nav2 fails with little/no progress and `local_clearance_recovery.status='succeeded'`, call `micro_adjust_to_pose` to `local_clearance_recovery.recovery_pose`, then `relocalize_here`, then retry the original navigation
- `relocalize_here()`
- resolve again from the updated/corrected pose if the waypoint was not final

Region visual-search flow:
- `execute_region_exploration_plan("kitchen", object_label="coke can")`
- the tool generates red inspection stops and blue shot cones from region shape plus saved occupied/free space
- it uses `navigate_to_waypoint` for each stop, so auto-rotation and direct fallback still apply
- it uses bounded local rotation to face each planned shot
- it saves RGB debug shots plus metadata in the agent vision report
- if object detection is configured, it runs after each shot and returns `detection_status`: `matched`, `not_found`, `not_configured`, `unavailable`, or `failed`
- when `detection_status='matched'`, remaining shots are aborted and `selected_detection` contains the selected bounding box

Object approach flow:
- `focus_detected_object(detection_id, object_label)` first uses the tracked bbox from the successful detection; if needed it rotates once from bbox-center error without re-calling the detector
- `approach_detected_object(detection_id, object_label)` first asks the exploration backend to solve the tracked bbox depth from the latest aligned RGB-D depth image + camera intrinsics
- before close approach, it projects the detected object into the saved occupancy map, infers the nearby occupied support surface, and uses `micro_adjust_to_pose` to move to a perpendicular standoff when the shallow angle would make the robot body scrape a table/shelf/wall
- after that support-surface alignment, it calls `relocalize_here` by default because object search/grab tends to accumulate odometry error from local rotations and micro-motions
- after that alignment, it uses the fresh RGB bbox center for recenter rotations; RGB-D geometry is used for distance and local corridor safety
- after that alignment, it checks a small body corridor and calls `micro_adjust_to_pose` for forward steps; each step aims for 80% of the remaining gap to the staging distance, capped by the configured max step and RGB-D corridor safety
- the approach loop repeats until the object is in the configured staging range, with a small tolerance to avoid chasing millimeter-scale final steps; detector refresh happens after image-centering rotations, after every physical forward step by default, or when tracked-bbox depth fails
- if approach is retried after a support-surface alignment, the retry disables surface realignment by default and takes a fresh detection shot; if the object is still visible but unreachable, report that instead of restarting the full angle-alignment sequence
- `grab_object(object_label, detection_id, object_description)` is mocked for now; it is where the VLA grasp skill will connect

For real-robot approach, the exploration backend needs these ROS topics. The defaults match `real_ros_bridge.py`:

```bash
--ros-rgb-topic /camera/head/image_raw
--ros-depth-topic /camera/head/depth/image_raw
--ros-camera-info-topic /camera/head/camera_info
--ros-rgbd-fallback-horizontal-fov-deg 64.0
```

Object approach requires depth-image grounding. It uses `/camera/head/depth/image_raw` plus camera intrinsics, where intrinsics can come from `/camera/head/camera_info` or from the fallback FOV. The older point-cloud grounding path is deliberately disabled for approach because `/camera/head/points` is unorganized on this robot and cannot be indexed by bbox pixels.

Distance is estimated from the inner bbox window, not the full box edge. The default `bbox_sample_inner_ratio` is `0.65`: Robot42 samples valid depth pixels in that central window, filters invalid/too-near/too-far values, projects them using the intrinsics, then uses the median 3D point as the object estimate.

### Optional Online Object Detection

For the first runnable object-recognition prototype, use Replicate Grounding DINO. This avoids loading a detector beside a future VLA model on the same 16GB GPU.

Set the token:

```bash
export REPLICATE_API_TOKEN="r8_..."
```

Start the agent backend with detection enabled:

```bash
python examples/robot42_agent_backend.py \
  --memory-root ./artifacts/memories \
  --exploration-backend-url http://127.0.0.1:8770 \
  --provider openai \
  --model gpt-5.4 \
  --agent-tool-output-mode compact \
  --navigation-waypoint-horizon-m 2.0 \
  --max-turns 80 \
  --object-detector-provider replicate_grounding_dino \
  --object-detector-api-key "$REPLICATE_API_TOKEN" \
  --object-detector-model adirik/grounding-dino \
  --object-detector-box-threshold 0.25 \
  --object-detector-text-threshold 0.25 \
  --object-detector-min-confidence 0.55 \
  --object-detector-max-image-edge-px 1280 \
  --object-detector-jpeg-quality 85 \
  --object-approach-target-min-m 0.3 \
  --object-approach-target-max-m 0.35 \
  --object-approach-target-tolerance-m 0.025 \
  --object-approach-step-m 0.25 \
  --object-approach-step-fraction 0.8 \
  --object-approach-max-attempts 20
```

Robot42 resizes/re-encodes images sent to Replicate by default. This keeps detector calls small enough for Replicate/Cloudflare while mapping returned boxes back to the original saved RGB shot for focus and RGB-D approach.

Replicate is used for the initial search and occasional verification, not every visual-servo step. The tracked bbox is reused for focus and early approach, and HTTP 429 responses are retried using Replicate's retry delay.

For a no-network smoke test that always returns a centered fake detection:

```bash
--object-detector-provider mock
```

The default pre-Nav2 auto-rotation threshold is `45` degrees. Override it with:

```bash
--navigation-auto-rotate-threshold-deg 60
```

or through the environment:

```bash
export ROBOT42_NAVIGATION_AUTO_ROTATE_THRESHOLD_DEG=60
```

## Configure Or Update Environment

Use this only when you need to create, scan, edit, or approve a map.

```bash
python -m xlerobot_playground.real_agentic_exploration \
  --memory-root ./artifacts/memories
```

Then in Robot42 UI:
- open `Configure Environment`
- create or select an environment
- start/create the map
- scan or manually navigate scan positions
- draw manual walls/barriers
- draw and name regions
- add named places
- click `Set Dock Pose`
- click `Approve + Save Memory`

After approval, the live navigation agent can be run later with:

```bash
python examples/robot42_agent_backend.py \
  --memory-root ./artifacts/memories \
  --provider openai \
  --model gpt-5.5 \
  --agent-tool-output-mode compact \
  --exploration-backend-url http://127.0.0.1:8770 \
  --navigation-waypoint-horizon-m 1.5 \
  --navigation-auto-rotate-threshold-deg 45 \
  --agent-artifacts-root ./artifacts/agent_runs
```

## Load A Saved Environment For Review Or Editing

Use this when long-term memory already exists, but you want to reopen the exploration/configuration UI to inspect or edit the map.

Start the exploration backend with the same memory root:

```bash
python -m xlerobot_playground.real_agentic_exploration \
  --memory-root ./artifacts/memories
```

Then in Robot42 UI:
- open `Configure Environment`
- choose a saved environment from `Environments`
- click `Load`
- optionally click `Start Nav Session` to attach live ROS/Nav2 to the loaded map
- use `Preview` or `Go` to test manual waypoints from the saved dock/start pose

That loads:

```text
artifacts/memories/<memory_id>/environment_map.json
```

You can then adjust regions, walls, named places, or dock pose, and click `Approve + Save Memory` again. You do not need the agent backend running for this review/editing flow.

`Start Nav Session` does not run frontier exploration. It creates a live real ROS/Nav2 session over the loaded `environment_map.json`, publishes the saved map for navigation, applies the saved `start_pose`/dock pose as the initial robot pose, and reuses the existing `Preview` / `Go` waypoint controls.

## Important Separation

- Agent-only flow does **not** require exploration to be running.
- Exploration is only for creating/updating environment memory.
- Loading an environment in the exploration UI uses `environment_map.json`, not the distilled agent-only `home_memory.json`.
- The editable snapshot path is internal/default during normal runs; only override it when debugging storage.
- `--memory-root` is shared by exploration and agent. It stores the saved environment memory folders.
- `examples/xlerobot_exploration_review.py` is only a review-only helper for already-saved maps.
