# XLeRobot Command Reference

Run these commands from `/home/alin/Robot42` unless noted otherwise.

## Setup and Sanity Checks

Create or update the local ManiSkill environment:

```bash
PYTHON_BIN=/home/alin/lero_env/bin/python ./scripts/setup_xlerobot_maniskill_env.sh
```

What it offers:
- Creates `.venv-maniskill`
- Installs ManiSkill, SAPIEN, Rerun, pygame, OpenCV, and tyro
- Downloads ReplicaCAD assets by default

Check that the local XLeRobot fork is wired into ManiSkill correctly:

```bash
./scripts/run_xlerobot_maniskill_play.sh --check --repo-root /home/alin/XLeRobot
```

What it offers:
- Validates the XLeRobot fork path
- Confirms `xlerobot` and `SceneManipulation-v1` are registered
- Does not launch the simulator

Launch the legacy direct sim wrapper:

```bash
./scripts/run_xlerobot_maniskill_play.sh --repo-root /home/alin/XLeRobot --demo ee_keyboard
```

What it offers:
- Direct access to the local ManiSkill demos without the new playground wrappers

## Manipulation Playground

Sim keyboard teleop with SAPIEN + Rerun cameras:

```bash
python examples/xlerobot_manipulation_playground.py \
  --backend sim \
  --controller keyboard \
  --speed-profile normal \
  --sim-python-bin /home/alin/Robot42/.venv-maniskill/bin/python \
  --repo-root /home/alin/XLeRobot
```

What it offers:
- 3D SAPIEN scene
- Rerun camera visualization for `fetch_head` RGB-D plus `fetch_left_arm_camera` / `fetch_right_arm_camera` RGB
- Keyboard teleop

Same, but tuned faster to feel closer to the real robot:

```bash
python examples/xlerobot_manipulation_playground.py \
  --backend sim \
  --controller keyboard \
  --speed-profile fast \
  --sim-python-bin /home/alin/Robot42/.venv-maniskill/bin/python \
  --repo-root /home/alin/XLeRobot
```

What it offers:
- Faster base, arm, wrist, and head teleop increments in sim

Sim VR teleop:

```bash
python examples/xlerobot_manipulation_playground.py \
  --backend sim \
  --controller vr \
  --sim-python-bin /home/alin/Robot42/.venv-maniskill/bin/python \
  --repo-root /home/alin/XLeRobot
```

What it offers:
- VR teleoperation in sim
- SAPIEN scene rendering

Real robot keyboard teleop with three cameras streamed to Rerun:

```bash
python examples/xlerobot_manipulation_playground.py \
  --backend real \
  --controller keyboard \
  --runtime-python-bin /home/alin/lero_env/bin/python \
  --repo-root /home/alin/XLeRobot \
  --camera head=realsense:SERIAL \
  --camera left_wrist=opencv:/dev/video0 \
  --camera right_wrist=opencv:/dev/video2
```

What it offers:
- Real robot teleop
- Keyboard control
- Live camera streaming to Rerun when cameras are configured

Real robot VR teleop:

```bash
python examples/xlerobot_manipulation_playground.py \
  --backend real \
  --controller vr \
  --runtime-python-bin /home/alin/lero_env/bin/python \
  --repo-root /home/alin/XLeRobot \
  --camera head=realsense:SERIAL \
  --camera left_wrist=opencv:/dev/video0 \
  --camera right_wrist=opencv:/dev/video2
```

What it offers:
- Real robot VR control
- Live camera streaming to Rerun when cameras are configured

## Training Playground

Sim keyboard recording to a local dataset:

```bash
python examples/xlerobot_training_playground.py \
  --backend sim \
  --controller keyboard \
  --speed-profile fast \
  --sim-python-bin /home/alin/Robot42/.venv-maniskill/bin/python \
  --repo-root /home/alin/XLeRobot \
  --dataset-name xlerobot_sim_playground \
  --dataset-root ./datasets \
  --task "dual-arm teleop practice"
```

What it offers:
- Sim teleop
- Local dataset recording
- Same scene/camera stack as the manipulation playground

Sim recording hotkeys:
- `SPACE`: start or stop recording
- `R`: close the current episode and start a new one
- `ESC`: save and quit

Real robot keyboard recording to LeRobot format:

```bash
python examples/xlerobot_training_playground.py \
  --backend real \
  --controller keyboard \
  --runtime-python-bin /home/alin/lero_env/bin/python \
  --repo-root /home/alin/XLeRobot \
  --dataset-id local/xlerobot_real_playground \
  --dataset-root ./datasets \
  --task "dual-arm teleop practice" \
  --camera head=realsense:SERIAL \
  --camera left_wrist=opencv:/dev/video0 \
  --camera right_wrist=opencv:/dev/video2
```

What it offers:
- Real robot teleop
- Local LeRobot dataset recording
- Rerun camera visualization when cameras are configured

Real robot VR recording:

```bash
python examples/xlerobot_training_playground.py \
  --backend real \
  --controller vr \
  --runtime-python-bin /home/alin/lero_env/bin/python \
  --repo-root /home/alin/XLeRobot \
  --dataset-id local/xlerobot_real_vr_playground \
  --dataset-root ./datasets \
  --task "dual-arm vr practice" \
  --camera head=realsense:SERIAL \
  --camera left_wrist=opencv:/dev/video0 \
  --camera right_wrist=opencv:/dev/video2
```

What it offers:
- Real robot VR teleop
- Local LeRobot dataset recording with VR-triggered episode control

Real keyboard recording hotkeys:
- `[`: start recording
- `]`: stop recording and save the current episode
- `\\`: quit

Real VR recording controls:
- Left thumbstick right: start recording, or stop and save the current episode
- Left thumbstick left: discard the current episode buffer
- Left thumbstick up: save the active episode and quit the recording session
- Left thumbstick down: reset the robot pose
- `\\`: optional keyboard quit fallback

## Inference Playground

Basic typed instruction run:

```bash
python examples/xlerobot_inference_playground.py \
  --instruction "go to the kitchen and open the fridge" \
  --visible-object fridge \
  --observation fridge_handle_visible
```

What it offers:
- Runs the current agent runtime end-to-end
- Chooses navigation and manipulation skills from the mock planner/runtime stack

Same, but emit JSON:

```bash
python examples/xlerobot_inference_playground.py \
  --instruction "go to the kitchen and open the fridge" \
  --visible-object fridge \
  --observation fridge_handle_visible \
  --json
```

What it offers:
- Machine-readable record of subgoals, selected skills, reasoning, and execution status

Try the delegated navigation backend explicitly:

```bash
python examples/xlerobot_inference_playground.py \
  --instruction "go to the kitchen and open the fridge" \
  --navigation-mode delegated_navigation_module \
  --delegated-backend global_map \
  --visible-landmark kitchen \
  --observation fridge_handle_visible
```

Try the VLA navigation skill mode explicitly:

```bash
python examples/xlerobot_inference_playground.py \
  --instruction "find the bread and grab it" \
  --navigation-mode vla_navigation_skills \
  --visible-object bread \
  --observation bread_visible
```

Try the voice-transcript path:

```bash
python examples/xlerobot_inference_playground.py \
  --voice-transcript "hey xlerobot go to the kitchen and open the fridge"
```

What it offers:
- Exercises the mock wake-word and translation path

## Agent Playground

Launch the new live agent playground UI:

```bash
python examples/xlerobot_agent_playground.py \
  --serve-ui \
  --backend sim \
  --provider mock \
  --ui-port 8765
```

Then open `http://127.0.0.1:8765`.

What it offers:
- Local web UI with real-time plan, action, review, and reflection events
- Pause, resume, and stop controls
- Pluggable model routing for planner / critic / coder roles
- Skill-first execution with simple navigation, mapping, and perception tools plus bounded code execution fallback

Run once without the UI and print the full machine-readable session state:

```bash
python examples/xlerobot_agent_playground.py \
  --instruction "go to the kitchen and open the fridge" \
  --backend sim \
  --provider mock
```

Use an OpenAI-compatible endpoint instead of the mock router:

```bash
python examples/xlerobot_agent_playground.py \
  --serve-ui \
  --backend sim \
  --provider openai-compatible \
  --model moonshotai/kimi-k2-instruct \
  --base-url http://127.0.0.1:8000/v1/chat/completions \
  --api-key "$OPENAI_API_KEY" \
  --thinking \
  --reasoning-effort medium
```

Load additional trained skills from a catalog JSON file:

```bash
python examples/xlerobot_agent_playground.py \
  --serve-ui \
  --backend sim \
  --skill-catalog ./skills/catalog.json
```

Use the playground with a remote Ubuntu offload server:

```bash
python examples/xlerobot_agent_playground.py \
  --serve-ui \
  --backend sim \
  --provider mock \
  --offload-server-url http://192.168.1.10:8890 \
  --brain-name xlerobot-macos-brain
```

What it offers:
- Registers the current brain with the Ubuntu offload node
- Publishes world-state updates during planning and execution
- Offloads `go_to_pose`, `get_map`, `explore`, `create_map`, `perceive_scene`, `ground_object_3d`, `set_waypoint_from_object`, and skill execution through the remote API

Without `--offload-server-url`, the same playground keeps everything local, which is the intended path for single-machine Ubuntu sim runs.

## Exploration Playground

Run a live ManiSkill/XLeRobot exploration pass that actually moves the robot in sim, builds the map from the head RGB-D stream, saves the result, and optionally opens the review UI afterward:

```bash
python examples/xlerobot_exploration_playground.py \
  --repo-root /home/alin/XLeRobot \
  --persist-path ./artifacts/xlerobot_exploration_map.json \
  --area workspace \
  --session house_v1 \
  --render-mode human \
  --serve-review-ui
```

For a headless smoke run:

```bash
python examples/xlerobot_exploration_playground.py \
  --repo-root /home/alin/XLeRobot \
  --persist-path ./artifacts/xlerobot_exploration_map.json \
  --area workspace \
  --session house_v1 \
  --render-mode rgb_array
```

What it offers:
- Boots the real `SceneManipulation-v1` sim under the ManiSkill interpreter
- Uses the live `fetch_head` RGB-D stream and point cloud positions to build a 2D occupancy map
- Drives the XLeRobot base toward frontiers until exploration stalls
- Captures RGB keyframes and writes a reviewable map snapshot for later correction

## Robot42 Real Exploration End To End

This is the normal Robot42 environment-configuration flow for the real robot. Use it when you need to create, scan, load, edit, or approve an environment map. It starts the real agentic exploration backend, serves the environment/configuration backend on port `8770`, and saves approved long-term memories under `artifacts/memories`.

```bash
python -m xlerobot_playground.real_agentic_exploration \
  --memory-root ./artifacts/memories
```

See the full agent flow in [Robot42 Agentic End To End](./robot42_agentic_end_to_end.md).

Then run the React UI:

```bash
cd frontend/robot42
npm install
npm run dev
```

And run the Robot42 agent backend:

```bash
python examples/robot42_agent_backend.py \
  --memory-root ./artifacts/memories \
  --provider openai \
  --model gpt-5.5 \
  --exploration-backend-url http://127.0.0.1:8770 \
  --navigation-waypoint-horizon-m 2.0 \
  --navigation-auto-rotate-threshold-deg 45 \
  --agent-artifacts-root ./artifacts/agent_runs
```

What gets saved after `Approve + Save Memory`:

```text
artifacts/memories/<memory_id>/
  manifest.json
  environment_map.json
  home_memory.json
```

What gets saved after an agent visual-search run:

```text
artifacts/agent_runs/<run_id>/vision_report/
  manifest.json
  shots/
    <region>_<stop>_<shot>.png
    <region>_<stop>_<shot>.json
```

Notes:
- `python -m xlerobot_playground.real_agentic_exploration` is the normal real robot mapping/configuration entrypoint.
- You do not need this command for agent-only runs when a memory already exists.
- To reopen an existing saved environment, start this backend, open `Configure Environment`, choose a memory, and click `Load`.
- The editable snapshot path is internal/default during normal runs; only override it when debugging storage.
- `--memory-root` is the folder of saved long-term environments.
- Agent visual-search commands such as `scan the kitchen for a coke can` save RGB debug shots and show them in the React Agent UI under `What Robot Saw`.
- `examples/xlerobot_exploration_review.py` is only a review-only helper for already-saved maps; it is not the normal robot exploration entrypoint.

## Exploration Review

Run the review UI on a saved local map snapshot:

```bash
python examples/xlerobot_exploration_review.py \
  --mode local \
  --host 127.0.0.1 \
  --port 8770 \
  --persist-path ./artifacts/xlerobot_exploration_map.json
```

Then open `http://127.0.0.1:8770`.

What it offers:
- Post-run visualization with occupancy, trajectory, semantic regions, keyframes, and named places
- Region relabeling, polygon edits, merge/split, waypoint edits, and map approval

Use the same UI against the remote offload server:

```bash
python examples/xlerobot_exploration_review.py \
  --mode offload \
  --offload-server-url http://192.168.1.10:8890 \
  --brain-name xlerobot-macos-brain
```

## Distributed Offload

Run the Ubuntu offload server:

```bash
python examples/xlerobot_offload_server.py \
  --host 0.0.0.0 \
  --port 8890
```

Optional upstream service routing:

```bash
python examples/xlerobot_offload_server.py \
  --host 0.0.0.0 \
  --port 8890 \
  --nav2-service-url http://127.0.0.1:9001 \
  --perception-service-url http://127.0.0.1:9002 \
  --vla-service-url http://127.0.0.1:9003
```

Run the macOS brain bridge service:

```bash
python examples/xlerobot_brain_service.py \
  --host 127.0.0.1 \
  --port 8891 \
  --offload-server-url http://192.168.1.10:8890 \
  --brain-name xlerobot-macos-brain
```

Then publish local state or sensor summaries into the brain bridge:

```bash
curl -X POST http://127.0.0.1:8891/api/state \
  -H 'Content-Type: application/json' \
  -d '{
    "world_state": {
      "current_task": "inspect scene",
      "current_pose": "kitchen",
      "metadata": {
        "sensors": {
          "rgb": {"width": 640, "height": 480},
          "point_cloud": {"points": 1234}
        }
      }
    },
    "sensors": {
      "rgb": {"width": 640, "height": 480},
      "point_cloud": {"points": 1234}
    },
    "reason": "manual_publish"
  }'
```

## Smaller Example Entry Points

Inspect the XLeRobot bootstrap surface:

```bash
python examples/use_xlerobot_interface.py
```

What it offers:
- Prints the resolved XLeRobot integration summary
- Shows which modules are available through the bootstrap wrapper

Run the smaller agent example:

```bash
python examples/use_xlerobot_agent.py
```

What it offers:
- Runs the agent runtime on a fixed mock world state
- Useful for checking planner and executor wiring without teleop

Run the mock voice CLI:

```bash
python examples/run_mock_voice_agent.py
```

What it offers:
- Interactive mock voice-command shell for the agent runtime

## Current Caveats

- `--speed-profile fast` currently matters for the sim keyboard-based manipulation and sim keyboard-based recording paths.
- Sim camera visualization uses Rerun. Viewer/SDK version mismatch warnings are noisy but not blocking.
- Real camera streaming only appears if you pass explicit `--camera NAME=DRIVER:SOURCE` entries.
- Sim VR recording is not wired in the upstream XLeRobot ManiSkill demos yet.
- The sim recording hotkeys are controlled by the upstream XLeRobot ManiSkill recorder, not the real-backend `[` `]` `\\` hotkeys.
