# Robot42 Agentic End To End

This guide is for running the agentic flow. If a long-term memory already exists, you do **not** need to start exploration.

## Normal Agent Flow

Use this when the environment has already been approved and saved under `artifacts/memories`.

### 1. Start Agent Backend

From the repo root:

For live agent navigation with OpenAI traces:

```bash
python examples/robot42_agent_backend.py \
  --memory-root ./artifacts/memories \
  --provider openai \
  --model gpt-5.5 \
  --exploration-backend-url http://127.0.0.1:8770 \
  --navigation-waypoint-horizon-m 2.0
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
```

Current behavior:
- the agent reads regions and occupancy from `home_memory.json`
- memory lookup is prompt/context, not a tool call
- region navigation starts with `resolve_navigation_to_region`, which resolves `kitchen` against saved occupancy/free space and returns a final safe pose plus a short-horizon waypoint
- the exposed navigation tools are `resolve_navigation_to_region`, `navigate_to_waypoint`, and `relocalize_here`
- perception, manipulation, and VLA/skill tools are intentionally not exposed yet

Region polygons are semantic labels, not navigation goals. The agent should not choose a target from the region shape directly. When it needs to move to a region, it calls the region navigation resolver, which searches known-free occupancy cells inside the named region, erodes free space by the robot footprint plus a small gap, and prefers centerline cells with higher clearance from occupied space. The resolver then walks along that same preview path to produce `next_waypoint`, defaulting to a `2.0 m` horizon.

Navigation flow:
- `resolve_navigation_to_region("kitchen")`
- agent reads `next_waypoint`
- `navigate_to_waypoint(waypoint_id, x, y, yaw)`
- wait for the Nav2 result from the exploration backend
- `relocalize_here()`
- resolve again from the updated/corrected pose if the waypoint was not final

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
  --exploration-backend-url http://127.0.0.1:8770 \
  --navigation-waypoint-horizon-m 1.5
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
