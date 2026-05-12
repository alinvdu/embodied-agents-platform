# Long-Term Memory And Agentic Robot Roadmap

Date: 2026-05-12

This is the next project direction I would take: stop treating autonomous frontier exploration as the center of the project, and instead make the robot good at using a manually approved, semantically annotated home map as long-term memory. The robot can still scan and update local space, but the main product effect should be: "I ask for a real task, the robot reasons over known places and objects, navigates to the right area, perceives what is in front of it, and invokes a trained or teleoperated skill."

## Current Repo Reality

The project already has most of the raw pieces for this direction.

- `xlerobot_agent/exploration_ui.py` already exposes a review UI with occupancy editing, region label/polygon/default-waypoint editing, named places, map approval, manual scans, waypoint preview, and waypoint navigation.
- `xlerobot_agent/exploration.py` already persists an exploration backend snapshot to `persist_path`, tracks `current_map`, stores approved maps, supports `approve_current_map`, `update_region`, `set_named_place`, and persists manual occupancy edits.
- `xlerobot_playground/map_editing.py` already models manual occupancy overrides and can overlay blocked/cleared cells onto the map. This is exactly the "I draw walls/window barriers and the saved map respects them" mechanism.
- `xlerobot_playground/sim_exploration_backend.py` already serializes a ROS/Nav2 map payload with occupancy cells, robot pose, trajectory, frontiers, keyframes, manual occupancy edits, and optional semantic memory. For real runs, `regions` are currently empty by default, and manual regions are intended to be operator authored.
- `xlerobot_playground/semantic_memory.py`, `semantic_evidence.py`, `semantic_anchors.py`, and `semantic_projection.py` are a parked semantic memory path. Useful, but not yet the primary long-term home-memory representation.
- `xlerobot_agent/tools.py`, `xlerobot_agent/playground.py`, `xlerobot_agent/runtime.py`, and tests already contain a mock/local agent loop with tools like `create_map`, `get_map`, `go_to_pose`, `perceive_scene`, `ground_object_3d`, and `set_waypoint_from_object`.
- `xlerobot_playground/robot_brain_agent.py` is a hardware endpoint for motion, stop, RGB-D, IMU, and camera pan/pitch state. It is not an LLM agent; it is the robot IO surface.

Important limitation: saving the exploration JSON through `--persist-path` works, but saving directly into a stable "robot long-term memory" store is still a separate integration step. The existing docs also call this out.

## Answering The Immediate Questions

Yes, there are regions in `exploration_ui`, but the current UI is mostly JSON-edit driven. It can display/select existing regions, update label/polygon/default waypoints, split/merge regions, and set named places. It does not yet have a polished "draw semantic region polygon on the map" workflow. You probably want to add that.

Yes, you can save the map as a JSON-backed map snapshot today through `--persist-path`. The saved map can include manual wall annotations because manual occupancy edits are overlaid into the map payload and stored under `artifacts.manual_occupancy_edits`. However, that saved exploration snapshot is not yet a clean long-term-memory artifact with versioning, start pose, region ontology, object affordances, and agent query APIs.

No, the production agentic AI setup is not really there yet. There is a good prototype scaffold and mock/local runtime, but not a proper OpenAI Agents SDK runtime wired to the robot's long-term memory, Nav2, perception, safety checks, and VLA skill registry.

The right next move is to formalize a `HomeMemory` artifact and make the agent consume that, instead of making the robot discover everything from scratch.

## Proposed Long-Term Memory Artifact

Create a new explicit artifact, probably:

```text
artifacts/home_memory/house_v1.home_memory.json
```

It should be generated from an approved exploration/map-review snapshot, not hand-maintained forever. The source of truth during authoring remains the exploration UI map plus your annotations.

Recommended top-level shape:

```json
{
  "schema_version": "home_memory.v1",
  "memory_id": "house_v1",
  "created_at": 1710000000.0,
  "updated_at": 1710000000.0,
  "frame": "map",
  "source_map_id": "house_v1",
  "approved": true,
  "start_pose": {
    "name": "dock",
    "pose": {"x": 0.0, "y": 0.0, "yaw": 0.0},
    "fixed": true,
    "notes": "Robot always starts here."
  },
  "occupancy": {
    "resolution": 0.25,
    "bounds": {"min_x": -4.0, "max_x": 4.0, "min_y": -2.0, "max_y": 6.0},
    "cells": []
  },
  "manual_occupancy_edits": {
    "blocked_cells": [],
    "cleared_cells": []
  },
  "regions": [],
  "places": [],
  "objects": [],
  "navigation_graph": {
    "waypoints": [],
    "edges": []
  },
  "skills": [],
  "provenance": {
    "source_snapshot_path": "artifacts/real_xlerobot_exploration_map.json",
    "operator": "alin",
    "notes": []
  }
}
```

Region shape:

```json
{
  "region_id": "region_kitchen",
  "label": "kitchen",
  "polygon_2d": [[1.0, 1.0], [4.0, 1.0], [4.0, 3.0], [1.0, 3.0]],
  "centroid": {"x": 2.5, "y": 2.0},
  "purpose": "food storage, drinks, cooking, sink",
  "entry_waypoints": [
    {"name": "kitchen_entry", "x": 1.2, "y": 2.0, "yaw": 0.0}
  ],
  "scan_waypoints": [
    {"name": "kitchen_scan_center", "x": 2.6, "y": 2.0, "yaw": 1.57}
  ],
  "default_waypoints": [
    {"name": "kitchen_center", "x": 2.6, "y": 2.0, "yaw": 1.57}
  ],
  "adjacent_region_ids": ["region_hallway"],
  "constraints": {
    "do_not_enter_polygons": [],
    "preferred_standoff_m": 0.8
  }
}
```

Object or fixture shape:

```json
{
  "object_id": "fixture_fridge",
  "label": "fridge",
  "category": "appliance",
  "region_id": "region_kitchen",
  "pose": {"x": 3.6, "y": 2.8, "yaw": -1.57},
  "approach_pose": {"x": 3.1, "y": 2.8, "yaw": 0.0},
  "observable_from": ["kitchen_scan_center"],
  "affordances": ["inspect_contents", "open_fridge"],
  "notes": "Use RGB-D grounding once near the approach pose."
}
```

Skill shape:

```json
{
  "skill_id": "open_fridge",
  "kind": "vla_or_scripted_skill",
  "target_categories": ["fridge"],
  "required_pose_class": "in_front_of_fridge",
  "required_observations": ["fridge_visible", "handle_visible"],
  "executor_binding": "vla_skill_runner",
  "safety": {
    "requires_human_approval": true,
    "max_retries": 1
  }
}
```

## Long-Term Versus Short-Term Memory

Use two layers, with clear authority boundaries.

Long-term memory is the approved home map. It contains static geometry, manually drawn blocked areas, semantic regions, fixed fixtures, known approach poses, start pose, navigation graph, and skill affordances.

Short-term memory is live and disposable. It contains current pose, local obstacles, recent RGB-D detections, object grounding results, local costmap deltas, and task-local facts like "the can of coke was detected on the second shelf." Short-term memory may veto long-term memory for collision and perception. Long-term memory should guide where to look and how to approach, but local perception decides whether the path or manipulation pose is currently safe.

Rule: local collision and local perception take precedence. If the long-term map says the path is free but the local costmap sees a chair, stop or replan locally. If the long-term map says the fridge is at a fixture pose, still use RGB-D grounding near the fridge before opening or reaching.

## Manual Exploration Workflow

This should be the primary mapping workflow for now.

1. Put robot at fixed start pose.
2. Run real exploration with review UI, initial scan, and operator approval pause.
3. Operator clicks waypoints or teleops the robot to useful scan positions.
4. Operator draws manual walls/barriers for places the robot sees but cannot navigate, such as sliding windows or unreachable glass.
5. Operator draws semantic region polygons: kitchen, living room, hallway, desk area, fridge zone, charging/dock area.
6. Operator adds named places and object/fixture anchors: fridge, sink, counter, couch, table, dock.
7. Operator sets start pose/dock pose.
8. Approve map.
9. Export approved annotated map into `house_v1.home_memory.json`.
10. Agent uses `HomeMemoryStore` as its default map context.

This avoids spending weeks on perfect frontier behavior before the robot can do useful tasks.

## Frontier Exploration Spec, If Kept

If you still keep automatic exploration, make it secondary and conservative.

The robot should not navigate to the edge of unknown space. It should choose scan poses in known safe free space, generally centered in navigable corridors or rooms, at about 2-3 m from the current sensed boundary, and orient the camera toward frontier clusters. The frontier is a gaze target, not necessarily a base target.

Candidate scan pose rules:

- Pose must be in known free space after manual occupancy edits are applied.
- Pose must satisfy robot footprint clearance and Nav2 path reachability.
- Pose should be at least `robot_radius + safety_margin` from occupied cells.
- Pose should maximize visible unknown boundary from the scan pose.
- Pose should prefer room/corridor midlines over wall-hugging.
- Pose yaw should face the frontier centroid or the semantic object/area of interest.
- Repeated failed frontiers should be remembered and deprioritized.

This is useful later, but not the highest-leverage path right now.

## Agent Architecture Direction

Use the OpenAI Agents SDK as the orchestration layer, not as a direct motor controller. The SDK is a good fit because it provides agents with tools, handoffs/agents-as-tools, sessions, human-in-the-loop, and tracing. The Python quickstart currently uses `openai-agents`, `Agent`, `Runner`, and `function_tool`; the SDK docs also emphasize tracing and session memory.

Recommended first production-ish agent:

- `HomeTaskAgent`: the one main agent. It understands the user request, queries long-term memory, chooses navigation targets, asks perception what is currently visible, checks skill preconditions, and decides which skill/tool to run next.

Start with tools/modules, not many specialist agents:

- Memory tools: `query_place`, `query_object`, `resolve_navigation_goal`, `get_approach_pose`.
- Navigation tools: `preview_path`, `navigate_to_pose`, `navigate_to_region`.
- Perception tools: `capture_rgbd`, `perceive_scene`, `ground_object_3d`, `update_short_term_memory`.
- Skill tools: `list_skills`, `check_skill_preconditions`, `run_vla_skill`, `run_scripted_skill`.
- Safety tools/guardrails: `ask_human_approval`, `stop_robot`, and confidence checks before movement/manipulation.

Do not add separate `MemoryAgent`, `NavigationAgent`, or `PerceptionAgent` yet. Those domains should be deterministic modules and typed tools first. More agents only become useful once a domain needs its own reasoning loop.

The one likely later addition is a `SkillAgent`, but only after manipulation becomes complex enough to need local retry/recovery. For example, opening a fridge may eventually need: align, try handle, observe failure, adjust gripper, retry, or ask for help. Until then, `run_vla_skill(...)` can just be a tool called by `HomeTaskAgent`.

Do not let the LLM emit raw motor commands. It should emit structured objectives and call typed tools. Example:

```text
User: open the fridge and tell me what I have
Agent:
1. query_memory("fridge")
2. navigate_to_pose(fridge.approach_pose)
3. perceive_scene(target="fridge")
4. ground_object_3d("fridge_handle")
5. check_skill_preconditions("open_fridge")
6. request_human_approval("open_fridge") while skill is not trusted
7. run_vla_skill("open_fridge")
8. perceive_scene(target="fridge_contents")
9. summarize_contents()
```

## Implementation Phases

### Phase 1: Make Approved Maps Into Home Memory

Goal: take the current annotated exploration map and save it as a stable memory artifact.

Tasks:

- Add `xlerobot_agent/home_memory.py` with dataclasses or typed dicts for `HomeMemory`, `MemoryRegion`, `MemoryPlace`, `MemoryObject`, `MemorySkill`, and `NavigationGraph`.
- Add `HomeMemoryStore` with `load(memory_id)`, `save(memory)`, `export_from_map_snapshot(snapshot)`, and `summarize_for_agent(memory)`.
- Add tests that convert an `ExplorationBackend` approved map with manual edits, regions, named places, and start pose into `home_memory.v1`.
- Add a CLI script, for example `scripts/export_home_memory.py --snapshot artifacts/real_xlerobot_exploration_map.json --memory-id house_v1 --out artifacts/home_memory/house_v1.home_memory.json`.
- Add start pose support to the UI/backend, not just current `robot_pose`. It should be a named fixed pose in the memory artifact.

Acceptance:

- A map edited in the UI, including drawn walls, regions, named places, and start pose, exports to a deterministic JSON file.
- Reloading that JSON gives the agent a compact summary of navigable regions, places, and approach poses.

### Phase 2: Improve The Annotation UI

Goal: make manual annotation pleasant enough that you will actually use it.

Tasks:

- Add a semantic region draw mode to `exploration_ui.py`, parallel to the occupancy edit and waypoint modes.
- Let clicks create polygon vertices, with save/cancel/undo.
- Let the operator set region label, purpose, default scan waypoint, and default entry waypoint.
- Add "Set Start Pose" from either current robot pose or map click/yaw.
- Add object/fixture anchor editing: fridge, sink, counter, table, dock, couch, etc.
- Store all of this in map payload before approval.

Acceptance:

- You can create kitchen/living/hallway regions without manually typing polygon JSON.
- You can mark a fridge approach pose and a dock/start pose from the UI.

### Phase 3: Agent Tools Over Home Memory

Goal: let the existing agent runtime reason over the real memory artifact.

Tasks:

- Add tools:
  - `load_home_memory(memory_id)`
  - `query_place(name_or_label)`
  - `query_object(label)`
  - `resolve_navigation_goal(goal_text)`
  - `get_approach_pose(target)`
  - `preview_path_to_pose(pose)`
  - `navigate_to_pose_nav2(pose)`
- Update `WorldState` to carry `home_memory_id`, `home_memory_summary`, and short-term observations separately from the old free-text `semantic_memory_summary` and `spatial_memory_summary`.
- Wire `get_map`/`go_to_pose` to use `HomeMemoryStore` first, then the current exploration backend as fallback.
- Keep the mock tests, but add at least one integration-style test: "bring me a coke" resolves kitchen/fridge, navigates to fridge approach, perceives fridge/coke, then proposes a manipulation skill.

Acceptance:

- A command like "go to the kitchen" resolves through `house_v1.home_memory.json`, not through a freshly generated sim map.
- A command like "look in the fridge" resolves `fridge -> approach_pose -> Nav2 goal -> RGB-D perception`.

### Phase 4: Production Agents SDK Runtime

Goal: replace the mock-ish planner loop with a real tool-calling agent runtime while keeping strict safety boundaries.

Tasks:

- Add an optional `openai-agents` dependency path and an `xlerobot_agent/agents_sdk_runtime.py`.
- Implement one `HomeTaskAgent` first.
- Wrap memory, navigation, perception, safety, and skill calls as `function_tool`s.
- Keep specialist agents out of the first production version. Add a `SkillAgent` later only if manipulation attempts need their own observe/retry/recover loop.
- Use SDK sessions for conversation continuity, but keep `HomeMemoryStore` as the durable environment memory. Do not rely on chat history as the map.
- Turn on tracing in development so failed plans/tool calls are inspectable.
- Add human approval gates for real Nav2 movement initially, and definitely for manipulation.

Acceptance:

- Running one command creates a trace with the plan, memory lookup, Nav2 target, perception result, and final response.
- The agent can be run in dry-run mode and real-execution mode.

### Phase 5: Short-Term Perception And Object Grounding

Goal: when the robot is near a region/object, use RGB-D to find the exact current object pose.

Tasks:

- Make `perceive_scene`, `ground_object_3d`, and `set_waypoint_from_object` work against live robot-brain RGB-D, not only metadata/mock anchors.
- Store short-term detections with timestamps, frame, confidence, object label, 3D anchor, and source frame id.
- Add local-memory invalidation: detections expire unless refreshed.
- For fixtures like fridge, use long-term pose to aim/search, then local RGB-D to align.
- For movable items like coke can, never rely only on long-term memory.

Acceptance:

- Near the fridge, `ground_object_3d("fridge")` or `ground_object_3d("coke can")` returns a current 3D anchor with confidence and an approach/alignment suggestion.

### Phase 6: VLA Data And Skill Runtime

Goal: create the runway for manipulation skills without forcing the agent to learn everything at once.

Tasks:

- Define a VLA dataset schema: observation frames, proprioception, action vector, teleop source, task label, object target, success/failure, environment memory id.
- Build VR teleoperation recording around the robot-brain streams: RGB-D, IMU, joint states/actions, base commands, and optional voice/task label.
- Add a `SkillRegistry` extension for trained skills:
  - `open_fridge`
  - `pick_can`
  - `place_item`
  - `inspect_fridge_contents`
- Add `run_vla_skill(skill_id, target_anchor, constraints)` as an agent tool.
- Keep scripted or teleop fallback for early tests.

Acceptance:

- You can record a teleop episode for "pick coke can from visible shelf" with synchronized observations/actions.
- The agent can select a skill only after navigation and perception preconditions are satisfied.

## Recommended Next Three PRs

1. Home memory exporter:
   - Add `home_memory.py`, exporter script, and tests.
   - Include start pose in schema, even if initially provided by CLI.

2. UI annotation upgrade:
   - Add region polygon draw mode, start pose tool, and fixture/object anchors.
   - Persist these annotations into the map snapshot.

3. Memory-aware agent tools:
   - Add `HomeMemoryStore`.
   - Teach `get_map`/`go_to_pose`/new query tools to use approved memory.
   - Add one dry-run agent test for "open the fridge and tell me what I have."

## Design Principle

Treat the manually approved map as the robot's home knowledge, not as a temporary exploration artifact. Let the agent reason over stable human-authored semantics, then use local RGB-D and Nav2 as reality checks at execution time.

That gives the project the effect you actually want: a robot that can understand "kitchen", "fridge", "bring me the can of coke", and "start from the dock" as grounded spatial tasks, instead of a robot that is forever trying to decide which frontier is interesting.

## References Checked

- OpenAI Agents SDK overview: agents, tools, handoffs/agents-as-tools, guardrails, sessions, human-in-the-loop, and tracing are first-class SDK concepts.
- OpenAI Agents SDK quickstart: current Python package is `openai-agents`, with `Agent`, `Runner`, and `function_tool`.
- OpenAI Agents SDK sessions docs: sessions maintain conversation history across runs, but should not replace durable environment memory.
- OpenAI Agents SDK tracing docs: tracing records LLM generations, tool calls, handoffs, guardrails, and custom events, useful for debugging robot plans.
