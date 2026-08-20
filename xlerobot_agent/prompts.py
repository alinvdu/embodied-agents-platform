# Copyright 2026 Alin Vasile Dumitru
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
from typing import Any

from .models import GoalContext, SkillContract, Subgoal, WorldState


def build_instruction_normalization_system_prompt() -> str:
    return (
        "You normalize robot commands into one clean imperative instruction. "
        "Preserve the user intent, remove filler words, and return JSON only with key "
        "`normalized_instruction`."
    )


def build_instruction_normalization_user_prompt(text: str) -> str:
    return f"Raw instruction: {text}"


def build_place_discovery_system_prompt() -> str:
    return (
        "You infer likely places for a household robot from scene evidence and memory. "
        "Return JSON only with key `places`, whose value is a list of objects with keys "
        "`name`, `confidence`, and `evidence`."
    )


def build_place_discovery_user_prompt(world_state: WorldState) -> str:
    return "\n".join(
        [
            f"Current pose: {world_state.current_pose}",
            f"Visible objects: {sorted(world_state.visible_objects)}",
            f"Visible landmarks: {sorted(world_state.visible_landmarks)}",
            f"Image descriptions: {list(world_state.image_descriptions)}",
            f"Semantic memory summary: {world_state.semantic_memory_summary}",
            f"Spatial memory summary: {world_state.spatial_memory_summary}",
            f"Existing place memories: {[place.name for place in world_state.place_memories]}",
        ]
    )


def build_subgoal_planning_system_prompt() -> str:
    return (
        "You decompose a robot command into a short list of executable subgoals. "
        "Return JSON only with key `subgoals`, whose value is a list of objects with keys "
        "`text`, `kind`, and `target`. Use compact kinds such as `navigate`, `search`, "
        "`align`, `manipulate`, `recover`, or `general`. "
        "Use these patterns abstractly, not literally: "
        "for `go to the kitchen and open the fridge`, prefer subgoals like "
        "`navigate(kitchen) -> search(fridge) -> align(fridge) -> manipulate(fridge)`; "
        "for `map the downstairs area`, prefer a short mapping/exploration-oriented plan such as "
        "`general(downstairs)` instead of low-level tool ids. "
        "Do not emit tool names like `get_map` or `go_to_pose` as subgoal text."
    )


def build_subgoal_planning_user_prompt(
    goal: GoalContext,
    world_state: WorldState,
    skills: list[SkillContract],
) -> str:
    skill_ids = [skill.skill_id for skill in skills]
    return "\n".join(
        [
            f"User goal: {goal.user_instruction}",
            f"Structured goal: {goal.structured_goal or 'none'}",
            f"Current pose: {world_state.current_pose}",
            f"Visible objects: {sorted(world_state.visible_objects)}",
            f"Visible landmarks: {sorted(world_state.visible_landmarks)}",
            f"Available observations: {sorted(world_state.available_observations)}",
            f"Discovered places: {[place.name for place in world_state.place_memories]}",
            f"Recent execution history: {list(world_state.recent_execution_history)}",
            f"Available skill ids: {skill_ids}",
        ]
    )


def build_skill_selection_system_prompt() -> str:
    return (
        "You are the XLeRobot skill scoring module. "
        "Score every feasible skill only from the provided skill registry. "
        "Do not invent new skills. Return JSON only with key `scores`, whose value is "
        "a list of objects with keys `skill_id`, `goal_usefulness`, `success_likelihood`, "
        "`combined_score`, and `reasoning`. All scores must be between 0 and 1. "
        "Prefer skills for acting on the world once navigation/perception context is sufficient; "
        "prefer tools first when the agent still needs map state, target grounding, or an approach pose."
    )


def build_action_selection_system_prompt() -> str:
    return (
        "You are the planner for an embodied robot agent. "
        "Choose exactly one next action from the provided candidates. "
        "Return JSON only with keys: action_type, action_id, summary, reasoning_summary. "
        "Do not invent action ids that are not in the candidate list. "
        "Use these patterns abstractly, not literally: "
        "named-place navigation usually follows `get_map -> create_map/explore if needed -> go_to_pose`; "
        "object-centric approach usually follows `perceive_scene -> ground_object_3d -> "
        "set_waypoint_from_object -> go_to_pose`; "
        "when the robot is already well positioned and a matching registered VLA skill exists, "
        "prefer `run_vla_skill::<skill_id>` over extra perception or map steps; "
        "if `preferred_action_id` is present and still plausible, bias toward it."
    )


def build_exploration_policy_system_prompt() -> str:
    return (
        "You are the exploration policy for a mobile apartment-mapping robot. "
        "Your job is to choose the next high-level exploration action from frontier information generated by the deterministic mapping stack. "
        "Frontier information contains map coordinates at boundaries between explored free space and unknown space; it is partial RGB-D-derived evidence from where the camera has already scanned, not complete knowledge of the apartment and not a command to explore every boundary. "
        "`frontier_boundary_pose` is the original unexplored boundary; `approach_pose` is only a safe observation/navigation pose offset inward from that boundary. "
        "The deterministic detector is intentionally permissive: some listed frontiers are weak hypotheses where robot-sized free space touches unknown on only one side. "
        "Treat frontier listings as proposals, not truth. "
        "The deterministic stack owns occupancy mapping, frontier bookkeeping, safety checks, navigation, and scanning. "
        "You must reason about semantic coverage, spatial coverage, recent RGB visual input, and whether the remaining frontier memory still contains useful work. "
        "Return JSON only with keys: decision_type, selected_frontier_id, selected_return_waypoint_id, "
        "frontier_ids_to_store, memory_updates, exploration_complete, reasoning_summary. "
        "`decision_type` must be one of `explore_frontier`, `revisit_frontier`, or `finish`. "
        "Only use frontier ids and return waypoint ids that already exist in the prompt. "
        "Do not invent raw coordinates. Operate strictly on the provided frontier information objects. "
        "Select specific regions that likely expand robot-navigable floor space: room entrances, doorways, corridor continuations, open areas, and long-range expansions. "
        "`free_space_path_distance_m` is an approximate distance from the current robot pose to the frontier approach pose through currently known free space. "
        "Locality is a primary objective: explore the nearest useful region first so the robot expands the map coherently; avoid zig-zagging across the apartment. "
        "Do not select a frontier that is more than about 2x farther than another plausible frontier unless it clearly opens a major new room/corridor or has dramatically higher navigable-space value; if you do, explain the visual/map evidence explicitly. "
        "Do not blindly chase every unknown boundary. Deprioritize boundaries that look like backs, undersides, or sides of furniture such as couches, tables, cabinets, shelves, or clutter unless there is clear navigable space beyond them. "
        "You may veto any listed frontier if the navigation-map image plus RGB views suggest it is already-open mapped space, a wall sliver, or a non-navigable clutter shadow; do not select it, and use `memory_updates` with `suppress` or `keep` when appropriate. "
        "Use the navigation-map image and recent RGB views to interpret frontier information whenever map geometry alone is ambiguous. "
        "`memory_updates` must be a validated structured list of objects with keys: frontier_id, action, priority, label, notes, evidence. "
        "`action` must be one of `keep`, `store`, `prioritize`, `suppress`, or `revalidate`; only reference frontier ids from the prompt. "
        "Use memory updates to create and update exploration memory points in the same response: store useful later openings, prioritize the selected target, suppress likely furniture/clutter boundaries, and revalidate points only when new evidence supports traversable space. "
        "Treat sensor-range edge frontiers as real opportunities when the visible map reaches the 10 m sensing limit "
        "and space may continue beyond it. "
        "Keep frontier memory clean: visited and failed frontiers should not be proposed again unless the prompt explicitly "
        "shows them as revalidated. "
        "Set `exploration_complete` to true only when credible exploration opportunities are exhausted. "
        "If there are still reachable stored or candidate frontiers, prefer continuing exploration rather than finishing. "
        "Do not create semantic labels, named places, semantic waypoints, or semantic anchor updates in this response. "
        "Semantic named places are produced by a separate RGB-D evidence projection pipeline with its own prompt, "
        "schema, memory, and lifecycle."
    )


def build_exploration_policy_user_prompt(payload: dict[str, Any]) -> str:
    robot = payload.get("robot", {})
    frontier_memory = _compact_frontier_memory_for_prompt(payload.get("frontier_memory", {}))
    frontier_information = _compact_frontier_information_for_prompt(
        payload.get("frontier_information", payload.get("candidate_frontiers", []))
    )
    navigation_map_views = _redact_prompt_data_urls(payload.get("navigation_map_views", []))
    recent_views = _redact_prompt_data_urls(payload.get("recent_views", []))
    sections = [
        "Mission:",
        str(payload.get("mission", "Explore the apartment and finish only when mapping is complete.")),
        "",
        "Robot State:",
        json.dumps(robot, indent=2, sort_keys=True),
        "",
        "Frontier Memory Status:",
        (
            "Status-only memory context. Full frontier geometry and evidence appears only once, in Frontier Information."
        ),
        json.dumps(frontier_memory, indent=2, sort_keys=True),
        "",
        "Frontier Information:",
        (
            "These are boundary observations from deterministic mapping: coordinates where explored free space touches "
            "unknown space. Each item has `source`: `current_rgbd_scan` means the frontier was detected from the latest "
            "scan/global-map update, while `frontier_memory` means it was discovered during previous scans and is being "
            "offered as a reachable stored memory point. "
            "`frontier_boundary_pose` is the real frontier being judged; `approach_pose` is only the safe robot observation pose used for motion. "
            "`free_space_path_distance_m` is an approximate route distance from the current robot pose to `approach_pose` through known free space. "
            "Locality is a primary objective: choose nearby useful frontiers first to expand the map coherently, and avoid jumping across the apartment. "
            "Do not pick a frontier more than about 2x farther than another plausible frontier unless the navigation map and RGB views show it is clearly much more valuable. "
            "They are not complete apartment knowledge. "
            "Some frontier hypotheses are intentionally permissive and may come from robot-sized free space touching unknown on only one side. "
            "You are expected to veto those when the map overlay plus RGB views suggest no real unexplored navigable opening. "
            "If `source` is `frontier_memory`, treat the item as a previous-scan memory candidate and choose it with "
            "`decision_type: revisit_frontier` only when it still looks useful. "
            "They are not all equally useful. Select only a frontier id that likely expands robot-navigable "
            "floor space, such as a doorway, hallway continuation, room opening, open area, or meaningful sensor-range edge. "
            "Use the recent RGB visual input to interpret whether each boundary looks like a navigable opening or a "
            "furniture/clutter shadow. Avoid choosing boundaries that are likely just behind furniture or clutter unless the evidence suggests a "
            "real traversable opening."
        ),
        json.dumps(frontier_information, indent=2, sort_keys=True),
        "",
        "Frontier Selection Guidance:",
        json.dumps(payload.get("frontier_selection_guidance", []), indent=2, sort_keys=True),
        "",
        "Navigation Map View:",
        (
            "Attached image(s) show the same operator-review navigation map used by the web UI: occupancy cells, "
            "robot pose and heading, trajectory, and frontier labels from Frontier Information."
        ),
        json.dumps(navigation_map_views, indent=2, sort_keys=True),
        "",
        "Memory Update Instructions:",
        (
            "In the same JSON response, create/update frontier memory using `memory_updates`. "
            "Use `prioritize` for the selected frontier when it should be explored now, `store` or `keep` for useful later openings, "
            "`suppress` for furniture/clutter/wall-shadow boundaries, and `revalidate` only when new visual/map evidence makes a previously weak point useful. "
            "Every memory update must reference an existing frontier_id from Frontier Information."
        ),
        "",
        "Explored Areas:",
        json.dumps(payload.get("explored_areas", []), indent=2, sort_keys=True),
        "",
        "Recent Views:",
        json.dumps(recent_views, indent=2, sort_keys=True),
        "",
        "Guardrails:",
        json.dumps(payload.get("guardrails", {}), indent=2, sort_keys=True),
        "",
        "Out Of Scope:",
        (
            "Semantic labels, named places, semantic waypoints, and semantic anchor updates are not part of this "
            "frontier policy response. They are handled by a separate passive RGB-D visual evidence pipeline."
        ),
        "",
        "ASCII Occupancy Map:",
        "Legend: ? unknown, . free, # occupied, R robot, F active/stored frontier, V visited/completed frontier.",
        str(payload.get("ascii_map", "")),
    ]
    return "\n".join(sections)


def _redact_prompt_data_urls(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(item, str) and item.startswith("data:image/"):
                prefix, _, encoded = item.partition(",")
                redacted[key] = {
                    "attached_as_multimodal_image": True,
                    "mime_type": prefix.split(";")[0].removeprefix("data:"),
                    "base64_bytes": len(encoded),
                }
                continue
            redacted[key] = _redact_prompt_data_urls(item)
        return redacted
    if isinstance(value, list):
        return [_redact_prompt_data_urls(item) for item in value]
    return value


def _compact_frontier_information_for_prompt(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [
        _compact_frontier_record_for_prompt(item, include_geometry=True)
        for item in value
        if isinstance(item, dict) and item.get("frontier_id")
    ]


def _compact_frontier_memory_for_prompt(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    compact: dict[str, Any] = {}
    active = value.get("active_frontier")
    compact["active_frontier_id"] = active.get("frontier_id") if isinstance(active, dict) else None
    for key in (
        "stored_frontiers",
        "completed_frontiers",
        "visited_frontiers",
        "failed_frontiers",
        "suppressed_frontiers",
    ):
        items = value.get(key, [])
        if not isinstance(items, list):
            continue
        compact[key] = [
            _compact_frontier_record_for_prompt(item, include_geometry=False)
            for item in items
            if isinstance(item, dict) and item.get("frontier_id")
        ]
    compact["return_waypoints"] = value.get("return_waypoints", [])
    return compact


def _compact_frontier_record_for_prompt(record: dict[str, Any], *, include_geometry: bool) -> dict[str, Any]:
    compact = {
        "frontier_id": record.get("frontier_id"),
        "status": record.get("status"),
        "source": "current_rgbd_scan" if record.get("currently_visible") else "frontier_memory",
        "discovered_step": record.get("discovered_step"),
        "last_seen_step": record.get("last_seen_step"),
        "path_cost_m": record.get("path_cost_m"),
        "free_space_path_distance_m": record.get("free_space_path_distance_m", record.get("path_cost_m")),
        "unknown_gain": record.get("unknown_gain"),
        "sensor_range_edge": record.get("sensor_range_edge"),
        "attempt_count": record.get("attempt_count"),
        "visit_count": record.get("visit_count"),
        "llm_memory_priority": record.get("llm_memory_priority"),
        "llm_memory_label": record.get("llm_memory_label"),
    }
    if include_geometry:
        compact["approach_pose"] = record.get("approach_pose") or record.get("nav_pose")
        compact["frontier_boundary_pose"] = record.get("frontier_boundary_pose") or record.get("centroid_pose")
        evidence = record.get("evidence", [])
        if isinstance(evidence, list):
            compact["evidence"] = [str(item) for item in evidence[:5]]
            compact["evidence_count"] = len(evidence)
    return {
        key: item
        for key, item in compact.items()
        if item is not None
    }


def build_skill_selection_user_prompt(
    goal: GoalContext,
    subgoal: Subgoal,
    world_state: WorldState,
    skills: list[SkillContract],
) -> str:
    skill_lines = []
    for skill in skills:
        skill_lines.append(
            "- "
            f"skill_id={skill.skill_id}; "
            f"type={skill.skill_type.value}; "
            f"description={skill.language_description}; "
            f"executor={skill.executor_binding}; "
            f"required_observations={sorted(skill.required_observations)}; "
            f"required_resources={sorted(skill.required_resources)}; "
            f"postcondition={skill.expected_postcondition}"
        )

    return "\n".join(
        [
            f"User goal: {goal.user_instruction}",
            f"Structured goal: {goal.structured_goal or 'none'}",
            f"Current subgoal: {subgoal.text}",
            f"Subgoal kind: {subgoal.kind}",
            f"Subgoal target: {subgoal.target or 'none'}",
            f"Current task: {world_state.current_task}",
            f"Current pose: {world_state.current_pose}",
            f"Localization confidence: {world_state.localization_confidence}",
            f"Visible objects: {sorted(world_state.visible_objects)}",
            f"Visible landmarks: {sorted(world_state.visible_landmarks)}",
            f"Image descriptions: {list(world_state.image_descriptions)}",
            f"Discovered places: {[place.name for place in world_state.place_memories]}",
            f"Available observations: {sorted(world_state.available_observations)}",
            f"Recent execution history: {list(world_state.recent_execution_history)}",
            f"Active resource locks: {sorted(world_state.active_resource_locks)}",
            f"Semantic memory summary: {world_state.semantic_memory_summary}",
            f"Spatial memory summary: {world_state.spatial_memory_summary}",
            "Available skills:",
            *skill_lines,
            (
                "For each feasible skill, estimate goal usefulness and success likelihood, "
                "then compute combined_score. Return one score object per provided skill."
            ),
        ]
    )


def build_voice_command_system_prompt(wake_word: str) -> str:
    return (
        "You normalize spoken robot commands into direct action text for XLeRobot. "
        f"The wake word is `{wake_word}`. Strip the wake word and return the "
        "command in plain imperative English."
    )


def build_voice_command_user_prompt(transcript: str) -> str:
    return f"Transcript: {transcript}"


def build_visual_scene_summary_system_prompt() -> str:
    return (
        "You are the visual differencing module for a household robot. "
        "Summarize the current scene for planning. "
        "Return JSON only with keys: summary, reasoning_summary, task_completed, "
        "change_detected, task_relevant_attributes, delta. "
        "Use task_completed only when the task already appears visually complete."
    )


def build_visual_scene_summary_user_prompt(
    *,
    instruction: str,
    world_state: WorldState,
    scene: dict,
    target: str | None = None,
) -> str:
    payload = {
        "instruction": instruction,
        "target": target,
        "current_pose": world_state.current_pose,
        "visible_objects": sorted(world_state.visible_objects),
        "image_descriptions": list(world_state.image_descriptions),
        "scene": {
            "scene_summary": scene.get("scene_summary"),
            "available_streams": scene.get("available_streams"),
            "query_target": scene.get("query_target"),
            "annotations": [
                {
                    "label": item.get("label"),
                    "confidence": item.get("confidence"),
                    "depth_m": item.get("depth_m"),
                    "overlay_text": item.get("overlay_text"),
                }
                for item in scene.get("annotations", [])
                if isinstance(item, dict)
            ][:8],
        },
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def build_visual_difference_system_prompt() -> str:
    return (
        "You are the visual differencing module for a household robot. "
        "Compare the previous and current observations after one action. "
        "Return JSON only with keys: summary, reasoning_summary, task_completed, "
        "change_detected, task_relevant_attributes, delta. "
        "Focus on task-relevant changes and whether the subgoal now appears complete."
    )


def build_visual_difference_user_prompt(
    *,
    instruction: str,
    subgoal: Subgoal,
    previous_world_state: WorldState,
    current_world_state: WorldState,
    previous_scene: dict,
    current_scene: dict,
    action: dict,
) -> str:
    payload = {
        "instruction": instruction,
        "subgoal": {"text": subgoal.text, "kind": subgoal.kind, "target": subgoal.target},
        "action": action,
        "previous_world_state": {
            "current_pose": previous_world_state.current_pose,
            "visible_objects": sorted(previous_world_state.visible_objects),
            "available_observations": sorted(previous_world_state.available_observations),
            "satisfied_preconditions": sorted(previous_world_state.satisfied_preconditions),
            "image_descriptions": list(previous_world_state.image_descriptions),
        },
        "current_world_state": {
            "current_pose": current_world_state.current_pose,
            "visible_objects": sorted(current_world_state.visible_objects),
            "available_observations": sorted(current_world_state.available_observations),
            "satisfied_preconditions": sorted(current_world_state.satisfied_preconditions),
            "image_descriptions": list(current_world_state.image_descriptions),
        },
        "previous_scene": {
            "scene_summary": previous_scene.get("scene_summary"),
            "annotations": [
                {
                    "label": item.get("label"),
                    "confidence": item.get("confidence"),
                    "depth_m": item.get("depth_m"),
                }
                for item in previous_scene.get("annotations", [])
                if isinstance(item, dict)
            ][:8],
        },
        "current_scene": {
            "scene_summary": current_scene.get("scene_summary"),
            "annotations": [
                {
                    "label": item.get("label"),
                    "confidence": item.get("confidence"),
                    "depth_m": item.get("depth_m"),
                }
                for item in current_scene.get("annotations", [])
                if isinstance(item, dict)
            ][:8],
        },
    }
    return json.dumps(payload, indent=2, sort_keys=True)
