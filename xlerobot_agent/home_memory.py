from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import json
import math
from pathlib import Path
import time
from typing import Any

from .memory_discovery import HOME_MEMORY_FILENAME, default_environment_memory_dir_for_map_path


HOME_MEMORY_SCHEMA_VERSION = "home_memory.v1"
XLEROBOT_FOOTPRINT_LENGTH_M = 0.3913
XLEROBOT_FOOTPRINT_WIDTH_M = 0.459
NAV2_FOOTPRINT_PADDING_M = 0.03
NAV2_INFLATION_RADIUS_M = 0.07
DEFAULT_ROBOT_FOOTPRINT_RADIUS_M = math.hypot(
    XLEROBOT_FOOTPRINT_LENGTH_M / 2.0,
    XLEROBOT_FOOTPRINT_WIDTH_M / 2.0,
)
DEFAULT_NAVIGATION_CLEARANCE_M = XLEROBOT_FOOTPRINT_WIDTH_M / 2.0 + NAV2_FOOTPRINT_PADDING_M
DEFAULT_NAVIGATION_CENTERLINE_PREFERENCE_M = XLEROBOT_FOOTPRINT_WIDTH_M / 2.0 + NAV2_INFLATION_RADIUS_M
DEFAULT_NAVIGATION_WAYPOINT_HORIZON_M = 2.0
DEFAULT_REGION_EXPLORATION_FOV_DEG = 65.0
DEFAULT_REGION_EXPLORATION_SHOTS_PER_STOP = 2
DEFAULT_REGION_EXPLORATION_BOUNDARY_MARGIN_M = 0.65
DEFAULT_REGION_EXPLORATION_MAX_RANGE_M = 3.0
DEFAULT_REGION_EXPLORATION_MIN_RANGE_M = 0.20
DEFAULT_REGION_EXPLORATION_MIN_STOP_SEPARATION_M = 0.45
DEFAULT_DIRECT_NAVIGATION_FALLBACK_MAX_DISTANCE_M = 1.0


@dataclass(frozen=True)
class HomeMemoryStore:
    path: Path

    def load(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        return json.loads(self.path.read_text())

    def save(self, memory: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(memory, indent=2, sort_keys=True))

    def export_from_map_snapshot(self, map_payload: dict[str, Any]) -> dict[str, Any]:
        memory = home_memory_from_map_snapshot(map_payload, source_path=str(self.path))
        self.save(memory)
        return memory


@dataclass
class HomeMemoryExportResult:
    memory: dict[str, Any]
    preserved_in_map_only: list[str] = field(default_factory=list)
    distilled_fields: list[str] = field(default_factory=list)


def default_home_memory_path_for_map_path(path: str | Path) -> Path:
    return default_environment_memory_dir_for_map_path(path) / HOME_MEMORY_FILENAME


def home_memory_from_map_snapshot(
    map_payload: dict[str, Any],
    *,
    source_path: str | None = None,
) -> dict[str, Any]:
    memory_id = str(map_payload.get("map_id") or "home")
    occupancy = _json_clone(map_payload.get("occupancy") or {})
    artifacts = map_payload.get("artifacts") if isinstance(map_payload.get("artifacts"), dict) else {}
    manual_edits = artifacts.get("manual_occupancy_edits") if isinstance(artifacts, dict) else None
    start_pose = _start_pose_from_map(map_payload)
    regions = [_memory_region(region) for region in map_payload.get("regions", []) if isinstance(region, dict)]
    places = [_memory_place(place) for place in map_payload.get("named_places", []) if isinstance(place, dict)]
    objects = _objects_from_map(map_payload)
    navigation_graph = _navigation_graph_from_regions(regions, places)
    skills = _skills_from_objects(objects)

    return {
        "schema_version": HOME_MEMORY_SCHEMA_VERSION,
        "memory_id": memory_id,
        "created_at": float(map_payload.get("created_at", time.time()) or time.time()),
        "updated_at": time.time(),
        "frame": str(map_payload.get("frame") or "map"),
        "source_map_id": memory_id,
        "approved": bool(map_payload.get("approved", False)),
        "approved_at": map_payload.get("approved_at"),
        "start_pose": start_pose,
        "occupancy": occupancy,
        "manual_occupancy_edits": _json_clone(manual_edits or {"blocked_cells": [], "cleared_cells": []}),
        "regions": regions,
        "places": places,
        "objects": objects,
        "navigation_graph": navigation_graph,
        "skills": skills,
        "provenance": {
            "source_snapshot_path": source_path,
            "source_mode": map_payload.get("mode"),
            "source_summary": map_payload.get("summary"),
            "source": map_payload.get("source"),
        },
        "export_notes": {
            "preserved_in_map_only": [
                "keyframes",
                "trajectory",
                "frontiers",
                "decision_log",
                "guardrail_events",
                "nav2_debug_artifacts",
                "raw_semantic_evidence",
            ],
            "distilled_fields": [
                "regions",
                "named_places",
                "occupancy",
                "manual_occupancy_edits",
                "start_pose",
                "semantic named places",
            ],
        },
    }


def summarize_home_memory(memory: dict[str, Any]) -> str:
    regions = ", ".join(region.get("label", "") for region in memory.get("regions", []) if region.get("label"))
    places = ", ".join(place.get("name", "") for place in memory.get("places", []) if place.get("name"))
    objects = ", ".join(item.get("label", "") for item in memory.get("objects", []) if item.get("label"))
    return (
        f"Home memory `{memory.get('memory_id', 'unknown')}` has regions [{regions}], "
        f"places [{places}], and objects [{objects}]."
    )


def home_memory_agent_context(memory: dict[str, Any]) -> dict[str, Any]:
    """Return the compact memory shape that should be placed in agent context."""
    return {
        "memory_id": memory.get("memory_id"),
        "schema_version": memory.get("schema_version"),
        "frame": memory.get("frame", "map"),
        "approved": bool(memory.get("approved", False)),
        "start_pose": _json_clone(memory.get("start_pose")),
        "regions": [
            {
                "region_id": region.get("region_id"),
                "label": region.get("label"),
                "purpose": region.get("purpose", ""),
                "default_waypoints": [
                    _json_clone(waypoint)
                    for waypoint in region.get("default_waypoints", [])
                    if isinstance(waypoint, dict) and not _is_auto_center_waypoint(region, waypoint)
                ],
                "exploration": _json_clone(region.get("exploration") or {}),
                "adjacent_region_ids": list(region.get("adjacent_region_ids") or []),
            }
            for region in memory.get("regions", [])
            if isinstance(region, dict)
        ],
        "places": [
            {
                "name": place.get("name"),
                "pose": _json_clone(place.get("pose") or {}),
                "region_id": place.get("region_id"),
            }
            for place in memory.get("places", [])
            if isinstance(place, dict)
        ],
        "objects": [
            {
                "object_id": item.get("object_id"),
                "label": item.get("label"),
                "category": item.get("category"),
                "region_id": item.get("region_id"),
                "pose": _json_clone(item.get("pose") or {}),
                "approach_pose": _json_clone(item.get("approach_pose") or {}),
                "affordances": list(item.get("affordances") or []),
                "confidence": item.get("confidence"),
            }
            for item in memory.get("objects", [])
            if isinstance(item, dict)
        ],
        "skills": [
            {
                "skill_id": skill.get("skill_id"),
                "kind": skill.get("kind"),
                "target_labels": list(skill.get("target_labels") or []),
                "target_categories": list(skill.get("target_categories") or []),
                "required_pose_class": skill.get("required_pose_class"),
                "required_observations": list(skill.get("required_observations") or []),
                "safety": _json_clone(skill.get("safety") or {}),
            }
            for skill in memory.get("skills", [])
            if isinstance(skill, dict)
        ],
    }


def home_memory_preview_map(memory: dict[str, Any]) -> dict[str, Any]:
    """Return geometry for UI previews without putting it in the agent prompt."""
    return {
        "memory_id": memory.get("memory_id"),
        "frame": memory.get("frame", "map"),
        "start_pose": _json_clone(memory.get("start_pose")),
        "occupancy": _json_clone(memory.get("occupancy") or {}),
        "manual_occupancy_edits": _json_clone(memory.get("manual_occupancy_edits") or {}),
        "regions": [
            {
                "region_id": region.get("region_id"),
                "label": region.get("label"),
                "polygon_2d": _json_clone(region.get("polygon_2d") or []),
                "exploration": _json_clone(region.get("exploration") or {}),
            }
            for region in memory.get("regions", [])
            if isinstance(region, dict)
        ],
    }


def resolve_home_memory_target(
    memory: dict[str, Any],
    name_or_label: str,
) -> dict[str, Any] | None:
    """Resolve a region/place/object name into a navigation pose from home memory."""
    query = _normalize_lookup_key(name_or_label)
    if not query:
        return None
    context = home_memory_agent_context(memory)
    candidates: list[dict[str, Any]] = []

    for place in context.get("places", []):
        label = str(place.get("name") or "")
        score = _lookup_score(query, label)
        pose = place.get("pose")
        if score > 0 and _has_xy_pose(pose):
            candidates.append(
                {
                    "target_type": "place",
                    "label": label,
                    "pose": _json_pose(pose),
                    "region_id": place.get("region_id"),
                    "confidence": score,
                    "source": "home_memory.places",
                }
            )

    for region in context.get("regions", []):
        label = str(region.get("label") or "")
        score = _lookup_score(query, label)
        pose = _first_pose(region.get("default_waypoints"))
        if score > 0 and _has_xy_pose(pose):
            candidates.append(
                {
                    "target_type": "region",
                    "label": label,
                    "pose": _json_pose(pose),
                    "region_id": region.get("region_id"),
                    "confidence": score,
                    "source": "home_memory.regions",
                }
            )

    for item in context.get("objects", []):
        label = str(item.get("label") or "")
        score = _lookup_score(query, label)
        pose = item.get("approach_pose") or item.get("pose")
        if score > 0 and _has_xy_pose(pose):
            candidates.append(
                {
                    "target_type": "object",
                    "label": label,
                    "pose": _json_pose(pose),
                    "region_id": item.get("region_id"),
                    "confidence": score,
                    "source": "home_memory.objects",
                    "affordances": list(item.get("affordances") or []),
                }
            )

    if not candidates:
        return None
    return max(candidates, key=lambda item: (float(item.get("confidence", 0.0)), _target_priority(item)))


def resolve_region_navigation_goal(
    memory: dict[str, Any],
    name_or_label: str,
    *,
    current_pose: dict[str, Any] | None = None,
    min_clearance_m: float = DEFAULT_NAVIGATION_CLEARANCE_M,
    waypoint_horizon_m: float = DEFAULT_NAVIGATION_WAYPOINT_HORIZON_M,
    waypoint_breakdown_enabled: bool = True,
    navigation_purpose: str = "",
    object_label: str = "",
    exploration_constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve a semantic region into a concrete known-free navigation pose."""
    region = _best_region_match(memory, name_or_label)
    if region is None:
        return {
            "tool": "resolve_region_navigation_goal",
            "status": "not_found",
            "target_label": name_or_label,
            "reason": "No matching region label was found in home memory.",
        }
    if _is_object_search_navigation_purpose(navigation_purpose):
        return _resolve_region_search_entry_navigation_goal(
            memory,
            region,
            name_or_label,
            current_pose=current_pose,
            min_clearance_m=min_clearance_m,
            waypoint_horizon_m=waypoint_horizon_m,
            waypoint_breakdown_enabled=waypoint_breakdown_enabled,
            object_label=object_label,
            exploration_constraints=exploration_constraints or {},
        )
    explicit = _first_pose(region.get("default_waypoints"))
    if explicit is not None and not _is_auto_center_waypoint(region, explicit):
        return {
            "tool": "resolve_region_navigation_goal",
            "status": "succeeded",
            "target_label": region.get("label") or name_or_label,
            "target_type": "region",
            "region_id": region.get("region_id"),
            "goal_pose": _json_pose(explicit),
            "next_waypoint": _explicit_waypoint_payload(
                region.get("label") or name_or_label,
                region.get("region_id"),
                _json_pose(explicit),
                current_pose,
                waypoint_horizon_m,
            ),
            "waypoint_breakdown_enabled": bool(waypoint_breakdown_enabled),
            "source": "region.default_waypoints",
            "candidate_count": 1,
            "clearance_m": None,
            "path": [],
        }

    polygon = region.get("polygon_2d")
    if not isinstance(polygon, list) or len(polygon) < 3:
        return {
            "tool": "resolve_region_navigation_goal",
            "status": "blocked",
            "target_label": region.get("label") or name_or_label,
            "target_type": "region",
            "region_id": region.get("region_id"),
            "reason": "The matched region has no polygon and no explicit waypoint.",
        }

    grid = _memory_occupancy_grid(memory)
    if not grid["free"]:
        return {
            "tool": "resolve_region_navigation_goal",
            "status": "blocked",
            "target_label": region.get("label") or name_or_label,
            "target_type": "region",
            "region_id": region.get("region_id"),
            "reason": "Home memory has no known-free occupancy cells.",
        }

    resolution = float(grid["resolution"])
    region_free_cells = [
        cell
        for cell in grid["free"]
        if _point_in_polygon(
            grid["origin_x"] + (cell[0] + 0.5) * resolution,
            grid["origin_y"] + (cell[1] + 0.5) * resolution,
            polygon,
        )
    ]
    if not region_free_cells:
        return {
            "tool": "resolve_region_navigation_goal",
            "status": "blocked",
            "target_label": region.get("label") or name_or_label,
            "target_type": "region",
            "region_id": region.get("region_id"),
            "reason": "No known-free occupancy cells were found inside the region.",
        }

    footprint_radius_m = DEFAULT_ROBOT_FOOTPRINT_RADIUS_M
    lateral_clearance_m = XLEROBOT_FOOTPRINT_WIDTH_M / 2.0 + NAV2_FOOTPRINT_PADDING_M
    safety_gap_m = max(float(min_clearance_m) - lateral_clearance_m, 0.0)
    footprint_safe_cells = _footprint_safe_cells(grid, min_clearance_m)
    if not footprint_safe_cells:
        return {
            "tool": "resolve_region_navigation_goal",
            "status": "blocked",
            "target_label": region.get("label") or name_or_label,
            "target_type": "region",
            "region_id": region.get("region_id"),
            "reason": "No known-free cells satisfy the robot footprint clearance.",
            "robot_footprint_radius_m": round(footprint_radius_m, 3),
            "robot_lateral_clearance_m": round(lateral_clearance_m, 3),
            "safety_gap_m": round(safety_gap_m, 3),
            "min_clearance_m": round(float(min_clearance_m), 3),
        }

    start_cell = _cell_for_pose(current_pose, grid) if current_pose else None
    reachable = _reachable_cells(grid, start_cell, footprint_safe_cells) if start_cell is not None else set(footprint_safe_cells)
    candidates = [cell for cell in region_free_cells if cell in reachable]
    if not candidates and start_cell is None:
        candidates = [cell for cell in region_free_cells if cell in footprint_safe_cells]

    inside_goal_candidate_count = len(candidates)
    approach_candidates: list[tuple[int, int]] = []
    goal_selection = "inside_region_clearance"
    if _should_use_inside_edge_region_approach_goal(
        polygon,
        region_free_cell_count=len(region_free_cells),
        footprint_safe_candidate_count=inside_goal_candidate_count,
    ):
        approach_candidates = _inside_edge_region_approach_cells(
            grid,
            polygon,
            inside_candidates=candidates,
            min_clearance_m=min_clearance_m,
        )
        if approach_candidates:
            candidates = approach_candidates
            goal_selection = "inside_region_edge_approach"

    if not candidates and start_cell is not None:
        return {
            "tool": "resolve_region_navigation_goal",
            "status": "blocked",
            "target_label": region.get("label") or name_or_label,
            "target_type": "region",
            "region_id": region.get("region_id"),
            "reason": "No footprint-safe known-free cells inside the region are reachable from the current pose.",
            "candidate_count": len(region_free_cells),
            "reachable_candidate_count": 0,
            "footprint_safe_candidate_count": len([cell for cell in region_free_cells if cell in footprint_safe_cells]),
            "robot_footprint_radius_m": round(footprint_radius_m, 3),
            "robot_lateral_clearance_m": round(lateral_clearance_m, 3),
            "safety_gap_m": round(safety_gap_m, 3),
            "min_clearance_m": round(float(min_clearance_m), 3),
        }
    if not candidates:
        return {
            "tool": "resolve_region_navigation_goal",
            "status": "blocked",
            "target_label": region.get("label") or name_or_label,
            "target_type": "region",
            "region_id": region.get("region_id"),
            "reason": "No footprint-safe known-free cells were found inside the region.",
            "candidate_count": len(region_free_cells),
            "robot_footprint_radius_m": round(footprint_radius_m, 3),
            "robot_lateral_clearance_m": round(lateral_clearance_m, 3),
            "safety_gap_m": round(safety_gap_m, 3),
            "min_clearance_m": round(float(min_clearance_m), 3),
        }

    polygon_axes = _polygon_axes(polygon) if goal_selection == "inside_region_edge_approach" else None
    scored = []
    for cell in candidates:
        clearance_m = _cell_clearance_m(cell, grid)
        path_distance_m = (
            float(_path_distance_cells(start_cell, cell, reachable)) * float(grid["resolution"])
            if start_cell is not None and cell in reachable
            else 0.0
        )
        if goal_selection == "inside_region_edge_approach":
            point_x, point_y = _cell_center_xy(cell, grid)
            edge_distance_m = _point_to_polygon_edge_distance_m(point_x, point_y, polygon)
            ideal_standoff_m = max(0.20, min(float(min_clearance_m) * 0.70, 0.32))
            standoff_error_m = abs(edge_distance_m - ideal_standoff_m)
            center_x, center_y = polygon_axes["center"] if polygon_axes else (point_x, point_y)
            major_axis = polygon_axes["major"] if polygon_axes else (1.0, 0.0)
            major_center_error_m = abs(_dot(point_x - center_x, point_y - center_y, major_axis))
            score = (
                clearance_m * 0.45
                - path_distance_m * 0.30
                - standoff_error_m * 1.25
                - major_center_error_m * 0.50
            )
            scored.append((score, clearance_m, -path_distance_m, -standoff_error_m, cell))
        else:
            scored.append((clearance_m, clearance_m, -path_distance_m, 0.0, cell))

    scored.sort(reverse=True)
    _, best_clearance, _, _, best = scored[0]
    path_cells = _centered_path_cells(grid, start_cell, best, footprint_safe_cells) if start_cell is not None and best in reachable else []
    path_clearance = min((_cell_clearance_m(cell, grid) for cell in path_cells), default=best_clearance)
    goal_pose = {
        "x": round(grid["origin_x"] + (best[0] + 0.5) * resolution, 3),
        "y": round(grid["origin_y"] + (best[1] + 0.5) * resolution, 3),
        "yaw": _resolved_goal_yaw(current_pose),
    }
    path_length_m = _path_length_m(_path_points_for_waypoint(path_cells, grid, current_pose, goal_pose))
    if waypoint_breakdown_enabled:
        next_waypoint = _next_waypoint_payload(
            path_cells,
            grid,
            current_pose,
            goal_pose,
            waypoint_horizon_m,
            target_label=region.get("label") or name_or_label,
            region_id=region.get("region_id"),
        )
        centerline_waypoints = _centerline_waypoints_payload(
            path_cells,
            grid,
            current_pose,
            goal_pose,
            waypoint_horizon_m,
            target_label=region.get("label") or name_or_label,
            region_id=region.get("region_id"),
        )
    else:
        next_waypoint = _direct_goal_waypoint_payload(
            path_cells,
            grid,
            current_pose,
            goal_pose,
            waypoint_horizon_m,
            target_label=region.get("label") or name_or_label,
            region_id=region.get("region_id"),
        )
        centerline_waypoints = [dict(next_waypoint)]
    status = "succeeded" if best_clearance >= min_clearance_m else "low_clearance"
    return {
        "tool": "resolve_region_navigation_goal",
        "status": status,
        "target_label": region.get("label") or name_or_label,
        "target_type": "region",
        "region_id": region.get("region_id"),
        "goal_pose": goal_pose,
        "next_waypoint": next_waypoint,
        "centerline_waypoints": centerline_waypoints,
        "waypoint_breakdown_enabled": bool(waypoint_breakdown_enabled),
        "source": (
            "home_memory.occupancy_region_inside_edge_approach"
            if goal_selection == "inside_region_edge_approach"
            else "home_memory.occupancy_region_free_space"
        ),
        "candidate_count": len(region_free_cells),
        "reachable_candidate_count": len([cell for cell in region_free_cells if cell in reachable]),
        "footprint_safe_candidate_count": len([cell for cell in region_free_cells if cell in footprint_safe_cells]),
        "clearance_m": round(best_clearance, 3),
        "path_clearance_m": round(path_clearance, 3),
        "min_clearance_m": round(float(min_clearance_m), 3),
        "robot_footprint_radius_m": round(footprint_radius_m, 3),
        "robot_lateral_clearance_m": round(lateral_clearance_m, 3),
        "safety_gap_m": round(safety_gap_m, 3),
        "path_strategy": "footprint_eroded_centerline_weighted_grid",
        "goal_selection": goal_selection,
        "inside_goal_candidate_count": inside_goal_candidate_count,
        "approach_goal_candidate_count": len(approach_candidates),
        "path_length_m": round(path_length_m, 3),
        "path": _path_payload(path_cells, grid),
    }


def _resolve_region_search_entry_navigation_goal(
    memory: dict[str, Any],
    region: dict[str, Any],
    name_or_label: str,
    *,
    current_pose: dict[str, Any] | None,
    min_clearance_m: float,
    waypoint_horizon_m: float,
    waypoint_breakdown_enabled: bool,
    object_label: str,
    exploration_constraints: dict[str, Any],
) -> dict[str, Any]:
    region_label = str(region.get("label") or name_or_label)
    plan = plan_region_exploration(
        memory,
        region_label,
        fov_deg=exploration_constraints.get("fov_deg"),
        max_stops=exploration_constraints.get("max_stops"),
        shots_per_stop=exploration_constraints.get("shots_per_stop"),
        min_clearance_m=min_clearance_m,
        boundary_margin_m=_bounded_float(
            exploration_constraints.get("boundary_margin_m"),
            DEFAULT_REGION_EXPLORATION_BOUNDARY_MARGIN_M,
            minimum=0.0,
            maximum=5.0,
        ),
        min_stop_separation_m=exploration_constraints.get("min_stop_separation_m"),
    )
    stops = plan.get("stops") if isinstance(plan.get("stops"), list) else []
    if plan.get("status") != "succeeded" or not stops or not isinstance(stops[0], dict):
        return {
            "tool": "resolve_region_navigation_goal",
            "status": "blocked",
            "target_label": region_label,
            "target_type": "region",
            "region_id": region.get("region_id"),
            "navigation_purpose": "object_search",
            "object_label": object_label,
            "reason": plan.get("reason") or "No exploration stop is available for object-search navigation.",
            "search_plan": plan,
        }

    grid = _memory_occupancy_grid(memory)
    footprint_safe_cells = _footprint_safe_cells(grid, min_clearance_m)
    if not footprint_safe_cells:
        return {
            "tool": "resolve_region_navigation_goal",
            "status": "blocked",
            "target_label": region_label,
            "target_type": "region",
            "region_id": region.get("region_id"),
            "navigation_purpose": "object_search",
            "object_label": object_label,
            "reason": "No known-free cells satisfy the robot footprint clearance.",
            "search_plan": plan,
        }
    start_cell = _cell_for_pose(current_pose, grid) if current_pose else None
    reachable = _reachable_cells(grid, start_cell, footprint_safe_cells) if start_cell is not None else set(footprint_safe_cells)

    stop: dict[str, Any] | None = None
    stop_index = 0
    stop_pose: dict[str, Any] | None = None
    goal_cell: tuple[int, int] | None = None
    for index, candidate in enumerate(stops, start=1):
        if not isinstance(candidate, dict):
            continue
        candidate_pose = candidate.get("pose") if isinstance(candidate.get("pose"), dict) else None
        candidate_cell = _cell_for_pose(candidate_pose, grid) if candidate_pose is not None else None
        if candidate_pose is not None and candidate_cell is not None and candidate_cell in reachable:
            stop = candidate
            stop_index = index
            stop_pose = candidate_pose
            goal_cell = candidate_cell
            break

    if stop is None or stop_pose is None or goal_cell is None:
        return {
            "tool": "resolve_region_navigation_goal",
            "status": "blocked",
            "target_label": region_label,
            "target_type": "region",
            "region_id": region.get("region_id"),
            "navigation_purpose": "object_search",
            "object_label": object_label,
            "reason": "No exploration stop is footprint-safe and reachable from the current pose.",
            "search_plan": plan,
            "first_stop": stops[0],
        }

    goal_pose = _json_pose(stop_pose)
    path_cells = _centered_path_cells(grid, start_cell, goal_cell, footprint_safe_cells) if start_cell is not None else []
    path_clearance = min((_cell_clearance_m(cell, grid) for cell in path_cells), default=_cell_clearance_m(goal_cell, grid))
    path_length_m = _path_length_m(_path_points_for_waypoint(path_cells, grid, current_pose, goal_pose))
    waypoint_builder = _next_waypoint_payload if waypoint_breakdown_enabled else _direct_goal_waypoint_payload
    next_waypoint = waypoint_builder(
        path_cells,
        grid,
        current_pose,
        goal_pose,
        waypoint_horizon_m,
        target_label=region_label,
        region_id=region.get("region_id"),
    )
    return {
        "tool": "resolve_region_navigation_goal",
        "status": "succeeded" if path_clearance >= min_clearance_m else "low_clearance",
        "target_label": region_label,
        "target_type": "region",
        "region_id": region.get("region_id"),
        "navigation_purpose": "object_search",
        "object_label": object_label,
        "goal_pose": goal_pose,
        "next_waypoint": next_waypoint,
        "waypoint_breakdown_enabled": bool(waypoint_breakdown_enabled),
        "source": "home_memory.region_search_entry",
        "goal_selection": "region_search_entry",
        "search_stop": stop,
        "search_stop_index": stop_index,
        "search_plan": plan,
        "clearance_m": round(_cell_clearance_m(goal_cell, grid), 3),
        "path_clearance_m": round(path_clearance, 3),
        "min_clearance_m": round(float(min_clearance_m), 3),
        "path_strategy": "footprint_eroded_centerline_weighted_grid",
        "path_length_m": round(path_length_m, 3),
        "path": _path_payload(path_cells, grid),
    }


def plan_region_exploration(
    memory: dict[str, Any],
    region_label: str,
    *,
    fov_deg: float | None = None,
    max_stops: int | None = None,
    shots_per_stop: int | None = None,
    min_clearance_m: float = DEFAULT_NAVIGATION_CLEARANCE_M,
    boundary_margin_m: float = DEFAULT_REGION_EXPLORATION_BOUNDARY_MARGIN_M,
    min_range_m: float = DEFAULT_REGION_EXPLORATION_MIN_RANGE_M,
    max_range_m: float = DEFAULT_REGION_EXPLORATION_MAX_RANGE_M,
    min_stop_separation_m: float | None = None,
) -> dict[str, Any]:
    """Generate simple visual-search stops and 65-degree shots for a known region."""
    region = _best_region_match(memory, region_label)
    if region is None:
        return {
            "tool": "plan_region_exploration",
            "status": "not_found",
            "target_label": region_label,
            "reason": "No matching region label was found in home memory.",
        }
    polygon = region.get("polygon_2d")
    if not isinstance(polygon, list) or len(polygon) < 3:
        return {
            "tool": "plan_region_exploration",
            "status": "blocked",
            "target_label": region.get("label") or region_label,
            "region_id": region.get("region_id"),
            "reason": "The matched region has no polygon for visual sweep planning.",
        }

    grid = _memory_occupancy_grid(memory)
    if not grid["free"]:
        return {
            "tool": "plan_region_exploration",
            "status": "blocked",
            "target_label": region.get("label") or region_label,
            "region_id": region.get("region_id"),
            "reason": "Home memory has no known-free occupancy cells.",
        }

    config = region.get("exploration") if isinstance(region.get("exploration"), dict) else {}
    fov_deg = _bounded_float(
        fov_deg if fov_deg is not None else config.get("fov_deg"),
        DEFAULT_REGION_EXPLORATION_FOV_DEG,
        minimum=20.0,
        maximum=160.0,
    )
    shots_per_stop = _bounded_int(
        shots_per_stop if shots_per_stop is not None else config.get("shots_per_stop"),
        DEFAULT_REGION_EXPLORATION_SHOTS_PER_STOP,
        minimum=1,
        maximum=8,
    )
    max_stops = _bounded_int(
        max_stops if max_stops is not None else config.get("max_stops"),
        _default_region_exploration_stop_count(polygon),
        minimum=1,
        maximum=8,
    )
    min_stop_separation_m = _bounded_float(
        min_stop_separation_m if min_stop_separation_m is not None else config.get("min_stop_separation_m"),
        DEFAULT_REGION_EXPLORATION_MIN_STOP_SEPARATION_M,
        minimum=0.0,
        maximum=2.0,
    )

    resolution = float(grid["resolution"])
    region_free_cells = [
        cell
        for cell in grid["free"]
        if _point_in_polygon(
            grid["origin_x"] + (cell[0] + 0.5) * resolution,
            grid["origin_y"] + (cell[1] + 0.5) * resolution,
            polygon,
        )
    ]
    if not region_free_cells:
        return {
            "tool": "plan_region_exploration",
            "status": "blocked",
            "target_label": region.get("label") or region_label,
            "region_id": region.get("region_id"),
            "reason": "No known-free occupancy cells were found inside the region.",
        }

    footprint_safe_cells = _footprint_safe_cells(grid, min_clearance_m)
    stop_candidates = [cell for cell in region_free_cells if cell in footprint_safe_cells]
    if not stop_candidates:
        stop_candidates = list(region_free_cells)
    boundary_cells = _region_exploration_boundary_cells(
        grid,
        polygon,
        margin_m=float(boundary_margin_m),
    )
    axes = _polygon_axes(polygon)
    stop_cells = _select_region_exploration_stops(
        grid,
        stop_candidates,
        axes,
        max_stops=max_stops,
        min_stop_separation_m=min_stop_separation_m,
    )

    region_area_cells = set(region_free_cells)
    covered_boundary_cells: set[tuple[int, int]] = set()
    covered_area_cells: set[tuple[int, int]] = set()
    stops: list[dict[str, Any]] = []
    for index, cell in enumerate(stop_cells, start=1):
        pose = _cell_center_pose(cell, grid)
        shot_plans, stop_boundary_covered, stop_area_covered = _region_exploration_shots_for_stop(
            grid,
            cell,
            boundary_cells,
            region_area_cells,
            already_covered_boundary_cells=covered_boundary_cells,
            already_covered_area_cells=covered_area_cells,
            fov_deg=fov_deg,
            shots_per_stop=shots_per_stop,
            min_range_m=float(min_range_m),
            max_range_m=float(max_range_m),
        )
        if shot_plans:
            pose["yaw"] = shot_plans[0]["yaw"]
        covered_boundary_cells.update(stop_boundary_covered)
        covered_area_cells.update(stop_area_covered)
        stops.append(
            {
                "stop_id": f"{_slug(str(region.get('label') or region_label))}_stop_{index}",
                "name": f"stop_{index}",
                "pose": pose,
                "clearance_m": round(_cell_clearance_m(cell, grid), 3),
                "shots": shot_plans,
            }
        )

    boundary_count = len(boundary_cells)
    covered_boundary_count = len(covered_boundary_cells)
    boundary_coverage_ratio = covered_boundary_count / boundary_count if boundary_count else 0.0
    area_count = len(region_area_cells)
    covered_area_count = len(covered_area_cells)
    area_coverage_ratio = covered_area_count / area_count if area_count else 0.0
    return {
        "tool": "plan_region_exploration",
        "status": "succeeded",
        "target_label": region.get("label") or region_label,
        "target_type": "region",
        "region_id": region.get("region_id"),
        "strategy": "occupancy_boundary_visual_sweep",
        "fov_deg": round(float(fov_deg), 3),
        "max_stops": int(max_stops),
        "shots_per_stop": int(shots_per_stop),
        "min_clearance_m": round(float(min_clearance_m), 3),
        "min_stop_separation_m": round(float(min_stop_separation_m), 3),
        "boundary_margin_m": round(float(boundary_margin_m), 3),
        "range_m": {
            "min": round(float(min_range_m), 3),
            "max": round(float(max_range_m), 3),
        },
        "coverage": {
            "boundary_cell_count": boundary_count,
            "covered_boundary_cell_count": covered_boundary_count,
            "coverage_ratio": round(boundary_coverage_ratio, 3),
            "region_area_cell_count": area_count,
            "covered_region_area_cell_count": covered_area_count,
            "region_area_coverage_ratio": round(area_coverage_ratio, 3),
        },
        "stops": stops,
    }


def resolve_direct_navigation_fallback(
    memory: dict[str, Any],
    start_pose: dict[str, Any] | None,
    goal_pose: dict[str, Any],
    *,
    min_clearance_m: float = DEFAULT_NAVIGATION_CLEARANCE_M,
    max_distance_m: float = DEFAULT_DIRECT_NAVIGATION_FALLBACK_MAX_DISTANCE_M,
) -> dict[str, Any]:
    """Check whether a short direct local motion is safe in the saved occupancy map."""
    start = _json_pose(start_pose or {})
    goal = _json_pose(goal_pose)
    distance_m = _pose_distance_m(start, goal)
    max_distance_m = max(float(max_distance_m), 0.0)
    if distance_m <= 1e-6:
        return {
            "tool": "resolve_direct_navigation_fallback",
            "status": "succeeded",
            "reason": "Already at the requested waypoint.",
            "start_pose": start,
            "goal_pose": goal,
            "distance_m": 0.0,
            "max_distance_m": round(max_distance_m, 3),
            "min_clearance_m": round(float(min_clearance_m), 3),
            "path": [start],
        }
    if distance_m > max_distance_m:
        return {
            "tool": "resolve_direct_navigation_fallback",
            "status": "too_far",
            "reason": "Direct primitive fallback is only allowed for short local moves.",
            "start_pose": start,
            "goal_pose": goal,
            "distance_m": round(distance_m, 3),
            "max_distance_m": round(max_distance_m, 3),
            "min_clearance_m": round(float(min_clearance_m), 3),
        }
    grid = _memory_occupancy_grid(memory)
    if not grid["free"]:
        return {
            "tool": "resolve_direct_navigation_fallback",
            "status": "blocked",
            "reason": "Home memory has no known-free occupancy cells.",
            "start_pose": start,
            "goal_pose": goal,
            "distance_m": round(distance_m, 3),
            "max_distance_m": round(max_distance_m, 3),
            "min_clearance_m": round(float(min_clearance_m), 3),
        }
    start_cell = _cell_for_pose(start, grid)
    goal_cell = _cell_for_pose(goal, grid)
    if start_cell is None or goal_cell is None:
        return {
            "tool": "resolve_direct_navigation_fallback",
            "status": "blocked",
            "reason": "Could not project start or goal pose into the occupancy grid.",
            "start_pose": start,
            "goal_pose": goal,
            "distance_m": round(distance_m, 3),
            "max_distance_m": round(max_distance_m, 3),
            "min_clearance_m": round(float(min_clearance_m), 3),
        }
    footprint_safe_cells = _footprint_safe_cells(grid, min_clearance_m)
    line_cells = _bresenham_cells(start_cell, goal_cell)
    unsafe_cells = [cell for cell in line_cells if cell not in footprint_safe_cells]
    if unsafe_cells:
        return {
            "tool": "resolve_direct_navigation_fallback",
            "status": "blocked",
            "reason": "Direct line is not footprint-clear in the saved occupancy map.",
            "start_pose": start,
            "goal_pose": goal,
            "distance_m": round(distance_m, 3),
            "max_distance_m": round(max_distance_m, 3),
            "min_clearance_m": round(float(min_clearance_m), 3),
            "checked_cell_count": len(line_cells),
            "unsafe_cell_count": len(unsafe_cells),
            "first_unsafe_cell": {"cell_x": unsafe_cells[0][0], "cell_y": unsafe_cells[0][1]},
            "path": _path_payload(line_cells, grid),
        }
    clearances = [_cell_clearance_m(cell, grid) for cell in line_cells]
    return {
        "tool": "resolve_direct_navigation_fallback",
        "status": "succeeded",
        "reason": "Short direct line is footprint-clear in the saved occupancy map.",
        "start_pose": start,
        "goal_pose": goal,
        "distance_m": round(distance_m, 3),
        "max_distance_m": round(max_distance_m, 3),
        "min_clearance_m": round(float(min_clearance_m), 3),
        "checked_cell_count": len(line_cells),
        "path_clearance_m": round(min(clearances), 3) if clearances else None,
        "path": _path_payload(line_cells, grid),
    }


def resolve_local_clearance_recovery(
    memory: dict[str, Any],
    start_pose: dict[str, Any] | None,
    goal_pose: dict[str, Any],
    *,
    min_clearance_m: float = DEFAULT_NAVIGATION_CLEARANCE_M,
    max_distance_m: float = 0.45,
) -> dict[str, Any]:
    """Suggest a short local move to a nearby footprint-safe pose before retrying Nav2."""
    start = _json_pose(start_pose or {})
    goal = _json_pose(goal_pose)
    grid = _memory_occupancy_grid(memory)
    if not grid["free"]:
        return {
            "tool": "resolve_local_clearance_recovery",
            "status": "blocked",
            "reason": "Home memory has no known-free occupancy cells.",
            "start_pose": start,
            "goal_pose": goal,
            "min_clearance_m": round(float(min_clearance_m), 3),
            "max_distance_m": round(float(max_distance_m), 3),
        }
    start_cell = _cell_for_pose(start, grid)
    if start_cell is None:
        return {
            "tool": "resolve_local_clearance_recovery",
            "status": "blocked",
            "reason": "Could not project current pose into the occupancy grid.",
            "start_pose": start,
            "goal_pose": goal,
            "min_clearance_m": round(float(min_clearance_m), 3),
            "max_distance_m": round(float(max_distance_m), 3),
        }
    footprint_safe_cells = _footprint_safe_cells(grid, min_clearance_m)
    resolution = float(grid["resolution"])
    max_distance_m = max(float(max_distance_m), resolution)
    max_radius_cells = max(1, int(math.ceil(max_distance_m / resolution)))
    start_goal_distance = _pose_distance_m(start, goal)
    start_clearance = _cell_clearance_m(start_cell, grid) if start_cell in grid["free"] else 0.0
    candidates: list[tuple[float, float, float, tuple[int, int], list[tuple[int, int]]]] = []
    for dx in range(-max_radius_cells, max_radius_cells + 1):
        for dy in range(-max_radius_cells, max_radius_cells + 1):
            cell = (start_cell[0] + dx, start_cell[1] + dy)
            if cell == start_cell or cell not in footprint_safe_cells:
                continue
            point = _path_payload([cell], grid)[0]
            distance_from_start = _pose_distance_m(start, point)
            if distance_from_start > max_distance_m or distance_from_start < resolution * 0.35:
                continue
            line = _bresenham_cells(start_cell, cell)
            unsafe_line = [line_cell for line_cell in line[1:] if line_cell not in footprint_safe_cells]
            if unsafe_line:
                continue
            clearance = _cell_clearance_m(cell, grid)
            progress = start_goal_distance - _pose_distance_m(point, goal)
            score = clearance * 2.0 + max(progress, -0.25) * 0.75 - distance_from_start * 0.25
            candidates.append((score, clearance, -distance_from_start, cell, line))
    if not candidates:
        return {
            "tool": "resolve_local_clearance_recovery",
            "status": "blocked",
            "reason": "No nearby footprint-safe recovery pose was found in saved occupancy.",
            "start_pose": start,
            "goal_pose": goal,
            "start_clearance_m": round(start_clearance, 3),
            "min_clearance_m": round(float(min_clearance_m), 3),
            "max_distance_m": round(max_distance_m, 3),
        }
    _, best_clearance, neg_distance, best_cell, best_line = max(candidates)
    recovery_xy = _path_payload([best_cell], grid)[0]
    recovery_yaw = math.atan2(goal["y"] - recovery_xy["y"], goal["x"] - recovery_xy["x"])
    recovery_pose = _json_pose({**recovery_xy, "yaw": recovery_yaw})
    return {
        "tool": "resolve_local_clearance_recovery",
        "status": "succeeded",
        "reason": "Move to a nearby footprint-safe pose to gain clearance before retrying Nav2.",
        "suggested_tool": "micro_adjust_to_pose",
        "recovery_pose": recovery_pose,
        "start_pose": start,
        "goal_pose": goal,
        "distance_m": round(-neg_distance, 3),
        "start_clearance_m": round(start_clearance, 3),
        "recovery_clearance_m": round(best_clearance, 3),
        "min_clearance_m": round(float(min_clearance_m), 3),
        "max_distance_m": round(max_distance_m, 3),
        "path": _path_payload(best_line, grid),
        "follow_up": "Call relocalize_here, then retry the original navigate_to_waypoint or region exploration step.",
    }


def resolve_object_surface_approach_pose(
    memory: dict[str, Any],
    current_pose: dict[str, Any] | None,
    object_pose: dict[str, Any],
    *,
    min_clearance_m: float = DEFAULT_NAVIGATION_CLEARANCE_M,
    standoff_m: float = 0.65,
    search_beyond_m: float = 0.9,
    support_radius_m: float = 0.75,
    max_alignment_distance_m: float = 1.0,
    yaw_tolerance_deg: float = 18.0,
    distance_tolerance_m: float = 0.12,
) -> dict[str, Any]:
    """Resolve a body-friendly standoff pose perpendicular to the occupied support surface."""
    start = _json_pose(current_pose or {})
    try:
        obj = {"x": float(object_pose["x"]), "y": float(object_pose["y"]), "yaw": 0.0}
    except Exception:
        return {
            "tool": "resolve_object_surface_approach_pose",
            "status": "blocked",
            "reason": "Object map pose is required for surface approach alignment.",
            "current_pose": start,
        }
    grid = _memory_occupancy_grid(memory)
    if not grid["free"] or not grid["occupied"]:
        return {
            "tool": "resolve_object_surface_approach_pose",
            "status": "unavailable",
            "reason": "Home memory needs known-free and occupied cells to infer an object support surface.",
            "current_pose": start,
            "object_pose": obj,
        }
    start_cell = _cell_for_pose(start, grid)
    if start_cell is None:
        return {
            "tool": "resolve_object_surface_approach_pose",
            "status": "blocked",
            "reason": "Could not project current pose into the occupancy grid.",
            "current_pose": start,
            "object_pose": obj,
        }
    dx = obj["x"] - start["x"]
    dy = obj["y"] - start["y"]
    distance_to_object_m = math.hypot(dx, dy)
    if distance_to_object_m <= 1e-6:
        return {
            "tool": "resolve_object_surface_approach_pose",
            "status": "blocked",
            "reason": "Object pose overlaps current pose; cannot infer a sightline.",
            "current_pose": start,
            "object_pose": obj,
        }
    direction = (dx / distance_to_object_m, dy / distance_to_object_m)
    ray_end = {
        "x": start["x"] + direction[0] * (distance_to_object_m + max(float(search_beyond_m), 0.0)),
        "y": start["y"] + direction[1] * (distance_to_object_m + max(float(search_beyond_m), 0.0)),
    }
    ray_end_cell = _cell_for_pose(ray_end, grid)
    ray_cells = _bresenham_cells(start_cell, ray_end_cell) if ray_end_cell is not None else []
    min_skip_cells = max(1, int(round(0.20 / float(grid["resolution"]))))
    hit_cell = next((cell for index, cell in enumerate(ray_cells[min_skip_cells:], start=min_skip_cells) if cell in grid["occupied"]), None)
    if hit_cell is None:
        hit_cell = _nearest_occupied_cell_to_point(grid, obj["x"], obj["y"], max_radius_m=max(float(support_radius_m), float(grid["resolution"])))
    if hit_cell is None:
        return {
            "tool": "resolve_object_surface_approach_pose",
            "status": "unavailable",
            "reason": "No occupied support surface was found along the centered object sightline.",
            "current_pose": start,
            "object_pose": obj,
            "ray": _path_payload(ray_cells, grid),
        }

    hit_point = _cell_center_pose(hit_cell, grid)
    support_cells = _nearby_occupied_cells(grid, hit_cell, radius_m=max(float(support_radius_m), float(grid["resolution"]) * 2.0))
    tangent = _fit_surface_tangent(grid, support_cells, fallback=(-direction[1], direction[0]))
    normal_a = (-tangent[1], tangent[0])
    to_robot = (start["x"] - hit_point["x"], start["y"] - hit_point["y"])
    normal = normal_a if _dot2(normal_a, to_robot) >= 0.0 else (-normal_a[0], -normal_a[1])
    normal = _normalize2(normal) or normal_a
    tangent = _normalize2(tangent) or (1.0, 0.0)

    footprint_safe_cells = _footprint_safe_cells(grid, min_clearance_m)
    desired_standoff_m = max(float(standoff_m), float(min_clearance_m))
    candidate = _surface_approach_candidate(
        grid=grid,
        start_cell=start_cell,
        footprint_safe_cells=footprint_safe_cells,
        hit_point=hit_point,
        tangent=tangent,
        normal=normal,
        desired_standoff_m=desired_standoff_m,
        min_clearance_m=min_clearance_m,
    )
    if candidate is None:
        return {
            "tool": "resolve_object_surface_approach_pose",
            "status": "blocked",
            "reason": "Could not compute a geometric standoff pose in front of the occupied support surface.",
            "current_pose": start,
            "object_pose": obj,
            "support_surface": _support_surface_payload(hit_point, tangent, normal, support_cells, grid),
            "ray": _path_payload(ray_cells, grid),
        }

    approach_pose = _json_pose(
        {
            "x": candidate["x"],
            "y": candidate["y"],
            "yaw": math.atan2(-normal[1], -normal[0]),
        }
    )
    distance_m = _pose_distance_m(start, approach_pose)
    yaw_delta_deg = math.degrees(_angle_delta_abs(float(start.get("yaw", 0.0) or 0.0), approach_pose["yaw"]))
    max_alignment_distance_m = max(float(max_alignment_distance_m), 0.0)
    if distance_m > max_alignment_distance_m:
        return {
            "tool": "resolve_object_surface_approach_pose",
            "status": "too_far",
            "reason": "Surface approach pose is farther than the allowed local alignment distance.",
            "current_pose": start,
            "object_pose": obj,
            "approach_pose": approach_pose,
            "distance_m": round(distance_m, 3),
            "max_alignment_distance_m": round(max_alignment_distance_m, 3),
            "yaw_delta_deg": round(yaw_delta_deg, 2),
            "support_surface": _support_surface_payload(hit_point, tangent, normal, support_cells, grid),
        }
    needs_alignment = distance_m > float(distance_tolerance_m) or yaw_delta_deg > float(yaw_tolerance_deg)
    return {
        "tool": "resolve_object_surface_approach_pose",
        "status": "succeeded",
        "reason": (
            "Resolved a perpendicular standoff pose from occupied support-surface geometry."
            if needs_alignment
            else "Current pose is already close to the perpendicular surface standoff."
        ),
        "needs_alignment": bool(needs_alignment),
        "current_pose": start,
        "object_pose": obj,
        "approach_pose": approach_pose,
        "distance_m": round(distance_m, 3),
        "yaw_delta_deg": round(yaw_delta_deg, 2),
        "min_clearance_m": round(float(min_clearance_m), 3),
        "standoff_m": round(desired_standoff_m, 3),
        "path": _path_payload(candidate["line_cells"], grid),
        "standoff_clearance_check": {
            "status": "disabled",
            "reason": "Saved-map footprint-clear standoff gating is disabled; using geometric support-surface alignment.",
            "candidate_cell": (
                {"cell_x": candidate["cell"][0], "cell_y": candidate["cell"][1]} if candidate.get("cell") else None
            ),
            "candidate_clearance_m": (
                round(float(candidate["clearance_m"]), 3) if candidate.get("clearance_m") is not None else None
            ),
            "would_have_been_footprint_clear": bool(candidate.get("footprint_clear", False)),
            "unsafe_cell_count": int(candidate.get("unsafe_cell_count", 0) or 0),
            "first_unsafe_cell": (
                {"cell_x": candidate["first_unsafe_cell"][0], "cell_y": candidate["first_unsafe_cell"][1]}
                if candidate.get("first_unsafe_cell")
                else None
            ),
        },
        "support_surface": _support_surface_payload(hit_point, tangent, normal, support_cells, grid),
        "ray": _path_payload(ray_cells, grid),
    }


def known_home_memory_labels(memory: dict[str, Any]) -> list[str]:
    context = home_memory_agent_context(memory)
    labels: list[str] = []
    labels.extend(str(item.get("label")) for item in context.get("regions", []) if item.get("label"))
    labels.extend(str(item.get("name")) for item in context.get("places", []) if item.get("name"))
    labels.extend(str(item.get("label")) for item in context.get("objects", []) if item.get("label"))
    return sorted(set(labels), key=lambda item: item.lower())


def _start_pose_from_map(map_payload: dict[str, Any]) -> dict[str, Any] | None:
    explicit = map_payload.get("start_pose")
    if isinstance(explicit, dict):
        pose = explicit.get("pose") if isinstance(explicit.get("pose"), dict) else explicit
        return {
            "name": str(explicit.get("name") or "start"),
            "pose": _json_pose(pose),
            "fixed": bool(explicit.get("fixed", True)),
            "source": str(explicit.get("source") or "map_start_pose"),
        }
    for place in map_payload.get("named_places", []):
        if not isinstance(place, dict):
            continue
        name = str(place.get("name", "")).lower()
        if name in {"dock", "start", "home", "charging_dock", "robot_start"}:
            return {
                "name": str(place.get("name") or "start"),
                "pose": _json_pose(place.get("pose") or {}),
                "fixed": True,
                "source": "named_place",
            }
    return {
        "name": "start",
        "pose": {"x": 0.0, "y": 0.0, "yaw": 0.0},
        "fixed": True,
        "source": "default_map_origin",
    }


def _memory_region(region: dict[str, Any]) -> dict[str, Any]:
    label = str(region.get("label") or region.get("region_id") or "region")
    default_waypoints = [
        _json_pose(waypoint) | {"name": str(waypoint.get("name") or f"{label}_waypoint")}
        for waypoint in region.get("default_waypoints", [])
        if isinstance(waypoint, dict) and not _is_auto_center_waypoint(region, waypoint)
    ]
    return {
        "region_id": str(region.get("region_id") or label),
        "label": label,
        "confidence": float(region.get("confidence", 1.0) or 1.0),
        "polygon_2d": _json_clone(region.get("polygon_2d") or []),
        "centroid": _json_clone(region.get("centroid") or {}),
        "purpose": str(region.get("purpose") or ""),
        "entry_waypoints": _json_clone(region.get("entry_waypoints") or []),
        "scan_waypoints": _json_clone(region.get("scan_waypoints") or []),
        "default_waypoints": default_waypoints,
        "exploration": _json_clone(region.get("exploration") or {}),
        "adjacent_region_ids": list(region.get("adjacency") or region.get("adjacent_region_ids") or []),
        "evidence": _json_clone(region.get("evidence") or []),
    }


def _is_auto_center_waypoint(region: dict[str, Any], waypoint: dict[str, Any]) -> bool:
    name = str(waypoint.get("name") or "").lower().replace(" ", "_")
    label_center = f"{_slug(str(region.get('label') or region.get('region_id') or 'region'))}_center"
    if not (name.endswith("_center") or name.endswith("_entry") or name == "center" or name == label_center):
        return False
    try:
        waypoint_x = float(waypoint.get("x", 0.0) or 0.0)
        waypoint_y = float(waypoint.get("y", 0.0) or 0.0)
    except Exception:
        return False

    centers: list[tuple[float, float]] = []
    centroid = region.get("centroid")
    if isinstance(centroid, dict):
        try:
            centers.append((float(centroid.get("x", 0.0) or 0.0), float(centroid.get("y", 0.0) or 0.0)))
        except Exception:
            pass
    polygon = region.get("polygon_2d")
    if isinstance(polygon, list) and len(polygon) >= 3:
        axes = _polygon_axes(polygon)
        centers.append(axes["center"])

    return any(math.hypot(waypoint_x - center_x, waypoint_y - center_y) <= 0.20 for center_x, center_y in centers)


def _best_region_match(memory: dict[str, Any], name_or_label: str) -> dict[str, Any] | None:
    query = _normalize_lookup_key(name_or_label)
    if not query:
        return None
    matches: list[tuple[float, dict[str, Any]]] = []
    for region in memory.get("regions", []):
        if not isinstance(region, dict):
            continue
        label = str(region.get("label") or region.get("region_id") or "")
        score = _lookup_score(query, label)
        if score > 0:
            matches.append((score, region))
    if not matches:
        return None
    return max(matches, key=lambda item: item[0])[1]


def _memory_occupancy_grid(memory: dict[str, Any]) -> dict[str, Any]:
    occupancy = memory.get("occupancy") if isinstance(memory.get("occupancy"), dict) else {}
    resolution = float(occupancy.get("resolution", 0.25) or 0.25)
    bounds = occupancy.get("bounds") if isinstance(occupancy.get("bounds"), dict) else {}
    origin_x = float(bounds.get("min_x", 0.0) or 0.0)
    origin_y = float(bounds.get("min_y", 0.0) or 0.0)
    free: set[tuple[int, int]] = set()
    occupied: set[tuple[int, int]] = set()

    for item in occupancy.get("cells", []):
        if not isinstance(item, dict):
            continue
        try:
            cell = (
                int(math.floor((float(item.get("x", 0.0)) - origin_x) / resolution)),
                int(math.floor((float(item.get("y", 0.0)) - origin_y) / resolution)),
            )
        except Exception:
            continue
        state = str(item.get("state") or "").lower()
        if state == "free":
            free.add(cell)
            occupied.discard(cell)
        elif state == "occupied":
            occupied.add(cell)
            free.discard(cell)

    edits = memory.get("manual_occupancy_edits") if isinstance(memory.get("manual_occupancy_edits"), dict) else {}
    for item in edits.get("blocked_cells", []) or []:
        if isinstance(item, dict) and "cell_x" in item and "cell_y" in item:
            cell = (int(item["cell_x"]), int(item["cell_y"]))
            occupied.add(cell)
            free.discard(cell)
    for item in edits.get("cleared_cells", []) or []:
        if isinstance(item, dict) and "cell_x" in item and "cell_y" in item:
            cell = (int(item["cell_x"]), int(item["cell_y"]))
            free.add(cell)
            occupied.discard(cell)

    return {
        "resolution": resolution,
        "origin_x": origin_x,
        "origin_y": origin_y,
        "free": free,
        "occupied": occupied,
    }


def _cell_for_pose(pose: dict[str, Any] | None, grid: dict[str, Any]) -> tuple[int, int] | None:
    if not isinstance(pose, dict):
        return None
    try:
        resolution = float(grid["resolution"])
        return (
            int(math.floor((float(pose.get("x", 0.0)) - float(grid["origin_x"])) / resolution)),
            int(math.floor((float(pose.get("y", 0.0)) - float(grid["origin_y"])) / resolution)),
        )
    except Exception:
        return None


def _cell_for_xy(x: float, y: float, grid: dict[str, Any]) -> tuple[int, int] | None:
    return _cell_for_pose({"x": x, "y": y}, grid)


def _footprint_safe_cells(grid: dict[str, Any], clearance_m: float) -> set[tuple[int, int]]:
    free = set(grid["free"])
    resolution = float(grid["resolution"])
    clearance_m = max(float(clearance_m), 0.0)
    radius_cells = max(0, int(math.ceil((clearance_m + resolution * 0.5) / resolution)))
    safe: set[tuple[int, int]] = set()
    for cell in free:
        ok = True
        for dx in range(-radius_cells, radius_cells + 1):
            if not ok:
                break
            for dy in range(-radius_cells, radius_cells + 1):
                distance_m = math.hypot(dx, dy) * resolution
                if distance_m > clearance_m + resolution * 0.5:
                    continue
                if (cell[0] + dx, cell[1] + dy) not in free:
                    ok = False
                    break
        if ok:
            safe.add(cell)
    return safe


def _reachable_cells(
    grid: dict[str, Any],
    start: tuple[int, int] | None,
    traversable: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    if not traversable:
        return set()
    if start not in traversable:
        start = _nearest_free_cell(start, traversable)
    if start is None:
        return set(traversable)
    visited = {start}
    queue = [start]
    while queue:
        cell = queue.pop(0)
        for neighbor in _neighbors4(cell):
            if neighbor in traversable and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return visited


def _nearest_free_cell(start: tuple[int, int] | None, free: set[tuple[int, int]]) -> tuple[int, int] | None:
    if not free:
        return None
    if start is None:
        return next(iter(free))
    return min(free, key=lambda cell: abs(cell[0] - start[0]) + abs(cell[1] - start[1]))


def _neighbors4(cell: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    x, y = cell
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _cell_clearance_m(cell: tuple[int, int], grid: dict[str, Any]) -> float:
    occupied = grid["occupied"]
    if not occupied:
        return 99.0
    resolution = float(grid["resolution"])
    nearest_cells = min(math.hypot(cell[0] - item[0], cell[1] - item[1]) for item in occupied)
    return max(0.0, nearest_cells * resolution - resolution * 0.5)


def _path_distance_cells(
    start: tuple[int, int] | None,
    goal: tuple[int, int],
    reachable: set[tuple[int, int]],
) -> int:
    if start is None:
        return 0
    if start not in reachable:
        start = _nearest_free_cell(start, reachable)
    if start is None:
        return 0
    return abs(start[0] - goal[0]) + abs(start[1] - goal[1])


def _centered_path_cells(
    grid: dict[str, Any],
    start: tuple[int, int] | None,
    goal: tuple[int, int],
    traversable: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    if not traversable:
        return []
    if start not in traversable:
        start = _nearest_free_cell(start, traversable)
    if start is None or goal not in traversable:
        return []
    open_heap: list[tuple[float, int, tuple[int, int]]] = [(0.0, 0, start)]
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    cost_so_far: dict[tuple[int, int], float] = {start: 0.0}
    sequence = 1
    while open_heap:
        _, _, cell = heapq.heappop(open_heap)
        if cell == goal:
            break
        for neighbor in _neighbors8(cell, traversable):
            clearance = max(_cell_clearance_m(neighbor, grid), float(grid["resolution"]) * 0.25)
            distance = math.hypot(neighbor[0] - cell[0], neighbor[1] - cell[1])
            lateral_balance_penalty = _lateral_obstacle_balance_penalty(cell, neighbor, grid)
            centerline_penalty = (
                1.0
                + (1.25 / max(clearance, float(grid["resolution"]) * 0.25))
                + lateral_balance_penalty * 0.45
            )
            new_cost = cost_so_far[cell] + distance * centerline_penalty
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + math.hypot(goal[0] - neighbor[0], goal[1] - neighbor[1])
                parent[neighbor] = cell
                heapq.heappush(open_heap, (priority, sequence, neighbor))
                sequence += 1
    if goal not in parent:
        return []
    path = []
    cell: tuple[int, int] | None = goal
    while cell is not None:
        path.append(cell)
        cell = parent[cell]
    return list(reversed(path))


def _neighbors8(cell: tuple[int, int], traversable: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    x, y = cell
    result: list[tuple[int, int]] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            neighbor = (x + dx, y + dy)
            if neighbor not in traversable:
                continue
            if dx != 0 and dy != 0 and ((x + dx, y) not in traversable or (x, y + dy) not in traversable):
                continue
            result.append(neighbor)
    return tuple(result)


def _lateral_obstacle_balance_penalty(
    previous: tuple[int, int],
    cell: tuple[int, int],
    grid: dict[str, Any],
) -> float:
    dx = cell[0] - previous[0]
    dy = cell[1] - previous[1]
    if dx == 0 and dy == 0:
        return 0.0
    left = _lateral_free_distance_m(cell, -dy, dx, grid)
    right = _lateral_free_distance_m(cell, dy, -dx, grid)
    if left is None or right is None:
        return 0.0
    channel_width = left + right
    if channel_width <= 1e-9:
        return 0.0
    return abs(left - right) / channel_width


def _lateral_free_distance_m(
    cell: tuple[int, int],
    step_x: int,
    step_y: int,
    grid: dict[str, Any],
) -> float | None:
    resolution = float(grid["resolution"])
    max_steps = max(1, int(math.ceil(1.2 / resolution)))
    for step in range(1, max_steps + 1):
        probe = (cell[0] + step_x * step, cell[1] + step_y * step)
        if probe not in grid["free"]:
            return max(0.0, (step - 0.5) * resolution)
    return max_steps * resolution


def _path_payload(path_cells: list[tuple[int, int]], grid: dict[str, Any]) -> list[dict[str, float]]:
    if not path_cells:
        return []
    goal = path_cells[-1]
    if len(path_cells) > 32:
        stride = max(1, len(path_cells) // 31)
        path_cells = path_cells[::stride]
        if path_cells[-1] != goal:
            path_cells.append(goal)
    resolution = float(grid["resolution"])
    return [
        {
            "x": round(float(grid["origin_x"]) + (cell[0] + 0.5) * resolution, 3),
            "y": round(float(grid["origin_y"]) + (cell[1] + 0.5) * resolution, 3),
        }
        for cell in path_cells
    ]


def _direct_goal_waypoint_payload(
    path_cells: list[tuple[int, int]],
    grid: dict[str, Any],
    current_pose: dict[str, Any] | None,
    goal_pose: dict[str, float],
    horizon_m: float,
    *,
    target_label: str,
    region_id: Any,
) -> dict[str, Any]:
    """Return the final planned goal without agent-level route segmentation."""
    points = _path_points_for_waypoint(path_cells, grid, current_pose, goal_pose)
    total_m = _path_length_m(points)
    waypoint = _waypoint_payload(
        target_label,
        region_id,
        goal_pose,
        horizon_m=max(float(horizon_m or DEFAULT_NAVIGATION_WAYPOINT_HORIZON_M), float(grid["resolution"])),
        distance_from_start_m=total_m,
        remaining_to_goal_m=0.0,
        is_final_waypoint=True,
    )
    waypoint_pose = _json_pose(goal_pose)
    waypoint["waypoint_id"] = (
        f"{_slug(str(target_label or region_id or 'target'))}_direct_"
        f"x{int(round(waypoint_pose['x'] * 100.0))}_"
        f"y{int(round(waypoint_pose['y'] * 100.0))}"
    )
    waypoint["navigation_mode"] = "direct_final_goal"
    return waypoint


def _next_waypoint_payload(
    path_cells: list[tuple[int, int]],
    grid: dict[str, Any],
    current_pose: dict[str, Any] | None,
    goal_pose: dict[str, float],
    horizon_m: float,
    *,
    target_label: str,
    region_id: Any,
) -> dict[str, Any]:
    horizon_m = max(float(horizon_m or DEFAULT_NAVIGATION_WAYPOINT_HORIZON_M), float(grid["resolution"]))
    points = _path_points_for_waypoint(path_cells, grid, current_pose, goal_pose)
    if len(points) < 2:
        return _waypoint_payload(
            target_label,
            region_id,
            goal_pose,
            horizon_m=horizon_m,
            distance_from_start_m=0.0,
            remaining_to_goal_m=0.0,
            is_final_waypoint=True,
        )

    total_m = _path_length_m(points)
    target_m = min(horizon_m, total_m)
    travelled_m = 0.0
    selected = dict(points[-1])
    yaw = _resolved_goal_yaw(current_pose)
    for previous, nxt in zip(points, points[1:]):
        segment_m = math.hypot(float(nxt["x"]) - float(previous["x"]), float(nxt["y"]) - float(previous["y"]))
        if segment_m <= 1e-9:
            continue
        if travelled_m + segment_m >= target_m:
            ratio = (target_m - travelled_m) / segment_m
            x = float(previous["x"]) + (float(nxt["x"]) - float(previous["x"])) * ratio
            y = float(previous["y"]) + (float(nxt["y"]) - float(previous["y"])) * ratio
            yaw = math.atan2(float(nxt["y"]) - float(previous["y"]), float(nxt["x"]) - float(previous["x"]))
            selected = {"x": round(x, 3), "y": round(y, 3), "yaw": round(yaw, 3)}
            break
        travelled_m += segment_m
    else:
        if len(points) >= 2:
            previous = points[-2]
            nxt = points[-1]
            yaw = math.atan2(float(nxt["y"]) - float(previous["y"]), float(nxt["x"]) - float(previous["x"]))
        selected = {**goal_pose, "yaw": round(yaw, 3)}

    remaining_m = max(total_m - target_m, 0.0)
    final_tolerance_m = max(float(grid["resolution"]) * 2.0, 0.20)
    return _waypoint_payload(
        target_label,
        region_id,
        selected,
        horizon_m=horizon_m,
        distance_from_start_m=target_m,
        remaining_to_goal_m=remaining_m,
        is_final_waypoint=remaining_m <= final_tolerance_m,
    )


def _centerline_waypoints_payload(
    path_cells: list[tuple[int, int]],
    grid: dict[str, Any],
    current_pose: dict[str, Any] | None,
    goal_pose: dict[str, float],
    horizon_m: float,
    *,
    target_label: str,
    region_id: Any,
) -> list[dict[str, Any]]:
    points = _path_points_for_waypoint(path_cells, grid, current_pose, goal_pose)
    if len(points) < 2:
        return [
            _waypoint_payload(
                target_label,
                region_id,
                goal_pose,
                horizon_m=max(float(horizon_m or DEFAULT_NAVIGATION_WAYPOINT_HORIZON_M), 0.01),
                distance_from_start_m=0.0,
                remaining_to_goal_m=0.0,
                is_final_waypoint=True,
            )
        ]

    total_m = _path_length_m(points)
    if total_m <= 1e-9:
        return []
    horizon_m = max(float(horizon_m or DEFAULT_NAVIGATION_WAYPOINT_HORIZON_M), float(grid["resolution"]))
    final_tolerance_m = max(float(grid["resolution"]) * 2.0, 0.20)
    distances: list[float] = []
    next_distance = min(horizon_m, total_m)
    while next_distance < total_m - final_tolerance_m:
        distances.append(next_distance)
        next_distance += horizon_m
    if not distances or abs(distances[-1] - total_m) > final_tolerance_m:
        distances.append(total_m)

    waypoints: list[dict[str, Any]] = []
    for distance_m in distances:
        pose = _pose_at_path_distance(points, distance_m)
        remaining_m = max(total_m - distance_m, 0.0)
        waypoints.append(
            _waypoint_payload(
                target_label,
                region_id,
                pose,
                horizon_m=horizon_m,
                distance_from_start_m=distance_m,
                remaining_to_goal_m=remaining_m,
                is_final_waypoint=remaining_m <= final_tolerance_m,
            )
        )
    return waypoints


def _pose_at_path_distance(points: list[dict[str, Any]], target_m: float) -> dict[str, float]:
    travelled_m = 0.0
    for previous, nxt in zip(points, points[1:]):
        dx = float(nxt["x"]) - float(previous["x"])
        dy = float(nxt["y"]) - float(previous["y"])
        segment_m = math.hypot(dx, dy)
        if segment_m <= 1e-9:
            continue
        if travelled_m + segment_m >= target_m:
            ratio = max(0.0, min((target_m - travelled_m) / segment_m, 1.0))
            return {
                "x": round(float(previous["x"]) + dx * ratio, 3),
                "y": round(float(previous["y"]) + dy * ratio, 3),
                "yaw": round(math.atan2(dy, dx), 3),
            }
        travelled_m += segment_m
    if len(points) >= 2:
        previous = points[-2]
        nxt = points[-1]
        yaw = math.atan2(float(nxt["y"]) - float(previous["y"]), float(nxt["x"]) - float(previous["x"]))
    else:
        yaw = _resolved_goal_yaw(points[-1] if points else None)
    return {**_json_pose(points[-1]), "yaw": round(yaw, 3)}


def _explicit_waypoint_payload(
    target_label: str,
    region_id: Any,
    goal_pose: dict[str, float],
    current_pose: dict[str, Any] | None,
    horizon_m: float,
) -> dict[str, Any]:
    distance_m = _pose_distance_m(current_pose, goal_pose) if current_pose else 0.0
    return _waypoint_payload(
        target_label,
        region_id,
        goal_pose,
        horizon_m=max(float(horizon_m or DEFAULT_NAVIGATION_WAYPOINT_HORIZON_M), 0.01),
        distance_from_start_m=distance_m,
        remaining_to_goal_m=0.0,
        is_final_waypoint=True,
    )


def _waypoint_payload(
    target_label: str,
    region_id: Any,
    pose: dict[str, Any],
    *,
    horizon_m: float,
    distance_from_start_m: float,
    remaining_to_goal_m: float,
    is_final_waypoint: bool,
) -> dict[str, Any]:
    waypoint_pose = _json_pose(pose)
    waypoint_id = (
        f"{_slug(str(target_label or region_id or 'target'))}_"
        f"h{int(round(horizon_m * 100.0))}_"
        f"x{int(round(waypoint_pose['x'] * 100.0))}_"
        f"y{int(round(waypoint_pose['y'] * 100.0))}"
    )
    return {
        "waypoint_id": waypoint_id,
        "target_label": target_label,
        "region_id": region_id,
        "x": waypoint_pose["x"],
        "y": waypoint_pose["y"],
        "yaw": waypoint_pose["yaw"],
        "horizon_m": round(horizon_m, 3),
        "distance_from_start_m": round(distance_from_start_m, 3),
        "remaining_to_goal_m": round(remaining_to_goal_m, 3),
        "is_final_waypoint": bool(is_final_waypoint),
    }


def _path_points_for_waypoint(
    path_cells: list[tuple[int, int]],
    grid: dict[str, Any],
    current_pose: dict[str, Any] | None,
    goal_pose: dict[str, float],
) -> list[dict[str, float]]:
    points: list[dict[str, float]] = []
    if isinstance(current_pose, dict):
        points.append(_json_pose(current_pose))
    points.extend(_cell_center_pose(cell, grid) for cell in path_cells)
    if not points or _pose_distance_m(points[-1], goal_pose) > float(grid["resolution"]) * 0.75:
        points.append(_json_pose(goal_pose))
    deduped: list[dict[str, float]] = []
    for point in points:
        if deduped and _pose_distance_m(deduped[-1], point) < 1e-6:
            continue
        deduped.append(point)
    return deduped


def _cell_center_pose(cell: tuple[int, int], grid: dict[str, Any]) -> dict[str, float]:
    resolution = float(grid["resolution"])
    return {
        "x": round(float(grid["origin_x"]) + (cell[0] + 0.5) * resolution, 3),
        "y": round(float(grid["origin_y"]) + (cell[1] + 0.5) * resolution, 3),
        "yaw": 0.0,
    }


def _cell_center_xy(cell: tuple[int, int], grid: dict[str, Any]) -> tuple[float, float]:
    pose = _cell_center_pose(cell, grid)
    return float(pose["x"]), float(pose["y"])


def _path_length_m(points: list[dict[str, Any]]) -> float:
    return sum(_pose_distance_m(previous, nxt) for previous, nxt in zip(points, points[1:]))


def _pose_distance_m(a: dict[str, Any] | None, b: dict[str, Any] | None) -> float:
    if not isinstance(a, dict) or not isinstance(b, dict):
        return 0.0
    return math.hypot(float(a.get("x", 0.0) or 0.0) - float(b.get("x", 0.0) or 0.0), float(a.get("y", 0.0) or 0.0) - float(b.get("y", 0.0) or 0.0))


def _nearest_occupied_cell_to_point(grid: dict[str, Any], x: float, y: float, *, max_radius_m: float) -> tuple[int, int] | None:
    occupied = grid["occupied"]
    if not occupied:
        return None
    max_radius_m = max(float(max_radius_m), 0.0)
    candidates = []
    for cell in occupied:
        cx, cy = _cell_center_xy(cell, grid)
        distance = math.hypot(cx - x, cy - y)
        if distance <= max_radius_m:
            candidates.append((distance, cell))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def _nearby_occupied_cells(grid: dict[str, Any], center: tuple[int, int], *, radius_m: float) -> list[tuple[int, int]]:
    resolution = float(grid["resolution"])
    radius_cells = max(1, int(math.ceil(max(float(radius_m), resolution) / resolution)))
    occupied = grid["occupied"]
    result = [
        cell
        for cell in occupied
        if abs(cell[0] - center[0]) <= radius_cells
        and abs(cell[1] - center[1]) <= radius_cells
        and math.hypot(cell[0] - center[0], cell[1] - center[1]) * resolution <= radius_m + resolution
    ]
    return result or [center]


def _fit_surface_tangent(
    grid: dict[str, Any],
    cells: list[tuple[int, int]],
    *,
    fallback: tuple[float, float],
) -> tuple[float, float]:
    if len(cells) < 2:
        return _normalize2(fallback) or (1.0, 0.0)
    points = [_cell_center_xy(cell, grid) for cell in cells]
    mean_x = sum(point[0] for point in points) / len(points)
    mean_y = sum(point[1] for point in points) / len(points)
    sxx = sum((point[0] - mean_x) ** 2 for point in points)
    syy = sum((point[1] - mean_y) ** 2 for point in points)
    sxy = sum((point[0] - mean_x) * (point[1] - mean_y) for point in points)
    if abs(sxx) + abs(syy) + abs(sxy) <= 1e-12:
        return _normalize2(fallback) or (1.0, 0.0)
    theta = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    tangent = (math.cos(theta), math.sin(theta))
    return _normalize2(tangent) or (_normalize2(fallback) or (1.0, 0.0))


def _surface_approach_candidate(
    *,
    grid: dict[str, Any],
    start_cell: tuple[int, int],
    footprint_safe_cells: set[tuple[int, int]],
    hit_point: dict[str, Any],
    tangent: tuple[float, float],
    normal: tuple[float, float],
    desired_standoff_m: float,
    min_clearance_m: float,
) -> dict[str, Any] | None:
    resolution = float(grid["resolution"])
    standoff_values = [
        desired_standoff_m,
        desired_standoff_m + resolution,
        desired_standoff_m + resolution * 2.0,
        max(float(min_clearance_m), desired_standoff_m - resolution),
    ]
    offset_values = [0.0]
    for step in (1, 2, 3):
        offset_values.extend([resolution * step, -resolution * step])
    scored: list[tuple[float, dict[str, Any]]] = []
    for standoff in standoff_values:
        for offset in offset_values:
            x = float(hit_point["x"]) + normal[0] * standoff + tangent[0] * offset
            y = float(hit_point["y"]) + normal[1] * standoff + tangent[1] * offset
            cell = _cell_for_xy(x, y, grid)
            line_cells = _bresenham_cells(start_cell, cell) if cell is not None else []
            unsafe = [line_cell for line_cell in line_cells[1:] if line_cell not in footprint_safe_cells]
            clearance_m = _cell_clearance_m(cell, grid) if cell is not None else None
            footprint_clear = cell in footprint_safe_cells and not unsafe if cell is not None else False
            score = (
                abs(standoff - desired_standoff_m)
                + abs(offset) * 0.5
                - min(float(clearance_m or 0.0), desired_standoff_m) * 0.05
            )
            scored.append(
                (
                    score,
                    {
                        "x": round(x, 3),
                        "y": round(y, 3),
                        "cell": cell,
                        "line_cells": line_cells,
                        "clearance_m": clearance_m,
                        "footprint_clear": footprint_clear,
                        "unsafe_cell_count": len(unsafe),
                        "first_unsafe_cell": unsafe[0] if unsafe else None,
                    },
                )
            )
    if not scored:
        return None
    return min(scored, key=lambda item: item[0])[1]


def _support_surface_payload(
    hit_point: dict[str, Any],
    tangent: tuple[float, float],
    normal: tuple[float, float],
    support_cells: list[tuple[int, int]],
    grid: dict[str, Any],
) -> dict[str, Any]:
    return {
        "hit_point": {"x": round(float(hit_point["x"]), 3), "y": round(float(hit_point["y"]), 3)},
        "tangent_yaw": round(math.atan2(tangent[1], tangent[0]), 3),
        "normal_yaw": round(math.atan2(normal[1], normal[0]), 3),
        "occupied_sample_count": len(support_cells),
        "sample_points": _path_payload(support_cells[:32], grid),
    }


def _normalize2(vector: tuple[float, float]) -> tuple[float, float] | None:
    length = math.hypot(float(vector[0]), float(vector[1]))
    if length <= 1e-12:
        return None
    return (float(vector[0]) / length, float(vector[1]) / length)


def _dot2(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(a[0]) * float(b[0]) + float(a[1]) * float(b[1])


def _slug(value: str) -> str:
    normalized = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
    return "_".join(part for part in normalized.split("_") if part) or "target"


def _resolved_goal_yaw(current_pose: dict[str, Any] | None) -> float:
    if isinstance(current_pose, dict) and "yaw" in current_pose:
        try:
            return round(float(current_pose.get("yaw", 0.0) or 0.0), 3)
        except Exception:
            pass
    return 0.0


def _point_in_polygon(point_x: float, point_y: float, polygon: list[Any]) -> bool:
    if len(polygon) < 3:
        return False
    inside = False
    previous = len(polygon) - 1
    for current in range(len(polygon)):
        try:
            xi, yi = float(polygon[current][0]), float(polygon[current][1])
            xj, yj = float(polygon[previous][0]), float(polygon[previous][1])
        except Exception:
            previous = current
            continue
        if ((yi > point_y) != (yj > point_y)) and (point_x < (xj - xi) * (point_y - yi) / ((yj - yi) or 1e-9) + xi):
            inside = not inside
        previous = current
    return inside


def _bounded_float(value: Any, fallback: float, *, minimum: float, maximum: float) -> float:
    try:
        result = float(value)
    except Exception:
        result = float(fallback)
    return min(max(result, minimum), maximum)


def _bounded_int(value: Any, fallback: int, *, minimum: int, maximum: int) -> int:
    try:
        result = int(value)
    except Exception:
        result = int(fallback)
    return min(max(result, minimum), maximum)


def _default_region_exploration_stop_count(polygon: list[Any]) -> int:
    axes = _polygon_axes(polygon)
    major_length = float(axes["major_length"])
    area = abs(_polygon_area(polygon))
    if area <= 0.90:
        return 1
    if major_length >= 2.25:
        return 3
    if major_length >= 1.20:
        return 2
    return 1


def _polygon_area(polygon: list[Any]) -> float:
    area = 0.0
    points = _valid_polygon_points(polygon)
    if len(points) < 3:
        return 0.0
    for previous, current in zip(points, points[1:] + points[:1]):
        area += previous[0] * current[1] - current[0] * previous[1]
    return area / 2.0


def _polygon_axes(polygon: list[Any]) -> dict[str, Any]:
    points = _valid_polygon_points(polygon)
    if not points:
        return {
            "center": (0.0, 0.0),
            "major": (1.0, 0.0),
            "minor": (0.0, 1.0),
            "major_length": 0.0,
            "minor_length": 0.0,
        }
    center_x = sum(point[0] for point in points) / len(points)
    center_y = sum(point[1] for point in points) / len(points)
    sxx = sum((point[0] - center_x) ** 2 for point in points) / len(points)
    syy = sum((point[1] - center_y) ** 2 for point in points) / len(points)
    sxy = sum((point[0] - center_x) * (point[1] - center_y) for point in points) / len(points)
    angle = 0.5 * math.atan2(2.0 * sxy, sxx - syy) if abs(sxy) > 1e-9 or abs(sxx - syy) > 1e-9 else 0.0
    major = (math.cos(angle), math.sin(angle))
    minor = (-major[1], major[0])
    major_projections = [_dot(point[0] - center_x, point[1] - center_y, major) for point in points]
    minor_projections = [_dot(point[0] - center_x, point[1] - center_y, minor) for point in points]
    return {
        "center": (center_x, center_y),
        "major": major,
        "minor": minor,
        "major_length": max(major_projections) - min(major_projections) if major_projections else 0.0,
        "minor_length": max(minor_projections) - min(minor_projections) if minor_projections else 0.0,
    }


def _should_use_inside_edge_region_approach_goal(
    polygon: list[Any],
    *,
    region_free_cell_count: int,
    footprint_safe_candidate_count: int,
) -> bool:
    if region_free_cell_count <= 0:
        return False
    axes = _polygon_axes(polygon)
    area_m2 = abs(_polygon_area(polygon))
    minor_length_m = float(axes["minor_length"])
    safe_ratio = float(footprint_safe_candidate_count) / float(region_free_cell_count)
    shallow_fixture_region = area_m2 <= 2.0 and minor_length_m <= 0.95
    mostly_unsafe_small_region = area_m2 <= 2.25 and safe_ratio <= 0.12
    return shallow_fixture_region or mostly_unsafe_small_region


def _is_object_search_navigation_purpose(value: str) -> bool:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return normalized in {"object_search", "region_object_search", "search", "search_region", "find_object", "grab_object"}


def _inside_edge_region_approach_cells(
    grid: dict[str, Any],
    polygon: list[Any],
    *,
    inside_candidates: list[tuple[int, int]],
    min_clearance_m: float,
) -> list[tuple[int, int]]:
    resolution = float(grid["resolution"])
    approach_band_m = max(0.35, float(min_clearance_m), resolution * 2.0)
    candidates: list[tuple[int, int]] = []
    for cell in inside_candidates:
        point_x, point_y = _cell_center_xy(cell, grid)
        if _point_to_polygon_edge_distance_m(point_x, point_y, polygon) <= approach_band_m:
            candidates.append(cell)
    return candidates


def _valid_polygon_points(polygon: list[Any]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for point in polygon:
        try:
            points.append((float(point[0]), float(point[1])))
        except Exception:
            continue
    return points


def _dot(x: float, y: float, axis: tuple[float, float]) -> float:
    return x * axis[0] + y * axis[1]


def _region_exploration_boundary_cells(
    grid: dict[str, Any],
    polygon: list[Any],
    *,
    margin_m: float,
) -> set[tuple[int, int]]:
    resolution = float(grid["resolution"])
    margin_m = max(float(margin_m), resolution)
    free = grid["free"]
    boundary: set[tuple[int, int]] = set()
    for cell in grid["occupied"]:
        x, y = _cell_center_xy(cell, grid)
        if not (_point_in_polygon(x, y, polygon) or _point_to_polygon_distance_m(x, y, polygon) <= margin_m):
            continue
        if any(neighbor in free for neighbor in _neighbors8_unchecked(cell)):
            boundary.add(cell)
    return boundary


def _select_region_exploration_stops(
    grid: dict[str, Any],
    candidates: list[tuple[int, int]],
    axes: dict[str, Any],
    *,
    max_stops: int,
    min_stop_separation_m: float,
) -> list[tuple[int, int]]:
    if not candidates:
        return []
    if max_stops <= 1:
        return [max(candidates, key=lambda cell: _cell_clearance_m(cell, grid))]

    center_x, center_y = axes["center"]
    major = axes["major"]
    minor = axes["minor"]
    projections = [
        (
            _dot(_cell_center_xy(cell, grid)[0] - center_x, _cell_center_xy(cell, grid)[1] - center_y, major),
            _dot(_cell_center_xy(cell, grid)[0] - center_x, _cell_center_xy(cell, grid)[1] - center_y, minor),
            cell,
        )
        for cell in candidates
    ]
    min_projection = min(item[0] for item in projections)
    max_projection = max(item[0] for item in projections)
    span = max(max_projection - min_projection, float(grid["resolution"]))
    selected: list[tuple[int, int]] = []
    for index in range(max_stops):
        target_projection = min_projection + (index + 0.5) * span / max_stops
        bin_radius = span / max_stops * 0.60
        eligible = [
            item
            for item in projections
            if item[2] not in selected
            and all(_cell_distance_m(item[2], selected_cell, grid) >= min_stop_separation_m for selected_cell in selected)
        ]
        in_bin = [item for item in eligible if abs(item[0] - target_projection) <= bin_radius]
        if not in_bin:
            in_bin = eligible
        if not in_bin:
            break
        _, _, best = max(
            in_bin,
            key=lambda item: (
                _cell_clearance_m(item[2], grid)
                - abs(item[0] - target_projection) * 0.35
                - abs(item[1]) * 0.20
            ),
        )
        selected.append(best)
    return selected


def _region_exploration_shots_for_stop(
    grid: dict[str, Any],
    stop_cell: tuple[int, int],
    boundary_cells: set[tuple[int, int]],
    area_cells: set[tuple[int, int]],
    *,
    already_covered_boundary_cells: set[tuple[int, int]],
    already_covered_area_cells: set[tuple[int, int]],
    fov_deg: float,
    shots_per_stop: int,
    min_range_m: float,
    max_range_m: float,
) -> tuple[list[dict[str, Any]], set[tuple[int, int]], set[tuple[int, int]]]:
    if not boundary_cells and not area_cells:
        pose = _cell_center_pose(stop_cell, grid)
        yaw = round(float(pose["yaw"]), 3)
        return (
            [
                _region_exploration_shot_payload(
                    stop_cell,
                    grid,
                    yaw=yaw,
                    fov_deg=fov_deg,
                    max_range_m=max_range_m,
                    covered_boundary_count=0,
                    covered_area_count=0,
                )
            ],
            set(),
            set(),
        )

    angle_step = math.radians(15.0)
    candidate_yaws = [index * angle_step for index in range(int(round((2.0 * math.pi) / angle_step)))]
    coverage_by_yaw = [
        (
            yaw,
            _visible_boundary_cells_for_shot(
                grid,
                stop_cell,
                boundary_cells,
                yaw=yaw,
                fov_deg=fov_deg,
                min_range_m=min_range_m,
                max_range_m=max_range_m,
            ),
            _visible_region_area_cells_for_shot(
                grid,
                stop_cell,
                area_cells,
                yaw=yaw,
                fov_deg=fov_deg,
                min_range_m=min_range_m,
                max_range_m=max_range_m,
            ),
        )
        for yaw in candidate_yaws
    ]
    selected: list[tuple[float, set[tuple[int, int]], set[tuple[int, int]]]] = []
    covered_boundary: set[tuple[int, int]] = set()
    covered_area: set[tuple[int, int]] = set()
    for _ in range(shots_per_stop):
        remaining = [
            (
                yaw,
                boundary,
                area,
                len(boundary - covered_boundary - already_covered_boundary_cells),
                len(boundary - covered_boundary),
                len(area - covered_area - already_covered_area_cells),
                len(area - covered_area),
            )
            for yaw, boundary, area in coverage_by_yaw
            if all(_angle_delta_abs(yaw, existing[0]) >= math.radians(25.0) for existing in selected)
        ]
        if not remaining:
            break
        yaw, boundary, area, new_boundary_gain, local_boundary_gain, new_area_gain, local_area_gain = max(
            remaining,
            key=lambda item: (
                item[3] * 4.0 + item[5] * 0.35 + item[4] * 0.5 + item[6] * 0.05,
                item[3],
                item[5],
                item[4],
            ),
        )
        if new_boundary_gain <= 0 and local_boundary_gain <= 0 and new_area_gain <= 0 and local_area_gain <= 0 and selected:
            break
        selected.append((yaw, boundary, area))
        covered_boundary.update(boundary)
        covered_area.update(area)

    shots = [
        _region_exploration_shot_payload(
            stop_cell,
            grid,
            yaw=yaw,
            fov_deg=fov_deg,
            max_range_m=max_range_m,
            covered_boundary_count=len(boundary),
            covered_area_count=len(area),
        )
        for yaw, boundary, area in selected
    ]
    return shots, covered_boundary, covered_area


def _visible_boundary_cells_for_shot(
    grid: dict[str, Any],
    stop_cell: tuple[int, int],
    boundary_cells: set[tuple[int, int]],
    *,
    yaw: float,
    fov_deg: float,
    min_range_m: float,
    max_range_m: float,
) -> set[tuple[int, int]]:
    origin_x, origin_y = _cell_center_xy(stop_cell, grid)
    half_fov = math.radians(float(fov_deg)) / 2.0
    visible: set[tuple[int, int]] = set()
    for cell in boundary_cells:
        target_x, target_y = _cell_center_xy(cell, grid)
        distance_m = math.hypot(target_x - origin_x, target_y - origin_y)
        if distance_m < min_range_m or distance_m > max_range_m:
            continue
        angle = math.atan2(target_y - origin_y, target_x - origin_x)
        if _angle_delta_abs(angle, yaw) > half_fov:
            continue
        if _line_of_sight_clear(grid, stop_cell, cell):
            visible.add(cell)
    return visible


def _visible_region_area_cells_for_shot(
    grid: dict[str, Any],
    stop_cell: tuple[int, int],
    area_cells: set[tuple[int, int]],
    *,
    yaw: float,
    fov_deg: float,
    min_range_m: float,
    max_range_m: float,
) -> set[tuple[int, int]]:
    origin_x, origin_y = _cell_center_xy(stop_cell, grid)
    half_fov = math.radians(float(fov_deg)) / 2.0
    visible: set[tuple[int, int]] = set()
    for cell in area_cells:
        if cell == stop_cell:
            continue
        target_x, target_y = _cell_center_xy(cell, grid)
        distance_m = math.hypot(target_x - origin_x, target_y - origin_y)
        if distance_m < min_range_m or distance_m > max_range_m:
            continue
        angle = math.atan2(target_y - origin_y, target_x - origin_x)
        if _angle_delta_abs(angle, yaw) > half_fov:
            continue
        if _line_of_sight_clear(grid, stop_cell, cell):
            visible.add(cell)
    return visible


def _region_exploration_shot_payload(
    stop_cell: tuple[int, int],
    grid: dict[str, Any],
    *,
    yaw: float,
    fov_deg: float,
    max_range_m: float,
    covered_boundary_count: int,
    covered_area_count: int,
) -> dict[str, Any]:
    origin_x, origin_y = _cell_center_xy(stop_cell, grid)
    half_fov = math.radians(float(fov_deg)) / 2.0
    left_yaw = yaw - half_fov
    right_yaw = yaw + half_fov
    center = {
        "x": round(origin_x + math.cos(yaw) * max_range_m, 3),
        "y": round(origin_y + math.sin(yaw) * max_range_m, 3),
    }
    left = {
        "x": round(origin_x + math.cos(left_yaw) * max_range_m, 3),
        "y": round(origin_y + math.sin(left_yaw) * max_range_m, 3),
    }
    right = {
        "x": round(origin_x + math.cos(right_yaw) * max_range_m, 3),
        "y": round(origin_y + math.sin(right_yaw) * max_range_m, 3),
    }
    yaw = math.atan2(math.sin(yaw), math.cos(yaw))
    return {
        "shot_id": f"yaw_{int(round(math.degrees(yaw)))}",
        "yaw": round(yaw, 3),
        "yaw_deg": round(math.degrees(yaw), 1),
        "fov_deg": round(float(fov_deg), 3),
        "covered_boundary_cell_count": int(covered_boundary_count),
        "covered_region_area_cell_count": int(covered_area_count),
        "cone": {
            "origin": {"x": round(origin_x, 3), "y": round(origin_y, 3)},
            "left": left,
            "center": center,
            "right": right,
        },
    }


def _cell_center_xy(cell: tuple[int, int], grid: dict[str, Any]) -> tuple[float, float]:
    resolution = float(grid["resolution"])
    return (
        float(grid["origin_x"]) + (cell[0] + 0.5) * resolution,
        float(grid["origin_y"]) + (cell[1] + 0.5) * resolution,
    )


def _cell_distance_m(a: tuple[int, int], b: tuple[int, int], grid: dict[str, Any]) -> float:
    resolution = float(grid["resolution"])
    return math.hypot(a[0] - b[0], a[1] - b[1]) * resolution


def _neighbors8_unchecked(cell: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    x, y = cell
    return tuple(
        (x + dx, y + dy)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if not (dx == 0 and dy == 0)
    )


def _point_to_polygon_distance_m(point_x: float, point_y: float, polygon: list[Any]) -> float:
    points = _valid_polygon_points(polygon)
    if not points:
        return 99.0
    if _point_in_polygon(point_x, point_y, polygon):
        return 0.0
    return _point_to_polygon_edge_distance_m(point_x, point_y, polygon)


def _point_to_polygon_edge_distance_m(point_x: float, point_y: float, polygon: list[Any]) -> float:
    points = _valid_polygon_points(polygon)
    if not points:
        return 99.0
    return min(
        _point_to_segment_distance_m(point_x, point_y, a[0], a[1], b[0], b[1])
        for a, b in zip(points, points[1:] + points[:1])
    )


def _point_to_segment_distance_m(
    point_x: float,
    point_y: float,
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
) -> float:
    dx = end_x - start_x
    dy = end_y - start_y
    segment_length_sq = dx * dx + dy * dy
    if segment_length_sq <= 1e-12:
        return math.hypot(point_x - start_x, point_y - start_y)
    t = ((point_x - start_x) * dx + (point_y - start_y) * dy) / segment_length_sq
    t = min(max(t, 0.0), 1.0)
    projection_x = start_x + t * dx
    projection_y = start_y + t * dy
    return math.hypot(point_x - projection_x, point_y - projection_y)


def _angle_delta_abs(a: float, b: float) -> float:
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def _line_of_sight_clear(
    grid: dict[str, Any],
    start: tuple[int, int],
    target: tuple[int, int],
) -> bool:
    cells = _bresenham_cells(start, target)
    for cell in cells[1:-1]:
        if cell in grid["occupied"]:
            return False
    return True


def _bresenham_cells(start: tuple[int, int], target: tuple[int, int]) -> list[tuple[int, int]]:
    x0, y0 = start
    x1, y1 = target
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    x, y = x0, y0
    cells = []
    while True:
        cells.append((x, y))
        if x == x1 and y == y1:
            break
        error2 = 2 * error
        if error2 >= dy:
            error += dy
            x += sx
        if error2 <= dx:
            error += dx
            y += sy
    return cells


def _memory_place(place: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": str(place.get("name") or place.get("label") or "place"),
        "pose": _json_pose(place.get("pose") or {}),
        "region_id": place.get("region_id"),
        "source": str(place.get("source") or "map"),
    }


def _objects_from_map(map_payload: dict[str, Any]) -> list[dict[str, Any]]:
    objects: dict[str, dict[str, Any]] = {}
    for item in map_payload.get("objects", []):
        if isinstance(item, dict):
            label = str(item.get("label") or item.get("name") or item.get("object_id") or "").strip()
            if label:
                objects[label] = _memory_object(item, label=label)
    for place in map_payload.get("named_places", []):
        if not isinstance(place, dict):
            continue
        name = str(place.get("name", "")).strip()
        if _looks_like_fixture(name) and name not in objects:
            objects[name] = _memory_object(
                {
                    "object_id": f"fixture_{name}",
                    "label": name,
                    "region_id": place.get("region_id"),
                    "pose": place.get("pose"),
                    "approach_pose": place.get("pose"),
                    "source": place.get("source"),
                },
                label=name,
            )
    semantic_memory = map_payload.get("semantic_memory") if isinstance(map_payload.get("semantic_memory"), dict) else {}
    for place in semantic_memory.get("named_places", []) if isinstance(semantic_memory, dict) else []:
        if not isinstance(place, dict):
            continue
        label = str(place.get("label") or "").strip()
        if label and _looks_like_fixture(label) and label not in objects:
            objects[label] = _memory_object(
                {
                    "object_id": place.get("place_id") or f"fixture_{label}",
                    "label": label,
                    "pose": place.get("anchor_pose"),
                    "approach_pose": place.get("anchor_pose"),
                    "source": "semantic_memory",
                    "confidence": place.get("confidence"),
                },
                label=label,
            )
    return list(objects.values())


def _memory_object(item: dict[str, Any], *, label: str) -> dict[str, Any]:
    return {
        "object_id": str(item.get("object_id") or item.get("id") or f"object_{label}"),
        "label": label,
        "category": str(item.get("category") or _category_for_label(label)),
        "region_id": item.get("region_id"),
        "pose": _json_pose(item.get("pose") or {}),
        "approach_pose": _json_pose(item.get("approach_pose") or item.get("pose") or {}),
        "observable_from": list(item.get("observable_from") or []),
        "affordances": list(item.get("affordances") or _affordances_for_label(label)),
        "source": str(item.get("source") or "map"),
        "confidence": float(item.get("confidence", 1.0) or 1.0),
    }


def _navigation_graph_from_regions(regions: list[dict[str, Any]], places: list[dict[str, Any]]) -> dict[str, Any]:
    waypoints: list[dict[str, Any]] = []
    for region in regions:
        for waypoint in region.get("default_waypoints", []):
            waypoints.append({"region_id": region["region_id"], **_json_clone(waypoint)})
        for waypoint in region.get("entry_waypoints", []):
            if isinstance(waypoint, dict):
                waypoints.append({"region_id": region["region_id"], **_json_clone(waypoint)})
    for place in places:
        waypoints.append({"name": place["name"], "region_id": place.get("region_id"), **_json_pose(place.get("pose") or {})})
    edges = []
    for region in regions:
        for adjacent in region.get("adjacent_region_ids", []):
            edges.append({"from_region_id": region["region_id"], "to_region_id": adjacent})
    return {"waypoints": _dedupe_waypoints(waypoints), "edges": edges}


def _skills_from_objects(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    skills: dict[str, dict[str, Any]] = {}
    for item in objects:
        for affordance in item.get("affordances", []):
            skill_id = str(affordance)
            skills.setdefault(
                skill_id,
                {
                    "skill_id": skill_id,
                    "kind": "vla_or_scripted_skill",
                    "target_categories": [item.get("category", "object")],
                    "target_labels": [item.get("label")],
                    "required_pose_class": f"in_front_of_{item.get('label')}",
                    "required_observations": [f"{item.get('label')}_visible"],
                    "executor_binding": "vla_skill_runner",
                    "safety": {"requires_human_approval": True, "max_retries": 1},
                },
            )
    return list(skills.values())


def _looks_like_fixture(label: str) -> bool:
    normalized = label.lower()
    return any(token in normalized for token in ("fridge", "sink", "counter", "table", "dock", "couch", "sofa", "oven", "cabinet"))


def _category_for_label(label: str) -> str:
    normalized = label.lower()
    if any(token in normalized for token in ("fridge", "oven", "sink", "counter", "cabinet")):
        return "fixture"
    if any(token in normalized for token in ("dock", "charger")):
        return "dock"
    return "object"


def _affordances_for_label(label: str) -> list[str]:
    normalized = label.lower()
    if "fridge" in normalized:
        return ["open_fridge", "inspect_fridge_contents"]
    if any(token in normalized for token in ("table", "counter")):
        return ["place_item"]
    if "dock" in normalized:
        return ["return_to_dock"]
    return []


def _dedupe_waypoints(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for item in items:
        name = str(item.get("name") or f"{item.get('x', 0)}:{item.get('y', 0)}")
        deduped[name] = item
    return list(deduped.values())


def _json_pose(payload: dict[str, Any]) -> dict[str, float]:
    return {
        "x": round(float(payload.get("x", 0.0) or 0.0), 3),
        "y": round(float(payload.get("y", 0.0) or 0.0), 3),
        "yaw": round(float(payload.get("yaw", 0.0) or 0.0), 3),
    }


def _json_clone(payload: Any) -> Any:
    return json.loads(json.dumps(payload))


def _normalize_lookup_key(value: str) -> str:
    return " ".join(str(value).replace("_", " ").lower().split())


def _lookup_score(query: str, label: str) -> float:
    normalized_label = _normalize_lookup_key(label)
    if not normalized_label:
        return 0.0
    if query == normalized_label:
        return 1.0
    if query in normalized_label or normalized_label in query:
        return 0.75
    query_tokens = set(query.split())
    label_tokens = set(normalized_label.split())
    overlap = query_tokens & label_tokens
    if not overlap:
        return 0.0
    return min(0.65, len(overlap) / max(len(label_tokens), 1))


def _has_xy_pose(value: Any) -> bool:
    return isinstance(value, dict) and "x" in value and "y" in value


def _first_pose(value: Any) -> dict[str, Any] | None:
    if isinstance(value, list):
        for item in value:
            if _has_xy_pose(item):
                return item
    return None


def _target_priority(item: dict[str, Any]) -> int:
    return {"place": 3, "object": 2, "region": 1}.get(str(item.get("target_type")), 0)
