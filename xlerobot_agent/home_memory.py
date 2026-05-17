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
DEFAULT_NAVIGATION_SAFETY_GAP_M = 0.05
DEFAULT_ROBOT_FOOTPRINT_RADIUS_M = math.hypot(
    XLEROBOT_FOOTPRINT_LENGTH_M / 2.0,
    XLEROBOT_FOOTPRINT_WIDTH_M / 2.0,
)
DEFAULT_NAVIGATION_CLEARANCE_M = DEFAULT_ROBOT_FOOTPRINT_RADIUS_M + DEFAULT_NAVIGATION_SAFETY_GAP_M
DEFAULT_NAVIGATION_WAYPOINT_HORIZON_M = 2.0


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
    safety_gap_m = max(float(min_clearance_m) - footprint_radius_m, 0.0)
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
            "safety_gap_m": round(safety_gap_m, 3),
            "min_clearance_m": round(float(min_clearance_m), 3),
        }

    start_cell = _cell_for_pose(current_pose, grid) if current_pose else None
    reachable = _reachable_cells(grid, start_cell, footprint_safe_cells) if start_cell is not None else set(footprint_safe_cells)
    candidates = [cell for cell in region_free_cells if cell in reachable]
    if not candidates and start_cell is None:
        candidates = [cell for cell in region_free_cells if cell in footprint_safe_cells]
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
            "safety_gap_m": round(safety_gap_m, 3),
            "min_clearance_m": round(float(min_clearance_m), 3),
        }
    scored = [
        (
            _cell_clearance_m(cell, grid),
            -float(_path_distance_cells(start_cell, cell, reachable) if start_cell is not None and cell in reachable else 0)
            * float(grid["resolution"]),
            cell,
        )
        for cell in candidates
    ]
    scored.sort(reverse=True)
    best_clearance, _, best = scored[0]
    path_cells = _centered_path_cells(grid, start_cell, best, footprint_safe_cells) if start_cell is not None and best in reachable else []
    path_clearance = min((_cell_clearance_m(cell, grid) for cell in path_cells), default=best_clearance)
    goal_pose = {
        "x": round(grid["origin_x"] + (best[0] + 0.5) * resolution, 3),
        "y": round(grid["origin_y"] + (best[1] + 0.5) * resolution, 3),
        "yaw": _resolved_goal_yaw(current_pose),
    }
    path_length_m = _path_length_m(_path_points_for_waypoint(path_cells, grid, current_pose, goal_pose))
    next_waypoint = _next_waypoint_payload(
        path_cells,
        grid,
        current_pose,
        goal_pose,
        waypoint_horizon_m,
        target_label=region.get("label") or name_or_label,
        region_id=region.get("region_id"),
    )
    status = "succeeded" if best_clearance >= min_clearance_m else "low_clearance"
    return {
        "tool": "resolve_region_navigation_goal",
        "status": status,
        "target_label": region.get("label") or name_or_label,
        "target_type": "region",
        "region_id": region.get("region_id"),
        "goal_pose": goal_pose,
        "next_waypoint": next_waypoint,
        "source": "home_memory.occupancy_region_free_space",
        "candidate_count": len(region_free_cells),
        "reachable_candidate_count": len([cell for cell in region_free_cells if cell in reachable]),
        "footprint_safe_candidate_count": len([cell for cell in region_free_cells if cell in footprint_safe_cells]),
        "clearance_m": round(best_clearance, 3),
        "path_clearance_m": round(path_clearance, 3),
        "min_clearance_m": round(float(min_clearance_m), 3),
        "robot_footprint_radius_m": round(footprint_radius_m, 3),
        "safety_gap_m": round(safety_gap_m, 3),
        "path_strategy": "footprint_eroded_centerline_weighted_grid",
        "path_length_m": round(path_length_m, 3),
        "path": _path_payload(path_cells, grid),
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
        "adjacent_region_ids": list(region.get("adjacency") or region.get("adjacent_region_ids") or []),
        "evidence": _json_clone(region.get("evidence") or []),
    }


def _is_auto_center_waypoint(region: dict[str, Any], waypoint: dict[str, Any]) -> bool:
    name = str(waypoint.get("name") or "").lower().replace(" ", "_")
    if not (name.endswith("_center") or name.endswith("_entry") or name == "center"):
        return False
    centroid = region.get("centroid")
    if not isinstance(centroid, dict):
        return False
    try:
        dx = float(waypoint.get("x", 0.0) or 0.0) - float(centroid.get("x", 0.0) or 0.0)
        dy = float(waypoint.get("y", 0.0) or 0.0) - float(centroid.get("y", 0.0) or 0.0)
    except Exception:
        return False
    return (dx * dx + dy * dy) ** 0.5 <= 0.10


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
            centerline_penalty = 1.0 + (1.25 / max(clearance, float(grid["resolution"]) * 0.25))
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


def _path_length_m(points: list[dict[str, Any]]) -> float:
    return sum(_pose_distance_m(previous, nxt) for previous, nxt in zip(points, points[1:]))


def _pose_distance_m(a: dict[str, Any] | None, b: dict[str, Any] | None) -> float:
    if not isinstance(a, dict) or not isinstance(b, dict):
        return 0.0
    return math.hypot(float(a.get("x", 0.0) or 0.0) - float(b.get("x", 0.0) or 0.0), float(a.get("y", 0.0) or 0.0) - float(b.get("y", 0.0) or 0.0))


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
