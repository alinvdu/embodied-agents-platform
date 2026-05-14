from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any

from .memory_discovery import HOME_MEMORY_FILENAME, default_environment_memory_dir_for_map_path


HOME_MEMORY_SCHEMA_VERSION = "home_memory.v1"


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
                "centroid": _json_clone(region.get("centroid") or {}),
                "default_waypoints": _json_clone(region.get("default_waypoints") or []),
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
        pose = _first_pose(region.get("default_waypoints")) or region.get("centroid")
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
    return None


def _memory_region(region: dict[str, Any]) -> dict[str, Any]:
    label = str(region.get("label") or region.get("region_id") or "region")
    default_waypoints = [_json_pose(waypoint) | {"name": str(waypoint.get("name") or f"{label}_waypoint")} for waypoint in region.get("default_waypoints", []) if isinstance(waypoint, dict)]
    centroid = region.get("centroid")
    if not default_waypoints and isinstance(centroid, dict):
        default_waypoints.append({"name": f"{label.replace(' ', '_')}_center", **_json_pose(centroid)})
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
