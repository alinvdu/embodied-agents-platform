from __future__ import annotations

import base64
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import threading
import time
from typing import Any, Protocol
import uuid

from .models import ExecutionStatus


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"x": round(self.x, 3), "y": round(self.y, 3), "yaw": round(self.yaw, 3)}


@dataclass(frozen=True)
class ExplorationFrame:
    frame_id: str
    pose: Pose2D
    region_id: str
    visible_objects: tuple[str, ...]
    point_count: int
    depth_min_m: float
    depth_max_m: float
    description: str
    thumbnail_data_url: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "pose": self.pose.to_dict(),
            "region_id": self.region_id,
            "visible_objects": list(self.visible_objects),
            "point_count": self.point_count,
            "depth_min_m": self.depth_min_m,
            "depth_max_m": self.depth_max_m,
            "description": self.description,
            "thumbnail_data_url": self.thumbnail_data_url,
        }


@dataclass(frozen=True)
class RegionTemplate:
    region_id: str
    polygon_2d: tuple[tuple[float, float], ...]
    objects: tuple[str, ...]
    descriptions: tuple[str, ...]
    viewpoints: tuple[Pose2D, ...]
    adjacency: tuple[str, ...]


@dataclass(frozen=True)
class ExplorationScenario:
    scenario_id: str
    area: str
    start_pose: Pose2D
    templates: tuple[RegionTemplate, ...]


class RegionLabelingProvider(Protocol):
    def label_region(self, region_id: str, objects: tuple[str, ...], descriptions: tuple[str, ...]) -> tuple[str, float, list[str]]:
        ...


class RGBDFrameProvider(Protocol):
    def reset(self, area: str, current_pose: str) -> None:
        ...


class PoseTrackingProvider(Protocol):
    def reset(self) -> None:
        ...


class ExplorationMotionProvider(Protocol):
    def reset(self, area: str) -> None:
        ...


class MapStore(Protocol):
    def load_snapshot(self) -> dict[str, Any] | None:
        ...

    def save_snapshot(self, payload: dict[str, Any]) -> None:
        ...


@dataclass
class ExplorationBackendConfig:
    mode: str = "sim"
    persist_path: str | None = None
    restore_persisted_state: bool = True
    step_interval_s: float = 0.05
    occupancy_resolution: float = 0.5


@dataclass
class _TaskState:
    task_id: str
    tool_id: str
    area: str
    session: str
    source: str
    state: str = ExecutionStatus.IN_PROGRESS.value
    progress: float = 0.0
    message: str = ""
    result: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    paused: bool = False
    canceled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "tool_id": self.tool_id,
            "area": self.area,
            "session": self.session,
            "source": self.source,
            "state": self.state,
            "progress": round(self.progress, 3),
            "message": self.message,
            "result": dict(self.result),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "paused": self.paused,
            "canceled": self.canceled,
        }


class HeuristicRegionLabeler:
    _rules: tuple[tuple[tuple[str, ...], str], ...] = (
        (("fridge", "sink", "oven", "counter"), "kitchen"),
        (("sofa", "tv", "coffee table"), "living_room"),
        (("bed", "wardrobe", "pillow"), "bedroom"),
        (("desk", "monitor", "chair"), "office"),
        (("charging dock", "shoe rack"), "hallway"),
    )

    def label_region(
        self,
        region_id: str,
        objects: tuple[str, ...],
        descriptions: tuple[str, ...],
    ) -> tuple[str, float, list[str]]:
        observed = {item.lower() for item in objects}
        description_text = " ".join(descriptions).lower()
        for keywords, label in self._rules:
            evidence = [keyword for keyword in keywords if keyword in observed or keyword in description_text]
            if evidence:
                confidence = min(0.55 + 0.1 * len(evidence), 0.96)
                return label, confidence, [f"{item} visible" for item in evidence]
        fallback = region_id.replace("region_", "").replace("_", " ").strip() or "unknown_area"
        return fallback, 0.45, ["geometry-only fallback"]


class FileMapStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load_snapshot(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        return json.loads(self.path.read_text())

    def save_snapshot(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2))


def _overlay_occupancy_payload_with_manual_edits(
    occupancy: dict[str, Any] | None,
    edits: dict[str, Any],
) -> dict[str, Any] | None:
    if not isinstance(occupancy, dict):
        return occupancy
    resolution = float(occupancy.get("resolution", 0.25) or 0.25)
    index: dict[tuple[int, int], dict[str, Any]] = {}
    for item in occupancy.get("cells", []):
        if not isinstance(item, dict):
            continue
        try:
            cell_x = int(math.floor(float(item["x"]) / resolution))
            cell_y = int(math.floor(float(item["y"]) / resolution))
        except Exception:
            continue
        index[(cell_x, cell_y)] = dict(item)

    def _apply(items: Any, *, state: str, override: str) -> None:
        if not isinstance(items, list):
            return
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                cell_x = int(item["cell_x"])
                cell_y = int(item["cell_y"])
            except Exception:
                continue
            index[(cell_x, cell_y)] = {
                "x": round(cell_x * resolution, 3),
                "y": round(cell_y * resolution, 3),
                "state": state,
                "manual_override": override,
            }

    _apply(edits.get("blocked_cells"), state="occupied", override="blocked")
    _apply(edits.get("cleared_cells"), state="free", override="cleared")
    payload = dict(occupancy)
    payload["cells"] = sorted(index.values(), key=lambda item: (float(item["y"]), float(item["x"])))
    return payload


def _summarize_map_payload(map_payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(map_payload, dict):
        return None
    occupancy = map_payload.get("occupancy")
    cells = occupancy.get("cells", []) if isinstance(occupancy, dict) else []
    free = 0
    occupied = 0
    for item in cells:
        if not isinstance(item, dict):
            continue
        state = item.get("state")
        if state == "free":
            free += 1
        elif state == "occupied":
            occupied += 1
    return {
        "map_id": map_payload.get("map_id"),
        "mode": map_payload.get("mode"),
        "frame": map_payload.get("frame"),
        "coverage": map_payload.get("coverage"),
        "cell_count": len(cells) if isinstance(cells, list) else 0,
        "free_cells": free,
        "occupied_cells": occupied,
        "bounds": occupancy.get("bounds") if isinstance(occupancy, dict) else None,
    }


class ExplorationBackend:
    def __init__(
        self,
        config: ExplorationBackendConfig | None = None,
        *,
        labeler: RegionLabelingProvider | None = None,
    ) -> None:
        self.config = config or ExplorationBackendConfig()
        self.labeler = labeler or HeuristicRegionLabeler()
        self._lock = threading.RLock()
        self._tasks: dict[str, _TaskState] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._current_map: dict[str, Any] | None = None
        self._maps: dict[str, dict[str, Any]] = {}
        self._manual_occupancy_edits: dict[str, dict[str, Any]] = {}
        self._map_store: FileMapStore | None = (
            FileMapStore(self.config.persist_path) if self.config.persist_path is not None else None
        )
        if self.config.restore_persisted_state:
            self._restore()
        self._persist()

    def start_explore(
        self,
        *,
        area: str,
        session: str | None = None,
        source: str = "planner",
        build_map: bool = True,
        world_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._start_task(
            tool_id="explore",
            area=area,
            session=session or f"explore_{uuid.uuid4().hex[:8]}",
            source=source,
            build_map=build_map,
            world_state=world_state or {},
        )

    def start_create_map(
        self,
        *,
        session: str,
        area: str,
        source: str = "planner",
        world_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._start_task(
            tool_id="create_map",
            area=area,
            session=session,
            source=source,
            build_map=True,
            world_state=world_state or {},
        )

    def begin_external_task(
        self,
        *,
        tool_id: str,
        area: str,
        session: str,
        source: str = "operator",
        message: str | None = None,
    ) -> dict[str, Any]:
        task = _TaskState(
            task_id=f"{tool_id}_{uuid.uuid4().hex[:8]}",
            tool_id=tool_id,
            area=area or "workspace",
            session=session,
            source=source,
            message=message or f"Accepted external `{tool_id}` request for `{area or 'workspace'}`.",
        )
        with self._lock:
            self._tasks[task.task_id] = task
            self._persist()
            return task.to_dict()

    def update_external_task(
        self,
        task_id: str,
        *,
        progress: float | None = None,
        message: str | None = None,
        result: dict[str, Any] | None = None,
        state: str | None = None,
        map_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            if progress is not None:
                task.progress = max(0.0, min(float(progress), 1.0))
            if message is not None:
                task.message = str(message)
            if result is not None:
                task.result = json.loads(json.dumps(result))
            if state is not None:
                task.state = str(state)
            if map_payload is not None:
                self._set_current_map(map_payload)
            task.updated_at = time.time()
            self._persist()
            return task.to_dict()

    def complete_external_task(
        self,
        task_id: str,
        *,
        map_payload: dict[str, Any] | None = None,
        message: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            if map_payload is not None:
                self._set_current_map(map_payload)
            task.state = ExecutionStatus.SUCCEEDED.value
            task.progress = 1.0
            task.message = message or f"Completed `{task.tool_id}` for `{task.area}`."
            if result is not None:
                task.result = json.loads(json.dumps(result))
            elif map_payload is not None:
                task.result = {
                    "map": json.loads(json.dumps(self._current_map)),
                    "coverage": float(self._current_map.get("coverage", 0.0)) if self._current_map else 0.0,
                    "region_count": len(self._current_map.get("regions", [])) if self._current_map else 0,
                }
            task.updated_at = time.time()
            self._persist()
            return task.to_dict()

    def fail_external_task(
        self,
        task_id: str,
        *,
        message: str,
        result: dict[str, Any] | None = None,
        state: str = ExecutionStatus.FAILED.value,
    ) -> dict[str, Any] | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            task.state = state
            task.message = str(message)
            if result is not None:
                task.result = json.loads(json.dumps(result))
            task.updated_at = time.time()
            self._persist()
            return task.to_dict()

    def get_task(self, task_id: str | None = None) -> dict[str, Any] | None:
        with self._lock:
            task = self._resolve_task(task_id)
            return None if task is None else task.to_dict()

    def pause_task(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            task.paused = True
            task.message = f"Paused task `{task_id}`."
            task.updated_at = time.time()
            self._persist()
            return task.to_dict()

    def resume_task(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            task.paused = False
            if task.state == ExecutionStatus.IN_PROGRESS.value:
                task.message = f"Resumed task `{task_id}`."
            task.updated_at = time.time()
            self._persist()
            return task.to_dict()

    def cancel_task(self, task_id: str | None = None) -> dict[str, Any] | None:
        with self._lock:
            task = self._resolve_task(task_id)
            if task is None:
                return None
            task.canceled = True
            task.paused = False
            task.state = ExecutionStatus.ABORTED.value
            task.message = f"Canceled task `{task.task_id}`."
            task.updated_at = time.time()
            self._persist()
            return task.to_dict()

    def get_map(self) -> dict[str, Any] | None:
        with self._lock:
            return None if self._current_map is None else json.loads(json.dumps(self._current_map))

    def list_maps(self) -> list[dict[str, Any]]:
        with self._lock:
            return [json.loads(json.dumps(item)) for item in self._maps.values()]

    def approve_current_map(self) -> dict[str, Any] | None:
        with self._lock:
            if self._current_map is None:
                return None
            self._current_map["approved"] = True
            self._current_map["approved_at"] = time.time()
            self._maps[self._current_map["map_id"]] = json.loads(json.dumps(self._current_map))
            self._persist()
            return json.loads(json.dumps(self._current_map))

    def update_region(
        self,
        region_id: str,
        *,
        label: str | None = None,
        polygon_2d: list[list[float]] | None = None,
        default_waypoints: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        with self._lock:
            region = self._find_region(region_id)
            if region is None:
                return None
            if label is not None:
                region["label"] = str(label)
            if polygon_2d is not None:
                region["polygon_2d"] = [[float(x), float(y)] for x, y in polygon_2d]
                region["centroid"] = _polygon_centroid(region["polygon_2d"])
            if default_waypoints is not None:
                region["default_waypoints"] = [_json_pose(item) for item in default_waypoints]
            self._rebuild_named_places()
            self._persist()
            return json.loads(json.dumps(region))

    def merge_regions(self, region_ids: list[str], *, new_label: str | None = None) -> dict[str, Any] | None:
        with self._lock:
            if self._current_map is None or len(region_ids) < 2:
                return None
            selected = [region for region in self._current_map["regions"] if region["region_id"] in set(region_ids)]
            if len(selected) < 2:
                return None
            merged_polygon = _bounding_polygon(point for region in selected for point in region["polygon_2d"])
            merged_region = {
                "region_id": f"region_{uuid.uuid4().hex[:8]}",
                "label": new_label or selected[0]["label"],
                "confidence": round(sum(region["confidence"] for region in selected) / len(selected), 3),
                "polygon_2d": merged_polygon,
                "centroid": _polygon_centroid(merged_polygon),
                "adjacency": sorted({item for region in selected for item in region.get("adjacency", []) if item not in set(region_ids)}),
                "representative_keyframes": [item for region in selected for item in region.get("representative_keyframes", [])][:6],
                "evidence": [item for region in selected for item in region.get("evidence", [])][:8],
                "default_waypoints": [
                    {
                        "name": f"{(new_label or selected[0]['label']).replace(' ', '_')}_center",
                        **_json_pose({"x": _polygon_centroid(merged_polygon)["x"], "y": _polygon_centroid(merged_polygon)["y"], "yaw": 0.0}),
                    }
                ],
            }
            self._current_map["regions"] = [
                region
                for region in self._current_map["regions"]
                if region["region_id"] not in set(region_ids)
            ]
            self._current_map["regions"].append(merged_region)
            self._rebuild_named_places()
            self._persist()
            return json.loads(json.dumps(merged_region))

    def split_region(self, region_id: str, polygons: list[list[list[float]]] | None = None) -> list[dict[str, Any]]:
        with self._lock:
            region = self._find_region(region_id)
            if region is None:
                return []
            if polygons is None:
                polygons = _split_polygon_vertically(region["polygon_2d"])
            if len(polygons) < 2:
                return []
            self._current_map["regions"] = [
                item for item in self._current_map["regions"] if item["region_id"] != region_id
            ]
            created: list[dict[str, Any]] = []
            for index, polygon in enumerate(polygons, start=1):
                centroid = _polygon_centroid(polygon)
                created_region = {
                    "region_id": f"{region_id}_part_{index}",
                    "label": f"{region['label']}_{index}",
                    "confidence": max(float(region.get("confidence", 0.5)) - 0.05, 0.3),
                    "polygon_2d": [[float(x), float(y)] for x, y in polygon],
                    "centroid": centroid,
                    "adjacency": list(region.get("adjacency", [])),
                    "representative_keyframes": list(region.get("representative_keyframes", [])),
                    "evidence": list(region.get("evidence", [])),
                    "default_waypoints": [
                        {
                            "name": f"{region['label']}_{index}_center",
                            **_json_pose({"x": centroid["x"], "y": centroid["y"], "yaw": 0.0}),
                        }
                    ],
                }
                created.append(created_region)
            self._current_map["regions"].extend(created)
            self._rebuild_named_places()
            self._persist()
            return json.loads(json.dumps(created))

    def set_named_place(self, name: str, pose: dict[str, Any], *, region_id: str | None = None) -> dict[str, Any] | None:
        with self._lock:
            if self._current_map is None:
                return None
            named_places = list(self._current_map.get("named_places", []))
            named_places = [item for item in named_places if item.get("name") != name]
            named_place = {
                "name": name,
                "pose": _json_pose(pose),
                "region_id": region_id,
                "source": "manual",
            }
            named_places.append(named_place)
            self._current_map["named_places"] = named_places
            self._persist()
            return json.loads(json.dumps(named_place))

    def occupancy_edit_snapshot(self, task_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            task = self._resolve_task(task_id)
            if task is None:
                return {"blocked_cells": [], "cleared_cells": []}
            payload = self._manual_occupancy_edits.get(task.task_id)
            if not isinstance(payload, dict):
                return {"blocked_cells": [], "cleared_cells": []}
            return json.loads(json.dumps(payload))

    def update_occupancy_edits(
        self,
        *,
        task_id: str | None = None,
        mode: str,
        cells: list[dict[str, Any]],
    ) -> dict[str, Any]:
        normalized_mode = str(mode).strip().lower()
        with self._lock:
            task = self._resolve_task(task_id)
            if task is None:
                return {"blocked_cells": [], "cleared_cells": []}
            edits = json.loads(json.dumps(self._manual_occupancy_edits.get(task.task_id, {"blocked_cells": [], "cleared_cells": []})))
            blocked = {(int(item["cell_x"]), int(item["cell_y"])) for item in edits.get("blocked_cells", []) if "cell_x" in item and "cell_y" in item}
            cleared = {(int(item["cell_x"]), int(item["cell_y"])) for item in edits.get("cleared_cells", []) if "cell_x" in item and "cell_y" in item}
            for item in cells:
                if not isinstance(item, dict):
                    continue
                try:
                    if "cell_x" in item and "cell_y" in item:
                        cell = (int(item["cell_x"]), int(item["cell_y"]))
                    else:
                        resolution = float(self._current_map.get("occupancy", {}).get("resolution", self.config.occupancy_resolution)) if self._current_map else self.config.occupancy_resolution
                        cell = (
                            int(float(item["x"]) // resolution),
                            int(float(item["y"]) // resolution),
                        )
                except Exception:
                    continue
                if normalized_mode == "block":
                    cleared.discard(cell)
                    blocked.add(cell)
                elif normalized_mode == "clear":
                    blocked.discard(cell)
                    cleared.add(cell)
                elif normalized_mode == "reset":
                    blocked.discard(cell)
                    cleared.discard(cell)
            resolution = float(self._current_map.get("occupancy", {}).get("resolution", self.config.occupancy_resolution)) if self._current_map else self.config.occupancy_resolution
            payload = {
                "blocked_cells": [
                    {
                        "cell_x": cell_x,
                        "cell_y": cell_y,
                        "x": round(cell_x * resolution, 3),
                        "y": round(cell_y * resolution, 3),
                    }
                    for cell_x, cell_y in sorted(blocked)
                ],
                "cleared_cells": [
                    {
                        "cell_x": cell_x,
                        "cell_y": cell_y,
                        "x": round(cell_x * resolution, 3),
                        "y": round(cell_y * resolution, 3),
                    }
                    for cell_x, cell_y in sorted(cleared)
                ],
            }
            self._manual_occupancy_edits[task.task_id] = payload
            if self._current_map is not None:
                self._attach_manual_edits(self._current_map, task.task_id)
            self._persist()
            return json.loads(json.dumps(payload))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            active_task = self._resolve_task(None)
            return {
                "mode": self.config.mode,
                "current_map": None if self._current_map is None else json.loads(json.dumps(self._current_map)),
                "maps": [json.loads(json.dumps(item)) for item in self._maps.values()],
                "active_task": None if active_task is None else active_task.to_dict(),
                "tasks": [task.to_dict() for task in self._tasks.values()],
            }

    def _start_task(
        self,
        *,
        tool_id: str,
        area: str,
        session: str,
        source: str,
        build_map: bool,
        world_state: dict[str, Any],
    ) -> dict[str, Any]:
        task = _TaskState(
            task_id=f"{tool_id}_{uuid.uuid4().hex[:8]}",
            tool_id=tool_id,
            area=area or "workspace",
            session=session,
            source=source,
            message=f"Accepted `{tool_id}` request for `{area or 'workspace'}`.",
        )
        with self._lock:
            self._tasks[task.task_id] = task
            thread = threading.Thread(
                target=self._run_task,
                kwargs={
                    "task_id": task.task_id,
                    "build_map": build_map,
                    "world_state": dict(world_state),
                },
                daemon=True,
            )
            self._threads[task.task_id] = thread
            self._persist()
            thread.start()
            return task.to_dict()

    def _run_task(self, *, task_id: str, build_map: bool, world_state: dict[str, Any]) -> None:
        scenario = _build_scenario(
            session=task_id,
            area=self._tasks[task_id].area,
            current_pose=str(world_state.get("current_pose", "hallway") or "hallway"),
        )
        templates_by_id = {item.region_id: item for item in scenario.templates}
        discovered_region_ids: list[str] = []
        trajectory: list[dict[str, Any]] = []
        keyframes: list[dict[str, Any]] = []
        total_frames = sum(len(item.viewpoints) for item in scenario.templates)

        processed = 0
        for region in scenario.templates:
            if not self._wait_until_resumed(task_id):
                return
            discovered_region_ids.append(region.region_id)
            for viewpoint in region.viewpoints:
                if not self._wait_until_resumed(task_id):
                    return
                processed += 1
                frame = _make_frame(region, viewpoint, processed)
                keyframes.append(frame.to_dict())
                trajectory.append(frame.pose.to_dict())
                with self._lock:
                    task = self._tasks[task_id]
                    if task.state == ExecutionStatus.ABORTED.value:
                        return
                    task.progress = min(processed / max(total_frames, 1), 0.92)
                    task.updated_at = time.time()
                    frontier_count = _frontier_count(discovered_region_ids, scenario.templates)
                    task.message = (
                        f"Exploring `{task.area}`: visited {len(discovered_region_ids)} region(s), "
                        f"{frontier_count} frontier(s) remaining."
                    )
                    task.result = {
                        "coverage": round(len(set(discovered_region_ids)) / max(len(scenario.templates), 1), 3),
                        "trajectory": list(trajectory),
                        "keyframes": list(keyframes[-8:]),
                    }
                    self._persist()
                time.sleep(self.config.step_interval_s)

        map_payload = self._build_map_payload(
            task=self._tasks[task_id],
            scenario=scenario,
            templates_by_id=templates_by_id,
            keyframes=keyframes,
            trajectory=trajectory,
            build_map=build_map,
        )
        with self._lock:
            task = self._tasks[task_id]
            if task.state == ExecutionStatus.ABORTED.value:
                return
            task.state = ExecutionStatus.SUCCEEDED.value
            task.progress = 1.0
            task.updated_at = time.time()
            task.message = f"Completed `{task.tool_id}` for `{task.area}`."
            task.result = {
                "map": map_payload,
                "coverage": map_payload["coverage"],
                "region_count": len(map_payload["regions"]),
            }
            self._current_map = json.loads(json.dumps(map_payload))
            self._maps[map_payload["map_id"]] = json.loads(json.dumps(map_payload))
            self._persist()

    def _build_map_payload(
        self,
        *,
        task: _TaskState,
        scenario: ExplorationScenario,
        templates_by_id: dict[str, RegionTemplate],
        keyframes: list[dict[str, Any]],
        trajectory: list[dict[str, Any]],
        build_map: bool,
    ) -> dict[str, Any]:
        keyframes_by_region: dict[str, list[dict[str, Any]]] = {}
        for keyframe in keyframes:
            keyframes_by_region.setdefault(str(keyframe["region_id"]), []).append(keyframe)

        regions: list[dict[str, Any]] = []
        for template in scenario.templates:
            region_keyframes = keyframes_by_region.get(template.region_id, [])
            label, confidence, evidence = self.labeler.label_region(
                template.region_id,
                template.objects,
                template.descriptions,
            )
            centroid = _polygon_centroid(template.polygon_2d)
            regions.append(
                {
                    "region_id": template.region_id,
                    "label": label,
                    "confidence": round(confidence, 3),
                    "polygon_2d": [[float(x), float(y)] for x, y in template.polygon_2d],
                    "centroid": centroid,
                    "adjacency": list(template.adjacency),
                    "representative_keyframes": [item["frame_id"] for item in region_keyframes[:3]],
                    "evidence": evidence,
                    "default_waypoints": [
                        {
                            "name": f"{label.replace(' ', '_')}_center",
                            **_json_pose({"x": centroid["x"], "y": centroid["y"], "yaw": 0.0}),
                        }
                    ],
                }
            )

        map_id = task.session
        occupancy = _rasterize_regions(
            [region["polygon_2d"] for region in regions],
            resolution=self.config.occupancy_resolution,
        ) if build_map else None
        map_payload = {
            "map_id": map_id,
            "frame": "map",
            "resolution": self.config.occupancy_resolution,
            "coverage": round(1.0 if regions else 0.0, 3),
            "summary": (
                f"{self.config.mode} exploration map for `{task.area}` with {len(regions)} semantic region(s)."
            ),
            "approved": False,
            "created_at": time.time(),
            "source": task.source,
            "mode": self.config.mode,
            "trajectory": trajectory,
            "keyframes": keyframes,
            "regions": regions,
            "named_places": [],
            "occupancy": occupancy,
        }
        self._current_map = map_payload
        self._rebuild_named_places()
        return json.loads(json.dumps(self._current_map))

    def _rebuild_named_places(self) -> None:
        if self._current_map is None:
            return
        named_places: list[dict[str, Any]] = [
            item
            for item in self._current_map.get("named_places", [])
            if item.get("source") == "manual"
        ]
        for region in self._current_map.get("regions", []):
            label = str(region["label"]).replace(" ", "_")
            centroid = region.get("centroid") or _polygon_centroid(region["polygon_2d"])
            named_places.append(
                {
                    "name": f"{label}_entry",
                    "pose": _json_pose({"x": centroid["x"], "y": centroid["y"], "yaw": 0.0}),
                    "region_id": region["region_id"],
                    "source": "derived",
                }
            )
            for waypoint in region.get("default_waypoints", []):
                named_places.append(
                    {
                        "name": waypoint["name"],
                        "pose": _json_pose(waypoint),
                        "region_id": region["region_id"],
                        "source": "derived",
                    }
                )
        deduped: dict[str, dict[str, Any]] = {}
        for place in named_places:
            deduped[place["name"]] = place
        self._current_map["named_places"] = list(deduped.values())
        if self._current_map is not None:
            self._maps[self._current_map["map_id"]] = json.loads(json.dumps(self._current_map))

    def _set_current_map(self, map_payload: dict[str, Any]) -> None:
        self._current_map = json.loads(json.dumps(map_payload))
        active_task = self._resolve_task(None)
        if active_task is not None:
            self._attach_manual_edits(self._current_map, active_task.task_id)
        self._rebuild_named_places()
        if self._current_map is not None:
            self._maps[self._current_map["map_id"]] = json.loads(json.dumps(self._current_map))
        print(f"[exploration_backend] stored current_map summary={_summarize_map_payload(self._current_map)}")

    def _attach_manual_edits(self, map_payload: dict[str, Any], task_id: str) -> None:
        artifacts = map_payload.setdefault("artifacts", {})
        edits = self._manual_occupancy_edits.get(task_id, {"blocked_cells": [], "cleared_cells": []})
        artifacts["manual_occupancy_edits"] = json.loads(json.dumps(edits))
        map_payload["occupancy"] = _overlay_occupancy_payload_with_manual_edits(
            map_payload.get("occupancy"),
            edits,
        )

    def _find_region(self, region_id: str) -> dict[str, Any] | None:
        if self._current_map is None:
            return None
        for region in self._current_map.get("regions", []):
            if region["region_id"] == region_id:
                return region
        return None

    def _resolve_task(self, task_id: str | None) -> _TaskState | None:
        if task_id:
            return self._tasks.get(task_id)
        if not self._tasks:
            return None
        latest_id = next(reversed(self._tasks))
        return self._tasks[latest_id]

    def _wait_until_resumed(self, task_id: str) -> bool:
        while True:
            with self._lock:
                task = self._tasks[task_id]
                if task.state == ExecutionStatus.ABORTED.value or task.canceled:
                    return False
                if not task.paused:
                    return True
            time.sleep(0.05)

    def _persist(self) -> None:
        if self._map_store is None:
            return
        payload = self.snapshot()
        self._map_store.save_snapshot(payload)

    def _restore(self) -> None:
        if self._map_store is None:
            return
        payload = self._map_store.load_snapshot()
        if not isinstance(payload, dict):
            return
        current_map = payload.get("current_map")
        if isinstance(current_map, dict):
            self._current_map = current_map
        maps = payload.get("maps")
        if isinstance(maps, list):
            for item in maps:
                if isinstance(item, dict) and item.get("map_id"):
                    self._maps[str(item["map_id"])] = item
        tasks = payload.get("tasks")
        if isinstance(tasks, list):
            for item in tasks:
                if not isinstance(item, dict) or not item.get("task_id"):
                    continue
                task = _TaskState(
                    task_id=str(item["task_id"]),
                    tool_id=str(item.get("tool_id", "unknown")),
                    area=str(item.get("area", "workspace")),
                    session=str(item.get("session", "restored")),
                    source=str(item.get("source", "restored")),
                    state=str(item.get("state", ExecutionStatus.IN_PROGRESS.value)),
                    progress=float(item.get("progress", 0.0) or 0.0),
                    message=str(item.get("message", "")),
                    result=dict(item.get("result", {})) if isinstance(item.get("result"), dict) else {},
                    created_at=float(item.get("created_at", time.time()) or time.time()),
                    updated_at=float(item.get("updated_at", time.time()) or time.time()),
                    paused=bool(item.get("paused", False)),
                    canceled=bool(item.get("canceled", False)),
                )
                self._tasks[task.task_id] = task
        if self._tasks and self._current_map is not None:
            edits = (
                self._current_map.get("artifacts", {}).get("manual_occupancy_edits")
                if isinstance(self._current_map, dict)
                else None
            )
            if isinstance(edits, dict):
                latest_task_id = next(reversed(self._tasks))
                self._manual_occupancy_edits[latest_task_id] = json.loads(json.dumps(edits))


def _build_scenario(*, session: str, area: str, current_pose: str) -> ExplorationScenario:
    seed = hashlib.sha256(f"{session}:{area}:{current_pose}".encode("utf-8")).hexdigest()
    floorplans = {
        "hallway": RegionTemplate(
            region_id="region_hallway",
            polygon_2d=((1.0, 2.0), (5.0, 2.0), (5.0, 4.0), (1.0, 4.0)),
            objects=("shoe rack", "charging dock"),
            descriptions=("a narrow transition area with the dock and shoes",),
            viewpoints=(Pose2D(1.5, 3.0, 0.0), Pose2D(4.5, 3.0, 0.0)),
            adjacency=("region_kitchen", "region_living_room", "region_bedroom"),
        ),
        "kitchen": RegionTemplate(
            region_id="region_kitchen",
            polygon_2d=((5.0, 4.0), (9.5, 4.0), (9.5, 7.5), (5.0, 7.5)),
            objects=("fridge", "sink", "counter", "oven"),
            descriptions=("a kitchen with a fridge, sink, and counter workspace",),
            viewpoints=(Pose2D(6.0, 5.0, 0.2), Pose2D(8.4, 6.2, 1.57)),
            adjacency=("region_hallway",),
        ),
        "living_room": RegionTemplate(
            region_id="region_living_room",
            polygon_2d=((0.5, 0.5), (5.0, 0.5), (5.0, 2.0), (0.5, 2.0)),
            objects=("sofa", "tv", "coffee table"),
            descriptions=("a living room with sofa seating and a television wall",),
            viewpoints=(Pose2D(1.3, 1.2, 0.0), Pose2D(4.1, 1.3, 0.0)),
            adjacency=("region_hallway",),
        ),
        "bedroom": RegionTemplate(
            region_id="region_bedroom",
            polygon_2d=((5.4, 0.5), (9.5, 0.5), (9.5, 4.0), (5.4, 4.0)),
            objects=("bed", "wardrobe", "pillow"),
            descriptions=("a bedroom with a bed, pillow, and wardrobe",),
            viewpoints=(Pose2D(6.2, 1.2, 0.0), Pose2D(8.5, 2.7, 1.2)),
            adjacency=("region_hallway",),
        ),
    }
    requested = area.lower().replace(" ", "_")
    if any(token in requested for token in ("kitchen", "fridge", "sink")):
        order = ("hallway", "kitchen")
    elif any(token in requested for token in ("living", "sofa", "tv")):
        order = ("hallway", "living_room")
    elif any(token in requested for token in ("bed", "bedroom")):
        order = ("hallway", "bedroom")
    else:
        order = ("hallway", "kitchen", "living_room", "bedroom")
    rotation = int(seed[0], 16) % len(order)
    ordered = order[rotation:] + order[:rotation]
    if "hallway" not in ordered:
        ordered = ("hallway",) + ordered
    templates = tuple(floorplans[item] for item in ordered)
    return ExplorationScenario(
        scenario_id=f"scenario_{seed[:10]}",
        area=area or "workspace",
        start_pose=Pose2D(1.5, 3.0, 0.0),
        templates=templates,
    )


def _make_frame(region: RegionTemplate, pose: Pose2D, index: int) -> ExplorationFrame:
    description = region.descriptions[0]
    return ExplorationFrame(
        frame_id=f"kf_{index:03d}",
        pose=pose,
        region_id=region.region_id,
        visible_objects=region.objects,
        point_count=1800 + index * 120,
        depth_min_m=0.4,
        depth_max_m=4.8,
        description=description,
        thumbnail_data_url=_thumbnail_data_url(
            title=region.region_id.replace("region_", "").replace("_", " ").title(),
            subtitle=", ".join(region.objects[:3]),
        ),
    )


def _frontier_count(discovered_region_ids: list[str], templates: tuple[RegionTemplate, ...]) -> int:
    known = set(discovered_region_ids)
    remaining = 0
    for template in templates:
        if template.region_id not in known:
            remaining += 1
    return remaining


def _thumbnail_data_url(*, title: str, subtitle: str) -> str:
    svg = f"""
<svg xmlns="http://www.w3.org/2000/svg" width="240" height="140" viewBox="0 0 240 140">
  <defs>
    <linearGradient id="g" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0%" stop-color="#0f766e"/>
      <stop offset="100%" stop-color="#b45309"/>
    </linearGradient>
  </defs>
  <rect width="240" height="140" rx="18" fill="#f8fafc"/>
  <rect x="10" y="10" width="220" height="120" rx="14" fill="url(#g)" opacity="0.16"/>
  <text x="20" y="46" font-family="IBM Plex Sans, Arial, sans-serif" font-size="24" fill="#102a43">{title}</text>
  <text x="20" y="78" font-family="IBM Plex Sans, Arial, sans-serif" font-size="13" fill="#486581">{subtitle}</text>
  <circle cx="190" cy="46" r="18" fill="#0f766e" opacity="0.22"/>
  <circle cx="160" cy="88" r="28" fill="#b45309" opacity="0.12"/>
</svg>
""".strip()
    encoded = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{encoded}"


def _json_pose(pose: dict[str, Any]) -> dict[str, Any]:
    return {
        "x": round(float(pose.get("x", 0.0)), 3),
        "y": round(float(pose.get("y", 0.0)), 3),
        "yaw": round(float(pose.get("yaw", 0.0)), 3),
    }


def _polygon_centroid(polygon: Any) -> dict[str, float]:
    points = [(float(x), float(y)) for x, y in polygon]
    if not points:
        return {"x": 0.0, "y": 0.0}
    x_sum = sum(x for x, _ in points)
    y_sum = sum(y for _, y in points)
    return {"x": round(x_sum / len(points), 3), "y": round(y_sum / len(points), 3)}


def _bounding_polygon(points: Any) -> list[list[float]]:
    normalized = [(float(x), float(y)) for x, y in points]
    min_x = min(x for x, _ in normalized)
    max_x = max(x for x, _ in normalized)
    min_y = min(y for _, y in normalized)
    max_y = max(y for _, y in normalized)
    return [
        [round(min_x, 3), round(min_y, 3)],
        [round(max_x, 3), round(min_y, 3)],
        [round(max_x, 3), round(max_y, 3)],
        [round(min_x, 3), round(max_y, 3)],
    ]


def _split_polygon_vertically(polygon: list[list[float]]) -> list[list[list[float]]]:
    centroid = _polygon_centroid(polygon)
    min_x = min(point[0] for point in polygon)
    max_x = max(point[0] for point in polygon)
    min_y = min(point[1] for point in polygon)
    max_y = max(point[1] for point in polygon)
    mid_x = centroid["x"]
    return [
        [[min_x, min_y], [mid_x, min_y], [mid_x, max_y], [min_x, max_y]],
        [[mid_x, min_y], [max_x, min_y], [max_x, max_y], [mid_x, max_y]],
    ]


def _point_in_polygon(point_x: float, point_y: float, polygon: list[list[float]]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = ((yi > point_y) != (yj > point_y)) and (
            point_x < (xj - xi) * (point_y - yi) / ((yj - yi) or 1e-9) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def _rasterize_regions(polygons: list[list[list[float]]], *, resolution: float) -> dict[str, Any]:
    all_points = [tuple(point) for polygon in polygons for point in polygon]
    if not all_points:
        return {
            "resolution": resolution,
            "bounds": {"min_x": 0.0, "max_x": 0.0, "min_y": 0.0, "max_y": 0.0},
            "cells": [],
        }
    min_x = min(point[0] for point in all_points)
    max_x = max(point[0] for point in all_points)
    min_y = min(point[1] for point in all_points)
    max_y = max(point[1] for point in all_points)
    cells: list[dict[str, Any]] = []
    x = min_x
    while x <= max_x + 1e-6:
        y = min_y
        while y <= max_y + 1e-6:
            point_x = x + resolution / 2.0
            point_y = y + resolution / 2.0
            state = None
            for polygon in polygons:
                if _point_in_polygon(point_x, point_y, polygon):
                    state = "free"
                    break
            if state is not None:
                cells.append(
                    {
                        "x": round(x, 3),
                        "y": round(y, 3),
                        "state": state,
                    }
                )
            y += resolution
        x += resolution
    return {
        "resolution": resolution,
        "bounds": {
            "min_x": round(min_x, 3),
            "max_x": round(max_x, 3),
            "min_y": round(min_y, 3),
            "max_y": round(max_y, 3),
        },
        "cells": cells,
    }
