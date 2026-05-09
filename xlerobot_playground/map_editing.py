from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from typing import Any, Iterable, Protocol

from xlerobot_agent.exploration import Pose2D


class CellLike(Protocol):
    x: int
    y: int


class OccupancyMapLike(Protocol):
    resolution: float
    width: int
    height: int
    origin_x: float
    origin_y: float

    def in_bounds(self, cell_x: int, cell_y: int) -> bool:
        ...

    def value(self, cell_x: int, cell_y: int) -> int:
        ...

    def is_unknown(self, cell_x: int, cell_y: int) -> bool:
        ...

    def is_free(self, cell_x: int, cell_y: int) -> bool:
        ...

    def is_occupied(self, cell_x: int, cell_y: int) -> bool:
        ...

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        ...

    def cell_to_pose(self, cell_x: int, cell_y: int, *, yaw: float = 0.0) -> Pose2D:
        ...

    def bounds(self) -> dict[str, float]:
        ...


@dataclass
class ManualOccupancyEdits:
    blocked_cells: set[Any] = field(default_factory=set)
    cleared_cells: set[Any] = field(default_factory=set)

    def clone(self) -> "ManualOccupancyEdits":
        return ManualOccupancyEdits(
            blocked_cells=set(self.blocked_cells),
            cleared_cells=set(self.cleared_cells),
        )

    def set_blocked(self, cell: Any) -> None:
        self.cleared_cells.discard(cell)
        self.blocked_cells.add(cell)

    def set_cleared(self, cell: Any) -> None:
        self.blocked_cells.discard(cell)
        self.cleared_cells.add(cell)

    def clear_override(self, cell: Any) -> None:
        self.blocked_cells.discard(cell)
        self.cleared_cells.discard(cell)

    def apply(
        self,
        *,
        cells: Iterable[Any],
        mode: str,
    ) -> None:
        normalized_mode = str(mode).strip().lower()
        for cell in cells:
            if normalized_mode == "block":
                self.set_blocked(cell)
            elif normalized_mode == "clear":
                self.set_cleared(cell)
            elif normalized_mode == "reset":
                self.clear_override(cell)

    def state_for_cell(self, base_state: str | None, cell: Any) -> str | None:
        if _cell_set_contains(self.blocked_cells, cell):
            return "occupied"
        if _cell_set_contains(self.cleared_cells, cell):
            return "free"
        return base_state

    def to_dict(self, *, resolution: float, origin_x: float = 0.0, origin_y: float = 0.0) -> dict[str, Any]:
        def _serialize(cells: set[Any]) -> list[dict[str, Any]]:
            payload: list[dict[str, Any]] = []
            for cell in sorted(cells, key=lambda item: (item.x, item.y)):
                payload.append(
                    {
                        "cell_x": int(cell.x),
                        "cell_y": int(cell.y),
                        "x": round(float(origin_x) + float(cell.x) * resolution, 3),
                        "y": round(float(origin_y) + float(cell.y) * resolution, 3),
                    }
                )
            return payload

        return {
            "blocked_cells": _serialize(self.blocked_cells),
            "cleared_cells": _serialize(self.cleared_cells),
        }


@dataclass(frozen=True)
class OccupancyFusionConfig:
    occupied_observation_weight: float = 2.0
    free_observation_weight: float = -1.0
    occupied_enter_threshold: float = 2.0
    occupied_exit_threshold: float = 0.5
    min_score: float = -4.0
    max_score: float = 8.0


DEFAULT_OCCUPANCY_FUSION_CONFIG = OccupancyFusionConfig()
ACTIVE_RGBD_SCAN_FUSION_CONFIG = OccupancyFusionConfig(
    occupied_observation_weight=1.0,
    free_observation_weight=-0.35,
    occupied_enter_threshold=2.0,
    occupied_exit_threshold=0.05,
    min_score=-3.0,
    max_score=5.0,
)


def cell_from_world(
    *,
    cell_type: type,
    resolution: float,
    x: float,
    y: float,
) -> Any:
    return cell_type(int(math.floor(x / resolution)), int(math.floor(y / resolution)))


def overlay_known_cells(
    known_cells: dict[Any, str],
    edits: ManualOccupancyEdits,
) -> dict[Any, str]:
    overlaid = dict(known_cells)
    for cell in edits.blocked_cells:
        overlaid[cell] = "occupied"
    for cell in edits.cleared_cells:
        overlaid[cell] = "free"
    return overlaid


def fuse_occupancy_state(
    existing_state: str | None,
    observed_state: str,
) -> str:
    """Merge one binary observation without evidence history.

    This is intentionally conservative for callers that do not track evidence:
    a later free observation should not erase a previously observed obstacle.
    Runtime exploration code should prefer ``merge_occupancy_observation`` with
    an evidence score dictionary so repeated free observations can clear noisy
    RGB-D endpoint hits without losing real walls immediately.
    """

    normalized_observed = str(observed_state).strip().lower()
    if normalized_observed == "occupied":
        return "occupied"
    if normalized_observed == "free":
        return "occupied" if existing_state == "occupied" else "free"
    return existing_state if existing_state is not None else normalized_observed


def merge_occupancy_observation(
    known_cells: dict[Any, str],
    cell: Any,
    observed_state: str,
    *,
    evidence_scores: dict[Any, float] | None = None,
    config: OccupancyFusionConfig = DEFAULT_OCCUPANCY_FUSION_CONFIG,
) -> None:
    normalized_observed = str(observed_state).strip().lower()
    if evidence_scores is None:
        known_cells[cell] = fuse_occupancy_state(known_cells.get(cell), normalized_observed)
        return

    existing_state = known_cells.get(cell)
    previous_score = evidence_scores.get(cell)
    if previous_score is None:
        previous_score = config.occupied_enter_threshold if existing_state == "occupied" else 0.0

    if normalized_observed == "occupied":
        updated_score = previous_score + config.occupied_observation_weight
    elif normalized_observed == "free":
        updated_score = previous_score + config.free_observation_weight
    else:
        return
    updated_score = max(config.min_score, min(config.max_score, updated_score))
    evidence_scores[cell] = updated_score

    if updated_score >= config.occupied_enter_threshold:
        known_cells[cell] = "occupied"
        return
    if existing_state == "occupied" and updated_score > config.occupied_exit_threshold:
        known_cells[cell] = "occupied"
        return
    if normalized_observed == "free" or existing_state is not None:
        known_cells[cell] = "free"


def merge_occupancy_observations(
    known_cells: dict[Any, str],
    observations: dict[Any, str] | Iterable[tuple[Any, str]],
    *,
    evidence_scores: dict[Any, float] | None = None,
    config: OccupancyFusionConfig = DEFAULT_OCCUPANCY_FUSION_CONFIG,
) -> None:
    items = observations.items() if isinstance(observations, dict) else observations
    for cell, state in items:
        merge_occupancy_observation(
            known_cells,
            cell,
            state,
            evidence_scores=evidence_scores,
            config=config,
        )


def overlay_occupancy_payload(
    occupancy: dict[str, Any] | None,
    *,
    edits: ManualOccupancyEdits,
) -> dict[str, Any] | None:
    if not isinstance(occupancy, dict):
        return occupancy
    resolution = float(occupancy.get("resolution", 0.25) or 0.25)
    bounds = occupancy.get("bounds") if isinstance(occupancy.get("bounds"), dict) else {}
    origin_x = float(bounds.get("min_x", 0.0) or 0.0)
    origin_y = float(bounds.get("min_y", 0.0) or 0.0)
    index: dict[tuple[int, int], dict[str, Any]] = {}
    for item in occupancy.get("cells", []):
        if item.get("manual_override"):
            continue
        cell_x = int(math.floor((float(item["x"]) - origin_x) / resolution))
        cell_y = int(math.floor((float(item["y"]) - origin_y) / resolution))
        index[(cell_x, cell_y)] = dict(item)
    for cell in edits.blocked_cells:
        index[(int(cell.x), int(cell.y))] = {
            "x": round(origin_x + float(cell.x) * resolution, 3),
            "y": round(origin_y + float(cell.y) * resolution, 3),
            "state": "occupied",
            "manual_override": "blocked",
        }
    for cell in edits.cleared_cells:
        index[(int(cell.x), int(cell.y))] = {
            "x": round(origin_x + float(cell.x) * resolution, 3),
            "y": round(origin_y + float(cell.y) * resolution, 3),
            "state": "free",
            "manual_override": "cleared",
        }
    payload = dict(occupancy)
    payload["cells"] = sorted(index.values(), key=lambda item: (float(item["y"]), float(item["x"])))
    return payload


@dataclass(frozen=True)
class EditableOccupancyMap:
    base_map: OccupancyMapLike
    edits: ManualOccupancyEdits

    @property
    def resolution(self) -> float:
        return float(self.base_map.resolution)

    @property
    def width(self) -> int:
        return int(self.base_map.width)

    @property
    def height(self) -> int:
        return int(self.base_map.height)

    @property
    def origin_x(self) -> float:
        return float(self.base_map.origin_x)

    @property
    def origin_y(self) -> float:
        return float(self.base_map.origin_y)

    def in_bounds(self, cell_x: int, cell_y: int) -> bool:
        return self.base_map.in_bounds(cell_x, cell_y)

    def value(self, cell_x: int, cell_y: int) -> int:
        cell_key = _TupleCell(cell_x, cell_y)
        if _cell_set_contains(self.edits.blocked_cells, cell_key):
            return 100
        if _cell_set_contains(self.edits.cleared_cells, cell_key):
            return 0
        return int(self.base_map.value(cell_x, cell_y))

    def is_unknown(self, cell_x: int, cell_y: int) -> bool:
        return self.value(cell_x, cell_y) < 0

    def is_free(self, cell_x: int, cell_y: int) -> bool:
        return self.value(cell_x, cell_y) == 0

    def is_occupied(self, cell_x: int, cell_y: int) -> bool:
        value = self.value(cell_x, cell_y)
        return value > 50 or value == 100

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        return self.base_map.world_to_cell(x, y)

    def cell_to_pose(self, cell_x: int, cell_y: int, *, yaw: float = 0.0) -> Pose2D:
        return self.base_map.cell_to_pose(cell_x, cell_y, yaw=yaw)

    def bounds(self) -> dict[str, float]:
        return self.base_map.bounds()

    def to_payload(self) -> dict[str, Any]:
        cells: list[dict[str, Any]] = []
        for y in range(self.height):
            for x in range(self.width):
                value = self.value(x, y)
                if value < 0:
                    continue
                pose = self.cell_to_pose(x, y)
                cell_key = _TupleCell(x, y)
                manual_override = None
                if _cell_set_contains(self.edits.blocked_cells, cell_key):
                    manual_override = "blocked"
                elif _cell_set_contains(self.edits.cleared_cells, cell_key):
                    manual_override = "cleared"
                payload = {
                    "x": round(pose.x - self.resolution / 2.0, 3),
                    "y": round(pose.y - self.resolution / 2.0, 3),
                    "state": "free" if value == 0 else "occupied",
                }
                if manual_override:
                    payload["manual_override"] = manual_override
                cells.append(payload)
        return {
            "resolution": self.resolution,
            "bounds": self.bounds(),
            "cells": cells,
        }


@dataclass(frozen=True)
class _TupleCell:
    x: int
    y: int


def _cell_set_contains(cells: set[Any], cell: Any) -> bool:
    if cell in cells:
        return True
    cell_x = getattr(cell, "x", None)
    cell_y = getattr(cell, "y", None)
    if cell_x is None or cell_y is None:
        return False
    return any(getattr(item, "x", None) == cell_x and getattr(item, "y", None) == cell_y for item in cells)


def edits_from_payload(
    payload: dict[str, Any] | None,
    *,
    cell_type: type,
) -> ManualOccupancyEdits:
    if not isinstance(payload, dict):
        return ManualOccupancyEdits()

    def _parse(items: Any) -> set[Any]:
        cells: set[Any] = set()
        if not isinstance(items, list):
            return cells
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                cells.add(cell_type(int(item["cell_x"]), int(item["cell_y"])))
            except Exception:
                continue
        return cells

    return ManualOccupancyEdits(
        blocked_cells=_parse(payload.get("blocked_cells")),
        cleared_cells=_parse(payload.get("cleared_cells")),
    )


def json_clone(payload: Any) -> Any:
    return json.loads(json.dumps(payload))
