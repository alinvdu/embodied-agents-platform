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

from collections import deque
import math
from typing import Mapping, TypeVar

from xlerobot_agent.exploration import Pose2D
from xlerobot_playground.semantic_evidence import SemanticAnchorCandidate, SemanticEvidence


CellT = TypeVar("CellT")


def build_semantic_anchor_candidate(
    *,
    anchor_id: str,
    evidence: SemanticEvidence,
    known_cells: Mapping[CellT, str],
    resolution: float,
    robot_cell: CellT,
    min_radius_m: float = 0.6,
    max_radius_m: float = 2.5,
    ideal_distance_m: float = 1.4,
) -> SemanticAnchorCandidate:
    free_cells = {cell for cell, state in known_cells.items() if state == "free"}
    reachable = _reachable_cells(robot_cell, free_cells)
    evidence_cell = _world_to_cell_like(robot_cell, evidence.evidence_pose.x, evidence.evidence_pose.y, resolution)
    candidates: list[tuple[float, CellT, float, float]] = []
    for cell in reachable:
        pose = _cell_center_pose(cell, resolution)
        distance = math.hypot(pose.x - evidence.evidence_pose.x, pose.y - evidence.evidence_pose.y)
        if distance < min_radius_m or distance > max_radius_m:
            continue
        line_of_sight = _line_of_sight_score(cell, evidence_cell, known_cells)
        clearance = _clearance_score(cell, free_cells)
        score = (
            evidence.confidence
            + line_of_sight * 0.4
            + clearance * 0.3
            - abs(distance - ideal_distance_m) * 0.2
        )
        candidates.append((score, cell, distance, line_of_sight))

    if not candidates:
        return SemanticAnchorCandidate(
            anchor_id=anchor_id,
            label_hint=evidence.label_hint,
            anchor_pose=evidence.evidence_pose,
            evidence_pose=evidence.evidence_pose,
            source_evidence_ids=(evidence.evidence_id,),
            source_frame_ids=evidence.source_frame_ids,
            confidence=evidence.confidence,
            reachability_status="unreachable",
            free_space_path_distance_m=None,
            line_of_sight_score=0.0,
            evidence=evidence.evidence,
            status="rejected",
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    _, best_cell, distance_m, line_of_sight = candidates[0]
    best_pose = _cell_center_pose(best_cell, resolution)
    yaw = math.atan2(evidence.evidence_pose.y - best_pose.y, evidence.evidence_pose.x - best_pose.x)
    return SemanticAnchorCandidate(
        anchor_id=anchor_id,
        label_hint=evidence.label_hint,
        anchor_pose=Pose2D(best_pose.x, best_pose.y, yaw),
        evidence_pose=evidence.evidence_pose,
        source_evidence_ids=(evidence.evidence_id,),
        source_frame_ids=evidence.source_frame_ids,
        confidence=evidence.confidence,
        reachability_status="reachable",
        free_space_path_distance_m=distance_m,
        line_of_sight_score=line_of_sight,
        evidence=evidence.evidence,
    )


def _reachable_cells(start: CellT, free_cells: set[CellT]) -> set[CellT]:
    if start not in free_cells:
        return set()
    reached = {start}
    queue: deque[CellT] = deque([start])
    while queue:
        cell = queue.popleft()
        for neighbor in _neighbors4_like(cell):
            if neighbor in free_cells and neighbor not in reached:
                reached.add(neighbor)
                queue.append(neighbor)
    return reached


def _line_of_sight_score(start: CellT, goal: CellT, known_cells: Mapping[CellT, str]) -> float:
    cells = _bresenham_like(start, goal)
    if not cells:
        return 0.0
    clear = 0
    for cell in cells:
        state = known_cells.get(cell)
        if state == "occupied":
            return 0.0
        if state == "free":
            clear += 1
    return clear / len(cells)


def _clearance_score(cell: CellT, free_cells: set[CellT]) -> float:
    neighbors = _neighbors8_like(cell)
    if not neighbors:
        return 0.0
    return sum(1 for item in neighbors if item in free_cells) / len(neighbors)


def _cell_xy(cell: object) -> tuple[int, int]:
    if hasattr(cell, "x") and hasattr(cell, "y"):
        return int(getattr(cell, "x")), int(getattr(cell, "y"))
    x, y = cell  # type: ignore[misc]
    return int(x), int(y)


def _make_cell_like(template: CellT, x: int, y: int) -> CellT:
    if hasattr(template, "__class__") and hasattr(template, "x") and hasattr(template, "y"):
        return template.__class__(x, y)  # type: ignore[call-arg,return-value]
    return (x, y)  # type: ignore[return-value]


def _world_to_cell_like(template: CellT, x: float, y: float, resolution: float) -> CellT:
    return _make_cell_like(template, int(math.floor(x / resolution)), int(math.floor(y / resolution)))


def _cell_center_pose(cell: CellT, resolution: float) -> Pose2D:
    x, y = _cell_xy(cell)
    return Pose2D((x + 0.5) * resolution, (y + 0.5) * resolution, 0.0)


def _neighbors4_like(cell: CellT) -> list[CellT]:
    x, y = _cell_xy(cell)
    return [_make_cell_like(cell, x + dx, y + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))]


def _neighbors8_like(cell: CellT) -> list[CellT]:
    x, y = _cell_xy(cell)
    return [
        _make_cell_like(cell, x + dx, y + dy)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if dx or dy
    ]


def _bresenham_like(start: CellT, goal: CellT) -> list[CellT]:
    x0, y0 = _cell_xy(start)
    x1, y1 = _cell_xy(goal)
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    cells: list[CellT] = []
    while True:
        cells.append(_make_cell_like(start, x, y))
        if x == x1 and y == y1:
            return cells
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
