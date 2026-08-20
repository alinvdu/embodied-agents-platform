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

from typing import Any, Callable, Iterable

from xlerobot_agent.exploration import Pose2D


def refresh_frontier_records(
    *,
    candidate_records: Iterable[Any],
    active_frontier_id: str | None,
    current_pose: Pose2D,
    current_pose_filter_m: float,
    path_cost_for_record: Callable[[Any], float | None],
    guardrail_events: list[dict[str, Any]],
    is_frontier_at_current_pose: Callable[[Any, float], bool],
    max_frontiers: int | None = None,
    is_frontier_near_visited_pose: Callable[[Any, float], bool] | None = None,
    visited_pose_filter_m: float | None = None,
    global_anchor_for_stored_record: Callable[[Any], tuple[Any | None, str | None]] | None = None,
    revalidate_stored_boundary: Callable[[Any, Any, str | None], None] | None = None,
    resnap_stored_nav_pose: Callable[[Any, Pose2D, Any], Pose2D | None] | None = None,
    apply_stored_resnap: Callable[[Any, Pose2D, Pose2D], None] | None = None,
    sort_key: Callable[[Any], tuple[Any, ...]] | None = None,
) -> list[Any]:
    reachable_records: list[Any] = []
    for record in candidate_records:
        if getattr(record, "status", None) == "active" or getattr(record, "frontier_id", None) == active_frontier_id:
            setattr(record, "path_cost_m", None)
            guardrail_events.append(
                {
                    "type": "active_frontier_filtered_from_prompt",
                    "frontier_id": getattr(record, "frontier_id", None),
                }
            )
            continue

        frontier_anchor_cell: Any | None = None
        frontier_anchor_mode: str | None = None
        if not bool(getattr(record, "currently_visible", False)) and global_anchor_for_stored_record is not None:
            frontier_anchor_cell, frontier_anchor_mode = global_anchor_for_stored_record(record)
            if frontier_anchor_cell is None:
                setattr(record, "path_cost_m", None)
                guardrail_events.append(
                    {
                        "type": "stored_frontier_not_global_boundary_after_merge",
                        "frontier_id": getattr(record, "frontier_id", None),
                        "frontier_boundary_pose": getattr(getattr(record, "centroid_pose", None), "to_dict", lambda: None)(),
                    }
                )
                continue
            if revalidate_stored_boundary is not None:
                revalidate_stored_boundary(record, frontier_anchor_cell, frontier_anchor_mode)

        path_cost_m = path_cost_for_record(record)
        if (
            path_cost_m is None
            and not bool(getattr(record, "currently_visible", False))
            and frontier_anchor_cell is not None
            and resnap_stored_nav_pose is not None
            and apply_stored_resnap is not None
        ):
            previous_pose = getattr(record, "nav_pose", None)
            resnapped_pose = resnap_stored_nav_pose(record, current_pose, frontier_anchor_cell)
            if resnapped_pose is not None and previous_pose is not None:
                apply_stored_resnap(record, resnapped_pose, previous_pose)
                path_cost_m = path_cost_for_record(record)
            if path_cost_m is None:
                setattr(record, "path_cost_m", None)
                guardrail_events.append(
                    {
                        "type": "stored_frontier_without_reachable_revisit_pose",
                        "frontier_id": getattr(record, "frontier_id", None),
                        "frontier_boundary_pose": getattr(getattr(record, "centroid_pose", None), "to_dict", lambda: None)(),
                    }
                )
                continue

        setattr(record, "path_cost_m", path_cost_m)
        if path_cost_m is None:
            guardrail_events.append(
                {
                    "type": "frontier_without_reachable_nav_pose",
                    "frontier_id": getattr(record, "frontier_id", None),
                    "currently_visible": bool(getattr(record, "currently_visible", False)),
                }
            )
            continue
        if is_frontier_at_current_pose(record, current_pose_filter_m):
            setattr(record, "path_cost_m", None)
            guardrail_events.append(
                {
                    "type": "frontier_at_current_pose_filtered",
                    "frontier_id": getattr(record, "frontier_id", None),
                    "path_cost_m": round(path_cost_m, 3),
                    "filter_radius_m": round(current_pose_filter_m, 3),
                }
            )
            continue
        if (
            is_frontier_near_visited_pose is not None
            and visited_pose_filter_m is not None
            and visited_pose_filter_m > 0.0
            and is_frontier_near_visited_pose(record, visited_pose_filter_m)
        ):
            setattr(record, "path_cost_m", None)
            if getattr(record, "status", None) not in {"active", "completed", "failed"}:
                setattr(record, "status", "suppressed")
            evidence = getattr(record, "evidence", None)
            if isinstance(evidence, list):
                note = "frontier suppressed because its boundary is close to a previously visited robot pose"
                if note not in evidence:
                    evidence.append(note)
            guardrail_events.append(
                {
                    "type": "frontier_near_visited_pose_filtered",
                    "frontier_id": getattr(record, "frontier_id", None),
                    "path_cost_m": round(path_cost_m, 3),
                    "filter_radius_m": round(visited_pose_filter_m, 3),
                    "frontier_boundary_pose": getattr(getattr(record, "centroid_pose", None), "to_dict", lambda: None)(),
                }
            )
            continue
        reachable_records.append(record)

    normalized_sort_key = sort_key or (
        lambda record: (
            getattr(record, "path_cost_m", None) if getattr(record, "path_cost_m", None) is not None else 1e9,
            not bool(getattr(record, "currently_visible", False)),
            -(getattr(record, "llm_memory_priority", None) or 0.0),
            -int(getattr(record, "unknown_gain", 0)),
        )
    )
    reachable_records.sort(key=normalized_sort_key)
    if max_frontiers is None or max_frontiers <= 0:
        return reachable_records
    return reachable_records[:max_frontiers]
