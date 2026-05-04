from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Sequence

from xlerobot_agent.exploration import Pose2D
from xlerobot_playground.map_editing import ACTIVE_RGBD_SCAN_FUSION_CONFIG, OccupancyFusionConfig, merge_occupancy_observation


@dataclass(frozen=True)
class ScanIntegrationSummary:
    point_count: int
    scan_beams: int
    integrated_beams: int


def integrate_planar_scan(
    *,
    pose: Pose2D,
    ranges: Sequence[float],
    angle_min: float,
    angle_increment: float,
    range_min_m: float,
    range_max_m: float,
    resolution_m: float,
    cell_from_world: Callable[[float, float], Any],
    known_cells: dict[Any, str] | None,
    evidence_scores: dict[Any, float] | None,
    range_edge_cells: set[Any] | None,
    visited_cells: set[Any] | None = None,
    beam_stride: int = 1,
    config: OccupancyFusionConfig = ACTIVE_RGBD_SCAN_FUSION_CONFIG,
    obstacle_hit_ratio: float = 0.98,
) -> ScanIntegrationSummary:
    if not ranges:
        return ScanIntegrationSummary(point_count=0, scan_beams=0, integrated_beams=0)

    normalized_beam_stride = max(int(beam_stride), 1)
    normalized_range_min = max(float(range_min_m), 0.0)
    normalized_range_max = max(float(range_max_m), normalized_range_min)
    step_m = max(float(resolution_m) * 0.5, 0.05)
    origin_cell = cell_from_world(float(pose.x), float(pose.y))

    if visited_cells is not None:
        visited_cells.add(origin_cell)
    if known_cells is not None:
        merge_occupancy_observation(
            known_cells,
            origin_cell,
            "free",
            evidence_scores=evidence_scores,
            config=config,
        )

    point_count = 0
    for index in range(0, len(ranges), normalized_beam_stride):
        beam_range = float(ranges[index])
        if not math.isfinite(beam_range):
            continue
        hit_obstacle = math.isfinite(beam_range) and beam_range < normalized_range_max * obstacle_hit_ratio
        ray_max_m = min(beam_range, normalized_range_max)
        ray_max_m = max(ray_max_m, normalized_range_min)
        if ray_max_m <= normalized_range_min:
            continue
        angle = float(pose.yaw) + float(angle_min) + index * float(angle_increment)
        last_free = None
        samples = max(1, int(ray_max_m / step_m))
        for sample_index in range(1, samples + 1):
            distance_m = min(sample_index * step_m, ray_max_m)
            cell = cell_from_world(
                float(pose.x) + math.cos(angle) * distance_m,
                float(pose.y) + math.sin(angle) * distance_m,
            )
            if visited_cells is not None:
                visited_cells.add(cell)
            if hit_obstacle and sample_index == samples and cell != origin_cell:
                if known_cells is not None:
                    merge_occupancy_observation(
                        known_cells,
                        cell,
                        "occupied",
                        evidence_scores=evidence_scores,
                        config=config,
                    )
                point_count += 1
            else:
                if known_cells is not None:
                    merge_occupancy_observation(
                        known_cells,
                        cell,
                        "free",
                        evidence_scores=evidence_scores,
                        config=config,
                    )
                last_free = cell
                point_count += 1
        if not hit_obstacle and last_free is not None and range_edge_cells is not None:
            range_edge_cells.add(last_free)

    return ScanIntegrationSummary(
        point_count=point_count,
        scan_beams=int(len(ranges)),
        integrated_beams=int(math.ceil(len(ranges) / normalized_beam_stride)),
    )
