from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import math
from pathlib import Path
import threading
import time
import webbrowser
from typing import Any

import numpy as np

from multido_xlerobot.bootstrap import resolve_xlerobot_repo_root
from xlerobot_agent.exploration import ExplorationBackend, ExplorationBackendConfig, Pose2D
from xlerobot_agent.prompts import build_exploration_policy_user_prompt
from xlerobot_playground.maniskill_ros_bridge import (
    HEAD_CAMERA_FOV_RAD,
    HEAD_CAMERA_UID,
    normalize_quaternion_wxyz,
    quaternion_to_yaw,
    synthesize_scan_from_depth,
)
from xlerobot_playground.frontier_runtime import refresh_frontier_records
from xlerobot_playground.map_editing import (
    ACTIVE_RGBD_SCAN_FUSION_CONFIG,
    ManualOccupancyEdits,
    merge_occupancy_observation,
    merge_occupancy_observations,
    overlay_known_cells,
    overlay_occupancy_payload,
)
from xlerobot_playground.nav2_defaults import default_nav2_behavior_tree
from xlerobot_playground.scan_fusion import integrate_planar_scan
from xlerobot_playground.interactive_react_ui import INTERACTIVE_REACT_HTML
from xlerobot_playground.ros_nav2_router import RemoteNav2RouterClient
from xlerobot_playground.sim_exploration_backend import (
    ExplorationDecision,
    ExplorationLLMPolicy,
    FrontierCandidate,
    FrontierMemory,
    FrontierRecord,
    GridCell,
    RosExplorationSession,
    RosOccupancyMap,
    SemanticWaypointObserver,
    SimExplorationConfig,
    _aggregate_semantic_updates,
    _build_simple_apartment,
    _centroid_cell,
    _dedupe_text,
    _frontier_opening_width_m,
    _frontier_selection_guidance,
    _grid_distance_cells,
    _neighbors4,
    _neighbors8,
    _pose_distance_m,
    _pose_from_mapping,
    _search_known_safe_path,
    _semantic_named_places_for_map,
    _simulate_scan,
)


XLEROBOT_IKEA_CART_FOOTPRINT_LENGTH_M = 0.35
XLEROBOT_IKEA_CART_FOOTPRINT_WIDTH_M = 0.45
XLEROBOT_IKEA_CART_FOOTPRINT_RADIUS_M = math.hypot(
    XLEROBOT_IKEA_CART_FOOTPRINT_LENGTH_M / 2.0,
    XLEROBOT_IKEA_CART_FOOTPRINT_WIDTH_M / 2.0,
)
XLEROBOT_IKEA_CART_CLEARANCE_PADDING_M = 0.06
BASE_TELEPORT_POSITION_TOLERANCE_M = 0.08
BASE_TELEPORT_YAW_TOLERANCE_RAD = 0.15
XLEROBOT_CAMERA_TO_BASE_OFFSET_M = 0.09
STRICT_NAVIGATION_CLEARANCE_MARGIN_M = 0.08
STRICT_NAVIGATION_KNOWN_FRACTION = 0.95
STORED_FRONTIER_REVALIDATION_RADIUS_M = 1.0
STORED_FRONTIER_REVISIT_APPROACH_RADIUS_M = 1.25
SIM_MOTION_SPEED_PRESETS = {
    "normal": {"linear_m_s": 0.30, "angular_multiplier": 1.0},
    "faster": {"linear_m_s": 0.60, "angular_multiplier": 1.8},
    "fastest": {"linear_m_s": 1.00, "angular_multiplier": 3.0},
}
LOCAL_PATH_GUARD_LOOKAHEAD_M = 0.25
LOCAL_PATH_GUARD_PADDING_M = 0.0
KEYBOARD_SPEED_PROFILES = {
    "slow": {"base_forward": 0.05, "base_turn": 0.02},
    "normal": {"base_forward": 0.1, "base_turn": 0.05},
    "fast": {"base_forward": 0.3, "base_turn": 0.1},
}


class Nav2LocalPathBlocked(RuntimeError):
    def __init__(self, blocker: dict[str, Any], *, travelled_distance_m: float) -> None:
        self.blocker = blocker
        self.travelled_distance_m = travelled_distance_m
        super().__init__(str(blocker.get("reason", "local RGB-D scan blocked the current path")))


class HumanAssistanceRequired(RuntimeError):
    """Raised when autonomous navigation must stop for operator takeover."""


def _local_scan_path_blocker(
    observation: dict[str, Any] | None,
    *,
    current_pose: Pose2D,
    target_pose: Pose2D,
    robot_length_m: float,
    robot_width_m: float,
    lookahead_m: float = LOCAL_PATH_GUARD_LOOKAHEAD_M,
    safety_padding_m: float = LOCAL_PATH_GUARD_PADDING_M,
) -> dict[str, Any] | None:
    if not isinstance(observation, dict):
        return None
    sensor_pose = observation.get("pose")
    if not isinstance(sensor_pose, Pose2D):
        return None
    dx = target_pose.x - current_pose.x
    dy = target_pose.y - current_pose.y
    segment_distance_m = math.hypot(dx, dy)
    if segment_distance_m <= 1e-4:
        return None
    ranges = observation.get("ranges", ())
    if not ranges:
        return None
    unit_x = dx / segment_distance_m
    unit_y = dy / segment_distance_m
    angle_min = float(observation.get("angle_min", 0.0) or 0.0)
    angle_increment = float(observation.get("angle_increment", 0.0) or 0.0)
    range_min = float(observation.get("range_min", 0.05) or 0.05)
    range_max = float(observation.get("range_max", 0.0) or 0.0)
    half_length_m = max(robot_length_m * 0.5 + safety_padding_m, 0.05)
    half_width_m = max(robot_width_m * 0.5 + safety_padding_m, 0.05)
    forward_limit_m = lookahead_m + half_length_m  # robot half + lookahead
    sweep_forward_min_m = 0.0
    sweep_forward_max_m = forward_limit_m
    closest: dict[str, Any] | None = None
    for index, raw_range in enumerate(ranges):
        try:
            range_m = float(raw_range)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(range_m) or range_m < range_min:
            continue
        if range_max > 0.0 and range_m >= range_max * 0.995:
            continue
        beam_angle = sensor_pose.yaw + angle_min + angle_increment * index
        point_x = sensor_pose.x + math.cos(beam_angle) * range_m
        point_y = sensor_pose.y + math.sin(beam_angle) * range_m
        rel_x = point_x - current_pose.x
        rel_y = point_y - current_pose.y
        forward_m = rel_x * unit_x + rel_y * unit_y
        if forward_m < sweep_forward_min_m or forward_m > sweep_forward_max_m:
            continue
        lateral_m = abs(rel_x * unit_y - rel_y * unit_x)
        if lateral_m > half_width_m:
            continue
        clearance_forward_m = max(forward_m - half_length_m, 0.0)
        if closest is None or clearance_forward_m < closest["forward_clearance_m"]:
            closest = {
                "reason": "local RGB-D scan observed an obstacle in the swept robot footprint",
                "beam_index": index,
                "range_m": round(range_m, 3),
                "forward_distance_m": round(forward_m, 3),
                "forward_clearance_m": round(clearance_forward_m, 3),
                "lateral_distance_m": round(lateral_m, 3),
                "point": {"x": round(point_x, 3), "y": round(point_y, 3)},
                "half_length_m": round(half_length_m, 3),
                    "half_width_m": round(half_width_m, 3),
                }
    
    return closest


def _manual_region_from_cells(
    *,
    region_id: str,
    label: str,
    description: str,
    cells: list[dict[str, int]],
    resolution: float,
) -> dict[str, Any]:
    unique = sorted({(int(item["cell_x"]), int(item["cell_y"])) for item in cells})
    if not unique:
        raise ValueError("region requires at least one selected free cell")
    xs = [x for x, _ in unique]
    ys = [y for _, y in unique]
    centroid_x = sum((x + 0.5) * resolution for x, _ in unique) / len(unique)
    centroid_y = sum((y + 0.5) * resolution for _, y in unique) / len(unique)
    waypoint_cell = min(
        unique,
        key=lambda item: ((item[0] + 0.5) * resolution - centroid_x) ** 2
        + ((item[1] + 0.5) * resolution - centroid_y) ** 2,
    )
    waypoint_x = (waypoint_cell[0] + 0.5) * resolution
    waypoint_y = (waypoint_cell[1] + 0.5) * resolution
    waypoint = {
        "name": f"{label.strip().replace(' ', '_') or region_id}_center",
        "x": round(waypoint_x, 3),
        "y": round(waypoint_y, 3),
        "yaw": 0.0,
        "kind": "primary",
    }
    polygon = [
        [min(xs) * resolution, min(ys) * resolution],
        [(max(xs) + 1) * resolution, min(ys) * resolution],
        [(max(xs) + 1) * resolution, (max(ys) + 1) * resolution],
        [min(xs) * resolution, (max(ys) + 1) * resolution],
    ]
    return {
        "region_id": region_id,
        "label": label.strip() or region_id,
        "description": description.strip(),
        "confidence": 1.0,
        "polygon_2d": [[round(float(x), 3), round(float(y), 3)] for x, y in polygon],
        "centroid": {"x": round(centroid_x, 3), "y": round(centroid_y, 3)},
        "selected_cells": [{"cell_x": x, "cell_y": y} for x, y in unique],
        "adjacency": [],
        "representative_keyframes": [],
        "evidence": ["operator selected free-space region in UI"],
        "default_waypoints": [waypoint],
        "source": "manual_region",
    }


def _named_places_from_manual_regions(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    places: list[dict[str, Any]] = []
    for region in regions:
        for waypoint in region.get("default_waypoints", []):
            if not isinstance(waypoint, dict) or not waypoint.get("name"):
                continue
            places.append(
                {
                    "name": str(waypoint["name"]),
                    "pose": {
                        "x": float(waypoint.get("x", 0.0)),
                        "y": float(waypoint.get("y", 0.0)),
                        "yaw": float(waypoint.get("yaw", 0.0)),
                    },
                    "region_id": region.get("region_id"),
                    "source": "manual_region",
                    "kind": waypoint.get("kind", "primary"),
                }
            )
    return places


class InteractiveRosNav2ExplorationSession(RosExplorationSession):
    """Step-gated React playground session that executes motion through live ROS/Nav2."""

    def __init__(self, config: SimExplorationConfig, backend: ExplorationBackend, task_id: str) -> None:
        super().__init__(config, backend, task_id)
        self.manual_regions: list[dict[str, Any]] = []

    def reset(self) -> dict[str, Any]:
        self.manual_regions = []
        return super().reset()

    def snapshot(self) -> dict[str, Any]:
        if self.runtime.latest_map is None:
            pose = self.runtime.current_pose()
            if pose is None and self.config.ros_navigation_map_source == "fused_scan":
                pose = self.runtime.current_pose_in_frame(self.config.ros_odom_frame)
            robot_pose = (pose or Pose2D(0.0, 0.0, 0.0)).to_dict()
            return {
                "status": self.status,
                "session": self.config.session,
                "coverage": 0.0,
                "robot_pose": robot_pose,
                "prompt": self.pending_prompt_text,
                "prompt_payload": self.pending_prompt_payload,
                "candidate_frontiers": [],
                "pending_decision": None,
                "pending_trace": self.pending_trace,
                "pending_target": None,
                "applied_memory_updates": list(self.applied_memory_updates),
                "capabilities": {"web_manual_control": False},
                "last_error": self.last_error,
                "map": {
                    "map_id": self.config.session,
                    "frame": self.config.ros_map_frame,
                    "resolution": float(self.config.occupancy_resolution),
                    "coverage": 0.0,
                    "summary": "ROS/Nav2 exploration has not started yet. Click Start Explore to wait for ROS data and run the initial scan.",
                    "approved": False,
                    "created_at": time.time(),
                    "source": self.config.source,
                    "mode": "interactive_ros_nav2",
                    "trajectory": self.trajectory,
                    "keyframes": self.keyframes,
                    "regions": list(self.manual_regions),
                    "named_places": _named_places_from_manual_regions(self.manual_regions),
                    "occupancy": {
                        "resolution": float(self.config.occupancy_resolution),
                        "bounds": {
                            "min_x": 0.0,
                            "min_y": 0.0,
                            "max_x": 0.0,
                            "max_y": 0.0,
                        },
                        "cells": [],
                    },
                    "frontiers": [],
                    "remembered_frontiers": [],
                    "semantic_area_candidates": [],
                    "semantic_memory": {},
                    "automatic_semantic_waypoints": self.config.automatic_semantic_waypoints,
                    "artifacts": {
                        "decision_log": self.decision_log,
                        "frontier_memory": self.frontier_memory.snapshot(),
                        "guardrail_events": self.guardrail_events,
                        "navigation": {
                            "mode": "ros_nav2",
                            "control_steps": self.control_steps,
                            "total_distance_m": round(self.total_distance_m, 3),
                        },
                        "ros_runtime": {
                            "map_topic": self.config.ros_map_topic,
                            "scan_topic": self.config.ros_scan_topic,
                            "rgb_topic": self.config.ros_rgb_topic,
                            "base_frame": self.config.ros_base_frame,
                            "odom_frame": self.config.ros_odom_frame,
                            "map_frame": self.config.ros_map_frame,
                        },
                    },
                },
            }
        return super().snapshot()

    def create_manual_region(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            occupancy_map = self._require_effective_map()
            cells = []
            for item in payload.get("cells", []):
                if not isinstance(item, dict):
                    continue
                try:
                    cell = GridCell(int(item.get("cell_x", -1)), int(item.get("cell_y", -1)))
                except Exception:
                    continue
                if occupancy_map.in_bounds(cell.x, cell.y) and occupancy_map.is_free(cell.x, cell.y):
                    cells.append({"cell_x": cell.x, "cell_y": cell.y})
            if not cells:
                self.last_error = "Region was not created because no selected cells are known free space."
                return self.snapshot()
            region = _manual_region_from_cells(
                region_id=f"manual_region_{len(self.manual_regions) + 1:03d}",
                label=str(payload.get("label", "")).strip() or f"region_{len(self.manual_regions) + 1}",
                description=str(payload.get("description", "")).strip(),
                cells=cells,
                resolution=occupancy_map.resolution,
            )
            self.manual_regions.append(region)
            self._push_progress_update(message=f"Added manual region `{region['label']}`.", frontier_id=None)
            return self.snapshot()

    def update_manual_region_waypoint(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            region_id = str(payload.get("region_id", ""))
            waypoint_name = str(payload.get("waypoint_name", ""))
            pose = payload.get("pose", {})
            region = next((item for item in self.manual_regions if item.get("region_id") == region_id), None)
            if region is None or not isinstance(pose, dict):
                self.last_error = "Could not update waypoint; region or pose was invalid."
                return self.snapshot()
            x = float(pose.get("x", 0.0))
            y = float(pose.get("y", 0.0))
            occupancy_map = self._require_effective_map()
            cell_x, cell_y = occupancy_map.world_to_cell(x, y)
            if not occupancy_map.in_bounds(cell_x, cell_y) or not occupancy_map.is_free(cell_x, cell_y):
                self.last_error = "Waypoint was not moved because the target is not known free space."
                return self.snapshot()
            waypoint = next(
                (item for item in region.get("default_waypoints", []) if item.get("name") == waypoint_name),
                None,
            )
            if waypoint is None:
                waypoint = {"name": waypoint_name or f"{region['label']}_waypoint", "kind": "subwaypoint"}
                region.setdefault("default_waypoints", []).append(waypoint)
            waypoint.update(
                {
                    "x": round(x, 3),
                    "y": round(y, 3),
                    "yaw": round(float(pose.get("yaw", 0.0)), 3),
                }
            )
            self._push_progress_update(message=f"Updated waypoint `{waypoint['name']}`.", frontier_id=None)
            return self.snapshot()

    def add_manual_region_subwaypoint(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload = dict(payload)
        payload["waypoint_name"] = str(payload.get("name", "")).strip() or "subwaypoint"
        return self.update_manual_region_waypoint(payload)

    def add_manual_frontier(self, payload: dict[str, Any]) -> dict[str, Any]:
        pose = payload.get("pose")
        if not isinstance(pose, dict):
            return self.snapshot()
        frontier = FrontierCandidate(
            nav_pose=Pose2D(float(pose.get("x", 0)), float(pose.get("y", 0)), float(pose.get("yaw", 0))),
            centroid_pose=Pose2D(float(pose.get("x", 0)), float(pose.get("y", 0)), float(pose.get("yaw", 0))),
            unknown_gain=1,
            sensor_range_edge=False,
            room_hint="manual",
            evidence=["manually added by user"],
        )
        records = self.frontier_memory.upsert_candidates([frontier], step_index=self.decision_index)
        if records:
            record = records[0]
            decision = ExplorationDecision(
                decision_type="explore_frontier",
                selected_frontier_id=record.frontier_id,
            )
            self.pending_decision = decision
        return self.snapshot()

    def _build_map_payload(self) -> dict[str, Any]:
        payload = super()._build_map_payload()
        regions = list(getattr(self, "manual_regions", []))
        semantic_places = payload.get("named_places", [])
        payload["mode"] = "interactive_ros_nav2"
        payload["regions"] = regions
        payload["named_places"] = _named_places_from_manual_regions(regions) + semantic_places
        pending_ids = {record.frontier_id for record in getattr(self, "pending_candidate_records", [])}
        payload["frontiers"] = [record.to_dict() for record in getattr(self, "pending_candidate_records", [])]
        payload["remembered_frontiers"] = [
            record.to_dict()
            for record in self.frontier_memory.records.values()
            if record.frontier_id not in pending_ids and record.status == "stored"
        ]
        payload.setdefault("artifacts", {}).setdefault("navigation", {})["mode"] = "ros_nav2"
        return payload


@dataclass(frozen=True)
class ManiSkillInteractiveOptions:
    repo_root: str
    env_id: str
    robot_uid: str
    display_yaw_offset_deg: float | None
    control_mode: str
    render_mode: str | None
    shader: str
    sim_backend: str
    num_envs: int
    force_reload: bool
    build_config_idx: int | None
    spawn_x: float | None
    spawn_y: float | None
    spawn_yaw: float
    spawn_facing: str
    scan_mode: str
    scan_yaw_samples: int
    depth_beam_stride: int
    teleport_z: float | None
    teleport_settle_steps: int
    max_frontiers: int
    use_keyboard_controls: bool
    keyboard_speed: str


class InteractiveNoNav2ExplorationSession:
    """Step-gated exploration sandbox for testing LLM frontier decisions without Nav2."""

    def __init__(self, config: SimExplorationConfig) -> None:
        self.config = config
        self.scenario = _build_simple_apartment(config.occupancy_resolution)
        self.policy = ExplorationLLMPolicy(config)
        self.frontier_memory = FrontierMemory(config.occupancy_resolution)
        self.manual_occupancy_edits = ManualOccupancyEdits()
        self._lock = threading.RLock()
        self.reset()

    def reset(self) -> dict[str, Any]:
        with self._lock:
            self.frontier_memory = FrontierMemory(self.config.occupancy_resolution)
            self.semantic_observer = SemanticWaypointObserver(self.config, scenario=self.scenario)
            self.manual_occupancy_edits = ManualOccupancyEdits()
            self.current_cell = self.scenario.start_cell
            self.current_yaw = 0.0
            self.known_cells: dict[GridCell, str] = {}
            self.occupancy_evidence: dict[GridCell, float] = {}
            self.range_edge_cells: set[GridCell] = set()
            self.trajectory: list[dict[str, Any]] = [
                self.current_cell.center_pose(self.scenario.resolution).to_dict()
            ]
            self.keyframes: list[dict[str, Any]] = []
            self.semantic_processed_frame_count = 0
            self.manual_regions: list[dict[str, Any]] = []
            self.room_frames: dict[str, list[str]] = {}
            self.room_objects_seen: dict[str, set[str]] = {room_id: set() for room_id in self.scenario.rooms}
            self.room_descriptions: dict[str, list[str]] = {room_id: [] for room_id in self.scenario.rooms}
            self.decision_log: list[dict[str, Any]] = []
            self.semantic_updates: list[dict[str, Any]] = []
            self.guardrail_events: list[dict[str, Any]] = []
            self.total_distance_m = 0.0
            self.control_steps = 0
            self.decision_index = 0
            self.status = "initial_scan_complete"
            self.pending_prompt_payload: dict[str, Any] | None = None
            self.pending_prompt_text: str | None = None
            self.pending_candidate_records: list[FrontierRecord] = []
            self.pending_decision: ExplorationDecision | None = None
            self.pending_trace: dict[str, Any] | None = None
            self.applied_memory_updates: list[dict[str, Any]] = []
            self.last_error: str | None = None
            self.paused = False
            self.frontier_memory.remember_return_waypoint(
                room_id=self.scenario.room_for_cell(self.current_cell),
                pose=self.current_cell.center_pose(self.scenario.resolution),
                step_index=0,
                reason="initial_pose",
            )
            self._perform_scan(full_turnaround=True, capture_frame=True, reason="mock_initial_turnaround_scan")
            self._prepare_decision_locked()
            return self.snapshot()

    def create_manual_region(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            effective = self._effective_known_cells()
            cells = [
                item
                for item in payload.get("cells", [])
                if isinstance(item, dict)
                and effective.get(GridCell(int(item.get("cell_x", -1)), int(item.get("cell_y", -1)))) == "free"
            ]
            if not cells:
                self.last_error = "Region was not created because no selected cells are known free space."
                return self.snapshot()
            region = _manual_region_from_cells(
                region_id=f"manual_region_{len(self.manual_regions) + 1:03d}",
                label=str(payload.get("label", "")).strip() or f"region_{len(self.manual_regions) + 1}",
                description=str(payload.get("description", "")).strip(),
                cells=cells,
                resolution=self.config.occupancy_resolution,
            )
            self.manual_regions.append(region)
            return self.snapshot()

    def update_manual_region_waypoint(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            region_id = str(payload.get("region_id", ""))
            waypoint_name = str(payload.get("waypoint_name", ""))
            pose = payload.get("pose", {})
            region = next((item for item in self.manual_regions if item.get("region_id") == region_id), None)
            if region is None or not isinstance(pose, dict):
                self.last_error = "Could not update waypoint; region or pose was invalid."
                return self.snapshot()
            x = float(pose.get("x", 0.0))
            y = float(pose.get("y", 0.0))
            cell = self.scenario.world_to_cell(x, y)
            if self._effective_known_cells().get(cell) != "free":
                self.last_error = "Waypoint was not moved because the target is not known free space."
                return self.snapshot()
            waypoint = next(
                (item for item in region.get("default_waypoints", []) if item.get("name") == waypoint_name),
                None,
            )
            if waypoint is None:
                waypoint = {"name": waypoint_name or f"{region['label']}_waypoint", "kind": "subwaypoint"}
                region.setdefault("default_waypoints", []).append(waypoint)
            waypoint.update(
                {
                    "x": round(x, 3),
                    "y": round(y, 3),
                    "yaw": round(float(pose.get("yaw", 0.0)), 3),
                }
            )
            return self.snapshot()

    def add_manual_region_subwaypoint(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload = dict(payload)
        payload["waypoint_name"] = str(payload.get("name", "")).strip() or "subwaypoint"
        return self.update_manual_region_waypoint(payload)

    def add_manual_frontier(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            pose = payload.get("pose")
            if not isinstance(pose, dict):
                return self.snapshot()
            frontier = FrontierCandidate(
                nav_pose=Pose2D(float(pose.get("x", 0)), float(pose.get("y", 0)), float(pose.get("yaw", 0))),
                centroid_pose=Pose2D(float(pose.get("x", 0)), float(pose.get("y", 0)), float(pose.get("yaw", 0))),
                unknown_gain=1,
                sensor_range_edge=False,
                room_hint="manual",
                evidence=["manually added by user"],
            )
            records = self.frontier_memory.upsert_candidates([frontier], step_index=self.decision_index)
            if records:
                record = records[0]
                decision = ExplorationDecision(
                    decision_type="explore_frontier",
                    selected_frontier_id=record.frontier_id,
                )
                self.pending_decision = decision
            return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": self.status,
                "session": self.config.session,
                "coverage": round(self._coverage(), 3),
                "robot_pose": self.current_cell.center_pose(self.scenario.resolution, yaw=self.current_yaw).to_dict(),
                "prompt": self.pending_prompt_text,
                "prompt_payload": self.pending_prompt_payload,
                "candidate_frontiers": [record.to_dict() for record in self.pending_candidate_records],
                "pending_decision": None if self.pending_decision is None else self.pending_decision.to_dict(),
                "pending_trace": self.pending_trace,
                "pending_target": self._pending_target(),
                "applied_memory_updates": list(self.applied_memory_updates),
                "capabilities": {"web_manual_control": False},
                "last_error": self.last_error,
                "map": self._build_map_payload(),
            }

    def call_llm(self) -> dict[str, Any]:
        with self._lock:
            self.last_error = None
            if self.paused:
                self.last_error = "Exploration is paused. Resume before calling the LLM."
                return self.snapshot()
            if self.pending_prompt_payload is None:
                self._prepare_decision_locked()
            assert self.pending_prompt_payload is not None
            coverage = self._coverage()
            decision, trace = self.policy.decide(
                prompt_payload=self.pending_prompt_payload,
                frontiers=self.pending_candidate_records,
                return_waypoints=list(self.frontier_memory.return_waypoints.values()),
                coverage=coverage,
                current_room_id=self.scenario.room_for_cell(self.current_cell),
            )
            decision = self._apply_finish_guardrail(decision, self.pending_candidate_records)
            applied_memory_updates = self.frontier_memory.apply_model_memory_updates(
                decision.memory_updates,
                selected_frontier_id=decision.selected_frontier_id,
            )
            trace["applied_memory_updates"] = applied_memory_updates
            self.pending_decision = decision
            self.pending_trace = trace
            self.applied_memory_updates = applied_memory_updates
            self.status = "llm_response_ready"
            return self.snapshot()

    def call_semantic_llm(self) -> dict[str, Any]:
        with self._lock:
            self.last_error = None
            if not self.config.automatic_semantic_waypoints:
                self.last_error = (
                    "Automatic semantic waypoints are disabled. Start with --automatic-semantic-waypoints "
                    "to enable the legacy semantic waypoint pipeline."
                )
                return self.snapshot()
            frames = self.keyframes[self.semantic_processed_frame_count :]
            if not frames:
                self.last_error = "No new spin keyframes are waiting for semantic waypoint processing."
                return self.snapshot()
            trace = self.semantic_observer.observe_keyframe_batch(
                frames=json.loads(json.dumps(frames)),
                known_cells=self._effective_known_cells(),
                robot_cell=self.current_cell,
                resolution=self.scenario.resolution,
            )
            self.semantic_processed_frame_count = len(self.keyframes)
            self.status = "semantic_response_ready"
            self.pending_trace = {"semantic_trace": trace}
            return self.snapshot()

    def apply_decision(self) -> dict[str, Any]:
        with self._lock:
            self.last_error = None
            if self.paused:
                self.last_error = "Exploration is paused. Resume before applying a decision."
                return self.snapshot()
            if self.pending_decision is None:
                self.last_error = "No pending LLM decision is ready yet."
                return self.snapshot()
            decision = self.pending_decision
            if decision.decision_type == "finish" or decision.exploration_complete:
                self.status = "finished"
                self._log_decision("finish_without_motion")
                self.pending_decision = None
                return self.snapshot()
            if not decision.selected_frontier_id:
                self.last_error = "The LLM decision did not select a frontier."
                return self.snapshot()
            record = self.frontier_memory.activate(decision.selected_frontier_id)
            if record is None:
                self.last_error = f"Selected frontier `{decision.selected_frontier_id}` no longer exists."
                return self.snapshot()

            start_pose = self.current_cell.center_pose(self.scenario.resolution, yaw=self.current_yaw)
            target_cell = self.scenario.world_to_cell(record.nav_pose.x, record.nav_pose.y)
            path_cells = self._known_free_path_to_cell(target_cell)
            if not path_cells:
                self.frontier_memory.fail(record.frontier_id, "mock mover target is not known free space")
                self.last_error = (
                    f"Mock mover rejected `{record.frontier_id}` because there is no known-free path "
                    "from the robot to the target."
                )
                self._log_decision("mock_move_rejected")
                self._prepare_decision_locked()
                return self.snapshot()

            self.current_yaw = math.atan2(target_cell.y - self.current_cell.y, target_cell.x - self.current_cell.x)
            self.current_cell = target_cell
            reached_pose = self.current_cell.center_pose(self.scenario.resolution, yaw=self.current_yaw)
            distance_m = max(len(path_cells) - 1, 0) * self.scenario.resolution
            self.total_distance_m += distance_m
            self.control_steps += 1
            for cell in path_cells[1:-1]:
                self.trajectory.append(cell.center_pose(self.scenario.resolution, yaw=self.current_yaw).to_dict())
            self.trajectory.append(reached_pose.to_dict())
            self._perform_scan(
                full_turnaround=True,
                capture_frame=True,
                reason=f"mock_arrive_frontier::{record.frontier_id}",
            )
            self.frontier_memory.complete(record.frontier_id)
            self.frontier_memory.remember_return_waypoint(
                room_id=self.scenario.room_for_cell(self.current_cell),
                pose=reached_pose,
                step_index=self.decision_index,
                reason=f"completed_frontier::{record.frontier_id}",
            )
            self.semantic_updates.extend(decision.semantic_updates)
            self._log_decision("mock_move_applied")
            self.pending_decision = None
            self.pending_trace = None
            self.applied_memory_updates = []
            if self._coverage() >= self.config.finish_coverage_threshold:
                self.status = "finished"
            else:
                self._prepare_decision_locked()
            return self.snapshot()

    def _prepare_decision_locked(self) -> None:
        self.decision_index += 1
        visible_candidates = self._detect_frontier_candidates()
        self.frontier_memory.upsert_candidates(visible_candidates, step_index=self.decision_index)
        candidate_records = self._refresh_candidate_paths()
        prompt_payload = self._build_prompt_payload(candidate_records)
        self.pending_candidate_records = candidate_records
        self.pending_prompt_payload = prompt_payload
        self.pending_prompt_text = build_exploration_policy_user_prompt(prompt_payload)
        self.pending_decision = None
        self.pending_trace = None
        self.applied_memory_updates = []
        self.status = "waiting_for_llm"
        if not candidate_records:
            self.status = "finished"

    def _perform_scan(self, *, full_turnaround: bool, capture_frame: bool, reason: str) -> None:
        scan = _simulate_scan(
            self.scenario,
            self.current_cell,
            yaw=self.current_yaw,
            max_range_m=self.config.sensor_range_m,
            full_turnaround=full_turnaround,
        )
        for cell in scan.observed_free:
            merge_occupancy_observation(
                self.known_cells,
                cell,
                "free",
                evidence_scores=self.occupancy_evidence,
            )
        for cell in scan.observed_occupied:
            merge_occupancy_observation(
                self.known_cells,
                cell,
                "occupied",
                evidence_scores=self.occupancy_evidence,
            )
        self.range_edge_cells |= scan.range_edge_frontiers
        current_room_id = self.scenario.room_for_cell(self.current_cell)
        for label in scan.visible_objects:
            if current_room_id:
                self.room_objects_seen.setdefault(current_room_id, set()).add(label)
        if current_room_id and scan.description not in self.room_descriptions.setdefault(current_room_id, []):
            self.room_descriptions[current_room_id].append(scan.description)
        if capture_frame:
            frame_id = f"kf_{len(self.keyframes) + 1:03d}"
            frame = {
                "frame_id": frame_id,
                "pose": self.current_cell.center_pose(self.scenario.resolution, yaw=self.current_yaw).to_dict(),
                "region_id": current_room_id or "unknown",
                "visible_objects": list(scan.visible_objects),
                "point_count": scan.point_count,
                "depth_min_m": scan.depth_min_m,
                "depth_max_m": scan.depth_max_m,
                "description": scan.description,
                "thumbnail_data_url": scan.thumbnail_data_url,
            }
            self.keyframes.append(frame)
            if current_room_id:
                self.room_frames.setdefault(current_room_id, []).append(frame_id)

    def _known_free_cells(self) -> set[GridCell]:
        return {cell for cell, state in self._effective_known_cells().items() if state == "free"}

    def _effective_known_cells(self) -> dict[GridCell, str]:
        edits = getattr(self, "manual_occupancy_edits", ManualOccupancyEdits())
        return overlay_known_cells(self.known_cells, edits)

    def pause(self) -> dict[str, Any]:
        with self._lock:
            self.paused = True
            self.status = "paused"
            return self.snapshot()

    def control_robot(self) -> dict[str, Any]:
        return self.pause()

    def manual_drive(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.last_error = "Manual drive is only available for robot-backed exploration sessions."
            return self.snapshot()

    def manual_stop(self) -> dict[str, Any]:
        with self._lock:
            self.last_error = "Manual stop is only available for robot-backed exploration sessions."
            return self.snapshot()

    def manual_scan(self) -> dict[str, Any]:
        with self._lock:
            self.last_error = "Manual scan is only available for robot-backed exploration sessions."
            return self.snapshot()

    def resume(self) -> dict[str, Any]:
        with self._lock:
            self.paused = False
            if self.pending_decision is not None:
                self.status = "llm_response_ready"
            elif self.pending_prompt_payload is not None:
                self.status = "waiting_for_llm"
            return self.snapshot()

    def update_occupancy_edits(self, *, mode: str, cells: list[dict[str, Any]]) -> dict[str, Any]:
        with self._lock:
            resolution = self.scenario.resolution
            parsed_cells: list[GridCell] = []
            for item in cells:
                if not isinstance(item, dict):
                    continue
                try:
                    if "cell_x" in item and "cell_y" in item:
                        parsed_cells.append(GridCell(int(item["cell_x"]), int(item["cell_y"])))
                    else:
                        parsed_cells.append(
                            GridCell(
                                int(math.floor(float(item["x"]) / resolution)),
                                int(math.floor(float(item["y"]) / resolution)),
                            )
                        )
                except Exception:
                    continue
            self.manual_occupancy_edits.apply(cells=parsed_cells, mode=mode)
            self._prepare_decision_locked()
            return self.snapshot()

    def _detect_frontier_candidates(self) -> list[FrontierCandidate]:
        frontier_cells: set[GridCell] = set()
        unknown_neighbors_by_frontier: dict[GridCell, set[GridCell]] = {}
        known_free = self._known_free_cells()
        effective_known = self._effective_known_cells()
        for cell in known_free:
            unknown_neighbors = {
                neighbor
                for neighbor in _neighbors4(cell)
                if self.scenario.in_bounds(neighbor) and neighbor not in effective_known
            }
            if unknown_neighbors:
                frontier_cells.add(cell)
                unknown_neighbors_by_frontier[cell] = unknown_neighbors

        clusters: list[list[GridCell]] = []
        visited: set[GridCell] = set()
        for cell in frontier_cells:
            if cell in visited:
                continue
            cluster: list[GridCell] = []
            queue = [cell]
            visited.add(cell)
            while queue:
                current = queue.pop(0)
                cluster.append(current)
                for neighbor in _neighbors8(current):
                    if neighbor in frontier_cells and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            clusters.append(cluster)

        candidates: list[FrontierCandidate] = []
        required_opening_m = self._min_frontier_opening_width_m()
        for cluster in clusters:
            cluster_unknown = set().union(*(unknown_neighbors_by_frontier.get(cell, set()) for cell in cluster))
            if not cluster_unknown:
                continue
            opening_width_m = _frontier_opening_width_m(cluster, self.scenario.resolution)
            if opening_width_m < required_opening_m:
                self.guardrail_events.append(
                    {
                        "type": "frontier_opening_too_narrow",
                        "cluster_size": len(cluster),
                        "opening_width_m": round(opening_width_m, 3),
                        "required_width_m": round(required_opening_m, 3),
                        "frontier_boundary_pose": _cell_mean_pose(cluster, self.scenario.resolution).to_dict(),
                    }
                )
                continue
            boundary_pose = _cell_mean_pose(cluster, self.scenario.resolution)
            visited_filter_radius_m = self._visited_frontier_filter_radius_m()
            if self._is_pose_near_previous_visit(boundary_pose, visited_filter_radius_m):
                self.guardrail_events.append(
                    {
                        "type": "frontier_near_visited_pose_filtered",
                        "cluster_size": len(cluster),
                        "unknown_gain": len(cluster_unknown),
                        "filter_radius_m": round(visited_filter_radius_m, 3),
                        "frontier_boundary_pose": boundary_pose.to_dict(),
                    }
                )
                continue
            max_unknown_neighbor_count = max(len(unknown_neighbors_by_frontier.get(cell, set())) for cell in cluster)
            nav_cell = min(cluster, key=lambda cell: _grid_distance_cells(cell, self.current_cell))
            centroid_cell = _centroid_cell(cluster)
            room_hint = self.scenario.room_for_cell(nav_cell)
            evidence = [
                f"{len(cluster_unknown)} unknown neighbor cells",
                f"cluster size {len(cluster)}",
                f"frontier opening width is {opening_width_m:.2f} m, above robot-sized threshold {required_opening_m:.2f} m",
                "frontier information from partial RGB-D scan, not complete apartment knowledge",
            ]
            if max_unknown_neighbor_count >= 2:
                evidence.append("frontier signal is stronger: multiple unknown-facing neighbors support possible expansion")
            else:
                evidence.append(
                    "frontier signal is weaker: a single unknown-facing edge can still be useful, but the LLM should veto it if RGB/map context shows no meaningful navigable expansion"
                )
            if any(cell in self.range_edge_cells for cell in cluster):
                evidence.append("visible frontier reaches sensor range limit")
            candidates.append(
                FrontierCandidate(
                    frontier_id=None,
                    member_cells=tuple(sorted(cluster)),
                    nav_cell=nav_cell,
                    centroid_cell=centroid_cell,
                    nav_pose=nav_cell.center_pose(self.scenario.resolution),
                    centroid_pose=centroid_cell.center_pose(self.scenario.resolution),
                    unknown_gain=len(cluster_unknown),
                    sensor_range_edge=any(cell in self.range_edge_cells for cell in cluster),
                    room_hint=room_hint,
                    evidence=evidence,
                )
            )
        return candidates

    def _min_frontier_opening_width_m(self) -> float:
        configured = self.config.frontier_min_opening_m
        if configured is not None and configured > 0.0:
            return max(float(configured), self.scenario.resolution)
        return max(self.config.robot_radius_m * 2.0 + 0.10, self.scenario.resolution * 2.0)

    def _refresh_candidate_paths(self) -> list[FrontierRecord]:
        current_pose = self.current_cell.center_pose(self.scenario.resolution, yaw=self.current_yaw)
        current_pose_filter_m = max(self.config.occupancy_resolution * 1.5, self.config.robot_radius_m)

        def _path_cost(record: FrontierRecord) -> float | None:
            target_cell = self.scenario.world_to_cell(record.nav_pose.x, record.nav_pose.y)
            path_cells = self._known_free_path_to_cell(target_cell)
            return max(len(path_cells) - 1, 0) * self.scenario.resolution if path_cells else None

        return refresh_frontier_records(
            candidate_records=self.frontier_memory.candidate_records(),
            active_frontier_id=self.frontier_memory.active_frontier_id,
            current_pose=current_pose,
            current_pose_filter_m=current_pose_filter_m,
            path_cost_for_record=_path_cost,
            guardrail_events=self.guardrail_events,
            is_frontier_at_current_pose=self._is_frontier_at_current_pose,
            is_frontier_near_visited_pose=self._is_frontier_near_visited_pose,
            visited_pose_filter_m=self._visited_frontier_filter_radius_m(),
        )

    def _is_frontier_at_current_pose(self, record: FrontierRecord, current_pose_filter_m: float) -> bool:
        current_pose = self.current_cell.center_pose(self.scenario.resolution, yaw=self.current_yaw)
        boundary_cell = self.scenario.world_to_cell(record.centroid_pose.x, record.centroid_pose.y)
        return boundary_cell == self.current_cell or _pose_distance_m(record.centroid_pose, current_pose) <= current_pose_filter_m

    def _visited_frontier_filter_radius_m(self) -> float:
        configured = self.config.visited_frontier_filter_radius_m
        if configured is not None and configured > 0.0:
            return max(float(configured), self.scenario.resolution)
        return max(self.config.robot_radius_m + 0.35, self.scenario.resolution * 2.0)

    def _previous_trajectory_poses(self) -> list[Pose2D]:
        trajectory = getattr(self, "trajectory", [])
        return [
            pose
            for pose in (_pose_from_mapping(item) for item in trajectory[:-1])
            if pose is not None
        ]

    def _is_pose_near_previous_visit(self, pose: Pose2D, radius_m: float) -> bool:
        return any(_pose_distance_m(pose, visited_pose) <= radius_m for visited_pose in self._previous_trajectory_poses())

    def _is_frontier_near_visited_pose(self, record: FrontierRecord, radius_m: float) -> bool:
        return self._is_pose_near_previous_visit(record.centroid_pose, radius_m)

    def _known_free_path_to_cell(self, target_cell: GridCell) -> list[GridCell]:
        if not self.scenario.in_bounds(target_cell):
            return []
        return _search_known_safe_path(
            self.current_cell,
            target_cell,
            self._known_free_cells(),
        )

    def _build_prompt_payload(self, candidate_records: list[FrontierRecord]) -> dict[str, Any]:
        frontier_memory_snapshot = self.frontier_memory.snapshot()
        explored_areas = []
        for room_id, room in self.scenario.rooms.items():
            explored_areas.append(
                {
                    "region_id": room_id,
                    "label": room.label,
                    "observed_fraction": round(self._room_coverage(room_id), 3),
                    "objects_seen": sorted(self.room_objects_seen.get(room_id, set())),
                    "representative_frames": self.room_frames.get(room_id, [])[:3],
                }
            )
        return {
            "mission": (
                "Interactive no-Nav2 test: evaluate whether the LLM can use partial RGB-D frontier information, "
                "recent RGB views, and the 2D occupancy map to choose useful robot-navigable exploration regions."
            ),
            "robot": {
                "pose": self.current_cell.center_pose(self.scenario.resolution, yaw=self.current_yaw).to_dict(),
                "room_id": self.scenario.room_for_cell(self.current_cell),
                "coverage": round(self._coverage(), 3),
                "trajectory_points": len(self.trajectory),
                "sensor_range_m": self.config.sensor_range_m,
                "navigation_mode": "mock_direct_move_no_nav2",
            },
            "frontier_memory": frontier_memory_snapshot,
            "frontier_information": [record.to_dict() for record in candidate_records],
            "candidate_frontiers": [record.to_dict() for record in candidate_records],
            "frontier_selection_guidance": _frontier_selection_guidance(),
            "explored_areas": explored_areas,
            "recent_views": self.keyframes[-3:],
            "guardrails": {
                "frontier_information_is_partial_rgbd_scan_evidence": True,
                "navigation_is_mocked_without_nav2": True,
                "frontier_ids_must_come_from_prompt": True,
                "memory_updates_must_use_existing_frontier_ids": True,
                "avoid_furniture_shadow_boundaries_without_clear_open_space": True,
            },
            "ascii_map": self._ascii_map(candidate_records),
        }

    def _ascii_map(self, candidate_records: list[FrontierRecord]) -> str:
        frontier_cells = {
            self.scenario.world_to_cell(record.nav_pose.x, record.nav_pose.y): record.status
            for record in candidate_records
        }
        effective_known = self._effective_known_cells()
        lines: list[str] = []
        for y in reversed(range(self.scenario.height_cells)):
            row: list[str] = []
            for x in range(self.scenario.width_cells):
                cell = GridCell(x, y)
                if cell == self.current_cell:
                    row.append("R")
                    continue
                if cell in frontier_cells:
                    row.append("V" if frontier_cells[cell] in {"completed", "failed", "suppressed"} else "F")
                    continue
                state = effective_known.get(cell)
                if state == "free":
                    row.append(".")
                elif state == "occupied":
                    row.append("#")
                else:
                    row.append("?")
            lines.append("".join(row))
        return "\n".join(lines)

    def _apply_finish_guardrail(
        self,
        decision: ExplorationDecision,
        candidate_records: list[FrontierRecord],
    ) -> ExplorationDecision:
        reachable = [record for record in candidate_records if record.path_cost_m is not None]
        if decision.decision_type == "finish" and reachable and self._coverage() < self.config.finish_coverage_threshold:
            fallback = self.policy._heuristic_decision(
                candidate_records,
                list(self.frontier_memory.return_waypoints.values()),
                self._coverage(),
                self.scenario.room_for_cell(self.current_cell),
            )
            self.guardrail_events.append(
                {
                    "type": "finish_override",
                    "requested": decision.to_dict(),
                    "fallback": fallback.to_dict(),
                }
            )
            return fallback
        return decision

    def _log_decision(self, event_type: str) -> None:
        if self.pending_decision is None:
            return
        self.decision_log.append(
            {
                "step_index": self.decision_index,
                "event_type": event_type,
                "coverage": round(self._coverage(), 3),
                "decision": self.pending_decision.to_dict(),
                "trace": self.pending_trace or {},
                "applied_memory_updates": list(self.applied_memory_updates),
            }
        )

    def _pending_target(self) -> dict[str, Any] | None:
        if self.pending_decision is None or not self.pending_decision.selected_frontier_id:
            return None
        record = self.frontier_memory.records.get(self.pending_decision.selected_frontier_id)
        if record is None:
            return None
        return {
            "frontier_id": record.frontier_id,
            "nav_pose": record.nav_pose.to_dict(),
            "centroid_pose": record.centroid_pose.to_dict(),
            "status": record.status,
            "path_cost_m": record.path_cost_m,
        }

    def _coverage(self) -> float:
        effective = self._effective_known_cells()
        discovered_free = sum(1 for cell in self.scenario.free_cells if effective.get(cell) == "free")
        return round(discovered_free / max(self.scenario.total_free_cells(), 1), 6)

    def _room_coverage(self, room_id: str) -> float:
        room_cells = [cell for cell, observed_room in self.scenario.room_by_cell.items() if observed_room == room_id]
        if not room_cells:
            return 0.0
        effective = self._effective_known_cells()
        known = sum(1 for cell in room_cells if effective.get(cell) == "free")
        return known / len(room_cells)

    def _build_map_payload(self) -> dict[str, Any]:
        edits = getattr(self, "manual_occupancy_edits", ManualOccupancyEdits())
        occupancy_cells = []
        for cell, state in sorted(self._effective_known_cells().items()):
            item = {
                "x": round(cell.x * self.scenario.resolution, 3),
                "y": round(cell.y * self.scenario.resolution, 3),
                "state": state,
            }
            if cell in edits.blocked_cells:
                item["manual_override"] = "blocked"
            elif cell in edits.cleared_cells:
                item["manual_override"] = "cleared"
            occupancy_cells.append(item)
        semantic_area_candidates = []
        if self.config.experimental_free_space_semantic_waypoints:
            semantic_area_candidates = _aggregate_semantic_updates(self.semantic_updates)
        semantic_memory = self.semantic_observer.snapshot() if self.config.automatic_semantic_waypoints else {}
        regions = list(getattr(self, "manual_regions", []))
        return {
            "map_id": self.config.session,
            "frame": "map",
            "resolution": float(self.scenario.resolution),
            "coverage": round(self._coverage(), 3),
            "mode": "interactive_no_nav2_llm_frontier",
            "trajectory": self.trajectory,
            "keyframes": self.keyframes,
            "regions": regions,
            "named_places": _named_places_from_manual_regions(regions)
            + (_semantic_named_places_for_map(semantic_memory) if self.config.automatic_semantic_waypoints else []),
            "frontiers": [record.to_dict() for record in self.frontier_memory.records.values()],
            "semantic_area_candidates": semantic_area_candidates,
            "semantic_memory": semantic_memory,
            "automatic_semantic_waypoints": self.config.automatic_semantic_waypoints,
            "occupancy": {
                "resolution": float(self.scenario.resolution),
                "bounds": self.scenario.bounds(),
                "cells": occupancy_cells,
            },
            "artifacts": {
                "decision_log": self.decision_log,
                "frontier_memory": self.frontier_memory.snapshot(),
                "guardrail_events": self.guardrail_events,
                "manual_occupancy_edits": edits.to_dict(
                    resolution=self.scenario.resolution,
                ),
                "navigation": {
                    "mode": "mock_direct_move_no_nav2",
                    "control_steps": self.control_steps,
                    "total_distance_m": round(self.total_distance_m, 3),
                },
                "llm_policy": {
                    "explorer_policy": self.config.explorer_policy,
                    "provider": self.config.llm_provider,
                    "model": self.config.llm_model,
                },
            },
        }


class ManiSkillTeleportExplorationSession:
    """Interactive no-Nav2 session backed by the real ManiSkill scene and XLeRobot RGB-D."""

    def __init__(self, config: SimExplorationConfig, options: ManiSkillInteractiveOptions) -> None:
        self.config = config
        self.options = options
        self.policy = ExplorationLLMPolicy(config)
        self.frontier_memory = FrontierMemory(config.occupancy_resolution)
        self.manual_occupancy_edits = ManualOccupancyEdits()
        self._lock = threading.RLock()
        display_offset_deg = 0.0 if options.display_yaw_offset_deg is None else float(options.display_yaw_offset_deg)
        self._display_yaw_offset_rad = math.radians(display_offset_deg)
        self._initialize_environment()
        self._keyboard_control_active = False
        if self.options.use_keyboard_controls:
            import pygame
            pygame.init()
            pygame.display.set_mode((200, 100))
            self._keyboard_control_active = True
        self.reset()
        self._pending_keyboard_scan = False

    def close(self) -> None:
        env = getattr(self, "env", None)
        if env is not None:
            env.close()

    def pump_viewer(self) -> None:
        if self.options.render_mode != "human":
            return
        if not self._lock.acquire(blocking=False):
            return
        try:
            if self._keyboard_control_active:
                self._poll_keyboard_controls()
                self.env.step(self.action)
                if self._pending_keyboard_scan:
                    self._pending_keyboard_scan = False
                    self._perform_scan(full_turnaround=True, capture_frame=True, reason="keyboard_manual_scan")
            self.env.render()
        finally:
            self._lock.release()

    def _poll_keyboard_controls(self) -> None:
        import pygame
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        speed = KEYBOARD_SPEED_PROFILES[self.options.keyboard_speed]

        forward = 0.0
        turn = 0.0
        head_tilt_delta = 0.0
        
        if keys[pygame.K_UP]:
            forward = speed["base_forward"]
        elif keys[pygame.K_DOWN]:
            forward = -speed["base_forward"]
        if keys[pygame.K_LEFT]:
            turn = speed["base_turn"]
        elif keys[pygame.K_RIGHT]:
            turn = -speed["base_turn"]
        
        HEAD_TILT_STEP = 0.1
        
        head_tilt_delta = 0.0
        if keys[pygame.K_f]:
            head_tilt_delta = -HEAD_TILT_STEP  # F = up (negative)
        elif keys[pygame.K_v]:
            head_tilt_delta = HEAD_TILT_STEP   # V = down (positive)

        if head_tilt_delta != 0.0:
            self.action[15] = head_tilt_delta

        if keys[pygame.K_s]:
            self._pending_keyboard_scan = True

        if forward != 0.0 or turn != 0.0:
            current_pose = self._current_pose()
            target_pose = Pose2D(
                current_pose.x + forward * math.cos(current_pose.yaw),
                current_pose.y + forward * math.sin(current_pose.yaw),
                current_pose.yaw + turn,
            )
            if self._check_keyboard_collision(current_pose, target_pose):
                return
            print(f"KEYBOARD: forward={forward}, turn={turn}")
            self.action[0] = forward
            self.action[1] = turn
        else:
            self.action[0] = 0.0
            self.action[1] = 0.0
        
        if head_tilt_delta == 0.0:
            self.action[15] = 0.0

    def _check_keyboard_collision(self, current_pose: Pose2D, target_pose: Pose2D) -> bool:
        head_data = self._capture_head_camera()
        observation, _ = self._build_scan_observation_from_head_data(head_data)
        
        dx = target_pose.x - current_pose.x
        dy = target_pose.y - current_pose.y
        segment_distance = math.hypot(dx, dy)
        
        if segment_distance < 0.01:
            return False
        
        sensor_pose = observation.get("pose") if observation else None
        if sensor_pose is None:
            return False
        
        blocker = _local_scan_path_blocker(
            observation,
            current_pose=sensor_pose,
            target_pose=target_pose,
            robot_length_m=XLEROBOT_IKEA_CART_FOOTPRINT_LENGTH_M,
            robot_width_m=XLEROBOT_IKEA_CART_FOOTPRINT_WIDTH_M,
            lookahead_m=LOCAL_PATH_GUARD_LOOKAHEAD_M,
            safety_padding_m=LOCAL_PATH_GUARD_PADDING_M,
        )
        if blocker is not None:
            print(f"COLLISION: {blocker.get('reason', 'unknown')}")
            return True
        return False

    def reset(self) -> dict[str, Any]:
        with self._lock:
            self._reset_environment()
            self.frontier_memory = FrontierMemory(self.config.occupancy_resolution)
            self.semantic_observer = SemanticWaypointObserver(self.config, scenario=None)
            self.manual_occupancy_edits = ManualOccupancyEdits()
            self.known_cells: dict[GridCell, str] = {}
            self.occupancy_evidence: dict[GridCell, float] = {}
            self.range_edge_cells: set[GridCell] = set()
            self.latest_scan_cells: set[GridCell] = set()
            self.latest_scan_known_cells: dict[GridCell, str] = {}
            self.latest_scan_range_edge_cells: set[GridCell] = set()
            self.trajectory: list[dict[str, Any]] = [self._current_pose().to_dict()]
            self.keyframes: list[dict[str, Any]] = []
            self.semantic_processed_frame_count = 0
            self.manual_regions: list[dict[str, Any]] = []
            self.decision_log: list[dict[str, Any]] = []
            self.semantic_updates: list[dict[str, Any]] = []
            self.guardrail_events: list[dict[str, Any]] = []
            self.total_distance_m = 0.0
            self.control_steps = 0
            self.decision_index = 0
            self.status = "initial_scan_complete"
            self.pending_prompt_payload: dict[str, Any] | None = None
            self.pending_prompt_text: str | None = None
            self.pending_candidate_records: list[FrontierRecord] = []
            self.pending_decision: ExplorationDecision | None = None
            self.pending_trace: dict[str, Any] | None = None
            self.applied_memory_updates: list[dict[str, Any]] = []
            self.last_error: str | None = None
            self.paused = False
            self.frontier_memory.remember_return_waypoint(
                room_id=None,
                pose=self._current_pose(),
                step_index=0,
                reason="initial_pose",
            )
            self._perform_scan(
                full_turnaround=self._scan_uses_turnaround(),
                capture_frame=True,
                reason=self._scan_reason("initial"),
            )
            self._prepare_decision_locked()
            return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            raw_pose = self._current_pose()
            return {
                "status": self.status,
                "session": self.config.session,
                "coverage": round(self._coverage(), 3),
                "robot_pose": self._display_pose(raw_pose).to_dict(),
                "robot_raw_pose": raw_pose.to_dict(),
                "prompt": self.pending_prompt_text,
                "prompt_payload": self.pending_prompt_payload,
                "candidate_frontiers": [record.to_dict() for record in self.pending_candidate_records],
                "pending_decision": None if self.pending_decision is None else self.pending_decision.to_dict(),
                "pending_trace": self.pending_trace,
                "pending_target": self._pending_target(),
                "applied_memory_updates": list(self.applied_memory_updates),
                "capabilities": {"web_manual_control": False},
                "last_error": self.last_error,
                "map": self._build_map_payload(),
            }

    def call_llm(self) -> dict[str, Any]:
        with self._lock:
            self.last_error = None
            if self.paused:
                self.last_error = "Exploration is paused. Resume before calling the LLM."
                return self.snapshot()
            if self.pending_prompt_payload is None:
                self._prepare_decision_locked()
            assert self.pending_prompt_payload is not None
            decision_index = self.decision_index
            prompt_payload = json.loads(json.dumps(self.pending_prompt_payload))
            candidate_records = list(self.pending_candidate_records)
            return_waypoints = list(self.frontier_memory.return_waypoints.values())
            coverage = self._coverage()

        decision, trace = self.policy.decide(
            prompt_payload=prompt_payload,
            frontiers=candidate_records,
            return_waypoints=return_waypoints,
            coverage=coverage,
            current_room_id=None,
        )

        with self._lock:
            if decision_index != self.decision_index:
                self.last_error = "Discarded stale LLM response because the session changed while the model was running."
                return self.snapshot()
            decision = self._apply_finish_guardrail(decision, self.pending_candidate_records)
            applied_memory_updates = self.frontier_memory.apply_model_memory_updates(
                decision.memory_updates,
                selected_frontier_id=decision.selected_frontier_id,
            )
            trace["applied_memory_updates"] = applied_memory_updates
            self.pending_decision = decision
            self.pending_trace = trace
            self.applied_memory_updates = applied_memory_updates
            self.status = "llm_response_ready"
            return self.snapshot()

    def call_semantic_llm(self) -> dict[str, Any]:
        with self._lock:
            self.last_error = None
            if not self.config.automatic_semantic_waypoints:
                self.last_error = (
                    "Automatic semantic waypoints are disabled. Start with --automatic-semantic-waypoints "
                    "to enable the legacy semantic waypoint pipeline."
                )
                return self.snapshot()
            frames = self.keyframes[self.semantic_processed_frame_count :]
            if not frames:
                self.last_error = "No new spin keyframes are waiting for semantic waypoint processing."
                return self.snapshot()
            pose = self._current_pose()
            semantic_known = self._effective_known_cells()
            trace = self.semantic_observer.observe_keyframe_batch(
                frames=json.loads(json.dumps(frames)),
                known_cells=semantic_known,
                robot_cell=self._world_to_cell(pose.x, pose.y),
                resolution=self.config.occupancy_resolution,
            )
            self.semantic_processed_frame_count = len(self.keyframes)
            self.status = "semantic_response_ready"
            self.pending_trace = {"semantic_trace": trace}
            return self.snapshot()

    def create_manual_region(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            effective = self._effective_known_cells()
            cells = [
                item
                for item in payload.get("cells", [])
                if isinstance(item, dict)
                and effective.get(GridCell(int(item.get("cell_x", -1)), int(item.get("cell_y", -1)))) == "free"
            ]
            if not cells:
                self.last_error = "Region was not created because no selected cells are known free space."
                return self.snapshot()
            region = _manual_region_from_cells(
                region_id=f"manual_region_{len(self.manual_regions) + 1:03d}",
                label=str(payload.get("label", "")).strip() or f"region_{len(self.manual_regions) + 1}",
                description=str(payload.get("description", "")).strip(),
                cells=cells,
                resolution=self.config.occupancy_resolution,
            )
            self.manual_regions.append(region)
            return self.snapshot()

    def update_manual_region_waypoint(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            region_id = str(payload.get("region_id", ""))
            waypoint_name = str(payload.get("waypoint_name", ""))
            pose = payload.get("pose", {})
            region = next((item for item in self.manual_regions if item.get("region_id") == region_id), None)
            if region is None or not isinstance(pose, dict):
                self.last_error = "Could not update waypoint; region or pose was invalid."
                return self.snapshot()
            x = float(pose.get("x", 0.0))
            y = float(pose.get("y", 0.0))
            cell = self._world_to_cell(x, y)
            if self._effective_known_cells().get(cell) != "free":
                self.last_error = "Waypoint was not moved because the target is not known free space."
                return self.snapshot()
            waypoint = next(
                (item for item in region.get("default_waypoints", []) if item.get("name") == waypoint_name),
                None,
            )
            if waypoint is None:
                waypoint = {"name": waypoint_name or f"{region['label']}_waypoint", "kind": "subwaypoint"}
                region.setdefault("default_waypoints", []).append(waypoint)
            waypoint.update(
                {
                    "x": round(x, 3),
                    "y": round(y, 3),
                    "yaw": round(float(pose.get("yaw", 0.0)), 3),
                }
            )
            return self.snapshot()

    def add_manual_region_subwaypoint(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload = dict(payload)
        payload["waypoint_name"] = str(payload.get("name", "")).strip() or "subwaypoint"
        return self.update_manual_region_waypoint(payload)

    def apply_decision(self) -> dict[str, Any]:
        with self._lock:
            self.last_error = None
            if self.paused:
                self.last_error = "Exploration is paused. Resume before applying a decision."
                return self.snapshot()
            if self.pending_decision is None:
                self.last_error = "No pending LLM decision is ready yet."
                return self.snapshot()
            decision = self.pending_decision
            if decision.decision_type == "finish" or decision.exploration_complete:
                self.status = "finished"
                self._log_decision("finish_without_motion")
                self.pending_decision = None
                return self.snapshot()
            if not decision.selected_frontier_id:
                self.last_error = "The LLM decision did not select a frontier."
                return self.snapshot()
            record = self.frontier_memory.activate(decision.selected_frontier_id)
            if record is None:
                self.last_error = f"Selected frontier `{decision.selected_frontier_id}` no longer exists."
                return self.snapshot()

            start_pose = self._current_pose()
            current_cell = self._world_to_cell(start_pose.x, start_pose.y)
            reachable_safe_cells = self._reachable_safe_navigation_cells(current_cell)
            target_pose, rejection_reason = self._resolve_physically_valid_target_pose(
                record=record,
                start_pose=start_pose,
                reachable_safe_cells=reachable_safe_cells,
            )
            if target_pose is None:
                if self._is_close_enough_to_complete_frontier(start_pose, record):
                    record.evidence = _dedupe_text(
                        record.evidence
                        + [
                            "frontier marked completed from the current pose because the exact projected approach pose was not physically valid, but the robot was already close enough to observe the frontier boundary"
                        ]
                    )
                    self.last_error = (
                        f"`{record.frontier_id}` was marked completed from the current pose; "
                        f"no collision-free teleport target was found nearby ({rejection_reason or 'unknown reason'})."
                    )
                    return self._complete_active_frontier_locked(
                        record=record,
                        decision=decision,
                        event_type="teleport_close_enough_completed",
                    )
                self.frontier_memory.fail(
                    record.frontier_id,
                    f"no physically valid teleport target near frontier: {rejection_reason or 'unknown reason'}",
                )
                self.last_error = (
                    f"Teleport rejected `{record.frontier_id}` because no nearby collision-free target pose "
                    f"could be found ({rejection_reason or 'unknown reason'})."
                )
                self._log_decision("teleport_target_invalid")
                self._prepare_decision_locked()
                return self.snapshot()

            if _pose_distance_m(target_pose, record.nav_pose) > self.config.occupancy_resolution * 0.5:
                self.guardrail_events.append(
                    {
                        "type": "teleport_target_pose_adjusted",
                        "frontier_id": record.frontier_id,
                        "original_nav_pose": record.nav_pose.to_dict(),
                        "adjusted_nav_pose": target_pose.to_dict(),
                    }
                )
                record.evidence = _dedupe_text(
                    record.evidence
                    + [
                        "approach pose adjusted to a nearby robot-connected, physically valid free-space pose before teleporting"
                    ]
                )
                record.nav_pose = target_pose

            target_cell = self._world_to_cell(target_pose.x, target_pose.y)
            path_cells = self._known_safe_path_to_cell(current_cell, target_cell)
            if not path_cells:
                if self._is_close_enough_to_complete_frontier(start_pose, record):
                    record.evidence = _dedupe_text(
                        record.evidence
                        + [
                            "frontier marked completed from the current pose because no known-free path to the approach pose remained, but the boundary was already close enough to inspect"
                        ]
                    )
                    self.last_error = (
                        f"`{record.frontier_id}` was marked completed from the current pose; "
                        "no known-free path to the approach pose remained."
                    )
                    return self._complete_active_frontier_locked(
                        record=record,
                        decision=decision,
                        event_type="teleport_path_close_enough_completed",
                    )
                self.frontier_memory.fail(
                    record.frontier_id,
                    "teleport target has no robot-connected, footprint-eroded known-free path",
                )
                self.last_error = (
                    f"Teleport rejected `{record.frontier_id}` because there is no robot-connected, "
                    "footprint-eroded known-free path to the target."
                )
                self._log_decision("teleport_rejected")
                self._prepare_decision_locked()
                return self.snapshot()

            try:
                self._teleport_robot(target_pose)
            except Exception as exc:
                reached_pose = self._current_pose()
                if self._is_close_enough_to_complete_frontier(reached_pose, record):
                    record.evidence = _dedupe_text(
                        record.evidence
                        + [
                            f"teleport pose check reported an error, but the robot landed close enough to the frontier boundary to mark it completed: {exc}"
                        ]
                    )
                    self.last_error = f"Teleport landed close enough for `{record.frontier_id}` despite a pose check warning: {exc}"
                    return self._complete_active_frontier_locked(
                        record=record,
                        decision=decision,
                        event_type="teleport_warning_close_enough_completed",
                    )
                self.frontier_memory.fail(record.frontier_id, f"teleport failed and frontier was not close enough to complete: {exc}")
                self.last_error = f"Teleport failed for `{record.frontier_id}`: {exc}"
                self._log_decision("teleport_failed")
                self._prepare_decision_locked()
                return self.snapshot()

            reached_pose = self._current_pose()
            distance_m = max(len(path_cells) - 1, 0) * self.config.occupancy_resolution
            self.total_distance_m += distance_m
            self.control_steps += 1
            for cell in path_cells[1:-1]:
                self.trajectory.append(cell.center_pose(self.config.occupancy_resolution, yaw=record.nav_pose.yaw).to_dict())
            self.trajectory.append(reached_pose.to_dict())
            try:
                self._perform_scan(
                    full_turnaround=self._scan_uses_turnaround(),
                    capture_frame=True,
                    reason=self._scan_reason(f"teleport_arrive_frontier::{record.frontier_id}"),
                )
            except Exception as exc:
                self.guardrail_events.append(
                    {
                        "type": "arrival_scan_failed_after_teleport",
                        "frontier_id": record.frontier_id,
                        "reason": str(exc),
                    }
                )
                record.evidence = _dedupe_text(
                    record.evidence
                    + [f"arrival scan failed after reaching the frontier approach pose; frontier still marked completed: {exc}"]
                )
                self.last_error = f"Arrival scan failed for `{record.frontier_id}`, but the frontier was reached: {exc}"
            return self._complete_active_frontier_locked(
                record=record,
                decision=decision,
                event_type="teleport_applied",
            )

    def _complete_active_frontier_locked(
        self,
        *,
        record: FrontierRecord,
        decision: ExplorationDecision,
        event_type: str,
    ) -> dict[str, Any]:
        self.frontier_memory.complete(record.frontier_id)
        record.path_cost_m = None
        record.currently_visible = False
        self.frontier_memory.remember_return_waypoint(
            room_id=None,
            pose=self._current_pose(),
            step_index=self.decision_index,
            reason=f"completed_frontier::{record.frontier_id}",
        )
        self.semantic_updates.extend(decision.semantic_updates)
        self._log_decision(event_type)
        self.pending_decision = None
        self.pending_trace = None
        self.applied_memory_updates = []
        self._prepare_decision_locked()
        return self.snapshot()

    def _initialize_environment(self) -> None:
        try:
            import gymnasium as gym
            from multido_xlerobot.maniskill import bootstrap_xlerobot_maniskill
        except Exception as exc:  # pragma: no cover - runtime dependency guard.
            raise RuntimeError(
                "The ManiSkill interactive backend requires gymnasium and the XLeRobot ManiSkill package "
                "in the active Python environment."
            ) from exc

        bootstrap_xlerobot_maniskill(self.options.repo_root, force_reload=self.options.force_reload)
        env_kwargs: dict[str, Any] = {
            "obs_mode": "sensor_data",
            "control_mode": self.options.control_mode,
            "render_mode": self.options.render_mode,
            "sensor_configs": {"shader_pack": self.options.shader},
            "human_render_camera_configs": {"shader_pack": self.options.shader},
            "viewer_camera_configs": {"shader_pack": self.options.shader},
            "robot_uids": self.options.robot_uid,
            "num_envs": self.options.num_envs,
            "sim_backend": self.options.sim_backend,
            "enable_shadow": True,
            "parallel_in_single_scene": False,
        }
        self.env = gym.make(self.options.env_id, **env_kwargs)

    def _reset_environment(self) -> None:
        reset_options: dict[str, Any] = {"reconfigure": True}
        if self.options.build_config_idx is not None:
            reset_options["build_config_idxs"] = self.options.build_config_idx
        self.env.reset(seed=2022, options=reset_options)
        self.robot = self.env.unwrapped.agent.robot
        self.base_link = self.env.unwrapped.agent.base_link
        self.head_link = self.env.unwrapped.agent.head_camera_link
        self.scene = self.env.unwrapped.scene
        self.action = np.zeros(self.env.action_space.shape, dtype=np.float32)
        self._base_qpos_anchor = np.asarray(self._tensor_to_numpy(self.robot.get_qpos()), dtype=np.float64).copy()
        self._base_world_anchor = self._current_pose()
        start_pose = _resolve_manishkill_start_pose(
            self._current_pose(),
            spawn_x=self.options.spawn_x,
            spawn_y=self.options.spawn_y,
            spawn_yaw=self.options.spawn_yaw,
            spawn_facing=self.options.spawn_facing,
        )
        if start_pose is not None:
            self._teleport_robot(start_pose)
            return
        self._settle_after_teleport()

    def _teleport_robot(self, pose: Pose2D) -> None:
        self._set_base_pose_via_qpos(pose)
        achieved = self._current_pose()
        position_error_m = math.hypot(achieved.x - pose.x, achieved.y - pose.y)
        yaw_error_rad = abs(_angle_wrap(achieved.yaw - pose.yaw))
        if position_error_m > BASE_TELEPORT_POSITION_TOLERANCE_M or yaw_error_rad > BASE_TELEPORT_YAW_TOLERANCE_RAD:
            raise RuntimeError(
                "Base teleport landed at "
                f"({achieved.x:.3f}, {achieved.y:.3f}, yaw={achieved.yaw:.3f}) instead of "
                f"({pose.x:.3f}, {pose.y:.3f}, yaw={pose.yaw:.3f}). "
                "This usually means the target pose is colliding with scene geometry or the base-frame/qpos conversion is still wrong."
            )

    def _set_base_pose_via_qpos(self, pose: Pose2D) -> None:
        updated_qpos = _updated_mobile_base_qpos(
            self.robot.get_qpos(),
            pose,
            anchor_pose=self._base_world_anchor,
            anchor_qpos=self._base_qpos_anchor,
        )
        self.robot.set_qpos(updated_qpos)
        try:
            self.robot.set_qvel(_zero_mobile_base_qvel(self.robot.get_qvel()))
        except Exception:
            pass
        self._settle_after_teleport()

    def _probe_teleport_pose(self, pose: Pose2D) -> tuple[bool, str | None]:
        saved_qpos = self.robot.get_qpos()
        saved_qvel = self.robot.get_qvel()
        try:
            self._teleport_robot(pose)
        except Exception as exc:
            try:
                self.robot.set_qpos(saved_qpos)
                self.robot.set_qvel(saved_qvel)
                self._settle_after_teleport()
            except Exception:
                pass
            return False, str(exc)
        try:
            self.robot.set_qpos(saved_qpos)
            self.robot.set_qvel(saved_qvel)
            self._settle_after_teleport()
        except Exception:
            pass
        return True, None

    def _resolve_physically_valid_target_pose(
        self,
        *,
        record: FrontierRecord,
        start_pose: Pose2D,
        reachable_safe_cells: set[GridCell],
    ) -> tuple[Pose2D | None, str | None]:
        candidates = self._candidate_target_poses(
            record=record,
            start_pose=start_pose,
            reachable_safe_cells=reachable_safe_cells,
        )
        rejection_reasons: list[str] = []
        for candidate in candidates:
            ok, reason = self._probe_teleport_pose(candidate)
            if ok:
                return candidate, None
            if reason:
                rejection_reasons.append(reason)
        if rejection_reasons:
            return None, rejection_reasons[-1]
        return None, "no physically valid inward pose candidate was found"

    def _is_close_enough_to_complete_frontier(self, pose: Pose2D, record: FrontierRecord) -> bool:
        completion_radius_m = max(
            STORED_FRONTIER_REVISIT_APPROACH_RADIUS_M,
            self._robot_footprint_radius_m() + 0.75,
            self.config.occupancy_resolution * 4.0,
        )
        return _pose_distance_m(pose, record.centroid_pose) <= completion_radius_m

    def _candidate_target_poses(
        self,
        *,
        record: FrontierRecord,
        start_pose: Pose2D,
        reachable_safe_cells: set[GridCell],
    ) -> list[Pose2D]:
        boundary = record.centroid_pose
        nav = record.nav_pose
        inward_x = nav.x - boundary.x
        inward_y = nav.y - boundary.y
        inward_norm = math.hypot(inward_x, inward_y)
        if inward_norm <= 1e-6:
            inward_x = nav.x - start_pose.x
            inward_y = nav.y - start_pose.y
            inward_norm = math.hypot(inward_x, inward_y)
        if inward_norm <= 1e-6:
            inward_x, inward_y, inward_norm = 1.0, 0.0, 1.0
        inward_x /= inward_norm
        inward_y /= inward_norm

        offsets_m = [0.0, 0.10, 0.20, 0.35, 0.50, 0.70]
        resolution = self.config.occupancy_resolution
        search_radius_cells = max(1, int(math.ceil(0.50 / resolution)))
        candidates: list[Pose2D] = []
        seen_cells: set[GridCell] = set()
        for offset_m in offsets_m:
            desired_x = nav.x + inward_x * offset_m
            desired_y = nav.y + inward_y * offset_m
            desired_cell = self._world_to_cell(desired_x, desired_y)
            snapped_cell: GridCell | None = None
            snapped_score: float | None = None
            for dx in range(-search_radius_cells, search_radius_cells + 1):
                for dy in range(-search_radius_cells, search_radius_cells + 1):
                    cell = GridCell(desired_cell.x + dx, desired_cell.y + dy)
                    if cell not in reachable_safe_cells or cell in seen_cells:
                        continue
                    cell_pose = cell.center_pose(resolution)
                    score = (
                        math.hypot(cell_pose.x - desired_x, cell_pose.y - desired_y)
                        + 0.02 * _pose_distance_m(cell_pose, start_pose)
                    )
                    if snapped_score is None or score < snapped_score:
                        snapped_cell = cell
                        snapped_score = score
            if snapped_cell is None:
                continue
            seen_cells.add(snapped_cell)
            snapped_pose = snapped_cell.center_pose(
                resolution,
                yaw=math.atan2(boundary.y - snapped_cell.center_pose(resolution).y, boundary.x - snapped_cell.center_pose(resolution).x),
            )
            candidates.append(snapped_pose)
        return candidates

    def _settle_after_teleport(self) -> None:
        steps = max(int(self.options.teleport_settle_steps), 0)
        for _ in range(steps):
            self.env.step(self.action)
        if self.options.render_mode == "human":
            self.env.render()

    def _tensor_to_numpy(self, value: Any) -> np.ndarray:
        array = value.cpu().numpy() if hasattr(value, "cpu") else np.asarray(value)
        return np.asarray(array, dtype=np.float64).squeeze()

    def _current_pose(self) -> Pose2D:
        position = self._tensor_to_numpy(self.base_link.pose.p)
        quaternion = normalize_quaternion_wxyz(self._tensor_to_numpy(self.base_link.pose.q))
        return Pose2D(float(position[0]), float(position[1]), quaternion_to_yaw(quaternion))

    def _display_pose(self, pose: Pose2D) -> Pose2D:
        return Pose2D(pose.x, pose.y, _angle_wrap(pose.yaw + self._display_yaw_offset_rad))

    def _scan_uses_turnaround(self) -> bool:
        return self.options.scan_mode == "turnaround"

    def _scan_reason(self, stage: str) -> str:
        if self._scan_uses_turnaround():
            return f"maniskill_{stage}_turnaround_scan"
        return f"maniskill_{stage}_forward_scan"

    def _world_to_cell(self, x: float, y: float) -> GridCell:
        resolution = self.config.occupancy_resolution
        return GridCell(int(math.floor(x / resolution)), int(math.floor(y / resolution)))

    def _capture_head_camera(self) -> dict[str, np.ndarray]:
        self.scene.update_render(update_sensors=True, update_human_render_cameras=False)
        camera = self.scene.sensors[HEAD_CAMERA_UID]
        camera.capture()
        data = camera.get_obs(rgb=True, depth=True, position=False, segmentation=False)
        converted: dict[str, np.ndarray] = {}
        for key, value in data.items():
            array = value.cpu().numpy() if hasattr(value, "cpu") else np.asarray(value)
            converted[key] = np.asarray(array).squeeze()
        return converted

    def _perform_scan(self, *, full_turnaround: bool, capture_frame: bool, reason: str) -> None:
        original_pose = self._current_pose()
        sample_count = max(int(self.options.scan_yaw_samples), 1) if full_turnaround else 1
        yaws = [original_pose.yaw + (math.tau * index / sample_count) for index in range(sample_count)]
        capture_indices = {0}
        if full_turnaround:
            capture_indices = {min(sample_count - 1, round(sample_count * fraction)) for fraction in (0.0, 0.25, 0.5, 0.75)}
        captured_any = False
        previous_active_scan_cells = getattr(self, "_active_scan_cells", None)
        previous_active_scan_known_cells = getattr(self, "_active_scan_known_cells", None)
        previous_active_scan_range_edge_cells = getattr(self, "_active_scan_range_edge_cells", None)
        previous_active_scan_occupancy_evidence = getattr(self, "_active_scan_occupancy_evidence", None)
        self._active_scan_cells: set[GridCell] | None = set()
        self._active_scan_known_cells: dict[GridCell, str] | None = {}
        self._active_scan_range_edge_cells: set[GridCell] | None = set()
        self._active_scan_occupancy_evidence: dict[GridCell, float] | None = {}
        try:
            for index, yaw in enumerate(yaws):
                scan_pose = Pose2D(original_pose.x, original_pose.y, _angle_wrap(yaw))
                try:
                    self._move_to_scan_sample_pose(
                        scan_pose,
                        full_turnaround=full_turnaround,
                        scan_reason=reason,
                        yaw_sample_index=index,
                    )
                except Exception as exc:
                    self.guardrail_events.append(
                        {
                            "type": "scan_yaw_sample_teleport_failed",
                            "reason": str(exc),
                            "scan_reason": reason,
                            "yaw_sample_index": index,
                            "target_pose": scan_pose.to_dict(),
                        }
                    )
                    continue
                head_data = self._capture_head_camera()
                summary = self._integrate_depth_scan(head_data)
                if capture_frame and index in capture_indices:
                    self._append_keyframe(
                        head_data=head_data,
                        summary=summary,
                        reason=f"{reason}::yaw_sample_{index:02d}",
                    )
                    captured_any = True
            if full_turnaround:
                try:
                    self._move_to_scan_sample_pose(
                        Pose2D(original_pose.x, original_pose.y, _angle_wrap(original_pose.yaw)),
                        full_turnaround=full_turnaround,
                        scan_reason=reason,
                        yaw_sample_index=sample_count,
                    )
                except Exception as exc:
                    self.guardrail_events.append(
                        {
                            "type": "scan_restore_pose_failed",
                            "reason": str(exc),
                            "scan_reason": reason,
                            "target_pose": original_pose.to_dict(),
                        }
                    )
            if capture_frame and not captured_any:
                head_data = self._capture_head_camera()
                self._append_keyframe(
                    head_data=head_data,
                    summary=self._integrate_depth_scan(head_data),
                    reason=reason,
                )
        finally:
            self.latest_scan_known_cells = dict(self._active_scan_known_cells or {})
            self.latest_scan_range_edge_cells = set(self._active_scan_range_edge_cells or set())
            self.latest_scan_cells = set(self.latest_scan_known_cells)
            self._merge_latest_scan_into_global()
            self._active_scan_cells = previous_active_scan_cells
            self._active_scan_known_cells = previous_active_scan_known_cells
            self._active_scan_range_edge_cells = previous_active_scan_range_edge_cells
            self._active_scan_occupancy_evidence = previous_active_scan_occupancy_evidence

    def _move_to_scan_sample_pose(
        self,
        pose: Pose2D,
        *,
        full_turnaround: bool,
        scan_reason: str,
        yaw_sample_index: int,
    ) -> None:
        self._teleport_robot(pose)

    def _build_scan_observation_from_head_data(
        self,
        head_data: dict[str, np.ndarray],
    ) -> tuple[dict[str, Any], np.ndarray]:
        depth = np.asarray(head_data.get("depth"))
        if depth.ndim == 3:
            depth = depth[..., 0]
        depth_for_scan = _depth_image_to_millimeters(depth)
        ranges, angles = synthesize_scan_from_depth(
            depth_for_scan,
            horizontal_fov_rad=HEAD_CAMERA_FOV_RAD,
            band_height_px=12,
            range_min_m=0.05,
            range_max_m=self.config.sensor_range_m,
        )
        head_position = self._tensor_to_numpy(self.head_link.pose.p)
        head_quaternion = normalize_quaternion_wxyz(self._tensor_to_numpy(self.head_link.pose.q))
        laser_yaw = quaternion_to_yaw(head_quaternion)
        observation = {
            "frame_id": "maniskill_head_scan",
            "reference_frame": "maniskill_world",
            "pose": Pose2D(float(head_position[0]), float(head_position[1]), laser_yaw),
            "range_min": 0.05,
            "range_max": float(self.config.sensor_range_m),
            "angle_min": float(angles[0]) if len(angles) else -HEAD_CAMERA_FOV_RAD / 2.0,
            "angle_increment": float(angles[1] - angles[0]) if len(angles) > 1 else 0.0,
            "ranges": tuple(float(item) for item in ranges),
        }
        return observation, depth_for_scan

    def _integrate_depth_scan(self, head_data: dict[str, np.ndarray]) -> dict[str, Any]:
        observation, depth_for_scan = self._build_scan_observation_from_head_data(head_data)
        active_scan_cells = getattr(self, "_active_scan_cells", None)
        active_scan_known_cells = getattr(self, "_active_scan_known_cells", None)
        active_scan_range_edge_cells = getattr(self, "_active_scan_range_edge_cells", None)
        active_scan_occupancy_evidence = getattr(self, "_active_scan_occupancy_evidence", None)
        beam_stride = max(int(self.options.depth_beam_stride), 1)
        summary = integrate_planar_scan(
            pose=observation["pose"],
            ranges=observation["ranges"],
            angle_min=float(observation["angle_min"]),
            angle_increment=float(observation["angle_increment"]),
            range_min_m=0.05,
            range_max_m=self.config.sensor_range_m,
            resolution_m=self.config.occupancy_resolution,
            cell_from_world=lambda x, y: self._world_to_cell(x, y),
            known_cells=active_scan_known_cells,
            evidence_scores=active_scan_occupancy_evidence,
            range_edge_cells=active_scan_range_edge_cells,
            visited_cells=active_scan_cells,
            beam_stride=beam_stride,
            config=ACTIVE_RGBD_SCAN_FUSION_CONFIG,
        )
        self._latest_scan_observation = observation

        depth_summary = _depth_summary(depth_for_scan, max_range_m=self.config.sensor_range_m)
        depth_summary.update(
            {
                "point_count": summary.point_count,
                "scan_beams": summary.scan_beams,
                "integrated_beams": summary.integrated_beams,
                "source": "maniskill_xlerobot_head_rgbd",
            }
        )
        return depth_summary

    def _merge_latest_scan_into_global(self) -> None:
        if not hasattr(self, "occupancy_evidence"):
            self.occupancy_evidence = {}
        merge_occupancy_observations(
            self.known_cells,
            self.latest_scan_known_cells,
            evidence_scores=self.occupancy_evidence,
        )
        self.range_edge_cells |= self.latest_scan_range_edge_cells

    def _append_keyframe(self, *, head_data: dict[str, np.ndarray], summary: dict[str, Any], reason: str) -> None:
        pose = self._current_pose()
        frame_id = f"kf_{len(self.keyframes) + 1:03d}"
        frame = {
            "frame_id": frame_id,
            "pose": pose.to_dict(),
            "region_id": "unknown",
            "visible_objects": [],
            "point_count": int(summary.get("point_count", 0)),
            "depth_min_m": summary.get("depth_min_m"),
            "depth_max_m": summary.get("depth_max_m"),
            "description": (
                f"Real ManiSkill XLeRobot head RGB-D frame captured for `{reason}` at "
                f"world pose ({pose.x:.2f}, {pose.y:.2f}, yaw={pose.yaw:.2f})."
            ),
            "rgbd_summary": summary,
            "thumbnail_data_url": _rgb_array_to_data_url(head_data.get("rgb")),
        }
        self.keyframes.append(frame)

    def _prepare_decision_locked(self) -> None:
        self.decision_index += 1
        visible_candidates = self._detect_frontier_candidates()
        self.frontier_memory.upsert_candidates(visible_candidates, step_index=self.decision_index)
        candidate_records = self._refresh_candidate_paths()
        prompt_payload = self._build_prompt_payload(candidate_records)
        self.pending_candidate_records = candidate_records
        self.pending_prompt_payload = prompt_payload
        self.pending_prompt_text = build_exploration_policy_user_prompt(prompt_payload)
        self.pending_decision = None
        self.pending_trace = None
        self.applied_memory_updates = []
        self.status = "waiting_for_llm" if candidate_records else "finished"

    def _known_free_cells(self) -> set[GridCell]:
        return {cell for cell, state in self._effective_known_cells().items() if state == "free"}

    def _effective_known_cells(self) -> dict[GridCell, str]:
        edits = getattr(self, "manual_occupancy_edits", ManualOccupancyEdits())
        return overlay_known_cells(self.known_cells, edits)

    def pause(self) -> dict[str, Any]:
        self.paused = True
        self.status = "paused"
        self.last_error = "Autonomous navigation is paused."
        self._stop_robot_action()
        return self.snapshot()

    def control_robot(self) -> dict[str, Any]:
        self.paused = True
        self.status = "human_control"
        self.last_error = (
            "Manual robot control is active. Use arrow keys to drive, F/V to tilt the head, "
            "and S to capture a scan when keyboard controls are enabled."
        )
        self._stop_robot_action()
        return self.snapshot()

    def _stop_robot_action(self) -> None:
        action = getattr(self, "action", None)
        if action is None:
            return
        try:
            action[...] = 0.0
        except Exception:
            try:
                self.action = np.zeros_like(action)
            except Exception:
                pass

    def resume(self) -> dict[str, Any]:
        with self._lock:
            self.paused = False
            if self.pending_decision is not None:
                self.status = "llm_response_ready"
            elif self.pending_prompt_payload is not None:
                self.status = "waiting_for_llm"
            return self.snapshot()

    def mark_selected_frontier_solved(self) -> dict[str, Any]:
        with self._lock:
            frontier_id = self.frontier_memory.active_frontier_id
            if frontier_id is None and self.pending_decision is not None:
                frontier_id = self.pending_decision.selected_frontier_id
            if not frontier_id:
                self.last_error = "No active or selected frontier is available to mark solved."
                return self.snapshot()
            record = self.frontier_memory.records.get(frontier_id)
            if record is None:
                self.last_error = f"Frontier `{frontier_id}` no longer exists."
                return self.snapshot()
            if self.frontier_memory.active_frontier_id != frontier_id:
                self.frontier_memory.activate(frontier_id)
            decision = self.pending_decision or ExplorationDecision(
                decision_type="explore_frontier",
                selected_frontier_id=frontier_id,
            )
            self.paused = False
            self.last_error = None
            return self._complete_active_frontier_locked(
                record=record,
                decision=decision,
                event_type="human_marked_frontier_solved",
            )

    def update_occupancy_edits(self, *, mode: str, cells: list[dict[str, Any]]) -> dict[str, Any]:
        with self._lock:
            parsed_cells: list[GridCell] = []
            resolution = self.config.occupancy_resolution
            for item in cells:
                if not isinstance(item, dict):
                    continue
                try:
                    if "cell_x" in item and "cell_y" in item:
                        parsed_cells.append(GridCell(int(item["cell_x"]), int(item["cell_y"])))
                    else:
                        parsed_cells.append(
                            GridCell(
                                int(math.floor(float(item["x"]) / resolution)),
                                int(math.floor(float(item["y"]) / resolution)),
                            )
                        )
                except Exception:
                    continue
            self.manual_occupancy_edits.apply(cells=parsed_cells, mode=mode)
            self._prepare_decision_locked()
            return self.snapshot()

    def _global_frontier_components(
        self,
        cluster: list[GridCell],
        *,
        scan_range_edge_cells: set[GridCell],
    ) -> list[tuple[list[GridCell], set[GridCell], int]]:
        effective_known = self._effective_known_cells()
        valid_frontier_cells: set[GridCell] = set()
        unknown_neighbors_by_frontier: dict[GridCell, set[GridCell]] = {}
        for cell in cluster:
            if effective_known.get(cell) != "free":
                continue
            unknown_neighbors = {neighbor for neighbor in _neighbors4(cell) if neighbor not in effective_known}
            if unknown_neighbors:
                valid_frontier_cells.add(cell)
                unknown_neighbors_by_frontier[cell] = unknown_neighbors

        components: list[tuple[list[GridCell], set[GridCell], int]] = []
        visited: set[GridCell] = set()
        for cell in valid_frontier_cells:
            if cell in visited:
                continue
            component: list[GridCell] = []
            queue = [cell]
            visited.add(cell)
            while queue:
                current = queue.pop(0)
                component.append(current)
                for neighbor in _neighbors8(current):
                    if neighbor in valid_frontier_cells and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            component_unknown = set().union(
                *(unknown_neighbors_by_frontier.get(component_cell, set()) for component_cell in component)
            )
            if component_unknown:
                max_unknown_neighbor_count = max(
                    len(unknown_neighbors_by_frontier.get(component_cell, set()))
                    for component_cell in component
                )
                components.append((component, component_unknown, max_unknown_neighbor_count))
        return components

    def _global_frontier_anchor_cell_near_record(
        self,
        record: FrontierRecord,
    ) -> tuple[GridCell | None, str | None]:
        known_cells = self._effective_known_cells()
        range_edge_cells = getattr(self, "range_edge_cells", set())
        resolution = self.config.occupancy_resolution
        current_pose = self._current_pose()
        reachable_free_cells = self._reachable_known_free_cells(
            self._world_to_cell(current_pose.x, current_pose.y)
        )
        boundary_cell = self._world_to_cell(record.centroid_pose.x, record.centroid_pose.y)
        search_radius_cells = max(1, int(math.ceil(STORED_FRONTIER_REVALIDATION_RADIUS_M / resolution)))
        strong_candidates: list[tuple[int, int, GridCell]] = []
        relaxed_candidates: list[tuple[int, int, GridCell]] = []
        unreachable_boundary_candidates = 0
        for dx in range(-search_radius_cells, search_radius_cells + 1):
            for dy in range(-search_radius_cells, search_radius_cells + 1):
                cell = GridCell(boundary_cell.x + dx, boundary_cell.y + dy)
                distance_cells = _grid_distance_cells(cell, boundary_cell)
                if distance_cells > search_radius_cells:
                    continue
                if known_cells.get(cell) != "free":
                    continue
                unknown_neighbors = {neighbor for neighbor in _neighbors4(cell) if neighbor not in known_cells}
                if not unknown_neighbors:
                    continue
                if cell not in reachable_free_cells:
                    unreachable_boundary_candidates += 1
                    continue
                if len(unknown_neighbors) >= 2 or (cell in range_edge_cells and unknown_neighbors):
                    strong_candidates.append((distance_cells, -len(unknown_neighbors), cell))
                else:
                    relaxed_candidates.append((distance_cells, -len(unknown_neighbors), cell))
        if strong_candidates:
            return min(strong_candidates, key=lambda item: (item[0], item[1]))[2], "strong"
        if relaxed_candidates:
            return min(relaxed_candidates, key=lambda item: (item[0], item[1]))[2], "relaxed"
        if unreachable_boundary_candidates:
            self.guardrail_events.append(
                {
                    "type": "stored_frontier_boundary_unreachable_through_known_free",
                    "frontier_id": record.frontier_id,
                    "frontier_boundary_pose": record.centroid_pose.to_dict(),
                    "unreachable_candidate_count": unreachable_boundary_candidates,
                }
            )
        return None, None

    def _record_still_has_global_frontier_boundary(self, record: FrontierRecord) -> bool:
        anchor_cell, _anchor_mode = self._global_frontier_anchor_cell_near_record(record)
        return anchor_cell is not None

    def _revalidate_stored_frontier_boundary(
        self,
        *,
        record: FrontierRecord,
        anchor_cell: GridCell,
        anchor_mode: str | None,
    ) -> None:
        anchor_pose = anchor_cell.center_pose(self.config.occupancy_resolution)
        relaxed_note = (
            "stored frontier memory was kept using relaxed revalidation because nearby free space still borders "
            "unknown map area even though the stricter current-frontier rule no longer matches the original point"
            if anchor_mode == "relaxed"
            else None
        )
        if (
            _pose_distance_m(record.centroid_pose, anchor_pose) <= self.config.occupancy_resolution * 0.5
            and not relaxed_note
        ):
            return
        previous_pose = record.centroid_pose
        record.centroid_pose = anchor_pose
        record.evidence = _dedupe_text(
            record.evidence
            + [
                *(
                    [
                        (
                            "stored frontier boundary was revalidated against the current global occupancy map "
                            f"near the original memory point ({previous_pose.x:.2f}, {previous_pose.y:.2f})"
                        )
                    ]
                    if _pose_distance_m(previous_pose, anchor_pose) > self.config.occupancy_resolution * 0.5
                    else []
                ),
                *( [relaxed_note] if relaxed_note else [] ),
            ]
        )

    def _resnap_stored_frontier_revisit_pose(
        self,
        *,
        record: FrontierRecord,
        current_pose: Pose2D,
        reachable_safe_cells: set[GridCell],
        anchor_cell: GridCell,
    ) -> Pose2D | None:
        candidates = self._candidate_target_poses(
            record=record,
            start_pose=current_pose,
            reachable_safe_cells=reachable_safe_cells,
        )
        if not candidates:
            resolution = self.config.occupancy_resolution
            target_distance_m = self._robot_footprint_radius_m() + 0.25
            max_radius_cells = max(1, int(math.ceil(STORED_FRONTIER_REVISIT_APPROACH_RADIUS_M / resolution)))
            anchor_pose = anchor_cell.center_pose(resolution)
            scored_cells: list[tuple[float, GridCell]] = []
            for cell in reachable_safe_cells:
                distance_cells = _grid_distance_cells(cell, anchor_cell)
                if distance_cells > max_radius_cells:
                    continue
                cell_pose = cell.center_pose(resolution)
                score = (
                    abs((distance_cells * resolution) - target_distance_m)
                    + 0.03 * _pose_distance_m(cell_pose, current_pose)
                    + 0.02 * _pose_distance_m(cell_pose, record.nav_pose)
                )
                scored_cells.append((score, cell))
            for _score, cell in sorted(scored_cells, key=lambda item: item[0]):
                cell_pose = cell.center_pose(resolution)
                candidates.append(
                    cell.center_pose(
                        resolution,
                        yaw=math.atan2(anchor_pose.y - cell_pose.y, anchor_pose.x - cell_pose.x),
                    )
                )
                break
        if not candidates:
            return None
        return min(candidates, key=lambda pose: _pose_distance_m(current_pose, pose))

    def _apply_stored_frontier_resnap(
        self,
        *,
        record: FrontierRecord,
        target_pose: Pose2D,
        previous_pose: Pose2D,
    ) -> None:
        if _pose_distance_m(target_pose, previous_pose) <= self.config.occupancy_resolution * 0.5:
            return
        record.nav_pose = target_pose
        record.evidence = _dedupe_text(
            record.evidence
            + [
                (
                    "stored frontier revisit approach pose was re-snapped to robot-connected, "
                    "footprint-eroded free space before sending the frontier back to the LLM"
                )
            ]
        )
        self.guardrail_events.append(
            {
                "type": "stored_frontier_revisit_pose_resnapped",
                "frontier_id": record.frontier_id,
                "previous_nav_pose": previous_pose.to_dict(),
                "resnapped_nav_pose": target_pose.to_dict(),
            }
        )

    def _detect_frontier_candidates(self) -> list[FrontierCandidate]:
        robot_pose = self._current_pose()
        robot_cell = self._world_to_cell(robot_pose.x, robot_pose.y)
        reachable_free_cells = self._reachable_known_free_cells(robot_cell)
        reachable_safe_cells = self._reachable_safe_navigation_cells(robot_cell)
        scan_known_cells = dict(getattr(self, "latest_scan_known_cells", {}))
        scan_range_edge_cells = set(getattr(self, "latest_scan_range_edge_cells", set()))
        if not scan_known_cells:
            scan_known_cells = dict(self.known_cells)
            scan_range_edge_cells = set(self.range_edge_cells)
        candidate_source_cells = {
            cell
            for cell, state in scan_known_cells.items()
            if state == "free"
        }
        frontier_cells: set[GridCell] = set()
        unknown_neighbors_by_frontier: dict[GridCell, set[GridCell]] = {}
        for cell in candidate_source_cells:
            unknown_neighbors = {neighbor for neighbor in _neighbors4(cell) if neighbor not in scan_known_cells}
            if unknown_neighbors:
                frontier_cells.add(cell)
                unknown_neighbors_by_frontier[cell] = unknown_neighbors

        clusters: list[list[GridCell]] = []
        visited: set[GridCell] = set()
        for cell in frontier_cells:
            if cell in visited:
                continue
            cluster: list[GridCell] = []
            queue = [cell]
            visited.add(cell)
            while queue:
                current = queue.pop(0)
                cluster.append(current)
                for neighbor in _neighbors8(current):
                    if neighbor in frontier_cells and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            clusters.append(cluster)

        candidates: list[FrontierCandidate] = []
        for local_cluster in clusters:
            local_cluster_unknown = set().union(
                *(unknown_neighbors_by_frontier.get(cell, set()) for cell in local_cluster)
            )
            if not local_cluster_unknown:
                continue
            global_components = self._global_frontier_components(
                local_cluster,
                scan_range_edge_cells=scan_range_edge_cells,
            )
            if not global_components:
                self.guardrail_events.append(
                    {
                        "type": "frontier_not_global_boundary_after_merge",
                        "local_cluster_size": len(local_cluster),
                        "local_unknown_gain": len(local_cluster_unknown),
                        "frontier_boundary_pose": _cell_mean_pose(local_cluster, self.config.occupancy_resolution).to_dict(),
                    }
                )
                continue

            required_opening_m = self._min_frontier_opening_width_m()
            for cluster, cluster_unknown, max_unknown_neighbor_count in global_components:
                if not any(cell in reachable_free_cells for cell in cluster):
                    self.guardrail_events.append(
                        {
                            "type": "frontier_boundary_unreachable_through_known_free",
                            "cluster_size": len(cluster),
                            "unknown_gain": len(cluster_unknown),
                            "frontier_boundary_pose": _cell_mean_pose(cluster, self.config.occupancy_resolution).to_dict(),
                        }
                    )
                    continue
                opening_width_m = _frontier_opening_width_m(cluster, self.config.occupancy_resolution)
                if opening_width_m < required_opening_m:
                    self.guardrail_events.append(
                        {
                            "type": "frontier_opening_too_narrow",
                            "cluster_size": len(cluster),
                            "unknown_gain": len(cluster_unknown),
                            "opening_width_m": round(opening_width_m, 3),
                            "required_width_m": round(required_opening_m, 3),
                            "frontier_boundary_pose": _cell_mean_pose(cluster, self.config.occupancy_resolution).to_dict(),
                        }
                    )
                    continue
                boundary_pose = _cell_mean_pose(cluster, self.config.occupancy_resolution)
                visited_filter_radius_m = self._visited_frontier_filter_radius_m()
                if self._is_pose_near_previous_visit(boundary_pose, visited_filter_radius_m):
                    self.guardrail_events.append(
                        {
                            "type": "frontier_near_visited_pose_filtered",
                            "cluster_size": len(cluster),
                            "unknown_gain": len(cluster_unknown),
                            "filter_radius_m": round(visited_filter_radius_m, 3),
                            "frontier_boundary_pose": boundary_pose.to_dict(),
                        }
                    )
                    continue
                approach_cell = self._select_frontier_approach_cell(
                    cluster=cluster,
                    unknown_cells=cluster_unknown,
                    robot_cell=robot_cell,
                    reachable_safe_cells=reachable_safe_cells,
                )
                if approach_cell is None:
                    self.guardrail_events.append(
                        {
                            "type": "frontier_without_safe_approach_pose",
                            "cluster_size": len(cluster),
                            "unknown_gain": len(cluster_unknown),
                            "frontier_boundary_pose": _cell_mean_pose(cluster, self.config.occupancy_resolution).to_dict(),
                            "robot_footprint_radius_m": round(self._robot_footprint_radius_m(), 3),
                        }
                    )
                    continue
                centroid_cell = _centroid_cell(cluster)
                nav_yaw = math.atan2(boundary_pose.y - approach_cell.center_pose(self.config.occupancy_resolution).y, boundary_pose.x - approach_cell.center_pose(self.config.occupancy_resolution).x)
                nav_pose = approach_cell.center_pose(self.config.occupancy_resolution, yaw=nav_yaw)
                centroid_pose = centroid_cell.center_pose(self.config.occupancy_resolution)
                evidence = [
                    f"{len(cluster_unknown)} unknown neighbor cells still unknown after merging this RGB-D scan into the global occupancy map",
                    f"cluster size {len(cluster)}",
                    f"frontier opening width is {opening_width_m:.2f} m, above robot-sized threshold {required_opening_m:.2f} m",
                    f"frontier boundary centroid is at ({boundary_pose.x:.2f}, {boundary_pose.y:.2f})",
                    "frontier passed merged-map validation, so local scan edges buried inside known free space are rejected",
                    (
                        "approach pose is offset inward and drawn from robot-connected, footprint-eroded free space "
                        f"for a {self._robot_footprint_radius_m():.2f} m XLeRobot/IKEA-cart footprint radius"
                    ),
                    "frontier information from actual XLeRobot head RGB-D, not complete apartment knowledge",
                    "movement in this playground is teleport-only and does not prove Nav2 reachability",
                ]
                if max_unknown_neighbor_count >= 2:
                    evidence.append(
                        "frontier signal is stronger: multiple unknown-facing neighbors support likely map expansion"
                    )
                else:
                    evidence.append(
                        "frontier signal is weaker: a single unknown-facing edge was kept because robot-sized free space exists nearby; the LLM should veto it if RGB/map evidence suggests no useful opening"
                    )
                if any(cell in scan_range_edge_cells for cell in cluster):
                    evidence.append("frontier aligns with an RGB-D sensor range limit")
                candidates.append(
                    FrontierCandidate(
                        frontier_id=None,
                        member_cells=tuple(sorted(cluster)),
                        nav_cell=approach_cell,
                        centroid_cell=centroid_cell,
                        nav_pose=nav_pose,
                        centroid_pose=centroid_pose,
                        unknown_gain=len(cluster_unknown),
                        sensor_range_edge=any(cell in scan_range_edge_cells for cell in cluster),
                        room_hint=None,
                        evidence=evidence,
                        currently_visible=_pose_distance_m(nav_pose, robot_pose) <= self.config.sensor_range_m + 0.5,
                    )
                )
        candidates.sort(
            key=lambda candidate: (
                _grid_distance_cells(candidate.nav_cell, robot_cell),
                -candidate.unknown_gain,
            )
        )
        return candidates[: max(int(self.options.max_frontiers), 1)]

    def _select_frontier_approach_cell(
        self,
        *,
        cluster: list[GridCell],
        unknown_cells: set[GridCell],
        robot_cell: GridCell,
        reachable_safe_cells: set[GridCell],
    ) -> GridCell | None:
        resolution = self.config.occupancy_resolution
        known_free = reachable_safe_cells
        if not cluster or not unknown_cells:
            return None

        frontier_x = sum((cell.x + 0.5) * resolution for cell in cluster) / len(cluster)
        frontier_y = sum((cell.y + 0.5) * resolution for cell in cluster) / len(cluster)
        unknown_x = sum((cell.x + 0.5) * resolution for cell in unknown_cells) / len(unknown_cells)
        unknown_y = sum((cell.y + 0.5) * resolution for cell in unknown_cells) / len(unknown_cells)
        inward_x = frontier_x - unknown_x
        inward_y = frontier_y - unknown_y
        inward_norm = math.hypot(inward_x, inward_y)
        if inward_norm <= 1e-6:
            robot_pose = robot_cell.center_pose(resolution)
            inward_x = robot_pose.x - frontier_x
            inward_y = robot_pose.y - frontier_y
            inward_norm = math.hypot(inward_x, inward_y)
        if inward_norm <= 1e-6:
            inward_x, inward_y, inward_norm = 1.0, 0.0, 1.0
        inward_x /= inward_norm
        inward_y /= inward_norm

        footprint_radius_m = self._robot_footprint_radius_m()
        offsets_m = [
            footprint_radius_m + 0.05,
            footprint_radius_m + 0.20,
            footprint_radius_m + 0.35,
            footprint_radius_m + 0.55,
            footprint_radius_m + 0.80,
        ]
        projected_candidates: list[tuple[float, GridCell]] = []
        for offset_m in offsets_m:
            desired_x = frontier_x + inward_x * offset_m
            desired_y = frontier_y + inward_y * offset_m
            desired_cell = self._world_to_cell(desired_x, desired_y)
            search_radius_cells = max(1, int(math.ceil(0.40 / resolution)))
            for dx in range(-search_radius_cells, search_radius_cells + 1):
                for dy in range(-search_radius_cells, search_radius_cells + 1):
                    cell = GridCell(desired_cell.x + dx, desired_cell.y + dy)
                    if cell not in known_free:
                        continue
                    cell_pose = cell.center_pose(resolution)
                    score = (
                        math.hypot(cell_pose.x - desired_x, cell_pose.y - desired_y)
                        + 0.04 * _grid_distance_cells(cell, robot_cell) * resolution
                        + 0.10 / max(_grid_distance_cells(cell, _centroid_cell(cluster)), 1)
                    )
                    projected_candidates.append((score, cell))

        if not projected_candidates:
            fallback_radius_cells = max(1, int(math.ceil((footprint_radius_m + 0.65) / resolution)))
            boundary = _centroid_cell(cluster)
            for cell in known_free:
                distance_cells = _grid_distance_cells(cell, boundary)
                if distance_cells > fallback_radius_cells:
                    continue
                score = (
                    abs((distance_cells * resolution) - (footprint_radius_m + 0.25))
                    + 0.04 * _grid_distance_cells(cell, robot_cell) * resolution
                )
                projected_candidates.append((score, cell))

        seen: set[GridCell] = set()
        for _score, cell in sorted(projected_candidates, key=lambda item: item[0]):
            if cell in seen:
                continue
            seen.add(cell)
            if self._is_valid_robot_center_cell(
                cell,
                required_known_fraction=STRICT_NAVIGATION_KNOWN_FRACTION,
                unknown_is_blocking=True,
                extra_clearance_m=STRICT_NAVIGATION_CLEARANCE_MARGIN_M,
            ):
                return cell
        return None

    def _reachable_safe_navigation_cells(self, robot_cell: GridCell) -> set[GridCell]:
        effective_known = self._effective_known_cells()
        if effective_known.get(robot_cell) != "free":
            return set()
        safe_cells = {
            cell
            for cell in self._known_free_cells()
            if self._is_valid_robot_center_cell(
                cell,
                required_known_fraction=STRICT_NAVIGATION_KNOWN_FRACTION,
                unknown_is_blocking=True,
                extra_clearance_m=STRICT_NAVIGATION_CLEARANCE_MARGIN_M,
            )
        }
        if not safe_cells:
            return set()
        traversable = set(safe_cells)
        traversable.add(robot_cell)
        reachable: set[GridCell] = set()
        queue = [robot_cell]
        while queue:
            current = queue.pop(0)
            if current in reachable:
                continue
            reachable.add(current)
            for neighbor in _neighbors4(current):
                if neighbor in traversable and neighbor not in reachable:
                    queue.append(neighbor)
        return reachable & safe_cells

    def _reachable_known_free_cells(self, robot_cell: GridCell) -> set[GridCell]:
        effective_known = self._effective_known_cells()
        if effective_known.get(robot_cell) != "free":
            return set()
        reachable: set[GridCell] = set()
        queue = [robot_cell]
        while queue:
            current = queue.pop(0)
            if current in reachable:
                continue
            if effective_known.get(current) != "free":
                continue
            reachable.add(current)
            for neighbor in _neighbors4(current):
                if neighbor not in reachable and effective_known.get(neighbor) == "free":
                    queue.append(neighbor)
        return reachable

    def _known_safe_path_to_cell(self, start_cell: GridCell, target_cell: GridCell) -> list[GridCell]:
        reachable_safe_cells = self._reachable_safe_navigation_cells(start_cell)
        if target_cell not in reachable_safe_cells:
            return []
        traversable = set(reachable_safe_cells)
        traversable.add(start_cell)
        return _search_known_safe_path(start_cell, target_cell, traversable)

    def _is_valid_robot_center_cell(
        self,
        cell: GridCell,
        *,
        required_known_fraction: float = 0.55,
        unknown_is_blocking: bool = False,
        extra_clearance_m: float = 0.0,
    ) -> bool:
        effective_known = self._effective_known_cells()
        if effective_known.get(cell) != "free":
            return False
        resolution = self.config.occupancy_resolution
        clearance_radius_m = self._robot_footprint_radius_m() + max(float(extra_clearance_m), 0.0)
        radius_cells = max(1, int(math.ceil(clearance_radius_m / resolution)))
        known_count = 0
        footprint_count = 0
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                sample = GridCell(cell.x + dx, cell.y + dy)
                distance_m = math.hypot(dx * resolution, dy * resolution)
                if distance_m > clearance_radius_m:
                    continue
                footprint_count += 1
                state = effective_known.get(sample)
                if state == "occupied":
                    return False
                if state is None and unknown_is_blocking:
                    return False
                if state == "free":
                    known_count += 1
        if footprint_count == 0:
            return False
        return known_count / footprint_count >= required_known_fraction

    def _robot_footprint_radius_m(self) -> float:
        return max(
            float(self.config.robot_radius_m),
            XLEROBOT_IKEA_CART_FOOTPRINT_RADIUS_M + XLEROBOT_IKEA_CART_CLEARANCE_PADDING_M,
        )

    def _visited_frontier_filter_radius_m(self) -> float:
        configured = self.config.visited_frontier_filter_radius_m
        if configured is not None and configured > 0.0:
            return max(float(configured), self.config.occupancy_resolution)
        return max(self._robot_footprint_radius_m() + 0.35, self.config.occupancy_resolution * 2.0)

    def _previous_trajectory_poses(self) -> list[Pose2D]:
        trajectory = getattr(self, "trajectory", [])
        return [
            pose
            for pose in (_pose_from_mapping(item) for item in trajectory[:-1])
            if pose is not None
        ]

    def _is_pose_near_previous_visit(self, pose: Pose2D, radius_m: float) -> bool:
        return any(_pose_distance_m(pose, visited_pose) <= radius_m for visited_pose in self._previous_trajectory_poses())

    def _is_frontier_near_visited_pose(self, record: FrontierRecord, radius_m: float) -> bool:
        return self._is_pose_near_previous_visit(record.centroid_pose, radius_m)

    def _min_frontier_opening_width_m(self) -> float:
        configured = self.config.frontier_min_opening_m
        if configured is not None and configured > 0.0:
            return max(float(configured), self.config.occupancy_resolution)
        cart_width_with_clearance_m = (
            XLEROBOT_IKEA_CART_FOOTPRINT_WIDTH_M
            + 2.0 * XLEROBOT_IKEA_CART_CLEARANCE_PADDING_M
            + 0.05
        )
        return max(
            self.config.robot_radius_m * 2.0 + 0.10,
            cart_width_with_clearance_m,
            self.config.occupancy_resolution * 2.0,
        )

    def _refresh_candidate_paths(self) -> list[FrontierRecord]:
        current_pose = self._current_pose()
        current_cell = self._world_to_cell(current_pose.x, current_pose.y)
        current_pose_filter_m = max(self.config.occupancy_resolution * 1.5, self._robot_footprint_radius_m())
        def _path_cost(record: FrontierRecord) -> float | None:
            target_cell = self._world_to_cell(record.nav_pose.x, record.nav_pose.y)
            path_cells = self._known_safe_path_to_cell(current_cell, target_cell)
            return max(len(path_cells) - 1, 0) * self.config.occupancy_resolution if path_cells else None

        reachable_safe_cells = self._reachable_safe_navigation_cells(current_cell)

        def _is_frontier_at_current_pose(record: FrontierRecord, filter_radius_m: float) -> bool:
            boundary_cell = self._world_to_cell(record.centroid_pose.x, record.centroid_pose.y)
            return boundary_cell == current_cell or _pose_distance_m(record.centroid_pose, current_pose) <= filter_radius_m

        return refresh_frontier_records(
            candidate_records=self.frontier_memory.candidate_records(),
            active_frontier_id=self.frontier_memory.active_frontier_id,
            current_pose=current_pose,
            current_pose_filter_m=current_pose_filter_m,
            path_cost_for_record=_path_cost,
            guardrail_events=self.guardrail_events,
            is_frontier_at_current_pose=_is_frontier_at_current_pose,
            is_frontier_near_visited_pose=self._is_frontier_near_visited_pose,
            visited_pose_filter_m=self._visited_frontier_filter_radius_m(),
            global_anchor_for_stored_record=self._global_frontier_anchor_cell_near_record,
            revalidate_stored_boundary=lambda record, anchor_cell, anchor_mode: self._revalidate_stored_frontier_boundary(
                record=record,
                anchor_cell=anchor_cell,
                anchor_mode=anchor_mode,
            ),
            resnap_stored_nav_pose=lambda record, pose, anchor_cell: self._resnap_stored_frontier_revisit_pose(
                record=record,
                current_pose=pose,
                reachable_safe_cells=reachable_safe_cells,
                anchor_cell=anchor_cell,
            ),
            apply_stored_resnap=lambda record, target_pose, previous_pose: self._apply_stored_frontier_resnap(
                record=record,
                target_pose=target_pose,
                previous_pose=previous_pose,
            ),
            max_frontiers=max(int(self.options.max_frontiers), 1),
        )

    def _build_prompt_payload(self, candidate_records: list[FrontierRecord]) -> dict[str, Any]:
        effective_known = self._effective_known_cells()
        known_free = sum(1 for state in effective_known.values() if state == "free")
        occupied = sum(1 for state in effective_known.values() if state == "occupied")
        navigation_map_image = _navigation_map_data_url(
            known_cells=effective_known,
            resolution=self.config.occupancy_resolution,
            trajectory=self.trajectory,
            robot_pose=self._current_pose(),
            candidate_records=candidate_records,
            remembered_records=[],
        )
        return {
            "mission": (
                "Interactive ManiSkill no-Nav2 test: use actual XLeRobot head RGB-D images and a deterministic "
                "2D RGB-D occupancy projection to choose useful robot-navigable exploration regions. "
                "The movement executor will teleport the robot after operator review, so this tests LLM frontier choice "
                "and visual interpretation, not Nav2 control."
            ),
            "robot": {
                "pose": self._current_pose().to_dict(),
                "room_id": None,
                "coverage": round(self._coverage(), 3),
                "trajectory_points": len(self.trajectory),
                "sensor_range_m": self.config.sensor_range_m,
                "navigation_mode": "maniskill_teleport_no_nav2",
                "rgbd_source": "actual_manishkill_xlerobot_head_camera",
            },
            "frontier_memory": _frontier_memory_prompt_context(self.frontier_memory),
            "frontier_information": [record.to_dict() for record in candidate_records],
            "frontier_selection_guidance": _frontier_selection_guidance(),
            "explored_areas": [
                {
                    "region_id": "maniskill_mapped_space",
                    "label": "mapped_space",
                    "observed_fraction": round(self._coverage(), 3),
                    "objects_seen": [],
                    "representative_frames": [item["frame_id"] for item in self.keyframes[-6:]],
                }
            ],
            "navigation_map_views": [
                {
                    "frame_id": f"nav_map_{self.decision_index:03d}",
                    "description": (
                        "Operator-review navigation map rendered from the same occupancy/frontier data as the web UI. "
                        "Green frontier labels correspond exactly to Frontier Information; red is the robot and heading."
                    ),
                    "thumbnail_data_url": navigation_map_image,
                    "frontier_count": len(candidate_records),
                }
            ] if navigation_map_image else [],
            "recent_views": self.keyframes[-6:],
            "map_stats": {
                "known_free_cells": known_free,
                "occupied_cells": occupied,
                "known_cells": len(effective_known),
                "map_bounds": self._map_bounds(),
                "robot_footprint": {
                    "source": "xlerobot URDF base_link IKEA/RASKOG cart collision boxes",
                    "length_m": XLEROBOT_IKEA_CART_FOOTPRINT_LENGTH_M,
                    "width_m": XLEROBOT_IKEA_CART_FOOTPRINT_WIDTH_M,
                    "clearance_radius_m": round(self._robot_footprint_radius_m(), 3),
                },
            },
            "guardrails": {
                "frontier_information_is_partial_rgbd_scan_evidence": True,
                "uses_actual_manishkill_rgbd": True,
                "navigation_is_teleport_mocked_without_nav2": True,
                "frontier_ids_must_come_from_prompt": True,
                "memory_updates_must_use_existing_frontier_ids": True,
                "avoid_furniture_shadow_boundaries_without_clear_open_space": True,
                "do_not_claim_nav2_reachability_from_this_playground": True,
            },
            "ascii_map": self._ascii_map(candidate_records),
        }

    def _ascii_map(self, candidate_records: list[FrontierRecord]) -> str:
        effective_known = self._effective_known_cells()
        if not effective_known:
            return "map unavailable"
        robot_pose = self._current_pose()
        robot_cell = self._world_to_cell(robot_pose.x, robot_pose.y)
        frontier_cells = {
            self._world_to_cell(record.nav_pose.x, record.nav_pose.y): record.status
            for record in candidate_records
        }
        interesting = set(effective_known) | set(frontier_cells) | {robot_cell}
        min_x = min(cell.x for cell in interesting) - 2
        max_x = max(cell.x for cell in interesting) + 2
        min_y = min(cell.y for cell in interesting) - 2
        max_y = max(cell.y for cell in interesting) + 2
        max_width = 72
        max_height = 48
        if max_x - min_x + 1 > max_width:
            half = max_width // 2
            min_x = robot_cell.x - half
            max_x = min_x + max_width - 1
        if max_y - min_y + 1 > max_height:
            half = max_height // 2
            min_y = robot_cell.y - half
            max_y = min_y + max_height - 1
        lines: list[str] = []
        for y in reversed(range(min_y, max_y + 1)):
            row: list[str] = []
            for x in range(min_x, max_x + 1):
                cell = GridCell(x, y)
                if cell == robot_cell:
                    row.append("R")
                elif cell in frontier_cells:
                    row.append("V" if frontier_cells[cell] in {"completed", "failed", "suppressed"} else "F")
                else:
                    state = effective_known.get(cell)
                    row.append("." if state == "free" else "#" if state == "occupied" else "?")
            lines.append("".join(row))
        return "\n".join(lines)

    def _apply_finish_guardrail(
        self,
        decision: ExplorationDecision,
        candidate_records: list[FrontierRecord],
    ) -> ExplorationDecision:
        reachable = [record for record in candidate_records if record.path_cost_m is not None]
        if decision.decision_type == "finish" and reachable:
            fallback = self.policy._heuristic_decision(
                candidate_records,
                list(self.frontier_memory.return_waypoints.values()),
                self._coverage(),
                None,
            )
            self.guardrail_events.append(
                {
                    "type": "finish_with_reachable_frontiers",
                    "requested": decision.to_dict(),
                    "fallback": fallback.to_dict(),
                    "reachable_frontier_ids": [record.frontier_id for record in reachable],
                }
            )
            return fallback
        return decision

    def _pending_target(self) -> dict[str, Any] | None:
        if self.pending_decision is None or not self.pending_decision.selected_frontier_id:
            return None
        record = self.frontier_memory.records.get(self.pending_decision.selected_frontier_id)
        if record is None:
            return None
        return {
            "frontier_id": record.frontier_id,
            "nav_pose": record.nav_pose.to_dict(),
            "centroid_pose": record.centroid_pose.to_dict(),
            "status": record.status,
            "path_cost_m": record.path_cost_m,
        }

    def _log_decision(self, event_type: str) -> None:
        if self.pending_decision is None:
            return
        self.decision_log.append(
            {
                "step_index": self.decision_index,
                "event_type": event_type,
                "coverage": round(self._coverage(), 3),
                "decision": self.pending_decision.to_dict(),
                "trace": self.pending_trace or {},
                "applied_memory_updates": list(self.applied_memory_updates),
            }
        )

    def _coverage(self) -> float:
        effective_known = self._effective_known_cells()
        if not effective_known:
            return 0.0
        min_x = min(cell.x for cell in effective_known) - 3
        max_x = max(cell.x for cell in effective_known) + 3
        min_y = min(cell.y for cell in effective_known) - 3
        max_y = max(cell.y for cell in effective_known) + 3
        bbox_cells = max((max_x - min_x + 1) * (max_y - min_y + 1), 1)
        return min(len(effective_known) / bbox_cells, 1.0)

    def _map_bounds(self) -> dict[str, float]:
        resolution = self.config.occupancy_resolution
        effective_known = self._effective_known_cells()
        if not effective_known:
            pose = self._current_pose()
            return {
                "min_x": round(pose.x - 1.0, 3),
                "max_x": round(pose.x + 1.0, 3),
                "min_y": round(pose.y - 1.0, 3),
                "max_y": round(pose.y + 1.0, 3),
            }
        min_x = min(cell.x for cell in effective_known) * resolution
        max_x = (max(cell.x for cell in effective_known) + 1) * resolution
        min_y = min(cell.y for cell in effective_known) * resolution
        max_y = (max(cell.y for cell in effective_known) + 1) * resolution
        return {
            "min_x": round(min_x, 3),
            "max_x": round(max_x, 3),
            "min_y": round(min_y, 3),
            "max_y": round(max_y, 3),
        }

    def _build_map_payload(self) -> dict[str, Any]:
        resolution = self.config.occupancy_resolution
        edits = getattr(self, "manual_occupancy_edits", ManualOccupancyEdits())
        occupancy_cells = []
        for cell, state in sorted(self._effective_known_cells().items()):
            item = {
                "x": round(cell.x * resolution, 3),
                "y": round(cell.y * resolution, 3),
                "state": state,
            }
            if cell in edits.blocked_cells:
                item["manual_override"] = "blocked"
            elif cell in edits.cleared_cells:
                item["manual_override"] = "cleared"
            occupancy_cells.append(item)
        semantic_area_candidates = []
        if self.config.experimental_free_space_semantic_waypoints:
            semantic_area_candidates = _aggregate_semantic_updates(self.semantic_updates)
        semantic_memory = self.semantic_observer.snapshot() if self.config.automatic_semantic_waypoints else {}
        regions = list(getattr(self, "manual_regions", []))
        pending_ids = {record.frontier_id for record in self.pending_candidate_records}
        remembered_frontiers = [
            record.to_dict()
            for record in self.frontier_memory.records.values()
            if record.frontier_id not in pending_ids and record.status == "stored"
        ]
        return {
            "map_id": self.config.session,
            "frame": "maniskill_world",
            "resolution": float(resolution),
            "coverage": round(self._coverage(), 3),
            "mode": "interactive_manishkill_rgbd_teleport_no_nav2",
            "summary": (
                "Robot exploration mode using the actual ManiSkill scene and XLeRobot head RGB-D, "
                "with teleport-only movement after operator-approved LLM decisions."
            ),
            "trajectory": self.trajectory,
            "keyframes": self.keyframes,
            "regions": regions,
            "named_places": _named_places_from_manual_regions(regions)
            + (_semantic_named_places_for_map(semantic_memory) if self.config.automatic_semantic_waypoints else []),
            "frontiers": [record.to_dict() for record in self.pending_candidate_records],
            "remembered_frontiers": remembered_frontiers,
            "semantic_area_candidates": semantic_area_candidates,
            "semantic_memory": semantic_memory,
            "automatic_semantic_waypoints": self.config.automatic_semantic_waypoints,
            "occupancy": {
                "resolution": float(resolution),
                "bounds": self._map_bounds(),
                "cells": occupancy_cells,
            },
            "artifacts": {
                "decision_log": self.decision_log,
                "frontier_memory": self.frontier_memory.snapshot(),
                "all_frontiers": [record.to_dict() for record in self.frontier_memory.records.values()],
                "guardrail_events": self.guardrail_events,
                "manual_occupancy_edits": edits.to_dict(
                    resolution=resolution,
                ),
                "navigation": {
                    "mode": "maniskill_teleport_no_nav2",
                    "control_steps": self.control_steps,
                    "total_distance_m": round(self.total_distance_m, 3),
                },
                "maniskill": {
                    "env_id": self.options.env_id,
                    "robot_uid": self.options.robot_uid,
                    "render_mode": self.options.render_mode,
                    "scan_mode": self.options.scan_mode,
                    "scan_yaw_samples": self.options.scan_yaw_samples,
                    "spawn_facing": self.options.spawn_facing,
                    "spawn_yaw": self.options.spawn_yaw,
                    "depth_beam_stride": self.options.depth_beam_stride,
                    "latest_scan_cells": len(self.latest_scan_known_cells),
                    "latest_scan_range_edge_cells": len(self.latest_scan_range_edge_cells),
                    "current_frontier_candidates": len(self.pending_candidate_records),
                    "stored_memory_frontiers": len(remembered_frontiers),
                },
                "llm_policy": {
                    "explorer_policy": self.config.explorer_policy,
                    "provider": self.config.llm_provider,
                    "model": self.config.llm_model,
                },
            },
        }


class ManiSkillNav2RouterExplorationSession(ManiSkillTeleportExplorationSession):
    """ManiSkill-backed brain session that asks a remote ROS/Nav2 router to validate paths."""

    def __init__(self, config: SimExplorationConfig, options: ManiSkillInteractiveOptions) -> None:
        if not config.ros_adapter_url:
            raise RuntimeError("`--ros-adapter-url` is required for the ManiSkill + Nav2 router flow.")
        self.router = RemoteNav2RouterClient(config.ros_adapter_url, timeout_s=config.ros_adapter_timeout_s)
        super().__init__(config, options)

    def reset(self) -> dict[str, Any]:
        self._transient_navigation_obstacle_cells: set[GridCell] = set()
        self._planned_nav_path: list[dict[str, Any]] = []
        snapshot = super().reset()
        self._publish_router_state()
        return snapshot

    def visualize_planned_nav_path(self) -> dict[str, Any]:
        with self._lock:
            self.last_error = None
            self._planned_nav_path = []
            if self.paused:
                self.last_error = "Exploration is paused. Resume before visualizing a Nav2 path."
                return self.snapshot()
            if self.pending_decision is None:
                self.last_error = "No pending LLM decision is ready yet."
                return self.snapshot()
            decision = self.pending_decision
            if decision.decision_type == "finish" or decision.exploration_complete:
                self.last_error = "The pending LLM decision is finish; there is no frontier path to visualize."
                return self.snapshot()
            if not decision.selected_frontier_id:
                self.last_error = "The LLM decision did not select a frontier."
                return self.snapshot()
            record = self.frontier_memory.records.get(decision.selected_frontier_id)
            if record is None:
                self.last_error = f"Selected frontier `{decision.selected_frontier_id}` no longer exists."
                return self.snapshot()

            start_pose = self._current_pose()
            current_cell = self._world_to_cell(start_pose.x, start_pose.y)
            reachable_safe_cells = self._reachable_safe_navigation_cells(current_cell)
            target_pose, path_poses, rejection_reason = self._plan_router_path_to_frontier(
                record=record,
                start_pose=start_pose,
                reachable_safe_cells=reachable_safe_cells,
            )
            self._planned_nav_path = [p.to_dict() for p in path_poses]
            if target_pose is None or not path_poses:
                self.last_error = (
                    f"Nav2 could not compute a preview path for `{record.frontier_id}` "
                    f"({rejection_reason or 'unknown reason'})."
                )
            else:
                self.guardrail_events.append(
                    {
                        "type": "nav2_planned_path_visualized",
                        "frontier_id": record.frontier_id,
                        "path_pose_count": len(path_poses),
                        "target_pose": target_pose.to_dict(),
                    }
                )
            return self.snapshot()

    def apply_decision(self) -> dict[str, Any]:
        with self._lock:
            self.last_error = None
            if self.paused:
                self.last_error = "Exploration is paused. Resume before applying a decision."
                return self.snapshot()
            if self.pending_decision is None:
                self.last_error = "No pending LLM decision is ready yet."
                return self.snapshot()
            decision = self.pending_decision
            if decision.decision_type == "finish" or decision.exploration_complete:
                self.status = "finished"
                self._log_decision("finish_without_motion")
                self.pending_decision = None
                return self.snapshot()
            if not decision.selected_frontier_id:
                self.last_error = "The LLM decision did not select a frontier."
                return self.snapshot()
            record = self.frontier_memory.activate(decision.selected_frontier_id)
            if record is None:
                self.last_error = f"Selected frontier `{decision.selected_frontier_id}` no longer exists."
                return self.snapshot()

            start_pose = self._current_pose()
            current_cell = self._world_to_cell(start_pose.x, start_pose.y)
            reachable_safe_cells = self._reachable_safe_navigation_cells(current_cell)
            target_pose, path_poses, rejection_reason = self._plan_router_path_to_frontier(
                record=record,
                start_pose=start_pose,
                reachable_safe_cells=reachable_safe_cells,
            )
            self._planned_nav_path = [p.to_dict() for p in path_poses]
            if target_pose is None or not path_poses:
                if self._is_close_enough_to_complete_frontier(start_pose, record):
                    record.evidence = _dedupe_text(
                        record.evidence
                        + [
                            "frontier marked completed from the current pose because no router-approved approach path was found, but the robot was already close enough to observe the frontier boundary"
                        ]
                    )
                    self.last_error = (
                        f"`{record.frontier_id}` was marked completed from the current pose; "
                        f"no router-approved approach path was found nearby ({rejection_reason or 'unknown reason'})."
                    )
                    return self._complete_active_frontier_locked(
                        record=record,
                        decision=decision,
                        event_type="router_close_enough_completed",
                    )
                self.frontier_memory.fail(
                    record.frontier_id,
                    f"no router-approved path near frontier: {rejection_reason or 'unknown reason'}",
                )
                self.last_error = (
                    f"Router move rejected `{record.frontier_id}` because no nearby target pose "
                    f"could be found ({rejection_reason or 'unknown reason'})."
                )
                self._log_decision("router_target_invalid")
                self._prepare_decision_locked()
                return self.snapshot()

            try:
                travelled_distance_m = self._execute_router_path(path_poses, final_pose=target_pose)
            except HumanAssistanceRequired:
                return self.snapshot()
            except Exception as exc:
                self.frontier_memory.fail(record.frontier_id, f"router movement failed: {exc}")
                self.last_error = f"Router movement failed for `{record.frontier_id}`: {exc}"
                self._log_decision("router_move_failed")
                self._prepare_decision_locked()
                return self.snapshot()

            reached_pose = self._current_pose()
            self.total_distance_m += travelled_distance_m
            self.trajectory.append(reached_pose.to_dict())
            try:
                self._perform_scan(
                    full_turnaround=self._scan_uses_turnaround(),
                    capture_frame=True,
                    reason=self._scan_reason(f"router_arrive_frontier::{record.frontier_id}"),
                )
            except Exception as exc:
                self.guardrail_events.append(
                    {
                        "type": "arrival_scan_failed_after_router_move",
                        "frontier_id": record.frontier_id,
                        "reason": str(exc),
                    }
                )
                record.evidence = _dedupe_text(
                    record.evidence
                    + [f"arrival scan failed after router-approved movement; frontier still marked completed: {exc}"]
                )
                self.last_error = f"Arrival scan failed for `{record.frontier_id}`, but the frontier was reached: {exc}"
            return self._complete_active_frontier_locked(
                record=record,
                decision=decision,
                event_type="router_path_applied",
            )

    def _validate_router_path(
        self,
        path_poses: list[Pose2D],
        *,
        start_pose: Pose2D,
        target_pose: Pose2D,
    ) -> None:
        if not path_poses:
            raise RuntimeError("Nav2 returned an empty path.")
        start_error_m = _pose_distance_m(path_poses[0], start_pose)
        end_error_m = _pose_distance_m(path_poses[-1], target_pose)
        tolerance_m = max(self.config.occupancy_resolution * 2.0, 0.50)
        if start_error_m > tolerance_m:
            raise RuntimeError(
                "Nav2 path starts too far from the current simulator pose "
                f"({start_error_m:.2f} m). The router likely planned from stale pose state."
            )
        if end_error_m > tolerance_m:
            raise RuntimeError(
                "Nav2 path ends too far from the requested frontier target "
                f"({end_error_m:.2f} m)."
            )
        crossing = self._first_known_occupied_path_crossing(
            [start_pose] + list(path_poses) + [target_pose],
        )
        if crossing is not None:
            raise RuntimeError(
                "Nav2 path crosses a known occupied map cell "
                f"at ({crossing['x']:.2f}, {crossing['y']:.2f}) "
                f"cell=({crossing['cell_x']},{crossing['cell_y']})."
            )

    def _first_known_occupied_path_crossing(self, poses: list[Pose2D]) -> dict[str, Any] | None:
        if len(poses) < 2:
            return None
        resolution = max(float(self.config.occupancy_resolution), 1e-6)
        effective = self._effective_known_cells()
        
        # Check for static occupied cells (these block the path)
        for segment_index, (start, end) in enumerate(zip(poses, poses[1:])):
            dx = end.x - start.x
            dy = end.y - start.y
            distance_m = math.hypot(dx, dy)
            sample_count = max(int(math.ceil(distance_m / max(resolution * 0.5, 0.05))), 1)
            for sample_index in range(sample_count + 1):
                fraction = sample_index / sample_count
                x = start.x + dx * fraction
                y = start.y + dy * fraction
                cell = self._world_to_cell(x, y)
                if effective.get(cell) == "occupied":
                    return {
                        "segment_index": segment_index,
                        "sample_index": sample_index,
                        "x": x,
                        "y": y,
                        "cell_x": cell.x,
                        "cell_y": cell.y,
                    }
        return None

    def _navigation_planning_cell_state(self, cell: GridCell) -> str | None:
        if cell in getattr(self, "_transient_navigation_obstacle_cells", set()):
            return "occupied"
        return self._effective_known_cells().get(cell)

    def _plan_router_path_to_frontier(
        self,
        *,
        record: FrontierRecord,
        start_pose: Pose2D,
        reachable_safe_cells: set[GridCell],
    ) -> tuple[Pose2D | None, list[Pose2D], str | None]:
        candidates = self._candidate_target_poses(
            record=record,
            start_pose=start_pose,
            reachable_safe_cells=reachable_safe_cells,
        )
        if not candidates:
            return None, [], "no safe target pose candidates were available near the frontier"

        rejection_reasons: list[str] = []
        for index, candidate in enumerate(candidates, start=1):
            try:
                self._publish_router_state(required=True)
                status, path_poses, status_label = self.router.compute_path(
                    goal_pose=candidate,
                    planner_id=self.config.nav2_planner_id,
                )
                if status_label != "succeeded" or not path_poses:
                    rejection_reasons.append(
                        f"candidate {index} Nav2 returned `{status_label}` with {len(path_poses)} poses"
                    )
                    continue
                self._validate_router_path(path_poses, start_pose=start_pose, target_pose=candidate)
                if index > 1:
                    self.guardrail_events.append(
                        {
                            "type": "router_frontier_target_candidate_recovered",
                            "frontier_id": record.frontier_id,
                            "candidate_index": index,
                            "rejected_candidates": rejection_reasons[-5:],
                        }
                    )
                return candidate, path_poses, None
            except Exception as exc:
                rejection_reasons.append(f"candidate {index} rejected: {exc}")

        return None, [], "; ".join(rejection_reasons[-5:]) if rejection_reasons else "all candidates were rejected"

    def _execute_router_path(self, path_poses: list[Pose2D], *, final_pose: Pose2D) -> float:
        travelled_distance_m = 0.0
        active_path = list(path_poses)
        try:
            travelled_distance_m += self._execute_router_path_once(active_path, final_pose=final_pose)
            self._transient_navigation_obstacle_cells.clear()
            self._publish_router_state()
            return travelled_distance_m
        except Nav2LocalPathBlocked as exc:
            travelled_distance_m += exc.travelled_distance_m
            self._stop_robot_action()
            try:
                self._publish_router_state()
            except Exception:
                pass
            self.paused = True
            self.status = "human_assistance_required"
            self.last_error = (
                "Human assistance required: autonomous navigation stopped because the local RGB-D guard "
                f"detected a collision risk ({exc}). Take over the robot, scan with S if keyboard controls "
                "are enabled, mark the frontier solved, then resume exploration."
            )
            self.guardrail_events.append(
                {
                    "type": "human_assistance_requested",
                    "reason": "nav2_local_path_blocked",
                    "blocker": exc.blocker,
                    "travelled_distance_m": round(travelled_distance_m, 3),
                }
            )
            raise HumanAssistanceRequired(self.last_error) from exc

    def _execute_router_path_once(self, path_poses: list[Pose2D], *, final_pose: Pose2D) -> float:
        self._raise_if_manual_stop_requested()
        if not path_poses:
            self._teleport_robot(final_pose)
            self.control_steps += 1
            return 0.0
        waypoints = [self._current_pose()] + list(path_poses)
        if _pose_distance_m(waypoints[-1], final_pose) > max(self.config.occupancy_resolution, 0.10):
            waypoints.append(final_pose)
        else:
            waypoints[-1] = final_pose

        travelled_distance_m = 0.0
        pose_publish_stride = 5
        for waypoint in waypoints[1:]:
            self._raise_if_manual_stop_requested()
            current = self._current_pose()
            dx = waypoint.x - current.x
            dy = waypoint.y - current.y
            distance = math.hypot(dx, dy)
            if distance > 1e-4:
                segment_yaw = math.atan2(dy, dx)
                self._rotate_in_place(segment_yaw, publish_stride=pose_publish_stride)
                try:
                    travelled_distance_m += self._drive_straight_to(
                        Pose2D(waypoint.x, waypoint.y, segment_yaw),
                        publish_stride=pose_publish_stride,
                    )
                except Nav2LocalPathBlocked as exc:
                    exc.travelled_distance_m += travelled_distance_m
                    raise
        self._rotate_in_place(final_pose.yaw, publish_stride=pose_publish_stride)
        return travelled_distance_m

    def _raise_if_manual_stop_requested(self) -> None:
        if not getattr(self, "paused", False):
            return
        self._stop_robot_action()
        raise HumanAssistanceRequired(
            self.last_error or "Autonomous navigation is paused for operator control."
        )

    def _stop_robot_action(self) -> None:
        action = getattr(self, "action", None)
        if action is None:
            return
        try:
            action[...] = 0.0
        except Exception:
            try:
                self.action = np.zeros_like(action)
            except Exception:
                pass

    def _rotate_in_place(self, target_yaw: float, *, publish_stride: int) -> None:
        self._raise_if_manual_stop_requested()
        current = self._current_pose()
        yaw_delta = _angle_wrap(target_yaw - current.yaw)
        if abs(yaw_delta) <= 1e-3:
            return
        angular_speed = self._sim_motion_angular_speed_rad_s()
        publish_hz = max(float(self.config.ros_manual_spin_publish_hz), 1.0)
        step_s = 1.0 / publish_hz
        step_yaw = angular_speed * step_s
        steps = max(int(math.ceil(abs(yaw_delta) / max(step_yaw, 1e-6))), 1)
        for step in range(1, steps + 1):
            self._raise_if_manual_stop_requested()
            fraction = step / steps
            pose = Pose2D(current.x, current.y, _angle_wrap(current.yaw + yaw_delta * fraction))
            self._teleport_robot(pose)
            self.control_steps += 1
            if step % publish_stride == 0 or step == steps:
                self._publish_navigation_only_observation()
            self._sleep_after_motion_step(step_s)

    def _drive_straight_to(self, target_pose: Pose2D, *, publish_stride: int) -> float:
        self._raise_if_manual_stop_requested()
        current = self._current_pose()
        dx = target_pose.x - current.x
        dy = target_pose.y - current.y
        distance = math.hypot(dx, dy)
        if distance <= 1e-4:
            return 0.0
        linear_speed_m_s = self._sim_motion_linear_speed_m_s()
        publish_hz = max(float(self.config.ros_manual_spin_publish_hz), 1.0)
        step_s = 1.0 / publish_hz
        step_distance = max(linear_speed_m_s * step_s, 0.01)
        steps = max(int(math.ceil(distance / step_distance)), 1)
        travelled_distance_m = 0.0
        previous = current
        for step in range(1, steps + 1):
            self._raise_if_manual_stop_requested()
            observation = self._publish_navigation_only_observation()
            self._raise_if_manual_stop_requested()
            blocker = self._local_path_blocker_from_observation(observation, target_pose=target_pose)
            if blocker is not None:
                print(f"NAV2 COLLISION: {blocker.get('reason', 'unknown')}")
                print(f"  obstacle_point=({blocker.get('point', {}).get('x', 0):.3f}, {blocker.get('point', {}).get('y', 0):.3f})")
                print(f"  forward_distance={blocker.get('forward_distance_m', 0):.3f}m, lateral={blocker.get('lateral_distance_m', 0):.3f}m")
                raise Nav2LocalPathBlocked(blocker, travelled_distance_m=travelled_distance_m)
            fraction = step / steps
            pose = Pose2D(
                current.x + dx * fraction,
                current.y + dy * fraction,
                target_pose.yaw,
            )
            self._teleport_robot(pose)
            self.control_steps += 1
            travelled_distance_m += _pose_distance_m(previous, pose)
            previous = pose
            if step % publish_stride == 0 or step == steps:
                self.trajectory.append(pose.to_dict())
            self._sleep_after_motion_step(step_s)
        return travelled_distance_m

    def _sleep_after_motion_step(self, step_s: float) -> None:
        if self.options.render_mode == "human":
            time.sleep(max(step_s, 0.0))
        elif self.config.realtime_sleep_s > 0:
            time.sleep(self.config.realtime_sleep_s)

    def _sim_motion_speed_preset(self) -> dict[str, float]:
        return SIM_MOTION_SPEED_PRESETS.get(self.config.sim_motion_speed, SIM_MOTION_SPEED_PRESETS["normal"])

    def _sim_motion_linear_speed_m_s(self) -> float:
        return max(float(self._sim_motion_speed_preset()["linear_m_s"]), 0.05)

    def _sim_motion_angular_speed_rad_s(self) -> float:
        multiplier = float(self._sim_motion_speed_preset()["angular_multiplier"])
        return max(float(self.config.ros_manual_spin_angular_speed_rad_s) * multiplier, 0.05)

    def _perform_scan(self, *, full_turnaround: bool, capture_frame: bool, reason: str) -> None:
        super()._perform_scan(full_turnaround=full_turnaround, capture_frame=capture_frame, reason=reason)
        self._publish_router_state()

    def _publish_navigation_only_observation(self) -> dict[str, Any] | None:
        try:
            head_data = self._capture_head_camera()
            observation, _depth_for_scan = self._build_scan_observation_from_head_data(head_data)
            self._latest_scan_observation = observation
            self.router.update_state(
                occupancy_map=None,
                pose=self._current_pose(),
                scan_observation=observation,
                image_data_url=None,
            )
            return observation
        except Exception as exc:
            self.last_error = f"Nav2 local observation update failed: {exc}"
            self.guardrail_events.append(
                {
                    "type": "nav2_local_observation_update_failed",
                    "reason": str(exc),
                }
            )
            return None

    def _local_path_blocker_from_observation(
        self,
        observation: dict[str, Any] | None,
        *,
        target_pose: Pose2D,
    ) -> dict[str, Any] | None:
        sensor_pose = observation.get("pose") if observation else None
        if sensor_pose is None:
            return None
        blocker = _local_scan_path_blocker(
            observation,
            current_pose=sensor_pose,
            target_pose=target_pose,
            robot_length_m=XLEROBOT_IKEA_CART_FOOTPRINT_LENGTH_M,
            robot_width_m=XLEROBOT_IKEA_CART_FOOTPRINT_WIDTH_M,
        )
        return blocker

    def _remember_transient_navigation_obstacle(self, blocker: dict[str, Any]) -> None:
        point = blocker.get("point")
        if not isinstance(point, dict):
            return
        try:
            center = self._world_to_cell(float(point["x"]), float(point["y"]))
        except Exception:
            return
        cells = getattr(self, "_transient_navigation_obstacle_cells", set())
        cells.add(center)
        self._transient_navigation_obstacle_cells = cells

    def _move_to_scan_sample_pose(
        self,
        pose: Pose2D,
        *,
        full_turnaround: bool,
        scan_reason: str,
        yaw_sample_index: int,
    ) -> None:
        if not full_turnaround:
            self._teleport_robot(pose)
            return
        current = self._current_pose()
        yaw_delta = _angle_wrap(pose.yaw - current.yaw)
        angular_speed = self._sim_motion_angular_speed_rad_s()
        publish_hz = max(float(self.config.ros_manual_spin_publish_hz), 1.0)
        step_s = 1.0 / publish_hz
        step_yaw = angular_speed * step_s
        steps = max(int(math.ceil(abs(yaw_delta) / max(step_yaw, 1e-6))), 1)
        for step in range(1, steps + 1):
            fraction = step / steps
            intermediate = Pose2D(
                pose.x,
                pose.y,
                _angle_wrap(current.yaw + yaw_delta * fraction),
            )
            self._teleport_robot(intermediate)
            if step < steps:
                time.sleep(step_s)

    def _publish_router_state(self, *, required: bool = False) -> None:
        nav_map = self._router_navigation_map()
        try:
            self.router.update_state(
                occupancy_map=nav_map,
                pose=self._current_pose(),
                scan_observation=getattr(self, "_latest_scan_observation", None),
                image_data_url=None,
            )
        except Exception as exc:
            self.last_error = f"Nav2 router state update failed: {exc}"
            self.guardrail_events.append(
                {
                    "type": "nav2_router_state_update_failed",
                    "reason": str(exc),
                }
            )
            if required:
                raise

    def _router_navigation_map(self) -> RosOccupancyMap:
        effective_known = self._effective_known_cells()
        transient_obstacles = set(getattr(self, "_transient_navigation_obstacle_cells", set()))
        resolution = float(self.config.occupancy_resolution)
        map_cells = set(effective_known) | transient_obstacles
        if not map_cells:
            pose = self._current_pose()
            base_x = int(math.floor(pose.x / resolution))
            base_y = int(math.floor(pose.y / resolution))
            return RosOccupancyMap(
                resolution=resolution,
                width=1,
                height=1,
                origin_x=base_x * resolution,
                origin_y=base_y * resolution,
                data=(0,),
            )
        min_x = min(cell.x for cell in map_cells) - 2
        min_y = min(cell.y for cell in map_cells) - 2
        max_x = max(cell.x for cell in map_cells) + 2
        max_y = max(cell.y for cell in map_cells) + 2
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        data = [-1] * (width * height)
        for cell, state in effective_known.items():
            local_x = cell.x - min_x
            local_y = cell.y - min_y
            data[local_y * width + local_x] = 0 if state == "free" else 100
        for cell in transient_obstacles:
            local_x = cell.x - min_x
            local_y = cell.y - min_y
            if 0 <= local_x < width and 0 <= local_y < height:
                data[local_y * width + local_x] = 100
        return RosOccupancyMap(
            resolution=resolution,
            width=width,
            height=height,
            origin_x=min_x * resolution,
            origin_y=min_y * resolution,
            data=tuple(data),
        )

    def _build_map_payload(self) -> dict[str, Any]:
        payload = super()._build_map_payload()
        payload["mode"] = "interactive_manishkill_rgbd_nav2_router"
        payload["summary"] = (
            "Robot exploration mode using ManiSkill as the robot brain, with fused RGB-D mapping kept local "
            "and Nav2 path approval routed through the external ROS adapter."
        )
        artifacts = payload.setdefault("artifacts", {})
        artifacts["navigation"] = {
            "mode": "maniskill_brain_with_nav2_router",
            "control_steps": self.control_steps,
            "total_distance_m": round(self.total_distance_m, 3),
            "router_url": self.config.ros_adapter_url,
            "sim_motion_speed": self.config.sim_motion_speed,
        }
        try:
            artifacts["nav2_router"] = self.router.snapshot()
        except Exception as exc:
            artifacts["nav2_router"] = {"status": "unavailable", "error": str(exc)}
        artifacts["planned_nav_path"] = getattr(self, "_planned_nav_path", [])
        return payload


def _depth_image_to_millimeters(depth: np.ndarray) -> np.ndarray:
    array = np.asarray(depth)
    valid = array[np.isfinite(array) & (array > 0)]
    if valid.size and float(np.nanmedian(valid)) < 50.0:
        array = array.astype(np.float32) * 1000.0
    array = np.nan_to_num(array, nan=0.0, posinf=float(np.iinfo(np.uint16).max), neginf=0.0)
    return np.asarray(np.clip(array, 0, np.iinfo(np.uint16).max), dtype=np.uint16)


def _updated_mobile_base_qpos(
    current_qpos: Any,
    pose: Pose2D,
    *,
    anchor_pose: Pose2D,
    anchor_qpos: Any,
) -> Any:
    updated = current_qpos.clone() if hasattr(current_qpos, "clone") else np.array(current_qpos, copy=True)
    anchor_qpos_array = np.asarray(anchor_qpos, dtype=np.float64)
    dx_world = float(pose.x) - float(anchor_pose.x)
    dy_world = float(pose.y) - float(anchor_pose.y)
    cos_yaw = math.cos(float(anchor_pose.yaw))
    sin_yaw = math.sin(float(anchor_pose.yaw))
    dx_local = cos_yaw * dx_world + sin_yaw * dy_world
    dy_local = -sin_yaw * dx_world + cos_yaw * dy_world
    updated[..., 0] = float(anchor_qpos_array[..., 0] + dx_local)
    updated[..., 1] = float(anchor_qpos_array[..., 1] + dy_local)
    updated[..., 2] = float(anchor_qpos_array[..., 2] + _angle_wrap(float(pose.yaw) - float(anchor_pose.yaw)))
    return updated


def _zero_mobile_base_qvel(current_qvel: Any) -> Any:
    updated = current_qvel.clone() if hasattr(current_qvel, "clone") else np.array(current_qvel, copy=True)
    updated[..., 0] = 0.0
    updated[..., 1] = 0.0
    updated[..., 2] = 0.0
    return updated


def _cell_mean_pose(cells: list[GridCell], resolution: float, *, yaw: float = 0.0) -> Pose2D:
    if not cells:
        return Pose2D(0.0, 0.0, yaw)
    return Pose2D(
        sum((cell.x + 0.5) * resolution for cell in cells) / len(cells),
        sum((cell.y + 0.5) * resolution for cell in cells) / len(cells),
        yaw,
    )


def _spawn_facing_yaw_offset(spawn_facing: str) -> float:
    offsets = {
        "front": 0.0,
        "left": -math.pi / 2.0,
        "right": math.pi / 2.0,
        "back": math.pi,
    }
    return offsets[spawn_facing]


def _resolve_manishkill_start_pose(
    current_pose: Pose2D,
    *,
    spawn_x: float | None,
    spawn_y: float | None,
    spawn_yaw: float,
    spawn_facing: str,
) -> Pose2D | None:
    if spawn_x is not None and spawn_y is not None:
        return Pose2D(float(spawn_x), float(spawn_y), float(spawn_yaw))
    yaw_offset = _spawn_facing_yaw_offset(spawn_facing) + float(spawn_yaw)
    if abs(yaw_offset) <= 1e-9:
        return None
    return Pose2D(current_pose.x, current_pose.y, _angle_wrap(current_pose.yaw + yaw_offset))


def _angle_wrap(angle: float) -> float:
    wrapped = (angle + math.pi) % (2.0 * math.pi) - math.pi
    if wrapped == -math.pi:
        return math.pi
    return wrapped


def _depth_summary(depth_mm: np.ndarray, *, max_range_m: float) -> dict[str, Any]:
    depth_m = depth_mm.astype(np.float32) / 1000.0
    valid = depth_m[np.isfinite(depth_m) & (depth_m > 0.0) & (depth_m <= max_range_m)]
    if valid.size == 0:
        return {
            "valid_depth_fraction": 0.0,
            "depth_min_m": None,
            "depth_p25_m": None,
            "depth_median_m": None,
            "depth_p75_m": None,
            "depth_max_m": None,
        }
    return {
        "valid_depth_fraction": round(float(valid.size) / max(depth_m.size, 1), 3),
        "depth_min_m": round(float(np.nanmin(valid)), 3),
        "depth_p25_m": round(float(np.nanpercentile(valid, 25)), 3),
        "depth_median_m": round(float(np.nanmedian(valid)), 3),
        "depth_p75_m": round(float(np.nanpercentile(valid, 75)), 3),
        "depth_max_m": round(float(np.nanmax(valid)), 3),
    }


def _rgb_array_to_data_url(rgb: Any) -> str:
    if rgb is None:
        return ""
    try:
        from PIL import Image as PILImage
    except Exception:
        return ""
    array = np.asarray(rgb)
    if array.ndim == 4:
        array = np.squeeze(array, axis=0)
    if array.ndim != 3:
        return ""
    if array.shape[-1] > 3:
        array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array, 0.0, 1.0) * 255.0
    array = np.asarray(np.clip(array, 0, 255), dtype=np.uint8)
    image = PILImage.fromarray(array, mode="RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _frontier_memory_prompt_context(memory: FrontierMemory) -> dict[str, Any]:
    records = sorted(memory.records.values(), key=lambda record: record.frontier_id)
    return {
        "active_frontier_id": memory.active_frontier_id,
        "frontier_status": [
            {
                "frontier_id": record.frontier_id,
                "status": record.status,
                "discovered_step": record.discovered_step,
                "last_seen_step": record.last_seen_step,
                "attempt_count": record.attempt_count,
                "visit_count": record.visit_count,
                "source": "current_rgbd_scan" if record.currently_visible else "frontier_memory",
                "llm_memory_priority": record.llm_memory_priority,
            }
            for record in records
        ],
        "return_waypoints": list(memory.return_waypoints.values()),
        "note": (
            "Status-only memory context. `source=frontier_memory` means the frontier was discovered during previous "
            "scans. Choose only from Frontier Information."
        ),
    }


def _navigation_map_data_url(
    *,
    known_cells: dict[GridCell, str],
    resolution: float,
    trajectory: list[dict[str, Any]],
    robot_pose: Pose2D,
    candidate_records: list[FrontierRecord],
    remembered_records: list[FrontierRecord],
) -> str:
    try:
        from PIL import Image as PILImage
        from PIL import ImageDraw
    except Exception:
        return ""
    width_px = 1000
    height_px = 760
    pad_px = 36
    poses: list[Pose2D] = [robot_pose]
    for item in trajectory:
        if isinstance(item, dict):
            poses.append(Pose2D(float(item.get("x", robot_pose.x)), float(item.get("y", robot_pose.y)), float(item.get("yaw", 0.0))))
    for record in candidate_records + remembered_records:
        poses.extend([record.nav_pose, record.centroid_pose])
    if known_cells:
        min_x = min(cell.x for cell in known_cells) * resolution
        max_x = (max(cell.x for cell in known_cells) + 1) * resolution
        min_y = min(cell.y for cell in known_cells) * resolution
        max_y = (max(cell.y for cell in known_cells) + 1) * resolution
    else:
        min_x = robot_pose.x - 1.0
        max_x = robot_pose.x + 1.0
        min_y = robot_pose.y - 1.0
        max_y = robot_pose.y + 1.0
    if poses:
        min_x = min(min_x, *(pose.x for pose in poses)) - 0.35
        max_x = max(max_x, *(pose.x for pose in poses)) + 0.35
        min_y = min(min_y, *(pose.y for pose in poses)) - 0.35
        max_y = max(max_y, *(pose.y for pose in poses)) + 0.35
    world_w = max(max_x - min_x, resolution)
    world_h = max(max_y - min_y, resolution)

    def project(x: float, y: float) -> tuple[float, float]:
        px = pad_px + ((x - min_x) / world_w) * (width_px - pad_px * 2)
        py = height_px - pad_px - ((y - min_y) / world_h) * (height_px - pad_px * 2)
        return px, py

    image = PILImage.new("RGB", (width_px, height_px), (250, 250, 247))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width_px - 1, height_px - 1), fill=(250, 250, 247), outline=(218, 224, 211), width=2)
    for cell, state in sorted(known_cells.items()):
        x0, y1 = project(cell.x * resolution, cell.y * resolution)
        x1, y0 = project((cell.x + 1) * resolution, (cell.y + 1) * resolution)
        fill = (115, 125, 112) if state == "occupied" else (225, 232, 219)
        draw.rectangle((x0, y0, x1, y1), fill=fill)

    trajectory_points = [
        project(float(item.get("x", robot_pose.x)), float(item.get("y", robot_pose.y)))
        for item in trajectory
        if isinstance(item, dict)
    ]
    if len(trajectory_points) >= 2:
        draw.line(trajectory_points, fill=(79, 119, 45), width=5)

    def draw_frontier(record: FrontierRecord, *, color: tuple[int, int, int], memory: bool) -> None:
        bx, by = project(record.centroid_pose.x, record.centroid_pose.y)
        nx, ny = project(record.nav_pose.x, record.nav_pose.y)
        draw.line((bx, by, nx, ny), fill=color, width=2)
        draw.ellipse((bx - 5, by - 5, bx + 5, by + 5), outline=color, width=2)
        radius = 5 if memory else 8
        draw.ellipse((nx - radius, ny - radius, nx + radius, ny + radius), fill=color)
        label = record.frontier_id
        if record.path_cost_m is not None:
            label = f"{label} {record.path_cost_m:.1f}m"
        if memory:
            label = f"{label} memory"
        draw.text((nx + 10, ny - 12), label, fill=color)

    for record in remembered_records:
        draw_frontier(record, color=(100, 112, 103), memory=True)
    for record in candidate_records:
        draw_frontier(record, color=(49, 87, 44), memory=False)

    rx, ry = project(robot_pose.x, robot_pose.y)
    hx, hy = project(
        robot_pose.x + math.cos(robot_pose.yaw) * max(resolution * 2.5, 0.45),
        robot_pose.y + math.sin(robot_pose.yaw) * max(resolution * 2.5, 0.45),
    )
    draw.ellipse((rx - 13, ry - 13, rx + 13, ry + 13), fill=(165, 40, 32))
    draw.line((rx, ry, hx, hy), fill=(109, 15, 10), width=5)
    draw.ellipse((hx - 4, hy - 4, hx + 4, hy + 4), fill=(109, 15, 10))
    draw.text((rx + 15, ry - 16), "robot", fill=(165, 40, 32))
    draw.text((18, 16), "Navigation map: free, occupied, robot, trajectory, frontier labels", fill=(24, 35, 15))

    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


INTERACTIVE_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Robot Exploration Mode</title>
  <style>
    :root {
      --bg: #eef2e6;
      --panel: rgba(255,255,255,0.82);
      --ink: #18230f;
      --muted: #5d6b52;
      --line: rgba(24,35,15,0.14);
      --green: #31572c;
      --leaf: #4f772d;
      --gold: #b0891f;
      --red: #a52820;
      --shadow: 0 24px 54px rgba(24,35,15,0.13);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Aptos", "Segoe UI", sans-serif;
      color: var(--ink);
      min-height: 100vh;
      background:
        radial-gradient(circle at 12% 6%, rgba(79,119,45,0.24), transparent 28%),
        radial-gradient(circle at 85% 16%, rgba(176,137,31,0.22), transparent 24%),
        linear-gradient(145deg, #f6f1db 0%, #e4ecd2 50%, #d9e7da 100%);
    }
    .shell { max-width: 1760px; margin: 0 auto; padding: 24px; }
    header { display: flex; justify-content: space-between; gap: 18px; align-items: end; margin-bottom: 18px; }
    h1 { margin: 0; font-family: Georgia, serif; font-size: clamp(32px, 5vw, 62px); line-height: .9; max-width: 11ch; }
    .subtitle { color: var(--muted); max-width: 760px; }
    .layout { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 18px; align-items: start; }
    .layout > * { min-width: 0; }
    .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 24px; padding: 16px; box-shadow: var(--shadow); backdrop-filter: blur(18px); overflow: hidden; }
    .stack { display: grid; gap: 16px; }
    .eyebrow { font-size: 12px; text-transform: uppercase; letter-spacing: .14em; color: var(--muted); margin-bottom: 10px; }
    .stats { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
    .stat { border: 1px solid var(--line); border-radius: 16px; background: rgba(255,255,255,.66); padding: 11px; }
    .key { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .08em; }
    .value { font-weight: 750; margin-top: 4px; }
    button { border: 0; border-radius: 999px; padding: 11px 14px; font-weight: 750; cursor: pointer; }
    button.primary { color: white; background: var(--green); }
    button.secondary { color: var(--green); background: #f5f0d6; border: 1px solid rgba(79,119,45,.22); }
    button.danger { color: var(--red); background: #fee8e1; }
    .buttons { display: flex; flex-wrap: wrap; gap: 9px; }
    #map { width: 100%; height: 600px; border-radius: 20px; border: 1px solid var(--line); background: rgba(255,255,255,.9); }
    pre, textarea { width: 100%; border: 1px solid var(--line); border-radius: 16px; background: rgba(255,255,255,.72); color: #12210f; padding: 12px; overflow: auto; }
    pre { white-space: pre-wrap; max-height: 360px; margin: 0; }
    textarea { min-height: 320px; resize: vertical; font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 12px; }
    .frontier-list { display: grid; gap: 8px; max-height: 420px; overflow: auto; }
    .frontier { border: 1px solid var(--line); border-radius: 15px; padding: 10px; background: rgba(255,255,255,.66); }
    .frontier.pending { border-color: var(--gold); background: rgba(176,137,31,.12); }
    .frontier.suppressed { opacity: .55; }
    .thumbs { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; }
    .thumbs img { width: 100%; border-radius: 14px; border: 1px solid var(--line); background: white; }
    .muted { color: var(--muted); }
    .error { color: var(--red); font-weight: 750; }
    @media (max-width: 980px) { .layout { grid-template-columns: 1fr; } #map { height: 560px; } }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div>
        <h1>Robot Exploration Mode</h1>
      </div>
    </header>
    <div class="layout">
      <div class="stack left-column">
        <section class="panel">
          <div class="eyebrow">Controls</div>
          <div class="buttons">
            <button class="secondary" id="reset">Reset + Scan</button>
            <button class="secondary" id="pause">Pause</button>
            <button class="secondary" id="resume">Resume</button>
            <button class="primary" id="call-semantic" style="display:none;">Call Semantic LLM</button>
            <button class="primary" id="control-robot">Control Robot</button>
          </div>
          <div class="buttons" style="margin-top:10px;">
            <button class="secondary" id="edit-block">Draw Wall</button>
            <button class="secondary" id="edit-clear">Erase Wall</button>
            <button class="secondary" id="edit-reset">Reset Cell</button>
            <button class="secondary" id="region-add">Add Region</button>
            <button class="primary" id="region-done">Done Region</button>
            <button class="secondary" id="subwaypoint-add">Add Subwaypoint</button>
          </div>
          <div id="error" class="error"></div>
        </section>
        <section class="panel">
          <div class="eyebrow">State</div>
          <div class="stats" id="stats"></div>
        </section>
        <section class="panel">
          <div class="eyebrow">Frontier Information</div>
          <div id="frontiers" class="frontier-list"></div>
        </section>
        <section class="panel">
          <div class="eyebrow">Navigation Memory</div>
          <div id="manual-regions" class="frontier-list"></div>
        </section>
        <section class="panel" id="semantic-panel" style="display:none;">
          <div class="eyebrow">Automatic Semantic Places</div>
          <div id="semantic" class="frontier-list"></div>
        </section>
      </div>
      <div class="stack right-column">
        <section class="panel">
          <div class="eyebrow">Scanned 2D Map</div>
          <svg id="map" viewBox="0 0 1000 760"></svg>
        </section>
      </div>
    </div>
  </div>
  <script>
    let state = null;
    let mapEditMode = 'block';
    let isPaintingMap = false;
    let pendingPaintCells = new Map();
    let paintFlushTimer = null;
    let lastPaintedCellKey = null;
    let currentOccupancyCellStates = new Map();
    let regionMode = null;
    let selectedRegionCells = new Map();
    let pendingSubwaypoint = null;
    let draggingWaypoint = null;

    function esc(v) {
      return String(v ?? '').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
    }
    async function post(url, payload) {
      const res = await fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload || {})});
      if (!res.ok) throw new Error(await res.text());
      state = await res.json();
      render();
    }
    async function refresh() {
      const res = await fetch('/api/state');
      state = await res.json();
      render();
    }
    function bounds(map) {
      return map.occupancy.bounds || {min_x:0,max_x:10,min_y:0,max_y:8};
    }
    function projector(b) {
      const pad = 32;
      const w = Math.max(b.max_x - b.min_x, 1);
      const h = Math.max(b.max_y - b.min_y, 1);
      return (p) => ({
        x: pad + ((p.x - b.min_x) / w) * (1000 - pad * 2),
        y: 760 - pad - ((p.y - b.min_y) / h) * (760 - pad * 2)
      });
    }
    function svgPointFromClient(svg, clientX, clientY) {
      const point = svg.createSVGPoint();
      point.x = clientX;
      point.y = clientY;
      const matrix = svg.getScreenCTM();
      if (!matrix) return {x: 0, y: 0};
      return point.matrixTransform(matrix.inverse());
    }
    function worldFromSvgViewPoint(b, svgX, svgY) {
      const pad = 32;
      const w = Math.max(b.max_x - b.min_x, 1);
      const h = Math.max(b.max_y - b.min_y, 1);
      const normalizedX = Math.min(Math.max((svgX - pad) / Math.max(1000 - pad * 2, 1), 0), 1);
      const normalizedY = Math.min(Math.max((svgY - pad) / Math.max(760 - pad * 2, 1), 0), 1);
      return {
        x: b.min_x + normalizedX * w,
        y: b.max_y - normalizedY * h,
      };
    }
    function cellFromMapEvent(map, event) {
      const svg = document.getElementById('map');
      const b = bounds(map);
      const point = svgPointFromClient(svg, event.clientX, event.clientY);
      const world = worldFromSvgViewPoint(b, point.x, point.y);
      const resolution = map.occupancy?.resolution || 0.25;
      const cell = {
        cell_x: Math.floor(world.x / resolution),
        cell_y: Math.floor(world.y / resolution),
      };
      cell.key = `${cell.cell_x}:${cell.cell_y}`;
      return cell;
    }
    function cellCenterPose(map, cell) {
      const resolution = map.occupancy?.resolution || 0.25;
      return {
        x: (cell.cell_x + 0.5) * resolution,
        y: (cell.cell_y + 0.5) * resolution,
        yaw: 0
      };
    }
    function selectRegionCell(cell) {
      const state = currentOccupancyCellStates.get(cell.key);
      if (!state || state.state !== 'free') return;
      selectedRegionCells.set(cell.key, {cell_x: cell.cell_x, cell_y: cell.cell_y});
      renderMap();
    }
    function shouldPaintCell(cell) {
      if (mapEditMode !== 'clear') return true;
      const state = currentOccupancyCellStates.get(cell.key);
      return !!state && (state.state === 'occupied' || state.manual_override === 'blocked');
    }
    function enqueuePaintCell(cell) {
      if (!shouldPaintCell(cell)) return;
      pendingPaintCells.set(cell.key, {cell_x: cell.cell_x, cell_y: cell.cell_y});
      if (!paintFlushTimer) {
        paintFlushTimer = setTimeout(flushPaintCells, 80);
      }
    }
    async function flushPaintCells() {
      if (paintFlushTimer) {
        clearTimeout(paintFlushTimer);
        paintFlushTimer = null;
      }
      const cells = Array.from(pendingPaintCells.values());
      pendingPaintCells.clear();
      if (!cells.length) return;
      await post('/api/map/edit', {mode: mapEditMode, cells});
    }
    function renderStats() {
      const pose = state.robot_pose || {};
      const items = [
        ['Status', state.status],
        ['Coverage', state.coverage],
        ['Pose', `${Number(pose.x || 0).toFixed(2)}, ${Number(pose.y || 0).toFixed(2)}`],
        ['Frontiers', (state.candidate_frontiers || []).length],
        ['Stored Memory', (state.map?.remembered_frontiers || []).length],
        ['Manual Regions', (state.map?.regions || []).length],
        ['Pending', state.pending_target?.frontier_id || 'none'],
        ['Provider', state.map?.artifacts?.llm_policy?.provider || 'unknown'],
        ['Edit Mode', mapEditMode]
      ];
      document.getElementById('stats').innerHTML = items.map(([k,v]) => `<div class="stat"><div class="key">${esc(k)}</div><div class="value">${esc(v)}</div></div>`).join('');
      document.getElementById('error').textContent = state.last_error || '';
    }
    function renderFrontiers() {
      const pending = state.pending_target?.frontier_id;
      document.getElementById('frontiers').innerHTML = (state.candidate_frontiers || []).map((f) => `
        <div class="frontier ${f.frontier_id === pending ? 'pending' : ''} ${f.status === 'suppressed' ? 'suppressed' : ''}">
          <strong>${esc(f.frontier_id)}</strong> · ${esc(f.status)}<br>
          gain ${esc(f.unknown_gain)} · path ${esc(f.path_cost_m ?? 'n/a')}m · priority ${esc(f.llm_memory_priority ?? 'n/a')}<br>
          <span class="muted">${esc((f.evidence || []).slice(0, 2).join(' | '))}</span>
        </div>
      `).join('') || '<div class="muted">No active frontier information.</div>';
    }
    function renderThumbs() {
      const frames = (state.map?.keyframes || []).slice(-4);
      document.getElementById('thumbs').innerHTML = frames.map((f) => `
        <div>
          <img src="${esc(f.thumbnail_data_url)}" alt="${esc(f.frame_id)}">
          <div class="muted">${esc(f.frame_id)} · ${esc(f.description)}</div>
        </div>
      `).join('');
    }
    function renderSemantic() {
      const visible = !!state.map?.automatic_semantic_waypoints;
      document.getElementById('semantic-panel').style.display = visible ? '' : 'none';
      document.getElementById('call-semantic').style.display = visible ? '' : 'none';
      if (!visible) return;
      const memory = state.map?.semantic_memory || {};
      const places = memory.named_places || [];
      const anchors = memory.anchors || [];
      const traces = memory.traces || [];
      const anchorById = new Map(anchors.map((a) => [a.anchor_id, a]));
      document.getElementById('semantic').innerHTML = places.map((place) => {
        const primaryAnchor = (place.source_anchor_ids || []).map((id) => anchorById.get(id)).find(Boolean);
        const reachability = primaryAnchor?.reachability_status || 'unknown';
        const pose = place.anchor_pose || {};
        return `<div class="frontier">
          <strong>${esc(place.label)}</strong> · ${esc(place.status)} · ${esc(Number(place.confidence || 0).toFixed(2))}<br>
          anchor ${esc(Number(pose.x || 0).toFixed(2))}, ${esc(Number(pose.y || 0).toFixed(2))} · ${esc(reachability)}<br>
          <span class="muted">${esc((place.evidence || []).slice(0, 2).join(' | '))}</span>
        </div>`;
      }).join('') || `<div class="muted">No semantic places yet. ${traces.length ? esc(traces[traces.length - 1].status || '') : ''}</div>`;
    }
    function renderManualRegions() {
      const regions = state.map?.regions || [];
      document.getElementById('manual-regions').innerHTML = regions.map((region) => {
        const waypoints = region.default_waypoints || [];
        return `<div class="frontier">
          <strong>${esc(region.label)}</strong> · ${esc(region.region_id)}<br>
          <span class="muted">${esc(region.description || '')}</span><br>
          ${(waypoints || []).map((wp) => `<span>${esc(wp.name)} (${Number(wp.x || 0).toFixed(2)}, ${Number(wp.y || 0).toFixed(2)})</span>`).join('<br>')}
        </div>`;
      }).join('') || '<div class="muted">No manual regions yet. Click Add Region, select free cells, then Done Region.</div>';
    }
    function renderMap() {
      const svg = document.getElementById('map');
      const map = state.map;
      if (!map) { svg.innerHTML = '<text x="40" y="60">No map.</text>'; return; }
      const project = projector(bounds(map));
      const res = map.occupancy.resolution || 0.25;
      currentOccupancyCellStates = new Map();
      const cells = (map.occupancy.cells || []).map((c) => {
        const cellX = Math.floor(c.x / res);
        const cellY = Math.floor(c.y / res);
        currentOccupancyCellStates.set(`${cellX}:${cellY}`, {
          state: c.state,
          manual_override: c.manual_override || null,
        });
        const p = project({x:c.x, y:c.y});
        const p2 = project({x:c.x + res, y:c.y + res});
        const fill = c.manual_override === 'blocked'
          ? 'rgba(24,35,15,.86)'
          : c.manual_override === 'cleared'
            ? 'rgba(79,119,45,.16)'
            : c.state === 'occupied'
              ? 'rgba(24,35,15,.58)'
              : 'rgba(79,119,45,.16)';
        return `<rect x="${p.x}" y="${p2.y}" width="${Math.max(2, p2.x-p.x)}" height="${Math.max(2, p.y-p2.y)}" fill="${fill}"/>`;
      }).join('');
      const traj = (map.trajectory || []).map((p0) => { const p = project(p0); return `${p.x},${p.y}`; }).join(' ');
      const frontiers = (map.frontiers || []).map((f) => {
        const p = project(f.nav_pose);
        const b = f.frontier_boundary_pose ? project(f.frontier_boundary_pose) : p;
        const selected = f.frontier_id === state.pending_target?.frontier_id;
        const color = f.status === 'suppressed' ? '#71717a' : selected ? '#b0891f' : '#31572c';
        const r = selected ? 11 : 7;
        return `<circle cx="${b.x}" cy="${b.y}" r="4" fill="none" stroke="${color}" stroke-width="2" opacity=".72"><title>${esc(f.frontier_id)} boundary</title></circle>
                <line x1="${b.x}" y1="${b.y}" x2="${p.x}" y2="${p.y}" stroke="${color}" stroke-width="1.5" stroke-dasharray="4 4" opacity=".5"/>
                <circle cx="${p.x}" cy="${p.y}" r="${r}" fill="${color}" opacity="${selected ? 1 : .76}"><title>${esc(f.frontier_id)} approach</title></circle>
                <text x="${p.x + 10}" y="${p.y - 8}" font-size="12" fill="${color}">${esc(f.frontier_id)}</text>`;
      }).join('');
      const remembered = (map.remembered_frontiers || []).map((f) => {
        const p = project(f.nav_pose);
        const b = f.frontier_boundary_pose ? project(f.frontier_boundary_pose) : p;
        return `<circle cx="${b.x}" cy="${b.y}" r="3.5" fill="none" stroke="#647067" stroke-width="1.5" opacity=".38"><title>${esc(f.frontier_id)} stored memory boundary</title></circle>
                <line x1="${b.x}" y1="${b.y}" x2="${p.x}" y2="${p.y}" stroke="#647067" stroke-width="1" stroke-dasharray="2 5" opacity=".28"/>
                <circle cx="${p.x}" cy="${p.y}" r="5" fill="#647067" opacity=".32"><title>${esc(f.frontier_id)} stored memory approach</title></circle>
                <text x="${p.x + 8}" y="${p.y + 14}" font-size="11" fill="#647067" opacity=".7">${esc(f.frontier_id)} memory</text>`;
      }).join('');
      const manualRegionCells = (map.regions || []).map((region, idx) => {
        const color = ['#2563eb', '#0891b2', '#9333ea', '#db2777', '#0f766e'][idx % 5];
        return (region.selected_cells || []).map((cell) => {
          const x = Number(cell.cell_x) * res;
          const y = Number(cell.cell_y) * res;
          const p = project({x, y});
          const p2 = project({x: x + res, y: y + res});
          return `<rect x="${p.x}" y="${p2.y}" width="${Math.max(2, p2.x-p.x)}" height="${Math.max(2, p.y-p2.y)}" fill="${color}" opacity=".28"><title>${esc(region.label)}</title></rect>`;
        }).join('');
      }).join('');
      const selectedCells = Array.from(selectedRegionCells.values()).map((cell) => {
        const x = Number(cell.cell_x) * res;
        const y = Number(cell.cell_y) * res;
        const p = project({x, y});
        const p2 = project({x: x + res, y: y + res});
        return `<rect x="${p.x}" y="${p2.y}" width="${Math.max(2, p2.x-p.x)}" height="${Math.max(2, p.y-p2.y)}" fill="#f59e0b" opacity=".46"/>`;
      }).join('');
      const manualWaypoints = (map.regions || []).map((region) => (region.default_waypoints || []).map((wp) => {
        const p = project(wp);
        const primary = (wp.kind || 'primary') === 'primary';
        return `<g data-region-id="${esc(region.region_id)}" data-waypoint-name="${esc(wp.name)}" class="manual-waypoint">
          <circle cx="${p.x}" cy="${p.y}" r="${primary ? 8 : 6}" fill="${primary ? '#1d4ed8' : '#0e7490'}"><title>${esc(region.label)} · ${esc(wp.name)}</title></circle>
          <text x="${p.x + 10}" y="${p.y + 4}" font-size="12" fill="#1e3a8a" font-weight="800">${esc(wp.name)}</text>
        </g>`;
      }).join('')).join('');
      const semanticMemory = map.automatic_semantic_waypoints ? (map.semantic_memory || {}) : {};
      const evidencePoints = (semanticMemory.evidence || []).map((ev) => {
        const p = project(ev.evidence_pose || {x:0,y:0});
        return `<circle cx="${p.x}" cy="${p.y}" r="5" fill="#7c3aed" opacity=".72"><title>${esc(ev.label_hint)} evidence: ${esc((ev.evidence || []).join(' | '))}</title></circle>`;
      }).join('');
      const semanticPlaces = (semanticMemory.named_places || []).map((place) => {
        const anchor = project(place.anchor_pose || {x:0,y:0});
        const evidence = place.evidence_pose ? project(place.evidence_pose) : null;
        return `${evidence ? `<line x1="${evidence.x}" y1="${evidence.y}" x2="${anchor.x}" y2="${anchor.y}" stroke="#7c3aed" stroke-width="1.6" stroke-dasharray="3 5" opacity=".58"/>` : ''}
                <rect x="${anchor.x - 7}" y="${anchor.y - 7}" width="14" height="14" rx="3" fill="#7c3aed" opacity=".9"><title>${esc(place.label)} semantic anchor</title></rect>
                <text x="${anchor.x + 10}" y="${anchor.y + 4}" font-size="12" fill="#4c1d95" font-weight="800">${esc(place.label)}</text>`;
      }).join('');
      const robot = project(state.robot_pose || {x:0,y:0});
      const robotYaw = Number(state.robot_pose?.yaw || 0);
      const headingWorld = {
        x: (state.robot_pose?.x || 0) + Math.cos(robotYaw) * Math.max(res * 2.5, 0.45),
        y: (state.robot_pose?.y || 0) + Math.sin(robotYaw) * Math.max(res * 2.5, 0.45),
      };
      const heading = project(headingWorld);
      svg.innerHTML = `<rect width="1000" height="760" fill="rgba(255,255,255,.92)"/>${cells}
        <polyline points="${traj}" fill="none" stroke="#4f772d" stroke-width="4" stroke-linecap="round"/>
        ${manualRegionCells}
        ${selectedCells}
        ${remembered}
        ${manualWaypoints}
        ${evidencePoints}
        ${semanticPlaces}
        ${frontiers}
        <circle cx="${robot.x}" cy="${robot.y}" r="13" fill="#a52820"/>
        <line x1="${robot.x}" y1="${robot.y}" x2="${heading.x}" y2="${heading.y}" stroke="#6d0f0a" stroke-width="5" stroke-linecap="round"/>
        <circle cx="${heading.x}" cy="${heading.y}" r="4.5" fill="#6d0f0a"/>
        <text x="${robot.x + 14}" y="${robot.y - 12}" fill="#a52820" font-weight="800">robot</text>`;
      svg.onpointerdown = (event) => {
        event.preventDefault();
        if (regionMode === 'select') {
          const cell = cellFromMapEvent(map, event);
          selectRegionCell(cell);
          isPaintingMap = true;
          return;
        }
        if (pendingSubwaypoint) {
          const cell = cellFromMapEvent(map, event);
          const pose = cellCenterPose(map, cell);
          post('/api/region/subwaypoint', {region_id: pendingSubwaypoint.region_id, name: pendingSubwaypoint.name, pose});
          pendingSubwaypoint = null;
          return;
        }
        isPaintingMap = true;
        lastPaintedCellKey = null;
        const cell = cellFromMapEvent(map, event);
        lastPaintedCellKey = cell.key;
        enqueuePaintCell(cell);
      };
      svg.onpointermove = (event) => {
        if (!isPaintingMap) return;
        event.preventDefault();
        if (regionMode === 'select') {
          selectRegionCell(cellFromMapEvent(map, event));
          return;
        }
        const cell = cellFromMapEvent(map, event);
        if (cell.key === lastPaintedCellKey) return;
        lastPaintedCellKey = cell.key;
        enqueuePaintCell(cell);
      };
      svg.onpointerup = async (event) => {
        if (!isPaintingMap) return;
        event.preventDefault();
        isPaintingMap = false;
        lastPaintedCellKey = null;
        if (regionMode === 'select') return;
        await flushPaintCells();
      };
      svg.onpointerleave = async () => {
        if (!isPaintingMap) return;
        isPaintingMap = false;
        lastPaintedCellKey = null;
        if (regionMode === 'select') return;
        await flushPaintCells();
      };
      for (const element of svg.querySelectorAll('.manual-waypoint')) {
        element.onpointerdown = (event) => {
          event.stopPropagation();
          draggingWaypoint = {
            region_id: element.getAttribute('data-region-id'),
            waypoint_name: element.getAttribute('data-waypoint-name')
          };
        };
      }
      svg.onpointerup = async (event) => {
        if (draggingWaypoint) {
          const cell = cellFromMapEvent(map, event);
          const pose = cellCenterPose(map, cell);
          const payload = {...draggingWaypoint, pose};
          draggingWaypoint = null;
          await post('/api/region/waypoint', payload);
          return;
        }
        if (!isPaintingMap) return;
        event.preventDefault();
        isPaintingMap = false;
        lastPaintedCellKey = null;
        if (regionMode === 'select') return;
        await flushPaintCells();
      };
    }
    function renderPromptAndResponse() {
      document.getElementById('prompt').value = state.prompt || '';
      document.getElementById('response').textContent = state.pending_decision
        ? JSON.stringify({decision: state.pending_decision, applied_memory_updates: state.applied_memory_updates}, null, 2)
        : 'No response yet.';
    }
    function render() {
      renderStats();
      renderFrontiers();
      renderManualRegions();
      renderSemantic();
      renderThumbs();
      renderMap();
      renderPromptAndResponse();
    }
    document.getElementById('reset').onclick = () => post('/api/reset');
    document.getElementById('pause').onclick = () => post('/api/pause');
    document.getElementById('resume').onclick = () => post('/api/resume');
    document.getElementById('call').onclick = () => post('/api/call_llm');
    document.getElementById('call-semantic').onclick = () => post('/api/call_semantic_llm');
    document.getElementById('apply').onclick = () => post('/api/apply_decision');
    document.getElementById('control-robot').onclick = () => post('/api/control_robot');
    document.getElementById('edit-block').onclick = () => { mapEditMode = 'block'; renderStats(); };
    document.getElementById('edit-clear').onclick = () => { mapEditMode = 'clear'; renderStats(); };
    document.getElementById('edit-reset').onclick = () => { mapEditMode = 'reset'; renderStats(); };
    document.getElementById('region-add').onclick = () => {
      regionMode = 'select';
      selectedRegionCells = new Map();
      document.getElementById('error').textContent = 'Select free cells for the region, then click Done Region.';
      renderMap();
    };
    document.getElementById('region-done').onclick = async () => {
      if (!selectedRegionCells.size) {
        document.getElementById('error').textContent = 'Select at least one free cell before finishing the region.';
        return;
      }
      const label = prompt('Region name');
      if (!label) return;
      const description = prompt('Region description') || '';
      const cells = Array.from(selectedRegionCells.values());
      regionMode = null;
      selectedRegionCells = new Map();
      await post('/api/region/create', {label, description, cells});
    };
    document.getElementById('subwaypoint-add').onclick = () => {
      const regions = state.map?.regions || [];
      if (!regions.length) {
        document.getElementById('error').textContent = 'Create a region before adding subwaypoints.';
        return;
      }
      const region_id = prompt(`Region id for subwaypoint:\n${regions.map((r) => `${r.region_id}: ${r.label}`).join('\n')}`);
      if (!region_id) return;
      const name = prompt('Subwaypoint name') || 'subwaypoint';
      pendingSubwaypoint = {region_id, name};
      document.getElementById('error').textContent = 'Click a free map cell to place the subwaypoint.';
    };
    refresh();
    setInterval(refresh, 1000);
  </script>
</body>
</html>
"""


class InteractiveExplorationServer:
    def __init__(self, session: Any, *, host: str, port: int, ui_flavor: str = "developer") -> None:
        self.session = session
        self.host = host
        self.port = port
        self.ui_flavor = ui_flavor
        self._server: ThreadingHTTPServer | None = None
        self._serve_error: BaseException | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    @property
    def serve_error(self) -> BaseException | None:
        return self._serve_error

    def serve_in_background(self) -> threading.Thread:
        thread = threading.Thread(target=self.serve_forever, name="interactive_exploration_http", daemon=True)
        thread.start()
        return thread

    def serve_forever(self) -> None:
        session = self.session
        html = INTERACTIVE_REACT_HTML.replace("__UI_FLAVOR__", self.ui_flavor)

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path in {"/", "/index.html"}:
                    self._send_html(html)
                    return
                if self.path == "/api/state":
                    self._send_json(session.snapshot())
                    return
                self.send_error(HTTPStatus.NOT_FOUND, "not found")

            def do_POST(self) -> None:
                if self.path == "/api/reset":
                    self._send_json(session.reset())
                    return
                if self.path == "/api/pause":
                    self._send_json(session.pause())
                    return
                if self.path == "/api/resume":
                    self._send_json(session.resume())
                    return
                if self.path == "/api/control_robot":
                    controller = getattr(session, "control_robot", None)
                    if callable(controller):
                        self._send_json(controller())
                    else:
                        self._send_json(session.pause())
                    return
                if self.path == "/api/manual_drive":
                    controller = getattr(session, "manual_drive", None)
                    if callable(controller):
                        self._send_json(controller(self._read_json()))
                    else:
                        setattr(session, "last_error", "Manual drive is not available in this exploration mode.")
                        self._send_json(session.snapshot())
                    return
                if self.path == "/api/manual_stop":
                    controller = getattr(session, "manual_stop", None)
                    if callable(controller):
                        self._send_json(controller())
                    else:
                        setattr(session, "last_error", "Manual stop is not available in this exploration mode.")
                        self._send_json(session.snapshot())
                    return
                if self.path == "/api/manual_scan":
                    controller = getattr(session, "manual_scan", None)
                    if callable(controller):
                        self._send_json(controller())
                    else:
                        setattr(session, "last_error", "Manual scan is not available in this exploration mode.")
                        self._send_json(session.snapshot())
                    return
                if self.path == "/api/auto_explore":
                    self._send_json(self._auto_explore_step())
                    return
                if self.path == "/api/call_llm":
                    self._send_json(session.call_llm())
                    return
                if self.path == "/api/call_semantic_llm":
                    self._send_json(session.call_semantic_llm())
                    return
                if self.path == "/api/apply_decision":
                    self._send_json(session.apply_decision())
                    return
                if self.path == "/api/visualize_nav2_path":
                    planner = getattr(session, "visualize_planned_nav_path", None)
                    if callable(planner):
                        self._send_json(planner())
                    else:
                        setattr(session, "last_error", "Nav2 path visualization is only available in the Nav2 router mode.")
                        self._send_json(session.snapshot())
                    return
                if self.path == "/api/region/create":
                    self._send_json(session.create_manual_region(self._read_json()))
                    return
                if self.path == "/api/region/waypoint":
                    self._send_json(session.update_manual_region_waypoint(self._read_json()))
                    return
                if self.path == "/api/region/subwaypoint":
                    self._send_json(session.add_manual_region_subwaypoint(self._read_json()))
                    return
                if self.path == "/api/frontier/solved":
                    solver = getattr(session, "mark_selected_frontier_solved", None)
                    if callable(solver):
                        self._send_json(solver())
                    else:
                        setattr(session, "last_error", "Manual frontier solving is not available in this robot exploration mode.")
                        self._send_json(session.snapshot())
                    return
                if self.path == "/api/frontier/manual":
                    self._send_json(session.add_manual_frontier(self._read_json()))
                    return
                if self.path == "/api/map/edit":
                    payload = self._read_json()
                    self._send_json(
                        session.update_occupancy_edits(
                            mode=str(payload.get("mode", "block")),
                            cells=list(payload.get("cells", [])),
                        )
                    )
                    return
                self.send_error(HTTPStatus.NOT_FOUND, "not found")

            def log_message(self, format: str, *args: Any) -> None:
                return

            def _read_json(self) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length", "0"))
                if not length:
                    return {}
                return json.loads(self.rfile.read(length).decode("utf-8"))

            def _auto_explore_step(self) -> dict[str, Any]:
                snapshot = session.snapshot()
                status = str(snapshot.get("status", ""))
                if status in {"finished", "paused", "human_control", "human_assistance_required"}:
                    return snapshot
                started_from_llm_wait = status in {"waiting_for_llm", "initial_scan_complete"}
                if status in {"waiting_for_llm", "initial_scan_complete"}:
                    snapshot = session.call_llm()
                    status = str(snapshot.get("status", ""))
                planner = getattr(session, "visualize_planned_nav_path", None)
                if callable(planner) and status == "llm_response_ready":
                    snapshot = planner()
                    status = str(snapshot.get("status", ""))
                if started_from_llm_wait:
                    return snapshot
                if status in {"llm_response_ready", "semantic_response_ready"}:
                    snapshot = session.apply_decision()
                return snapshot

            def _send_html(self, content: str) -> None:
                encoded = content.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def _send_json(self, payload: dict[str, Any]) -> None:
                encoded = json.dumps(payload).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        try:
            self._server = ThreadingHTTPServer((self.host, self.port), Handler)
            self._server.serve_forever()
        except BaseException as exc:
            self._serve_error = exc
            raise

    def shutdown(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Robot Exploration Mode for autonomous frontier exploration."
    )
    parser.add_argument("--backend", choices=("synthetic", "maniskill", "ros"), default="synthetic")
    parser.add_argument("--repo-root", default=str(resolve_xlerobot_repo_root()))
    parser.add_argument("--session", default="robot_exploration")
    parser.add_argument("--area", default="apartment")
    parser.add_argument("--source", default="operator")
    parser.add_argument("--env-id", default="SceneManipulation-v1")
    parser.add_argument("--robot-uid", default="xlerobot")
    parser.add_argument("--display-yaw-offset-deg", type=float, default=None)
    parser.add_argument("--control-mode", default="pd_joint_delta_pos_dual_arm")
    parser.add_argument("--render-mode", default="human")
    parser.add_argument("--shader", default="default")
    parser.add_argument("--sim-backend", default="auto")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--force-reload", action="store_true")
    parser.add_argument("--build-config-idx", type=int, default=None)
    parser.add_argument("--spawn-x", type=float, default=None)
    parser.add_argument("--spawn-y", type=float, default=None)
    parser.add_argument("--spawn-yaw", type=float, default=0.0)
    parser.add_argument("--spawn-facing", choices=("front", "left", "right", "back"), default="front")
    parser.add_argument("--scan-mode", choices=("turnaround", "front_only"), default="turnaround")
    parser.add_argument("--scan-yaw-samples", type=int, default=12)
    parser.add_argument("--depth-beam-stride", type=int, default=2)
    parser.add_argument("--teleport-z", type=float, default=None)
    parser.add_argument("--teleport-settle-steps", type=int, default=1)
    parser.add_argument("--max-frontiers", type=int, default=12)
    parser.add_argument("--occupancy-resolution", type=float, default=0.25)
    parser.add_argument("--sensor-range-m", type=float, default=10.0)
    parser.add_argument("--robot-radius-m", type=float, default=0.22)
    parser.add_argument(
        "--frontier-min-opening-m",
        type=float,
        default=None,
        help="Override the minimum frontier opening width. Defaults to robot diameter plus clearance.",
    )
    parser.add_argument(
        "--visited-frontier-filter-radius-m",
        type=float,
        default=None,
        help=(
            "Suppress frontier boundaries within this distance of previous robot poses. "
            "Defaults to robot footprint plus clearance."
        ),
    )
    parser.add_argument("--finish-coverage-threshold", type=float, default=0.96)
    parser.add_argument("--max-decisions", type=int, default=64)
    parser.add_argument("--explorer-policy", choices=("heuristic", "llm"), default="llm")
    parser.add_argument("--llm-provider", default="mock")
    parser.add_argument("--llm-model", default="mock")
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--llm-temperature", type=float, default=0.1)
    parser.add_argument("--llm-max-tokens", type=int, default=1200)
    parser.add_argument("--llm-reasoning-effort", default=None)
    parser.add_argument("--semantic-waypoints-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--automatic-semantic-waypoints", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--semantic-llm-provider", default=None)
    parser.add_argument("--semantic-llm-model", default=None)
    parser.add_argument("--semantic-llm-base-url", default=None)
    parser.add_argument("--semantic-llm-api-key", default=None)
    parser.add_argument("--semantic-vlm-async", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nav2-planner-id", default="GridBased")
    parser.add_argument("--nav2-controller-id", default="FollowPath")
    parser.add_argument("--nav2-behavior-tree", default=default_nav2_behavior_tree())
    parser.add_argument("--nav2-recovery-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--ros-navigation-map-source",
        choices=("fused_scan", "fused_point_cloud", "external"),
        default="fused_scan",
    )
    parser.add_argument("--ros-map-topic", default="/map")
    parser.add_argument("--ros-scan-topic", default="/scan")
    parser.add_argument("--ros-rgb-topic", default="/camera/head/image_raw")
    parser.add_argument("--ros-cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--ros-map-frame", default="map")
    parser.add_argument("--ros-adapter-url", default=None)
    parser.add_argument("--ros-adapter-timeout-s", type=float, default=30.0)
    parser.add_argument(
        "--ros-odom-frame",
        default="odom",
        help=(
            "Accepted for command compatibility. Robot Exploration Mode uses TF from map to base_link; "
            "set odom in the bridge/Nav2 launch files."
        ),
    )
    parser.add_argument("--ros-base-frame", default="base_link")
    parser.add_argument("--ros-server-timeout-s", type=float, default=10.0)
    parser.add_argument("--ros-ready-timeout-s", type=float, default=20.0)
    parser.add_argument("--ros-turn-scan-timeout-s", type=float, default=45.0)
    parser.add_argument("--ros-turn-scan-settle-s", type=float, default=1.0)
    parser.add_argument("--ros-manual-spin-angular-speed-rad-s", type=float, default=0.25)
    parser.add_argument("--ros-manual-spin-publish-hz", type=float, default=20.0)
    parser.add_argument("--ros-turn-scan-mode", choices=("camera_pan", "robot_spin"), default="camera_pan")
    parser.add_argument("--robot-brain-url", default="http://127.0.0.1:8765")
    parser.add_argument("--camera-pan-action-key", default="head_motor_1.pos")
    parser.add_argument("--camera-pan-settle-s", type=float, default=0.5)
    parser.add_argument("--camera-pan-sample-count", type=int, default=12)
    parser.add_argument("--sim-motion-speed", choices=("normal", "faster", "fastest"), default="normal")
    parser.add_argument("--ros-allow-multiple-action-servers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8781)
    parser.add_argument("--open-browser", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ui-flavor", choices=("user", "developer"), default="developer")
    parser.add_argument(
        "--experimental-free-space-semantic-waypoints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Deprecated debugging path: render frontier-policy semantic area candidates. "
            "Normal runs keep semantic waypointing passive and RGB-D evidence grounded."
        ),
    )
    parser.add_argument(
        "--use-keyboard-controls",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable arrow key controls for manual robot teleoperation.",
    )
    parser.add_argument(
        "--keyboard-speed",
        choices=("slow", "normal", "fast"),
        default="normal",
        help="Speed profile for keyboard movement.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = SimExplorationConfig(
        repo_root=args.repo_root,
        persist_path="",
        area=args.area,
        session=args.session,
        source=args.source,
        env_id=args.env_id,
        robot_uid=args.robot_uid,
        control_mode=args.control_mode,
        render_mode=None if args.render_mode == "none" else args.render_mode,
        shader=args.shader,
        sim_backend=args.sim_backend,
        num_envs=args.num_envs,
        force_reload=args.force_reload,
        occupancy_resolution=args.occupancy_resolution,
        sensor_range_m=args.sensor_range_m,
        robot_radius_m=args.robot_radius_m,
        frontier_min_opening_m=args.frontier_min_opening_m,
        visited_frontier_filter_radius_m=args.visited_frontier_filter_radius_m,
        finish_coverage_threshold=args.finish_coverage_threshold,
        max_decisions=args.max_decisions,
        explorer_policy=args.explorer_policy,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        llm_api_key=args.llm_api_key,
        llm_temperature=args.llm_temperature,
        llm_max_tokens=args.llm_max_tokens,
        llm_reasoning_effort=args.llm_reasoning_effort,
        nav2_mode="ros" if args.backend == "ros" else "simulated",
        nav2_planner_id=args.nav2_planner_id,
        nav2_controller_id=args.nav2_controller_id,
        nav2_behavior_tree=args.nav2_behavior_tree,
        nav2_recovery_enabled=args.nav2_recovery_enabled,
        ros_navigation_map_source=args.ros_navigation_map_source,
        ros_map_topic=args.ros_map_topic,
        ros_scan_topic=args.ros_scan_topic,
        ros_rgb_topic=args.ros_rgb_topic,
        ros_cmd_vel_topic=args.ros_cmd_vel_topic,
        ros_map_frame=args.ros_map_frame,
        ros_adapter_url=args.ros_adapter_url,
        ros_adapter_timeout_s=args.ros_adapter_timeout_s,
        ros_odom_frame=args.ros_odom_frame,
        ros_base_frame=args.ros_base_frame,
        ros_server_timeout_s=args.ros_server_timeout_s,
        ros_ready_timeout_s=args.ros_ready_timeout_s,
        ros_turn_scan_timeout_s=args.ros_turn_scan_timeout_s,
        ros_turn_scan_settle_s=args.ros_turn_scan_settle_s,
        ros_manual_spin_angular_speed_rad_s=args.ros_manual_spin_angular_speed_rad_s,
        ros_manual_spin_publish_hz=args.ros_manual_spin_publish_hz,
        ros_turn_scan_mode=args.ros_turn_scan_mode,
        robot_brain_url=args.robot_brain_url,
        camera_pan_action_key=args.camera_pan_action_key,
        camera_pan_settle_s=args.camera_pan_settle_s,
        camera_pan_sample_count=args.camera_pan_sample_count,
        sim_motion_speed=args.sim_motion_speed,
        ros_allow_multiple_action_servers=args.ros_allow_multiple_action_servers,
        realtime_sleep_s=0.0,
        experimental_free_space_semantic_waypoints=args.experimental_free_space_semantic_waypoints,
        semantic_waypoints_enabled=args.semantic_waypoints_enabled,
        automatic_semantic_waypoints=args.automatic_semantic_waypoints,
        semantic_llm_provider=args.semantic_llm_provider,
        semantic_llm_model=args.semantic_llm_model,
        semantic_llm_base_url=args.semantic_llm_base_url,
        semantic_llm_api_key=args.semantic_llm_api_key,
        semantic_vlm_async=args.semantic_vlm_async,
        use_keyboard_controls=args.use_keyboard_controls,
        keyboard_speed=args.keyboard_speed,
    )
    if args.backend == "ros":
        if args.ros_adapter_url:
            session = ManiSkillNav2RouterExplorationSession(
                config,
                ManiSkillInteractiveOptions(
                    repo_root=args.repo_root,
                    env_id=args.env_id,
                    robot_uid=args.robot_uid,
                    display_yaw_offset_deg=args.display_yaw_offset_deg,
                    control_mode=args.control_mode,
                    render_mode=None if args.render_mode == "none" else args.render_mode,
                    shader=args.shader,
                    sim_backend=args.sim_backend,
                    num_envs=args.num_envs,
                    force_reload=args.force_reload,
                    build_config_idx=args.build_config_idx,
                    spawn_x=args.spawn_x,
                    spawn_y=args.spawn_y,
                    spawn_yaw=args.spawn_yaw,
                    spawn_facing=args.spawn_facing,
                    scan_mode=args.scan_mode,
                    scan_yaw_samples=args.scan_yaw_samples,
                    depth_beam_stride=args.depth_beam_stride,
                    teleport_z=args.teleport_z,
                    teleport_settle_steps=args.teleport_settle_steps,
                    max_frontiers=args.max_frontiers,
                    use_keyboard_controls=args.use_keyboard_controls,
                    keyboard_speed=args.keyboard_speed,
                ),
            )
        else:
            backend = ExplorationBackend(
                ExplorationBackendConfig(
                    mode="ros",
                    persist_path=None,
                    occupancy_resolution=args.occupancy_resolution,
                )
            )
            task = backend.begin_external_task(
                tool_id="explore",
                area=args.area,
                session=args.session,
                source=args.source,
                message="Starting interactive ROS/Nav2 frontier exploration.",
            )
            session = InteractiveRosNav2ExplorationSession(config, backend, str(task["task_id"]))
    elif args.backend == "maniskill":
        session = ManiSkillTeleportExplorationSession(
            config,
            ManiSkillInteractiveOptions(
                repo_root=args.repo_root,
                env_id=args.env_id,
                robot_uid=args.robot_uid,
                display_yaw_offset_deg=args.display_yaw_offset_deg,
                control_mode=args.control_mode,
                render_mode=None if args.render_mode == "none" else args.render_mode,
                shader=args.shader,
                sim_backend=args.sim_backend,
                num_envs=args.num_envs,
                force_reload=args.force_reload,
                build_config_idx=args.build_config_idx,
                spawn_x=args.spawn_x,
                spawn_y=args.spawn_y,
                spawn_yaw=args.spawn_yaw,
                spawn_facing=args.spawn_facing,
                scan_mode=args.scan_mode,
                scan_yaw_samples=args.scan_yaw_samples,
                depth_beam_stride=args.depth_beam_stride,
                teleport_z=args.teleport_z,
                teleport_settle_steps=args.teleport_settle_steps,
                max_frontiers=args.max_frontiers,
                use_keyboard_controls=args.use_keyboard_controls,
                keyboard_speed=args.keyboard_speed,
            ),
        )
    else:
        session = InteractiveNoNav2ExplorationSession(config)
    server = InteractiveExplorationServer(session, host=args.host, port=args.port, ui_flavor=args.ui_flavor)
    print(
        f"Robot exploration mode running at {server.url} "
        f"(backend={args.backend}, flavor={args.ui_flavor})"
    )
    print("Flow: automatic LLM decision -> Nav2 path preview when available -> automatic frontier motion.")
    if args.open_browser:
        webbrowser.open(server.url)
    try:
        if callable(getattr(session, "pump_viewer", None)):
            server_thread = server.serve_in_background()
            print("Pumping the ManiSkill/SAPIEN viewer on the main thread to keep the window responsive.")
            while server_thread.is_alive():
                if server.serve_error is not None:
                    raise server.serve_error
                pump_viewer = getattr(session, "pump_viewer", None)
                if callable(pump_viewer):
                    pump_viewer()
                time.sleep(1.0 / 30.0)
            if server.serve_error is not None:
                raise server.serve_error
        else:
            server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down robot exploration mode.")
    finally:
        server.shutdown()
        close = getattr(session, "close", None)
        if callable(close):
            close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
