from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
import heapq
import json
import math
from pathlib import Path
import threading
import time
from typing import Any, Callable, Iterable
import webbrowser

from multido_xlerobot.bootstrap import resolve_xlerobot_repo_root
from xlerobot_agent.exploration import ExplorationBackend, ExplorationBackendConfig, Pose2D
from xlerobot_agent.exploration_ui import ExplorationReviewServer, LocalExplorationUIController
from xlerobot_agent.llm import AgentLLMRouter, AgentModelSuite, ModelConfig
from xlerobot_agent.prompts import (
    build_exploration_policy_system_prompt,
    build_exploration_policy_user_prompt,
)
from xlerobot_agent.semantic_prompts import (
    build_semantic_evidence_extraction_system_prompt,
    build_semantic_evidence_extraction_user_prompt,
)
from xlerobot_playground.semantic_anchors import build_semantic_anchor_candidate
from xlerobot_playground.semantic_evidence import (
    PixelRegion,
    SemanticEvidence,
    SemanticObservation,
    deterministic_semantic_id,
    parse_semantic_observation_payload,
)
from xlerobot_playground.semantic_memory import SemanticMemory, normalize_label
from xlerobot_playground.semantic_projection import (
    CameraIntrinsics,
    project_pixel_region_to_map,
    project_pixel_to_map,
    representative_pixel_for_image_position,
)
from xlerobot_playground.ros_nav2_runtime import (
    GoalStatus,
    RosExplorationRuntime,
    RosOccupancyMap,
    RosRuntimeConfig,
    path_length_m,
    require_runtime_dependencies as require_ros_nav2_runtime_dependencies,
    ros_goal_status_label,
    rclpy,
    seconds_since,
)
from xlerobot_playground.ros_nav2_adapter import RemoteRosExplorationRuntime
from xlerobot_playground.frontier_runtime import refresh_frontier_records
from xlerobot_playground.map_editing import (
    ACTIVE_RGBD_SCAN_FUSION_CONFIG,
    EditableOccupancyMap,
    edits_from_payload,
    merge_occupancy_observation,
    overlay_known_cells,
    overlay_occupancy_payload,
)
from xlerobot_playground.nav2_defaults import default_nav2_behavior_tree
from xlerobot_playground.point_cloud_fusion import (
    PointCloudFusionConfig,
    integrate_transformed_point_cloud_observation,
)
from xlerobot_playground.scan_fusion import integrate_planar_scan


STORED_FRONTIER_REVALIDATION_RADIUS_M = 1.0
STORED_FRONTIER_REVISIT_APPROACH_RADIUS_M = 1.25


@dataclass
class SimExplorationConfig:
    repo_root: str
    persist_path: str
    env_id: str = "SceneManipulation-v1"
    robot_uid: str = "xlerobot"
    control_mode: str = "pd_joint_delta_pos_dual_arm"
    render_mode: str | None = "human"
    shader: str = "default"
    sim_backend: str = "auto"
    num_envs: int = 1
    force_reload: bool = False
    area: str = "apartment"
    session: str = "house_v1"
    source: str = "operator"
    occupancy_resolution: float = 0.25
    max_control_steps: int | None = None
    max_episode_steps: int | None = None
    show_cameras: bool = True
    use_rerun: bool = False
    camera_log_stride: int = 2
    realtime_sleep_s: float = 0.01
    explorer_policy: str = "llm"
    llm_provider: str = "mock"
    llm_model: str = "mock"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1200
    llm_reasoning_effort: str | None = None
    trace_policy_stdout: bool = False
    trace_llm_stdout: bool = False
    review_host: str = "127.0.0.1"
    review_port: int = 8770
    serve_review_ui: bool = False
    review_ui_flavor: str = "user"
    sensor_range_m: float = 10.0
    robot_radius_m: float = 0.22
    frontier_min_opening_m: float | None = None
    visited_frontier_filter_radius_m: float | None = None
    finish_coverage_threshold: float = 0.96
    max_decisions: int = 32
    nav2_planner_id: str = "GridBased"
    nav2_controller_id: str = "FollowPath"
    nav2_behavior_tree: str = field(default_factory=default_nav2_behavior_tree)
    nav2_recovery_enabled: bool = True
    nav2_mode: str = "simulated"
    ros_navigation_map_source: str = "fused_scan"
    ros_map_topic: str = "/map"
    ros_map_updates_topic: str | None = None
    ros_fuse_external_projected_map_snapshots: bool = False
    ros_scan_topic: str = "/scan"
    ros_point_cloud_topic: str = "/camera/head/points"
    ros_scan_active_topic: str = "/xlerobot/scan_active"
    ros_nav_active_topic: str = "/xlerobot/nav_active"
    ros_scan_active_release_delay_s: float = 3.0
    ros_rgb_topic: str = "/camera/head/image_raw"
    ros_imu_topic: str = "/imu/filtered_yaw"
    ros_cmd_vel_topic: str = "/cmd_vel"
    ros_map_frame: str = "map"
    ros_odom_frame: str = "odom"
    ros_base_frame: str = "base_link"
    ros_adapter_url: str | None = None
    ros_adapter_timeout_s: float = 30.0
    ros_server_timeout_s: float = 10.0
    ros_ready_timeout_s: float = 20.0
    ros_turn_scan_timeout_s: float = 45.0
    ros_turn_scan_settle_s: float = 1.0
    ros_manual_spin_angular_speed_rad_s: float = 0.25
    ros_manual_spin_publish_hz: float = 20.0
    ros_turn_scan_mode: str = "camera_pan"
    robot_brain_url: str | None = "http://127.0.0.1:8765"
    camera_pan_action_key: str = "head_motor_1.pos"
    camera_pan_settle_s: float = 0.5
    camera_pan_step_deg: float = 60.0
    camera_pan_compute_s: float = 2.0
    camera_pan_sample_count: int = 12
    point_cloud_range_min_m: float = 0.25
    point_cloud_range_max_m: float = 4.0
    point_cloud_floor_free_max_z_m: float = 0.08
    point_cloud_obstacle_min_z_m: float = 0.08
    point_cloud_robot_clearance_height_m: float = 1.50
    point_cloud_obstacle_max_z_m: float = 1.80
    point_cloud_max_rays: int = 2400
    sim_motion_speed: str = "normal"
    ros_allow_multiple_action_servers: bool = False
    experimental_free_space_semantic_waypoints: bool = False
    semantic_waypoints_enabled: bool = True
    automatic_semantic_waypoints: bool = False
    semantic_llm_provider: str | None = None
    semantic_llm_model: str | None = None
    semantic_llm_base_url: str | None = None
    semantic_llm_api_key: str | None = None
    semantic_vlm_async: bool = True
    wait_for_ui_start: bool = False
    pause_for_operator_approval: bool = False
    stop_after_initial_scan: bool = False
    use_keyboard_controls: bool = False
    keyboard_speed: str = "normal"


@dataclass(frozen=True, order=True)
class GridCell:
    x: int
    y: int

    def center_pose(self, resolution: float, *, yaw: float = 0.0) -> Pose2D:
        return Pose2D((self.x + 0.5) * resolution, (self.y + 0.5) * resolution, yaw)

    def to_xy(self, resolution: float) -> tuple[float, float]:
        pose = self.center_pose(resolution)
        return pose.x, pose.y


@dataclass(frozen=True)
class SubAreaSpec:
    area_id: str
    label: str
    polygon_2d: tuple[tuple[float, float], ...]
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class RoomSpec:
    region_id: str
    label: str
    polygon_2d: tuple[tuple[float, float], ...]
    adjacency: tuple[str, ...]
    objects: tuple[str, ...]
    descriptions: tuple[str, ...]
    entry_pose: Pose2D
    center_pose: Pose2D
    sub_areas: tuple[SubAreaSpec, ...] = tuple()


@dataclass(frozen=True)
class WorldObject:
    object_id: str
    label: str
    room_id: str
    cell: GridCell
    description: str


@dataclass
class ApartmentScenario:
    scenario_id: str
    layout_id: str
    resolution: float
    width_cells: int
    height_cells: int
    start_cell: GridCell
    free_cells: set[GridCell]
    obstacle_cells: set[GridCell]
    room_by_cell: dict[GridCell, str]
    rooms: dict[str, RoomSpec]
    objects: list[WorldObject]

    def in_bounds(self, cell: GridCell) -> bool:
        return 0 <= cell.x < self.width_cells and 0 <= cell.y < self.height_cells

    def is_free(self, cell: GridCell) -> bool:
        return cell in self.free_cells

    def is_occupied(self, cell: GridCell) -> bool:
        return not self.in_bounds(cell) or cell in self.obstacle_cells or cell not in self.free_cells

    def room_for_cell(self, cell: GridCell) -> str | None:
        return self.room_by_cell.get(cell)

    def room_for_pose(self, pose: Pose2D) -> str | None:
        return self.room_for_cell(self.world_to_cell(pose.x, pose.y))

    def world_to_cell(self, x: float, y: float) -> GridCell:
        return GridCell(int(math.floor(x / self.resolution)), int(math.floor(y / self.resolution)))

    def bounds(self) -> dict[str, float]:
        return {
            "min_x": 0.0,
            "max_x": round(self.width_cells * self.resolution, 3),
            "min_y": 0.0,
            "max_y": round(self.height_cells * self.resolution, 3),
        }

    def total_free_cells(self) -> int:
        return len(self.free_cells)


@dataclass
class ScanResult:
    observed_free: set[GridCell]
    observed_occupied: set[GridCell]
    range_edge_frontiers: set[GridCell]
    visible_objects: list[str]
    visible_room_ids: list[str]
    point_count: int
    depth_min_m: float
    depth_max_m: float
    description: str
    thumbnail_data_url: str


@dataclass
class FrontierCandidate:
    frontier_id: str | None
    member_cells: tuple[GridCell, ...]
    nav_cell: GridCell
    centroid_cell: GridCell
    nav_pose: Pose2D
    centroid_pose: Pose2D
    unknown_gain: int
    sensor_range_edge: bool
    room_hint: str | None
    evidence: list[str]
    currently_visible: bool = True
    path_cost_m: float | None = None

    def to_prompt_dict(self) -> dict[str, Any]:
        path_distance_m = None if self.path_cost_m is None else round(self.path_cost_m, 3)
        return {
            "frontier_id": self.frontier_id,
            "nav_pose": self.nav_pose.to_dict(),
            "centroid_pose": self.centroid_pose.to_dict(),
            "unknown_gain": self.unknown_gain,
            "sensor_range_edge": self.sensor_range_edge,
            "room_hint": self.room_hint,
            "currently_visible": self.currently_visible,
            "path_cost_m": path_distance_m,
            "free_space_path_distance_m": path_distance_m,
            "evidence": list(self.evidence),
        }


@dataclass
class FrontierRecord:
    frontier_id: str
    nav_pose: Pose2D
    centroid_pose: Pose2D
    status: str
    discovered_step: int
    last_seen_step: int
    unknown_gain: int
    sensor_range_edge: bool
    room_hint: str | None
    evidence: list[str] = field(default_factory=list)
    llm_memory_label: str | None = None
    llm_memory_priority: float | None = None
    llm_memory_notes: list[str] = field(default_factory=list)
    attempt_count: int = 0
    visit_count: int = 0
    path_cost_m: float | None = None
    currently_visible: bool = False

    def to_dict(self) -> dict[str, Any]:
        path_distance_m = None if self.path_cost_m is None else round(self.path_cost_m, 3)
        return {
            "frontier_id": self.frontier_id,
            "nav_pose": self.nav_pose.to_dict(),
            "approach_pose": self.nav_pose.to_dict(),
            "centroid_pose": self.centroid_pose.to_dict(),
            "frontier_boundary_pose": self.centroid_pose.to_dict(),
            "status": self.status,
            "discovered_step": self.discovered_step,
            "last_seen_step": self.last_seen_step,
            "unknown_gain": self.unknown_gain,
            "sensor_range_edge": self.sensor_range_edge,
            "room_hint": self.room_hint,
            "evidence": list(self.evidence),
            "llm_memory_label": self.llm_memory_label,
            "llm_memory_priority": self.llm_memory_priority,
            "llm_memory_notes": list(self.llm_memory_notes),
            "attempt_count": self.attempt_count,
            "visit_count": self.visit_count,
            "path_cost_m": path_distance_m,
            "free_space_path_distance_m": path_distance_m,
            "currently_visible": self.currently_visible,
        }


@dataclass
class ExplorationDecision:
    decision_type: str
    selected_frontier_id: str | None
    selected_return_waypoint_id: str | None
    frontier_ids_to_store: list[str]
    exploration_complete: bool
    reasoning_summary: str
    semantic_updates: list[dict[str, Any]]
    memory_updates: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_type": self.decision_type,
            "selected_frontier_id": self.selected_frontier_id,
            "selected_return_waypoint_id": self.selected_return_waypoint_id,
            "frontier_ids_to_store": list(self.frontier_ids_to_store),
            "exploration_complete": self.exploration_complete,
            "reasoning_summary": self.reasoning_summary,
            "semantic_updates": json.loads(json.dumps(self.semantic_updates)),
            "memory_updates": json.loads(json.dumps(self.memory_updates)),
        }


@dataclass(frozen=True)
class Nav2GoalRequest:
    goal_id: str
    goal_type: str
    target_pose: Pose2D
    planner_id: str
    controller_id: str
    behavior_tree: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "goal_type": self.goal_type,
            "target_pose": self.target_pose.to_dict(),
            "planner_id": self.planner_id,
            "controller_id": self.controller_id,
            "behavior_tree": self.behavior_tree,
            "metadata": json.loads(json.dumps(self.metadata)),
        }


@dataclass(frozen=True)
class Nav2GoalValidation:
    accepted: bool
    goal: Nav2GoalRequest
    normalized_pose: Pose2D | None
    goal_cell: GridCell | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "goal": self.goal.to_dict(),
            "normalized_pose": None if self.normalized_pose is None else self.normalized_pose.to_dict(),
            "goal_cell": None if self.goal_cell is None else {"x": self.goal_cell.x, "y": self.goal_cell.y},
            "reason": self.reason,
        }


@dataclass(frozen=True)
class Nav2PlanResult:
    status: str
    goal: Nav2GoalRequest
    planner_id: str
    reason: str
    goal_cell: GridCell | None
    path_cells: tuple[GridCell, ...] = tuple()
    path_poses: tuple[Pose2D, ...] = tuple()
    path_length_m: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "goal": self.goal.to_dict(),
            "planner_id": self.planner_id,
            "reason": self.reason,
            "goal_cell": None if self.goal_cell is None else {"x": self.goal_cell.x, "y": self.goal_cell.y},
            "path_length_m": round(self.path_length_m, 3),
            "waypoint_count": len(self.path_cells),
            "path": [
                {"x": point.x, "y": point.y}
                for point in self.path_cells
            ],
            "path_poses": [pose.to_dict() for pose in self.path_poses],
        }


@dataclass(frozen=True)
class Nav2NavigateResult:
    status: str
    goal: Nav2GoalRequest
    plan: Nav2PlanResult
    reached_pose: Pose2D | None
    travelled_distance_m: float
    reason: str
    feedback_samples: tuple[dict[str, Any], ...] = tuple()
    recovery_events: tuple[dict[str, Any], ...] = tuple()

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "goal": self.goal.to_dict(),
            "plan": self.plan.to_dict(),
            "reached_pose": None if self.reached_pose is None else self.reached_pose.to_dict(),
            "travelled_distance_m": round(self.travelled_distance_m, 3),
            "reason": self.reason,
            "feedback_samples": list(self.feedback_samples),
            "recovery_events": list(self.recovery_events),
        }


class Nav2NavigationModule:
    def validate_goal(self, goal: Nav2GoalRequest) -> Nav2GoalValidation:
        raise NotImplementedError

    def compute_path(self, goal: Nav2GoalRequest, *, record: bool = True) -> Nav2PlanResult:
        raise NotImplementedError

    def navigate_to_pose(
        self,
        goal: Nav2GoalRequest,
        *,
        ignore_pause_cancel: bool = False,
        feedback_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> Nav2NavigateResult:
        raise NotImplementedError

    def recover(self, goal: Nav2GoalRequest, *, reason: str) -> dict[str, Any]:
        raise NotImplementedError

    def snapshot(self) -> dict[str, Any]:
        raise NotImplementedError


class SimulatedNav2NavigationModule(Nav2NavigationModule):
    def __init__(
        self,
        config: SimExplorationConfig,
        scenario: ApartmentScenario,
        *,
        get_current_cell: Callable[[], GridCell],
        get_current_yaw: Callable[[], float],
        known_free_cells: Callable[[], set[GridCell]],
        on_motion_step: Callable[[GridCell, GridCell, Nav2GoalRequest, int, int], None],
        on_runtime_obstacle: Callable[[GridCell], None],
        is_runtime_blocked: Callable[[GridCell], bool],
        budget_exhausted: Callable[[], bool],
        should_stop_execution: Callable[[], bool],
    ) -> None:
        self.config = config
        self.scenario = scenario
        self._get_current_cell = get_current_cell
        self._get_current_yaw = get_current_yaw
        self._known_free_cells = known_free_cells
        self._on_motion_step = on_motion_step
        self._on_runtime_obstacle = on_runtime_obstacle
        self._is_runtime_blocked = is_runtime_blocked
        self._budget_exhausted = budget_exhausted
        self._should_stop_execution = should_stop_execution
        self._goal_history: list[dict[str, Any]] = []
        self._plan_history: list[dict[str, Any]] = []
        self._recovery_history: list[dict[str, Any]] = []

    def validate_goal(self, goal: Nav2GoalRequest) -> Nav2GoalValidation:
        goal_cell = self.scenario.world_to_cell(goal.target_pose.x, goal.target_pose.y)
        if not self.scenario.in_bounds(goal_cell):
            return Nav2GoalValidation(
                accepted=False,
                goal=goal,
                normalized_pose=None,
                goal_cell=goal_cell,
                reason="goal lies outside the known map bounds",
            )
        normalized_pose = goal_cell.center_pose(self.scenario.resolution, yaw=goal.target_pose.yaw)
        known_free = self._known_free_cells()
        if goal_cell not in known_free:
            return Nav2GoalValidation(
                accepted=False,
                goal=goal,
                normalized_pose=normalized_pose,
                goal_cell=goal_cell,
                reason="goal is not currently reachable known free space for Nav2",
            )
        return Nav2GoalValidation(
            accepted=True,
            goal=goal,
            normalized_pose=normalized_pose,
            goal_cell=goal_cell,
            reason="goal accepted by simulated Nav2 goal checker",
        )

    def compute_path(self, goal: Nav2GoalRequest, *, record: bool = True) -> Nav2PlanResult:
        validation = self.validate_goal(goal)
        if not validation.accepted:
            result = Nav2PlanResult(
                status="rejected",
                goal=goal,
                planner_id=goal.planner_id,
                reason=validation.reason,
                goal_cell=validation.goal_cell,
            )
            if record:
                self._plan_history.append(result.to_dict())
            return result
        path_cells = _search_known_safe_path(
            self._get_current_cell(),
            validation.goal_cell,
            self._known_free_cells(),
        )
        if not path_cells:
            result = Nav2PlanResult(
                status="failed",
                goal=goal,
                planner_id=goal.planner_id,
                reason="planner could not find a known-safe path to the goal",
                goal_cell=validation.goal_cell,
            )
            if record:
                self._plan_history.append(result.to_dict())
            return result
        result = Nav2PlanResult(
            status="succeeded",
            goal=goal,
            planner_id=goal.planner_id,
            reason="planner computed a path on the current known-free occupancy map",
            goal_cell=validation.goal_cell,
            path_cells=tuple(path_cells),
            path_poses=tuple(
                point.center_pose(self.scenario.resolution, yaw=goal.target_pose.yaw)
                for point in path_cells
            ),
            path_length_m=max(len(path_cells) - 1, 0) * self.scenario.resolution,
        )
        if record:
            self._plan_history.append(result.to_dict())
        return result

    def navigate_to_pose(
        self,
        goal: Nav2GoalRequest,
        *,
        ignore_pause_cancel: bool = False,
        feedback_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> Nav2NavigateResult:
        self._goal_history.append(goal.to_dict())
        plan = self.compute_path(goal, record=True)
        recovery_events: list[dict[str, Any]] = []
        if plan.status != "succeeded":
            if self.config.nav2_recovery_enabled:
                recovery_events.append(self.recover(goal, reason=plan.reason))
            return Nav2NavigateResult(
                status="failed",
                goal=goal,
                plan=plan,
                reached_pose=None,
                travelled_distance_m=0.0,
                reason=plan.reason,
                recovery_events=tuple(recovery_events),
            )

        feedback_samples: list[dict[str, Any]] = []
        travelled_distance_m = 0.0
        path_cells = list(plan.path_cells)
        for index, (previous, nxt) in enumerate(zip(path_cells, path_cells[1:]), start=1):
            if self._budget_exhausted():
                reason = "Nav2 execution stopped because the control-step budget was exhausted"
                if self.config.nav2_recovery_enabled:
                    recovery_events.append(self.recover(goal, reason=reason))
                return Nav2NavigateResult(
                    status="failed",
                    goal=goal,
                    plan=plan,
                    reached_pose=previous.center_pose(self.scenario.resolution, yaw=self._get_current_yaw()),
                    travelled_distance_m=travelled_distance_m,
                    reason=reason,
                    feedback_samples=tuple(feedback_samples),
                    recovery_events=tuple(recovery_events),
                )
            if self._should_stop_execution():
                reason = "Nav2 execution stopped because the task was canceled"
                return Nav2NavigateResult(
                    status="failed",
                    goal=goal,
                    plan=plan,
                    reached_pose=previous.center_pose(self.scenario.resolution, yaw=self._get_current_yaw()),
                    travelled_distance_m=travelled_distance_m,
                    reason=reason,
                    feedback_samples=tuple(feedback_samples),
                    recovery_events=tuple(recovery_events),
                )
            if self._is_runtime_blocked(nxt):
                self._on_runtime_obstacle(nxt)
                reason = "Nav2 execution encountered a runtime obstacle on the current path"
                if self.config.nav2_recovery_enabled:
                    recovery_events.append(self.recover(goal, reason=reason))
                return Nav2NavigateResult(
                    status="failed",
                    goal=goal,
                    plan=plan,
                    reached_pose=previous.center_pose(self.scenario.resolution, yaw=self._get_current_yaw()),
                    travelled_distance_m=travelled_distance_m,
                    reason=reason,
                    feedback_samples=tuple(feedback_samples),
                    recovery_events=tuple(recovery_events),
                )
            self._on_motion_step(previous, nxt, goal, index, len(path_cells))
            travelled_distance_m += self.scenario.resolution
            remaining_distance_m = max((len(path_cells) - index - 1), 0) * self.scenario.resolution
            sample = {
                "step_index": index,
                "remaining_distance_m": round(remaining_distance_m, 3),
                "current_pose": nxt.center_pose(self.scenario.resolution).to_dict(),
                "status": "moving",
            }
            feedback_samples.append(sample)
            if feedback_callback is not None:
                feedback_callback(sample)

        reached_pose = path_cells[-1].center_pose(self.scenario.resolution, yaw=goal.target_pose.yaw)
        return Nav2NavigateResult(
            status="succeeded",
            goal=goal,
            plan=plan,
            reached_pose=reached_pose,
            travelled_distance_m=travelled_distance_m,
            reason="Nav2 reached the requested goal pose",
            feedback_samples=tuple(feedback_samples),
            recovery_events=tuple(recovery_events),
        )

    def recover(self, goal: Nav2GoalRequest, *, reason: str) -> dict[str, Any]:
        event = {
            "goal_id": goal.goal_id,
            "behavior_tree": goal.behavior_tree,
            "recovery_action": "spin_and_rescan",
            "reason": reason,
            "timestamp": time.time(),
        }
        self._recovery_history.append(event)
        return event

    def snapshot(self) -> dict[str, Any]:
        return {
            "module": "simulated_nav2",
            "planner_id": self.config.nav2_planner_id,
            "controller_id": self.config.nav2_controller_id,
            "behavior_tree": self.config.nav2_behavior_tree,
            "goals": list(self._goal_history),
            "plans": list(self._plan_history),
            "recoveries": list(self._recovery_history),
        }


class RosNav2NavigationModule(Nav2NavigationModule):
    def __init__(
        self,
        config: SimExplorationConfig,
        runtime: RosExplorationRuntime,
        *,
        current_map: Callable[[], RosOccupancyMap | EditableOccupancyMap | None],
        should_cancel: Callable[[], bool] | None = None,
    ) -> None:
        self.config = config
        self.runtime = runtime
        self._current_map = current_map
        self._should_cancel = should_cancel
        self._goal_history: list[dict[str, Any]] = []
        self._plan_history: list[dict[str, Any]] = []
        self._recovery_history: list[dict[str, Any]] = []

    def validate_goal(self, goal: Nav2GoalRequest) -> Nav2GoalValidation:
        occupancy_map = self._current_map()
        if occupancy_map is None:
            return Nav2GoalValidation(
                accepted=False,
                goal=goal,
                normalized_pose=None,
                goal_cell=None,
                reason="ROS occupancy map is not available yet",
            )
        goal_cell = GridCell(*occupancy_map.world_to_cell(goal.target_pose.x, goal.target_pose.y))
        normalized_cell = self._normalize_goal_cell(occupancy_map, goal_cell)
        if normalized_cell is None:
            return Nav2GoalValidation(
                accepted=False,
                goal=goal,
                normalized_pose=None,
                goal_cell=None,
                reason="goal does not land on mapped free space or a nearby reachable free cell",
            )
        normalized_pose = occupancy_map.cell_to_pose(normalized_cell.x, normalized_cell.y, yaw=goal.target_pose.yaw)
        reason = "goal accepted by ROS/Nav2 goal checker"
        if normalized_cell != goal_cell:
            reason = "goal normalized onto the nearest mapped free cell before Nav2 planning"
        return Nav2GoalValidation(
            accepted=True,
            goal=goal,
            normalized_pose=normalized_pose,
            goal_cell=normalized_cell,
            reason=reason,
        )

    def compute_path(self, goal: Nav2GoalRequest, *, record: bool = True) -> Nav2PlanResult:
        validation = self.validate_goal(goal)
        if not validation.accepted:
            result = Nav2PlanResult(
                status="rejected",
                goal=goal,
                planner_id=goal.planner_id,
                reason=validation.reason,
                goal_cell=validation.goal_cell,
            )
            if record:
                self._plan_history.append(result.to_dict())
            return result
        try:
            status, path_poses, _raw_result = self.runtime.compute_path(
                goal_pose=validation.normalized_pose or goal.target_pose,
                planner_id=goal.planner_id,
            )
        except Exception as exc:
            result = Nav2PlanResult(
                status="failed",
                goal=goal,
                planner_id=goal.planner_id,
                reason=f"ROS Nav2 planner call failed: {exc}",
                goal_cell=validation.goal_cell,
            )
            if record:
                self._plan_history.append(result.to_dict())
            return result
        occupancy_map = self._current_map()
        path_cells = tuple(
            GridCell(*occupancy_map.world_to_cell(item.x, item.y))
            for item in path_poses
        ) if occupancy_map is not None else tuple()
        result = Nav2PlanResult(
            status="succeeded" if status == GoalStatus.STATUS_SUCCEEDED and path_poses else "failed",
            goal=goal,
            planner_id=goal.planner_id,
            reason=(
                "Nav2 planner returned a valid path on the live occupancy map"
                if status == GoalStatus.STATUS_SUCCEEDED and path_poses
                else f"Nav2 planner returned status `{ros_goal_status_label(status)}` with {len(path_poses)} path poses"
            ),
            goal_cell=validation.goal_cell,
            path_cells=path_cells,
            path_poses=tuple(path_poses),
            path_length_m=path_length_m(path_poses),
        )
        if record:
            self._plan_history.append(result.to_dict())
        return result

    def navigate_to_pose(
        self,
        goal: Nav2GoalRequest,
        *,
        ignore_pause_cancel: bool = False,
        feedback_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> Nav2NavigateResult:
        self._goal_history.append(goal.to_dict())
        plan = self.compute_path(goal, record=True)
        if plan.status != "succeeded":
            recovery_events: list[dict[str, Any]] = []
            if self.config.nav2_recovery_enabled:
                recovery_events.append(self.recover(goal, reason=plan.reason))
            return Nav2NavigateResult(
                status="failed",
                goal=goal,
                plan=plan,
                reached_pose=self.runtime.current_pose(),
                travelled_distance_m=0.0,
                reason=plan.reason,
                recovery_events=tuple(recovery_events),
            )
        validation = self.validate_goal(goal)
        try:
            self.runtime.set_point_cloud_map_updates_enabled(False)
            try:
                outcome, feedback_samples = self.runtime.navigate_to_pose(
                    goal_pose=validation.normalized_pose or goal.target_pose,
                    behavior_tree=goal.behavior_tree,
                    should_cancel=None if ignore_pause_cancel else self._should_cancel,
                    feedback_callback=feedback_callback,
                )
            finally:
                self.runtime.set_point_cloud_map_updates_enabled(False)
        except Exception as exc:
            reason = f"ROS Nav2 navigate_to_pose call failed: {exc}"
            recovery_events: list[dict[str, Any]] = []
            if self.config.nav2_recovery_enabled:
                recovery_events.append(self.recover(goal, reason=reason))
            return Nav2NavigateResult(
                status="failed",
                goal=goal,
                plan=plan,
                reached_pose=self.runtime.current_pose(),
                travelled_distance_m=0.0,
                reason=reason,
                recovery_events=tuple(recovery_events),
            )
        status = _goal_status_from_outcome(outcome)
        if status == GoalStatus.STATUS_SUCCEEDED:
            return Nav2NavigateResult(
                status="succeeded",
                goal=goal,
                plan=plan,
                reached_pose=self.runtime.current_pose(),
                travelled_distance_m=plan.path_length_m,
                reason="Nav2 reached the requested goal pose on the live map",
                feedback_samples=tuple(feedback_samples),
            )
        reason = f"Nav2 returned status `{ros_goal_status_label(status)}`"
        recovery_events = []
        if self.config.nav2_recovery_enabled:
            recovery_events.append(self.recover(goal, reason=reason))
        return Nav2NavigateResult(
            status="failed",
            goal=goal,
            plan=plan,
            reached_pose=self.runtime.current_pose(),
            travelled_distance_m=plan.path_length_m,
            reason=reason,
            feedback_samples=tuple(feedback_samples),
            recovery_events=tuple(recovery_events),
        )

    def recover(self, goal: Nav2GoalRequest, *, reason: str) -> dict[str, Any]:
        event = {
            "goal_id": goal.goal_id,
            "behavior_tree": goal.behavior_tree,
            "recovery_action": "nav2_spin_turnaround_scan",
            "reason": reason,
            "timestamp": time.time(),
        }
        self._recovery_history.append(event)
        return event

    def snapshot(self) -> dict[str, Any]:
        snapshot = self.runtime.snapshot()
        snapshot.update(
            {
                "planner_id": self.config.nav2_planner_id,
                "controller_id": self.config.nav2_controller_id,
                "behavior_tree": self.config.nav2_behavior_tree,
                "goals": list(self._goal_history),
                "plans": list(self._plan_history),
                "recoveries": list(self._recovery_history),
            }
        )
        return snapshot

    def _normalize_goal_cell(
        self,
        occupancy_map: RosOccupancyMap | EditableOccupancyMap,
        goal_cell: GridCell,
    ) -> GridCell | None:
        def _safe_free(cell: GridCell) -> bool:
            if not occupancy_map.in_bounds(cell.x, cell.y):
                return False
            edge_margin_cells = 3
            if (
                cell.x < edge_margin_cells
                or cell.y < edge_margin_cells
                or cell.x >= occupancy_map.width - edge_margin_cells
                or cell.y >= occupancy_map.height - edge_margin_cells
            ):
                return False
            return occupancy_map.is_free(cell.x, cell.y)

        if _safe_free(goal_cell):
            return goal_cell
        for radius in range(1, 9):
            candidates: list[GridCell] = []
            for offset_x in range(-radius, radius + 1):
                for offset_y in range(-radius, radius + 1):
                    cell = GridCell(goal_cell.x + offset_x, goal_cell.y + offset_y)
                    if not _safe_free(cell):
                        continue
                    candidates.append(cell)
            if candidates:
                return min(
                    candidates,
                    key=lambda item: _grid_distance_cells(item, goal_cell),
                )
        return None


def _occupancy_map_like_to_ros_map(
    occupancy_map: RosOccupancyMap | EditableOccupancyMap,
) -> RosOccupancyMap:
    data: list[int] = []
    for y in range(int(occupancy_map.height)):
        for x in range(int(occupancy_map.width)):
            data.append(int(occupancy_map.value(x, y)))
    return RosOccupancyMap(
        resolution=float(occupancy_map.resolution),
        width=int(occupancy_map.width),
        height=int(occupancy_map.height),
        origin_x=float(occupancy_map.origin_x),
        origin_y=float(occupancy_map.origin_y),
        data=tuple(data),
    )


def _pose_from_map_payload(payload: Any) -> Pose2D | None:
    if not isinstance(payload, dict) or "x" not in payload or "y" not in payload:
        return None
    try:
        return Pose2D(float(payload["x"]), float(payload["y"]), float(payload.get("yaw", 0.0) or 0.0))
    except (TypeError, ValueError):
        return None


def _start_pose_from_environment_payload(map_payload: dict[str, Any]) -> Pose2D | None:
    start_pose = map_payload.get("start_pose")
    if isinstance(start_pose, dict):
        pose = _pose_from_map_payload(start_pose.get("pose"))
        if pose is not None:
            return pose
    for place in map_payload.get("named_places", []):
        if not isinstance(place, dict):
            continue
        if str(place.get("name", "")).lower() not in {"dock", "start", "home"}:
            continue
        pose = _pose_from_map_payload(place.get("pose"))
        if pose is not None:
            return pose
    return Pose2D(0.0, 0.0, 0.0)


def _ros_map_from_environment_payload(map_payload: dict[str, Any]) -> RosOccupancyMap | None:
    occupancy = map_payload.get("occupancy")
    if not isinstance(occupancy, dict):
        return None
    cells = [item for item in occupancy.get("cells", []) if isinstance(item, dict)]
    if not cells:
        return None
    resolution = float(occupancy.get("resolution") or map_payload.get("resolution") or 0.25)
    resolution = max(resolution, 1e-6)
    bounds = occupancy.get("bounds") if isinstance(occupancy.get("bounds"), dict) else {}
    try:
        min_x = float(bounds.get("min_x"))
        min_y = float(bounds.get("min_y"))
        max_x = float(bounds.get("max_x"))
        max_y = float(bounds.get("max_y"))
    except (TypeError, ValueError):
        xs = [float(item.get("x", 0.0)) for item in cells]
        ys = [float(item.get("y", 0.0)) for item in cells]
        min_x = min(xs)
        min_y = min(ys)
        max_x = max(xs) + resolution
        max_y = max(ys) + resolution
    width = max(int(round((max_x - min_x) / resolution)), 1)
    height = max(int(round((max_y - min_y) / resolution)), 1)
    data = [-1] * (width * height)
    occupancy_map = RosOccupancyMap(
        resolution=resolution,
        width=width,
        height=height,
        origin_x=min_x,
        origin_y=min_y,
        data=tuple(data),
    )
    for item in cells:
        try:
            x = float(item.get("x", 0.0)) + resolution / 2.0
            y = float(item.get("y", 0.0)) + resolution / 2.0
        except (TypeError, ValueError):
            continue
        cell_x, cell_y = occupancy_map.world_to_cell(x, y)
        if not occupancy_map.in_bounds(cell_x, cell_y):
            continue
        state = str(item.get("manual_override") or item.get("state") or "").lower()
        if state == "blocked" or state == "occupied":
            value = 100
        elif state == "cleared" or state == "free":
            value = 0
        else:
            continue
        data[cell_y * width + cell_x] = value
    return RosOccupancyMap(
        resolution=resolution,
        width=width,
        height=height,
        origin_x=min_x,
        origin_y=min_y,
        data=tuple(data),
    )


class FrontierMemory:
    def __init__(self, resolution: float) -> None:
        self.resolution = resolution
        self.records: dict[str, FrontierRecord] = {}
        self.active_frontier_id: str | None = None
        self.return_waypoints: dict[str, dict[str, Any]] = {}
        self._counter = 0

    def upsert_candidates(self, candidates: list[FrontierCandidate], *, step_index: int) -> list[FrontierRecord]:
        visible_ids: set[str] = set()
        for candidate in candidates:
            match_id, distance_m = self._find_match_id(candidate)
            if match_id is not None:
                record = self.records[match_id]
                if record.status in {"completed", "failed", "suppressed"} and distance_m <= self._dedupe_tolerance_m():
                    if (
                        distance_m <= self.resolution * 2.0
                        and candidate.unknown_gain <= record.unknown_gain + 1
                        and not candidate.sensor_range_edge
                    ):
                        continue
                    match_id = None
            if match_id is None:
                frontier_id = f"frontier_{self._counter:03d}"
                self._counter += 1
                record = FrontierRecord(
                    frontier_id=frontier_id,
                    nav_pose=candidate.nav_pose,
                    centroid_pose=candidate.centroid_pose,
                    status="stored",
                    discovered_step=step_index,
                    last_seen_step=step_index,
                    unknown_gain=candidate.unknown_gain,
                    sensor_range_edge=candidate.sensor_range_edge,
                    room_hint=candidate.room_hint,
                    evidence=list(candidate.evidence),
                    currently_visible=True,
                )
                self.records[frontier_id] = record
            else:
                record = self.records[match_id]
                record.nav_pose = candidate.nav_pose
                record.centroid_pose = candidate.centroid_pose
                record.last_seen_step = step_index
                record.unknown_gain = max(record.unknown_gain, candidate.unknown_gain)
                record.sensor_range_edge = record.sensor_range_edge or candidate.sensor_range_edge
                record.room_hint = candidate.room_hint or record.room_hint
                record.currently_visible = True
                record.evidence = _dedupe_text(record.evidence + candidate.evidence)
                if record.status not in {"active", "failed", "completed", "suppressed"}:
                    record.status = "stored"
            visible_ids.add(record.frontier_id)

        for record in self.records.values():
            if record.frontier_id not in visible_ids:
                record.currently_visible = False
        return [self.records[item] for item in visible_ids]

    def candidate_records(self) -> list[FrontierRecord]:
        return [
            record
            for record in self.records.values()
            if record.status not in {"completed", "failed", "suppressed"}
        ]

    def apply_model_memory_updates(
        self,
        updates: list[dict[str, Any]],
        *,
        selected_frontier_id: str | None,
    ) -> list[dict[str, Any]]:
        applied: list[dict[str, Any]] = []
        for update in updates:
            frontier_id = str(update.get("frontier_id", "")).strip()
            record = self.records.get(frontier_id)
            if record is None:
                continue
            action = str(update.get("action", "keep")).strip().lower()
            if action not in {"keep", "store", "prioritize", "suppress", "revalidate"}:
                action = "keep"
            if frontier_id == selected_frontier_id and action == "suppress":
                action = "prioritize"
            priority = _clamp_float(update.get("priority"), default=record.llm_memory_priority or 0.5, low=0.0, high=1.0)
            label = str(update.get("label", "")).strip()
            notes = str(update.get("notes", "")).strip()
            evidence = [
                str(item).strip()
                for item in update.get("evidence", [])
                if str(item).strip()
            ] if isinstance(update.get("evidence", []), list) else []
            if label:
                record.llm_memory_label = label
            record.llm_memory_priority = priority
            if notes:
                record.llm_memory_notes = _dedupe_text(record.llm_memory_notes + [notes])
            if evidence:
                record.evidence = _dedupe_text(record.evidence + evidence[:4])
            if action in {"store", "keep"} and record.status not in {"active", "completed", "failed"}:
                record.status = "stored"
            elif action == "prioritize" and record.status not in {"active", "completed", "failed"}:
                record.status = "stored"
                record.llm_memory_priority = max(priority, 0.8)
            elif action == "suppress" and record.status not in {"active", "completed", "failed"}:
                record.status = "suppressed"
            elif action == "revalidate" and record.status in {"suppressed", "failed"}:
                record.status = "stored"
                record.llm_memory_priority = max(priority, 0.65)
            applied.append(
                {
                    "frontier_id": frontier_id,
                    "action": action,
                    "status": record.status,
                    "priority": record.llm_memory_priority,
                    "label": record.llm_memory_label,
                    "notes": notes,
                }
            )
        return applied

    def activate(self, frontier_id: str) -> FrontierRecord | None:
        record = self.records.get(frontier_id)
        if record is None:
            return None
        if self.active_frontier_id and self.active_frontier_id != frontier_id:
            previous = self.records.get(self.active_frontier_id)
            if previous is not None and previous.status == "active":
                previous.status = "stored"
        self.active_frontier_id = frontier_id
        record.status = "active"
        record.attempt_count += 1
        return record

    def complete(self, frontier_id: str) -> FrontierRecord | None:
        record = self.records.get(frontier_id)
        if record is None:
            return None
        record.status = "completed"
        record.visit_count += 1
        if self.active_frontier_id == frontier_id:
            self.active_frontier_id = None
        return record

    def fail(self, frontier_id: str, reason: str) -> FrontierRecord | None:
        record = self.records.get(frontier_id)
        if record is None:
            return None
        record.status = "failed"
        record.evidence = _dedupe_text(record.evidence + [reason])
        if self.active_frontier_id == frontier_id:
            self.active_frontier_id = None
        return record

    def remember_return_waypoint(self, *, room_id: str | None, pose: Pose2D, step_index: int, reason: str) -> dict[str, Any]:
        prefix = (room_id or "unknown").replace("region_", "")
        waypoint_id = f"return_{prefix}_{step_index:03d}"
        entry = {
            "waypoint_id": waypoint_id,
            "room_id": room_id,
            "pose": pose.to_dict(),
            "step_index": step_index,
            "reason": reason,
        }
        self.return_waypoints[waypoint_id] = entry
        return entry

    def get_return_waypoint(self, waypoint_id: str | None) -> dict[str, Any] | None:
        if waypoint_id is None:
            return None
        return self.return_waypoints.get(waypoint_id)

    def snapshot(self) -> dict[str, Any]:
        stored = [record.to_dict() for record in self.records.values() if record.status == "stored"]
        active = None if self.active_frontier_id is None else self.records[self.active_frontier_id].to_dict()
        visited = [record.to_dict() for record in self.records.values() if record.visit_count > 0]
        failed = [record.to_dict() for record in self.records.values() if record.status == "failed"]
        suppressed = [record.to_dict() for record in self.records.values() if record.status == "suppressed"]
        completed = [record.to_dict() for record in self.records.values() if record.status == "completed"]
        return {
            "active_frontier": active,
            "stored_frontiers": stored,
            "visited_frontiers": visited,
            "failed_frontiers": failed,
            "suppressed_frontiers": suppressed,
            "completed_frontiers": completed,
            "return_waypoints": list(self.return_waypoints.values()),
        }

    def _dedupe_tolerance_m(self) -> float:
        return max(self.resolution * 4.0, 0.75)

    def _find_match_id(self, candidate: FrontierCandidate) -> tuple[str | None, float]:
        best_id = None
        best_distance = 1e9
        for record in self.records.values():
            distance = _pose_distance_m(record.centroid_pose, candidate.centroid_pose)
            if distance < best_distance:
                best_distance = distance
                best_id = record.frontier_id
        if best_id is None or best_distance > self._dedupe_tolerance_m():
            return None, best_distance
        return best_id, best_distance


def _mark_frontier_unreachable_as_visited(
    frontier_memory: FrontierMemory,
    frontier_id: str,
    reason: str,
) -> FrontierRecord | None:
    frontier_memory.fail(
        frontier_id,
        f"{reason}; marked visited because low-level navigation could not reach a safe approach pose",
    )
    return frontier_memory.complete(frontier_id)


class ExplorationLLMPolicy:
    def __init__(self, config: SimExplorationConfig) -> None:
        self.config = config
        self.system_prompt = build_exploration_policy_system_prompt()
        self.router: AgentLLMRouter | None = None
        if self.config.explorer_policy == "llm" and self.config.llm_provider != "mock":
            model = ModelConfig(
                provider=self.config.llm_provider,
                model=self.config.llm_model,
                base_url=self.config.llm_base_url,
                api_key=self.config.llm_api_key,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
                reasoning_effort=self.config.llm_reasoning_effort,
            )
            self.router = AgentLLMRouter(
                AgentModelSuite(
                    planner=model,
                    critic=model,
                    coder=model,
                )
            )

    def decide(
        self,
        *,
        prompt_payload: dict[str, Any],
        frontiers: list[FrontierRecord],
        return_waypoints: list[dict[str, Any]],
        coverage: float,
        current_room_id: str | None,
    ) -> tuple[ExplorationDecision, dict[str, Any]]:
        heuristic = self._heuristic_decision(frontiers, return_waypoints, coverage, current_room_id)
        user_prompt = build_exploration_policy_user_prompt(prompt_payload)
        trace: dict[str, Any] = {
            "mode": self.config.explorer_policy,
            "provider": self.config.llm_provider,
            "model": self.config.llm_model,
            "prompt": user_prompt,
        }

        if self.config.explorer_policy != "llm":
            trace["response"] = heuristic.to_dict()
            return heuristic, trace

        if self.router is None:
            trace["response"] = heuristic.to_dict()
            return heuristic, trace

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    *[
                        {"type": "image_url", "image_url": {"url": item["thumbnail_data_url"]}}
                        for item in prompt_payload.get("navigation_map_views", [])[-2:]
                        if item.get("thumbnail_data_url")
                    ],
                    *[
                        {"type": "image_url", "image_url": {"url": item["thumbnail_data_url"]}}
                        for item in prompt_payload.get("recent_views", [])[-4:]
                        if item.get("thumbnail_data_url")
                    ],
                ],
            },
        ]
        parsed, llm_trace = self.router.complete_json_messages(
            config=self.router.model_suite.planner,
            messages=messages,
        )
        trace["llm_trace"] = {
            "duration_s": llm_trace.duration_s,
            "error": llm_trace.error,
            "response_text": llm_trace.response_text,
        }
        if parsed is None:
            trace["response"] = heuristic.to_dict()
            trace["fallback_reason"] = "llm_parse_failure"
            return heuristic, trace

        decision = self._parse_model_decision(parsed, frontiers, return_waypoints)
        if decision is None:
            trace["response"] = heuristic.to_dict()
            trace["fallback_reason"] = "llm_invalid_decision"
            return heuristic, trace
        ignored_legacy_semantic_updates = self._ignored_legacy_semantic_updates(parsed)
        if ignored_legacy_semantic_updates:
            trace["ignored_legacy_frontier_semantic_update"] = {
                "count": len(ignored_legacy_semantic_updates),
                "reason": (
                    "frontier policy semantic updates are deprecated; visual semantic places are handled by "
                    "the passive RGB-D evidence pipeline"
                ),
                "items": ignored_legacy_semantic_updates,
            }
        trace["response"] = decision.to_dict()
        return decision, trace

    def _heuristic_decision(
        self,
        frontiers: list[FrontierRecord],
        return_waypoints: list[dict[str, Any]],
        coverage: float,
        current_room_id: str | None,
    ) -> ExplorationDecision:
        reachable = [record for record in frontiers if record.path_cost_m is not None]
        if not reachable:
            return ExplorationDecision(
                decision_type="finish",
                selected_frontier_id=None,
                selected_return_waypoint_id=None,
                frontier_ids_to_store=[],
                exploration_complete=True,
                reasoning_summary=(
                    "No reachable frontier remains in memory, so the exploration loop should finish."
                ),
                semantic_updates=[],
            )

        def score(record: FrontierRecord) -> float:
            distance_penalty = record.path_cost_m or 0.0
            novelty_bonus = 0.25 if record.currently_visible else 0.05
            range_bonus = 0.45 if record.sensor_range_edge else 0.0
            revisit_penalty = min(record.attempt_count * 0.5, 1.5)
            room_bonus = 0.25 if record.room_hint and record.room_hint != current_room_id else 0.0
            memory_priority_bonus = (record.llm_memory_priority or 0.5) * 0.2
            # Distance is deliberately dominant. Extra gain can break local ties, but should not
            # make the robot zig-zag across the apartment for a modestly larger frontier.
            gain_bonus = min(record.unknown_gain, 25) * 0.07
            return (
                gain_bonus
                + novelty_bonus
                + range_bonus
                + room_bonus
                + memory_priority_bonus
                - distance_penalty
                - revisit_penalty
            )

        ranked = sorted(reachable, key=score, reverse=True)
        best = ranked[0]
        hallway_waypoint = next(
            (
                item
                for item in reversed(return_waypoints)
                if item.get("room_id") == "region_hallway"
            ),
            None,
        )
        selected_return_waypoint_id = None
        if hallway_waypoint and current_room_id not in {None, "region_hallway"} and best.room_hint != current_room_id:
            selected_return_waypoint_id = str(hallway_waypoint["waypoint_id"])
        semantic_updates: list[dict[str, Any]] = []
        if self.config.experimental_free_space_semantic_waypoints and best.room_hint:
            semantic_updates.append(
                {
                    "label": best.room_hint.replace("region_", ""),
                    "kind": "room_hint",
                    "target_id": best.frontier_id,
                    "confidence": 0.58 if best.sensor_range_edge else 0.51,
                    "evidence": best.evidence[:3],
                }
            )
        decision = ExplorationDecision(
            decision_type="revisit_frontier" if not best.currently_visible else "explore_frontier",
            selected_frontier_id=best.frontier_id,
            selected_return_waypoint_id=selected_return_waypoint_id,
            frontier_ids_to_store=[record.frontier_id for record in ranked[1:] if record.frontier_id],
            exploration_complete=coverage >= self.config.finish_coverage_threshold and len(reachable) <= 1,
            reasoning_summary=(
                f"Select {best.frontier_id} because it is the nearest useful reachable frontier after "
                f"weighing known-free route distance, expected navigable-space gain, and visual/map novelty."
            ),
            semantic_updates=semantic_updates,
        )
        decision.memory_updates = self._heuristic_memory_updates(decision, frontiers)
        return decision

    def _heuristic_memory_updates(
        self,
        decision: ExplorationDecision,
        frontiers: list[FrontierRecord],
    ) -> list[dict[str, Any]]:
        updates: list[dict[str, Any]] = []
        for record in frontiers:
            if record.frontier_id == decision.selected_frontier_id:
                action = "prioritize"
                priority = 1.0
                notes = "Selected as the next active exploration target."
            elif record.frontier_id in decision.frontier_ids_to_store:
                action = "store"
                priority = 0.65
                notes = "Keep as a useful later exploration memory point."
            else:
                action = "keep"
                priority = 0.5
                notes = "Keep unless later visual/map evidence shows this is not useful."
            updates.append(
                {
                    "frontier_id": record.frontier_id,
                    "action": action,
                    "priority": priority,
                    "label": record.room_hint or "frontier_boundary",
                    "notes": notes,
                    "evidence": record.evidence[:3],
                }
            )
        return updates

    def _parse_model_decision(
        self,
        payload: dict[str, Any],
        frontiers: list[FrontierRecord],
        return_waypoints: list[dict[str, Any]],
    ) -> ExplorationDecision | None:
        valid_frontier_ids = {record.frontier_id for record in frontiers}
        valid_waypoint_ids = {str(item["waypoint_id"]) for item in return_waypoints}
        decision_type = str(payload.get("decision_type", "")).strip()
        if decision_type not in {"explore_frontier", "revisit_frontier", "finish"}:
            return None
        selected_frontier_id = payload.get("selected_frontier_id")
        if decision_type != "finish" and selected_frontier_id not in valid_frontier_ids:
            return None
        if decision_type == "finish":
            selected_frontier_id = None
        selected_return_waypoint_id = payload.get("selected_return_waypoint_id")
        if selected_return_waypoint_id is not None and selected_return_waypoint_id not in valid_waypoint_ids:
            selected_return_waypoint_id = None
        frontier_ids_to_store = [
            str(item)
            for item in payload.get("frontier_ids_to_store", [])
            if str(item) in valid_frontier_ids and str(item) != selected_frontier_id
        ]
        semantic_updates = []
        if self.config.experimental_free_space_semantic_waypoints:
            semantic_updates = [
                item
                for item in payload.get("semantic_updates", [])
                if isinstance(item, dict)
            ][:6]
        decision = ExplorationDecision(
            decision_type=decision_type,
            selected_frontier_id=selected_frontier_id,
            selected_return_waypoint_id=selected_return_waypoint_id,
            frontier_ids_to_store=frontier_ids_to_store,
            exploration_complete=bool(payload.get("exploration_complete", decision_type == "finish")),
            reasoning_summary=str(payload.get("reasoning_summary", "")).strip() or "Model selected the next exploration action.",
            semantic_updates=json.loads(json.dumps(semantic_updates)),
            memory_updates=self._parse_memory_updates(payload, frontiers, selected_frontier_id),
        )
        if not decision.memory_updates:
            decision.memory_updates = self._heuristic_memory_updates(decision, frontiers)
        return decision

    def _parse_memory_updates(
        self,
        payload: dict[str, Any],
        frontiers: list[FrontierRecord],
        selected_frontier_id: str | None,
    ) -> list[dict[str, Any]]:
        valid_frontier_ids = {record.frontier_id for record in frontiers}
        parsed_updates: list[dict[str, Any]] = []
        raw_updates = payload.get("memory_updates", [])
        if not isinstance(raw_updates, list):
            return []
        for item in raw_updates:
            if not isinstance(item, dict):
                continue
            frontier_id = str(item.get("frontier_id", "")).strip()
            if frontier_id not in valid_frontier_ids:
                continue
            action = str(item.get("action", "keep")).strip().lower()
            if action not in {"keep", "store", "prioritize", "suppress", "revalidate"}:
                action = "keep"
            if frontier_id == selected_frontier_id and action == "suppress":
                action = "prioritize"
            evidence = item.get("evidence", [])
            parsed_updates.append(
                {
                    "frontier_id": frontier_id,
                    "action": action,
                    "priority": _clamp_float(item.get("priority"), default=0.5, low=0.0, high=1.0),
                    "label": str(item.get("label", "")).strip()[:80],
                    "notes": str(item.get("notes", "")).strip()[:400],
                    "evidence": [
                        str(token).strip()[:240]
                        for token in evidence
                        if str(token).strip()
                    ][:6] if isinstance(evidence, list) else [],
                }
            )
        return parsed_updates

    def _ignored_legacy_semantic_updates(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        if self.config.experimental_free_space_semantic_waypoints:
            return []
        raw_updates = payload.get("semantic_updates", [])
        if not isinstance(raw_updates, list):
            return []
        ignored: list[dict[str, Any]] = []
        for item in raw_updates:
            if isinstance(item, dict):
                ignored.append(json.loads(json.dumps(item)))
        return ignored[:6]


OBJECT_SEMANTIC_LABEL_HINTS = {
    "fridge": "kitchen",
    "sink": "kitchen",
    "oven": "kitchen",
    "counter": "kitchen",
    "island": "kitchen",
    "cabinet": "kitchen",
    "sofa": "living_room",
    "tv": "living_room",
    "coffee table": "living_room",
    "rug": "living_room",
    "desk": "desk_area",
    "monitor": "desk_area",
    "keyboard": "desk_area",
    "office chair": "desk_area",
    "chair": "desk_area",
    "table": "dining_area",
    "dining table": "dining_area",
    "dining chair": "dining_area",
    "bed": "bedroom",
    "wardrobe": "bedroom",
    "nightstand": "bedroom",
    "toilet": "bathroom_entry",
    "bathtub": "bathroom_entry",
    "mirror": "bathroom_entry",
    "charging dock": "hallway",
    "shoe rack": "hallway",
}


class SemanticWaypointObserver:
    def __init__(self, config: SimExplorationConfig, *, scenario: ApartmentScenario | None = None) -> None:
        self.config = config
        self.scenario = scenario
        self.memory = SemanticMemory()
        self.traces: list[dict[str, Any]] = []
        self._evidence_index = 0
        self._anchor_index = 0
        self._observation_index = 0
        self._lock = threading.RLock()
        self.router: AgentLLMRouter | None = None
        provider = config.semantic_llm_provider or config.llm_provider
        model = config.semantic_llm_model or config.llm_model
        if config.automatic_semantic_waypoints and config.semantic_waypoints_enabled and provider != "mock" and model != "mock":
            model_config = ModelConfig(
                provider=provider,
                model=model,
                base_url=config.semantic_llm_base_url or config.llm_base_url,
                api_key=config.semantic_llm_api_key or config.llm_api_key,
                temperature=config.llm_temperature,
                max_tokens=config.llm_max_tokens,
                reasoning_effort=config.llm_reasoning_effort,
            )
            self.router = AgentLLMRouter(
                AgentModelSuite(planner=model_config, critic=model_config, coder=model_config, visual_summary=model_config)
            )

    def observe_keyframe(
        self,
        *,
        frame: dict[str, Any],
        known_cells: dict[GridCell, str],
        robot_cell: GridCell,
        resolution: float,
        depth_image: Any = None,
        intrinsics: CameraIntrinsics | None = None,
    ) -> None:
        if not self.config.semantic_waypoints_enabled:
            return
        trace: dict[str, Any] = {
            "type": "semantic_keyframe_observation",
            "frame_id": frame.get("frame_id"),
            "source": "heuristic_visible_objects",
            "created_evidence_ids": [],
            "created_anchor_ids": [],
            "warnings": [],
        }
        observations = self._heuristic_observations_from_frame(frame)
        if not observations and self.router is not None and frame.get("thumbnail_data_url"):
            if self.config.semantic_vlm_async:
                self._queue_vlm_observation(
                    frame=frame,
                    known_cells=known_cells,
                    robot_cell=robot_cell,
                    resolution=resolution,
                    depth_image=depth_image,
                    intrinsics=intrinsics,
                    trace=trace,
                )
                return
            observations, trace = self._vlm_observations_from_frame(frame, trace)
        if not observations:
            trace["status"] = "no_semantic_observations"
            with self._lock:
                self.traces.append(trace)
            return
        self._process_observations(
            observations=observations,
            frame=frame,
            known_cells=known_cells,
            robot_cell=robot_cell,
            resolution=resolution,
            depth_image=depth_image,
            intrinsics=intrinsics,
            trace=trace,
        )

    def observe_keyframe_batch(
        self,
        *,
        frames: list[dict[str, Any]],
        known_cells: dict[GridCell, str],
        robot_cell: GridCell,
        resolution: float,
    ) -> dict[str, Any]:
        if not self.config.semantic_waypoints_enabled:
            return {"status": "disabled", "frame_count": len(frames)}
        frame_ids = [str(frame.get("frame_id", "")) for frame in frames if frame.get("frame_id")]
        trace: dict[str, Any] = {
            "type": "semantic_spin_batch_observation",
            "frame_ids": frame_ids,
            "source": "heuristic_visible_objects_batch",
            "created_evidence_ids": [],
            "created_anchor_ids": [],
            "warnings": [],
        }
        observations_by_frame = [
            (frame, self._heuristic_observations_from_frame(frame))
            for frame in frames
        ]
        if any(observations for _, observations in observations_by_frame):
            for frame, observations in observations_by_frame:
                if not observations:
                    continue
                self._process_observations(
                    observations=observations,
                    frame=frame,
                    known_cells=known_cells,
                    robot_cell=robot_cell,
                    resolution=resolution,
                    depth_image=None,
                    intrinsics=None,
                    trace=trace,
                    append_trace=False,
                )
            trace["status"] = "processed"
            with self._lock:
                self.traces.append(trace)
            return trace

        if self.router is None or not frames:
            trace["status"] = "no_semantic_observations"
            with self._lock:
                self.traces.append(trace)
            return trace

        observations, trace = self._vlm_observations_from_frames(frames, trace)
        if not observations:
            trace["status"] = "no_semantic_observations"
            with self._lock:
                self.traces.append(trace)
            return trace
        representative_frame = frames[-1]
        self._process_observations(
            observations=observations,
            frame=representative_frame,
            known_cells=known_cells,
            robot_cell=robot_cell,
            resolution=resolution,
            depth_image=None,
            intrinsics=None,
            trace=trace,
        )
        return trace

    def _process_observations(
        self,
        *,
        observations: list[SemanticObservation],
        frame: dict[str, Any],
        known_cells: dict[GridCell, str],
        robot_cell: GridCell,
        resolution: float,
        depth_image: Any,
        intrinsics: CameraIntrinsics | None,
        trace: dict[str, Any],
        append_trace: bool = True,
    ) -> None:
        for observation in observations:
            evidence_pose = self._project_observation(
                observation=observation,
                frame=frame,
                depth_image=depth_image,
                intrinsics=intrinsics,
            )
            if evidence_pose is None:
                trace["warnings"].append(f"could not project observation {observation.observation_id}")
                continue
            with self._lock:
                self._evidence_index += 1
                evidence_id = deterministic_semantic_id("sem_ev", self._evidence_index)
            evidence = SemanticEvidence(
                evidence_id=evidence_id,
                label_hint=normalize_label(observation.label_hint),
                evidence_pose=evidence_pose,
                source_frame_ids=(observation.frame_id,),
                source_pixels=observation.pixel_regions,
                confidence=observation.confidence,
                evidence=tuple(
                    _dedupe_text(
                        [
                            *observation.visual_cues,
                            observation.reasoning_summary,
                            "semantic evidence projected from RGB-D keyframe into map coordinates",
                        ]
                    )
                ),
            )
            with self._lock:
                self.memory.add_evidence(evidence)
                self._anchor_index += 1
                anchor_id = deterministic_semantic_id("sem_anchor", self._anchor_index)
            anchor = build_semantic_anchor_candidate(
                anchor_id=anchor_id,
                evidence=evidence,
                known_cells=known_cells,
                resolution=resolution,
                robot_cell=robot_cell,
            )
            with self._lock:
                self.memory.add_anchor(anchor)
            trace["created_evidence_ids"].append(evidence.evidence_id)
            trace["created_anchor_ids"].append(anchor.anchor_id)
        trace["status"] = "processed"
        if append_trace:
            with self._lock:
                self.traces.append(trace)

    def _queue_vlm_observation(
        self,
        *,
        frame: dict[str, Any],
        known_cells: dict[GridCell, str],
        robot_cell: GridCell,
        resolution: float,
        depth_image: Any,
        intrinsics: CameraIntrinsics | None,
        trace: dict[str, Any],
    ) -> None:
        trace["source"] = "semantic_vlm_async"
        trace["status"] = "queued"
        trace["warnings"].append("semantic VLM queued in background so exploration UI startup is not blocked")
        with self._lock:
            self.traces.append(trace)

        def _worker() -> None:
            worker_trace: dict[str, Any] = {
                "type": "semantic_keyframe_observation",
                "frame_id": frame.get("frame_id"),
                "source": "semantic_vlm_async",
                "created_evidence_ids": [],
                "created_anchor_ids": [],
                "warnings": [],
            }
            observations, worker_trace = self._vlm_observations_from_frame(dict(frame), worker_trace)
            if not observations:
                worker_trace["status"] = "no_semantic_observations"
                with self._lock:
                    self.traces.append(worker_trace)
                return
            self._process_observations(
                observations=observations,
                frame=dict(frame),
                known_cells=dict(known_cells),
                robot_cell=robot_cell,
                resolution=resolution,
                depth_image=depth_image,
                intrinsics=intrinsics,
                trace=worker_trace,
            )

        threading.Thread(target=_worker, name=f"semantic-vlm-{frame.get('frame_id', 'frame')}", daemon=True).start()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            snapshot = self.memory.snapshot()
            snapshot["traces"] = list(self.traces)
        snapshot["enabled"] = self.config.semantic_waypoints_enabled
        snapshot["llm_provider"] = self.config.semantic_llm_provider or self.config.llm_provider
        snapshot["llm_model"] = self.config.semantic_llm_model or self.config.llm_model
        return snapshot

    def _heuristic_observations_from_frame(self, frame: dict[str, Any]) -> list[SemanticObservation]:
        frame_id = str(frame.get("frame_id", "")).strip()
        visible_objects = [str(item).strip().lower() for item in frame.get("visible_objects", []) if str(item).strip()]
        if not frame_id or not visible_objects:
            return []
        grouped: dict[str, list[str]] = {}
        for object_label in visible_objects:
            label = OBJECT_SEMANTIC_LABEL_HINTS.get(object_label)
            if label is None:
                continue
            grouped.setdefault(label, []).append(object_label)
        observations: list[SemanticObservation] = []
        for label, objects in grouped.items():
            self._observation_index += 1
            regions = tuple(
                PixelRegion(
                    frame_id=frame_id,
                    bbox_xyxy=(120, 120, 520, 380),
                    center_uv=(320, 240),
                    depth_m=None,
                    image_position="center",
                    object_label=object_label,
                    description=f"{object_label} visible in simulated RGB-D scan",
                )
                for object_label in objects[:4]
            )
            observations.append(
                SemanticObservation(
                    observation_id=deterministic_semantic_id("sem_obs", self._observation_index),
                    frame_id=frame_id,
                    label_hint=label,
                    confidence=min(0.55 + 0.08 * len(objects), 0.86),
                    pixel_regions=regions,
                    visual_cues=tuple(f"{item} visible" for item in objects[:6]),
                    reasoning_summary=f"{', '.join(objects[:4])} support a {label} place label.",
                )
            )
        return observations

    def _vlm_observations_from_frame(
        self,
        frame: dict[str, Any],
        trace: dict[str, Any],
    ) -> tuple[list[SemanticObservation], dict[str, Any]]:
        assert self.router is not None
        payload = {
            "frame_id": frame.get("frame_id"),
            "camera_pose": frame.get("pose"),
            "rgbd_summary": frame.get("rgbd_summary", {}),
            "description": frame.get("description"),
            "prior_semantic_memory": {
                "named_places": self.memory.snapshot().get("named_places", [])[-8:],
                "clusters": self.memory.snapshot().get("clusters", [])[-12:],
            },
            "rgb_image": frame.get("thumbnail_data_url"),
        }
        user_prompt = build_semantic_evidence_extraction_user_prompt(payload)
        messages = [
            {"role": "system", "content": build_semantic_evidence_extraction_system_prompt()},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": frame["thumbnail_data_url"]}},
                ],
            },
        ]
        parsed, llm_trace = self.router.complete_json_messages(
            config=self.router.model_suite.planner,
            messages=messages,
        )
        trace["source"] = "semantic_vlm"
        trace["llm_trace"] = {
            "provider": llm_trace.provider,
            "model": llm_trace.model,
            "duration_s": llm_trace.duration_s,
            "error": llm_trace.error,
            "response_text": llm_trace.response_text[:1000],
        }
        if parsed is None:
            trace["warnings"].append("semantic VLM returned no parseable JSON")
            return [], trace
        observations, warnings = parse_semantic_observation_payload(
            parsed,
            fallback_frame_id=str(frame.get("frame_id", "")),
            id_start=self._observation_index + 1,
        )
        self._observation_index += len(observations)
        trace["warnings"].extend(warnings)
        return observations, trace

    def _vlm_observations_from_frames(
        self,
        frames: list[dict[str, Any]],
        trace: dict[str, Any],
    ) -> tuple[list[SemanticObservation], dict[str, Any]]:
        assert self.router is not None
        payload = {
            "frame_id": f"spin_{frames[0].get('frame_id', 'start')}_to_{frames[-1].get('frame_id', 'end')}",
            "frames": [
                {
                    "frame_id": frame.get("frame_id"),
                    "camera_pose": frame.get("pose"),
                    "rgbd_summary": frame.get("rgbd_summary", {}),
                    "description": frame.get("description"),
                    "rgb_image": frame.get("thumbnail_data_url"),
                }
                for frame in frames
            ],
            "instruction": "Analyze these images as one completed spin. Return one JSON object for the whole spin.",
            "prior_semantic_memory": {
                "named_places": self.memory.snapshot().get("named_places", [])[-8:],
                "clusters": self.memory.snapshot().get("clusters", [])[-12:],
            },
        }
        user_prompt = build_semantic_evidence_extraction_user_prompt(payload)
        image_items = [
            {"type": "image_url", "image_url": {"url": frame["thumbnail_data_url"]}}
            for frame in frames
            if frame.get("thumbnail_data_url")
        ]
        messages = [
            {"role": "system", "content": build_semantic_evidence_extraction_system_prompt()},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}, *image_items]},
        ]
        parsed, llm_trace = self.router.complete_json_messages(
            config=self.router.model_suite.planner,
            messages=messages,
        )
        trace["source"] = "semantic_vlm_spin_batch"
        trace["llm_trace"] = {
            "provider": llm_trace.provider,
            "model": llm_trace.model,
            "duration_s": llm_trace.duration_s,
            "error": llm_trace.error,
            "response_text": llm_trace.response_text[:1000],
        }
        if parsed is None:
            trace["warnings"].append("semantic VLM returned no parseable JSON for spin batch")
            return [], trace
        observations, warnings = parse_semantic_observation_payload(
            parsed,
            fallback_frame_id=str(payload["frame_id"]),
            id_start=self._observation_index + 1,
        )
        self._observation_index += len(observations)
        trace["warnings"].extend(warnings)
        return observations, trace

    def _project_observation(
        self,
        *,
        observation: SemanticObservation,
        frame: dict[str, Any],
        depth_image: Any,
        intrinsics: CameraIntrinsics | None,
    ) -> Pose2D | None:
        object_pose = self._scenario_object_pose(observation)
        if object_pose is not None:
            return object_pose
        frame_pose = _pose_from_mapping(frame.get("pose", {}))
        if frame_pose is None:
            return None
        if depth_image is not None and intrinsics is not None:
            for region in observation.pixel_regions:
                pose = project_pixel_region_to_map(
                    pixel_region=region,
                    depth_image=depth_image,
                    intrinsics=intrinsics,
                    camera_pose=frame_pose,
                    fallback_depth_m=_frame_fallback_depth_m(frame),
                )
                if pose is not None:
                    return pose
        first_region = observation.pixel_regions[0] if observation.pixel_regions else None
        if first_region is None or intrinsics is None:
            depth_m = _frame_fallback_depth_m(frame)
            if depth_m is None:
                return None
            return Pose2D(
                frame_pose.x + math.cos(frame_pose.yaw) * depth_m,
                frame_pose.y + math.sin(frame_pose.yaw) * depth_m,
                0.0,
            )
        u, v = representative_pixel_for_image_position(first_region.image_position, intrinsics)
        depth_m = _frame_fallback_depth_m(frame) or 2.0
        return project_pixel_to_map(u=u, v=v, depth_m=depth_m, intrinsics=intrinsics, camera_pose=frame_pose)

    def _scenario_object_pose(self, observation: SemanticObservation) -> Pose2D | None:
        if self.scenario is None:
            return None
        labels = {
            str(region.object_label).lower()
            for region in observation.pixel_regions
            if region.object_label
        }
        matches = [item for item in self.scenario.objects if item.label.lower() in labels]
        if not matches:
            return None
        x = sum((item.cell.x + 0.5) * self.scenario.resolution for item in matches) / len(matches)
        y = sum((item.cell.y + 0.5) * self.scenario.resolution for item in matches) / len(matches)
        return Pose2D(x, y, 0.0)


class _ApartmentExplorationSession:
    def __init__(self, config: SimExplorationConfig, backend: ExplorationBackend, task_id: str) -> None:
        self.config = config
        self.backend = backend
        self.task_id = task_id
        self.scenario = _build_simple_apartment(config.occupancy_resolution)
        self.policy = ExplorationLLMPolicy(config)
        self.frontier_memory = FrontierMemory(config.occupancy_resolution)
        self.semantic_observer = SemanticWaypointObserver(config, scenario=self.scenario)
        self.current_cell = self.scenario.start_cell
        self.current_yaw = 0.0
        self.known_cells: dict[GridCell, str] = {}
        self.occupancy_evidence: dict[GridCell, float] = {}
        self.manual_occupancy_edits = edits_from_payload({}, cell_type=GridCell)
        self.range_edge_cells: set[GridCell] = set()
        self.trajectory: list[dict[str, Any]] = [self.current_cell.center_pose(self.scenario.resolution).to_dict()]
        self.keyframes: list[dict[str, Any]] = []
        self.room_frames: dict[str, list[str]] = {}
        self.room_objects_seen: dict[str, set[str]] = {room_id: set() for room_id in self.scenario.rooms}
        self.room_descriptions: dict[str, list[str]] = {room_id: [] for room_id in self.scenario.rooms}
        self.decision_log: list[dict[str, Any]] = []
        self.guardrail_events: list[dict[str, Any]] = []
        self.semantic_updates: list[dict[str, Any]] = []
        self.scan_known_cells: dict[GridCell, str] = {}
        self.scan_occupancy_evidence: dict[GridCell, float] = {}
        self.scan_range_edge_cells: set[GridCell] = set()
        self.scan_observation_index = 0
        self.scan_map_resolution = config.occupancy_resolution
        self.total_distance_m = 0.0
        self.control_steps = 0
        self.decision_index = 0
        self.nav2_goal_counter = 0
        self.nav2 = SimulatedNav2NavigationModule(
            config,
            self.scenario,
            get_current_cell=lambda: self.current_cell,
            get_current_yaw=lambda: self.current_yaw,
            known_free_cells=self._known_free_cells,
            on_motion_step=self._on_nav2_motion_step,
            on_runtime_obstacle=self._on_nav2_runtime_obstacle,
            is_runtime_blocked=self._is_runtime_blocked,
            budget_exhausted=self._budget_exhausted,
            should_stop_execution=self._should_stop_execution,
        )

    def run(self) -> dict[str, Any]:
        start_pose = self.current_cell.center_pose(self.scenario.resolution)
        self.frontier_memory.remember_return_waypoint(
            room_id=self.scenario.room_for_cell(self.current_cell),
            pose=start_pose,
            step_index=0,
            reason="initial_pose",
        )
        self._perform_scan(
            full_turnaround=True,
            capture_frame=True,
            reason="initial_turnaround_scan",
        )
        self._publish_live_map("Initial scan complete.")

        while self.decision_index < self.config.max_decisions:
            if not self._wait_until_task_active():
                break
            if self._budget_exhausted():
                self.guardrail_events.append(
                    {
                        "type": "budget_exhausted",
                        "control_steps": self.control_steps,
                        "decision_index": self.decision_index,
                    }
                )
                break

            self.decision_index += 1
            self._sync_manual_occupancy_edits()
            visible_candidates = self._detect_frontier_candidates()
            self.frontier_memory.upsert_candidates(visible_candidates, step_index=self.decision_index)
            candidate_records = self._refresh_candidate_paths()
            prompt_payload = self._build_prompt_payload(candidate_records)
            coverage = self._coverage()
            if not self._wait_until_task_active():
                break
            decision, trace = self.policy.decide(
                prompt_payload=prompt_payload,
                frontiers=candidate_records,
                return_waypoints=list(self.frontier_memory.return_waypoints.values()),
                coverage=coverage,
                current_room_id=self.scenario.room_for_cell(self.current_cell),
            )
            decision = self._apply_finish_guardrail(decision, candidate_records)
            applied_memory_updates = self.frontier_memory.apply_model_memory_updates(
                decision.memory_updates,
                selected_frontier_id=decision.selected_frontier_id,
            )
            trace["applied_memory_updates"] = applied_memory_updates
            self.semantic_updates.extend(decision.semantic_updates)
            self._log_policy_step(prompt_payload, decision, trace)
            if decision.decision_type == "finish" or decision.exploration_complete and not candidate_records:
                break

            if decision.selected_return_waypoint_id:
                waypoint = self.frontier_memory.get_return_waypoint(decision.selected_return_waypoint_id)
                if waypoint is not None:
                    target_pose = waypoint["pose"]
                    return_goal = self._make_nav2_goal(
                        Pose2D(float(target_pose["x"]), float(target_pose["y"]), float(target_pose.get("yaw", 0.0))),
                        goal_type="return_waypoint",
                        reason=f"return_waypoint::{waypoint['waypoint_id']}",
                    )
                    return_result = self.nav2.navigate_to_pose(return_goal)
                    if return_result.status != "succeeded":
                        self.guardrail_events.append(
                            {
                                "type": "return_waypoint_failed",
                                "waypoint_id": waypoint["waypoint_id"],
                                "nav2_result": return_result.to_dict(),
                            }
                        )

            if not decision.selected_frontier_id:
                break

            record = self.frontier_memory.activate(decision.selected_frontier_id)
            if record is None:
                self.guardrail_events.append(
                    {
                        "type": "missing_frontier",
                        "frontier_id": decision.selected_frontier_id,
                    }
                )
                continue

            frontier_goal = self._make_nav2_goal(
                record.nav_pose,
                goal_type="frontier",
                reason=f"frontier::{record.frontier_id}",
            )
            nav_result = self.nav2.navigate_to_pose(frontier_goal)
            if nav_result.status != "succeeded":
                _mark_frontier_unreachable_as_visited(self.frontier_memory, record.frontier_id, nav_result.reason)
                self.guardrail_events.append(
                    {
                        "type": "nav2_frontier_marked_visited_after_failure",
                        "frontier_id": record.frontier_id,
                        "nav2_result": nav_result.to_dict(),
                    }
                )
                self._push_progress_update(
                    message=f"Marked {record.frontier_id} visited after Nav2 failed to reach it: {nav_result.reason}",
                    frontier_id=record.frontier_id,
                )
                continue

            self._perform_scan(
                full_turnaround=True,
                capture_frame=True,
                reason=f"arrive_frontier::{record.frontier_id}",
            )
            self.frontier_memory.complete(record.frontier_id)
            self.frontier_memory.remember_return_waypoint(
                room_id=self.scenario.room_for_cell(self.current_cell),
                pose=self.current_cell.center_pose(self.scenario.resolution, yaw=self.current_yaw),
                step_index=self.decision_index,
                reason=f"completed_frontier::{record.frontier_id}",
            )
            self._push_progress_update(
                message=f"Explored {record.frontier_id} from room `{self.scenario.room_for_cell(self.current_cell)}`.",
                frontier_id=record.frontier_id,
            )

        return self._build_map_payload()

    def _wait_until_task_active(self) -> bool:
        while True:
            task = self.backend.get_task(self.task_id)
            if task is None:
                return False
            if task.get("state") == "aborted":
                return False
            if not bool(task.get("paused", False)):
                self._sync_manual_occupancy_edits()
                return True
            time.sleep(0.1)

    def _sync_manual_occupancy_edits(self) -> None:
        self.manual_occupancy_edits = edits_from_payload(
            self.backend.occupancy_edit_snapshot(self.task_id),
            cell_type=GridCell,
        )

    def perform_manual_scan(self) -> dict[str, Any]:
        self._sync_manual_occupancy_edits()
        self._perform_scan(
            full_turnaround=True,
            capture_frame=True,
            reason="operator_manual_turnaround_scan",
        )
        self._publish_live_map("Manual 360 degree scan complete.")
        return {
            "status": "succeeded",
            "message": "Manual 360 degree scan complete.",
            "map": self._build_map_payload(),
        }

    def _cell_state(self, cell: GridCell) -> str | None:
        return self.manual_occupancy_edits.state_for_cell(self.known_cells.get(cell), cell)

    def _effective_known_cells(self) -> dict[GridCell, str]:
        return overlay_known_cells(self.known_cells, self.manual_occupancy_edits)

    def _is_runtime_blocked(self, cell: GridCell) -> bool:
        return self._cell_state(cell) == "occupied" or self.scenario.is_occupied(cell)

    def _publish_live_map(self, message: str) -> None:
        self.backend.update_external_task(
            self.task_id,
            progress=min(self._coverage(), 0.98),
            message=message,
            result={
                "coverage": round(self._coverage(), 3),
                "trajectory": self.trajectory[-12:],
                "keyframes": self.keyframes[-4:],
                "frontier_memory": self.frontier_memory.snapshot(),
                "active_frontier_id": self.frontier_memory.active_frontier_id,
            },
            map_payload=self._build_map_payload(),
        )

    def _should_stop_execution(self) -> bool:
        task = self.backend.get_task(self.task_id)
        if task is None:
            return True
        return bool(task.get("state") == "aborted" or task.get("canceled", False))

    def _budget_exhausted(self) -> bool:
        if self.config.max_control_steps is not None and self.control_steps >= self.config.max_control_steps:
            return True
        if self.config.max_episode_steps is not None and self.decision_index >= self.config.max_episode_steps:
            return True
        return False

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
            if self.config.automatic_semantic_waypoints:
                self.semantic_observer.observe_keyframe(
                    frame=frame,
                    known_cells=self._effective_known_cells(),
                    robot_cell=self.current_cell,
                    resolution=self.scenario.resolution,
                )
            if current_room_id:
                self.room_frames.setdefault(current_room_id, []).append(frame_id)
        self._sleep()

    def _known_free_cells(self) -> set[GridCell]:
        effective = self._effective_known_cells()
        return {cell for cell, state in effective.items() if state == "free"}

    def _global_frontier_anchor_cell_near_record(
        self,
        record: FrontierRecord,
    ) -> tuple[GridCell | None, str | None]:
        effective_known = self._effective_known_cells()
        reachable_free_cells = self._reachable_known_free_cells(self.current_cell)
        boundary_cell = self.scenario.world_to_cell(record.centroid_pose.x, record.centroid_pose.y)
        search_radius_cells = max(1, int(math.ceil(STORED_FRONTIER_REVALIDATION_RADIUS_M / self.scenario.resolution)))
        strong_candidates: list[tuple[int, int, GridCell]] = []
        relaxed_candidates: list[tuple[int, int, GridCell]] = []
        unreachable_boundary_candidates = 0
        for dx in range(-search_radius_cells, search_radius_cells + 1):
            for dy in range(-search_radius_cells, search_radius_cells + 1):
                cell = GridCell(boundary_cell.x + dx, boundary_cell.y + dy)
                if not self.scenario.in_bounds(cell):
                    continue
                distance_cells = _grid_distance_cells(cell, boundary_cell)
                if distance_cells > search_radius_cells:
                    continue
                if effective_known.get(cell) != "free":
                    continue
                unknown_neighbors = {
                    neighbor
                    for neighbor in _neighbors4(cell)
                    if self.scenario.in_bounds(neighbor) and neighbor not in effective_known
                }
                if not unknown_neighbors:
                    continue
                if cell not in reachable_free_cells:
                    unreachable_boundary_candidates += 1
                    continue
                if len(unknown_neighbors) >= 2 or (cell in self.range_edge_cells and unknown_neighbors):
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

    def _revalidate_stored_frontier_boundary(
        self,
        record: FrontierRecord,
        anchor_cell: GridCell,
        anchor_mode: str | None,
    ) -> None:
        anchor_pose = anchor_cell.center_pose(self.scenario.resolution)
        if _pose_distance_m(record.centroid_pose, anchor_pose) <= self.scenario.resolution * 0.5 and anchor_mode != "relaxed":
            return
        previous_pose = record.centroid_pose
        record.centroid_pose = anchor_pose
        notes = [
            (
                "stored frontier boundary was revalidated against the current global occupancy map "
                f"near the original memory point ({previous_pose.x:.2f}, {previous_pose.y:.2f})"
            )
        ]
        if anchor_mode == "relaxed":
            notes.append(
                "stored frontier memory was kept using relaxed revalidation because nearby free space still borders unknown map area"
            )
        record.evidence = _dedupe_text(record.evidence + notes)

    def _resnap_stored_frontier_revisit_pose(
        self,
        record: FrontierRecord,
        current_pose: Pose2D,
        anchor_cell: GridCell,
    ) -> Pose2D | None:
        reachable_safe_cells = self._known_free_cells()
        resolution = self.scenario.resolution
        anchor_pose = anchor_cell.center_pose(resolution)
        max_radius_cells = max(1, int(math.ceil(STORED_FRONTIER_REVISIT_APPROACH_RADIUS_M / resolution)))
        scored_cells: list[tuple[float, GridCell]] = []
        for cell in reachable_safe_cells:
            distance_cells = _grid_distance_cells(cell, anchor_cell)
            if distance_cells > max_radius_cells:
                continue
            cell_pose = cell.center_pose(resolution)
            score = (
                abs(distance_cells * resolution - (self.config.robot_radius_m + 0.25))
                + 0.03 * _pose_distance_m(cell_pose, current_pose)
                + 0.02 * _pose_distance_m(cell_pose, record.nav_pose)
            )
            scored_cells.append((score, cell))
        if not scored_cells:
            return None
        best = min(scored_cells, key=lambda item: item[0])[1]
        best_pose = best.center_pose(
            resolution,
            yaw=math.atan2(anchor_pose.y - best.center_pose(resolution).y, anchor_pose.x - best.center_pose(resolution).x),
        )
        return best_pose

    def _apply_stored_frontier_resnap(
        self,
        record: FrontierRecord,
        target_pose: Pose2D,
        previous_pose: Pose2D,
    ) -> None:
        if _pose_distance_m(target_pose, previous_pose) <= self.scenario.resolution * 0.5:
            return
        record.nav_pose = target_pose
        record.evidence = _dedupe_text(
            record.evidence
            + [
                "stored frontier revisit approach pose was re-snapped to nearby robot-connected free space before LLM selection"
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

    def _reachable_known_free_cells(self, start_cell: GridCell) -> set[GridCell]:
        effective_known = self._effective_known_cells()
        if effective_known.get(start_cell) != "free":
            return set()
        reachable: set[GridCell] = set()
        queue = deque([start_cell])
        while queue:
            current = queue.popleft()
            if current in reachable:
                continue
            if not self.scenario.in_bounds(current) or effective_known.get(current) != "free":
                continue
            reachable.add(current)
            for neighbor in _neighbors4(current):
                if neighbor not in reachable and self.scenario.in_bounds(neighbor) and effective_known.get(neighbor) == "free":
                    queue.append(neighbor)
        return reachable

    def _min_frontier_opening_width_m(self) -> float:
        configured = self.config.frontier_min_opening_m
        if configured is not None and configured > 0.0:
            return max(float(configured), self.scenario.resolution)
        return max(self.config.robot_radius_m * 2.0 + 0.10, self.scenario.resolution * 2.0)

    def _detect_frontier_candidates(self) -> list[FrontierCandidate]:
        frontier_cells: set[GridCell] = set()
        unknown_neighbors_by_frontier: dict[GridCell, set[GridCell]] = {}
        known_free = self._known_free_cells()
        effective_known = self._effective_known_cells()
        reachable_free_cells = self._reachable_known_free_cells(self.current_cell)
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
            queue = deque([cell])
            visited.add(cell)
            while queue:
                current = queue.popleft()
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
            if not any(cell in reachable_free_cells for cell in cluster):
                self.guardrail_events.append(
                    {
                        "type": "frontier_boundary_unreachable_through_known_free",
                        "cluster_size": len(cluster),
                        "unknown_gain": len(cluster_unknown),
                        "frontier_boundary_pose": _cell_mean_pose(cluster, self.scenario.resolution).to_dict(),
                    }
                )
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
            nav_cell = min(
                cluster,
                key=lambda cell: _grid_distance_cells(cell, self.current_cell),
            )
            centroid_cell = _centroid_cell(cluster)
            room_hint = self.scenario.room_for_cell(nav_cell)
            evidence = [
                f"{len(cluster_unknown)} unknown neighbor cells",
                f"cluster size {len(cluster)}",
                f"frontier opening width is {opening_width_m:.2f} m, above robot-sized threshold {required_opening_m:.2f} m",
            ]
            if max_unknown_neighbor_count >= 2:
                evidence.append("frontier signal is stronger: multiple unknown-facing neighbors support likely expansion")
            else:
                evidence.append(
                    "frontier signal is weaker: a single unknown-facing edge can still be useful, but it should be vetoed if view/map context shows no meaningful navigable opening"
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

    def _refresh_candidate_paths(self) -> list[FrontierRecord]:
        current_pose = self.current_cell.center_pose(self.scenario.resolution, yaw=self.current_yaw)
        current_pose_filter_m = max(self.config.occupancy_resolution * 1.5, self.config.robot_radius_m)

        def _path_cost(record: FrontierRecord) -> float | None:
            preview_goal = Nav2GoalRequest(
                goal_id=f"preview_{self.decision_index:03d}_{record.frontier_id}",
                goal_type="frontier_preview",
                target_pose=record.nav_pose,
                planner_id=self.config.nav2_planner_id,
                controller_id=self.config.nav2_controller_id,
                behavior_tree=self.config.nav2_behavior_tree,
                metadata={"frontier_id": record.frontier_id},
            )
            plan = self.nav2.compute_path(preview_goal, record=False)
            return plan.path_length_m if plan.status == "succeeded" else None

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
            global_anchor_for_stored_record=self._global_frontier_anchor_cell_near_record,
            revalidate_stored_boundary=self._revalidate_stored_frontier_boundary,
            resnap_stored_nav_pose=self._resnap_stored_frontier_revisit_pose,
            apply_stored_resnap=self._apply_stored_frontier_resnap,
        )

    def _build_prompt_payload(self, candidate_records: list[FrontierRecord]) -> dict[str, Any]:
        recent_views = self.keyframes[-3:]
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
        frontier_memory_snapshot = self.frontier_memory.snapshot()
        return {
            "mission": (
                "Explore the apartment until the accessible map is complete, keep frontier memory stable, "
                "and prefer fast coverage growth over redundant local motion."
            ),
            "robot": {
                "pose": self.current_cell.center_pose(self.scenario.resolution, yaw=self.current_yaw).to_dict(),
                "room_id": self.scenario.room_for_cell(self.current_cell),
                "coverage": round(self._coverage(), 3),
                "trajectory_points": len(self.trajectory),
                "sensor_range_m": self.config.sensor_range_m,
            },
            "frontier_memory": frontier_memory_snapshot,
            "frontier_information": [record.to_dict() for record in candidate_records],
            "candidate_frontiers": [record.to_dict() for record in candidate_records],
            "explored_areas": explored_areas,
            "recent_views": recent_views,
            "frontier_selection_guidance": _frontier_selection_guidance(),
            "guardrails": {
                "finish_requires_frontier_exhaustion": True,
                "finish_coverage_threshold": self.config.finish_coverage_threshold,
                "navigation_must_use_nav2_goal_validation": True,
                "frontier_ids_must_come_from_prompt": True,
                "frontier_information_is_boundary_evidence_not_a_command": True,
                "select_regions_that_expand_robot_navigable_space": True,
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
                    row.append("V" if frontier_cells[cell] in {"completed", "failed"} else "F")
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

    def _apply_finish_guardrail(self, decision: ExplorationDecision, candidate_records: list[FrontierRecord]) -> ExplorationDecision:
        reachable = [record for record in candidate_records if record.path_cost_m is not None]
        if decision.decision_type == "finish" and reachable and self._coverage() < self.config.finish_coverage_threshold:
            fallback = self.policy._heuristic_decision(  # intentional guardrail fallback
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

    def _log_policy_step(self, prompt_payload: dict[str, Any], decision: ExplorationDecision, trace: dict[str, Any]) -> None:
        if self.config.trace_policy_stdout:
            print(
                f"[exploration-policy] step={self.decision_index} coverage={self._coverage():.3f} "
                f"decision={decision.decision_type} frontier={decision.selected_frontier_id}"
            )
        if self.config.trace_llm_stdout:
            print("[exploration-llm] prompt")
            print(trace.get("prompt", ""))
            print("[exploration-llm] response")
            print(json.dumps(trace.get("response", {}), indent=2, sort_keys=True))
            if trace.get("llm_trace", {}).get("error"):
                print(f"[exploration-llm] error={trace['llm_trace']['error']}")
        self.decision_log.append(
            {
                "step_index": self.decision_index,
                "coverage": round(self._coverage(), 3),
                "decision": decision.to_dict(),
                "trace": trace,
                "frontier_information_ids": [
                    item.get("frontier_id")
                    for item in prompt_payload.get("frontier_information", prompt_payload.get("candidate_frontiers", []))
                ],
            }
        )

    def _push_progress_update(self, *, message: str, frontier_id: str | None) -> None:
        coverage = self._coverage()
        result = {
            "coverage": round(coverage, 3),
            "trajectory": self.trajectory[-12:],
            "keyframes": self.keyframes[-4:],
            "frontier_memory": self.frontier_memory.snapshot(),
            "active_frontier_id": frontier_id,
        }
        self.backend.update_external_task(
            self.task_id,
            progress=min(coverage, 0.98),
            message=message,
            result=result,
            map_payload=self._build_map_payload(),
        )

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
        semantic_memory = self.semantic_observer.snapshot() if self.config.automatic_semantic_waypoints else {}
        occupancy_cells = []
        for cell, state in sorted(self._effective_known_cells().items()):
            item = {
                "x": round(cell.x * self.scenario.resolution, 3),
                "y": round(cell.y * self.scenario.resolution, 3),
                "state": state,
            }
            if cell in self.manual_occupancy_edits.blocked_cells:
                item["manual_override"] = "blocked"
            elif cell in self.manual_occupancy_edits.cleared_cells:
                item["manual_override"] = "cleared"
            occupancy_cells.append(item)

        regions = []
        semantic_area_candidates: list[dict[str, Any]] = []
        for room_id, room in self.scenario.rooms.items():
            coverage = self._room_coverage(room_id)
            if coverage <= 0.12 and not self.room_objects_seen.get(room_id):
                continue
            evidence = [f"{item} visible" for item in sorted(self.room_objects_seen.get(room_id, set()))]
            if not evidence:
                evidence = ["geometry observed", "frontier traversal reached this room"]
            confidence = min(0.55 + coverage * 0.35 + min(len(evidence), 4) * 0.03, 0.98)
            regions.append(
                {
                    "region_id": room.region_id,
                    "label": room.label,
                    "confidence": round(confidence, 3),
                    "polygon_2d": [[float(x), float(y)] for x, y in room.polygon_2d],
                    "centroid": {"x": room.center_pose.x, "y": room.center_pose.y},
                    "adjacency": list(room.adjacency),
                    "representative_keyframes": self.room_frames.get(room_id, [])[:3],
                    "evidence": evidence[:8],
                    "default_waypoints": [
                        {
                            "name": f"{room.label}_center",
                            **room.center_pose.to_dict(),
                        },
                        {
                            "name": f"{room.label}_door",
                            **room.entry_pose.to_dict(),
                        },
                    ],
                }
            )
            for sub_area in room.sub_areas:
                if all(token.split()[0] in " ".join(evidence).lower() for token in sub_area.evidence):
                    semantic_area_candidates.append(
                        {
                            "area_id": sub_area.area_id,
                            "label": sub_area.label,
                            "parent_region_id": room.region_id,
                            "polygon_2d": [[float(x), float(y)] for x, y in sub_area.polygon_2d],
                            "confidence": 0.74,
                            "evidence": list(sub_area.evidence),
                        }
                    )

        if self.config.experimental_free_space_semantic_waypoints:
            aggregated_semantics = _aggregate_semantic_updates(self.semantic_updates)
            semantic_area_candidates.extend(aggregated_semantics)

        summary = (
            f"Agentic apartment exploration completed with coverage {self._coverage():.3f}, "
            f"{len(regions)} mapped regions, and {len(self.decision_log)} exploration decisions."
        )
        return {
            "map_id": self.config.session,
            "frame": "map",
            "resolution": float(self.config.occupancy_resolution),
            "coverage": round(self._coverage(), 3),
            "summary": summary,
            "approved": False,
            "created_at": time.time(),
            "source": self.config.source,
            "mode": "sim_agentic",
            "trajectory": self.trajectory,
            "keyframes": self.keyframes,
            "regions": regions,
            "named_places": _semantic_named_places_for_map(semantic_memory) if self.config.automatic_semantic_waypoints else [],
            "occupancy": {
                "resolution": float(self.config.occupancy_resolution),
                "bounds": self.scenario.bounds(),
                "cells": occupancy_cells,
            },
            "frontiers": [record.to_dict() for record in self.frontier_memory.records.values()],
            "semantic_area_candidates": semantic_area_candidates,
            "semantic_memory": semantic_memory,
            "automatic_semantic_waypoints": self.config.automatic_semantic_waypoints,
            "artifacts": {
                "layout_id": self.scenario.layout_id,
                "decision_log": self.decision_log,
                "frontier_memory": self.frontier_memory.snapshot(),
                "guardrail_events": self.guardrail_events,
                "manual_occupancy_edits": self.manual_occupancy_edits.to_dict(
                    resolution=self.scenario.resolution,
                    origin_x=self.scenario.bounds()["min_x"],
                    origin_y=self.scenario.bounds()["min_y"],
                ),
                "navigation": {
                    "control_steps": self.control_steps,
                    "total_distance_m": round(self.total_distance_m, 3),
                },
                "nav2": self.nav2.snapshot(),
                "llm_policy": {
                    "explorer_policy": self.config.explorer_policy,
                    "provider": self.config.llm_provider,
                    "model": self.config.llm_model,
                },
            },
        }

    def _sleep(self) -> None:
        if self.config.realtime_sleep_s > 0:
            time.sleep(self.config.realtime_sleep_s)

    def _make_nav2_goal(self, pose: Pose2D, *, goal_type: str, reason: str) -> Nav2GoalRequest:
        self.nav2_goal_counter += 1
        return Nav2GoalRequest(
            goal_id=f"nav2_goal_{self.nav2_goal_counter:03d}",
            goal_type=goal_type,
            target_pose=pose,
            planner_id=self.config.nav2_planner_id,
            controller_id=self.config.nav2_controller_id,
            behavior_tree=self.config.nav2_behavior_tree,
            metadata={"reason": reason},
        )

    def _on_nav2_motion_step(
        self,
        previous: GridCell,
        nxt: GridCell,
        goal: Nav2GoalRequest,
        step_index: int,
        total_waypoints: int,
    ) -> None:
        if not self._wait_until_task_active():
            return
        self.current_yaw = math.atan2(nxt.y - previous.y, nxt.x - previous.x)
        self.current_cell = nxt
        self.control_steps += 1
        self.total_distance_m += self.scenario.resolution
        self.trajectory.append(self.current_cell.center_pose(self.scenario.resolution, yaw=self.current_yaw).to_dict())
        capture_frame = self.config.show_cameras and self.control_steps % max(self.config.camera_log_stride, 1) == 0
        self._perform_scan(
            full_turnaround=False,
            capture_frame=capture_frame,
            reason=f"travel::{goal.goal_id}",
        )
        self._push_progress_update(
            message=(
                f"Nav2 executing {goal.goal_type} {goal.goal_id} "
                f"({step_index}/{max(total_waypoints - 1, 1)}). Coverage {self._coverage():.3f}."
            ),
            frontier_id=goal.metadata.get("reason"),
        )

    def _on_nav2_runtime_obstacle(self, cell: GridCell) -> None:
        merge_occupancy_observation(
            self.known_cells,
            cell,
            "occupied",
            evidence_scores=self.occupancy_evidence,
        )
        self._publish_live_map(f"Runtime obstacle observed at cell ({cell.x}, {cell.y}).")


class RosExplorationSession:
    def __init__(self, config: SimExplorationConfig, backend: ExplorationBackend, task_id: str) -> None:
        self.config = config
        self.backend = backend
        self.task_id = task_id
        self.policy = ExplorationLLMPolicy(config)
        self.frontier_memory = FrontierMemory(config.occupancy_resolution)
        self.semantic_observer = SemanticWaypointObserver(config, scenario=None)
        self.manual_occupancy_edits = edits_from_payload({}, cell_type=GridCell)
        self.keyframes: list[dict[str, Any]] = []
        self.trajectory: list[dict[str, Any]] = []
        self.decision_log: list[dict[str, Any]] = []
        self.guardrail_events: list[dict[str, Any]] = []
        self.semantic_updates: list[dict[str, Any]] = []
        self.total_distance_m = 0.0
        self.control_steps = 0
        self.decision_index = 0
        self.nav2_goal_counter = 0
        self.status = "not_started"
        self.pending_prompt_payload: dict[str, Any] | None = None
        self.pending_prompt_text: str | None = None
        self.pending_candidate_records: list[FrontierRecord] = []
        self.pending_decision: ExplorationDecision | None = None
        self.pending_trace: dict[str, Any] | None = None
        self.applied_memory_updates: list[dict[str, Any]] = []
        self.last_error: str | None = None
        self.last_alert: str | None = None
        self.last_alert_id = 0
        self.last_nav2_preview: dict[str, Any] | None = None
        self._navigation_only_map: RosOccupancyMap | None = None
        self._lock = threading.RLock()
        self._last_pose: Pose2D | None = None
        self._manual_scan_in_progress = False
        self._owns_rclpy = False
        if config.ros_adapter_url and config.ros_navigation_map_source == "fused_point_cloud":
            raise RuntimeError(
                "`fused_point_cloud` requires the local ROS runtime because the HTTP ROS adapter "
                "does not serialize large PointCloud2 observations. Run real_agentic_exploration without "
                "--ros-adapter-url, or use --ros-navigation-map-source fused_scan with the adapter."
            )
        if config.ros_adapter_url:
            self.runtime = RemoteRosExplorationRuntime(
                config.ros_adapter_url,
                timeout_s=config.ros_adapter_timeout_s,
                turn_scan_mode=config.ros_turn_scan_mode,
                robot_brain_url=config.robot_brain_url,
                camera_pan_action_key=config.camera_pan_action_key,
                camera_pan_settle_s=config.camera_pan_settle_s,
                camera_pan_step_deg=config.camera_pan_step_deg,
                camera_pan_compute_s=config.camera_pan_compute_s,
                camera_pan_sample_count=config.camera_pan_sample_count,
            )
        else:
            require_ros_nav2_runtime_dependencies()
            if not rclpy.ok():
                rclpy.init(args=None)
                self._owns_rclpy = True
            self.runtime = RosExplorationRuntime(
                RosRuntimeConfig(
                    map_topic=config.ros_map_topic,
                    map_updates_topic=config.ros_map_updates_topic,
                    scan_topic=config.ros_scan_topic,
                    point_cloud_topic=config.ros_point_cloud_topic,
                    scan_active_topic=config.ros_scan_active_topic,
                    nav_active_topic=config.ros_nav_active_topic,
                    scan_active_release_delay_s=config.ros_scan_active_release_delay_s,
                    rgb_topic=config.ros_rgb_topic,
                    imu_topic=config.ros_imu_topic,
                    cmd_vel_topic=config.ros_cmd_vel_topic,
                    map_frame=config.ros_map_frame,
                    odom_frame=config.ros_odom_frame,
                    base_frame=config.ros_base_frame,
                    server_timeout_s=config.ros_server_timeout_s,
                    ready_timeout_s=config.ros_ready_timeout_s,
                    turn_scan_radians=math.tau,
                    turn_scan_timeout_s=config.ros_turn_scan_timeout_s,
                    turn_scan_settle_s=config.ros_turn_scan_settle_s,
                    manual_spin_angular_speed_rad_s=config.ros_manual_spin_angular_speed_rad_s,
                    manual_spin_publish_hz=config.ros_manual_spin_publish_hz,
                    turn_scan_mode=config.ros_turn_scan_mode,
                    robot_brain_url=config.robot_brain_url,
                    camera_pan_action_key=config.camera_pan_action_key,
                    camera_pan_settle_s=config.camera_pan_settle_s,
                    camera_pan_step_deg=config.camera_pan_step_deg,
                    camera_pan_compute_s=config.camera_pan_compute_s,
                    camera_pan_sample_count=config.camera_pan_sample_count,
                    allow_multiple_action_servers=config.ros_allow_multiple_action_servers,
                    publish_internal_navigation_map=config.ros_navigation_map_source
                    in ("fused_scan", "fused_point_cloud"),
                    navigation_map_source=config.ros_navigation_map_source,
                    fuse_external_projected_map_snapshots=config.ros_fuse_external_projected_map_snapshots,
                )
            )
        self.scan_known_cells: dict[GridCell, str] = {}
        self.scan_occupancy_evidence: dict[GridCell, float] = {}
        self.scan_range_edge_cells: set[GridCell] = set()
        self.point_cloud_integration_summaries: list[dict[str, Any]] = []
        self._latest_fused_projected_map: RosOccupancyMap | None = None
        self.point_cloud_fusion_config = PointCloudFusionConfig(
            range_min_m=float(config.point_cloud_range_min_m),
            range_max_m=float(config.point_cloud_range_max_m),
            floor_free_max_z_m=float(config.point_cloud_floor_free_max_z_m),
            obstacle_min_z_m=float(config.point_cloud_obstacle_min_z_m),
            robot_clearance_height_m=float(config.point_cloud_robot_clearance_height_m),
            obstacle_max_z_m=float(config.point_cloud_obstacle_max_z_m),
            max_rays=int(config.point_cloud_max_rays),
        )
        if self.runtime.latest_map is not None:
            self.scan_map_resolution = float(self.runtime.latest_map.resolution)
        else:
            self.scan_map_resolution = config.occupancy_resolution
        self.scan_observation_index = self.runtime.scan_observation_count()
        self.point_cloud_observation_index = (
            self.runtime.point_cloud_observation_count()
            if hasattr(self.runtime, "point_cloud_observation_count")
            else 0
        )
        self._last_map_payload_log_s = 0.0
        self._last_live_pose_publish_s = 0.0
        self.nav2 = RosNav2NavigationModule(
            config,
            self.runtime,
            current_map=self._current_effective_map,
            should_cancel=self._pause_requested_or_canceled,
        )

    def close(self) -> None:
        try:
            self.runtime.close()
        finally:
            if self._owns_rclpy and rclpy.ok():
                rclpy.shutdown()

    def run(self) -> dict[str, Any]:
        self._initialize_scan_state()
        self._publish_live_map("Initial ROS/Nav2 scan complete.")
        if self.config.stop_after_initial_scan:
            self.backend.update_external_task(
                self.task_id,
                message="Initial 360 degree scan complete. Stopping before frontier navigation as requested.",
            )
            self.status = "finished"
            return self._build_map_payload()
        if not self._pause_for_operator_approval(
            "Initial 360 degree scan complete. Review the map and click Resume to continue."
        ):
            return self._build_map_payload()

        while self.decision_index < self.config.max_decisions:
            if not self._wait_until_task_active():
                break
            self.runtime.spin_for(0.15)
            if self._budget_exhausted():
                self.guardrail_events.append(
                    {
                        "type": "budget_exhausted",
                        "decision_index": self.decision_index,
                    }
                )
                break

            self.decision_index += 1
            self._sync_manual_occupancy_edits()
            occupancy_map = self._require_effective_map()
            pose = self._require_pose()
            self._update_pose_history(pose)
            visible_candidates = self._detect_frontier_candidates(occupancy_map, pose)
            self.frontier_memory.upsert_candidates(visible_candidates, step_index=self.decision_index)
            candidate_records = self._refresh_candidate_paths()
            prompt_payload = self._build_prompt_payload(occupancy_map, pose, candidate_records)
            coverage = self._coverage(occupancy_map)
            if not self._wait_until_task_active():
                break
            decision, trace = self.policy.decide(
                prompt_payload=prompt_payload,
                frontiers=candidate_records,
                return_waypoints=list(self.frontier_memory.return_waypoints.values()),
                coverage=coverage,
                current_room_id=None,
            )
            decision = self._apply_finish_guardrail(decision, candidate_records)
            applied_memory_updates = self.frontier_memory.apply_model_memory_updates(
                decision.memory_updates,
                selected_frontier_id=decision.selected_frontier_id,
            )
            trace["applied_memory_updates"] = applied_memory_updates
            self.semantic_updates.extend(decision.semantic_updates)
            self._log_policy_step(prompt_payload, decision, trace)
            if decision.decision_type == "finish" or (decision.exploration_complete and not candidate_records):
                break

            if decision.selected_return_waypoint_id:
                waypoint = self.frontier_memory.get_return_waypoint(decision.selected_return_waypoint_id)
                if waypoint is not None:
                    target_pose = waypoint["pose"]
                    return_goal = self._make_nav2_goal(
                        Pose2D(
                            float(target_pose["x"]),
                            float(target_pose["y"]),
                            float(target_pose.get("yaw", 0.0)),
                        ),
                        goal_type="return_waypoint",
                        reason=f"return_waypoint::{waypoint['waypoint_id']}",
                    )
                    return_result = self.nav2.navigate_to_pose(
                        return_goal,
                        feedback_callback=self._publish_navigation_feedback,
                    )
                    self._consume_nav_result(return_result)
                    if return_result.status != "succeeded":
                        self.guardrail_events.append(
                            {
                                "type": "return_waypoint_failed",
                                "waypoint_id": waypoint["waypoint_id"],
                                "nav2_result": return_result.to_dict(),
                            }
                        )

            if not decision.selected_frontier_id:
                break

            record = self.frontier_memory.activate(decision.selected_frontier_id)
            if record is None:
                self.guardrail_events.append(
                    {
                        "type": "missing_frontier",
                        "frontier_id": decision.selected_frontier_id,
                    }
                )
                continue

            frontier_goal = self._make_nav2_goal(
                record.nav_pose,
                goal_type="frontier",
                reason=f"frontier::{record.frontier_id}",
            )
            nav_result = self.nav2.navigate_to_pose(
                frontier_goal,
                feedback_callback=self._publish_navigation_feedback,
            )
            self._consume_nav_result(nav_result)
            if nav_result.status != "succeeded":
                _mark_frontier_unreachable_as_visited(self.frontier_memory, record.frontier_id, nav_result.reason)
                self.guardrail_events.append(
                    {
                        "type": "nav2_frontier_marked_visited_after_failure",
                        "frontier_id": record.frontier_id,
                        "nav2_result": nav_result.to_dict(),
                    }
                )
                self._push_progress_update(
                    message=f"Marked {record.frontier_id} visited after Nav2 failed to reach it: {nav_result.reason}",
                    frontier_id=record.frontier_id,
                )
                if not self._pause_for_operator_approval(
                    f"Step {self.decision_index} complete. {record.frontier_id} was marked visited after Nav2 failed. "
                    "Review the state and click Resume to continue."
                ):
                    break
                continue

            self._perform_turnaround_scan(reason=f"arrive_frontier::{record.frontier_id}")
            self.frontier_memory.complete(record.frontier_id)
            current_pose = self._require_pose()
            self.frontier_memory.remember_return_waypoint(
                room_id=None,
                pose=current_pose,
                step_index=self.decision_index,
                reason=f"completed_frontier::{record.frontier_id}",
            )
            self._push_progress_update(
                message=f"Explored {record.frontier_id} from live ROS/Nav2 state.",
                frontier_id=record.frontier_id,
            )
            if not self._pause_for_operator_approval(
                f"Step {self.decision_index} complete. Explored {record.frontier_id}. "
                "Review the state and click Resume to continue."
            ):
                break

        return self._build_map_payload()

    def reset(self) -> dict[str, Any]:
        with self._lock:
            self.frontier_memory = FrontierMemory(self.config.occupancy_resolution)
            self.semantic_observer = SemanticWaypointObserver(self.config, scenario=None)
            self.manual_occupancy_edits = edits_from_payload({}, cell_type=GridCell)
            self._navigation_only_map = None
            self.keyframes = []
            self.trajectory = []
            self.decision_log = []
            self.guardrail_events = []
            self.semantic_updates = []
            self.scan_known_cells = {}
            self.scan_occupancy_evidence = {}
            self.scan_range_edge_cells = set()
            self.point_cloud_integration_summaries = []
            self.scan_observation_index = self.runtime.scan_observation_count()
            self.point_cloud_observation_index = (
                self.runtime.point_cloud_observation_count()
                if hasattr(self.runtime, "point_cloud_observation_count")
                else 0
            )
            if self.runtime.latest_map is not None:
                self.scan_map_resolution = float(self.runtime.latest_map.resolution)
            else:
                self.scan_map_resolution = self.config.occupancy_resolution
            self._latest_fused_projected_map = None
            self.total_distance_m = 0.0
            self.control_steps = 0
            self.decision_index = 0
            self.nav2_goal_counter = 0
            self.pending_prompt_payload = None
            self.pending_prompt_text = None
            self.pending_candidate_records = []
            self.pending_decision = None
            self.pending_trace = None
            self.applied_memory_updates = []
            self.last_error = None
            self.last_alert: str | None = None
            self.last_alert_id = 0
            self._last_pose = None
            self.backend.resume_task(self.task_id)
            self._initialize_scan_state()
            self._prepare_decision_locked()
            return self.snapshot()

    def start_navigation_only(self, map_payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.last_error = None
            saved_map = _ros_map_from_environment_payload(map_payload)
            if saved_map is None:
                self.status = "navigation_only_failed"
                self.last_error = "Loaded environment does not contain an occupancy map."
                return {"status": "failed", "reason": self.last_error}
            self._navigation_only_map = saved_map
            self.scan_map_resolution = float(saved_map.resolution)
            edits = map_payload.get("artifacts", {}).get("manual_occupancy_edits")
            self.manual_occupancy_edits = edits_from_payload(
                edits if isinstance(edits, dict) else {},
                cell_type=GridCell,
            )
            start_pose = _start_pose_from_environment_payload(map_payload)
            initial_pose_result = None
            if start_pose is not None:
                self.status = "navigation_only_setting_initial_pose"
                initial_pose_result = self.runtime.set_initial_pose(start_pose)
            effective_map = EditableOccupancyMap(saved_map, self.manual_occupancy_edits)
            self.runtime.publish_navigation_map(
                _occupancy_map_like_to_ros_map(effective_map),
                map_to_odom=_pose_from_map_payload(initial_pose_result.get("map_to_odom"))
                if isinstance(initial_pose_result, dict)
                else None,
                force_publish=True,
            )
            self.runtime.spin_for(0.3)
            live_pose = self.runtime.current_pose()
            if live_pose is not None:
                self._update_pose_history(live_pose)
            self.status = "navigation_only_active"
            self.backend.update_external_task(
                self.task_id,
                message=(
                    "Navigation-only session active on the loaded environment. "
                    "Use Preview or Go to test manual Nav2 waypoints."
                ),
            )
            return {
                "status": "active",
                "mode": "navigation_only",
                "initial_pose": None if start_pose is None else start_pose.to_dict(),
                "initial_pose_result": initial_pose_result,
                "robot_pose": None if live_pose is None else live_pose.to_dict(),
                "map": map_payload,
            }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            pose = self.runtime.current_pose()
            if pose is not None:
                self._update_pose_history(pose)
            return {
                "status": self.status,
                "session": self.config.session,
                "coverage": round(self._coverage(self._require_effective_map()), 3),
                "robot_pose": (pose or Pose2D(0.0, 0.0, 0.0)).to_dict(),
                "prompt": self.pending_prompt_text,
                "prompt_payload": self.pending_prompt_payload,
                "candidate_frontiers": [record.to_dict() for record in self.pending_candidate_records],
                "pending_decision": None if self.pending_decision is None else self.pending_decision.to_dict(),
                "pending_trace": self.pending_trace,
                "pending_target": self._pending_target(),
                "applied_memory_updates": list(self.applied_memory_updates),
                "last_error": self.last_error,
                "last_alert": self.last_alert,
                "last_alert_id": self.last_alert_id,
                "map": self._build_map_payload(),
            }

    def call_llm(self) -> dict[str, Any]:
        with self._lock:
            self.last_error = None
            task = self.backend.get_task(self.task_id)
            if task and bool(task.get("paused", False)):
                self.last_error = "Exploration is paused. Resume before calling the LLM."
                return self.snapshot()
            if self.pending_prompt_payload is None:
                self._prepare_decision_locked()
            assert self.pending_prompt_payload is not None
            decision, trace = self.policy.decide(
                prompt_payload=self.pending_prompt_payload,
                frontiers=self.pending_candidate_records,
                return_waypoints=list(self.frontier_memory.return_waypoints.values()),
                coverage=self._coverage(self._require_effective_map()),
                current_room_id=None,
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
            self._push_progress_update(message="LLM frontier decision ready.", frontier_id=decision.selected_frontier_id)
            return self.snapshot()

    def apply_decision(self) -> dict[str, Any]:
        with self._lock:
            self.last_error = None
            task = self.backend.get_task(self.task_id)
            if task and bool(task.get("paused", False)):
                self.last_error = "Exploration is paused. Resume before applying a decision."
                return self.snapshot()
            if self.pending_decision is None:
                self.last_error = "No pending LLM decision. Click `Call LLM` first."
                return self.snapshot()
            decision = self.pending_decision
            if decision.decision_type == "finish" or decision.exploration_complete:
                self.status = "finished"
                self.pending_decision = None
                self._push_progress_update(message="LLM marked exploration finished.", frontier_id=None)
                return self.snapshot()

            if decision.selected_return_waypoint_id:
                waypoint = self.frontier_memory.get_return_waypoint(decision.selected_return_waypoint_id)
                if waypoint is not None:
                    target_pose = waypoint["pose"]
                    return_goal = self._make_nav2_goal(
                        Pose2D(
                            float(target_pose["x"]),
                            float(target_pose["y"]),
                            float(target_pose.get("yaw", 0.0)),
                        ),
                        goal_type="return_waypoint",
                        reason=f"return_waypoint::{waypoint['waypoint_id']}",
                    )
                    return_result = self.nav2.navigate_to_pose(
                        return_goal,
                        feedback_callback=self._publish_navigation_feedback,
                    )
                    self._consume_nav_result(return_result)
                    if return_result.status != "succeeded":
                        self.guardrail_events.append(
                            {
                                "type": "return_waypoint_failed",
                                "waypoint_id": waypoint["waypoint_id"],
                                "nav2_result": return_result.to_dict(),
                            }
                        )

            if not decision.selected_frontier_id:
                self.last_error = "The LLM decision did not select a frontier."
                return self.snapshot()

            record = self.frontier_memory.activate(decision.selected_frontier_id)
            if record is None:
                self.last_error = f"Selected frontier `{decision.selected_frontier_id}` no longer exists."
                return self.snapshot()

            self.status = "nav2_goal_active"
            self._push_progress_update(message=f"Sending Nav2 goal for {record.frontier_id}.", frontier_id=record.frontier_id)
            frontier_goal = self._make_nav2_goal(
                record.nav_pose,
                goal_type="frontier",
                reason=f"frontier::{record.frontier_id}",
            )
            nav_result = self.nav2.navigate_to_pose(
                frontier_goal,
                feedback_callback=self._publish_navigation_feedback,
            )
            self._consume_nav_result(nav_result)
            if nav_result.status != "succeeded":
                _mark_frontier_unreachable_as_visited(self.frontier_memory, record.frontier_id, nav_result.reason)
                self.guardrail_events.append(
                    {
                        "type": "nav2_frontier_marked_visited_after_failure",
                        "frontier_id": record.frontier_id,
                        "nav2_result": nav_result.to_dict(),
                    }
                )
                self.status = "nav2_goal_failed"
                self.pending_decision = None
                self._push_progress_update(
                    message=f"Marked {record.frontier_id} visited after Nav2 failed to reach it: {nav_result.reason}",
                    frontier_id=record.frontier_id,
                )
                self._prepare_decision_locked()
                return self.snapshot()

            self._perform_turnaround_scan(reason=f"arrive_frontier::{record.frontier_id}")
            self.frontier_memory.complete(record.frontier_id)
            current_pose = self._require_pose()
            self.frontier_memory.remember_return_waypoint(
                room_id=None,
                pose=current_pose,
                step_index=self.decision_index,
                reason=f"completed_frontier::{record.frontier_id}",
            )
            self.pending_decision = None
            self.pending_trace = None
            self.applied_memory_updates = []
            self._set_alert("Waypoint achieved.")
            self._push_progress_update(
                message=f"Explored {record.frontier_id} from live ROS/Nav2 state.",
                frontier_id=record.frontier_id,
            )
            self._prepare_decision_locked()
            return self.snapshot()

    def pause(self) -> dict[str, Any]:
        self.backend.pause_task(self.task_id)
        self.status = "paused"
        return self.snapshot()

    def resume(self) -> dict[str, Any]:
        self.backend.resume_task(self.task_id)
        if self.pending_decision is not None:
            self.status = "llm_response_ready"
        elif self.pending_prompt_payload is not None:
            self.status = "waiting_for_llm"
        return self.snapshot()

    def update_occupancy_edits(self, *, mode: str, cells: list[dict[str, Any]]) -> dict[str, Any]:
        with self._lock:
            self.backend.update_occupancy_edits(task_id=self.task_id, mode=mode, cells=cells)
            self._sync_manual_occupancy_edits()
            self._publish_navigation_map()
            self._prepare_decision_locked()
            return self.snapshot()

    def perform_manual_scan(self) -> dict[str, Any]:
        with self._lock:
            self.last_error = None
            if self._manual_scan_in_progress:
                return {"status": "busy", "reason": "A manual scan is already running."}
            if self._task_canceled_or_aborted():
                return {"status": "unavailable", "reason": "The exploration task is not active."}
            self._sync_manual_occupancy_edits()
            self.status = "manual_scan_active"
            self._manual_scan_in_progress = True
            try:
                self._push_progress_update(message="Running operator-requested 360 degree scan.", frontier_id=None)
            except Exception as exc:
                self._manual_scan_in_progress = False
                self.status = "manual_scan_failed"
                self.last_error = f"Manual scan failed: {exc}"
                return {"status": "failed", "reason": str(exc), "map": None}
        try:
            self._perform_turnaround_scan(reason="operator_manual_turnaround_scan", ignore_pause=True)
            with self._lock:
                self._sync_manual_occupancy_edits()
                self.pending_decision = None
                self.pending_trace = None
                self.applied_memory_updates = []
                self._prepare_decision_locked()
                self._push_progress_update(message="Manual 360 degree scan complete.", frontier_id=None)
                self._manual_scan_in_progress = False
                return {
                    "status": "succeeded",
                    "message": "Manual 360 degree scan complete.",
                    "map": self._build_map_payload(),
                }
        except Exception as exc:
            with self._lock:
                self._manual_scan_in_progress = False
                self.status = "manual_scan_failed"
                self.last_error = f"Manual scan failed: {exc}"
                self.guardrail_events.append({"type": "manual_scan_failed", "reason": str(exc)})
                try:
                    self._push_progress_update(message=self.last_error, frontier_id=None)
                except Exception:
                    pass
                try:
                    map_payload = self._build_map_payload()
                except Exception:
                    map_payload = None
                return {"status": "failed", "reason": str(exc), "map": map_payload}

    def navigate_to_manual_waypoint(self, pose_payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._sync_manual_occupancy_edits()
            try:
                current_pose = self.runtime.current_pose()
                target_pose = Pose2D(
                    float(pose_payload["x"]),
                    float(pose_payload["y"]),
                    float(pose_payload.get("yaw", current_pose.yaw if current_pose is not None else 0.0)),
                )
            except Exception as exc:
                return {"status": "rejected", "reason": f"Invalid waypoint pose: {exc}"}
            print(
                "[real_exploration] manual waypoint requested "
                f"x={target_pose.x:.3f} y={target_pose.y:.3f} yaw={target_pose.yaw:.3f}",
                flush=True,
            )
            goal = self._make_nav2_goal(
                target_pose,
                goal_type="manual_waypoint",
                reason="operator_click_waypoint",
            )
            self.status = "manual_waypoint_active"
            self._push_progress_update(
                message=f"Sending clicked waypoint ({target_pose.x:.2f}, {target_pose.y:.2f}) to Nav2.",
                frontier_id=None,
            )
            result = self.nav2.navigate_to_pose(
                goal,
                ignore_pause_cancel=True,
                feedback_callback=self._publish_navigation_feedback,
            )
            self._consume_nav_result(result)
            self.status = "manual_waypoint_succeeded" if result.status == "succeeded" else "manual_waypoint_failed"
            if result.status == "succeeded":
                self._set_alert("Waypoint achieved.")
            print(
                "[real_exploration] manual waypoint result "
                f"status={result.status} reason={result.reason}",
                flush=True,
            )
            self.guardrail_events.append(
                {
                    "type": "manual_waypoint_navigation",
                    "requested_pose": target_pose.to_dict(),
                    "nav2_result": result.to_dict(),
                }
            )
            self._push_progress_update(
                message=f"Clicked waypoint navigation {result.status}: {result.reason}",
                frontier_id=None,
            )
            normalized_pose = None
            if result.plan and result.plan.status == "succeeded" and result.plan.path_cells:
                normalized_pose = self._require_effective_map().cell_to_pose(
                    result.plan.path_cells[-1].x,
                    result.plan.path_cells[-1].y,
                    yaw=target_pose.yaw,
                ).to_dict()
            return {
                "status": result.status,
                "reason": result.reason,
                "requested_pose": target_pose.to_dict(),
                "normalized_pose": normalized_pose,
                "nav2_result": result.to_dict(),
                "map": self._build_map_payload(),
            }

    def preview_manual_waypoint(self, pose_payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._sync_manual_occupancy_edits()
            try:
                current_pose = self.runtime.current_pose()
                target_pose = Pose2D(
                    float(pose_payload["x"]),
                    float(pose_payload["y"]),
                    float(pose_payload.get("yaw", current_pose.yaw if current_pose is not None else 0.0)),
                )
            except Exception as exc:
                return {"status": "rejected", "reason": f"Invalid waypoint pose: {exc}"}
            goal = self._make_nav2_goal(
                target_pose,
                goal_type="manual_waypoint_preview",
                reason="operator_click_waypoint_preview",
            )
            plan = self.nav2.compute_path(goal, record=True)
            self.last_nav2_preview = {
                "requested_pose": target_pose.to_dict(),
                "plan": plan.to_dict(),
                "created_at": time.time(),
            }
            self._push_progress_update(
                message=f"Previewed Nav2 path to clicked waypoint: {plan.status}.",
                frontier_id=None,
            )
            return {
                "status": plan.status,
                "reason": plan.reason,
                "requested_pose": target_pose.to_dict(),
                "plan": plan.to_dict(),
                "map": self._build_map_payload(),
            }

    def call_semantic_llm(self) -> dict[str, Any]:
        with self._lock:
            self.last_error = (
                "Automatic semantic waypoints are disabled. Start with --automatic-semantic-waypoints "
                "to enable the legacy semantic waypoint pipeline."
            )
            return self.snapshot()

    def _initialize_scan_state(self) -> None:
        self.backend.update_external_task(
            self.task_id,
            message="Waiting for ROS pose and scan data before the initial 360 degree scan.",
        )
        self.runtime.spin_until_ready(timeout_s=self.config.ros_ready_timeout_s)
        initial_pose = self._require_pose()
        self._update_pose_history(initial_pose)
        self.frontier_memory.remember_return_waypoint(
            room_id=None,
            pose=initial_pose,
            step_index=0,
            reason="initial_pose",
        )
        self.backend.update_external_task(
            self.task_id,
            message="Running the initial 360 degree scan.",
        )
        self._perform_turnaround_scan(reason="initial_turnaround_scan")
        self._push_progress_update(message="Initial 360 degree scan complete.", frontier_id=None)

    def _prepare_decision_locked(self) -> None:
        if self._budget_exhausted():
            self.status = "finished"
            self.pending_prompt_payload = None
            self.pending_prompt_text = None
            self.pending_candidate_records = []
            return
        self.decision_index += 1
        self._consume_runtime_scan_observations()
        self._sync_manual_occupancy_edits()
        occupancy_map = self._require_effective_map()
        pose = self._require_pose()
        self._update_pose_history(pose)
        visible_candidates = self._detect_frontier_candidates(occupancy_map, pose)
        self.frontier_memory.upsert_candidates(visible_candidates, step_index=self.decision_index)
        candidate_records = self._refresh_candidate_paths()
        prompt_payload = self._build_prompt_payload(occupancy_map, pose, candidate_records)
        self.pending_candidate_records = candidate_records
        self.pending_prompt_payload = prompt_payload
        self.pending_prompt_text = build_exploration_policy_user_prompt(prompt_payload)
        self.pending_decision = None
        self.pending_trace = None
        self.applied_memory_updates = []
        if candidate_records:
            self.status = "waiting_for_llm"
        elif self._coverage(occupancy_map) >= self.config.finish_coverage_threshold:
            self.status = "finished"
        else:
            self.status = "waiting_for_more_map_frontiers"
            self.last_error = (
                "No reachable frontier was detected yet on the live ROS map. "
                "Let SLAM/Nav2 publish more map data or run Reset + Scan again."
            )
        self._push_progress_update(message="Waiting for LLM frontier decision.", frontier_id=None)

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

    def _current_ros_map(self) -> RosOccupancyMap | None:
        if self._navigation_only_map is not None:
            return self._navigation_only_map
        if (
            self.config.ros_navigation_map_source == "external"
            and self.config.ros_fuse_external_projected_map_snapshots
            and self._latest_fused_projected_map is not None
        ):
            return self._latest_fused_projected_map
        return self.runtime.latest_map

    def _current_map(self) -> RosOccupancyMap | None:
        self._consume_runtime_scan_observations()
        if self.config.ros_navigation_map_source in ("fused_scan", "fused_point_cloud"):
            return self._current_scan_fused_map() or self._current_ros_map()
        return self._current_ros_map() or self._current_scan_fused_map()

    def _current_effective_map(self) -> EditableOccupancyMap | None:
        raw = self._current_map()
        if raw is None:
            return None
        return EditableOccupancyMap(raw, self.manual_occupancy_edits)

    def _sync_manual_occupancy_edits(self) -> None:
        self.manual_occupancy_edits = edits_from_payload(
            self.backend.occupancy_edit_snapshot(self.task_id),
            cell_type=GridCell,
        )

    def _require_effective_map(self) -> EditableOccupancyMap:
        return EditableOccupancyMap(self._require_map(), self.manual_occupancy_edits)

    def _scan_world_cell(self, x: float, y: float) -> GridCell:
        resolution = max(float(self.scan_map_resolution), 1e-6)
        return GridCell(int(math.floor(x / resolution)), int(math.floor(y / resolution)))

    def _world_cell_center(self, cell: GridCell) -> tuple[float, float]:
        resolution = max(float(self.scan_map_resolution), 1e-6)
        return ((cell.x + 0.5) * resolution, (cell.y + 0.5) * resolution)

    def _consume_runtime_scan_observations(self) -> None:
        if self.runtime.latest_map is not None:
            self.scan_map_resolution = float(self.runtime.latest_map.resolution)
        observations, stop_index = self.runtime.drain_scan_observations(self.scan_observation_index)
        point_observations, point_stop_index = self._drain_point_cloud_observations()
        if self.config.ros_navigation_map_source != "fused_point_cloud":
            for observation in observations:
                self._integrate_scan_observation(observation)
        if self.config.ros_navigation_map_source == "fused_point_cloud":
            for observation in point_observations:
                self._integrate_point_cloud_observation(observation)
        self.scan_observation_index = stop_index
        self.point_cloud_observation_index = point_stop_index
        if observations or point_observations:
            self._publish_navigation_map()

    def _drain_point_cloud_observations(self) -> tuple[list[dict[str, Any]], int]:
        drain = getattr(self.runtime, "drain_point_cloud_observations", None)
        if drain is None:
            return [], getattr(self, "point_cloud_observation_index", 0)
        return drain(getattr(self, "point_cloud_observation_index", 0))

    def _integrate_scan_observation(self, observation: dict[str, Any]) -> None:
        pose = observation.get("pose")
        if not isinstance(pose, Pose2D):
            return
        ranges = observation.get("ranges")
        if not isinstance(ranges, tuple) or not ranges:
            return
        integrate_planar_scan(
            pose=pose,
            ranges=ranges,
            angle_min=float(observation.get("angle_min", 0.0) or 0.0),
            angle_increment=float(observation.get("angle_increment", 0.0) or 0.0),
            range_min_m=float(observation.get("range_min", 0.05) or 0.05),
            range_max_m=float(observation.get("range_max", self.config.sensor_range_m) or self.config.sensor_range_m),
            resolution_m=self.scan_map_resolution,
            cell_from_world=lambda x, y: self._scan_world_cell(x, y),
            known_cells=self.scan_known_cells,
            evidence_scores=self.scan_occupancy_evidence,
            range_edge_cells=self.scan_range_edge_cells,
            beam_stride=2,
            config=ACTIVE_RGBD_SCAN_FUSION_CONFIG,
        )

    def _integrate_point_cloud_observation(self, observation: dict[str, Any]) -> None:
        points = observation.get("points_xyz")
        origin = observation.get("sensor_origin_xyz")
        if points is None or origin is None:
            return
        summary = integrate_transformed_point_cloud_observation(
            sensor_origin_xyz=origin,
            points_xyz_map=points,
            map_resolution_m=self.scan_map_resolution,
            cell_from_world=lambda x, y: self._scan_world_cell(x, y),
            known_cells=self.scan_known_cells,
            evidence_scores=self.scan_occupancy_evidence,
            range_edge_cells=self.scan_range_edge_cells,
            config=self.point_cloud_fusion_config,
        )
        self.point_cloud_integration_summaries.append(
            {
                "raw_point_count": summary.raw_point_count,
                "valid_point_count": summary.valid_point_count,
                "voxel_point_count": summary.voxel_point_count,
                "floor_point_count": summary.floor_point_count,
                "obstacle_point_count": summary.obstacle_point_count,
                "occupied_cell_count": summary.occupied_cell_count,
                "free_cell_count": summary.free_cell_count,
                "invalid_point_count": summary.invalid_point_count,
                "integrated_rays": summary.integrated_rays,
                "frame_id": str(observation.get("frame_id", "")),
            }
        )
        if len(self.point_cloud_integration_summaries) > 64:
            self.point_cloud_integration_summaries = self.point_cloud_integration_summaries[-64:]

    def _current_scan_fused_map(self) -> RosOccupancyMap | None:
        if not self.scan_known_cells:
            return None
        resolution = max(float(self.scan_map_resolution), 1e-6)
        world_cells = list(self.scan_known_cells)
        min_x = min(cell.x for cell in world_cells) - 4
        min_y = min(cell.y for cell in world_cells) - 4
        max_x = max(cell.x for cell in world_cells) + 4
        max_y = max(cell.y for cell in world_cells) + 4
        raw_map = self._current_ros_map()
        if raw_map is not None:
            raw_min_x = int(math.floor(float(raw_map.origin_x) / resolution))
            raw_min_y = int(math.floor(float(raw_map.origin_y) / resolution))
            raw_max_x = raw_min_x + int(raw_map.width) - 1
            raw_max_y = raw_min_y + int(raw_map.height) - 1
            min_x = min(min_x, raw_min_x)
            min_y = min(min_y, raw_min_y)
            max_x = max(max_x, raw_max_x)
            max_y = max(max_y, raw_max_y)
        origin_x = min_x * resolution
        origin_y = min_y * resolution
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        data = [-1] * (width * height)
        fused_map = RosOccupancyMap(
            resolution=resolution,
            width=width,
            height=height,
            origin_x=origin_x,
            origin_y=origin_y,
            data=tuple(data),
        )
        for cell, state in self.scan_known_cells.items():
            world_x, world_y = self._world_cell_center(cell)
            cell_x, cell_y = fused_map.world_to_cell(world_x, world_y)
            if not fused_map.in_bounds(cell_x, cell_y):
                continue
            data[cell_y * width + cell_x] = 100 if state == "occupied" else 0
        return RosOccupancyMap(
            resolution=resolution,
            width=width,
            height=height,
            origin_x=origin_x,
            origin_y=origin_y,
            data=tuple(data),
        )

    def _publish_navigation_map(self) -> None:
        if self.config.ros_navigation_map_source not in ("fused_scan", "fused_point_cloud"):
            return
        raw_map = self._current_scan_fused_map() or self._current_ros_map()
        if raw_map is None:
            return
        effective_map = EditableOccupancyMap(raw_map, self.manual_occupancy_edits)
        self.runtime.publish_navigation_map(
            _occupancy_map_like_to_ros_map(effective_map),
            map_to_odom=Pose2D(0.0, 0.0, 0.0),
        )

    def _known_cells_from_occupancy_map(self, occupancy_map: RosOccupancyMap | None) -> dict[GridCell, str]:
        if occupancy_map is None:
            return {}
        known: dict[GridCell, str] = {}
        for y in range(occupancy_map.height):
            for x in range(occupancy_map.width):
                value = occupancy_map.value(x, y)
                if value < 0:
                    continue
                known[GridCell(x, y)] = "free" if value == 0 else "occupied"
        return known

    def _wait_until_task_active(self) -> bool:
        while True:
            task = self.backend.get_task(self.task_id)
            if task is None:
                return False
            if task.get("state") == "aborted":
                return False
            if not bool(task.get("paused", False)):
                self._sync_manual_occupancy_edits()
                return True
            time.sleep(0.1)

    def _pause_for_operator_approval(self, message: str) -> bool:
        if not self.config.pause_for_operator_approval:
            return True
        self.backend.pause_task(self.task_id)
        self.backend.update_external_task(
            self.task_id,
            message=message,
        )
        self.status = "paused"
        return self._wait_until_task_active()

    def _pause_requested_or_canceled(self) -> bool:
        task = self.backend.get_task(self.task_id)
        if task is None:
            return True
        return bool(task.get("paused", False) or task.get("state") == "aborted")

    def _task_canceled_or_aborted(self) -> bool:
        task = self.backend.get_task(self.task_id)
        if task is None:
            return True
        return bool(task.get("state") == "aborted" or task.get("canceled", False))

    def _require_map(self) -> RosOccupancyMap:
        occupancy_map = self._current_map()
        if occupancy_map is None:
            raise RuntimeError("ROS occupancy map is not available")
        return occupancy_map

    def _require_pose(self) -> Pose2D:
        pose = self.runtime.current_pose()
        if pose is None and self.config.ros_navigation_map_source in ("fused_scan", "fused_point_cloud"):
            pose = self.runtime.current_pose_in_frame(self.config.ros_odom_frame)
        if pose is None:
            raise RuntimeError(
                (
                    f"Robot pose in `{self.config.ros_map_frame}` is not available from TF yet"
                    if self.config.ros_navigation_map_source not in ("fused_scan", "fused_point_cloud")
                    else (
                        f"Robot pose is not available from TF yet in either `{self.config.ros_map_frame}` "
                        f"or `{self.config.ros_odom_frame}`."
                    )
                )
            )
        return pose

    def _budget_exhausted(self) -> bool:
        if self.config.max_control_steps is not None and self.control_steps >= self.config.max_control_steps:
            return True
        if self.config.max_episode_steps is not None and self.decision_index >= self.config.max_episode_steps:
            return True
        return False

    def _perform_turnaround_scan(self, *, reason: str, ignore_pause: bool = False) -> None:
        self.runtime.set_point_cloud_map_updates_enabled(True)
        try:
            event = self.runtime.perform_turnaround_scan(
                reason=reason,
                should_cancel=self._task_canceled_or_aborted if ignore_pause else self._pause_requested_or_canceled,
            )
        finally:
            self.runtime.set_point_cloud_map_updates_enabled(False)
        fused_projected_map = event.pop("fused_projected_map", None)
        if self.config.ros_fuse_external_projected_map_snapshots and isinstance(fused_projected_map, RosOccupancyMap):
            self._latest_fused_projected_map = fused_projected_map
        observations = list(event.pop("observations", []))
        self.guardrail_events.append({"type": "turnaround_scan", "event": event})
        self.runtime.spin_for(0.25)
        if self.config.ros_navigation_map_source != "fused_point_cloud":
            for observation in observations:
                self._integrate_scan_observation(observation)
        point_observations, point_stop_index = self._drain_point_cloud_observations()
        if self.config.ros_navigation_map_source == "fused_point_cloud":
            for observation in point_observations:
                self._integrate_point_cloud_observation(observation)
        self.scan_observation_index = int(event.get("observation_stop_index", self.scan_observation_index))
        self.point_cloud_observation_index = point_stop_index
        event["selected_count"] = len(observations)
        event["raw_count"] = len(observations)
        event["point_cloud_observation_count"] = len(point_observations)
        self._publish_navigation_map()
        self._capture_keyframe(reason=reason)
        pose = self.runtime.current_pose()
        if pose is not None:
            self._update_pose_history(pose)

    def _capture_keyframe(self, *, reason: str) -> None:
        pose = self.runtime.current_pose()
        if pose is None:
            return
        scan = self.runtime.latest_scan
        latest_point_cloud = getattr(self.runtime, "latest_point_cloud_stats", None)
        point_count = (
            int(latest_point_cloud.get("point_count", 0))
            if isinstance(latest_point_cloud, dict) and self.config.ros_navigation_map_source == "fused_point_cloud"
            else len(getattr(scan, "ranges", []) or [])
        )
        frame_id = f"kf_{len(self.keyframes) + 1:03d}"
        frame = {
            "frame_id": frame_id,
            "pose": pose.to_dict(),
            "region_id": "unknown",
            "visible_objects": [],
            "point_count": point_count,
            "depth_min_m": float(getattr(scan, "range_min", 0.0) or 0.0),
            "depth_max_m": float(getattr(scan, "range_max", self.config.sensor_range_m) or self.config.sensor_range_m),
            "description": (
                f"Live head-camera observation captured after `{reason}` at "
                f"map pose ({pose.x:.2f}, {pose.y:.2f}, yaw={pose.yaw:.2f})."
            ),
            "thumbnail_data_url": self.runtime.latest_image_data_url or "",
        }
        if isinstance(latest_point_cloud, dict):
            frame["rgbd_summary"] = {
                "source": "orbbec_gemini2_point_cloud",
                "latest_point_cloud": latest_point_cloud,
                "recent_fusion": list(self.point_cloud_integration_summaries[-4:]),
            }
        self.keyframes.append(frame)
        if self.config.automatic_semantic_waypoints:
            self.semantic_observer.observe_keyframe(
                frame=frame,
                known_cells=self._known_cells_from_occupancy_map(self.runtime.latest_map),
                robot_cell=GridCell(
                    int(math.floor(pose.x / self.config.occupancy_resolution)),
                    int(math.floor(pose.y / self.config.occupancy_resolution)),
                ),
                resolution=self.config.occupancy_resolution,
            )

    def _update_pose_history(self, pose: Pose2D) -> None:
        if self._last_pose is not None:
            delta = _pose_distance_m(self._last_pose, pose)
            if delta > 1e-3:
                self.control_steps += 1
        if self._last_pose is None or _pose_distance_m(self._last_pose, pose) > 0.02:
            self.trajectory.append(pose.to_dict())
            self._last_pose = pose

    def _detect_frontier_candidates(
        self,
        occupancy_map: RosOccupancyMap,
        pose: Pose2D,
    ) -> list[FrontierCandidate]:
        frontier_cells: set[GridCell] = set()
        unknown_neighbors_by_frontier: dict[GridCell, set[GridCell]] = {}
        range_edge_cells = self._range_edge_cells(occupancy_map, pose)
        for y in range(occupancy_map.height):
            for x in range(occupancy_map.width):
                if not occupancy_map.is_free(x, y):
                    continue
                cell = GridCell(x, y)
                unknown_neighbors = {
                    neighbor
                    for neighbor in _neighbors4(cell)
                    if (
                        not occupancy_map.in_bounds(neighbor.x, neighbor.y)
                        or occupancy_map.is_unknown(neighbor.x, neighbor.y)
                    )
                }
                if unknown_neighbors and (len(unknown_neighbors) >= 1 or cell in range_edge_cells):
                    frontier_cells.add(cell)
                    unknown_neighbors_by_frontier[cell] = unknown_neighbors

        clusters: list[list[GridCell]] = []
        visited: set[GridCell] = set()
        for cell in frontier_cells:
            if cell in visited:
                continue
            cluster: list[GridCell] = []
            queue = deque([cell])
            visited.add(cell)
            while queue:
                current = queue.popleft()
                cluster.append(current)
                for neighbor in _neighbors8(current):
                    if neighbor in frontier_cells and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            clusters.append(cluster)

        robot_cell = GridCell(*occupancy_map.world_to_cell(pose.x, pose.y))
        reachable_free_cells = self._reachable_free_cells(occupancy_map, pose)
        candidates: list[FrontierCandidate] = []
        configured_opening_m = self.config.frontier_min_opening_m
        required_opening_m = (
            max(float(configured_opening_m), occupancy_map.resolution)
            if configured_opening_m is not None and configured_opening_m > 0.0
            else max(self.config.robot_radius_m * 2.0 + 0.10, occupancy_map.resolution * 2.0)
        )
        for cluster in clusters:
            cluster_unknown = set().union(*(unknown_neighbors_by_frontier.get(cell, set()) for cell in cluster))
            if not cluster_unknown:
                continue
            if not any(cell in reachable_free_cells for cell in cluster):
                self.guardrail_events.append(
                    {
                        "type": "frontier_boundary_unreachable_through_known_free",
                        "cluster_size": len(cluster),
                        "unknown_gain": len(cluster_unknown),
                        "frontier_boundary_pose": _cell_mean_pose(cluster, occupancy_map.resolution).to_dict(),
                    }
                )
                continue
            opening_width_m = _frontier_opening_width_m(cluster, occupancy_map.resolution)
            if opening_width_m < required_opening_m:
                self.guardrail_events.append(
                    {
                        "type": "frontier_opening_too_narrow",
                        "cluster_size": len(cluster),
                        "opening_width_m": round(opening_width_m, 3),
                        "required_width_m": round(required_opening_m, 3),
                        "frontier_boundary_pose": _cell_mean_pose(cluster, occupancy_map.resolution).to_dict(),
                    }
                )
                continue
            boundary_pose = _cell_mean_pose(cluster, occupancy_map.resolution)
            boundary_pose = Pose2D(
                boundary_pose.x + occupancy_map.origin_x,
                boundary_pose.y + occupancy_map.origin_y,
                boundary_pose.yaw,
            )
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
            nav_cell = min(cluster, key=lambda cell: _grid_distance_cells(cell, robot_cell))
            centroid_cell = _centroid_cell(cluster)
            nav_pose = occupancy_map.cell_to_pose(nav_cell.x, nav_cell.y)
            centroid_pose = occupancy_map.cell_to_pose(centroid_cell.x, centroid_cell.y)
            evidence = [
                f"{len(cluster_unknown)} unknown neighbor cells on the live occupancy map",
                f"cluster size {len(cluster)}",
                f"frontier opening width is {opening_width_m:.2f} m, above robot-sized threshold {required_opening_m:.2f} m",
            ]
            if any(cell in range_edge_cells for cell in cluster):
                evidence.append("frontier also aligns with the depth-derived sensor range limit")
            candidates.append(
                FrontierCandidate(
                    frontier_id=None,
                    member_cells=tuple(sorted(cluster)),
                    nav_cell=nav_cell,
                    centroid_cell=centroid_cell,
                    nav_pose=nav_pose,
                    centroid_pose=centroid_pose,
                    unknown_gain=len(cluster_unknown),
                    sensor_range_edge=any(cell in range_edge_cells for cell in cluster),
                    room_hint=None,
                    evidence=evidence,
                    currently_visible=_pose_distance_m(nav_pose, pose) <= self.config.sensor_range_m + 0.5,
                )
            )
        return candidates

    def _range_edge_cells(self, occupancy_map: RosOccupancyMap, pose: Pose2D) -> set[GridCell]:
        if self.scan_range_edge_cells:
            result: set[GridCell] = set()
            for cell in self.scan_range_edge_cells:
                world_x, world_y = self._world_cell_center(cell)
                cell_x, cell_y = occupancy_map.world_to_cell(world_x, world_y)
                if occupancy_map.in_bounds(cell_x, cell_y):
                    result.add(GridCell(cell_x, cell_y))
            if result:
                return result
        scan = self.runtime.latest_scan
        if scan is None or not getattr(scan, "ranges", None):
            return set()
        result: set[GridCell] = set()
        beam_stride = max(len(scan.ranges) // 96, 1)
        step_m = max(occupancy_map.resolution * 0.5, 0.05)
        for index in range(0, len(scan.ranges), beam_stride):
            beam_range = float(scan.ranges[index])
            if math.isfinite(beam_range) and beam_range < float(scan.range_max) * 0.98:
                continue
            angle = pose.yaw + float(scan.angle_min) + index * float(scan.angle_increment)
            last_free: GridCell | None = None
            samples = int(max(float(scan.range_max) / step_m, 1))
            for sample_idx in range(1, samples + 1):
                distance = min(sample_idx * step_m, float(scan.range_max))
                world_x = pose.x + distance * math.cos(angle)
                world_y = pose.y + distance * math.sin(angle)
                cell_x, cell_y = occupancy_map.world_to_cell(world_x, world_y)
                if not occupancy_map.in_bounds(cell_x, cell_y):
                    break
                if occupancy_map.is_occupied(cell_x, cell_y):
                    break
                if occupancy_map.is_unknown(cell_x, cell_y):
                    if last_free is not None:
                        result.add(last_free)
                    break
                last_free = GridCell(cell_x, cell_y)
        return result

    def _reachable_free_cells(
        self,
        occupancy_map: EditableOccupancyMap,
        pose: Pose2D,
    ) -> set[GridCell]:
        origin_cell = GridCell(*occupancy_map.world_to_cell(pose.x, pose.y))
        if not occupancy_map.in_bounds(origin_cell.x, origin_cell.y):
            return set()
        if not occupancy_map.is_free(origin_cell.x, origin_cell.y):
            replacement = self._nearest_free_cell(occupancy_map, origin_cell, max_radius_cells=8)
            if replacement is None:
                self.guardrail_events.append(
                    {
                        "type": "ros_pose_cell_not_free",
                        "pose_cell": {"cell_x": origin_cell.x, "cell_y": origin_cell.y},
                    }
                )
                return set()
            self.guardrail_events.append(
                {
                    "type": "ros_pose_cell_snapped_to_nearest_free",
                    "pose_cell": {"cell_x": origin_cell.x, "cell_y": origin_cell.y},
                    "nearest_free_cell": {"cell_x": replacement.x, "cell_y": replacement.y},
                }
            )
            origin_cell = replacement
        reachable: set[GridCell] = set()
        queue = deque([origin_cell])
        while queue:
            current = queue.popleft()
            if current in reachable:
                continue
            reachable.add(current)
            for neighbor in _neighbors4(current):
                if neighbor in reachable:
                    continue
                if not occupancy_map.in_bounds(neighbor.x, neighbor.y):
                    continue
                if not occupancy_map.is_free(neighbor.x, neighbor.y):
                    continue
                queue.append(neighbor)
        return reachable

    def _nearest_free_cell(
        self,
        occupancy_map: EditableOccupancyMap,
        origin_cell: GridCell,
        *,
        max_radius_cells: int,
    ) -> GridCell | None:
        candidates: list[tuple[int, GridCell]] = []
        for radius in range(1, max_radius_cells + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if max(abs(dx), abs(dy)) != radius:
                        continue
                    cell = GridCell(origin_cell.x + dx, origin_cell.y + dy)
                    if not occupancy_map.in_bounds(cell.x, cell.y):
                        continue
                    if occupancy_map.is_free(cell.x, cell.y):
                        candidates.append((_grid_distance_cells(origin_cell, cell), cell))
            if candidates:
                return min(candidates, key=lambda item: item[0])[1]
        return None

    def _global_frontier_anchor_cell_near_record(
        self,
        record: FrontierRecord,
    ) -> tuple[GridCell | None, str | None]:
        occupancy_map = self._require_effective_map()
        pose = self._require_pose()
        reachable_free_cells = self._reachable_free_cells(occupancy_map, pose)
        range_edge_cells = self._range_edge_cells(occupancy_map, pose)
        boundary_cell = GridCell(*occupancy_map.world_to_cell(record.centroid_pose.x, record.centroid_pose.y))
        search_radius_cells = max(1, int(math.ceil(STORED_FRONTIER_REVALIDATION_RADIUS_M / occupancy_map.resolution)))
        strong_candidates: list[tuple[int, int, GridCell]] = []
        relaxed_candidates: list[tuple[int, int, GridCell]] = []
        unreachable_boundary_candidates = 0
        for dx in range(-search_radius_cells, search_radius_cells + 1):
            for dy in range(-search_radius_cells, search_radius_cells + 1):
                cell = GridCell(boundary_cell.x + dx, boundary_cell.y + dy)
                if not occupancy_map.in_bounds(cell.x, cell.y):
                    continue
                distance_cells = _grid_distance_cells(cell, boundary_cell)
                if distance_cells > search_radius_cells or not occupancy_map.is_free(cell.x, cell.y):
                    continue
                unknown_neighbors = {
                    neighbor
                    for neighbor in _neighbors4(cell)
                    if occupancy_map.in_bounds(neighbor.x, neighbor.y) and occupancy_map.is_unknown(neighbor.x, neighbor.y)
                }
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

    def _revalidate_stored_frontier_boundary(
        self,
        record: FrontierRecord,
        anchor_cell: GridCell,
        anchor_mode: str | None,
    ) -> None:
        occupancy_map = self._require_effective_map()
        anchor_pose = occupancy_map.cell_to_pose(anchor_cell.x, anchor_cell.y)
        if _pose_distance_m(record.centroid_pose, anchor_pose) <= occupancy_map.resolution * 0.5 and anchor_mode != "relaxed":
            return
        previous_pose = record.centroid_pose
        record.centroid_pose = anchor_pose
        notes = [
            (
                "stored frontier boundary was revalidated against the current global occupancy map "
                f"near the original memory point ({previous_pose.x:.2f}, {previous_pose.y:.2f})"
            )
        ]
        if anchor_mode == "relaxed":
            notes.append(
                "stored frontier memory was kept using relaxed revalidation because nearby free space still borders unknown map area"
            )
        record.evidence = _dedupe_text(record.evidence + notes)

    def _resnap_stored_frontier_revisit_pose(
        self,
        record: FrontierRecord,
        current_pose: Pose2D,
        anchor_cell: GridCell,
    ) -> Pose2D | None:
        occupancy_map = self._require_effective_map()
        reachable_safe_cells = self._reachable_free_cells(occupancy_map, current_pose)
        resolution = occupancy_map.resolution
        anchor_pose = occupancy_map.cell_to_pose(anchor_cell.x, anchor_cell.y)
        max_radius_cells = max(1, int(math.ceil(STORED_FRONTIER_REVISIT_APPROACH_RADIUS_M / resolution)))
        scored_cells: list[tuple[float, GridCell]] = []
        for cell in reachable_safe_cells:
            distance_cells = _grid_distance_cells(cell, anchor_cell)
            if distance_cells > max_radius_cells:
                continue
            cell_pose = occupancy_map.cell_to_pose(cell.x, cell.y)
            score = (
                abs(distance_cells * resolution - (self.config.robot_radius_m + 0.25))
                + 0.03 * _pose_distance_m(cell_pose, current_pose)
                + 0.02 * _pose_distance_m(cell_pose, record.nav_pose)
            )
            scored_cells.append((score, cell))
        if not scored_cells:
            return None
        best = min(scored_cells, key=lambda item: item[0])[1]
        best_pose = occupancy_map.cell_to_pose(best.x, best.y)
        return Pose2D(
            best_pose.x,
            best_pose.y,
            math.atan2(anchor_pose.y - best_pose.y, anchor_pose.x - best_pose.x),
        )

    def _apply_stored_frontier_resnap(
        self,
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
                "stored frontier revisit approach pose was re-snapped to nearby mapped free space before LLM selection"
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

    def _is_frontier_at_current_pose(self, record: FrontierRecord, current_pose_filter_m: float) -> bool:
        occupancy_map = self._require_effective_map()
        current_pose = self._require_pose()
        target_cell = GridCell(*occupancy_map.world_to_cell(record.centroid_pose.x, record.centroid_pose.y))
        current_cell = GridCell(*occupancy_map.world_to_cell(current_pose.x, current_pose.y))
        return target_cell == current_cell or _pose_distance_m(record.centroid_pose, current_pose) <= current_pose_filter_m

    def _visited_frontier_filter_radius_m(self) -> float:
        configured = self.config.visited_frontier_filter_radius_m
        if configured is not None and configured > 0.0:
            return max(float(configured), self.config.occupancy_resolution)
        return max(self.config.robot_radius_m + 0.35, self.config.occupancy_resolution * 2.0)

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

    def _refresh_candidate_paths(self) -> list[FrontierRecord]:
        pose = self._require_pose()
        current_pose_filter_m = max(self.config.occupancy_resolution * 1.5, self.config.robot_radius_m)

        def _path_cost(record: FrontierRecord) -> float | None:
            preview_goal = Nav2GoalRequest(
                goal_id=f"preview_{self.decision_index:03d}_{record.frontier_id}",
                goal_type="frontier_preview",
                target_pose=record.nav_pose,
                planner_id=self.config.nav2_planner_id,
                controller_id=self.config.nav2_controller_id,
                behavior_tree=self.config.nav2_behavior_tree,
                metadata={"frontier_id": record.frontier_id},
            )
            plan = self.nav2.compute_path(preview_goal, record=False)
            return plan.path_length_m if plan.status == "succeeded" else None

        return refresh_frontier_records(
            candidate_records=self.frontier_memory.candidate_records(),
            active_frontier_id=self.frontier_memory.active_frontier_id,
            current_pose=pose,
            current_pose_filter_m=current_pose_filter_m,
            path_cost_for_record=_path_cost,
            guardrail_events=self.guardrail_events,
            is_frontier_at_current_pose=self._is_frontier_at_current_pose,
            is_frontier_near_visited_pose=self._is_frontier_near_visited_pose,
            visited_pose_filter_m=self._visited_frontier_filter_radius_m(),
            global_anchor_for_stored_record=self._global_frontier_anchor_cell_near_record,
            revalidate_stored_boundary=self._revalidate_stored_frontier_boundary,
            resnap_stored_nav_pose=self._resnap_stored_frontier_revisit_pose,
            apply_stored_resnap=self._apply_stored_frontier_resnap,
        )

    def _build_prompt_payload(
        self,
        occupancy_map: RosOccupancyMap,
        pose: Pose2D,
        candidate_records: list[FrontierRecord],
    ) -> dict[str, Any]:
        known_free = 0
        occupied = 0
        unknown = 0
        for y in range(occupancy_map.height):
            for x in range(occupancy_map.width):
                item = occupancy_map.value(x, y)
                if item < 0:
                    unknown += 1
                elif item == 0:
                    known_free += 1
                else:
                    occupied += 1
        return {
            "mission": (
                "Explore the live ManiSkill apartment through the ROS/Nav2 stack, keep frontier memory stable, "
                "and prefer fast global coverage over redundant local scans."
            ),
            "robot": {
                "pose": pose.to_dict(),
                "room_id": None,
                "coverage": round(self._coverage(occupancy_map), 3),
                "trajectory_points": len(self.trajectory),
                "sensor_range_m": self.config.sensor_range_m,
            },
            "frontier_memory": self.frontier_memory.snapshot(),
            "frontier_information": [record.to_dict() for record in candidate_records],
            "candidate_frontiers": [record.to_dict() for record in candidate_records],
            "explored_areas": [
                {
                    "region_id": "mapped_space",
                    "label": "mapped_space",
                    "observed_fraction": round(self._coverage(occupancy_map), 3),
                    "objects_seen": [],
                    "representative_frames": [item["frame_id"] for item in self.keyframes[-3:]],
                }
            ],
            "recent_views": self.keyframes[-3:],
            "frontier_selection_guidance": _frontier_selection_guidance(),
            "map_stats": {
                "known_free_cells": known_free,
                "occupied_cells": occupied,
                "unknown_cells": unknown,
                "map_bounds": occupancy_map.bounds(),
                "map_age_s": round(seconds_since(self.runtime.latest_map_stamp_s), 3),
            },
            "guardrails": {
                "finish_requires_frontier_exhaustion": True,
                "navigation_must_use_nav2_goal_validation": True,
                "frontier_ids_must_come_from_prompt": True,
                "frontier_information_is_boundary_evidence_not_a_command": True,
                "select_regions_that_expand_robot_navigable_space": True,
                "avoid_furniture_shadow_boundaries_without_clear_open_space": True,
            },
            "ascii_map": self._ascii_map(occupancy_map, pose, candidate_records),
        }

    def _ascii_map(
        self,
        occupancy_map: RosOccupancyMap,
        pose: Pose2D,
        candidate_records: list[FrontierRecord],
    ) -> str:
        robot_cell = GridCell(*occupancy_map.world_to_cell(pose.x, pose.y))
        frontier_cells = {
            occupancy_map.world_to_cell(record.nav_pose.x, record.nav_pose.y): record.status
            for record in candidate_records
        }
        interesting_x = [robot_cell.x]
        interesting_y = [robot_cell.y]
        for (cell_x, cell_y) in frontier_cells:
            interesting_x.append(cell_x)
            interesting_y.append(cell_y)
        for y in range(occupancy_map.height):
            for x in range(occupancy_map.width):
                value = occupancy_map.value(x, y)
                if value >= 0:
                    interesting_x.append(x)
                    interesting_y.append(y)
        if not interesting_x or not interesting_y:
            return "map unavailable"
        min_x = max(min(interesting_x) - 2, 0)
        max_x = min(max(interesting_x) + 2, occupancy_map.width - 1)
        min_y = max(min(interesting_y) - 2, 0)
        max_y = min(max(interesting_y) + 2, occupancy_map.height - 1)
        max_width = 72
        max_height = 48
        if max_x - min_x + 1 > max_width:
            half = max_width // 2
            min_x = max(robot_cell.x - half, 0)
            max_x = min(min_x + max_width - 1, occupancy_map.width - 1)
        if max_y - min_y + 1 > max_height:
            half = max_height // 2
            min_y = max(robot_cell.y - half, 0)
            max_y = min(min_y + max_height - 1, occupancy_map.height - 1)

        lines: list[str] = []
        for y in reversed(range(min_y, max_y + 1)):
            row: list[str] = []
            for x in range(min_x, max_x + 1):
                if x == robot_cell.x and y == robot_cell.y:
                    row.append("R")
                    continue
                if (x, y) in frontier_cells:
                    row.append("V" if frontier_cells[(x, y)] in {"completed", "failed"} else "F")
                    continue
                value = occupancy_map.value(x, y)
                if value < 0:
                    row.append("?")
                elif value == 0:
                    row.append(".")
                else:
                    row.append("#")
            lines.append("".join(row))
        return "\n".join(lines)

    def _apply_finish_guardrail(
        self,
        decision: ExplorationDecision,
        candidate_records: list[FrontierRecord],
    ) -> ExplorationDecision:
        reachable = [record for record in candidate_records if record.path_cost_m is not None]
        if decision.decision_type != "finish":
            return decision
        if not reachable:
            return decision
        fallback = self.policy._heuristic_decision(  # intentional guardrail fallback
            candidate_records,
            list(self.frontier_memory.return_waypoints.values()),
            0.0,
            None,
        )
        self.guardrail_events.append(
            {
                "type": "finish_override",
                "requested": decision.to_dict(),
                "fallback": fallback.to_dict(),
            }
        )
        return fallback

    def _log_policy_step(self, prompt_payload: dict[str, Any], decision: ExplorationDecision, trace: dict[str, Any]) -> None:
        if self.config.trace_policy_stdout:
            print(
                f"[exploration-policy] step={self.decision_index} coverage={self._coverage(self._require_map()):.3f} "
                f"decision={decision.decision_type} frontier={decision.selected_frontier_id}"
            )
        if self.config.trace_llm_stdout:
            print("[exploration-llm] prompt")
            print(trace.get("prompt", ""))
            print("[exploration-llm] response")
            print(json.dumps(trace.get("response", {}), indent=2, sort_keys=True))
            if trace.get("llm_trace", {}).get("error"):
                print(f"[exploration-llm] error={trace['llm_trace']['error']}")
        self.decision_log.append(
            {
                "step_index": self.decision_index,
                "coverage": round(self._coverage(self._require_map()), 3),
                "decision": decision.to_dict(),
                "trace": trace,
                "frontier_information_ids": [
                    item.get("frontier_id")
                    for item in prompt_payload.get("frontier_information", prompt_payload.get("candidate_frontiers", []))
                ],
            }
        )

    def _push_progress_update(self, *, message: str, frontier_id: str | None) -> None:
        coverage = self._coverage(self._require_effective_map())
        result = {
            "coverage": round(coverage, 3),
            "trajectory": self.trajectory[-12:],
            "keyframes": self.keyframes[-4:],
            "frontier_memory": self.frontier_memory.snapshot(),
            "active_frontier_id": frontier_id,
        }
        self.backend.update_external_task(
            self.task_id,
            progress=min(coverage, 0.98),
            message=message,
            result=result,
            map_payload=self._build_map_payload(),
        )

    def _publish_live_map(self, message: str) -> None:
        self._push_progress_update(message=message, frontier_id=self.frontier_memory.active_frontier_id)

    def _set_alert(self, message: str) -> None:
        self.last_alert = message
        self.last_alert_id += 1

    def _live_robot_pose(self) -> Pose2D | None:
        pose = self.runtime.current_pose()
        if pose is None and self.config.ros_navigation_map_source in ("fused_scan", "fused_point_cloud"):
            pose = self.runtime.current_pose_in_frame(self.config.ros_odom_frame)
        return pose

    def _publish_navigation_feedback(self, sample: dict[str, Any]) -> None:
        now = time.time()
        if now - self._last_live_pose_publish_s < 0.35:
            return
        self._last_live_pose_publish_s = now
        pose_payload = sample.get("current_pose")
        if isinstance(pose_payload, dict):
            try:
                pose = Pose2D(
                    float(pose_payload["x"]),
                    float(pose_payload["y"]),
                    float(pose_payload.get("yaw", 0.0)),
                )
            except Exception:
                pose = None
            if pose is not None:
                self._update_pose_history(pose)
        distance_remaining = sample.get("distance_remaining_m")
        message = "Nav2 waypoint active."
        if distance_remaining is not None:
            message = f"Nav2 waypoint active, {float(distance_remaining):.2f} m remaining."
        self._push_progress_update(message=message, frontier_id=self.frontier_memory.active_frontier_id)

    def _consume_nav_result(self, result: Nav2NavigateResult) -> None:
        if result.reached_pose is not None:
            self._update_pose_history(result.reached_pose)
        self.total_distance_m += max(result.travelled_distance_m, 0.0)
        self.runtime.spin_for(0.2)
        self._consume_runtime_scan_observations()

    def _coverage(self, occupancy_map: RosOccupancyMap | EditableOccupancyMap) -> float:
        known = 0
        total = max(int(occupancy_map.width) * int(occupancy_map.height), 1)
        for y in range(occupancy_map.height):
            for x in range(occupancy_map.width):
                if occupancy_map.value(x, y) >= 0:
                    known += 1
        return round(known / total, 6)

    def _build_map_payload(self) -> dict[str, Any]:
        occupancy_map = self._require_effective_map()
        live_robot_pose = self._live_robot_pose()
        semantic_memory = self.semantic_observer.snapshot() if self.config.automatic_semantic_waypoints else {}
        occupancy_cells = []
        for y in range(occupancy_map.height):
            for x in range(occupancy_map.width):
                value = occupancy_map.value(x, y)
                if value < 0:
                    continue
                pose = occupancy_map.cell_to_pose(x, y)
                occupancy_cells.append(
                    {
                        "x": round(pose.x - occupancy_map.resolution / 2.0, 3),
                        "y": round(pose.y - occupancy_map.resolution / 2.0, 3),
                        "state": "free" if value == 0 else "occupied",
                        **(
                            {"manual_override": "blocked"}
                            if GridCell(x, y) in self.manual_occupancy_edits.blocked_cells
                            else {"manual_override": "cleared"}
                            if GridCell(x, y) in self.manual_occupancy_edits.cleared_cells
                            else {}
                        ),
                    }
                )
        semantic_area_candidates = []
        if self.config.experimental_free_space_semantic_waypoints:
            semantic_area_candidates = _aggregate_semantic_updates(self.semantic_updates)
        summary = (
            f"ROS/Nav2 exploration completed with coverage {self._coverage(occupancy_map):.3f}, "
            f"{len(self.frontier_memory.records)} tracked frontiers, and {len(self.decision_log)} decisions."
        )
        payload = {
            "map_id": self.config.session,
            "frame": self.config.ros_map_frame,
            "resolution": float(occupancy_map.resolution),
            "coverage": round(self._coverage(occupancy_map), 3),
            "summary": summary,
            "approved": False,
            "created_at": time.time(),
            "source": self.config.source,
            "mode": "ros_nav2_agentic",
            "robot_pose": None if live_robot_pose is None else live_robot_pose.to_dict(),
            "trajectory": self.trajectory,
            "keyframes": self.keyframes,
            "regions": [],
            "named_places": _semantic_named_places_for_map(semantic_memory) if self.config.automatic_semantic_waypoints else [],
            "occupancy": {
                "resolution": float(occupancy_map.resolution),
                "bounds": occupancy_map.bounds(),
                "cells": occupancy_cells,
            },
            "frontiers": [record.to_dict() for record in self.frontier_memory.records.values()],
            "semantic_area_candidates": semantic_area_candidates,
            "semantic_memory": semantic_memory,
            "automatic_semantic_waypoints": self.config.automatic_semantic_waypoints,
            "artifacts": {
                "decision_log": self.decision_log,
                "frontier_memory": self.frontier_memory.snapshot(),
                "guardrail_events": self.guardrail_events,
                "manual_occupancy_edits": self.manual_occupancy_edits.to_dict(
                    resolution=occupancy_map.resolution,
                    origin_x=occupancy_map.bounds()["min_x"],
                    origin_y=occupancy_map.bounds()["min_y"],
                ),
                "navigation": {
                    "control_steps": self.control_steps,
                    "total_distance_m": round(self.total_distance_m, 3),
                },
                "nav2": self.nav2.snapshot(),
                "nav2_preview": self.last_nav2_preview,
                "ros_runtime": {
                    "map_topic": self.config.ros_map_topic,
                    "scan_topic": self.config.ros_scan_topic,
                    "point_cloud_topic": self.config.ros_point_cloud_topic,
                    "rgb_topic": self.config.ros_rgb_topic,
                    "imu_topic": self.config.ros_imu_topic,
                    "navigation_map_source": self.config.ros_navigation_map_source,
                    "base_frame": self.config.ros_base_frame,
                    "odom_frame": self.config.ros_odom_frame,
                    "map_frame": self.config.ros_map_frame,
                    "point_cloud_fusion": {
                        "recent_summaries": list(self.point_cloud_integration_summaries[-8:]),
                    },
                },
                "llm_policy": {
                    "explorer_policy": self.config.explorer_policy,
                    "provider": self.config.llm_provider,
                    "model": self.config.llm_model,
                },
            },
        }
        now = time.time()
        if now - self._last_map_payload_log_s >= 2.0:
            self._last_map_payload_log_s = now
            raw_summary = self.runtime.latest_map_summary() if hasattr(self.runtime, "latest_map_summary") else None
            fused_summary = None
            if self._latest_fused_projected_map is not None:
                fused_summary = {
                    "resolution": round(float(self._latest_fused_projected_map.resolution), 4),
                    "width": int(self._latest_fused_projected_map.width),
                    "height": int(self._latest_fused_projected_map.height),
                    "free_cells": sum(1 for item in self._latest_fused_projected_map.data if int(item) == 0),
                    "occupied_cells": sum(1 for item in self._latest_fused_projected_map.data if int(item) > 50),
                    "unknown_cells": sum(1 for item in self._latest_fused_projected_map.data if int(item) < 0),
                    "bounds": self._latest_fused_projected_map.bounds(),
                }
            free_cells = 0
            occupied_cells = 0
            unknown_cells = 0
            for y in range(occupancy_map.height):
                for x in range(occupancy_map.width):
                    value = occupancy_map.value(x, y)
                    if value == 0:
                        free_cells += 1
                    elif value > 50:
                        occupied_cells += 1
                    elif value < 0:
                        unknown_cells += 1
            print(
                "[ros_exploration_session] UI map payload "
                f"source_topic={self.config.ros_map_topic} raw_summary={raw_summary} "
                f"fused_projected_summary={fused_summary} "
                f"payload_cells={len(occupancy_cells)} free={free_cells} "
                f"occupied={occupied_cells} unknown={unknown_cells} "
                f"coverage={payload['coverage']} bounds={payload['occupancy']['bounds']}"
            )
        return payload

    def _make_nav2_goal(self, pose: Pose2D, *, goal_type: str, reason: str) -> Nav2GoalRequest:
        self.nav2_goal_counter += 1
        return Nav2GoalRequest(
            goal_id=f"nav2_goal_{self.nav2_goal_counter:03d}",
            goal_type=goal_type,
            target_pose=pose,
            planner_id=self.config.nav2_planner_id,
            controller_id=self.config.nav2_controller_id,
            behavior_tree=self.config.nav2_behavior_tree,
            metadata={"reason": reason},
        )


class _ExplorationStartGate:
    def __init__(self, config: SimExplorationConfig, *, apply_payload_metadata: bool = False) -> None:
        self.config = config
        self.apply_payload_metadata = apply_payload_metadata
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._payload: dict[str, str | None] = {}

    def request_start(self, *, area: str, session: str | None, source: str = "operator") -> dict[str, Any]:
        with self._lock:
            if area:
                self._payload["area"] = area
            if session:
                self._payload["session"] = session
            self._payload["source"] = source
            self._event.set()
        return {
            "status": "starting",
            "message": "Accepted UI start request. Initial 360 degree scan will begin now.",
            "area": self.config.area,
            "session": self.config.session,
            "source": source,
        }

    def wait(self) -> None:
        self._event.wait()
        if not self.apply_payload_metadata:
            return
        with self._lock:
            area = self._payload.get("area")
            session = self._payload.get("session")
            source = self._payload.get("source")
        if area:
            self.config.area = area
        if session:
            self.config.session = session
        if source:
            self.config.source = source


class _GatedExplorationUIController(LocalExplorationUIController):
    def __init__(
        self,
        backend: ExplorationBackend,
        start_gate: _ExplorationStartGate,
        *,
        waypoint_navigator: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        waypoint_previewer: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        scan_performer: Callable[[], dict[str, Any]] | None = None,
        navigation_session_starter: Callable[[], dict[str, Any]] | None = None,
        navigation_session_stopper: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(
            backend,
            waypoint_navigator=waypoint_navigator,
            waypoint_previewer=waypoint_previewer,
            scan_performer=scan_performer,
            navigation_session_starter=navigation_session_starter,
            navigation_session_stopper=navigation_session_stopper,
        )
        self.start_gate = start_gate

    def start_explore(self, *, area: str, session: str | None = None, source: str = "operator") -> dict[str, Any]:
        return self.start_gate.request_start(area=area, session=session, source=source)


class ExplorationRunner:
    def __init__(
        self,
        config: SimExplorationConfig,
        backend: ExplorationBackend,
        *,
        start_gate: _ExplorationStartGate | None = None,
    ) -> None:
        self.config = config
        self.backend = backend
        self.start_gate = start_gate
        self._active_session: _ApartmentExplorationSession | RosExplorationSession | None = None
        self._active_session_kind: str | None = None
        self._active_session_lock = threading.RLock()

    def start_navigation_session(self) -> dict[str, Any]:
        if self.config.nav2_mode != "ros":
            return {"status": "unavailable", "reason": "Navigation-only sessions require real ROS/Nav2 mode."}
        with self._active_session_lock:
            if self._active_session is not None:
                return {
                    "status": "active",
                    "mode": self._active_session_kind or "unknown",
                    "reason": "A live ROS/Nav2 session is already active.",
                }
        map_payload = self.backend.get_map()
        if not isinstance(map_payload, dict):
            return {"status": "unavailable", "reason": "Load a saved environment before starting a nav session."}
        task = self.backend.begin_external_task(
            tool_id="navigation_session",
            area=self.config.area,
            session=self.config.session,
            source="operator",
            message="Starting navigation-only ROS/Nav2 session from loaded environment.",
        )
        edits = map_payload.get("artifacts", {}).get("manual_occupancy_edits")
        if isinstance(edits, dict):
            self.backend.set_occupancy_edits(str(task["task_id"]), edits)
            map_payload = self.backend.get_map() or map_payload
        session = RosExplorationSession(self.config, self.backend, str(task["task_id"]))
        with self._active_session_lock:
            self._active_session = session
            self._active_session_kind = "navigation_only"
        try:
            result = session.start_navigation_only(map_payload)
        except Exception as exc:
            with self._active_session_lock:
                if self._active_session is session:
                    self._active_session = None
                    self._active_session_kind = None
            close = getattr(session, "close", None)
            if callable(close):
                close()
            self.backend.fail_external_task(
                str(task["task_id"]),
                message=f"Navigation-only session failed: {exc}",
                result={"error": str(exc)},
            )
            return {"status": "failed", "reason": str(exc)}
        if result.get("status") != "active":
            with self._active_session_lock:
                if self._active_session is session:
                    self._active_session = None
                    self._active_session_kind = None
            close = getattr(session, "close", None)
            if callable(close):
                close()
            self.backend.fail_external_task(
                str(task["task_id"]),
                message=str(result.get("reason") or "Navigation-only session failed."),
                result=result,
            )
        return result

    def stop_navigation_session(self) -> dict[str, Any]:
        with self._active_session_lock:
            session = self._active_session
            kind = self._active_session_kind
            if session is None:
                return {"status": "inactive", "reason": "No live ROS/Nav2 session is active."}
            if kind != "navigation_only":
                return {"status": "unavailable", "reason": f"The active session is `{kind}`, not navigation-only."}
            self._active_session = None
            self._active_session_kind = None
        task_id = str(getattr(session, "task_id", ""))
        close = getattr(session, "close", None)
        if callable(close):
            close()
        if task_id:
            self.backend.complete_external_task(
                task_id,
                map_payload=self.backend.get_map() or {},
                message="Stopped navigation-only ROS/Nav2 session.",
                result={"mode": "navigation_only"},
            )
        return {"status": "stopped", "mode": "navigation_only"}

    def navigate_to_waypoint(self, pose: dict[str, Any]) -> dict[str, Any]:
        with self._active_session_lock:
            session = self._active_session
        if session is None:
            return {
                "status": "unavailable",
                "reason": (
                    "No live navigation session is running. Load an environment and click Start Nav Session, "
                    "or keep an exploration session alive with --pause-for-operator-approval."
                ),
            }
        navigate = getattr(session, "navigate_to_manual_waypoint", None)
        if not callable(navigate):
            return {"status": "unavailable", "reason": "The active exploration session does not support waypoint navigation."}
        return navigate(pose)

    def preview_waypoint(self, pose: dict[str, Any]) -> dict[str, Any]:
        with self._active_session_lock:
            session = self._active_session
        if session is None:
            return {
                "status": "unavailable",
                "reason": (
                    "No live navigation session is running. Load an environment and click Start Nav Session, "
                    "or keep an exploration session alive with --pause-for-operator-approval."
                ),
            }
        preview = getattr(session, "preview_manual_waypoint", None)
        if not callable(preview):
            return {"status": "unavailable", "reason": "The active exploration session does not support waypoint previews."}
        return preview(pose)

    def perform_manual_scan(self) -> dict[str, Any]:
        with self._active_session_lock:
            session = self._active_session
        if session is None:
            return {
                "status": "unavailable",
                "reason": "No live exploration session is running yet. Wait for Start Explore to begin the session.",
            }
        perform_scan = getattr(session, "perform_manual_scan", None)
        if not callable(perform_scan):
            return {"status": "unavailable", "reason": "The active exploration session does not support manual scans."}
        return perform_scan()

    def run(self) -> dict[str, Any]:
        if self.start_gate is not None:
            self.start_gate.wait()
        task = self.backend.begin_external_task(
            tool_id="explore",
            area=self.config.area,
            session=self.config.session,
            source=self.config.source,
            message="Starting agentic apartment exploration run.",
        )
        session: _ApartmentExplorationSession | RosExplorationSession
        if self.config.nav2_mode == "ros":
            session = RosExplorationSession(self.config, self.backend, str(task["task_id"]))
        else:
            session = _ApartmentExplorationSession(self.config, self.backend, str(task["task_id"]))
        with self._active_session_lock:
            self._active_session = session
            self._active_session_kind = "exploration"
        try:
            map_payload = session.run()
        except Exception as exc:
            self.backend.fail_external_task(
                str(task["task_id"]),
                message=f"Exploration backend failed: {exc}",
                result={"error": str(exc)},
            )
            raise
        finally:
            with self._active_session_lock:
                if self._active_session is session:
                    self._active_session = None
                    self._active_session_kind = None
            close = getattr(session, "close", None)
            if callable(close):
                close()

        self.backend.complete_external_task(
            str(task["task_id"]),
            map_payload=map_payload,
            message=(
                f"Completed apartment exploration with coverage {map_payload['coverage']:.3f} "
                f"and {len(map_payload['regions'])} semantic region(s)."
            ),
            result={
                "map": map_payload,
                "coverage": map_payload["coverage"],
                "region_count": len(map_payload["regions"]),
                "decision_count": len(map_payload["artifacts"]["decision_log"]),
            },
        )
        return self.backend.snapshot()


ManiSkillExplorationRunner = ExplorationRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the simulated XLeRobot apartment exploration backend with frontier memory, "
            "occupancy mapping, and an optional LLM frontier-selection policy."
        )
    )
    parser.add_argument("--repo-root", default=str(resolve_xlerobot_repo_root()))
    parser.add_argument("--persist-path", default="./artifacts/xlerobot_exploration_map.json")
    parser.add_argument("--memory-root", default=None)
    parser.add_argument("--restore-persisted-state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--area", default="apartment")
    parser.add_argument("--session", default="house_v1")
    parser.add_argument("--source", default="operator")
    parser.add_argument("--env-id", default="SceneManipulation-v1")
    parser.add_argument("--robot-uid", default="xlerobot")
    parser.add_argument("--control-mode", default="pd_joint_delta_pos_dual_arm")
    parser.add_argument("--render-mode", default="human")
    parser.add_argument("--shader", default="default")
    parser.add_argument("--sim-backend", default="auto")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--force-reload", action="store_true")
    parser.add_argument("--occupancy-resolution", type=float, default=0.25)
    parser.add_argument("--max-control-steps", type=int, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--show-cameras", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-rerun", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--camera-log-stride", type=int, default=2)
    parser.add_argument("--realtime-sleep-s", type=float, default=0.01)
    parser.add_argument("--explorer-policy", choices=("heuristic", "llm"), default="llm")
    parser.add_argument("--llm-provider", default="mock")
    parser.add_argument("--llm-model", default="mock")
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--llm-temperature", type=float, default=0.1)
    parser.add_argument("--llm-max-tokens", type=int, default=1200)
    parser.add_argument("--llm-reasoning-effort", default=None)
    parser.add_argument("--trace-policy-stdout", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trace-llm-stdout", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--serve-review-ui", action="store_true")
    parser.add_argument("--review-host", default="127.0.0.1")
    parser.add_argument("--review-port", type=int, default=8770)
    parser.add_argument("--review-ui-flavor", choices=("user", "developer"), default="user")
    parser.add_argument("--open-browser", action=argparse.BooleanOptionalAction, default=False)
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
            "Defaults to robot radius plus clearance."
        ),
    )
    parser.add_argument("--finish-coverage-threshold", type=float, default=0.96)
    parser.add_argument("--max-decisions", type=int, default=32)
    parser.add_argument("--nav2-mode", choices=("simulated", "ros"), default="simulated")
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
    parser.add_argument(
        "--ros-map-updates-topic",
        default=None,
        help="OccupancyGridUpdate topic paired with --ros-map-topic. Defaults to '<ros-map-topic>_updates'.",
    )
    parser.add_argument(
        "--ros-fuse-external-projected-map-snapshots",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Fuse external /projected_map snapshots captured during camera-pan scans. "
            "Keep disabled for OctoMap, where /projected_map plus updates already contains accumulated evidence."
        ),
    )
    parser.add_argument("--ros-scan-topic", default="/scan")
    parser.add_argument("--ros-point-cloud-topic", default="/camera/head/points")
    parser.add_argument("--ros-scan-active-topic", default="/xlerobot/scan_active")
    parser.add_argument("--ros-nav-active-topic", default="/xlerobot/nav_active")
    parser.add_argument("--ros-scan-active-release-delay-s", type=float, default=3.0)
    parser.add_argument("--ros-rgb-topic", default="/camera/head/image_raw")
    parser.add_argument("--ros-imu-topic", default="/imu/filtered_yaw")
    parser.add_argument("--ros-cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--ros-map-frame", default="map")
    parser.add_argument("--ros-adapter-url", default=None)
    parser.add_argument("--ros-adapter-timeout-s", type=float, default=30.0)
    parser.add_argument("--ros-odom-frame", default="odom")
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
    parser.add_argument("--camera-pan-step-deg", type=float, default=60.0)
    parser.add_argument("--camera-pan-compute-s", type=float, default=2.0)
    parser.add_argument("--camera-pan-sample-count", type=int, default=12)
    parser.add_argument("--point-cloud-range-min-m", type=float, default=0.25)
    parser.add_argument("--point-cloud-range-max-m", type=float, default=4.0)
    parser.add_argument("--point-cloud-floor-free-max-z-m", type=float, default=0.08)
    parser.add_argument("--point-cloud-obstacle-min-z-m", type=float, default=0.08)
    parser.add_argument("--point-cloud-robot-clearance-height-m", type=float, default=1.50)
    parser.add_argument("--point-cloud-obstacle-max-z-m", type=float, default=1.80)
    parser.add_argument("--point-cloud-max-rays", type=int, default=2400)
    parser.add_argument("--sim-motion-speed", choices=("normal", "faster", "fastest"), default="normal")
    parser.add_argument("--ros-allow-multiple-action-servers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--semantic-waypoints-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--automatic-semantic-waypoints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show and enable the parked automatic semantic waypoint/VLM path. Manual regions are the default.",
    )
    parser.add_argument("--semantic-llm-provider", default=None)
    parser.add_argument("--semantic-llm-model", default=None)
    parser.add_argument("--semantic-llm-base-url", default=None)
    parser.add_argument("--semantic-llm-api-key", default=None)
    parser.add_argument("--semantic-vlm-async", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wait-for-ui-start", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--pause-for-operator-approval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pause after the initial scan and after each completed exploration step until the operator clicks Resume.",
    )
    parser.add_argument(
        "--stop-after-initial-scan",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Perform the initial 360 degree scan, publish the map state, and exit before frontier navigation.",
    )
    parser.add_argument(
        "--experimental-free-space-semantic-waypoints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Deprecated debugging path: allow the frontier policy to emit geometry/frontier-derived semantic "
            "area candidates. Normal exploration keeps this disabled so semantic places come from RGB-D evidence."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    backend = ExplorationBackend(
        ExplorationBackendConfig(
            mode="ros" if args.nav2_mode == "ros" else "sim",
            persist_path=args.persist_path,
            memory_root_path=args.memory_root,
            restore_persisted_state=args.restore_persisted_state,
            occupancy_resolution=args.occupancy_resolution,
        )
    )
    runner = ExplorationRunner(
        SimExplorationConfig(
            repo_root=args.repo_root,
            persist_path=args.persist_path,
            env_id=args.env_id,
            robot_uid=args.robot_uid,
            control_mode=args.control_mode,
            render_mode=args.render_mode,
            shader=args.shader,
            sim_backend=args.sim_backend,
            num_envs=args.num_envs,
            force_reload=args.force_reload,
            area=args.area,
            session=args.session,
            source=args.source,
            occupancy_resolution=args.occupancy_resolution,
            max_control_steps=args.max_control_steps,
            max_episode_steps=args.max_episode_steps,
            show_cameras=args.show_cameras,
            use_rerun=args.use_rerun,
            camera_log_stride=args.camera_log_stride,
            realtime_sleep_s=args.realtime_sleep_s,
            explorer_policy=args.explorer_policy,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            llm_base_url=args.llm_base_url,
            llm_api_key=args.llm_api_key,
            llm_temperature=args.llm_temperature,
            llm_max_tokens=args.llm_max_tokens,
            llm_reasoning_effort=args.llm_reasoning_effort,
            trace_policy_stdout=args.trace_policy_stdout,
            trace_llm_stdout=args.trace_llm_stdout,
            serve_review_ui=args.serve_review_ui,
            review_host=args.review_host,
            review_port=args.review_port,
            review_ui_flavor=args.review_ui_flavor,
            sensor_range_m=args.sensor_range_m,
            robot_radius_m=args.robot_radius_m,
            frontier_min_opening_m=args.frontier_min_opening_m,
            visited_frontier_filter_radius_m=args.visited_frontier_filter_radius_m,
            finish_coverage_threshold=args.finish_coverage_threshold,
            max_decisions=args.max_decisions,
            nav2_mode=args.nav2_mode,
            nav2_planner_id=args.nav2_planner_id,
            nav2_controller_id=args.nav2_controller_id,
            nav2_behavior_tree=args.nav2_behavior_tree,
            nav2_recovery_enabled=args.nav2_recovery_enabled,
            ros_navigation_map_source=args.ros_navigation_map_source,
            ros_map_topic=args.ros_map_topic,
            ros_map_updates_topic=args.ros_map_updates_topic,
            ros_fuse_external_projected_map_snapshots=args.ros_fuse_external_projected_map_snapshots,
            ros_scan_topic=args.ros_scan_topic,
            ros_point_cloud_topic=args.ros_point_cloud_topic,
            ros_scan_active_topic=args.ros_scan_active_topic,
            ros_nav_active_topic=args.ros_nav_active_topic,
            ros_scan_active_release_delay_s=args.ros_scan_active_release_delay_s,
            ros_rgb_topic=args.ros_rgb_topic,
            ros_imu_topic=args.ros_imu_topic,
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
            camera_pan_step_deg=args.camera_pan_step_deg,
            camera_pan_compute_s=args.camera_pan_compute_s,
            camera_pan_sample_count=args.camera_pan_sample_count,
            point_cloud_range_min_m=args.point_cloud_range_min_m,
            point_cloud_range_max_m=args.point_cloud_range_max_m,
            point_cloud_floor_free_max_z_m=args.point_cloud_floor_free_max_z_m,
            point_cloud_obstacle_min_z_m=args.point_cloud_obstacle_min_z_m,
            point_cloud_robot_clearance_height_m=args.point_cloud_robot_clearance_height_m,
            point_cloud_obstacle_max_z_m=args.point_cloud_obstacle_max_z_m,
            point_cloud_max_rays=args.point_cloud_max_rays,
            sim_motion_speed=args.sim_motion_speed,
            ros_allow_multiple_action_servers=args.ros_allow_multiple_action_servers,
            experimental_free_space_semantic_waypoints=args.experimental_free_space_semantic_waypoints,
            semantic_waypoints_enabled=args.semantic_waypoints_enabled,
            automatic_semantic_waypoints=args.automatic_semantic_waypoints,
            semantic_llm_provider=args.semantic_llm_provider,
            semantic_llm_model=args.semantic_llm_model,
            semantic_llm_base_url=args.semantic_llm_base_url,
            semantic_llm_api_key=args.semantic_llm_api_key,
            semantic_vlm_async=args.semantic_vlm_async,
            wait_for_ui_start=args.wait_for_ui_start,
            pause_for_operator_approval=args.pause_for_operator_approval,
            stop_after_initial_scan=args.stop_after_initial_scan,
        ),
        backend,
    )
    server: ExplorationReviewServer | None = None
    start_gate: _ExplorationStartGate | None = None
    if args.serve_review_ui:
        if args.wait_for_ui_start:
            start_gate = _ExplorationStartGate(runner.config)
            controller = _GatedExplorationUIController(
                backend,
                start_gate,
                waypoint_navigator=runner.navigate_to_waypoint,
                waypoint_previewer=runner.preview_waypoint,
                scan_performer=runner.perform_manual_scan,
                navigation_session_starter=runner.start_navigation_session,
                navigation_session_stopper=runner.stop_navigation_session,
            )
        else:
            controller = LocalExplorationUIController(
                backend,
                waypoint_navigator=runner.navigate_to_waypoint,
                waypoint_previewer=runner.preview_waypoint,
                scan_performer=runner.perform_manual_scan,
                navigation_session_starter=runner.start_navigation_session,
                navigation_session_stopper=runner.stop_navigation_session,
            )
        server = ExplorationReviewServer(
            controller,
            host=args.review_host,
            port=args.review_port,
            allow_task_controls=False,
            allow_task_launch_controls=bool(args.wait_for_ui_start),
            allow_task_state_controls=True,
            allow_map_approval=True,
            ui_flavor=args.review_ui_flavor,
        )
        server.serve_in_background()
        print(
            f"XLeRobot exploration review UI: http://{args.review_host}:{args.review_port} "
            f"(flavor={args.review_ui_flavor})"
        )
        if args.open_browser:
            webbrowser.open(f"http://{args.review_host}:{args.review_port}")
    if args.wait_for_ui_start and start_gate is None:
        print("`--wait-for-ui-start` was set without `--serve-review-ui`; starting immediately.")
    runner.start_gate = start_gate
    try:
        snapshot = runner.run()
        current_map = snapshot.get("current_map") or {}
        print(
            f"Saved exploration map `{current_map.get('map_id', args.session)}` "
            f"with {len(current_map.get('regions', []))} region(s), "
            f"coverage {current_map.get('coverage', 0.0)}, to {args.persist_path}."
        )
        if args.serve_review_ui:
            print("Review UI remains live for pause/edit/approval inspection. Press Ctrl-C to exit.")
            try:
                while True:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                pass
    finally:
        if server is not None:
            server.shutdown()
    return 0


def _build_simple_apartment(resolution: float) -> ApartmentScenario:
    width_m = 11.0
    height_m = 8.5
    width_cells = int(math.ceil(width_m / resolution))
    height_cells = int(math.ceil(height_m / resolution))
    free_cells: set[GridCell] = set()
    room_by_cell: dict[GridCell, str] = {}

    def carve(room_id: str, bounds: tuple[float, float, float, float]) -> None:
        x0, y0, x1, y1 = bounds
        for cell in _rect_cells(bounds, resolution, width_cells, height_cells):
            free_cells.add(cell)
            room_by_cell[cell] = room_id

    room_bounds = {
        "region_hallway": (4.25, 1.0, 6.25, 7.5),
        "region_living_room": (0.75, 0.75, 4.25, 3.75),
        "region_bathroom": (0.75, 4.5, 3.25, 7.5),
        "region_kitchen": (6.25, 4.5, 10.25, 7.5),
        "region_bedroom": (6.25, 0.75, 10.25, 3.75),
    }
    door_bounds = {
        "region_hallway": [
            (4.25, 2.0, 4.75, 2.75),
            (3.25, 5.5, 4.25, 6.25),
            (6.25, 5.5, 6.75, 6.25),
            (6.25, 2.0, 6.75, 2.75),
        ]
    }
    for room_id, bounds in room_bounds.items():
        carve(room_id, bounds)
    for bounds in door_bounds["region_hallway"]:
        carve("region_hallway", bounds)

    obstacle_cells: set[GridCell] = set()
    for bounds in (
        (2.0, 1.75, 2.75, 2.5),
        (7.75, 5.5, 8.75, 6.25),
        (7.25, 1.25, 9.5, 2.75),
        (1.25, 5.0, 2.5, 6.25),
    ):
        for cell in _rect_cells(bounds, resolution, width_cells, height_cells):
            if cell in free_cells:
                obstacle_cells.add(cell)
                free_cells.discard(cell)
                room_by_cell.pop(cell, None)

    rooms = {
        "region_hallway": RoomSpec(
            region_id="region_hallway",
            label="hallway",
            polygon_2d=_polygon_from_bounds(room_bounds["region_hallway"]),
            adjacency=("region_living_room", "region_bathroom", "region_kitchen", "region_bedroom"),
            objects=("charging dock", "shoe rack", "wall art"),
            descriptions=("a central hallway that connects the apartment rooms",),
            entry_pose=Pose2D(5.25, 3.0, 0.0),
            center_pose=Pose2D(5.25, 4.25, 0.0),
        ),
        "region_living_room": RoomSpec(
            region_id="region_living_room",
            label="living_room",
            polygon_2d=_polygon_from_bounds(room_bounds["region_living_room"]),
            adjacency=("region_hallway",),
            objects=("sofa", "tv", "coffee table", "desk", "monitor", "chair"),
            descriptions=("a living room with a sofa, television wall, and a small desk corner",),
            entry_pose=Pose2D(4.5, 2.25, 0.0),
            center_pose=Pose2D(2.5, 2.25, 0.0),
            sub_areas=(
                SubAreaSpec(
                    area_id="subarea_living_room_office",
                    label="office_area",
                    polygon_2d=_polygon_from_bounds((0.75, 0.75, 2.25, 1.75)),
                    evidence=("desk visible", "monitor visible", "chair visible"),
                ),
            ),
        ),
        "region_bathroom": RoomSpec(
            region_id="region_bathroom",
            label="bathroom",
            polygon_2d=_polygon_from_bounds(room_bounds["region_bathroom"]),
            adjacency=("region_hallway",),
            objects=("sink", "mirror", "toilet", "bathtub"),
            descriptions=("a compact bathroom with a sink, mirror, toilet, and bathtub",),
            entry_pose=Pose2D(3.75, 5.75, 0.0),
            center_pose=Pose2D(2.0, 6.0, 0.0),
        ),
        "region_kitchen": RoomSpec(
            region_id="region_kitchen",
            label="kitchen",
            polygon_2d=_polygon_from_bounds(room_bounds["region_kitchen"]),
            adjacency=("region_hallway",),
            objects=("fridge", "sink", "oven", "counter", "island"),
            descriptions=("a kitchen with a fridge, oven, sink, counter, and central island",),
            entry_pose=Pose2D(6.5, 5.75, 0.0),
            center_pose=Pose2D(8.25, 6.0, 0.0),
        ),
        "region_bedroom": RoomSpec(
            region_id="region_bedroom",
            label="bedroom",
            polygon_2d=_polygon_from_bounds(room_bounds["region_bedroom"]),
            adjacency=("region_hallway",),
            objects=("bed", "wardrobe", "lamp", "nightstand"),
            descriptions=("a bedroom with a bed, wardrobe, and bedside furniture",),
            entry_pose=Pose2D(6.5, 2.25, 0.0),
            center_pose=Pose2D(8.25, 2.25, 0.0),
        ),
    }

    objects = [
        WorldObject("obj_dock", "charging dock", "region_hallway", GridCell(int(4.75 / resolution), int(6.5 / resolution)), "robot charging dock"),
        WorldObject("obj_shoes", "shoe rack", "region_hallway", GridCell(int(5.5 / resolution), int(1.5 / resolution)), "shoe rack by the wall"),
        WorldObject("obj_sofa", "sofa", "region_living_room", GridCell(int(1.5 / resolution), int(2.75 / resolution)), "large sofa"),
        WorldObject("obj_tv", "tv", "region_living_room", GridCell(int(3.5 / resolution), int(3.0 / resolution)), "television on the far wall"),
        WorldObject("obj_coffee", "coffee table", "region_living_room", GridCell(int(2.25 / resolution), int(1.5 / resolution)), "low coffee table"),
        WorldObject("obj_desk", "desk", "region_living_room", GridCell(int(1.25 / resolution), int(1.25 / resolution)), "small office desk"),
        WorldObject("obj_monitor", "monitor", "region_living_room", GridCell(int(1.5 / resolution), int(1.0 / resolution)), "computer monitor"),
        WorldObject("obj_chair", "chair", "region_living_room", GridCell(int(1.75 / resolution), int(1.25 / resolution)), "desk chair"),
        WorldObject("obj_sink_bath", "sink", "region_bathroom", GridCell(int(2.75 / resolution), int(6.75 / resolution)), "bathroom sink"),
        WorldObject("obj_mirror", "mirror", "region_bathroom", GridCell(int(2.75 / resolution), int(7.0 / resolution)), "bathroom mirror"),
        WorldObject("obj_toilet", "toilet", "region_bathroom", GridCell(int(2.5 / resolution), int(5.5 / resolution)), "toilet"),
        WorldObject("obj_tub", "bathtub", "region_bathroom", GridCell(int(1.25 / resolution), int(5.5 / resolution)), "bathtub"),
        WorldObject("obj_fridge", "fridge", "region_kitchen", GridCell(int(9.5 / resolution), int(6.75 / resolution)), "kitchen fridge"),
        WorldObject("obj_sink_kitchen", "sink", "region_kitchen", GridCell(int(9.0 / resolution), int(5.0 / resolution)), "kitchen sink"),
        WorldObject("obj_oven", "oven", "region_kitchen", GridCell(int(7.0 / resolution), int(4.75 / resolution)), "kitchen oven"),
        WorldObject("obj_counter", "counter", "region_kitchen", GridCell(int(8.0 / resolution), int(4.75 / resolution)), "counter workspace"),
        WorldObject("obj_island", "island", "region_kitchen", GridCell(int(8.25 / resolution), int(6.0 / resolution)), "kitchen island"),
        WorldObject("obj_bed", "bed", "region_bedroom", GridCell(int(8.25 / resolution), int(2.25 / resolution)), "bed"),
        WorldObject("obj_wardrobe", "wardrobe", "region_bedroom", GridCell(int(9.5 / resolution), int(3.0 / resolution)), "wardrobe"),
        WorldObject("obj_lamp", "lamp", "region_bedroom", GridCell(int(6.75 / resolution), int(3.0 / resolution)), "floor lamp"),
        WorldObject("obj_nightstand", "nightstand", "region_bedroom", GridCell(int(7.0 / resolution), int(1.25 / resolution)), "nightstand"),
    ]

    free_cells -= obstacle_cells
    start_cell = GridCell(int(5.25 / resolution), int(4.25 / resolution))
    return ApartmentScenario(
        scenario_id="scenario_simple_apartment",
        layout_id="simple_apartment",
        resolution=resolution,
        width_cells=width_cells,
        height_cells=height_cells,
        start_cell=start_cell,
        free_cells=free_cells,
        obstacle_cells=obstacle_cells,
        room_by_cell=room_by_cell,
        rooms=rooms,
        objects=objects,
    )


def _simulate_scan(
    scenario: ApartmentScenario,
    origin_cell: GridCell,
    *,
    yaw: float,
    max_range_m: float,
    full_turnaround: bool,
) -> ScanResult:
    origin_pose = origin_cell.center_pose(scenario.resolution, yaw=yaw)
    angles = _scan_angles(yaw, full_turnaround=full_turnaround)
    observed_free: set[GridCell] = set()
    observed_occupied: set[GridCell] = set()
    range_edge_frontiers: set[GridCell] = set()
    depth_hits: list[float] = []
    substep_m = scenario.resolution * 0.5
    max_steps = max(1, int(max_range_m / substep_m))

    for angle in angles:
        last_free: GridCell | None = None
        hit_obstacle = False
        seen_cells: set[GridCell] = set()
        for step in range(1, max_steps + 1):
            distance_m = step * substep_m
            x = origin_pose.x + math.cos(angle) * distance_m
            y = origin_pose.y + math.sin(angle) * distance_m
            cell = scenario.world_to_cell(x, y)
            if cell in seen_cells:
                continue
            seen_cells.add(cell)
            if not scenario.in_bounds(cell):
                break
            if scenario.is_occupied(cell):
                observed_occupied.add(cell)
                depth_hits.append(distance_m)
                hit_obstacle = True
                break
            observed_free.add(cell)
            last_free = cell
        if not hit_obstacle and last_free is not None:
            range_edge_frontiers.add(last_free)

    visible_objects: list[str] = []
    visible_room_ids: set[str] = set()
    for cell in observed_free:
        room_id = scenario.room_for_cell(cell)
        if room_id:
            visible_room_ids.add(room_id)

    for item in scenario.objects:
        if item.cell == origin_cell:
            visible_objects.append(item.label)
            continue
        distance_m = _grid_distance_cells(origin_cell, item.cell) * scenario.resolution
        if distance_m > max_range_m:
            continue
        if not full_turnaround and _angle_difference(yaw, math.atan2(item.cell.y - origin_cell.y, item.cell.x - origin_cell.x)) > math.radians(60):
            continue
        if _line_of_sight_clear(scenario, origin_cell, item.cell):
            visible_objects.append(item.label)
            depth_hits.append(distance_m)
            visible_room_ids.add(item.room_id)

    visible_objects = sorted(set(visible_objects))
    current_room_id = scenario.room_for_cell(origin_cell)
    room_label = scenario.rooms.get(current_room_id).label if current_room_id in scenario.rooms else "unknown_space"
    description = (
        f"{'360 degree' if full_turnaround else 'forward'} scan from {room_label}. "
        f"Visible objects: {', '.join(visible_objects[:6]) or 'none'}."
    )
    thumbnail = _thumbnail_data_url(
        title=room_label.replace("_", " ").title(),
        subtitle=", ".join(visible_objects[:3]) or "mapping scan",
    )
    return ScanResult(
        observed_free=observed_free,
        observed_occupied=observed_occupied,
        range_edge_frontiers=range_edge_frontiers,
        visible_objects=visible_objects,
        visible_room_ids=sorted(visible_room_ids),
        point_count=max((len(observed_free) + len(observed_occupied)) * 18, 900),
        depth_min_m=round(min(depth_hits), 3) if depth_hits else 0.35,
        depth_max_m=round(max(depth_hits), 3) if depth_hits else max_range_m,
        description=description,
        thumbnail_data_url=thumbnail,
    )


def _scan_angles(yaw: float, *, full_turnaround: bool) -> list[float]:
    if full_turnaround:
        return [math.radians(index * 5.0) for index in range(72)]
    return [yaw + math.radians(offset) for offset in range(-55, 56, 5)]


def _rect_cells(bounds: tuple[float, float, float, float], resolution: float, width_cells: int, height_cells: int) -> set[GridCell]:
    x0, y0, x1, y1 = bounds
    cells: set[GridCell] = set()
    for x in range(width_cells):
        for y in range(height_cells):
            center_x = (x + 0.5) * resolution
            center_y = (y + 0.5) * resolution
            if x0 <= center_x <= x1 and y0 <= center_y <= y1:
                cells.add(GridCell(x, y))
    return cells


def _polygon_from_bounds(bounds: tuple[float, float, float, float]) -> tuple[tuple[float, float], ...]:
    x0, y0, x1, y1 = bounds
    return (
        (round(x0, 3), round(y0, 3)),
        (round(x1, 3), round(y0, 3)),
        (round(x1, 3), round(y1, 3)),
        (round(x0, 3), round(y1, 3)),
    )


def _neighbors4(cell: GridCell) -> tuple[GridCell, ...]:
    return (
        GridCell(cell.x + 1, cell.y),
        GridCell(cell.x - 1, cell.y),
        GridCell(cell.x, cell.y + 1),
        GridCell(cell.x, cell.y - 1),
    )


def _neighbors8(cell: GridCell) -> tuple[GridCell, ...]:
    return tuple(
        GridCell(cell.x + dx, cell.y + dy)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if not (dx == 0 and dy == 0)
    )


def _grid_distance_cells(a: GridCell, b: GridCell) -> int:
    return abs(a.x - b.x) + abs(a.y - b.y)


def _centroid_cell(cells: Iterable[GridCell]) -> GridCell:
    normalized = list(cells)
    if not normalized:
        return GridCell(0, 0)
    mean_x = round(sum(cell.x for cell in normalized) / len(normalized))
    mean_y = round(sum(cell.y for cell in normalized) / len(normalized))
    return GridCell(mean_x, mean_y)


def _cell_mean_pose(cells: Iterable[GridCell], resolution: float, *, yaw: float = 0.0) -> Pose2D:
    normalized = list(cells)
    if not normalized:
        return Pose2D(0.0, 0.0, yaw)
    mean_x = sum((cell.x + 0.5) * resolution for cell in normalized) / len(normalized)
    mean_y = sum((cell.y + 0.5) * resolution for cell in normalized) / len(normalized)
    return Pose2D(mean_x, mean_y, yaw)


def _frontier_opening_width_m(cells: Iterable[GridCell], resolution: float) -> float:
    normalized = list(cells)
    if not normalized:
        return 0.0
    span_x_cells = max(cell.x for cell in normalized) - min(cell.x for cell in normalized) + 1
    span_y_cells = max(cell.y for cell in normalized) - min(cell.y for cell in normalized) + 1
    # Bound by member count so sparse diagonal/noisy components do not look wider than their evidence.
    effective_width_cells = min(max(span_x_cells, span_y_cells), len(normalized))
    return max(0.0, effective_width_cells * resolution)


def _search_known_safe_path(start: GridCell, goal: GridCell, traversable: set[GridCell]) -> list[GridCell]:
    if start == goal:
        return [start]
    if start not in traversable or goal not in traversable:
        return []
    open_heap: list[tuple[int, int, GridCell]] = []
    heapq.heappush(open_heap, (_grid_distance_cells(start, goal), 0, start))
    came_from: dict[GridCell, GridCell] = {}
    g_score: dict[GridCell, int] = {start: 0}
    seen: set[GridCell] = set()
    while open_heap:
        _, cost, current = heapq.heappop(open_heap)
        if current in seen:
            continue
        seen.add(current)
        if current == goal:
            return _reconstruct_path(came_from, current)
        for neighbor in _neighbors4(current):
            if neighbor not in traversable:
                continue
            tentative = cost + 1
            if tentative < g_score.get(neighbor, 1_000_000):
                g_score[neighbor] = tentative
                came_from[neighbor] = current
                heapq.heappush(open_heap, (tentative + _grid_distance_cells(neighbor, goal), tentative, neighbor))
    return []


def _reconstruct_path(came_from: dict[GridCell, GridCell], current: GridCell) -> list[GridCell]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def _line_of_sight_clear(scenario: ApartmentScenario, start: GridCell, goal: GridCell) -> bool:
    start_pose = start.center_pose(scenario.resolution)
    goal_pose = goal.center_pose(scenario.resolution)
    distance = math.hypot(goal_pose.x - start_pose.x, goal_pose.y - start_pose.y)
    if distance <= 1e-6:
        return True
    steps = max(1, int(distance / (scenario.resolution * 0.35)))
    seen: set[GridCell] = set()
    for step in range(1, steps):
        ratio = step / steps
        x = start_pose.x + (goal_pose.x - start_pose.x) * ratio
        y = start_pose.y + (goal_pose.y - start_pose.y) * ratio
        cell = scenario.world_to_cell(x, y)
        if cell in seen or cell == start or cell == goal:
            continue
        seen.add(cell)
        if scenario.is_occupied(cell):
            return False
    return True


def _angle_difference(a: float, b: float) -> float:
    delta = (a - b + math.pi) % (2 * math.pi) - math.pi
    return abs(delta)


def _unwrap_yaw_sequence(yaws: Iterable[float]) -> list[float]:
    sequence = list(yaws)
    if not sequence:
        return []
    unwrapped = [float(sequence[0])]
    previous = float(sequence[0])
    for value in sequence[1:]:
        current = float(value)
        delta = math.atan2(math.sin(current - previous), math.cos(current - previous))
        unwrapped.append(unwrapped[-1] + delta)
        previous = current
    return unwrapped


def _select_turnaround_scan_observations(
    observations: list[dict[str, Any]],
    *,
    sample_count: int,
) -> list[dict[str, Any]]:
    if len(observations) <= sample_count:
        return list(observations)
    poses = [item.get("pose") for item in observations]
    if not all(isinstance(pose, Pose2D) for pose in poses):
        stride = max(len(observations) // max(sample_count, 1), 1)
        return observations[::stride][:sample_count]
    yaws = _unwrap_yaw_sequence([float(pose.yaw) for pose in poses if isinstance(pose, Pose2D)])
    if len(yaws) != len(observations):
        stride = max(len(observations) // max(sample_count, 1), 1)
        return observations[::stride][:sample_count]
    span = yaws[-1] - yaws[0]
    if abs(span) < math.radians(45.0):
        stride = max(len(observations) // max(sample_count, 1), 1)
        return observations[::stride][:sample_count]
    direction = 1.0 if span >= 0.0 else -1.0
    usable_span = min(abs(span), math.tau)
    start_yaw = yaws[0]
    if sample_count <= 1:
        targets = [start_yaw]
    else:
        targets = [
            start_yaw + direction * (usable_span * index / (sample_count - 1))
            for index in range(sample_count)
        ]
    chosen_indices: set[int] = set()
    for target in targets:
        best_index = min(
            range(len(yaws)),
            key=lambda index: (abs(yaws[index] - target), abs(index - len(yaws) // 2)),
        )
        chosen_indices.add(best_index)
    if 0 not in chosen_indices:
        chosen_indices.add(0)
    if len(observations) - 1 not in chosen_indices:
        chosen_indices.add(len(observations) - 1)
    ordered = sorted(chosen_indices)
    if len(ordered) > sample_count:
        stride = max(len(ordered) / float(sample_count), 1.0)
        compacted = [ordered[min(int(round(index * stride)), len(ordered) - 1)] for index in range(sample_count)]
        ordered = sorted(set(compacted))
    return [observations[index] for index in ordered]


def _pose_distance_m(a: Pose2D, b: Pose2D) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _goal_status_from_outcome(value: Any) -> int:
    if isinstance(value, dict):
        try:
            return int(value.get("status", GoalStatus.STATUS_UNKNOWN))
        except Exception:
            return GoalStatus.STATUS_UNKNOWN
    return int(getattr(value, "status", GoalStatus.STATUS_UNKNOWN))


def _pose_from_mapping(value: Any) -> Pose2D | None:
    if not isinstance(value, dict):
        return None
    try:
        return Pose2D(
            float(value["x"]),
            float(value["y"]),
            float(value.get("yaw", 0.0)),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _clamp_float(value: Any, *, default: float, low: float, high: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return min(max(parsed, low), high)


def _thumbnail_data_url(*, title: str, subtitle: str) -> str:
    import base64

    svg = f"""
<svg xmlns="http://www.w3.org/2000/svg" width="320" height="180" viewBox="0 0 320 180">
  <defs>
    <linearGradient id="g" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0%" stop-color="#0f766e"/>
      <stop offset="100%" stop-color="#9a3412"/>
    </linearGradient>
  </defs>
  <rect width="320" height="180" rx="20" fill="#f8fafc"/>
  <rect x="14" y="14" width="292" height="152" rx="16" fill="url(#g)" opacity="0.16"/>
  <text x="28" y="64" font-family="IBM Plex Sans, Arial, sans-serif" font-size="28" fill="#102a43">{title}</text>
  <text x="28" y="98" font-family="IBM Plex Sans, Arial, sans-serif" font-size="14" fill="#486581">{subtitle}</text>
  <circle cx="250" cy="58" r="20" fill="#0f766e" opacity="0.2"/>
  <circle cx="218" cy="118" r="34" fill="#9a3412" opacity="0.12"/>
</svg>
""".strip()
    encoded = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{encoded}"


def _frontier_selection_guidance() -> list[str]:
    return [
        "Treat frontier coordinates as partial RGB-D-derived boundary information from what the camera has scanned, not as complete apartment knowledge or a command to visit every boundary.",
        "Select frontiers that likely expand robot-navigable floor space: doors, room entrances, hallway continuations, open areas, or meaningful sensor-range-limit expansions.",
        "`free_space_path_distance_m` is an approximate route through currently known free cells from the robot to the frontier approach pose. Locality is a primary objective: choose nearby useful frontiers first and avoid zig-zagging across the apartment.",
        "Do not choose a frontier more than about 2x farther than another plausible frontier unless the navigation map and RGB views show it is clearly much more valuable, such as opening a major new room or corridor.",
        "Deterministic frontier candidates are proposals, not truth. They should have a robot-sized opening width, but you should still veto them if the navigation-map image and RGB views suggest already-mapped empty space, a wall sliver, or clutter.",
        "Use recent RGB views to interpret ambiguous frontier information, especially when geometry alone could be a doorway, furniture edge, or clutter shadow.",
        "Deprioritize frontiers that look like furniture-shadow boundaries behind or under couches, tables, cabinets, shelves, or clutter unless evidence suggests a real traversable opening.",
        "If a listed frontier looks wrong after overlaying the navigation map, frontier label, and camera images, do not select it; use memory_updates with `suppress` or `keep` to veto it explicitly.",
        "Use free_space_path_distance_m first, then unknown_gain, source, recent views, navigation-map image, and frontier memory; choose a far frontier only when it clearly offers substantially better navigable expansion than nearby alternatives.",
        "If remaining frontiers are reachable but likely not useful for robot navigation, explain that in reasoning_summary before finishing or choosing a better stored frontier.",
    ]


def _dedupe_text(items: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = str(item).strip()
        if not normalized or normalized in seen:
            continue
        output.append(normalized)
        seen.add(normalized)
    return output


def _frame_fallback_depth_m(frame: dict[str, Any]) -> float | None:
    depths: list[float] = []
    for key in ("depth_min_m", "depth_max_m"):
        value = frame.get(key)
        if value is None:
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed) and parsed > 0.0:
            depths.append(parsed)
    if len(depths) == 2:
        return sum(depths) / 2.0
    if depths:
        return depths[0]
    summary = frame.get("rgbd_summary")
    if isinstance(summary, dict):
        return _frame_fallback_depth_m(summary)
    return None


def _semantic_named_places_for_map(semantic_memory: dict[str, Any]) -> list[dict[str, Any]]:
    places: list[dict[str, Any]] = []
    for item in semantic_memory.get("named_places", []):
        if not isinstance(item, dict):
            continue
        anchor_pose = item.get("anchor_pose")
        if not isinstance(anchor_pose, dict):
            continue
        label = str(item.get("label") or item.get("place_id") or "").strip()
        if not label:
            continue
        places.append(
            {
                "name": label,
                "pose": anchor_pose,
                "place_id": item.get("place_id"),
                "confidence": item.get("confidence"),
                "evidence": item.get("evidence", []),
                "source": "semantic_memory",
            }
        )
    return places


def _aggregate_semantic_updates(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregated: dict[tuple[str, str], dict[str, Any]] = {}
    for item in items:
        label = str(item.get("label", "")).strip()
        target_id = str(item.get("target_id", "")).strip()
        if not label or not target_id:
            continue
        key = (label, target_id)
        existing = aggregated.get(key)
        confidence = float(item.get("confidence", 0.5))
        evidence = [str(token) for token in item.get("evidence", []) if str(token).strip()]
        candidate = {
            "area_id": f"semantic_{target_id}_{label}",
            "label": label,
            "kind": str(item.get("kind", "room_hint")),
            "target_id": target_id,
            "confidence": round(confidence, 3),
            "evidence": evidence[:4],
        }
        if existing is None or candidate["confidence"] > existing["confidence"]:
            aggregated[key] = candidate
    return list(aggregated.values())


if __name__ == "__main__":
    raise SystemExit(main())
