from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from xlerobot_agent.exploration import ExplorationBackend, ExplorationBackendConfig, Pose2D
from xlerobot_playground.sim_exploration_backend import (
    ExplorationDecision,
    ExplorationLLMPolicy,
    FrontierCandidate,
    FrontierMemory,
    FrontierRecord,
    GridCell,
    ManiSkillExplorationRunner,
    SimExplorationConfig,
    _occupancy_map_like_to_ros_map,
    _select_turnaround_scan_observations,
    _mark_frontier_unreachable_as_visited,
    build_parser,
)
from xlerobot_playground.interactive_exploration_playground import (
    HumanAssistanceRequired,
    InteractiveNoNav2ExplorationSession,
    ManiSkillNav2RouterExplorationSession,
    ManiSkillTeleportExplorationSession,
    Nav2LocalPathBlocked,
)
from xlerobot_playground.interactive_exploration_playground import (
    _local_scan_path_blocker,
    _navigation_map_data_url,
    _resolve_manishkill_start_pose,
    _updated_mobile_base_qpos,
    _zero_mobile_base_qvel,
    build_parser as build_interactive_playground_parser,
)
from xlerobot_playground.interactive_react_ui import INTERACTIVE_REACT_HTML
from xlerobot_playground.map_editing import (
    ACTIVE_RGBD_SCAN_FUSION_CONFIG,
    DEFAULT_OCCUPANCY_FUSION_CONFIG,
    EditableOccupancyMap,
    ManualOccupancyEdits,
    merge_occupancy_observation,
)
from xlerobot_playground.ros_nav2_adapter import (
    map_from_payload,
    pose_from_payload,
    scan_observation_from_payload,
    serialize_map,
    serialize_pose,
    serialize_scan_observation,
)
from xlerobot_playground.ros_nav2_runtime import RosOccupancyMap
from xlerobot_playground.point_cloud_fusion import (
    PointCloudFusionConfig,
    integrate_transformed_point_cloud_observation,
)
from xlerobot_playground.scan_fusion import integrate_planar_scan


class SimExplorationBackendTests(unittest.TestCase):
    def test_local_scan_path_guard_detects_obstacle_in_path_corridor(self) -> None:
        observation = {
            "pose": Pose2D(0.0, 0.0, 0.0),
            "range_min": 0.05,
            "range_max": 10.0,
            "angle_min": -0.2,
            "angle_increment": 0.1,
            "ranges": (10.0, 10.0, 0.4, 10.0, 10.0),
        }

        blocker = _local_scan_path_blocker(
            observation,
            current_pose=Pose2D(0.0, 0.0, 0.0),
            target_pose=Pose2D(2.0, 0.0, 0.0),
            robot_length_m=0.3913,
            robot_width_m=0.459,
        )

        self.assertIsNotNone(blocker)
        assert blocker is not None
        self.assertEqual(blocker["beam_index"], 2)
        self.assertLess(blocker["forward_distance_m"], 0.7)

    def test_local_scan_path_guard_ignores_obstacle_outside_corridor(self) -> None:
        observation = {
            "pose": Pose2D(0.0, 0.0, 0.0),
            "range_min": 0.05,
            "range_max": 10.0,
            "angle_min": 0.45,
            "angle_increment": 0.1,
            "ranges": (0.7, 10.0, 10.0),
        }

        blocker = _local_scan_path_blocker(
            observation,
            current_pose=Pose2D(0.0, 0.0, 0.0),
            target_pose=Pose2D(2.0, 0.0, 0.0),
            robot_length_m=0.3913,
            robot_width_m=0.459,
        )

        self.assertIsNone(blocker)

    def test_local_scan_path_guard_uses_padded_rectangular_footprint(self) -> None:
        observation = {
            "pose": Pose2D(0.0, 0.0, 0.0),
            "range_min": 0.05,
            "range_max": 10.0,
            "angle_min": 0.48,
            "angle_increment": 0.0,
            "ranges": (0.4,),
        }

        blocker = _local_scan_path_blocker(
            observation,
            current_pose=Pose2D(0.0, 0.0, 0.0),
            target_pose=Pose2D(2.0, 0.0, 0.0),
            robot_length_m=0.3913,
            robot_width_m=0.459,
            safety_padding_m=0.06,
        )

        self.assertIsNotNone(blocker)
        assert blocker is not None
        self.assertAlmostEqual(blocker["half_length_m"], 0.256, places=3)
        self.assertAlmostEqual(blocker["half_width_m"], 0.289, places=3)
        self.assertLess(blocker["lateral_distance_m"], blocker["half_width_m"])

    def test_local_scan_path_guard_rejects_point_outside_rectangular_width(self) -> None:
        observation = {
            "pose": Pose2D(0.0, 0.0, 0.0),
            "range_min": 0.05,
            "range_max": 10.0,
            "angle_min": 0.62,
            "angle_increment": 0.0,
            "ranges": (0.6,),
        }

        blocker = _local_scan_path_blocker(
            observation,
            current_pose=Pose2D(0.0, 0.0, 0.0),
            target_pose=Pose2D(2.0, 0.0, 0.0),
            robot_length_m=0.3913,
            robot_width_m=0.459,
            safety_padding_m=0.06,
        )

        self.assertIsNone(blocker)

    def test_nav2_router_path_guard_stops_on_known_static_obstacles(self) -> None:
        session = ManiSkillNav2RouterExplorationSession.__new__(ManiSkillNav2RouterExplorationSession)
        session.config = SimExplorationConfig(
            repo_root="/tmp/XLeRobot",
            persist_path="",
            occupancy_resolution=0.25,
        )
        session.known_cells = {GridCell(2, 0): "occupied"}
        session.manual_occupancy_edits = ManualOccupancyEdits()
        session.guardrail_events = []
        session._current_pose = lambda: Pose2D(0.0, 0.0, 0.0)
        observation = {
            "pose": Pose2D(0.0, 0.0, 0.0),
            "range_min": 0.05,
            "range_max": 10.0,
            "angle_min": 0.0,
            "angle_increment": 0.0,
            "ranges": (0.4,),
        }

        blocker = session._local_path_blocker_from_observation(
            observation,
            target_pose=Pose2D(2.0, 0.0, 0.0),
        )

        self.assertIsNotNone(blocker)
        assert blocker is not None
        self.assertEqual(blocker["beam_index"], 0)
        self.assertFalse(
            any(event["type"] == "local_path_guard_ignored_known_static_obstacle" for event in session.guardrail_events)
        )

    def test_nav2_router_collision_stop_requests_human_assistance(self) -> None:
        session = ManiSkillNav2RouterExplorationSession.__new__(ManiSkillNav2RouterExplorationSession)
        session.guardrail_events = []
        session.paused = False
        session.status = "moving"
        session.last_error = None
        session.action = np.ones(4, dtype=np.float32)
        publish_calls = []
        blocker = {
            "reason": "test obstacle",
            "point": {"x": 0.5, "y": 0.0},
            "forward_distance_m": 0.5,
            "lateral_distance_m": 0.0,
        }

        def raise_blocked(_path_poses, *, final_pose):
            raise Nav2LocalPathBlocked(blocker, travelled_distance_m=0.2)

        session._execute_router_path_once = raise_blocked
        session._publish_router_state = lambda: publish_calls.append("published")

        with self.assertRaises(HumanAssistanceRequired):
            session._execute_router_path([Pose2D(1.0, 0.0, 0.0)], final_pose=Pose2D(1.0, 0.0, 0.0))

        self.assertTrue(session.paused)
        self.assertEqual(session.status, "human_assistance_required")
        self.assertIn("Human assistance required", session.last_error)
        self.assertTrue(np.all(session.action == 0.0))
        self.assertEqual(publish_calls, ["published"])
        self.assertTrue(any(event["type"] == "human_assistance_requested" for event in session.guardrail_events))

    def test_nav2_router_motion_loop_honors_pause_stop(self) -> None:
        session = ManiSkillNav2RouterExplorationSession.__new__(ManiSkillNav2RouterExplorationSession)
        session.paused = True
        session.last_error = "Autonomous navigation is paused."
        session.action = np.ones(4, dtype=np.float32)

        with self.assertRaises(HumanAssistanceRequired):
            session._drive_straight_to(Pose2D(1.0, 0.0, 0.0), publish_stride=1)

        self.assertTrue(np.all(session.action == 0.0))

    def test_nav2_router_rejects_path_crossing_known_occupied_cell(self) -> None:
        session = ManiSkillNav2RouterExplorationSession.__new__(ManiSkillNav2RouterExplorationSession)
        session.config = SimExplorationConfig(
            repo_root="/tmp/XLeRobot",
            persist_path="",
            occupancy_resolution=0.25,
        )
        session.known_cells = {GridCell(2, 0): "occupied"}
        session.manual_occupancy_edits = ManualOccupancyEdits()
        session._transient_navigation_obstacle_cells = set()

        with self.assertRaisesRegex(RuntimeError, "known occupied map cell"):
            session._validate_router_path(
                [Pose2D(0.0, 0.1, 0.0), Pose2D(1.0, 0.1, 0.0)],
                start_pose=Pose2D(0.0, 0.1, 0.0),
                target_pose=Pose2D(1.0, 0.1, 0.0),
            )

    def test_nav2_router_allows_path_through_unknown_cells(self) -> None:
        session = ManiSkillNav2RouterExplorationSession.__new__(ManiSkillNav2RouterExplorationSession)
        session.config = SimExplorationConfig(
            repo_root="/tmp/XLeRobot",
            persist_path="",
            occupancy_resolution=0.25,
        )
        session.known_cells = {}
        session.manual_occupancy_edits = ManualOccupancyEdits()
        session._transient_navigation_obstacle_cells = set()

        session._validate_router_path(
            [Pose2D(0.0, 0.1, 0.0), Pose2D(1.0, 0.1, 0.0)],
            start_pose=Pose2D(0.0, 0.1, 0.0),
            target_pose=Pose2D(1.0, 0.1, 0.0),
        )

    def test_nav2_router_tries_alternate_frontier_target_when_first_path_crosses_wall(self) -> None:
        session = ManiSkillNav2RouterExplorationSession.__new__(ManiSkillNav2RouterExplorationSession)
        session.config = SimExplorationConfig(
            repo_root="/tmp/XLeRobot",
            persist_path="",
            occupancy_resolution=0.25,
            nav2_planner_id="",
        )
        session.known_cells = {GridCell(2, 0): "occupied"}
        session.manual_occupancy_edits = ManualOccupancyEdits()
        session._transient_navigation_obstacle_cells = set()
        session.guardrail_events = []
        session._publish_router_state = lambda **_kwargs: None
        session._probe_teleport_pose = lambda _pose: (True, None)
        first = Pose2D(1.0, 0.1, 0.0)
        second = Pose2D(0.0, 1.0, 0.0)
        session._candidate_target_poses = lambda **_kwargs: [first, second]

        class FakeRouter:
            def __init__(self) -> None:
                self.goals: list[Pose2D] = []

            def compute_path(self, *, goal_pose: Pose2D, planner_id: str = ""):
                self.goals.append(goal_pose)
                if goal_pose == first:
                    return 4, [Pose2D(0.0, 0.1, 0.0), first], "succeeded"
                return 4, [Pose2D(0.0, 0.1, 0.0), Pose2D(0.0, 0.5, 0.0), second], "succeeded"

        router = FakeRouter()
        session.router = router

        target, path, reason = session._plan_router_path_to_frontier(
            record=FrontierRecord(
                frontier_id="frontier_retry",
                nav_pose=first,
                centroid_pose=Pose2D(1.0, 0.0, 0.0),
                status="active",
                discovered_step=1,
                last_seen_step=1,
                unknown_gain=1,
                sensor_range_edge=False,
                room_hint=None,
                evidence=[],
            ),
            start_pose=Pose2D(0.0, 0.1, 0.0),
            reachable_safe_cells={GridCell(0, 0)},
        )

        self.assertEqual(target, second)
        self.assertEqual(path[-1], second)
        self.assertIsNone(reason)
        self.assertEqual(router.goals, [first, second])
        self.assertTrue(
            any(event["type"] == "router_frontier_target_candidate_recovered" for event in session.guardrail_events)
        )

    def test_occupancy_fusion_preserves_observed_walls_against_later_free_updates(self) -> None:
        wall_cell = GridCell(2, 3)
        known_cells = {wall_cell: "occupied"}
        evidence = {wall_cell: 2.0}

        merge_occupancy_observation(known_cells, wall_cell, "free", evidence_scores=evidence)

        self.assertEqual(known_cells[wall_cell], "occupied")

    def test_occupancy_fusion_clears_wall_after_repeated_free_updates(self) -> None:
        wall_cell = GridCell(2, 3)
        known_cells = {wall_cell: "occupied"}
        evidence = {wall_cell: 2.0}

        merge_occupancy_observation(known_cells, wall_cell, "free", evidence_scores=evidence)
        merge_occupancy_observation(known_cells, wall_cell, "free", evidence_scores=evidence)

        self.assertEqual(known_cells[wall_cell], "free")

    def test_occupancy_fusion_promotes_free_cell_to_wall_when_obstacle_observed(self) -> None:
        wall_cell = GridCell(2, 3)
        known_cells = {wall_cell: "free"}
        evidence: dict[GridCell, float] = {}

        merge_occupancy_observation(known_cells, wall_cell, "occupied", evidence_scores=evidence)

        self.assertEqual(known_cells[wall_cell], "occupied")

    def test_active_rgbd_scan_requires_supported_obstacle_hit(self) -> None:
        noisy_cell = GridCell(2, 3)
        known_cells: dict[GridCell, str] = {}
        evidence: dict[GridCell, float] = {}

        merge_occupancy_observation(
            known_cells,
            noisy_cell,
            "occupied",
            evidence_scores=evidence,
            config=ACTIVE_RGBD_SCAN_FUSION_CONFIG,
        )
        self.assertNotIn(noisy_cell, known_cells)

        merge_occupancy_observation(
            known_cells,
            noisy_cell,
            "occupied",
            evidence_scores=evidence,
            config=ACTIVE_RGBD_SCAN_FUSION_CONFIG,
        )
        self.assertEqual(known_cells[noisy_cell], "occupied")

    def test_parser_accepts_llm_policy_controls(self) -> None:
        args = build_parser().parse_args(
            [
                "--session",
                "agentic_session",
                "--explorer-policy",
                "llm",
                "--llm-model",
                "gpt-4o-mini",
                "--sensor-range-m",
                "8.0",
                "--max-decisions",
                "12",
                "--nav2-mode",
                "ros",
                "--nav2-planner-id",
                "SmacPlanner2D",
                "--nav2-controller-id",
                "FollowPath",
                "--ros-navigation-map-source",
                "fused_scan",
                "--ros-adapter-url",
                "http://127.0.0.1:8891",
                "--ros-map-topic",
                "/map",
                "--serve-review-ui",
                "--visited-frontier-filter-radius-m",
                "0.7",
                "--no-semantic-waypoints-enabled",
                "--semantic-llm-provider",
                "ollama",
                "--semantic-llm-model",
                "gemma4:26b",
                "--sim-motion-speed",
                "faster",
                "--automatic-semantic-waypoints",
            ]
        )
        self.assertEqual(args.session, "agentic_session")
        self.assertEqual(args.explorer_policy, "llm")
        self.assertEqual(args.llm_model, "gpt-4o-mini")
        self.assertEqual(args.sensor_range_m, 8.0)
        self.assertEqual(args.max_decisions, 12)
        self.assertEqual(args.nav2_mode, "ros")
        self.assertEqual(args.nav2_planner_id, "SmacPlanner2D")
        self.assertEqual(args.nav2_controller_id, "FollowPath")
        self.assertEqual(args.ros_navigation_map_source, "fused_scan")
        self.assertEqual(args.ros_adapter_url, "http://127.0.0.1:8891")
        self.assertEqual(args.ros_map_topic, "/map")
        self.assertTrue(args.serve_review_ui)
        self.assertEqual(args.visited_frontier_filter_radius_m, 0.7)
        self.assertFalse(args.semantic_waypoints_enabled)
        self.assertEqual(args.semantic_llm_provider, "ollama")
        self.assertEqual(args.semantic_llm_model, "gemma4:26b")
        self.assertEqual(args.sim_motion_speed, "faster")
        self.assertTrue(args.automatic_semantic_waypoints)

    def test_occupancy_map_like_to_ros_map_applies_manual_edits(self) -> None:
        base_map = RosOccupancyMap(
            resolution=0.5,
            width=2,
            height=2,
            origin_x=0.0,
            origin_y=0.0,
            data=(0, -1, 100, 0),
        )
        edits = ManualOccupancyEdits(
            blocked_cells={GridCell(0, 0)},
            cleared_cells={GridCell(1, 0)},
        )
        editable = EditableOccupancyMap(base_map, edits)

        converted = _occupancy_map_like_to_ros_map(editable)

        self.assertEqual(converted.data, (100, 0, 100, 0))

    def test_turnaround_scan_selection_prefers_even_yaw_coverage(self) -> None:
        observations = [
            {"pose": Pose2D(0.0, 0.0, index * (2.0 * 3.141592653589793 / 24.0))}
            for index in range(24)
        ]

        selected = _select_turnaround_scan_observations(observations, sample_count=6)

        self.assertGreaterEqual(len(selected), 6)
        selected_yaws = [item["pose"].yaw for item in selected]
        self.assertAlmostEqual(selected_yaws[0], 0.0)
        self.assertGreater(selected_yaws[-1], 5.0)

    def test_common_scan_fusion_integrates_hit_and_range_edge(self) -> None:
        known_cells: dict[GridCell, str] = {}
        evidence: dict[GridCell, float] = {}
        range_edge_cells: set[GridCell] = set()
        visited_cells: set[GridCell] = set()

        summary = integrate_planar_scan(
            pose=Pose2D(0.5, 0.5, 0.0),
            ranges=(1.0, 2.0),
            angle_min=0.0,
            angle_increment=3.141592653589793 / 2.0,
            range_min_m=0.05,
            range_max_m=2.0,
            resolution_m=0.5,
            cell_from_world=lambda x, y: GridCell(int(x / 0.5), int(y / 0.5)),
            known_cells=known_cells,
            evidence_scores=evidence,
            range_edge_cells=range_edge_cells,
            visited_cells=visited_cells,
            beam_stride=1,
            config=DEFAULT_OCCUPANCY_FUSION_CONFIG,
        )

        self.assertEqual(summary.scan_beams, 2)
        self.assertEqual(summary.integrated_beams, 2)
        self.assertTrue(any(state == "occupied" for state in known_cells.values()))
        self.assertTrue(range_edge_cells)
        self.assertTrue(visited_cells)

    def test_common_scan_fusion_does_not_clear_infinite_no_return_beams(self) -> None:
        known_cells: dict[GridCell, str] = {}
        evidence: dict[GridCell, float] = {}
        range_edge_cells: set[GridCell] = set()

        summary = integrate_planar_scan(
            pose=Pose2D(0.5, 0.5, 0.0),
            ranges=(math.inf,),
            angle_min=0.0,
            angle_increment=0.0,
            range_min_m=0.05,
            range_max_m=2.0,
            resolution_m=0.5,
            cell_from_world=lambda x, y: GridCell(int(x / 0.5), int(y / 0.5)),
            known_cells=known_cells,
            evidence_scores=evidence,
            range_edge_cells=range_edge_cells,
            beam_stride=1,
            config=DEFAULT_OCCUPANCY_FUSION_CONFIG,
        )

        self.assertEqual(summary.scan_beams, 1)
        self.assertEqual(summary.integrated_beams, 1)
        self.assertEqual(known_cells, {GridCell(1, 1): "free"})
        self.assertFalse(range_edge_cells)

    def test_point_cloud_fusion_marks_floor_free_and_furniture_bulk_occupied(self) -> None:
        known_cells: dict[GridCell, str] = {}
        evidence: dict[GridCell, float] = {}
        points = np.asarray(
            [
                [0.5, -0.1, 0.0],
                [0.5, 0.1, 0.0],
                [1.0, 0.0, 0.7],
                [1.16, 0.02, 0.72],
                [1.03, 0.16, 0.68],
            ],
            dtype=np.float32,
        )

        summary = integrate_transformed_point_cloud_observation(
            sensor_origin_xyz=(0.0, 0.0, 0.35),
            points_xyz_map=points,
            map_resolution_m=0.25,
            cell_from_world=lambda x, y: GridCell(int(math.floor(x / 0.25)), int(math.floor(y / 0.25))),
            known_cells=known_cells,
            evidence_scores=evidence,
            config=PointCloudFusionConfig(
                min_points_per_occupied_cell=2,
                obstacle_inflation_radius_m=0.0,
                max_rays=64,
            ),
        )

        self.assertGreater(summary.free_cell_count, 0)
        self.assertGreater(summary.occupied_cell_count, 0)
        self.assertEqual(known_cells[GridCell(4, 0)], "occupied")
        self.assertTrue(any(state == "free" for state in known_cells.values()))

    def test_point_cloud_fusion_ignores_obstacles_above_robot_clearance(self) -> None:
        known_cells: dict[GridCell, str] = {}
        points = np.asarray([[1.0, 0.0, 1.7], [1.02, 0.0, 1.72]], dtype=np.float32)

        summary = integrate_transformed_point_cloud_observation(
            sensor_origin_xyz=(0.0, 0.0, 0.35),
            points_xyz_map=points,
            map_resolution_m=0.25,
            cell_from_world=lambda x, y: GridCell(int(math.floor(x / 0.25)), int(math.floor(y / 0.25))),
            known_cells=known_cells,
            evidence_scores={},
            config=PointCloudFusionConfig(robot_clearance_height_m=1.5, obstacle_inflation_radius_m=0.0),
        )

        self.assertEqual(summary.obstacle_point_count, 0)
        self.assertNotIn(GridCell(4, 0), known_cells)

    def test_point_cloud_fusion_only_marks_far_free_endpoints_as_range_edges(self) -> None:
        range_edge_cells: set[GridCell] = set()

        integrate_transformed_point_cloud_observation(
            sensor_origin_xyz=(0.0, 0.0, 0.35),
            points_xyz_map=np.asarray([[0.8, 0.0, 0.0], [3.95, 0.0, 0.0]], dtype=np.float32),
            map_resolution_m=0.25,
            cell_from_world=lambda x, y: GridCell(int(math.floor(x / 0.25)), int(math.floor(y / 0.25))),
            known_cells={},
            evidence_scores={},
            range_edge_cells=range_edge_cells,
            config=PointCloudFusionConfig(max_rays=64),
        )

        self.assertIn(GridCell(15, 0), range_edge_cells)
        self.assertNotIn(GridCell(3, 0), range_edge_cells)

    def test_point_cloud_fusion_does_not_clear_to_obstacle_only_points(self) -> None:
        known_cells: dict[GridCell, str] = {}
        evidence: dict[GridCell, float] = {}
        points = np.asarray(
            [
                [1.0, 0.03, 0.7],
                [1.12, 0.05, 0.72],
                [1.20, 0.08, 0.68],
            ],
            dtype=np.float32,
        )

        summary = integrate_transformed_point_cloud_observation(
            sensor_origin_xyz=(0.0, 0.0, 0.35),
            points_xyz_map=points,
            map_resolution_m=0.25,
            cell_from_world=lambda x, y: GridCell(int(math.floor(x / 0.25)), int(math.floor(y / 0.25))),
            known_cells=known_cells,
            evidence_scores=evidence,
            config=PointCloudFusionConfig(
                min_points_per_occupied_cell=2,
                obstacle_inflation_radius_m=0.0,
                max_rays=64,
            ),
        )

        self.assertEqual(summary.free_cell_count, 0)
        self.assertEqual(known_cells[GridCell(4, 0)], "occupied")
        self.assertNotIn(GridCell(1, 0), known_cells)
        self.assertNotIn(GridCell(2, 0), known_cells)
        self.assertNotIn(GridCell(3, 0), known_cells)

    def test_point_cloud_fusion_estimates_tilted_floor_before_height_filtering(self) -> None:
        known_cells: dict[GridCell, str] = {}
        evidence: dict[GridCell, float] = {}
        floor_points = [[x * 0.1 + 0.5, y * 0.1 - 0.3, 0.20 * (x * 0.1 + 0.5) - 0.12] for x in range(8) for y in range(6)]
        obstacle_points = [[1.0, 0.0, 0.20 * 1.0 - 0.12 + 0.75], [1.16, 0.02, 0.20 * 1.16 - 0.12 + 0.76]]
        points = np.asarray(floor_points + obstacle_points, dtype=np.float32)

        summary = integrate_transformed_point_cloud_observation(
            sensor_origin_xyz=(0.0, 0.0, 0.35),
            points_xyz_map=points,
            map_resolution_m=0.25,
            cell_from_world=lambda x, y: GridCell(int(math.floor(x / 0.25)), int(math.floor(y / 0.25))),
            known_cells=known_cells,
            evidence_scores=evidence,
            config=PointCloudFusionConfig(
                min_points_per_occupied_cell=2,
                obstacle_inflation_radius_m=0.0,
                floor_plane_min_points=12,
                max_rays=64,
            ),
        )

        self.assertGreater(summary.floor_point_count, 20)
        self.assertEqual(summary.obstacle_point_count, 2)
        self.assertEqual(known_cells[GridCell(4, 0)], "occupied")

    def test_ros_adapter_serialization_round_trips_pose_map_and_scan(self) -> None:
        pose = Pose2D(1.0, 2.0, 0.5)
        occupancy_map = RosOccupancyMap(
            resolution=0.25,
            width=2,
            height=2,
            origin_x=-1.0,
            origin_y=-2.0,
            data=(0, -1, 100, 0),
        )
        observation = {
            "frame_id": "laser",
            "reference_frame": "odom",
            "pose": pose,
            "range_min": 0.05,
            "range_max": 10.0,
            "angle_min": -0.5,
            "angle_increment": 0.1,
            "ranges": (1.0, 2.0, 3.0),
        }

        self.assertEqual(pose_from_payload(serialize_pose(pose)), pose)
        self.assertEqual(map_from_payload(serialize_map(occupancy_map)), occupancy_map)
        round_tripped = scan_observation_from_payload(serialize_scan_observation(observation))
        self.assertIsNotNone(round_tripped)
        assert round_tripped is not None
        self.assertEqual(round_tripped["pose"], pose)
        self.assertEqual(round_tripped["reference_frame"], "odom")
        self.assertEqual(round_tripped["ranges"], (1.0, 2.0, 3.0))

    def test_runner_builds_real_agentic_map_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = str(Path(tmpdir) / "agentic_map.json")
            backend = ExplorationBackend(
                ExplorationBackendConfig(
                    mode="sim",
                    persist_path=persist_path,
                    occupancy_resolution=0.25,
                )
            )
            runner = ManiSkillExplorationRunner(
                SimExplorationConfig(
                    repo_root="/tmp/XLeRobot",
                    persist_path=persist_path,
                    area="apartment",
                    session="agentic_session",
                    source="test",
                    realtime_sleep_s=0.0,
                    llm_provider="mock",
                    llm_model="mock",
                    explorer_policy="llm",
                ),
                backend,
            )
            snapshot = runner.run()

            current_map = snapshot["current_map"]
            self.assertIsNotNone(current_map)
            assert current_map is not None
            self.assertEqual(current_map["mode"], "sim_agentic")
            self.assertEqual(current_map["map_id"], "agentic_session")
            self.assertGreater(current_map["coverage"], 0.75)
            self.assertGreaterEqual(len(current_map["regions"]), 4)
            self.assertTrue(current_map["trajectory"])
            self.assertTrue(current_map["keyframes"])
            self.assertTrue(current_map["occupancy"]["cells"])
            self.assertTrue(any(cell["state"] == "occupied" for cell in current_map["occupancy"]["cells"]))
            labels = {region["label"] for region in current_map["regions"]}
            self.assertIn("hallway", labels)
            self.assertIn("kitchen", labels)
            self.assertIn("living_room", labels)
            named_places = {item["name"] for item in current_map["named_places"]}
            self.assertIn("kitchen_center", named_places)
            self.assertIn("living_room_door", named_places)
            self.assertIn("hallway_entry", named_places)
            artifacts = current_map["artifacts"]
            self.assertTrue(artifacts["decision_log"])
            self.assertIn("memory_updates", artifacts["decision_log"][0]["decision"])
            self.assertIn("applied_memory_updates", artifacts["decision_log"][0]["trace"])
            self.assertIn("frontier_memory", artifacts)
            self.assertIn("nav2", artifacts)
            self.assertEqual(artifacts["nav2"]["module"], "simulated_nav2")
            self.assertTrue(artifacts["nav2"]["goals"])
            self.assertIn("semantic_memory", current_map)
            self.assertFalse(current_map["automatic_semantic_waypoints"])
            self.assertEqual(current_map["semantic_memory"], {})
            self.assertTrue(Path(persist_path).exists())

            task = snapshot["active_task"]
            self.assertIsNotNone(task)
            assert task is not None
            self.assertEqual(task["state"], "succeeded")
            self.assertGreaterEqual(task["result"]["decision_count"], 1)

    def test_interactive_manual_regions_create_navigation_memory(self) -> None:
        session = InteractiveNoNav2ExplorationSession(
            SimExplorationConfig(
                repo_root="/tmp/XLeRobot",
                persist_path="",
                explorer_policy="heuristic",
            )
        )

        before = session.snapshot()["map"]
        self.assertFalse(before["automatic_semantic_waypoints"])
        self.assertEqual(before["semantic_memory"], {})

        free_cell = session.current_cell
        snapshot = session.create_manual_region(
            {
                "label": "kitchen",
                "description": "operator marked the kitchen standing area",
                "cells": [{"cell_x": free_cell.x, "cell_y": free_cell.y}],
            }
        )
        region = snapshot["map"]["regions"][0]
        self.assertEqual(region["label"], "kitchen")
        self.assertEqual(region["description"], "operator marked the kitchen standing area")
        self.assertEqual(region["source"], "manual_region")
        self.assertEqual(snapshot["map"]["named_places"][0]["source"], "manual_region")
        self.assertEqual(snapshot["map"]["named_places"][0]["name"], "kitchen_center")

        pose = free_cell.center_pose(session.config.occupancy_resolution).to_dict()
        snapshot = session.add_manual_region_subwaypoint(
            {
                "region_id": region["region_id"],
                "name": "counter_side",
                "pose": pose,
            }
        )
        waypoints = snapshot["map"]["regions"][0]["default_waypoints"]
        self.assertEqual([waypoint["name"] for waypoint in waypoints], ["kitchen_center", "counter_side"])

        disabled = session.call_semantic_llm()
        self.assertEqual(disabled["map"]["semantic_memory"], {})
        self.assertIn("Automatic semantic waypoints are disabled", disabled["last_error"])


    def test_manishkill_frontier_approach_pose_is_offset_to_known_side(self) -> None:
        session = ManiSkillTeleportExplorationSession.__new__(ManiSkillTeleportExplorationSession)
        session.config = SimExplorationConfig(
            repo_root="/tmp/XLeRobot",
            persist_path="",
            occupancy_resolution=0.25,
            robot_radius_m=0.22,
        )
        session.known_cells = {}
        session.guardrail_events = []
        for x in range(0, 8):
            for y in range(2, 8):
                session.known_cells[GridCell(x, y)] = "free"
        session.known_cells[GridCell(6, 5)] = "occupied"

        approach = session._select_frontier_approach_cell(
            cluster=[GridCell(5, 5)],
            unknown_cells={GridCell(6, 5)},
            robot_cell=GridCell(1, 5),
            reachable_safe_cells=session._reachable_safe_navigation_cells(GridCell(1, 5)),
        )

        self.assertIsNotNone(approach)
        assert approach is not None
        self.assertLess(approach.x, 5)
        self.assertTrue(session._is_valid_robot_center_cell(approach))

    def test_strict_robot_center_validation_rejects_unknown_footprint_space(self) -> None:
        session = ManiSkillTeleportExplorationSession.__new__(ManiSkillTeleportExplorationSession)
        session.config = SimExplorationConfig(
            repo_root="/tmp/XLeRobot",
            persist_path="",
            occupancy_resolution=0.25,
            robot_radius_m=0.22,
        )
        session.known_cells = {}
        session.guardrail_events = []
        for x in range(0, 3):
            for y in range(0, 3):
                session.known_cells[GridCell(x, y)] = "free"

        self.assertFalse(
            session._is_valid_robot_center_cell(
                GridCell(0, 0),
                required_known_fraction=0.95,
                unknown_is_blocking=True,
                extra_clearance_m=0.08,
            )
        )

    def test_mobile_base_qpos_helpers_update_only_base_dofs(self) -> None:
        qpos = [0.0, 0.0, 0.0, 1.2, -0.5]
        qvel = [0.3, -0.1, 0.2, 0.9, -0.4]
        updated_qpos = _updated_mobile_base_qpos(
            qpos,
            Pose2D(1.5, -2.0, 0.7),
            anchor_pose=Pose2D(-1.0, 0.0, 0.0),
            anchor_qpos=[0.0, 0.0, 0.0, 1.2, -0.5],
        )
        updated_qvel = _zero_mobile_base_qvel(qvel)

        self.assertAlmostEqual(float(updated_qpos[0]), 2.5)
        self.assertAlmostEqual(float(updated_qpos[1]), -2.0)
        self.assertAlmostEqual(float(updated_qpos[2]), 0.7)
        self.assertEqual(list(updated_qpos[3:]), [1.2, -0.5])
        self.assertEqual(list(updated_qvel[:3]), [0.0, 0.0, 0.0])
        self.assertEqual(list(updated_qvel[3:]), [0.9, -0.4])

    def test_manishkill_candidate_paths_prefer_current_reachable_frontiers(self) -> None:
        session = ManiSkillTeleportExplorationSession.__new__(ManiSkillTeleportExplorationSession)
        session.config = SimExplorationConfig(
            repo_root="/tmp/XLeRobot",
            persist_path="",
            occupancy_resolution=0.25,
        )
        session.options = type("Options", (), {"max_frontiers": 12})()
        session.frontier_memory = FrontierMemory(session.config.occupancy_resolution)
        session.known_cells = {
            GridCell(x, 0): "free"
            for x in range(0, 5)
        }
        session.range_edge_cells = set()
        session.guardrail_events = []
        session._current_pose = lambda: Pose2D(0.125, 0.125, 0.0)
        session._reachable_safe_navigation_cells = lambda _cell: {GridCell(x, 0) for x in range(0, 5)}

        visible = FrontierRecord(
            frontier_id="frontier_visible",
            nav_pose=GridCell(2, 0).center_pose(session.config.occupancy_resolution),
            centroid_pose=GridCell(2, 0).center_pose(session.config.occupancy_resolution),
            status="stored",
            discovered_step=1,
            last_seen_step=2,
            unknown_gain=5,
            sensor_range_edge=False,
            room_hint=None,
            currently_visible=True,
        )
        remembered = FrontierRecord(
            frontier_id="frontier_remembered",
            nav_pose=GridCell(4, 0).center_pose(session.config.occupancy_resolution),
            centroid_pose=GridCell(4, 0).center_pose(session.config.occupancy_resolution),
            status="stored",
            discovered_step=1,
            last_seen_step=1,
            unknown_gain=9,
            sensor_range_edge=False,
            room_hint=None,
            currently_visible=False,
        )
        inaccessible = FrontierRecord(
            frontier_id="frontier_inaccessible",
            nav_pose=GridCell(8, 0).center_pose(session.config.occupancy_resolution),
            centroid_pose=GridCell(8, 0).center_pose(session.config.occupancy_resolution),
            status="stored",
            discovered_step=1,
            last_seen_step=2,
            unknown_gain=99,
            sensor_range_edge=False,
            room_hint=None,
            currently_visible=True,
        )
        session.frontier_memory.records = {
            visible.frontier_id: visible,
            remembered.frontier_id: remembered,
            inaccessible.frontier_id: inaccessible,
        }

        records = session._refresh_candidate_paths()

        self.assertEqual([record.frontier_id for record in records], ["frontier_visible", "frontier_remembered"])
        self.assertIsNone(inaccessible.path_cost_m)

    def test_manishkill_reachability_does_not_jump_to_disconnected_safe_island(self) -> None:
        session = ManiSkillTeleportExplorationSession.__new__(ManiSkillTeleportExplorationSession)
        session.config = SimExplorationConfig(
            repo_root="/tmp/XLeRobot",
            persist_path="",
            occupancy_resolution=0.25,
        )
        session.known_cells = {
            GridCell(0, 0): "free",
            GridCell(10, 0): "free",
            GridCell(11, 0): "free",
        }
        safe_island = {GridCell(10, 0), GridCell(11, 0)}
        session._is_valid_robot_center_cell = lambda cell, **_kwargs: cell in safe_island

        reachable = session._reachable_safe_navigation_cells(GridCell(0, 0))

        self.assertEqual(reachable, set())

    def test_manishkill_candidate_paths_reject_disconnected_known_free_target(self) -> None:
        session = ManiSkillTeleportExplorationSession.__new__(ManiSkillTeleportExplorationSession)
        session.config = SimExplorationConfig(
            repo_root="/tmp/XLeRobot",
            persist_path="",
            occupancy_resolution=0.25,
        )
        session.options = type("Options", (), {"max_frontiers": 12})()
        session.frontier_memory = FrontierMemory(session.config.occupancy_resolution)
        session.known_cells = {
            GridCell(0, 0): "free",
            GridCell(2, 0): "free",
        }
        session.range_edge_cells = set()
        session.guardrail_events = []
        session._current_pose = lambda: Pose2D(0.125, 0.125, 0.0)
        session._reachable_safe_navigation_cells = lambda _cell: {GridCell(0, 0), GridCell(2, 0)}
        disconnected = FrontierRecord(
            frontier_id="frontier_disconnected",
            nav_pose=GridCell(2, 0).center_pose(session.config.occupancy_resolution),
            centroid_pose=GridCell(2, 0).center_pose(session.config.occupancy_resolution),
            status="stored",
            discovered_step=1,
            last_seen_step=2,
            unknown_gain=5,
            sensor_range_edge=False,
            room_hint=None,
            currently_visible=True,
        )
        session.frontier_memory.records = {disconnected.frontier_id: disconnected}

        records = session._refresh_candidate_paths()

        self.assertEqual(records, [])
        self.assertIsNone(disconnected.path_cost_m)

    def test_manishkill_candidate_paths_drop_stored_frontier_when_boundary_is_unreachable(self) -> None:
        session = ManiSkillTeleportExplorationSession.__new__(ManiSkillTeleportExplorationSession)
        session.config = SimExplorationConfig(
            repo_root="/tmp/XLeRobot",
            persist_path="",
            occupancy_resolution=0.25,
        )
        session.options = type("Options", (), {"max_frontiers": 12})()
        session.frontier_memory = FrontierMemory(session.config.occupancy_resolution)
        session.known_cells = {
            GridCell(0, 0): "free",
            GridCell(1, 0): "free",
            GridCell(6, 0): "free",
        }
        session.range_edge_cells = set()
        session.guardrail_events = []
        session._current_pose = lambda: Pose2D(0.125, 0.125, 0.0)
        session._reachable_safe_navigation_cells = lambda _cell: {GridCell(0, 0), GridCell(1, 0)}
        remembered = FrontierRecord(
            frontier_id="frontier_unreachable_boundary",
            nav_pose=GridCell(1, 0).center_pose(session.config.occupancy_resolution),
            centroid_pose=GridCell(6, 0).center_pose(session.config.occupancy_resolution),
            status="stored",
            discovered_step=1,
            last_seen_step=1,
            unknown_gain=8,
            sensor_range_edge=False,
            room_hint=None,
            currently_visible=False,
        )
        session.frontier_memory.records = {remembered.frontier_id: remembered}

        records = session._refresh_candidate_paths()

        self.assertEqual(records, [])
        self.assertIsNone(remembered.path_cost_m)
        self.assertTrue(
            any(
                event["type"] == "stored_frontier_boundary_unreachable_through_known_free"
                for event in session.guardrail_events
            )
        )

    def test_manishkill_detected_frontier_requires_boundary_reachable_through_known_free(self) -> None:
        session = ManiSkillTeleportExplorationSession.__new__(ManiSkillTeleportExplorationSession)
        session.config = SimExplorationConfig(
            repo_root="/tmp/XLeRobot",
            persist_path="",
            occupancy_resolution=0.25,
        )
        session.options = type("Options", (), {"max_frontiers": 12})()
        session.guardrail_events = []
        session.latest_scan_known_cells = {
            GridCell(0, 0): "free",
            GridCell(6, 0): "free",
            GridCell(7, 0): "free",
            GridCell(8, 0): "free",
        }
        session.latest_scan_range_edge_cells = set()
        session.known_cells = dict(session.latest_scan_known_cells)
        session.range_edge_cells = set()
        session._current_pose = lambda: Pose2D(0.125, 0.125, 0.0)
        session._reachable_safe_navigation_cells = lambda _cell: {GridCell(0, 0)}

        candidates = session._detect_frontier_candidates()

        self.assertEqual(candidates, [])
        self.assertTrue(
            any(event["type"] == "frontier_boundary_unreachable_through_known_free" for event in session.guardrail_events)
        )

    def test_manishkill_candidate_paths_filter_frontier_at_current_pose(self) -> None:
        session = ManiSkillTeleportExplorationSession.__new__(ManiSkillTeleportExplorationSession)
        session.config = SimExplorationConfig(
            repo_root="/tmp/XLeRobot",
            persist_path="",
            occupancy_resolution=0.25,
        )
        session.options = type("Options", (), {"max_frontiers": 12})()
        session.frontier_memory = FrontierMemory(session.config.occupancy_resolution)
        session.known_cells = {GridCell(x, 0): "free" for x in range(0, 4)}
        session.range_edge_cells = set()
        session.guardrail_events = []
        session._current_pose = lambda: Pose2D(0.125, 0.125, 0.0)
        session._reachable_safe_navigation_cells = lambda _cell: {GridCell(x, 0) for x in range(0, 4)}

        current = FrontierRecord(
            frontier_id="frontier_current",
            nav_pose=GridCell(0, 0).center_pose(session.config.occupancy_resolution),
            centroid_pose=GridCell(0, 0).center_pose(session.config.occupancy_resolution),
            status="stored",
            discovered_step=1,
            last_seen_step=2,
            unknown_gain=8,
            sensor_range_edge=False,
            room_hint=None,
            currently_visible=True,
        )
        nearby = FrontierRecord(
            frontier_id="frontier_nearby",
            nav_pose=GridCell(3, 0).center_pose(session.config.occupancy_resolution),
            centroid_pose=GridCell(3, 0).center_pose(session.config.occupancy_resolution),
            status="stored",
            discovered_step=1,
            last_seen_step=2,
            unknown_gain=4,
            sensor_range_edge=False,
            room_hint=None,
            currently_visible=True,
        )
        session.frontier_memory.records = {
            current.frontier_id: current,
            nearby.frontier_id: nearby,
        }

        records = session._refresh_candidate_paths()

        self.assertEqual([record.frontier_id for record in records], ["frontier_nearby"])
        self.assertIsNone(current.path_cost_m)
        self.assertTrue(any(event["type"] == "frontier_at_current_pose_filtered" for event in session.guardrail_events))

    def test_manishkill_candidate_paths_filter_frontier_near_previous_visit(self) -> None:
        session = ManiSkillTeleportExplorationSession.__new__(ManiSkillTeleportExplorationSession)
        session.config = SimExplorationConfig(
            repo_root="/tmp/XLeRobot",
            persist_path="",
            occupancy_resolution=0.25,
            visited_frontier_filter_radius_m=0.5,
        )
        session.options = type("Options", (), {"max_frontiers": 12})()
        session.frontier_memory = FrontierMemory(session.config.occupancy_resolution)
        session.known_cells = {GridCell(x, 0): "free" for x in range(0, 6)}
        session.range_edge_cells = set()
        session.guardrail_events = []
        session._current_pose = lambda: Pose2D(0.125, 0.125, 0.0)
        session._reachable_safe_navigation_cells = lambda _cell: {GridCell(x, 0) for x in range(0, 6)}
        session.trajectory = [
            GridCell(4, 0).center_pose(session.config.occupancy_resolution).to_dict(),
            GridCell(0, 0).center_pose(session.config.occupancy_resolution).to_dict(),
        ]

        revisited_area = FrontierRecord(
            frontier_id="frontier_revisited_area",
            nav_pose=GridCell(4, 0).center_pose(session.config.occupancy_resolution),
            centroid_pose=GridCell(4, 0).center_pose(session.config.occupancy_resolution),
            status="stored",
            discovered_step=1,
            last_seen_step=2,
            unknown_gain=8,
            sensor_range_edge=False,
            room_hint=None,
            currently_visible=True,
        )
        session.frontier_memory.records = {revisited_area.frontier_id: revisited_area}

        records = session._refresh_candidate_paths()

        self.assertEqual(records, [])
        self.assertEqual(revisited_area.status, "suppressed")
        self.assertIsNone(revisited_area.path_cost_m)
        self.assertTrue(any(event["type"] == "frontier_near_visited_pose_filtered" for event in session.guardrail_events))

    def test_manishkill_current_pose_filter_uses_boundary_not_approach_pose(self) -> None:
        session = ManiSkillTeleportExplorationSession.__new__(ManiSkillTeleportExplorationSession)
        session.config = SimExplorationConfig(
            repo_root="/tmp/XLeRobot",
            persist_path="",
            occupancy_resolution=0.25,
        )
        session.options = type("Options", (), {"max_frontiers": 12})()
        session.frontier_memory = FrontierMemory(session.config.occupancy_resolution)
        session.known_cells = {GridCell(x, 0): "free" for x in range(0, 5)}
        session.range_edge_cells = set()
        session.guardrail_events = []
        session._current_pose = lambda: Pose2D(0.125, 0.125, 0.0)
        session._reachable_safe_navigation_cells = lambda _cell: {GridCell(x, 0) for x in range(0, 5)}

        same_approach_far_boundary = FrontierRecord(
            frontier_id="frontier_far_boundary",
            nav_pose=GridCell(0, 0).center_pose(session.config.occupancy_resolution),
            centroid_pose=GridCell(4, 0).center_pose(session.config.occupancy_resolution),
            status="stored",
            discovered_step=1,
            last_seen_step=2,
            unknown_gain=6,
            sensor_range_edge=False,
            room_hint=None,
            currently_visible=True,
        )
        session.frontier_memory.records = {
            same_approach_far_boundary.frontier_id: same_approach_far_boundary,
        }

        records = session._refresh_candidate_paths()

        self.assertEqual([record.frontier_id for record in records], ["frontier_far_boundary"])
        self.assertEqual(same_approach_far_boundary.path_cost_m, 0.0)
        self.assertFalse(any(event["type"] == "frontier_at_current_pose_filtered" for event in session.guardrail_events))

    def test_frontier_record_exposes_free_space_path_distance_alias(self) -> None:
        record = FrontierRecord(
            frontier_id="frontier_distance",
            nav_pose=Pose2D(1.0, 0.0, 0.0),
            centroid_pose=Pose2D(1.5, 0.0, 0.0),
            status="stored",
            discovered_step=1,
            last_seen_step=1,
            unknown_gain=3,
            sensor_range_edge=False,
            room_hint=None,
            path_cost_m=1.2349,
        )

        payload = record.to_dict()

        self.assertEqual(payload["path_cost_m"], 1.235)
        self.assertEqual(payload["free_space_path_distance_m"], 1.235)

    def test_mock_policy_prioritizes_nearby_useful_frontier_over_far_high_gain(self) -> None:
        policy = ExplorationLLMPolicy(
            SimExplorationConfig(
                repo_root="/tmp/XLeRobot",
                persist_path="",
                explorer_policy="llm",
                llm_provider="mock",
                llm_model="mock",
            )
        )
        near = FrontierRecord(
            frontier_id="frontier_near",
            nav_pose=Pose2D(1.0, 0.0, 0.0),
            centroid_pose=Pose2D(1.5, 0.0, 0.0),
            status="stored",
            discovered_step=1,
            last_seen_step=1,
            unknown_gain=16,
            sensor_range_edge=False,
            room_hint=None,
            currently_visible=True,
            path_cost_m=2.75,
        )
        far = FrontierRecord(
            frontier_id="frontier_far",
            nav_pose=Pose2D(8.0, 0.0, 0.0),
            centroid_pose=Pose2D(8.5, 0.0, 0.0),
            status="stored",
            discovered_step=1,
            last_seen_step=1,
            unknown_gain=20,
            sensor_range_edge=False,
            room_hint=None,
            currently_visible=True,
            path_cost_m=8.75,
        )

        decision = policy._heuristic_decision(
            [far, near],
            return_waypoints=[],
            coverage=0.2,
            current_room_id=None,
        )

        self.assertEqual(decision.selected_frontier_id, "frontier_near")
        self.assertIn("nearest useful reachable frontier", decision.reasoning_summary)

    def test_manishkill_close_enough_frontier_completion_uses_boundary_pose(self) -> None:
        session = ManiSkillTeleportExplorationSession.__new__(ManiSkillTeleportExplorationSession)
        session.config = SimExplorationConfig(
            repo_root="/tmp/XLeRobot",
            persist_path="",
            occupancy_resolution=0.25,
            robot_radius_m=0.22,
        )
        near = FrontierRecord(
            frontier_id="frontier_near",
            nav_pose=Pose2D(0.0, 0.0, 0.0),
            centroid_pose=Pose2D(1.0, 0.0, 0.0),
            status="active",
            discovered_step=1,
            last_seen_step=1,
            unknown_gain=3,
            sensor_range_edge=False,
            room_hint=None,
        )
        far = FrontierRecord(
            frontier_id="frontier_far",
            nav_pose=Pose2D(0.0, 0.0, 0.0),
            centroid_pose=Pose2D(2.5, 0.0, 0.0),
            status="active",
            discovered_step=1,
            last_seen_step=1,
            unknown_gain=3,
            sensor_range_edge=False,
            room_hint=None,
        )

        self.assertTrue(session._is_close_enough_to_complete_frontier(Pose2D(0.0, 0.0, 0.0), near))
        self.assertFalse(session._is_close_enough_to_complete_frontier(Pose2D(0.0, 0.0, 0.0), far))

    def test_nav2_unreachable_frontier_is_marked_visited_to_avoid_retries(self) -> None:
        memory = FrontierMemory(0.25)
        record = FrontierRecord(
            frontier_id="frontier_blocked",
            nav_pose=Pose2D(1.0, 0.0, 0.0),
            centroid_pose=Pose2D(1.5, 0.0, 0.0),
            status="stored",
            discovered_step=1,
            last_seen_step=1,
            unknown_gain=3,
            sensor_range_edge=False,
            room_hint=None,
        )
        memory.records = {record.frontier_id: record}
        memory.active_frontier_id = record.frontier_id

        completed = _mark_frontier_unreachable_as_visited(memory, record.frontier_id, "Nav2 could not plan a path")

        self.assertIs(completed, record)
        self.assertEqual(record.status, "completed")
        self.assertEqual(record.visit_count, 1)
        self.assertIsNone(memory.active_frontier_id)
        self.assertTrue(any("Nav2 could not plan a path" in evidence for evidence in record.evidence))

    def test_frontier_memory_matches_candidates_by_boundary_not_approach_pose(self) -> None:
        memory = FrontierMemory(0.25)
        existing = FrontierRecord(
            frontier_id="frontier_existing",
            nav_pose=Pose2D(0.0, 0.0, 0.0),
            centroid_pose=Pose2D(5.0, 0.0, 0.0),
            status="stored",
            discovered_step=1,
            last_seen_step=1,
            unknown_gain=2,
            sensor_range_edge=False,
            room_hint=None,
        )
        memory.records = {existing.frontier_id: existing}

        candidate = FrontierCandidate(
            frontier_id=None,
            member_cells=(GridCell(0, 0),),
            nav_cell=GridCell(0, 0),
            centroid_cell=GridCell(32, 0),
            nav_pose=Pose2D(0.0, 0.0, 0.0),
            centroid_pose=Pose2D(8.0, 0.0, 0.0),
            unknown_gain=4,
            sensor_range_edge=False,
            room_hint=None,
            evidence=["same approach pose but different original frontier boundary"],
        )

        memory.upsert_candidates([candidate], step_index=2)

        self.assertEqual(len(memory.records), 2)
        self.assertEqual(memory.records["frontier_existing"].centroid_pose.x, 5.0)

    def test_manishkill_candidate_paths_do_not_expose_stored_memory_that_is_not_global_frontier(self) -> None:
        session = ManiSkillTeleportExplorationSession.__new__(ManiSkillTeleportExplorationSession)
        session.config = SimExplorationConfig(
            repo_root="/tmp/XLeRobot",
            persist_path="",
            occupancy_resolution=0.25,
        )
        session.options = type("Options", (), {"max_frontiers": 12})()
        session.frontier_memory = FrontierMemory(session.config.occupancy_resolution)
        session.known_cells = {
            GridCell(x, y): "free"
            for x in range(-6, 9)
            for y in range(-6, 7)
        }
        session.range_edge_cells = set()
        session.guardrail_events = []
        session._current_pose = lambda: Pose2D(0.125, 0.125, 0.0)
        session._reachable_safe_navigation_cells = lambda _cell: {GridCell(3, 0)}
        remembered = FrontierRecord(
            frontier_id="frontier_remembered",
            nav_pose=GridCell(3, 0).center_pose(session.config.occupancy_resolution),
            centroid_pose=GridCell(3, 0).center_pose(session.config.occupancy_resolution),
            status="stored",
            discovered_step=1,
            last_seen_step=1,
            unknown_gain=9,
            sensor_range_edge=False,
            room_hint=None,
            currently_visible=False,
        )
        session.frontier_memory.records = {remembered.frontier_id: remembered}

        records = session._refresh_candidate_paths()

        self.assertEqual(records, [])
        self.assertTrue(
            any(event["type"] == "stored_frontier_not_global_boundary_after_merge" for event in session.guardrail_events)
        )

    def test_manishkill_candidate_paths_resnap_stored_memory_frontier_to_reachable_pose(self) -> None:
        session = ManiSkillTeleportExplorationSession.__new__(ManiSkillTeleportExplorationSession)
        session.config = SimExplorationConfig(
            repo_root="/tmp/XLeRobot",
            persist_path="",
            occupancy_resolution=0.25,
        )
        session.options = type("Options", (), {"max_frontiers": 12})()
        session.frontier_memory = FrontierMemory(session.config.occupancy_resolution)
        session.known_cells = {
            GridCell(0, 0): "free",
            GridCell(1, 0): "free",
            GridCell(2, 0): "free",
            GridCell(3, 0): "free",
            GridCell(4, 0): "free",
            GridCell(5, 0): "free",
            GridCell(6, 0): "free",
        }
        session.range_edge_cells = set()
        session.guardrail_events = []
        session._current_pose = lambda: Pose2D(0.125, 0.125, 0.0)
        session._reachable_safe_navigation_cells = lambda _cell: {GridCell(0, 0), GridCell(1, 0), GridCell(2, 0)}
        remembered = FrontierRecord(
            frontier_id="frontier_remembered",
            nav_pose=GridCell(10, 0).center_pose(session.config.occupancy_resolution),
            centroid_pose=GridCell(5, 0).center_pose(session.config.occupancy_resolution),
            status="stored",
            discovered_step=1,
            last_seen_step=1,
            unknown_gain=5,
            sensor_range_edge=False,
            room_hint=None,
            currently_visible=False,
        )
        session.frontier_memory.records = {remembered.frontier_id: remembered}

        records = session._refresh_candidate_paths()

        self.assertEqual([record.frontier_id for record in records], ["frontier_remembered"])
        self.assertEqual(session._world_to_cell(remembered.nav_pose.x, remembered.nav_pose.y), GridCell(2, 0))
        self.assertEqual(session._world_to_cell(remembered.centroid_pose.x, remembered.centroid_pose.y), GridCell(5, 0))
        self.assertIsNotNone(remembered.path_cost_m)
        self.assertTrue(
            any(event["type"] == "stored_frontier_revisit_pose_resnapped" for event in session.guardrail_events)
        )

    def test_manishkill_candidate_paths_keep_relaxed_stored_memory_boundary(self) -> None:
        session = ManiSkillTeleportExplorationSession.__new__(ManiSkillTeleportExplorationSession)
        session.config = SimExplorationConfig(
            repo_root="/tmp/XLeRobot",
            persist_path="",
            occupancy_resolution=0.25,
        )
        session.options = type("Options", (), {"max_frontiers": 12})()
        session.frontier_memory = FrontierMemory(session.config.occupancy_resolution)
        session.known_cells = {
            GridCell(0, 0): "free",
            GridCell(1, 0): "free",
            GridCell(2, 0): "free",
            GridCell(3, 0): "free",
            GridCell(1, -1): "occupied",
            GridCell(1, 1): "occupied",
            GridCell(2, -1): "occupied",
            GridCell(2, 1): "occupied",
            GridCell(4, 0): "occupied",
            GridCell(3, -1): "occupied",
            GridCell(6, 0): "free",
            GridCell(5, 0): "occupied",
            GridCell(7, 0): "occupied",
            GridCell(6, -1): "occupied",
        }
        session.range_edge_cells = set()
        session.guardrail_events = []
        session._current_pose = lambda: Pose2D(0.125, 0.125, 0.0)
        session._reachable_safe_navigation_cells = lambda _cell: {
            GridCell(0, 0),
            GridCell(1, 0),
            GridCell(2, 0),
            GridCell(3, 0),
        }
        remembered = FrontierRecord(
            frontier_id="frontier_relaxed",
            nav_pose=GridCell(10, 0).center_pose(session.config.occupancy_resolution),
            centroid_pose=GridCell(5, 0).center_pose(session.config.occupancy_resolution),
            status="stored",
            discovered_step=1,
            last_seen_step=1,
            unknown_gain=4,
            sensor_range_edge=False,
            room_hint=None,
            currently_visible=False,
        )
        session.frontier_memory.records = {remembered.frontier_id: remembered}

        records = session._refresh_candidate_paths()

        self.assertEqual([record.frontier_id for record in records], ["frontier_relaxed"])
        self.assertTrue(any("relaxed revalidation" in evidence for evidence in remembered.evidence))

    def test_manishkill_candidate_paths_drop_stored_memory_without_reachable_revisit_pose(self) -> None:
        session = ManiSkillTeleportExplorationSession.__new__(ManiSkillTeleportExplorationSession)
        session.config = SimExplorationConfig(
            repo_root="/tmp/XLeRobot",
            persist_path="",
            occupancy_resolution=0.25,
        )
        session.options = type("Options", (), {"max_frontiers": 12})()
        session.frontier_memory = FrontierMemory(session.config.occupancy_resolution)
        session.known_cells = {
            GridCell(0, 0): "free",
            GridCell(1, 0): "free",
            GridCell(2, 0): "free",
            GridCell(3, 0): "free",
            GridCell(4, 0): "free",
            GridCell(5, 0): "free",
            GridCell(6, 0): "free",
        }
        session.range_edge_cells = set()
        session.guardrail_events = []
        session._current_pose = lambda: Pose2D(0.125, 0.125, 0.0)
        session._reachable_safe_navigation_cells = lambda _cell: {GridCell(0, 0)}
        remembered = FrontierRecord(
            frontier_id="frontier_remembered",
            nav_pose=GridCell(10, 0).center_pose(session.config.occupancy_resolution),
            centroid_pose=GridCell(6, 0).center_pose(session.config.occupancy_resolution),
            status="stored",
            discovered_step=1,
            last_seen_step=1,
            unknown_gain=5,
            sensor_range_edge=False,
            room_hint=None,
            currently_visible=False,
        )
        session.frontier_memory.records = {remembered.frontier_id: remembered}

        records = session._refresh_candidate_paths()

        self.assertEqual(records, [])
        self.assertTrue(
            any(event["type"] == "stored_frontier_without_reachable_revisit_pose" for event in session.guardrail_events)
        )

    def test_manishkill_rejects_single_cell_frontier_as_too_narrow_for_robot(self) -> None:
        session = ManiSkillTeleportExplorationSession.__new__(ManiSkillTeleportExplorationSession)
        session.config = SimExplorationConfig(
            repo_root="/tmp/XLeRobot",
            persist_path="",
            occupancy_resolution=0.25,
        )
        session.options = type("Options", (), {"max_frontiers": 12})()
        session.guardrail_events = []
        session.latest_scan_known_cells = {
            GridCell(0, 0): "free",
            GridCell(1, 0): "occupied",
            GridCell(-1, 0): "occupied",
            GridCell(0, 1): "occupied",
        }
        session.latest_scan_range_edge_cells = set()
        session.known_cells = dict(session.latest_scan_known_cells)
        session.range_edge_cells = set()
        session._current_pose = lambda: Pose2D(0.125, 0.125, 0.0)
        session._reachable_safe_navigation_cells = lambda _cell: {GridCell(0, 0)}
        session._select_frontier_approach_cell = lambda **_kwargs: GridCell(0, 0)

        candidates = session._detect_frontier_candidates()

        self.assertEqual(candidates, [])
        self.assertTrue(any(event["type"] == "frontier_opening_too_narrow" for event in session.guardrail_events))

    def test_manishkill_detects_robot_sized_frontier_opening(self) -> None:
        session = ManiSkillTeleportExplorationSession.__new__(ManiSkillTeleportExplorationSession)
        session.config = SimExplorationConfig(
            repo_root="/tmp/XLeRobot",
            persist_path="",
            occupancy_resolution=0.25,
        )
        session.options = type("Options", (), {"max_frontiers": 12})()
        session.guardrail_events = []
        session.latest_scan_known_cells = {
            GridCell(0, 0): "free",
            GridCell(1, 0): "free",
            GridCell(2, 0): "free",
        }
        session.latest_scan_range_edge_cells = set()
        session.known_cells = dict(session.latest_scan_known_cells)
        session.range_edge_cells = set()
        session._current_pose = lambda: Pose2D(0.125, 0.125, 0.0)
        session._reachable_safe_navigation_cells = lambda _cell: {GridCell(0, 0), GridCell(1, 0)}
        session._select_frontier_approach_cell = lambda **_kwargs: GridCell(0, 0)

        candidates = session._detect_frontier_candidates()

        self.assertEqual(len(candidates), 1)
        self.assertTrue(any("frontier opening width" in evidence for evidence in candidates[0].evidence))

    def test_manishkill_finish_guardrail_falls_back_to_reachable_stored_frontier(self) -> None:
        session = ManiSkillTeleportExplorationSession.__new__(ManiSkillTeleportExplorationSession)
        session.config = SimExplorationConfig(
            repo_root="/tmp/XLeRobot",
            persist_path="",
            occupancy_resolution=0.25,
        )
        session.frontier_memory = FrontierMemory(session.config.occupancy_resolution)
        session.frontier_memory.return_waypoints = {}
        session.guardrail_events = []
        session._coverage = lambda: 0.4
        stored = FrontierRecord(
            frontier_id="frontier_remembered",
            nav_pose=GridCell(1, 0).center_pose(session.config.occupancy_resolution),
            centroid_pose=GridCell(1, 0).center_pose(session.config.occupancy_resolution),
            status="stored",
            discovered_step=1,
            last_seen_step=1,
            unknown_gain=9,
            sensor_range_edge=False,
            room_hint=None,
            currently_visible=False,
            path_cost_m=0.25,
        )

        class DummyPolicy:
            def _heuristic_decision(self, *_args: object) -> ExplorationDecision:
                return ExplorationDecision(
                    decision_type="revisit_frontier",
                    selected_frontier_id="frontier_remembered",
                    selected_return_waypoint_id=None,
                    frontier_ids_to_store=[],
                    exploration_complete=False,
                    reasoning_summary="Fallback to reachable stored frontier.",
                    semantic_updates=[],
                )

        session.policy = DummyPolicy()
        finish = ExplorationDecision(
            decision_type="finish",
            selected_frontier_id=None,
            selected_return_waypoint_id=None,
            frontier_ids_to_store=[],
            exploration_complete=True,
            reasoning_summary="No reachable frontier remains.",
            semantic_updates=[],
        )

        guarded = session._apply_finish_guardrail(finish, [stored])

        self.assertEqual(guarded.decision_type, "revisit_frontier")
        self.assertEqual(guarded.selected_frontier_id, "frontier_remembered")
        self.assertTrue(any(event["type"] == "finish_with_reachable_frontiers" for event in session.guardrail_events))

    def test_manishkill_rejects_latest_scan_boundary_that_is_not_global_frontier(self) -> None:
        session = ManiSkillTeleportExplorationSession.__new__(ManiSkillTeleportExplorationSession)
        session.config = SimExplorationConfig(
            repo_root="/tmp/XLeRobot",
            persist_path="",
            occupancy_resolution=0.25,
        )
        session.options = type("Options", (), {"max_frontiers": 12})()
        session.guardrail_events = []
        session.latest_scan_known_cells = {GridCell(0, 0): "free"}
        session.latest_scan_range_edge_cells = set()
        session.known_cells = {
            GridCell(x, y): "free"
            for x in range(-1, 2)
            for y in range(-1, 2)
        }
        session._current_pose = lambda: Pose2D(0.125, 0.125, 0.0)
        session._reachable_safe_navigation_cells = lambda _cell: {GridCell(0, 0)}

        candidates = session._detect_frontier_candidates()

        self.assertEqual(candidates, [])
        self.assertTrue(
            any(event["type"] == "frontier_not_global_boundary_after_merge" for event in session.guardrail_events)
        )

    def test_manishkill_keeps_latest_scan_boundary_that_remains_global_frontier(self) -> None:
        session = ManiSkillTeleportExplorationSession.__new__(ManiSkillTeleportExplorationSession)
        session.config = SimExplorationConfig(
            repo_root="/tmp/XLeRobot",
            persist_path="",
            occupancy_resolution=0.25,
        )
        session.options = type("Options", (), {"max_frontiers": 12})()
        session.guardrail_events = []
        session.latest_scan_known_cells = {
            GridCell(0, 0): "free",
            GridCell(1, 0): "free",
            GridCell(2, 0): "free",
            GridCell(3, 0): "free",
        }
        session.latest_scan_range_edge_cells = set()
        session.known_cells = dict(session.latest_scan_known_cells)
        session._current_pose = lambda: Pose2D(0.125, 0.125, 0.0)
        session._reachable_safe_navigation_cells = lambda _cell: {GridCell(0, 0)}
        session._select_frontier_approach_cell = lambda **_kwargs: GridCell(0, 0)

        candidates = session._detect_frontier_candidates()

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].unknown_gain, 10)
        self.assertTrue(
            any("passed merged-map validation" in evidence for evidence in candidates[0].evidence)
        )

    def test_manishkill_scan_merge_preserves_previous_wall_cells(self) -> None:
        session = ManiSkillTeleportExplorationSession.__new__(ManiSkillTeleportExplorationSession)
        session.known_cells = {
            GridCell(1, 1): "occupied",
            GridCell(2, 1): "free",
        }
        session.occupancy_evidence = {
            GridCell(1, 1): 2.0,
            GridCell(2, 1): -1.0,
        }
        session.latest_scan_known_cells = {
            GridCell(1, 1): "free",
            GridCell(3, 1): "free",
        }
        session.range_edge_cells = set()
        session.latest_scan_range_edge_cells = set()

        session._merge_latest_scan_into_global()

        self.assertEqual(session.known_cells[GridCell(1, 1)], "occupied")
        self.assertEqual(session.known_cells[GridCell(2, 1)], "free")
        self.assertEqual(session.known_cells[GridCell(3, 1)], "free")

    def test_navigation_map_data_url_renders_selectable_and_memory_frontiers(self) -> None:
        current = FrontierRecord(
            frontier_id="frontier_current",
            nav_pose=GridCell(2, 0).center_pose(0.25),
            centroid_pose=GridCell(2, 1).center_pose(0.25),
            status="stored",
            discovered_step=1,
            last_seen_step=2,
            unknown_gain=4,
            sensor_range_edge=False,
            room_hint=None,
            currently_visible=True,
            path_cost_m=0.5,
        )
        remembered = FrontierRecord(
            frontier_id="frontier_memory",
            nav_pose=GridCell(0, 2).center_pose(0.25),
            centroid_pose=GridCell(0, 3).center_pose(0.25),
            status="stored",
            discovered_step=1,
            last_seen_step=1,
            unknown_gain=4,
            sensor_range_edge=False,
            room_hint=None,
            currently_visible=False,
            path_cost_m=1.0,
        )

        data_url = _navigation_map_data_url(
            known_cells={GridCell(0, 0): "free", GridCell(1, 0): "occupied"},
            resolution=0.25,
            trajectory=[Pose2D(0.125, 0.125, 0.0).to_dict()],
            robot_pose=Pose2D(0.125, 0.125, 0.0),
            candidate_records=[current],
            remembered_records=[remembered],
        )

        self.assertTrue(data_url.startswith("data:image/png;base64,"))

    def test_interactive_parser_accepts_spawn_facing_and_front_only_scan(self) -> None:
        args = build_interactive_playground_parser().parse_args(
            [
                "--backend",
                "maniskill",
                "--spawn-facing",
                "left",
                "--scan-mode",
                "front_only",
                "--scan-yaw-samples",
                "1",
                "--ros-navigation-map-source",
                "fused_scan",
                "--ros-adapter-url",
                "http://127.0.0.1:8891",
                "--visited-frontier-filter-radius-m",
                "0.8",
                "--sim-motion-speed",
                "fastest",
                "--automatic-semantic-waypoints",
            ]
        )

        self.assertEqual(args.backend, "maniskill")
        self.assertEqual(args.spawn_facing, "left")
        self.assertEqual(args.scan_mode, "front_only")
        self.assertEqual(args.scan_yaw_samples, 1)
        self.assertEqual(args.ros_navigation_map_source, "fused_scan")
        self.assertEqual(args.ros_adapter_url, "http://127.0.0.1:8891")
        self.assertEqual(args.visited_frontier_filter_radius_m, 0.8)
        self.assertEqual(args.sim_motion_speed, "fastest")
        self.assertTrue(args.automatic_semantic_waypoints)

    def test_interactive_playground_ui_is_react_app(self) -> None:
        self.assertIn("ReactDOM.createRoot", INTERACTIVE_REACT_HTML)
        self.assertIn("React.createElement", INTERACTIVE_REACT_HTML)
        self.assertIn("window.INTERACTIVE_UI_FLAVOR", INTERACTIVE_REACT_HTML)
        self.assertIn("/api/region/create", INTERACTIVE_REACT_HTML)
        self.assertIn("requestJson(path, payload || {})", INTERACTIVE_REACT_HTML)

    def test_resolve_manishkill_start_pose_rotates_default_spawn_in_place(self) -> None:
        pose = _resolve_manishkill_start_pose(
            Pose2D(-1.0, 0.0, 0.25),
            spawn_x=None,
            spawn_y=None,
            spawn_yaw=0.0,
            spawn_facing="left",
        )

        self.assertIsNotNone(pose)
        assert pose is not None
        self.assertAlmostEqual(pose.x, -1.0)
        self.assertAlmostEqual(pose.y, 0.0)
        self.assertAlmostEqual(pose.yaw, 0.25 - 1.5707963267948966, places=6)

    def test_resolve_manishkill_start_pose_keeps_absolute_custom_spawn_pose(self) -> None:
        pose = _resolve_manishkill_start_pose(
            Pose2D(-1.0, 0.0, 0.25),
            spawn_x=1.5,
            spawn_y=-0.75,
            spawn_yaw=0.6,
            spawn_facing="back",
        )

        self.assertIsNotNone(pose)
        assert pose is not None
        self.assertAlmostEqual(pose.x, 1.5)
        self.assertAlmostEqual(pose.y, -0.75)
        self.assertAlmostEqual(pose.yaw, 0.6)

    def test_display_pose_applies_heading_offset_without_moving_position(self) -> None:
        session = ManiSkillTeleportExplorationSession.__new__(ManiSkillTeleportExplorationSession)
        session._display_yaw_offset_rad = 3.141592653589793

        display_pose = session._display_pose(Pose2D(1.25, -0.75, 0.4))

        self.assertAlmostEqual(display_pose.x, 1.25)
        self.assertAlmostEqual(display_pose.y, -0.75)
        self.assertAlmostEqual(display_pose.yaw, -2.741592653589793, places=6)


if __name__ == "__main__":
    unittest.main()
