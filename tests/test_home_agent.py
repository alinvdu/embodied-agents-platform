import json
import math
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from xlerobot_agent.home_agent import (
    HomeAgentConfig,
    HomeAgentController,
    HomeAgentModelConfig,
    HomeAgentToolRuntime,
    HomeTaskAgent,
    discover_latest_home_memory_path,
)
from xlerobot_agent.home_memory import (
    DEFAULT_NAVIGATION_CLEARANCE_M,
    DEFAULT_NAVIGATION_WAYPOINT_HORIZON_M,
    home_memory_agent_context,
    plan_region_exploration,
    resolve_home_memory_target,
    resolve_object_surface_approach_pose,
    resolve_region_navigation_goal,
)


TEST_PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMB/axJ4qkAAAAASUVORK5CYII="
)


def sample_memory() -> dict:
    return {
        "schema_version": "home_memory.v1",
        "memory_id": "house_v1",
        "frame": "map",
        "approved": True,
        "start_pose": {"name": "dock", "pose": {"x": 0.0, "y": 0.0, "yaw": 0.0}},
        "regions": [
            {
                "region_id": "region_kitchen",
                "label": "kitchen",
                "purpose": "food prep",
                "centroid": {"x": 3.0, "y": 2.0, "yaw": 0.0},
                "default_waypoints": [{"name": "kitchen_center", "x": 3.1, "y": 2.2, "yaw": 1.57}],
            }
        ],
        "places": [
            {"name": "fridge_front", "pose": {"x": 3.5, "y": 2.6, "yaw": 1.57}, "region_id": "region_kitchen"}
        ],
        "objects": [
            {
                "object_id": "fixture_fridge",
                "label": "fridge",
                "category": "fixture",
                "region_id": "region_kitchen",
                "pose": {"x": 3.8, "y": 2.8, "yaw": 1.57},
                "approach_pose": {"x": 3.5, "y": 2.6, "yaw": 1.57},
                "affordances": ["open_fridge", "inspect_fridge_contents"],
            }
        ],
        "skills": [
            {
                "skill_id": "open_fridge",
                "kind": "vla_or_scripted_skill",
                "target_labels": ["fridge"],
                "executor_binding": "vla_skill_runner",
                "safety": {"requires_human_approval": True},
            }
        ],
    }


def visual_sweep_memory() -> dict:
    memory = sample_memory()
    memory["regions"][0]["default_waypoints"] = []
    memory["regions"][0]["polygon_2d"] = [[0.75, 1.0], [5.25, 1.0], [5.25, 4.0], [0.75, 4.0]]
    memory["regions"][0]["exploration"] = {"max_stops": 3, "shots_per_stop": 2, "fov_deg": 65}
    memory["occupancy"] = {
        "resolution": 0.25,
        "bounds": {"min_x": 0.0, "min_y": 0.0},
        "cells": [
            {
                "x": x * 0.25,
                "y": y * 0.25,
                "state": "occupied" if y in {3, 17} and 2 <= x <= 22 else "free",
            }
            for x in range(25)
            for y in range(21)
        ],
    }
    return memory


def direct_fallback_memory() -> dict:
    memory = sample_memory()
    memory["occupancy"] = {
        "resolution": 0.25,
        "bounds": {"min_x": 0.0, "min_y": 0.0},
        "cells": [
            {
                "x": x * 0.25,
                "y": y * 0.25,
                "state": "occupied" if x in {0, 11} or y in {0, 11} else "free",
            }
            for x in range(12)
            for y in range(12)
        ],
    }
    memory["start_pose"] = {"name": "dock", "pose": {"x": 1.0, "y": 1.0, "yaw": 0.0}}
    return memory


def object_surface_memory() -> dict:
    memory = sample_memory()
    memory["occupancy"] = {
        "resolution": 0.25,
        "bounds": {"min_x": 0.0, "min_y": 0.0},
        "cells": [
            {
                "x": x * 0.25,
                "y": y * 0.25,
                "state": "occupied" if y == 12 and 7 <= x <= 20 else "free",
            }
            for x in range(25)
            for y in range(18)
        ],
    }
    return memory


class FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class HomeMemoryAgentContextTests(unittest.TestCase):
    def test_agent_context_exposes_regions_places_objects_and_skills(self) -> None:
        context = home_memory_agent_context(sample_memory())
        self.assertEqual(context["memory_id"], "house_v1")
        self.assertEqual(context["regions"][0]["label"], "kitchen")
        self.assertEqual(context["places"][0]["name"], "fridge_front")
        self.assertEqual(context["objects"][0]["label"], "fridge")
        self.assertEqual(context["skills"][0]["skill_id"], "open_fridge")

    def test_resolve_home_memory_target_returns_navigation_pose(self) -> None:
        target = resolve_home_memory_target(sample_memory(), "go to the kitchen")
        self.assertIsNotNone(target)
        assert target is not None
        self.assertEqual(target["label"], "kitchen")
        self.assertEqual(target["pose"]["x"], 3.1)

    def test_object_surface_approach_pose_faces_occupied_support_perpendicularly(self) -> None:
        result = resolve_object_surface_approach_pose(
            object_surface_memory(),
            {"x": 1.25, "y": 1.25, "yaw": 0.0},
            {"x": 2.75, "y": 2.85},
            min_clearance_m=0.30,
            standoff_m=0.65,
            max_alignment_distance_m=3.0,
        )

        self.assertEqual(result["status"], "succeeded")
        self.assertTrue(result["needs_alignment"])
        self.assertAlmostEqual(result["approach_pose"]["yaw"], 1.57, delta=0.25)
        self.assertLess(result["approach_pose"]["y"], result["support_surface"]["hit_point"]["y"])
        self.assertGreater(result["support_surface"]["occupied_sample_count"], 1)

    def test_region_navigation_goal_uses_occupancy_when_no_explicit_pose(self) -> None:
        memory = sample_memory()
        memory["regions"][0]["default_waypoints"] = []
        memory["regions"][0]["polygon_2d"] = [[2.0, 1.0], [4.0, 1.0], [4.0, 3.0], [2.0, 3.0]]
        memory["occupancy"] = {
            "resolution": 0.5,
            "bounds": {"min_x": 0.0, "min_y": 0.0},
            "cells": [
                {
                    "x": x * 0.5,
                    "y": y * 0.5,
                    "state": "occupied" if x in {0, 9} or y in {0, 5} else "free",
                }
                for x in range(10)
                for y in range(6)
            ],
        }
        result = resolve_region_navigation_goal(memory, "go to kitchen", current_pose={"x": 0.5, "y": 0.5, "yaw": 0.0})
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["target_label"], "kitchen")
        self.assertIn("goal_pose", result)
        self.assertGreater(result["candidate_count"], 0)
        self.assertGreaterEqual(result["clearance_m"], DEFAULT_NAVIGATION_CLEARANCE_M)
        self.assertEqual(result["path_strategy"], "footprint_eroded_centerline_weighted_grid")
        self.assertIn("next_waypoint", result)
        self.assertLessEqual(result["next_waypoint"]["distance_from_start_m"], DEFAULT_NAVIGATION_WAYPOINT_HORIZON_M)
        self.assertIn("waypoint_id", result["next_waypoint"])

    def test_region_navigation_goal_uses_inside_edge_approach_for_shallow_regions(self) -> None:
        memory = sample_memory()
        memory["regions"][0]["label"] = "TV Area"
        memory["regions"][0]["default_waypoints"] = []
        memory["regions"][0]["polygon_2d"] = [[2.0, 1.0], [4.0, 1.0], [4.0, 1.5], [2.0, 1.5]]
        memory["regions"][0]["centroid"] = {"x": 3.0, "y": 1.25, "yaw": 0.0}
        memory["occupancy"] = {
            "resolution": 0.25,
            "bounds": {"min_x": 0.0, "min_y": 0.0},
            "cells": [
                {
                    "x": x * 0.25,
                    "y": y * 0.25,
                    "state": "occupied" if x in {0, 25} or y in {0, 19} else "free",
                }
                for x in range(26)
                for y in range(20)
            ],
        }
        result = resolve_region_navigation_goal(
            memory,
            "go to TV Area",
            current_pose={"x": 3.0, "y": 3.0, "yaw": 0.0},
        )
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["goal_selection"], "inside_region_edge_approach")
        self.assertEqual(result["source"], "home_memory.occupancy_region_inside_edge_approach")
        self.assertGreater(result["goal_pose"]["y"], 1.2)
        self.assertLess(result["goal_pose"]["y"], 1.5)
        self.assertAlmostEqual(result["goal_pose"]["x"], 3.0, delta=0.35)
        self.assertGreater(result["approach_goal_candidate_count"], 0)

    def test_region_navigation_goal_ignores_generated_center_waypoint_after_region_edit(self) -> None:
        memory = sample_memory()
        memory["regions"][0]["label"] = "TV Area"
        memory["regions"][0]["centroid"] = {"x": 3.0, "y": 1.25, "yaw": 0.0}
        memory["regions"][0]["default_waypoints"] = [{"name": "tv_area_center", "x": 3.0, "y": 1.10, "yaw": 0.0}]
        memory["regions"][0]["polygon_2d"] = [[2.0, 1.0], [4.0, 1.0], [4.0, 1.5], [2.0, 1.5]]
        memory["occupancy"] = {
            "resolution": 0.25,
            "bounds": {"min_x": 0.0, "min_y": 0.0},
            "cells": [
                {
                    "x": x * 0.25,
                    "y": y * 0.25,
                    "state": "occupied" if x in {0, 25} or y in {0, 19} else "free",
                }
                for x in range(26)
                for y in range(20)
            ],
        }
        result = resolve_region_navigation_goal(
            memory,
            "go to TV Area",
            current_pose={"x": 3.0, "y": 3.0, "yaw": 0.0},
        )
        self.assertEqual(result["source"], "home_memory.occupancy_region_inside_edge_approach")
        self.assertGreater(result["goal_pose"]["y"], 1.2)
        self.assertLess(result["goal_pose"]["y"], 1.5)

    def test_region_navigation_goal_blocks_footprint_unsafe_corridor(self) -> None:
        memory = sample_memory()
        memory["regions"][0]["default_waypoints"] = []
        memory["regions"][0]["polygon_2d"] = [[0.5, 0.25], [3.5, 0.25], [3.5, 0.75], [0.5, 0.75]]
        memory["occupancy"] = {
            "resolution": 0.25,
            "bounds": {"min_x": 0.0, "min_y": 0.0},
            "cells": [
                {
                    "x": x * 0.25,
                    "y": y * 0.25,
                    "state": "free" if y == 2 and 1 <= x <= 14 else "occupied",
                }
                for x in range(16)
                for y in range(5)
            ],
        }
        result = resolve_region_navigation_goal(memory, "go to kitchen", current_pose={"x": 0.75, "y": 0.5, "yaw": 0.0})
        self.assertEqual(result["status"], "blocked")
        self.assertIn("footprint", result["reason"])

    def test_region_navigation_goal_blocks_unreachable_region_from_current_pose(self) -> None:
        memory = sample_memory()
        memory["regions"][0]["default_waypoints"] = []
        memory["regions"][0]["polygon_2d"] = [[3.0, 0.5], [4.5, 0.5], [4.5, 2.5], [3.0, 2.5]]
        memory["occupancy"] = {
            "resolution": 0.25,
            "bounds": {"min_x": 0.0, "min_y": 0.0},
            "cells": [
                {
                    "x": x * 0.25,
                    "y": y * 0.25,
                    "state": "occupied" if x == 9 or y in {0, 11} or x in {0, 19} else "free",
                }
                for x in range(20)
                for y in range(12)
            ],
        }
        result = resolve_region_navigation_goal(
            memory,
            "go to kitchen",
            current_pose={"x": 0.5, "y": 1.0, "yaw": 0.0},
            min_clearance_m=0.25,
        )
        self.assertEqual(result["status"], "blocked")
        self.assertIn("reachable", result["reason"])

    def test_region_navigation_goal_accepts_configurable_short_horizon_waypoint(self) -> None:
        memory = sample_memory()
        memory["regions"][0]["default_waypoints"] = []
        memory["regions"][0]["polygon_2d"] = [[2.0, 1.0], [4.0, 1.0], [4.0, 3.0], [2.0, 3.0]]
        memory["occupancy"] = {
            "resolution": 0.25,
            "bounds": {"min_x": 0.0, "min_y": 0.0},
            "cells": [
                {
                    "x": x * 0.25,
                    "y": y * 0.25,
                    "state": "occupied" if x in {0, 19} or y in {0, 15} else "free",
                }
                for x in range(20)
                for y in range(16)
            ],
        }
        result = resolve_region_navigation_goal(
            memory,
            "go to kitchen",
            current_pose={"x": 0.5, "y": 0.5, "yaw": 0.0},
            waypoint_horizon_m=1.0,
        )
        self.assertEqual(result["status"], "succeeded")
        waypoint = result["next_waypoint"]
        self.assertAlmostEqual(waypoint["distance_from_start_m"], 1.0, delta=0.05)
        self.assertFalse(waypoint["is_final_waypoint"])
        self.assertNotEqual((waypoint["x"], waypoint["y"]), (result["goal_pose"]["x"], result["goal_pose"]["y"]))

    def test_region_exploration_plan_generates_stops_and_fov_shots(self) -> None:
        result = plan_region_exploration(visual_sweep_memory(), "kitchen", min_clearance_m=0.2)
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["strategy"], "occupancy_boundary_visual_sweep")
        self.assertEqual(len(result["stops"]), 3)
        self.assertGreater(result["coverage"]["covered_boundary_cell_count"], 0)
        self.assertGreater(result["coverage"]["coverage_ratio"], 0.4)
        stop_xs = [stop["pose"]["x"] for stop in result["stops"]]
        self.assertGreater(max(stop_xs) - min(stop_xs), 2.0)
        for stop in result["stops"]:
            self.assertGreater(len(stop["shots"]), 0)
            self.assertLessEqual(len(stop["shots"]), 2)
            for shot in stop["shots"]:
                self.assertEqual(shot["fov_deg"], 65.0)
                self.assertIn("cone", shot)

    def test_region_exploration_plan_uses_region_effort_annotation(self) -> None:
        memory = visual_sweep_memory()
        memory["regions"][0]["exploration"] = {"max_stops": 1, "shots_per_stop": 1, "fov_deg": 70}
        result = plan_region_exploration(memory, "kitchen", min_clearance_m=0.2)
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["fov_deg"], 70.0)
        self.assertEqual(len(result["stops"]), 1)
        self.assertEqual(len(result["stops"][0]["shots"]), 1)

    def test_region_exploration_plan_prunes_duplicate_stops(self) -> None:
        memory = sample_memory()
        memory["regions"][0]["default_waypoints"] = []
        memory["regions"][0]["polygon_2d"] = [[0.0, 0.0], [1.2, 0.0], [1.2, 1.2], [0.0, 1.2]]
        memory["regions"][0]["exploration"] = {"max_stops": 3, "shots_per_stop": 2}
        free_cells = {(2, 2), (2, 3), (3, 2), (3, 3)}
        memory["occupancy"] = {
            "resolution": 0.1,
            "bounds": {"min_x": 0.0, "min_y": 0.0},
            "cells": [
                {
                    "x": x * 0.1,
                    "y": y * 0.1,
                    "state": "free" if (x, y) in free_cells else "occupied",
                }
                for x in range(12)
                for y in range(12)
            ],
        }
        result = plan_region_exploration(memory, "kitchen", min_clearance_m=0.05)
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(len(result["stops"]), 1)


class HomeTaskAgentTests(unittest.TestCase):
    def test_mock_agent_resolves_navigation_to_known_region(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = Path(tmpdir) / "house.home_memory.json"
            memory_path.write_text(json.dumps(sample_memory()))
            events = []
            agent = HomeTaskAgent(
                HomeAgentConfig(
                    home_memory_path=str(memory_path),
                    model=HomeAgentModelConfig(provider="mock", model="mock"),
                ),
                emit=lambda kind, title, summary, details=None: events.append(
                    {"kind": kind, "title": title, "summary": summary, "details": details or {}}
                ),
            )
            record = agent.run("go to the kitchen")
        self.assertEqual(record.status, "completed")
        self.assertTrue(any(action.get("tool") == "resolve_region_navigation_goal" for action in record.actions))
        self.assertFalse(any(action.get("tool") == "preview_path_to_pose" for action in record.actions))
        self.assertTrue(any(event["kind"] == "memory_resolved" for event in events))

    def test_mock_agent_resolves_region_navigation_from_occupancy(self) -> None:
        memory = sample_memory()
        memory["regions"][0]["default_waypoints"] = []
        memory["regions"][0]["polygon_2d"] = [[2.0, 1.0], [4.0, 1.0], [4.0, 3.0], [2.0, 3.0]]
        memory["occupancy"] = {
            "resolution": 0.5,
            "bounds": {"min_x": 0.0, "min_y": 0.0},
            "cells": [
                {
                    "x": x * 0.5,
                    "y": y * 0.5,
                    "state": "occupied" if x in {0, 9} or y in {0, 5} else "free",
                }
                for x in range(10)
                for y in range(6)
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = Path(tmpdir) / "house.home_memory.json"
            memory_path.write_text(json.dumps(memory))
            record = HomeTaskAgent(
                HomeAgentConfig(
                    home_memory_path=str(memory_path),
                    model=HomeAgentModelConfig(provider="mock", model="mock"),
                )
            ).run("go to the kitchen")
        self.assertEqual(record.status, "completed")
        self.assertTrue(any(action.get("tool") == "resolve_region_navigation_goal" for action in record.actions))
        self.assertFalse(any(action.get("tool") == "preview_path_to_pose" for action in record.actions))

    def test_mock_agent_plans_region_exploration_for_search_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = Path(tmpdir) / "house.home_memory.json"
            memory_path.write_text(json.dumps(visual_sweep_memory()))
            record = HomeTaskAgent(
                HomeAgentConfig(
                    home_memory_path=str(memory_path),
                    model=HomeAgentModelConfig(provider="mock", model="mock"),
                )
            ).run("find a coke can in the kitchen")
        self.assertEqual(record.status, "completed")
        self.assertTrue(any(action.get("tool") == "plan_region_exploration" for action in record.actions))
        self.assertFalse(any(action.get("tool") == "resolve_region_navigation_goal" for action in record.actions))

    def test_agent_auto_discovers_latest_home_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            older = Path(tmpdir) / "old" / "home_memory" / "old.home_memory.json"
            newer = Path(tmpdir) / "new" / "home_memory" / "new.home_memory.json"
            older.parent.mkdir(parents=True)
            newer.parent.mkdir(parents=True)
            older.write_text(json.dumps({**sample_memory(), "memory_id": "old"}))
            newer.write_text(json.dumps(sample_memory()))
            os.utime(older, (1, 1))
            os.utime(newer, (2, 2))
            discovered = discover_latest_home_memory_path((tmpdir,))
            record = HomeTaskAgent(
                HomeAgentConfig(
                    home_memory_search_roots=(tmpdir,),
                    model=HomeAgentModelConfig(provider="mock", model="mock"),
                )
            ).run("go to the kitchen")
        self.assertEqual(discovered, newer)
        self.assertEqual(record.status, "completed")
        self.assertIn("house_v1", record.memory_summary)

    def test_controller_lists_and_selects_memory_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_dir = Path(tmpdir) / "memories" / "house_v1"
            memory_dir.mkdir(parents=True)
            memory_path = memory_dir / "home_memory.json"
            memory_path.write_text(json.dumps(sample_memory()))
            controller = HomeAgentController.from_config(
                HomeAgentConfig(
                    home_memory_search_roots=(tmpdir,),
                    model=HomeAgentModelConfig(provider="mock", model="mock"),
                )
            )

            listed = controller.list_environment_memories()
            selected = controller.select_environment_memory("house_v1")
            snapshot = controller.snapshot()

        self.assertEqual(listed[0]["memory_id"], "house_v1")
        assert selected is not None
        self.assertEqual(selected["home_memory_path"], str(memory_path))
        self.assertEqual(snapshot["home_memory"]["context"]["memory_id"], "house_v1")

    def test_mock_agent_does_not_expose_skill_tools_yet(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = Path(tmpdir) / "house.home_memory.json"
            memory_path.write_text(json.dumps(sample_memory()))
            record = HomeTaskAgent(
                HomeAgentConfig(
                    home_memory_path=str(memory_path),
                    model=HomeAgentModelConfig(provider="mock", model="mock"),
                )
            ).run("open the fridge")
        self.assertEqual(record.status, "blocked")
        self.assertFalse(any(action.get("tool") == "run_skill" for action in record.actions))

    def test_controller_snapshot_matches_react_chat_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = Path(tmpdir) / "house.home_memory.json"
            memory_path.write_text(json.dumps(sample_memory()))
            controller = HomeAgentController.from_config(
                HomeAgentConfig(
                    home_memory_path=str(memory_path),
                    model=HomeAgentModelConfig(provider="mock", model="mock"),
                )
            )
            self.assertTrue(controller.start("go to the kitchen"))
            deadline = time.time() + 3
            snapshot = controller.snapshot()
            while snapshot["status"] == "running" and time.time() < deadline:
                time.sleep(0.02)
                snapshot = controller.snapshot()
        self.assertIn(snapshot["status"], {"completed", "blocked"})
        self.assertIn("report", snapshot)
        self.assertIn("events", snapshot["report"])
        self.assertEqual(snapshot["home_memory"]["context"]["memory_id"], "house_v1")

    def test_optional_specialist_hook_can_be_configured(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = Path(tmpdir) / "house.home_memory.json"
            memory_path.write_text(json.dumps(sample_memory()))
            events = []
            agent = HomeTaskAgent(
                HomeAgentConfig(
                    home_memory_path=str(memory_path),
                    model=HomeAgentModelConfig(provider="mock", model="mock"),
                    specialist_model=HomeAgentModelConfig(provider="mock", model="gemini-robotics-er-placeholder"),
                ),
                emit=lambda kind, title, summary, details=None: events.append(
                    {"kind": kind, "title": title, "summary": summary, "details": details or {}}
                ),
            )
            memory = agent._load_memory()
            runtime = HomeAgentToolRuntime(
                memory=memory,
                config=agent.config,
                emit=lambda kind, title, summary, details=None: events.append(
                    {"kind": kind, "title": title, "summary": summary, "details": details or {}}
                ),
            )
            result = runtime.analyze_embodied_scene(target_label="fridge", question="is this reachable?")
        self.assertEqual(result["status"], "succeeded")
        self.assertTrue(any(event["kind"] == "specialist_result" for event in events))

    def test_runtime_navigate_to_waypoint_calls_exploration_backend(self) -> None:
        events = []
        runtime = HomeAgentToolRuntime(
            memory=sample_memory(),
            config=HomeAgentConfig(exploration_backend_url="http://explore.local"),
            emit=lambda kind, title, summary, details=None: events.append(
                {"kind": kind, "title": title, "summary": summary, "details": details or {}}
            ),
        )
        payload = {
            "status": "succeeded",
            "reason": "Nav2 reached the requested goal pose",
            "nav2_result": {
                "status": "succeeded",
                "reason": "Nav2 reached the requested goal pose",
                "reached_pose": {"x": 1.2, "y": 0.4, "yaw": 0.1},
                "travelled_distance_m": 1.1,
                "actual_pose_delta_m": 1.1,
                "actual_yaw_delta_deg": 5.0,
                "plan": {"status": "succeeded", "path_length_m": 1.1},
                "feedback_samples": [{"remaining_distance_m": 0.0}],
            },
            "map": {"robot_pose": {"x": 1.2, "y": 0.4, "yaw": 0.1}},
        }
        with patch("xlerobot_agent.home_agent.urllib.request.urlopen", return_value=FakeHTTPResponse(payload)) as mocked:
            result = runtime.navigate_to_waypoint(waypoint_id="kitchen_step", x=1.2, y=0.4, yaw=0.1)

        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["current_pose"], {"x": 1.2, "y": 0.4, "yaw": 0.1})
        self.assertEqual(runtime.current_pose, {"x": 1.2, "y": 0.4, "yaw": 0.1})
        self.assertEqual(result["distance_remaining_m"], 0.0)
        self.assertEqual(result["actual_pose_delta_m"], 1.1)
        self.assertEqual(result["nav2"]["feedback_summary"]["sample_count"], 1)
        request = mocked.call_args.args[0]
        self.assertEqual(request.full_url, "http://explore.local/api/nav/waypoint")
        self.assertTrue(any(event["details"].get("tool") == "navigate_to_waypoint" for event in events))

    def test_runtime_navigate_to_waypoint_uses_direct_fallback_after_nav2_failure(self) -> None:
        events = []
        runtime = HomeAgentToolRuntime(
            memory=direct_fallback_memory(),
            config=HomeAgentConfig(exploration_backend_url="http://explore.local"),
            emit=lambda kind, title, summary, details=None: events.append(
                {"kind": kind, "title": title, "summary": summary, "details": details or {}}
            ),
        )
        nav2_payload = {
            "status": "failed",
            "reason": "Nav2 failed to make progress",
            "nav2_result": {
                "status": "failed",
                "reason": "Failed to make progress",
                "actual_pose_delta_m": 0.0,
                "plan": {"status": "succeeded", "path_length_m": 0.5},
                "feedback_samples": [{"remaining_distance_m": 0.5, "current_pose": {"x": 1.0, "y": 1.0, "yaw": 0.0}}],
            },
            "map": {"robot_pose": {"x": 1.0, "y": 1.0, "yaw": 0.0}},
        }
        fallback_payload = {
            "status": "succeeded",
            "reason": "micro adjustment reached pose",
            "local_motion": {
                "primitive": "micro_adjust_to_pose",
                "status": "succeeded",
                "reason": "target_pose_reached",
                "start_pose": {"x": 1.0, "y": 1.0, "yaw": 0.0},
                "end_pose": {"x": 1.5, "y": 1.0, "yaw": 0.0},
                "actual_pose_delta_m": 0.5,
                "distance_remaining_m": 0.0,
            },
            "map": {"robot_pose": {"x": 1.5, "y": 1.0, "yaw": 0.0}},
        }

        def fake_urlopen(request, timeout=0):
            if request.full_url.endswith("/api/nav/waypoint"):
                return FakeHTTPResponse(nav2_payload)
            if request.full_url.endswith("/api/nav/local_motion"):
                return FakeHTTPResponse(fallback_payload)
            raise AssertionError(f"unexpected URL {request.full_url}")

        with patch("xlerobot_agent.home_agent.urllib.request.urlopen", side_effect=fake_urlopen) as mocked:
            result = runtime.navigate_to_waypoint(waypoint_id="short_step", x=1.5, y=1.0, yaw=0.0)

        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["nav2_status_before_fallback"], "failed")
        self.assertEqual(result["direct_fallback_plan"]["status"], "succeeded")
        self.assertEqual(result["fallback_navigation"]["status"], "succeeded")
        self.assertEqual(runtime.current_pose, {"x": 1.5, "y": 1.0, "yaw": 0.0})
        self.assertEqual([call.args[0].full_url for call in mocked.call_args_list], [
            "http://explore.local/api/state",
            "http://explore.local/api/nav/waypoint",
            "http://explore.local/api/nav/local_motion",
        ])
        self.assertTrue(any(event["details"].get("tool") == "navigate_to_waypoint" for event in events))

    def test_runtime_navigate_to_waypoint_auto_rotates_before_nav2_when_threshold_exceeded(self) -> None:
        runtime = HomeAgentToolRuntime(
            memory=direct_fallback_memory(),
            config=HomeAgentConfig(
                exploration_backend_url="http://explore.local",
                navigation_auto_rotate_threshold_deg=45.0,
            ),
            emit=lambda *_args, **_kwargs: None,
        )
        rotate_payload = {
            "status": "succeeded",
            "reason": "rotated toward point",
            "local_motion": {
                "primitive": "rotate_towards_point",
                "status": "succeeded",
                "reason": "target_yaw_reached",
                "start_pose": {"x": 1.0, "y": 1.0, "yaw": 0.0},
                "end_pose": {"x": 1.0, "y": 1.0, "yaw": 1.57},
            },
            "map": {"robot_pose": {"x": 1.0, "y": 1.0, "yaw": 1.57}},
        }
        nav2_payload = {
            "status": "succeeded",
            "reason": "Nav2 reached the requested goal pose",
            "nav2_result": {
                "status": "succeeded",
                "reason": "Nav2 reached the requested goal pose",
                "reached_pose": {"x": 1.0, "y": 1.5, "yaw": 1.57},
                "plan": {"status": "succeeded", "path_length_m": 0.5},
                "feedback_samples": [{"remaining_distance_m": 0.0}],
            },
            "map": {"robot_pose": {"x": 1.0, "y": 1.5, "yaw": 1.57}},
        }

        def fake_urlopen(request, timeout=0):
            if request.full_url.endswith("/api/nav/local_motion"):
                return FakeHTTPResponse(rotate_payload)
            if request.full_url.endswith("/api/nav/waypoint"):
                return FakeHTTPResponse(nav2_payload)
            raise AssertionError(f"unexpected URL {request.full_url}")

        with patch("xlerobot_agent.home_agent.urllib.request.urlopen", side_effect=fake_urlopen) as mocked:
            result = runtime.navigate_to_waypoint(waypoint_id="side_step", x=1.0, y=1.5, yaw=1.57)

        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["pre_nav_auto_rotation"]["status"], "succeeded")
        self.assertEqual(result["pre_nav_auto_rotation"]["threshold_deg"], 45.0)
        self.assertGreater(result["pre_nav_auto_rotation"]["bearing_error_deg"], 45.0)
        self.assertEqual(runtime.current_pose, {"x": 1.0, "y": 1.5, "yaw": 1.57})
        self.assertEqual([call.args[0].full_url for call in mocked.call_args_list], [
            "http://explore.local/api/state",
            "http://explore.local/api/nav/local_motion",
            "http://explore.local/api/nav/waypoint",
        ])

    def test_runtime_navigate_to_waypoint_refreshes_pose_before_auto_rotate(self) -> None:
        runtime = HomeAgentToolRuntime(
            memory=direct_fallback_memory(),
            config=HomeAgentConfig(
                exploration_backend_url="http://explore.local",
                navigation_auto_rotate_threshold_deg=45.0,
            ),
            emit=lambda *_args, **_kwargs: None,
        )
        runtime.current_pose = {"x": 1.0, "y": 1.0, "yaw": 0.0}

        def fake_urlopen(request, timeout=0):
            if request.full_url.endswith("/api/state"):
                return FakeHTTPResponse(
                    {
                        "current_map": {
                            "robot_pose": {"x": 1.0, "y": 1.0, "yaw": 3.14},
                        },
                    }
                )
            if request.full_url.endswith("/api/nav/local_motion"):
                return FakeHTTPResponse(
                    {
                        "status": "succeeded",
                        "reason": "rotated toward point",
                        "local_motion": {
                            "primitive": "rotate_towards_point",
                            "status": "succeeded",
                            "start_pose": {"x": 1.0, "y": 1.0, "yaw": 3.14},
                            "end_pose": {"x": 1.0, "y": 1.0, "yaw": 0.0},
                        },
                        "map": {"robot_pose": {"x": 1.0, "y": 1.0, "yaw": 0.0}},
                    }
                )
            if request.full_url.endswith("/api/nav/waypoint"):
                return FakeHTTPResponse(
                    {
                        "status": "succeeded",
                        "reason": "Nav2 reached the requested goal pose",
                        "nav2_result": {
                            "status": "succeeded",
                            "reason": "Nav2 reached the requested goal pose",
                            "reached_pose": {"x": 2.0, "y": 1.0, "yaw": 0.0},
                            "plan": {"status": "succeeded", "path_length_m": 1.0},
                            "feedback_samples": [{"remaining_distance_m": 0.0}],
                        },
                        "map": {"robot_pose": {"x": 2.0, "y": 1.0, "yaw": 0.0}},
                    }
                )
            raise AssertionError(f"unexpected URL {request.full_url}")

        with patch("xlerobot_agent.home_agent.urllib.request.urlopen", side_effect=fake_urlopen):
            result = runtime.navigate_to_waypoint(waypoint_id="stale_pose_step", x=2.0, y=1.0, yaw=0.0)

        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["pre_nav_auto_rotation"]["status"], "succeeded")
        self.assertAlmostEqual(abs(result["pre_nav_auto_rotation"]["bearing_error_deg"]), 180.0, delta=0.2)

    def test_runtime_navigate_to_waypoint_suggests_local_clearance_recovery(self) -> None:
        runtime = HomeAgentToolRuntime(
            memory=direct_fallback_memory(),
            config=HomeAgentConfig(exploration_backend_url="http://explore.local"),
            emit=lambda *_args, **_kwargs: None,
        )
        nav2_payload = {
            "status": "failed",
            "reason": "Nav2 returned status `aborted`",
            "nav2_result": {
                "status": "failed",
                "reason": "Nav2 returned status `aborted`",
                "actual_pose_delta_m": 0.0,
                "plan": {"status": "succeeded", "path_length_m": 1.5},
                "feedback_samples": [{"remaining_distance_m": 1.5}],
            },
            "map": {"robot_pose": {"x": 1.0, "y": 1.0, "yaw": 0.0}},
        }

        def fake_urlopen(request, timeout=0):
            if request.full_url.endswith("/api/nav/waypoint"):
                return FakeHTTPResponse(nav2_payload)
            raise AssertionError(f"unexpected URL {request.full_url}")

        with patch("xlerobot_agent.home_agent.urllib.request.urlopen", side_effect=fake_urlopen):
            result = runtime.navigate_to_waypoint(waypoint_id="long_step", x=2.5, y=1.0, yaw=0.0)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["direct_fallback_plan"]["status"], "too_far")
        self.assertEqual(result["local_clearance_recovery"]["status"], "succeeded")
        self.assertEqual(result["local_clearance_recovery"]["suggested_tool"], "micro_adjust_to_pose")
        self.assertIn("local_clearance_recovery", result["failure_hint"])

    def test_runtime_execute_region_exploration_plan_navigates_stops_and_aligns_shots(self) -> None:
        events = []
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        runtime = HomeAgentToolRuntime(
            memory=visual_sweep_memory(),
            config=HomeAgentConfig(
                exploration_backend_url="http://explore.local",
                agent_artifacts_root=tmpdir.name,
            ),
            emit=lambda kind, title, summary, details=None: events.append(
                {"kind": kind, "title": title, "summary": summary, "details": details or {}}
            ),
            run_id="test_run",
        )
        current_pose = dict(runtime.current_pose)
        calls = []

        def fake_urlopen(request, timeout=0):
            body = json.loads(request.data.decode("utf-8")) if request.data else {}
            calls.append((request.full_url, body))
            if request.full_url.endswith("/api/nav/waypoint"):
                pose = body["pose"]
                current_pose.update({"x": pose["x"], "y": pose["y"], "yaw": 0.0})
                return FakeHTTPResponse(
                    {
                        "status": "succeeded",
                        "reason": "Nav2 reached the requested goal pose",
                        "nav2_result": {
                            "status": "succeeded",
                            "reason": "Nav2 reached the requested goal pose",
                            "reached_pose": pose,
                            "plan": {"status": "succeeded", "path_length_m": 0.5},
                            "feedback_samples": [{"remaining_distance_m": 0.0}],
                        },
                        "map": {"robot_pose": dict(current_pose)},
                    }
                )
            if request.full_url.endswith("/api/nav/local_motion"):
                primitive = body.get("primitive")
                if primitive == "rotate_by":
                    current_pose["yaw"] = round(
                        current_pose.get("yaw", 0.0) + float(body.get("delta_yaw_deg", 0.0)) * 3.141592653589793 / 180.0,
                        3,
                    )
                elif primitive == "rotate_towards_point":
                    current_pose["yaw"] = 0.0
                return FakeHTTPResponse(
                    {
                        "status": "succeeded",
                        "reason": "local motion succeeded",
                        "local_motion": {
                            "primitive": primitive,
                            "status": "succeeded",
                            "reason": "target_reached",
                            "start_pose": dict(runtime.current_pose),
                            "end_pose": dict(current_pose),
                            "actual_pose_delta_m": 0.0,
                            "actual_yaw_delta_deg": abs(float(body.get("delta_yaw_deg", 0.0) or 0.0)),
                        },
                        "map": {"robot_pose": dict(current_pose)},
                    }
                )
            if request.full_url.endswith("/api/nav/capture_rgb"):
                return FakeHTTPResponse(
                    {
                        "status": "succeeded",
                        "reason": "captured",
                        "image_data_url": TEST_PNG_DATA_URL,
                        "robot_pose": dict(current_pose),
                        "captured_at": 123.0,
                    }
                )
            raise AssertionError(f"unexpected URL {request.full_url}")

        with patch("xlerobot_agent.home_agent.urllib.request.urlopen", side_effect=fake_urlopen):
            result = runtime.execute_region_exploration_plan(
                region_label="kitchen",
                object_label="coke can",
                constraints={"max_stops": 1, "shots_per_stop": 1, "allow_auto_rotate": False},
            )

        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["visited_stop_count"], 1)
        self.assertEqual(result["captured_shot_count"], 1)
        self.assertEqual(result["saved_rgb_count"], 1)
        self.assertEqual(result["detection_status"], "not_configured")
        self.assertEqual(result["stops"][0]["navigation"]["status"], "succeeded")
        self.assertEqual(result["stops"][0]["shots"][0]["capture"]["status"], "succeeded")
        self.assertEqual(result["stops"][0]["shots"][0]["detection"]["object_label"], "coke can")
        self.assertTrue(any(url.endswith("/api/nav/waypoint") for url, _body in calls))
        self.assertTrue(any(url.endswith("/api/nav/capture_rgb") for url, _body in calls))
        image_path = Path(result["stops"][0]["shots"][0]["capture"]["image_path"])
        manifest_path = Path(result["stops"][0]["shots"][0]["capture"]["manifest_path"])
        self.assertTrue(image_path.exists())
        self.assertTrue(manifest_path.exists())
        self.assertTrue(any(event["details"].get("tool") == "plan_region_exploration" for event in events))
        self.assertTrue(any(event["details"].get("tool") == "execute_region_exploration_plan" for event in events))

    def test_runtime_execute_region_exploration_plan_aborts_when_detector_matches(self) -> None:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        runtime = HomeAgentToolRuntime(
            memory=visual_sweep_memory(),
            config=HomeAgentConfig(
                exploration_backend_url="http://explore.local",
                agent_artifacts_root=tmpdir.name,
                object_detector_provider="mock",
            ),
            emit=lambda *_args, **_kwargs: None,
            run_id="test_detection_run",
        )
        current_pose = dict(runtime.current_pose)
        calls = []

        def fake_urlopen(request, timeout=0):
            body = json.loads(request.data.decode("utf-8")) if request.data else {}
            calls.append((request.full_url, body))
            if request.full_url.endswith("/api/nav/waypoint"):
                pose = body["pose"]
                current_pose.update({"x": pose["x"], "y": pose["y"], "yaw": 0.0})
                return FakeHTTPResponse(
                    {
                        "status": "succeeded",
                        "reason": "Nav2 reached the requested goal pose",
                        "nav2_result": {
                            "status": "succeeded",
                            "reason": "Nav2 reached the requested goal pose",
                            "reached_pose": pose,
                            "feedback_samples": [{"remaining_distance_m": 0.0}],
                        },
                        "map": {"robot_pose": dict(current_pose)},
                    }
                )
            if request.full_url.endswith("/api/nav/local_motion"):
                current_pose["yaw"] = round(
                    current_pose.get("yaw", 0.0)
                    + float(body.get("delta_yaw_deg", 0.0)) * 3.141592653589793 / 180.0,
                    3,
                )
                return FakeHTTPResponse(
                    {
                        "status": "succeeded",
                        "reason": "local motion succeeded",
                        "local_motion": {
                            "primitive": body.get("primitive"),
                            "status": "succeeded",
                            "end_pose": dict(current_pose),
                        },
                        "map": {"robot_pose": dict(current_pose)},
                    }
                )
            if request.full_url.endswith("/api/nav/capture_rgb"):
                return FakeHTTPResponse(
                    {
                        "status": "succeeded",
                        "reason": "captured",
                        "image_data_url": "data:image/png;base64,cG5n",
                        "robot_pose": dict(current_pose),
                        "captured_at": 123.0,
                    }
                )
            raise AssertionError(f"unexpected URL {request.full_url}")

        with patch("xlerobot_agent.home_agent.urllib.request.urlopen", side_effect=fake_urlopen):
            result = runtime.execute_region_exploration_plan(
                region_label="kitchen",
                object_label="coke can",
                constraints={"max_stops": 1, "shots_per_stop": 2, "allow_auto_rotate": False},
            )

        self.assertEqual(result["status"], "object_found")
        self.assertEqual(result["detection_status"], "matched")
        self.assertEqual(result["captured_shot_count"], 1)
        self.assertEqual(result["selected_detection"]["label"], "coke can")
        self.assertEqual(result["stops"][0]["shots"][0]["detection"]["status"], "matched")
        self.assertEqual(
            len([url for url, _body in calls if url.endswith("/api/nav/capture_rgb")]),
            1,
        )
        manifest_path = Path(result["stops"][0]["shots"][0]["capture"]["manifest_path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["captures"][0]["detection"]["status"], "matched")
        capture = result["stops"][0]["shots"][0]["capture"]
        try:
            import PIL  # noqa: F401
        except Exception:
            self.assertNotIn("annotated_artifact_url", capture)
        else:
            self.assertIn("annotated_artifact_url", capture)
            self.assertTrue(Path(capture["annotated_image_path"]).is_file())
            self.assertEqual(manifest["captures"][0]["annotated_artifact_url"], capture["annotated_artifact_url"])

    def test_runtime_execute_region_exploration_plan_aborts_when_detector_unavailable(self) -> None:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        runtime = HomeAgentToolRuntime(
            memory=visual_sweep_memory(),
            config=HomeAgentConfig(
                exploration_backend_url="http://explore.local",
                agent_artifacts_root=tmpdir.name,
                object_detector_provider="replicate_grounding_dino",
                object_detector_api_key=None,
            ),
            emit=lambda *_args, **_kwargs: None,
            run_id="test_detection_unavailable_run",
        )
        current_pose = dict(runtime.current_pose)
        calls = []

        def fake_urlopen(request, timeout=0):
            body = json.loads(request.data.decode("utf-8")) if request.data else {}
            calls.append((request.full_url, body))
            if request.full_url.endswith("/api/nav/waypoint"):
                pose = body["pose"]
                current_pose.update({"x": pose["x"], "y": pose["y"], "yaw": 0.0})
                return FakeHTTPResponse(
                    {
                        "status": "succeeded",
                        "reason": "Nav2 reached the requested goal pose",
                        "nav2_result": {
                            "status": "succeeded",
                            "reason": "Nav2 reached the requested goal pose",
                            "reached_pose": pose,
                            "feedback_samples": [{"remaining_distance_m": 0.0}],
                        },
                        "map": {"robot_pose": dict(current_pose)},
                    }
                )
            if request.full_url.endswith("/api/nav/local_motion"):
                current_pose["yaw"] = round(
                    current_pose.get("yaw", 0.0)
                    + float(body.get("delta_yaw_deg", 0.0)) * 3.141592653589793 / 180.0,
                    3,
                )
                return FakeHTTPResponse(
                    {
                        "status": "succeeded",
                        "reason": "local motion succeeded",
                        "local_motion": {
                            "primitive": body.get("primitive"),
                            "status": "succeeded",
                            "end_pose": dict(current_pose),
                        },
                        "map": {"robot_pose": dict(current_pose)},
                    }
                )
            if request.full_url.endswith("/api/nav/capture_rgb"):
                return FakeHTTPResponse(
                    {
                        "status": "succeeded",
                        "reason": "captured",
                        "image_data_url": "data:image/png;base64,cG5n",
                        "robot_pose": dict(current_pose),
                        "captured_at": 123.0,
                    }
                )
            raise AssertionError(f"unexpected URL {request.full_url}")

        with patch("xlerobot_agent.home_agent.urllib.request.urlopen", side_effect=fake_urlopen):
            result = runtime.execute_region_exploration_plan(
                region_label="kitchen",
                object_label="coke can",
                constraints={"max_stops": 1, "shots_per_stop": 2, "allow_auto_rotate": False},
            )

        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["detection_status"], "unavailable")
        self.assertIn("Missing Replicate API token", result["reason"])
        self.assertEqual(
            len([url for url, _body in calls if url.endswith("/api/nav/capture_rgb")]),
            1,
        )

    def test_runtime_focus_detected_object_centers_tracked_detection(self) -> None:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        events = []
        runtime = HomeAgentToolRuntime(
            memory=visual_sweep_memory(),
            config=HomeAgentConfig(
                exploration_backend_url="http://explore.local",
                agent_artifacts_root=tmpdir.name,
                object_detector_provider="mock",
            ),
            emit=lambda kind, title, summary, details=None: events.append(
                {"kind": kind, "title": title, "summary": summary, "details": details or {}}
            ),
            run_id="test_focus_run",
        )

        def fake_urlopen(request, timeout=0):
            if request.full_url.endswith("/api/nav/capture_rgb"):
                return FakeHTTPResponse(
                    {
                        "status": "succeeded",
                        "reason": "captured",
                        "image_data_url": TEST_PNG_DATA_URL,
                        "robot_pose": dict(runtime.current_pose),
                        "captured_at": 123.0,
                    }
                )
            raise AssertionError(f"unexpected URL {request.full_url}")

        with patch("xlerobot_agent.home_agent.urllib.request.urlopen", side_effect=fake_urlopen):
            result = runtime.focus_detected_object(object_label="coke can")

        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["object_label"], "coke can")
        self.assertIn(result["detection_id"], runtime.detection_tracking)
        self.assertAlmostEqual(result["center_error_norm"], 0.0)
        self.assertTrue(any(event["details"].get("tool") == "focus_detected_object" for event in events))

    def test_runtime_focus_detected_object_reuses_tracked_bbox_without_detector_refresh(self) -> None:
        runtime = HomeAgentToolRuntime(
            memory=visual_sweep_memory(),
            config=HomeAgentConfig(exploration_backend_url="http://explore.local"),
            emit=lambda *_args, **_kwargs: None,
        )
        runtime._track_detection_result(
            object_label="coke can",
            detection={
                "status": "matched",
                "selected_detection_id": "det_1",
                "selected_detection": {
                    "detection_id": "det_1",
                    "label": "coke can",
                    "confidence": 0.9,
                    "bbox_xyxy": [0.35, 0.25, 0.65, 0.75],
                },
            },
            capture={"shot_id": "shot_1", "image_width": 640, "image_height": 480},
        )

        def fake_urlopen(request, timeout=0):
            if request.full_url.endswith("/api/nav/capture_rgb"):
                raise AssertionError("focus should reuse the tracked bbox")
            raise AssertionError(f"unexpected URL {request.full_url}")

        with patch("xlerobot_agent.home_agent.urllib.request.urlopen", side_effect=fake_urlopen):
            result = runtime.focus_detected_object(object_label="coke can")

        self.assertEqual(result["status"], "succeeded")
        self.assertFalse(result["detector_refreshed"])
        self.assertEqual(result["attempt_count"], 0)

    def test_runtime_approach_detected_object_uses_rgbd_depth_and_micro_steps(self) -> None:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        runtime = HomeAgentToolRuntime(
            memory=visual_sweep_memory(),
            config=HomeAgentConfig(
                exploration_backend_url="http://explore.local",
                agent_artifacts_root=tmpdir.name,
                object_detector_provider="mock",
                object_approach_step_m=0.08,
            ),
            emit=lambda *_args, **_kwargs: None,
            run_id="test_approach_run",
        )
        current_pose = dict(runtime.current_pose)
        geometry_calls = 0
        urls = []
        geometry_bodies = []

        def fake_urlopen(request, timeout=0):
            nonlocal geometry_calls
            body = json.loads(request.data.decode("utf-8")) if request.data else {}
            urls.append(request.full_url)
            if request.full_url.endswith("/api/nav/capture_rgb"):
                return FakeHTTPResponse(
                    {
                        "status": "succeeded",
                        "reason": "captured",
                        "image_data_url": TEST_PNG_DATA_URL,
                        "robot_pose": dict(current_pose),
                        "captured_at": 123.0 + geometry_calls,
                    }
                )
            if request.full_url.endswith("/api/nav/estimate_detection_geometry"):
                geometry_calls += 1
                geometry_bodies.append(body)
                forward = 0.7 if geometry_calls == 1 else 0.42
                return FakeHTTPResponse(
                    {
                        "status": "succeeded",
                        "reason": "geometry solved",
                        "forward_m": forward,
                        "distance_m": forward,
                        "lateral_m": 0.0,
                        "bearing_error_deg": 0.0,
                        "estimated_pose_base": {"x": forward, "y": 0.0, "z": 0.0},
                        "current_pose": dict(current_pose),
                        "safety": {
                            "safe": True,
                            "safe_forward_step_m": 0.08 if geometry_calls == 1 else 0.0,
                            "reason": "clear",
                        },
                    }
                )
            if request.full_url.endswith("/api/nav/local_motion"):
                pose = body["pose"]
                current_pose.update(pose)
                return FakeHTTPResponse(
                    {
                        "status": "succeeded",
                        "reason": "micro adjustment completed",
                        "local_motion": {
                            "primitive": "micro_adjust_to_pose",
                            "status": "succeeded",
                            "start_pose": dict(runtime.current_pose),
                            "end_pose": dict(current_pose),
                            "distance_remaining_m": 0.0,
                        },
                        "map": {"robot_pose": dict(current_pose)},
                    }
                )
            raise AssertionError(f"unexpected URL {request.full_url}")

        with patch("xlerobot_agent.home_agent.urllib.request.urlopen", side_effect=fake_urlopen):
            result = runtime.approach_detected_object(
                object_label="coke can",
                constraints={"allow_surface_alignment": False},
            )

        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["geometry"]["forward_m"], 0.42)
        self.assertEqual(geometry_calls, 2)
        self.assertEqual(result["attempts"][1]["source"], "detector_refresh")
        self.assertTrue(any(url.endswith("/api/nav/local_motion") for url in urls))
        self.assertIn(result["detection_id"], runtime.detection_tracking)
        self.assertTrue(all(body.get("require_depth_image") is True for body in geometry_bodies))
        self.assertTrue(all(body.get("disable_point_cloud_fallback") is True for body in geometry_bodies))

    def test_runtime_approach_moves_fraction_of_remaining_gap(self) -> None:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        runtime = HomeAgentToolRuntime(
            memory=visual_sweep_memory(),
            config=HomeAgentConfig(
                exploration_backend_url="http://explore.local",
                agent_artifacts_root=tmpdir.name,
                object_detector_provider="mock",
            ),
            emit=lambda *_args, **_kwargs: None,
            run_id="test_fractional_approach_run",
        )
        current_pose = dict(runtime.current_pose)
        start_pose = dict(current_pose)
        geometry_calls = 0
        geometry_bodies = []
        local_motion_bodies = []

        def fake_urlopen(request, timeout=0):
            nonlocal geometry_calls
            body = json.loads(request.data.decode("utf-8")) if request.data else {}
            if request.full_url.endswith("/api/nav/capture_rgb"):
                return FakeHTTPResponse(
                    {
                        "status": "succeeded",
                        "reason": "captured",
                        "image_data_url": TEST_PNG_DATA_URL,
                        "robot_pose": dict(current_pose),
                        "captured_at": 123.0 + geometry_calls,
                    }
                )
            if request.full_url.endswith("/api/nav/estimate_detection_geometry"):
                geometry_calls += 1
                geometry_bodies.append(body)
                forward = 0.70 if geometry_calls == 1 else 0.32
                return FakeHTTPResponse(
                    {
                        "status": "succeeded",
                        "reason": "geometry solved",
                        "forward_m": forward,
                        "distance_m": forward,
                        "lateral_m": 0.0,
                        "bearing_error_deg": 0.0,
                        "estimated_pose_base": {"x": forward, "y": 0.0, "z": 0.0},
                        "current_pose": dict(current_pose),
                        "safety": {
                            "safe": True,
                            "safe_forward_step_m": 0.25 if geometry_calls == 1 else 0.0,
                            "reason": "clear",
                        },
                    }
                )
            if request.full_url.endswith("/api/nav/local_motion"):
                local_motion_bodies.append(body)
                pose = body["pose"]
                current_pose.update(pose)
                return FakeHTTPResponse(
                    {
                        "status": "succeeded",
                        "reason": "fractional approach step completed",
                        "local_motion": {
                            "primitive": "micro_adjust_to_pose",
                            "status": "succeeded",
                            "start_pose": dict(runtime.current_pose),
                            "end_pose": dict(current_pose),
                            "distance_remaining_m": 0.0,
                        },
                        "map": {"robot_pose": dict(current_pose)},
                    }
                )
            raise AssertionError(f"unexpected URL {request.full_url}")

        with patch("xlerobot_agent.home_agent.urllib.request.urlopen", side_effect=fake_urlopen):
            result = runtime.approach_detected_object(
                object_label="coke can",
                constraints={
                    "allow_surface_alignment": False,
                    "target_min_m": 0.25,
                    "target_max_m": 0.30,
                },
            )

        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(geometry_calls, 2)
        self.assertEqual(geometry_bodies[0]["max_step_m"], 0.25)
        self.assertEqual(result["attempts"][0]["approach_step"]["desired_forward_step_m"], 0.32)
        self.assertEqual(result["attempts"][0]["approach_step"]["chosen_forward_step_m"], 0.25)
        self.assertTrue(local_motion_bodies)
        first_pose = local_motion_bodies[0]["pose"]
        moved = math.hypot(first_pose["x"] - start_pose["x"], first_pose["y"] - start_pose["y"])
        self.assertAlmostEqual(moved, 0.25, delta=0.01)

    def test_runtime_approach_detected_object_reuses_tracked_bbox_for_depth(self) -> None:
        runtime = HomeAgentToolRuntime(
            memory=visual_sweep_memory(),
            config=HomeAgentConfig(exploration_backend_url="http://explore.local"),
            emit=lambda *_args, **_kwargs: None,
        )
        runtime._track_detection_result(
            object_label="coke can",
            detection={
                "status": "matched",
                "selected_detection_id": "det_1",
                "selected_detection": {
                    "detection_id": "det_1",
                    "label": "coke can",
                    "confidence": 0.9,
                    "bbox_xyxy": [220, 120, 420, 360],
                },
            },
            capture={"shot_id": "shot_1", "image_width": 640, "image_height": 480},
        )
        calls = []

        def fake_urlopen(request, timeout=0):
            body = json.loads(request.data.decode("utf-8")) if request.data else {}
            calls.append((request.full_url, body))
            if request.full_url.endswith("/api/nav/capture_rgb"):
                raise AssertionError("approach should reuse the tracked bbox for the first RGB-D solve")
            if request.full_url.endswith("/api/nav/estimate_detection_geometry"):
                return FakeHTTPResponse(
                    {
                        "status": "succeeded",
                        "reason": "geometry solved",
                        "forward_m": 0.42,
                        "distance_m": 0.42,
                        "lateral_m": 0.0,
                        "bearing_error_deg": 0.0,
                        "current_pose": dict(runtime.current_pose),
                        "safety": {"safe": True, "safe_forward_step_m": 0.0, "reason": "clear"},
                    }
                )
            raise AssertionError(f"unexpected URL {request.full_url}")

        with patch("xlerobot_agent.home_agent.urllib.request.urlopen", side_effect=fake_urlopen):
            result = runtime.approach_detected_object(object_label="coke can")

        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["attempts"][0]["source"], "tracked_detection")
        self.assertEqual(
            len([url for url, _body in calls if url.endswith("/api/nav/estimate_detection_geometry")]),
            1,
        )
        geometry_body = next(body for url, body in calls if url.endswith("/api/nav/estimate_detection_geometry"))
        self.assertTrue(geometry_body["require_depth_image"])
        self.assertTrue(geometry_body["disable_point_cloud_fallback"])
        self.assertEqual(geometry_body["bbox_sample_inner_ratio"], 0.65)
        self.assertEqual(geometry_body["min_valid_points"], 12)

    def test_runtime_approach_accepts_grasp_staging_tolerance(self) -> None:
        runtime = HomeAgentToolRuntime(
            memory=visual_sweep_memory(),
            config=HomeAgentConfig(exploration_backend_url="http://explore.local"),
            emit=lambda *_args, **_kwargs: None,
        )
        runtime._track_detection_result(
            object_label="small bottle of water",
            detection={
                "status": "matched",
                "selected_detection_id": "det_1",
                "selected_detection": {
                    "detection_id": "det_1",
                    "label": "small bottle",
                    "confidence": 0.9,
                    "bbox_xyxy": [260, 267, 319, 406],
                },
            },
            capture={"shot_id": "shot_1", "image_width": 640, "image_height": 480},
        )

        def fake_urlopen(request, timeout=0):
            if request.full_url.endswith("/api/nav/estimate_detection_geometry"):
                return FakeHTTPResponse(
                    {
                        "status": "succeeded",
                        "reason": "geometry solved",
                        "forward_m": 0.454,
                        "distance_m": 0.811,
                        "lateral_m": 0.033,
                        "bearing_error_deg": 4.17,
                        "current_pose": dict(runtime.current_pose),
                        "safety": {
                            "safe": True,
                            "safe_forward_step_m": 0.004,
                            "reason": "clear",
                        },
                    }
                )
            if request.full_url.endswith("/api/nav/local_motion"):
                raise AssertionError("approach should not chase a 4 mm tolerance step")
            if request.full_url.endswith("/api/nav/capture_rgb"):
                raise AssertionError("approach should reuse the tracked detection")
            raise AssertionError(f"unexpected URL {request.full_url}")

        with patch("xlerobot_agent.home_agent.urllib.request.urlopen", side_effect=fake_urlopen):
            result = runtime.approach_detected_object(
                object_label="small bottle of water",
                constraints={"allow_surface_alignment": False},
            )

        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["geometry"]["forward_m"], 0.454)
        self.assertEqual(result["target_tolerance_m"], 0.025)
        self.assertIn("tolerance", result["reason"])

    def test_runtime_approach_refreshes_detector_after_image_centering_rotation(self) -> None:
        runtime = HomeAgentToolRuntime(
            memory=visual_sweep_memory(),
            config=HomeAgentConfig(
                exploration_backend_url="http://explore.local",
                object_detector_provider="mock",
            ),
            emit=lambda *_args, **_kwargs: None,
        )
        runtime._track_detection_result(
            object_label="coke can",
            detection={
                "status": "matched",
                "selected_detection_id": "det_1",
                "selected_detection": {
                    "detection_id": "det_1",
                    "label": "coke can",
                    "confidence": 0.9,
                    "bbox_xyxy": [120, 120, 220, 360],
                },
            },
            capture={"shot_id": "shot_1", "image_width": 640, "image_height": 480},
        )
        geometry_calls = 0
        capture_calls = 0
        rotation_requests = []

        def fake_urlopen(request, timeout=0):
            nonlocal geometry_calls, capture_calls
            body = json.loads(request.data.decode("utf-8")) if request.data else {}
            if request.full_url.endswith("/api/nav/estimate_detection_geometry"):
                geometry_calls += 1
                return FakeHTTPResponse(
                    {
                        "status": "succeeded",
                        "reason": "geometry solved",
                        "forward_m": 0.7 if geometry_calls == 1 else 0.42,
                        "distance_m": 0.7 if geometry_calls == 1 else 0.42,
                        "lateral_m": 0.0,
                        "bearing_error_deg": -90.0 if geometry_calls == 1 else 0.0,
                        "current_pose": dict(runtime.current_pose),
                        "safety": {"safe": True, "safe_forward_step_m": 0.0, "reason": "clear"},
                    }
                )
            if request.full_url.endswith("/api/nav/local_motion"):
                self.assertEqual(body.get("primitive"), "rotate_by")
                rotation_requests.append(body.get("delta_yaw_deg"))
                return FakeHTTPResponse(
                    {
                        "status": "succeeded",
                        "reason": "rotation completed",
                        "local_motion": {
                            "primitive": "rotate_by",
                            "status": "succeeded",
                            "start_pose": dict(runtime.current_pose),
                            "end_pose": dict(runtime.current_pose),
                        },
                        "map": {"robot_pose": dict(runtime.current_pose)},
                    }
                )
            if request.full_url.endswith("/api/nav/capture_rgb"):
                capture_calls += 1
                return FakeHTTPResponse(
                    {
                        "status": "succeeded",
                        "reason": "captured",
                        "image_data_url": TEST_PNG_DATA_URL,
                        "robot_pose": dict(runtime.current_pose),
                        "captured_at": 123.0,
                    }
                )
            raise AssertionError(f"unexpected URL {request.full_url}")

        with patch("xlerobot_agent.home_agent.urllib.request.urlopen", side_effect=fake_urlopen):
            result = runtime.approach_detected_object(
                object_label="coke can",
                constraints={"allow_surface_alignment": False, "max_attempts": 3},
            )

        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(geometry_calls, 2)
        self.assertEqual(capture_calls, 1)
        self.assertEqual(result["attempts"][1]["source"], "detector_refresh")
        self.assertEqual(result["attempts"][0]["next_action"], "refresh_detector_after_image_centering_rotation")
        self.assertEqual(result["attempts"][0]["image_centering"]["rotation_source"], "image_bbox")
        self.assertEqual(rotation_requests, [12.0])

    def test_runtime_approach_recenters_negative_forward_geometry_instead_of_too_close(self) -> None:
        runtime = HomeAgentToolRuntime(
            memory=visual_sweep_memory(),
            config=HomeAgentConfig(
                exploration_backend_url="http://explore.local",
                object_detector_provider="mock",
            ),
            emit=lambda *_args, **_kwargs: None,
        )
        runtime._track_detection_result(
            object_label="small yellow bottle",
            detection={
                "status": "matched",
                "selected_detection_id": "det_1",
                "selected_detection": {
                    "detection_id": "det_1",
                    "label": "small yellow bottle",
                    "confidence": 0.93,
                    "bbox_xyxy": [167, 180, 223, 276],
                },
            },
            capture={"shot_id": "shot_1", "image_width": 640, "image_height": 480},
        )
        geometry_calls = 0
        capture_calls = 0
        rotation_requests = []

        def fake_urlopen(request, timeout=0):
            nonlocal geometry_calls, capture_calls
            body = json.loads(request.data.decode("utf-8")) if request.data else {}
            if request.full_url.endswith("/api/nav/estimate_detection_geometry"):
                geometry_calls += 1
                forward = -0.16 if geometry_calls == 1 else 0.42
                lateral = -0.623 if geometry_calls == 1 else 0.0
                return FakeHTTPResponse(
                    {
                        "status": "succeeded",
                        "reason": "geometry solved",
                        "forward_m": forward,
                        "distance_m": 0.958 if geometry_calls == 1 else 0.42,
                        "lateral_m": lateral,
                        "bearing_error_deg": -90.0 if geometry_calls == 1 else 0.0,
                        "current_pose": dict(runtime.current_pose),
                        "safety": {"safe": True, "safe_forward_step_m": 0.0, "reason": "clear"},
                    }
                )
            if request.full_url.endswith("/api/nav/local_motion"):
                self.assertEqual(body.get("primitive"), "rotate_by")
                rotation_requests.append(body.get("delta_yaw_deg"))
                return FakeHTTPResponse(
                    {
                        "status": "succeeded",
                        "reason": "rotation completed",
                        "local_motion": {
                            "primitive": "rotate_by",
                            "status": "succeeded",
                            "start_pose": dict(runtime.current_pose),
                            "end_pose": dict(runtime.current_pose),
                        },
                        "map": {"robot_pose": dict(runtime.current_pose)},
                    }
                )
            if request.full_url.endswith("/api/nav/capture_rgb"):
                capture_calls += 1
                return FakeHTTPResponse(
                    {
                        "status": "succeeded",
                        "reason": "captured",
                        "image_data_url": TEST_PNG_DATA_URL,
                        "robot_pose": dict(runtime.current_pose),
                        "captured_at": 123.0,
                    }
                )
            raise AssertionError(f"unexpected URL {request.full_url}")

        with patch("xlerobot_agent.home_agent.urllib.request.urlopen", side_effect=fake_urlopen):
            result = runtime.approach_detected_object(
                object_label="small yellow bottle",
                constraints={"allow_surface_alignment": False, "max_attempts": 3},
            )

        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(geometry_calls, 2)
        self.assertEqual(capture_calls, 1)
        self.assertEqual(result["attempts"][0]["geometry_consistency"]["status"], "invalid_forward")
        self.assertEqual(result["attempts"][0]["next_action"], "refresh_detector_after_image_centering_rotation")
        self.assertEqual(result["attempts"][0]["image_centering"]["rotation_source"], "image_bbox")
        self.assertEqual(rotation_requests, [12.0])

    def test_runtime_approach_aligns_body_to_occupied_surface_before_close_approach(self) -> None:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        runtime = HomeAgentToolRuntime(
            memory=object_surface_memory(),
            config=HomeAgentConfig(
                exploration_backend_url="http://explore.local",
                agent_artifacts_root=tmpdir.name,
                object_detector_provider="mock",
            ),
            emit=lambda *_args, **_kwargs: None,
            run_id="surface_align_run",
        )
        runtime.current_pose = {"x": 1.25, "y": 1.25, "yaw": 0.0}
        runtime._track_detection_result(
            object_label="coke can",
            detection={
                "status": "matched",
                "selected_detection_id": "det_1",
                "selected_detection": {
                    "detection_id": "det_1",
                    "label": "coke can",
                    "confidence": 0.9,
                    "bbox_xyxy": [220, 120, 420, 360],
                },
            },
            capture={"shot_id": "shot_1", "image_width": 640, "image_height": 480},
        )
        current_pose = dict(runtime.current_pose)
        geometry_calls = 0
        local_motion_bodies = []

        def fake_urlopen(request, timeout=0):
            nonlocal geometry_calls
            body = json.loads(request.data.decode("utf-8")) if request.data else {}
            if request.full_url.endswith("/api/nav/estimate_detection_geometry"):
                geometry_calls += 1
                forward = 0.8 if geometry_calls == 1 else 0.42
                return FakeHTTPResponse(
                    {
                        "status": "succeeded",
                        "reason": "geometry solved",
                        "forward_m": forward,
                        "distance_m": forward,
                        "lateral_m": 0.0,
                        "bearing_error_deg": 0.0,
                        "estimated_pose_map": {"x": 2.75, "y": 2.85, "z": 0.2},
                        "current_pose": dict(current_pose),
                        "safety": {
                            "safe": True,
                            "safe_forward_step_m": 0.08 if geometry_calls == 1 else 0.0,
                            "reason": "clear",
                        },
                    }
                )
            if request.full_url.endswith("/api/nav/local_motion"):
                local_motion_bodies.append(body)
                pose = body["pose"]
                current_pose.update(pose)
                return FakeHTTPResponse(
                    {
                        "status": "succeeded",
                        "reason": "micro adjustment completed",
                        "local_motion": {
                            "primitive": "micro_adjust_to_pose",
                            "status": "succeeded",
                            "start_pose": dict(runtime.current_pose),
                            "end_pose": dict(current_pose),
                            "distance_remaining_m": 0.0,
                        },
                        "map": {"robot_pose": dict(current_pose)},
                    }
                )
            if request.full_url.endswith("/api/nav/capture_rgb"):
                return FakeHTTPResponse(
                    {
                        "status": "succeeded",
                        "reason": "captured",
                        "image_data_url": TEST_PNG_DATA_URL,
                        "robot_pose": dict(current_pose),
                        "captured_at": 123.0,
                    }
                )
            if request.full_url.endswith("/api/nav/relocalize"):
                return FakeHTTPResponse(
                    {
                        "status": "skipped",
                        "message": "Relocalization skipped in test.",
                        "map": {"robot_pose": dict(current_pose)},
                    }
                )
            raise AssertionError(f"unexpected URL {request.full_url}")

        with patch("xlerobot_agent.home_agent.urllib.request.urlopen", side_effect=fake_urlopen):
            result = runtime.approach_detected_object(
                object_label="coke can",
                constraints={"surface_alignment_max_distance_m": 3.0},
            )

        self.assertEqual(result["status"], "succeeded")
        self.assertGreaterEqual(geometry_calls, 2)
        self.assertTrue(local_motion_bodies)
        alignment_pose = local_motion_bodies[0]["pose"]
        self.assertAlmostEqual(alignment_pose["yaw"], 1.57, delta=0.3)
        self.assertIn("surface_alignment", result["attempts"][0])

    def test_runtime_approach_retry_refreshes_detection_without_surface_realign(self) -> None:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        runtime = HomeAgentToolRuntime(
            memory=object_surface_memory(),
            config=HomeAgentConfig(
                exploration_backend_url="http://explore.local",
                agent_artifacts_root=tmpdir.name,
                object_detector_provider="mock",
            ),
            emit=lambda *_args, **_kwargs: None,
            run_id="surface_retry_run",
        )
        runtime.object_approach_state["coke can"] = {
            "object_label": "coke can",
            "surface_alignment_attempted": True,
        }
        runtime._track_detection_result(
            object_label="coke can",
            detection={
                "status": "matched",
                "selected_detection_id": "old_det",
                "selected_detection": {
                    "detection_id": "old_det",
                    "label": "coke can",
                    "confidence": 0.9,
                    "bbox_xyxy": [220, 120, 420, 360],
                },
            },
            capture={"shot_id": "old_shot", "image_width": 640, "image_height": 480},
        )
        current_pose = dict(runtime.current_pose)
        urls = []

        def fake_urlopen(request, timeout=0):
            body = json.loads(request.data.decode("utf-8")) if request.data else {}
            urls.append(request.full_url)
            if request.full_url.endswith("/api/nav/capture_rgb"):
                return FakeHTTPResponse(
                    {
                        "status": "succeeded",
                        "reason": "captured",
                        "image_data_url": TEST_PNG_DATA_URL,
                        "robot_pose": dict(current_pose),
                        "captured_at": 123.0,
                    }
                )
            if request.full_url.endswith("/api/nav/estimate_detection_geometry"):
                return FakeHTTPResponse(
                    {
                        "status": "succeeded",
                        "reason": "geometry solved",
                        "forward_m": 0.8,
                        "distance_m": 0.8,
                        "lateral_m": 0.0,
                        "bearing_error_deg": 0.0,
                        "estimated_pose_map": {"x": 2.75, "y": 2.85, "z": 0.2},
                        "current_pose": dict(current_pose),
                        "safety": {
                            "safe": False,
                            "safe_forward_step_m": 0.0,
                            "reason": "object corridor is not clear",
                        },
                    }
                )
            if request.full_url.endswith("/api/nav/local_motion"):
                raise AssertionError(f"retry should not surface-realign or move locally: {body}")
            raise AssertionError(f"unexpected URL {request.full_url}")

        with patch("xlerobot_agent.home_agent.urllib.request.urlopen", side_effect=fake_urlopen):
            default_result = runtime.approach_detected_object(object_label="coke can")
            default_urls = list(urls)
            urls.clear()
            explicit_result = runtime.approach_detected_object(
                object_label="coke can",
                constraints={"allow_surface_alignment": False},
            )
            explicit_urls = list(urls)

        for result, call_urls in ((default_result, default_urls), (explicit_result, explicit_urls)):
            self.assertEqual(result["status"], "blocked")
            self.assertTrue(call_urls[0].endswith("/api/nav/capture_rgb"))
            self.assertEqual(result["attempts"][0]["source"], "detector_refresh")
            self.assertNotIn("surface_alignment", result["attempts"][0])
            self.assertEqual(result["attempts"][0]["retry_policy"]["surface_alignment_disabled"], True)
            self.assertIn("object corridor is not clear", result["reason"])

    def test_runtime_grab_object_is_mock_vla_entrypoint(self) -> None:
        runtime = HomeAgentToolRuntime(
            memory=sample_memory(),
            config=HomeAgentConfig(),
            emit=lambda *_args, **_kwargs: None,
        )

        result = runtime.grab_object(
            object_label="coke can",
            detection_id="det_1",
            object_description="red can centered at grasp range",
        )

        self.assertEqual(result["status"], "mock_succeeded")
        self.assertEqual(result["tool"], "grab_object")
        self.assertIn("future VLA", result["reason"])

    def test_runtime_execute_region_exploration_plan_skips_capture_when_alignment_fails(self) -> None:
        runtime = HomeAgentToolRuntime(
            memory=visual_sweep_memory(),
            config=HomeAgentConfig(exploration_backend_url="http://explore.local"),
            emit=lambda *_args, **_kwargs: None,
        )
        current_pose = dict(runtime.current_pose)
        calls = []

        def fake_urlopen(request, timeout=0):
            body = json.loads(request.data.decode("utf-8")) if request.data else {}
            calls.append((request.full_url, body))
            if request.full_url.endswith("/api/nav/waypoint"):
                pose = body["pose"]
                current_pose.update({"x": pose["x"], "y": pose["y"], "yaw": 0.0})
                return FakeHTTPResponse(
                    {
                        "status": "succeeded",
                        "reason": "Nav2 reached the requested goal pose",
                        "nav2_result": {
                            "status": "succeeded",
                            "reason": "Nav2 reached the requested goal pose",
                            "reached_pose": dict(current_pose),
                        },
                        "map": {"robot_pose": dict(current_pose)},
                    }
                )
            if request.full_url.endswith("/api/nav/local_motion"):
                return FakeHTTPResponse(
                    {
                        "status": "failed",
                        "reason": "rotation did not complete",
                        "local_motion": {
                            "primitive": body.get("primitive"),
                            "status": "failed",
                            "reason": "rotation did not complete",
                            "start_pose": dict(current_pose),
                            "end_pose": dict(current_pose),
                        },
                        "map": {"robot_pose": dict(current_pose)},
                    }
                )
            if request.full_url.endswith("/api/nav/capture_rgb"):
                raise AssertionError("capture should not run after failed shot alignment")
            raise AssertionError(f"unexpected URL {request.full_url}")

        with patch("xlerobot_agent.home_agent.urllib.request.urlopen", side_effect=fake_urlopen):
            result = runtime.execute_region_exploration_plan(
                region_label="kitchen",
                object_label="coke can",
                constraints={"max_stops": 1, "shots_per_stop": 1, "allow_auto_rotate": False},
            )

        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["captured_shot_count"], 0)
        self.assertEqual(result["saved_rgb_count"], 0)
        self.assertEqual(result["stops"][0]["shots"][0]["capture"]["status"], "skipped")
        self.assertFalse(any(url.endswith("/api/nav/capture_rgb") for url, _body in calls))

    def test_runtime_navigate_does_not_treat_normalized_goal_as_current_pose(self) -> None:
        runtime = HomeAgentToolRuntime(
            memory=sample_memory(),
            config=HomeAgentConfig(exploration_backend_url="http://explore.local"),
            emit=lambda *_args, **_kwargs: None,
        )
        payload = {
            "status": "failed",
            "reason": "controller patience exceeded",
            "normalized_pose": {"x": 3.0, "y": 0.0, "yaw": 0.0},
            "nav2_result": {
                "status": "failed",
                "reason": "controller patience exceeded",
                "reached_pose": None,
                "plan": {"status": "succeeded", "path_length_m": 3.0},
            },
            "map": {"robot_pose": {"x": 0.8, "y": 0.1, "yaw": 0.0}},
        }
        with patch("xlerobot_agent.home_agent.urllib.request.urlopen", return_value=FakeHTTPResponse(payload)):
            result = runtime.navigate_to_waypoint(waypoint_id="kitchen_step", x=3.0, y=0.0, yaw=0.0)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["current_pose"], {"x": 0.8, "y": 0.1, "yaw": 0.0})
        self.assertEqual(result["normalized_pose"], {"x": 3.0, "y": 0.0, "yaw": 0.0})
        self.assertEqual(runtime.current_pose, {"x": 0.8, "y": 0.1, "yaw": 0.0})

    def test_runtime_local_motion_calls_exploration_backend(self) -> None:
        events = []
        runtime = HomeAgentToolRuntime(
            memory=sample_memory(),
            config=HomeAgentConfig(exploration_backend_url="http://explore.local"),
            emit=lambda kind, title, summary, details=None: events.append(
                {"kind": kind, "title": title, "summary": summary, "details": details or {}}
            ),
        )
        payload = {
            "status": "succeeded",
            "reason": "rotate toward target point",
            "local_motion": {
                "primitive": "rotate_towards_point",
                "status": "succeeded",
                "reason": "target_yaw_reached",
                "start_pose": {"x": 0.0, "y": 0.0, "yaw": 0.0},
                "end_pose": {"x": 0.0, "y": 0.0, "yaw": 1.57},
                "actual_pose_delta_m": 0.0,
                "actual_yaw_delta_deg": 90.0,
            },
            "map": {"robot_pose": {"x": 0.0, "y": 0.0, "yaw": 1.57}},
        }
        with patch("xlerobot_agent.home_agent.urllib.request.urlopen", return_value=FakeHTTPResponse(payload)) as mocked:
            result = runtime.rotate_towards_point(x=2.0, y=0.0, reason="face waypoint before Nav2")

        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["tool"], "rotate_towards_point")
        self.assertEqual(runtime.current_pose, {"x": 0.0, "y": 0.0, "yaw": 1.57})
        request = mocked.call_args.args[0]
        self.assertEqual(request.full_url, "http://explore.local/api/nav/local_motion")
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body["primitive"], "rotate_towards_point")
        self.assertTrue(any(event["details"].get("tool") == "rotate_towards_point" for event in events))

    def test_runtime_relocalize_here_reuses_exploration_backend(self) -> None:
        runtime = HomeAgentToolRuntime(
            memory=sample_memory(),
            config=HomeAgentConfig(exploration_backend_url="http://explore.local"),
            emit=lambda *_args, **_kwargs: None,
        )
        payload = {
            "status": "corrected",
            "message": "Relocalization correction applied to odometry pose.",
            "match": {
                "status": "matched",
                "confidence": 0.82,
                "delta": {"dx_m": -0.1, "dy_m": 0.2, "dyaw_deg": 2.0},
                "corrected_pose": {"x": 0.9, "y": 0.2, "yaw": 0.03},
            },
            "correction": {"status": "ok"},
        }
        with patch("xlerobot_agent.home_agent.urllib.request.urlopen", return_value=FakeHTTPResponse(payload)) as mocked:
            result = runtime.relocalize_here()

        self.assertEqual(result["status"], "corrected")
        self.assertEqual(result["current_pose"], {"x": 0.9, "y": 0.2, "yaw": 0.03})
        self.assertEqual(runtime.current_pose, {"x": 0.9, "y": 0.2, "yaw": 0.03})
        request = mocked.call_args.args[0]
        self.assertEqual(request.full_url, "http://explore.local/api/nav/relocalize")

    def test_navigation_happy_path_resolves_moves_relocalizes_and_resolves_again(self) -> None:
        memory = sample_memory()
        memory["start_pose"] = {"name": "dock", "pose": {"x": 0.75, "y": 2.0, "yaw": 0.0}}
        memory["regions"][0]["default_waypoints"] = []
        memory["regions"][0]["polygon_2d"] = [[5.0, 1.0], [7.0, 1.0], [7.0, 3.0], [5.0, 3.0]]
        memory["occupancy"] = {
            "resolution": 0.25,
            "bounds": {"min_x": 0.0, "min_y": 0.0},
            "cells": [
                {
                    "x": x * 0.25,
                    "y": y * 0.25,
                    "state": "occupied" if x in {0, 35} or y in {0, 19} else "free",
                }
                for x in range(36)
                for y in range(20)
            ],
        }
        runtime = HomeAgentToolRuntime(
            memory=memory,
            config=HomeAgentConfig(
                exploration_backend_url="http://explore.local",
                navigation_waypoint_horizon_m=1.0,
            ),
            emit=lambda *_args, **_kwargs: None,
        )

        first = runtime.resolve_navigation_to_region(target_label="kitchen")
        self.assertEqual(first["status"], "succeeded")
        self.assertFalse(first["next_waypoint"]["is_final_waypoint"])
        self.assertGreater(first["path_length_m"], 1.0)
        waypoint = first["next_waypoint"]
        last_pose = {}
        called_paths = []

        def fake_urlopen(request, timeout=0):
            called_paths.append(request.full_url)
            body = json.loads(request.data.decode("utf-8")) if request.data else {}
            if request.full_url.endswith("/api/nav/waypoint"):
                pose = body["pose"]
                last_pose.clear()
                last_pose.update(pose)
                return FakeHTTPResponse(
                    {
                        "status": "succeeded",
                        "reason": "Nav2 reached the requested goal pose",
                        "nav2_result": {
                            "status": "succeeded",
                            "reason": "Nav2 reached the requested goal pose",
                            "reached_pose": pose,
                            "travelled_distance_m": waypoint["distance_from_start_m"],
                            "plan": {"status": "succeeded", "path_length_m": waypoint["distance_from_start_m"]},
                            "feedback_samples": [{"remaining_distance_m": 0.0}],
                        },
                        "map": {"robot_pose": pose},
                    }
                )
            if request.full_url.endswith("/api/nav/relocalize"):
                corrected = {
                    "x": round(float(last_pose["x"]) + 0.05, 3),
                    "y": round(float(last_pose["y"]), 3),
                    "yaw": round(float(last_pose.get("yaw", 0.0)), 3),
                }
                return FakeHTTPResponse(
                    {
                        "status": "corrected",
                        "message": "Relocalization correction applied to odometry pose.",
                        "match": {
                            "status": "matched",
                            "confidence": 0.9,
                            "delta": {"dx_m": 0.05, "dy_m": 0.0, "dyaw_deg": 0.0},
                            "corrected_pose": corrected,
                        },
                        "correction": {"status": "ok"},
                    }
                )
            raise AssertionError(f"unexpected URL {request.full_url}")

        with patch("xlerobot_agent.home_agent.urllib.request.urlopen", side_effect=fake_urlopen):
            nav = runtime.navigate_to_waypoint(
                waypoint_id=waypoint["waypoint_id"],
                x=waypoint["x"],
                y=waypoint["y"],
                yaw=waypoint["yaw"],
            )
            relocalized = runtime.relocalize_here()
            second = runtime.resolve_navigation_to_region(target_label="kitchen")

        self.assertEqual(nav["status"], "succeeded")
        self.assertEqual(nav["distance_remaining_m"], 0.0)
        self.assertEqual(relocalized["status"], "corrected")
        self.assertEqual(second["status"], "succeeded")
        self.assertLess(second["path_length_m"], first["path_length_m"])
        self.assertEqual(
            called_paths,
            [
                "http://explore.local/api/state",
                "http://explore.local/api/nav/waypoint",
                "http://explore.local/api/nav/relocalize",
            ],
        )

    def test_agent_instructions_include_navigation_tool_loop_examples(self) -> None:
        agent = HomeTaskAgent(HomeAgentConfig(navigation_waypoint_horizon_m=2.0))
        instructions = agent._agent_instructions(sample_memory())
        self.assertIn("Example navigation loop", instructions)
        self.assertIn("navigate_to_waypoint", instructions)
        self.assertIn("relocalize_here", instructions)
        self.assertIn("rotate_towards_point", instructions)
        self.assertIn("micro_adjust_to_pose", instructions)
        self.assertIn("execute_region_exploration_plan", instructions)
        self.assertIn("focus_detected_object", instructions)
        self.assertIn("approach_detected_object", instructions)
        self.assertIn("grab_object", instructions)
        self.assertIn("saves RGB debug shots", instructions)
        self.assertIn("RGB-D", instructions)
        self.assertIn("Nav2 can sometimes fail to find paths", instructions)
        self.assertIn("constraints_json='{}'", instructions)
        self.assertIn("After each successful waypoint", instructions)
        self.assertIn("Do not use exploration stops as a shortcut for long-distance navigation", instructions)
        self.assertIn("Example far object-search request", instructions)
        self.assertIn("Bad example", instructions)

    def test_openai_provider_applies_cli_api_key_to_agents_sdk_env(self) -> None:
        agent = HomeTaskAgent(
            HomeAgentConfig(
                model=HomeAgentModelConfig(provider="openai", model="gpt-test", api_key="test-key"),
            )
        )
        previous = os.environ.pop("OPENAI_API_KEY", None)
        try:
            model = agent._sdk_model()
            self.assertEqual(model, "gpt-test")
            self.assertEqual(os.environ.get("OPENAI_API_KEY"), "test-key")
        finally:
            if previous is not None:
                os.environ["OPENAI_API_KEY"] = previous
            else:
                os.environ.pop("OPENAI_API_KEY", None)

    def test_gpt55_agents_sdk_settings_omit_temperature(self) -> None:
        class FakeModelSettings:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        agent = HomeTaskAgent(
            HomeAgentConfig(
                model=HomeAgentModelConfig(
                    provider="openai",
                    model="gpt-5.5",
                    temperature=0.7,
                    max_tokens=4096,
                    reasoning_effort="high",
                    verbosity="medium",
                ),
            )
        )

        settings = agent._sdk_model_settings(FakeModelSettings)

        self.assertNotIn("temperature", settings.kwargs)
        self.assertEqual(settings.kwargs["max_tokens"], 4096)
        self.assertEqual(settings.kwargs["verbosity"], "medium")
        reasoning = settings.kwargs["reasoning"]
        if isinstance(reasoning, dict):
            self.assertEqual(reasoning["effort"], "high")
        else:
            self.assertEqual(reasoning.effort, "high")

    def test_non_openai_gpt55_compatible_settings_keep_temperature(self) -> None:
        class FakeModelSettings:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        agent = HomeTaskAgent(
            HomeAgentConfig(
                model=HomeAgentModelConfig(
                    provider="openai-compatible",
                    model="gemini-1.6-er",
                    temperature=0.7,
                    max_tokens=4096,
                    reasoning_effort="high",
                    verbosity="medium",
                ),
            )
        )

        settings = agent._sdk_model_settings(FakeModelSettings)

        self.assertEqual(settings.kwargs["temperature"], 0.7)
        self.assertEqual(settings.kwargs["max_tokens"], 4096)
        self.assertNotIn("reasoning", settings.kwargs)
        self.assertNotIn("verbosity", settings.kwargs)


if __name__ == "__main__":
    unittest.main()
