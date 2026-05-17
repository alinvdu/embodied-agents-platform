import json
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
    resolve_home_memory_target,
    resolve_region_navigation_goal,
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
        request = mocked.call_args.args[0]
        self.assertEqual(request.full_url, "http://explore.local/api/nav/waypoint")
        self.assertTrue(any(event["details"].get("tool") == "navigate_to_waypoint" for event in events))

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
            ["http://explore.local/api/nav/waypoint", "http://explore.local/api/nav/relocalize"],
        )

    def test_agent_instructions_include_navigation_tool_loop_examples(self) -> None:
        agent = HomeTaskAgent(HomeAgentConfig(navigation_waypoint_horizon_m=2.0))
        instructions = agent._agent_instructions(sample_memory())
        self.assertIn("Example navigation loop", instructions)
        self.assertIn("navigate_to_waypoint", instructions)
        self.assertIn("relocalize_here", instructions)
        self.assertIn("constraints_json='{}'", instructions)
        self.assertIn("After each successful waypoint", instructions)

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
