import json
import os
from pathlib import Path
import tempfile
import time
import unittest

from xlerobot_agent.home_agent import (
    HomeAgentConfig,
    HomeAgentController,
    HomeAgentModelConfig,
    HomeAgentToolRuntime,
    HomeTaskAgent,
    discover_latest_home_memory_path,
)
from xlerobot_agent.home_memory import (
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


if __name__ == "__main__":
    unittest.main()
