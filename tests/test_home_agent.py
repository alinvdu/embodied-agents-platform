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
from xlerobot_agent.home_memory import home_memory_agent_context, resolve_home_memory_target


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


class HomeTaskAgentTests(unittest.TestCase):
    def test_mock_agent_previews_navigation_to_known_region(self) -> None:
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
        self.assertTrue(any(action.get("tool") == "preview_path_to_pose" for action in record.actions))
        self.assertTrue(any(event["kind"] == "memory_resolved" for event in events))

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

    def test_mock_agent_stages_skill_with_approval_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = Path(tmpdir) / "house.home_memory.json"
            memory_path.write_text(json.dumps(sample_memory()))
            record = HomeTaskAgent(
                HomeAgentConfig(
                    home_memory_path=str(memory_path),
                    model=HomeAgentModelConfig(provider="mock", model="mock"),
                )
            ).run("open the fridge")
        self.assertEqual(record.status, "completed")
        skill_actions = [action for action in record.actions if action.get("tool") == "run_skill"]
        self.assertEqual(skill_actions[-1]["status"], "approval_required")

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
