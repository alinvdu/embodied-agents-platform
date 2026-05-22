from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from xlerobot_agent.exploration import ExplorationBackend, ExplorationBackendConfig
from xlerobot_agent.home_memory import summarize_home_memory
from xlerobot_playground.map_editing import ManualOccupancyEdits, overlay_occupancy_payload


@dataclass(frozen=True)
class _Cell:
    x: int
    y: int


class ExplorationBackendExternalTaskTests(unittest.TestCase):
    def test_external_task_updates_and_completes_with_named_places(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ExplorationBackend(
                ExplorationBackendConfig(
                    mode="sim",
                    persist_path=f"{tmpdir}/map.json",
                    occupancy_resolution=0.25,
                )
            )
            task = backend.begin_external_task(
                tool_id="explore",
                area="workspace",
                session="house_v1",
                source="operator",
            )
            backend.update_external_task(
                task["task_id"],
                progress=0.4,
                message="Exploring",
                result={"trajectory": [{"x": 0.0, "y": 0.0, "yaw": 0.0}]},
            )
            map_payload = {
                "map_id": "house_v1",
                "frame": "map",
                "resolution": 0.25,
                "coverage": 12.0,
                "summary": "test map",
                "approved": False,
                "created_at": 1.0,
                "source": "operator",
                "mode": "sim",
                "trajectory": [{"x": 0.0, "y": 0.0, "yaw": 0.0}],
                "keyframes": [],
                "regions": [
                    {
                        "region_id": "region_01",
                        "label": "kitchen",
                        "confidence": 0.8,
                        "polygon_2d": [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],
                        "centroid": {"x": 1.0, "y": 1.0},
                        "adjacency": [],
                        "representative_keyframes": [],
                        "evidence": ["fridge visible"],
                        "default_waypoints": [{"name": "kitchen_center", "x": 1.0, "y": 1.0, "yaw": 0.0}],
                    }
                ],
                "named_places": [],
                "occupancy": {
                    "resolution": 0.25,
                    "bounds": {"min_x": 0.0, "max_x": 2.0, "min_y": 0.0, "max_y": 2.0},
                    "cells": [{"x": 0.0, "y": 0.0, "state": "free"}],
                },
            }
            backend.complete_external_task(task["task_id"], map_payload=map_payload)

            snapshot = backend.snapshot()
            self.assertEqual(snapshot["active_task"]["state"], "succeeded")
            self.assertEqual(snapshot["current_map"]["map_id"], "house_v1")
            named_places = {item["name"] for item in snapshot["current_map"]["named_places"]}
            self.assertNotIn("kitchen_entry", named_places)
            self.assertNotIn("kitchen_center", named_places)

    def test_manual_occupancy_edits_respect_shifted_map_origin(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ExplorationBackend(
                ExplorationBackendConfig(
                    mode="sim",
                    persist_path=f"{tmpdir}/map.json",
                    occupancy_resolution=0.25,
                )
            )
            task = backend.begin_external_task(
                tool_id="explore",
                area="workspace",
                session="shifted_origin",
                source="operator",
            )
            map_payload = {
                "map_id": "shifted_origin",
                "frame": "map",
                "resolution": 0.25,
                "coverage": 0.5,
                "summary": "shifted origin map",
                "approved": False,
                "created_at": 1.0,
                "source": "operator",
                "mode": "sim",
                "trajectory": [],
                "keyframes": [],
                "regions": [],
                "named_places": [],
                "occupancy": {
                    "resolution": 0.25,
                    "bounds": {"min_x": -4.0, "max_x": 4.0, "min_y": -2.0, "max_y": 6.0},
                    "cells": [
                        {"x": -4.0, "y": -2.0, "state": "free"},
                        {"x": 2.0, "y": 2.0, "state": "occupied", "manual_override": "blocked"},
                    ],
                },
            }
            backend.complete_external_task(task["task_id"], map_payload=map_payload)

            backend.update_occupancy_edits(
                task_id=task["task_id"],
                mode="block",
                cells=[{"cell_x": 8, "cell_y": 8}],
            )

            snapshot = backend.snapshot()
            manual_cells = [
                cell
                for cell in snapshot["current_map"]["occupancy"]["cells"]
                if cell.get("manual_override") == "blocked"
            ]
            self.assertEqual(manual_cells, [{"x": -2.0, "y": 0.0, "state": "occupied", "manual_override": "blocked"}])

    def test_shared_occupancy_overlay_respects_shifted_map_origin(self) -> None:
        occupancy = {
            "resolution": 0.25,
            "bounds": {"min_x": -4.0, "max_x": 4.0, "min_y": -2.0, "max_y": 6.0},
            "cells": [
                {"x": -4.0, "y": -2.0, "state": "free"},
                {"x": 2.0, "y": 2.0, "state": "occupied", "manual_override": "blocked"},
            ],
        }
        payload = overlay_occupancy_payload(
            occupancy,
            edits=ManualOccupancyEdits(blocked_cells={_Cell(8, 8)}),
        )

        assert payload is not None
        manual_cells = [cell for cell in payload["cells"] if cell.get("manual_override") == "blocked"]
        self.assertEqual(manual_cells, [{"x": -2.0, "y": 0.0, "state": "occupied", "manual_override": "blocked"}])

    def test_manual_edit_metadata_serializes_with_shifted_map_origin(self) -> None:
        edits = ManualOccupancyEdits(blocked_cells={_Cell(8, 8)})

        payload = edits.to_dict(resolution=0.25, origin_x=-4.0, origin_y=-2.0)

        self.assertEqual(payload["blocked_cells"], [{"cell_x": 8, "cell_y": 8, "x": -2.0, "y": 0.0}])
        self.assertEqual(payload["cleared_cells"], [])

    def test_restore_rebuilds_persisted_manual_overlay_with_map_origin(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = Path(tmpdir) / "map.json"
            persist_path.write_text(
                json.dumps(
                    {
                        "current_map": {
                            "map_id": "shifted_origin",
                            "frame": "map",
                            "resolution": 0.25,
                            "coverage": 0.5,
                            "summary": "shifted origin map",
                            "approved": False,
                            "created_at": 1.0,
                            "source": "operator",
                            "mode": "sim",
                            "trajectory": [],
                            "keyframes": [],
                            "regions": [],
                            "named_places": [],
                            "artifacts": {"manual_occupancy_edits": {"blocked_cells": [{"cell_x": 8, "cell_y": 8}], "cleared_cells": []}},
                            "occupancy": {
                                "resolution": 0.25,
                                "bounds": {"min_x": -4.0, "max_x": 4.0, "min_y": -2.0, "max_y": 6.0},
                                "cells": [
                                    {"x": -4.0, "y": -2.0, "state": "free"},
                                    {"x": 2.0, "y": 2.0, "state": "occupied", "manual_override": "blocked"},
                                ],
                            },
                        },
                        "maps": [],
                        "tasks": [{"task_id": "task-1", "tool_id": "explore", "state": "succeeded"}],
                    }
                )
            )

            backend = ExplorationBackend(
                ExplorationBackendConfig(
                    mode="sim",
                    persist_path=str(persist_path),
                    occupancy_resolution=0.25,
                )
            )

            snapshot = backend.snapshot()
            manual_cells = [
                cell
                for cell in snapshot["current_map"]["occupancy"]["cells"]
                if cell.get("manual_override") == "blocked"
            ]
            self.assertEqual(manual_cells, [{"x": -2.0, "y": 0.0, "state": "occupied", "manual_override": "blocked"}])

    def test_approve_current_map_exports_home_memory_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = Path(tmpdir) / "real_xlerobot_exploration_map.json"
            backend = ExplorationBackend(
                ExplorationBackendConfig(
                    mode="sim",
                    persist_path=str(persist_path),
                    occupancy_resolution=0.25,
                )
            )
            task = backend.begin_external_task(
                tool_id="explore",
                area="workspace",
                session="house_v1",
                source="operator",
            )
            map_payload = {
                "map_id": "house_v1",
                "frame": "map",
                "resolution": 0.25,
                "coverage": 0.5,
                "summary": "approved home map",
                "approved": False,
                "created_at": 1.0,
                "source": "operator",
                "mode": "sim",
                "robot_pose": {"x": 0.25, "y": 0.5, "yaw": 0.0},
                "trajectory": [{"x": 0.0, "y": 0.0, "yaw": 0.0}],
                "keyframes": [{"frame_id": "frame_1", "description": "debug-only frame"}],
                "regions": [
                    {
                        "region_id": "region_kitchen",
                        "label": "kitchen",
                        "confidence": 0.9,
                        "polygon_2d": [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],
                        "centroid": {"x": 1.0, "y": 1.0},
                        "adjacency": ["region_hallway"],
                        "representative_keyframes": ["frame_1"],
                        "evidence": ["fridge visible"],
                        "default_waypoints": [{"name": "kitchen_center", "x": 1.0, "y": 1.0, "yaw": 1.57}],
                    }
                ],
                "named_places": [
                    {
                        "name": "dock",
                        "pose": {"x": 0.0, "y": 0.0, "yaw": 0.0},
                        "region_id": "region_hallway",
                        "source": "manual",
                    },
                    {
                        "name": "fridge",
                        "pose": {"x": 1.8, "y": 1.0, "yaw": 3.14},
                        "region_id": "region_kitchen",
                        "source": "manual",
                    },
                ],
                "occupancy": {
                    "resolution": 0.25,
                    "bounds": {"min_x": 0.0, "max_x": 2.0, "min_y": 0.0, "max_y": 2.0},
                    "cells": [{"x": 0.0, "y": 0.0, "state": "free"}],
                },
                "frontiers": [{"frontier_id": "debug_frontier"}],
                "artifacts": {"decision_log": [{"decision": "debug-only"}]},
            }
            backend.complete_external_task(task["task_id"], map_payload=map_payload)
            backend.update_occupancy_edits(
                task_id=task["task_id"],
                mode="block",
                cells=[{"cell_x": 2, "cell_y": 2}],
            )

            approved = backend.approve_current_map()

            assert approved is not None
            memory_dir = persist_path.parent / "memories" / "house_v1"
            home_memory_path = memory_dir / "home_memory.json"
            environment_map_path = memory_dir / "environment_map.json"
            self.assertTrue(persist_path.exists())
            self.assertTrue(home_memory_path.exists())
            self.assertTrue(environment_map_path.exists())
            memory = json.loads(home_memory_path.read_text())
            self.assertEqual(memory["schema_version"], "home_memory.v1")
            self.assertEqual(memory["memory_id"], "house_v1")
            self.assertTrue(memory["approved"])
            self.assertEqual(memory["start_pose"]["name"], "start")
            self.assertEqual(memory["start_pose"]["source"], "default_map_origin")
            self.assertEqual(memory["regions"][0]["label"], "kitchen")
            self.assertEqual(memory["manual_occupancy_edits"]["blocked_cells"][0]["cell_x"], 2)
            self.assertIn("fridge", {item["label"] for item in memory["objects"]})
            self.assertIn("open_fridge", {item["skill_id"] for item in memory["skills"]})
            self.assertIn("keyframes", memory["export_notes"]["preserved_in_map_only"])
            self.assertIn("kitchen", summarize_home_memory(memory))
            persisted = json.loads(persist_path.read_text())
            self.assertEqual(persisted["current_map"]["frontiers"], [{"frontier_id": "debug_frontier"}])
            self.assertEqual(persisted["current_map"]["keyframes"], [{"frame_id": "frame_1", "description": "debug-only frame"}])
            self.assertIn("home_memory", persisted["current_map"]["artifacts"])
            self.assertEqual(persisted["current_map"]["artifacts"]["home_memory"]["directory"], str(memory_dir))

            restored = ExplorationBackend(
                ExplorationBackendConfig(
                    mode="sim",
                    persist_path=str(persist_path.parent / "fresh_backend.json"),
                    memory_root_path=str(persist_path.parent / "memories"),
                )
            )
            loaded = restored.load_environment_memory("house_v1")
            assert loaded is not None
            self.assertEqual(loaded["current_map"]["regions"][0]["label"], "kitchen")

    def test_set_dock_pose_sets_start_pose_and_memory_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = Path(tmpdir) / "map.json"
            backend = ExplorationBackend(
                ExplorationBackendConfig(mode="sim", persist_path=str(persist_path))
            )
            task = backend.begin_external_task(
                tool_id="explore",
                area="workspace",
                session="house_v1",
            )
            backend.complete_external_task(
                task["task_id"],
                map_payload={
                    "map_id": "house_v1",
                    "frame": "map",
                    "resolution": 0.25,
                    "coverage": 1.0,
                    "summary": "dock pose map",
                    "approved": False,
                    "created_at": 1.0,
                    "source": "operator",
                    "mode": "sim",
                    "robot_pose": {"x": 0.5, "y": 0.75, "yaw": 1.57},
                    "trajectory": [],
                    "keyframes": [],
                    "regions": [
                        {
                            "region_id": "region_hallway",
                            "label": "hallway",
                            "polygon_2d": [[0, 0], [1, 0], [1, 1], [0, 1]],
                            "centroid": {"x": 0.5, "y": 0.5},
                            "default_waypoints": [],
                        }
                    ],
                    "named_places": [],
                    "occupancy": {
                        "resolution": 0.25,
                        "bounds": {"min_x": 0.0, "max_x": 1.0, "min_y": 0.0, "max_y": 1.0},
                        "cells": [{"x": 0.0, "y": 0.0, "state": "free"}],
                    },
                },
            )

            dock = backend.set_dock_pose()
            approved = backend.approve_current_map()

            assert dock is not None
            assert approved is not None
            self.assertEqual(dock["name"], "dock")
            self.assertEqual(dock["pose"], {"x": 0.5, "y": 0.75, "yaw": 1.57})
            self.assertEqual(approved["start_pose"]["pose"], {"x": 0.5, "y": 0.75, "yaw": 1.57})
            memory_path = persist_path.parent / "memories" / "house_v1" / "home_memory.json"
            memory = json.loads(memory_path.read_text())
            self.assertEqual(memory["start_pose"]["pose"], {"x": 0.5, "y": 0.75, "yaw": 1.57})
            self.assertEqual(memory["start_pose"]["source"], "operator_dock_pose")

    def test_home_memory_defaults_start_pose_to_map_origin_without_dock(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = Path(tmpdir) / "map.json"
            backend = ExplorationBackend(
                ExplorationBackendConfig(mode="sim", persist_path=str(persist_path))
            )
            task = backend.begin_external_task(
                tool_id="explore",
                area="workspace",
                session="house_v1",
            )
            backend.complete_external_task(
                task["task_id"],
                map_payload={
                    "map_id": "house_v1",
                    "frame": "map",
                    "resolution": 0.25,
                    "coverage": 1.0,
                    "summary": "map without dock",
                    "approved": False,
                    "created_at": 1.0,
                    "source": "operator",
                    "mode": "sim",
                    "robot_pose": {"x": 0.5, "y": 0.75, "yaw": 1.57},
                    "trajectory": [{"x": 0.0, "y": 0.0, "yaw": 0.0}],
                    "keyframes": [],
                    "regions": [],
                    "named_places": [],
                    "occupancy": {
                        "resolution": 0.25,
                        "bounds": {"min_x": 0.0, "max_x": 1.0, "min_y": 0.0, "max_y": 1.0},
                        "cells": [{"x": 0.0, "y": 0.0, "state": "free"}],
                    },
                },
            )

            approved = backend.approve_current_map()

            assert approved is not None
            memory_path = persist_path.parent / "memories" / "house_v1" / "home_memory.json"
            memory = json.loads(memory_path.read_text())
            self.assertEqual(memory["start_pose"]["name"], "start")
            self.assertEqual(memory["start_pose"]["pose"], {"x": 0.0, "y": 0.0, "yaw": 0.0})
            self.assertEqual(memory["start_pose"]["source"], "default_map_origin")

    def test_create_region_adds_manual_polygon_and_derived_waypoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ExplorationBackend(
                ExplorationBackendConfig(
                    mode="sim",
                    persist_path=f"{tmpdir}/map.json",
                    occupancy_resolution=0.25,
                )
            )
            task = backend.begin_external_task(
                tool_id="explore",
                area="workspace",
                session="manual_regions",
                source="operator",
            )
            backend.complete_external_task(
                task["task_id"],
                map_payload={
                    "map_id": "manual_regions",
                    "frame": "map",
                    "resolution": 0.25,
                    "coverage": 0.5,
                    "summary": "manual region map",
                    "approved": False,
                    "created_at": 1.0,
                    "source": "operator",
                    "mode": "sim",
                    "trajectory": [],
                    "keyframes": [],
                    "regions": [],
                    "named_places": [],
                    "occupancy": {
                        "resolution": 0.25,
                        "bounds": {"min_x": 0.0, "max_x": 4.0, "min_y": 0.0, "max_y": 4.0},
                        "cells": [{"x": 0.0, "y": 0.0, "state": "free"}],
                    },
                },
            )

            region = backend.create_region(
                label="Kitchen Zone",
                polygon_2d=[[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]],
                purpose="food prep",
            )

            assert region is not None
            self.assertTrue(region["region_id"].startswith("region_kitchen_zone_"))
            self.assertEqual(region["label"], "Kitchen Zone")
            self.assertEqual(region["centroid"], {"x": 1.0, "y": 0.5})
            self.assertEqual(region["default_waypoints"], [])
            snapshot = backend.snapshot()
            self.assertEqual(snapshot["current_map"]["regions"][0]["purpose"], "food prep")
            self.assertEqual(snapshot["current_map"]["named_places"], [])

    def test_region_waypoints_without_names_are_normalized_on_save(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ExplorationBackend(
                ExplorationBackendConfig(
                    mode="sim",
                    persist_path=f"{tmpdir}/map.json",
                    memory_root_path=f"{tmpdir}/memories",
                    occupancy_resolution=0.25,
                )
            )
            task = backend.begin_external_task(
                tool_id="explore",
                area="workspace",
                session="manual_regions",
                source="operator",
            )
            backend.complete_external_task(
                task["task_id"],
                map_payload={
                    "map_id": "manual_regions",
                    "frame": "map",
                    "resolution": 0.25,
                    "coverage": 0.5,
                    "summary": "manual region map",
                    "approved": False,
                    "created_at": 1.0,
                    "source": "operator",
                    "mode": "sim",
                    "trajectory": [],
                    "keyframes": [],
                    "regions": [
                        {
                            "region_id": "region_kitchen",
                            "label": "Kitchen",
                            "confidence": 1.0,
                            "polygon_2d": [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]],
                            "centroid": {"x": 1.0, "y": 0.5},
                            "adjacency": [],
                            "representative_keyframes": [],
                            "evidence": [],
                            "default_waypoints": [{"x": 1.4, "y": 0.6, "yaw": 0.0}],
                        }
                    ],
                    "named_places": [],
                    "occupancy": {
                        "resolution": 0.25,
                        "bounds": {"min_x": 0.0, "max_x": 4.0, "min_y": 0.0, "max_y": 4.0},
                        "cells": [{"x": 0.0, "y": 0.0, "state": "free"}],
                    },
                },
            )

            snapshot = backend.snapshot()
            waypoint = snapshot["current_map"]["regions"][0]["default_waypoints"][0]
            self.assertEqual(waypoint["name"], "kitchen_waypoint_1")
            self.assertEqual(snapshot["current_map"]["named_places"][0]["name"], "kitchen_waypoint_1")

            updated = backend.update_region(
                "region_kitchen",
                default_waypoints=[
                    {"x": 1.5, "y": 0.7, "yaw": 0.1},
                    {"name": "kitchen_entry", "x": 0.2, "y": 0.5, "yaw": 1.57},
                ],
            )
            assert updated is not None
            self.assertEqual(updated["default_waypoints"][0]["name"], "kitchen_waypoint_1")
            self.assertEqual(updated["default_waypoints"][1]["name"], "kitchen_entry")
            approved = backend.approve_current_map()
            assert approved is not None
            self.assertEqual(approved["named_places"][0]["name"], "kitchen_waypoint_1")
            self.assertEqual(approved["named_places"][1]["name"], "kitchen_entry")


if __name__ == "__main__":
    unittest.main()
