from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from xlerobot_agent.exploration import ExplorationBackend, ExplorationBackendConfig
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
            self.assertIn("kitchen_entry", named_places)
            self.assertIn("kitchen_center", named_places)

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


if __name__ == "__main__":
    unittest.main()
