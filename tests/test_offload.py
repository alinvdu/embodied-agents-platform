import json
import time
import unittest
import urllib.request
from typing import Any

from xlerobot_agent import (
    GoalContext,
    PerceptionService,
    PerceptionServiceConfig,
    SkillContract,
    SkillType,
    Subgoal,
    WorldState,
)
from xlerobot_agent.brain_service import BrainBridge, BrainServiceServer
from xlerobot_agent.offload import OffloadClient, OffloadServer, OffloadServerConfig
from xlerobot_agent.tools import ToolCallContext


class OffloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = OffloadServer(OffloadServerConfig(host="127.0.0.1", port=0))
        self.server.start_in_thread()
        self.server_url = f"http://{self.server.host}:{self.server.port}"

    def tearDown(self) -> None:
        self.server.shutdown()

    def _wait_for_task(self, client: OffloadClient, task_id: str, world_state: WorldState, subgoal: Subgoal) -> dict[str, Any]:
        deadline = time.time() + 5.0
        while time.time() < deadline:
            status = client.execute_tool(
                "get_task_status",
                ToolCallContext(
                    goal=GoalContext(subgoal.text),
                    subgoal=subgoal,
                    world_state=world_state,
                    payload={"task_id": task_id},
                ),
            )
            task = status["details"]["task"]
            if task["state"] == "succeeded":
                return status
            time.sleep(0.05)
        self.fail(f"Timed out waiting for task {task_id}")

    def test_offload_client_registers_publishes_and_executes(self) -> None:
        client = OffloadClient(self.server_url, brain_name="test-brain")
        registration = client.register()
        self.assertTrue(registration.brain_id)

        world_state = WorldState(
            current_task="Navigate to the kitchen",
            current_pose="hallway",
            metadata={"sensors": {"rgb": {"width": 640}, "point_cloud": {"points": 1200}}},
        )
        client.publish_state(world_state, reason="unit_test")
        map_before = client.execute_tool(
            "get_map",
            ToolCallContext(
                goal=GoalContext("Navigate to the kitchen"),
                subgoal=Subgoal(text="go to the kitchen", kind="navigate", target="kitchen"),
                world_state=world_state,
                payload={"summary_only": True},
            ),
        )
        self.assertEqual(map_before["status"], "succeeded")
        self.assertFalse(map_before["details"]["available"])

        map_task = client.execute_tool(
            "create_map",
            ToolCallContext(
                goal=GoalContext("Map the kitchen"),
                subgoal=Subgoal(text="build a map", kind="general", target="kitchen"),
                world_state=world_state,
                payload={"session": "house_v1", "area": "kitchen"},
            ),
        )
        self.assertEqual(map_task["status"], "in_progress")
        self.assertEqual(map_task["recommended_action_id"], "get_task_status")
        map_status = self._wait_for_task(
            client,
            map_task["details"]["task"]["task_id"],
            world_state,
            Subgoal(text="build a map", kind="general", target="kitchen"),
        )
        self.assertEqual(map_status["details"]["task"]["result"]["map"]["map_id"], "house_v1")

        map_after = client.execute_tool(
            "get_map",
            ToolCallContext(
                goal=GoalContext("Navigate to the kitchen"),
                subgoal=Subgoal(text="go to the kitchen", kind="navigate", target="kitchen"),
                world_state=world_state,
                payload={"summary_only": True, "target": "kitchen"},
            ),
        )
        self.assertTrue(map_after["details"]["available"])
        self.assertEqual(map_after["recommended_action_id"], "go_to_pose")
        self.assertTrue(map_after["details"]["map"]["regions"])

        tool_result = client.execute_tool(
            "go_to_pose",
            ToolCallContext(
                goal=GoalContext("Navigate to the kitchen"),
                subgoal=Subgoal(text="go to the kitchen", kind="navigate", target="kitchen"),
                world_state=world_state,
                payload={"pose": "kitchen"},
            ),
        )
        self.assertEqual(tool_result["status"], "succeeded")
        self.assertEqual(tool_result["details"]["task"]["result"]["goal_label"], "kitchen")
        self.assertIsInstance(tool_result["details"]["task"]["result"]["goal_pose"], dict)
        self.assertTrue(tool_result["resolved_subgoal"])

        execution_result = client.execute_skill(
            SkillContract(
                skill_id="navigate_to_region",
                skill_type=SkillType.NAVIGATION,
                language_description="Navigate to a region.",
                executor_binding="xlerobot.navigation.delegated.global_map",
            ),
            GoalContext("Navigate to the kitchen"),
            world_state,
        )
        self.assertEqual(execution_result.status.value, "succeeded")

        snapshot = client.brain_snapshot()
        self.assertEqual(snapshot["brain_name"], "test-brain")
        self.assertEqual(snapshot["last_publish_reason"], "skill:navigate_to_region")

    def test_manual_mapping_review_endpoints(self) -> None:
        client = OffloadClient(self.server_url, brain_name="review-brain")
        client.register()
        client.publish_state_payload({"current_task": "map home", "current_pose": "hallway", "metadata": {}}, reason="review_seed")

        task = client.start_explore(area="downstairs", session="review_map", source="operator")
        self.assertEqual(task["tool_id"], "explore")
        client.pause_mapping_task(task["task_id"])
        paused = client.mapping_snapshot()
        self.assertTrue(paused["active_task"]["paused"])
        client.resume_mapping_task(task["task_id"])

        deadline = time.time() + 5.0
        while time.time() < deadline:
            snapshot = client.mapping_snapshot()
            active_task = snapshot.get("active_task")
            if active_task and active_task["state"] == "succeeded":
                break
            time.sleep(0.05)
        final_snapshot = client.mapping_snapshot()
        current_map = final_snapshot["current_map"]
        self.assertEqual(current_map["map_id"], "review_map")
        region_id = current_map["regions"][0]["region_id"]

        updated_region = client.update_mapping_region(region_id, label="custom_kitchen", purpose="food and appliances")
        self.assertEqual(updated_region["label"], "custom_kitchen")
        self.assertEqual(updated_region["purpose"], "food and appliances")

        named_place = client.set_named_place(
            name="custom_anchor",
            pose={"x": 1.0, "y": 2.0, "yaw": 0.0},
            region_id=region_id,
        )
        self.assertEqual(named_place["name"], "custom_anchor")

        approved = client.approve_mapping_map()
        self.assertTrue(approved["approved"])

    def test_brain_service_forwards_state_to_offload_server(self) -> None:
        bridge = BrainBridge(OffloadClient(self.server_url, brain_name="bridge-brain"))
        brain_service = BrainServiceServer(bridge, host="127.0.0.1", port=0)
        brain_service.start_in_thread()
        try:
            state_payload = {
                "world_state": {
                    "current_task": "Inspect scene",
                    "current_pose": "kitchen",
                    "metadata": {"sensors": {"rgb": {"width": 320}}},
                },
                "sensors": {"rgb": {"width": 320}, "depth": {"height": 240}},
                "reason": "bridge_push",
            }
            request = urllib.request.Request(
                f"http://{brain_service.host}:{brain_service.port}/api/state",
                data=json.dumps(state_payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                forwarded = json.loads(response.read().decode("utf-8"))
            self.assertEqual(forwarded["last_publish_reason"], "bridge_push")

            brains_request = urllib.request.Request(
                f"{self.server_url}/api/brains",
                method="GET",
            )
            with urllib.request.urlopen(brains_request, timeout=10) as response:
                brains = json.loads(response.read().decode("utf-8"))
            self.assertEqual(len(brains["brains"]), 1)
            self.assertEqual(brains["brains"][0]["last_publish_reason"], "bridge_push")
            self.assertIn("depth", brains["brains"][0]["latest_sensors"])
        finally:
            brain_service.shutdown()

    def test_offload_client_perception_tools(self) -> None:
        client = OffloadClient(self.server_url, brain_name="perception-brain")
        client.register()
        world_state = WorldState(
            current_task="Find the fridge",
            current_pose="kitchen",
            visible_objects=frozenset({"fridge"}),
            image_descriptions=("a fridge door is visible near the counter",),
            metadata={
                "sensors": {"rgb": {"width": 640}, "depth": {"height": 480}},
                "object_anchors": {"fridge": {"x": 2.1, "y": 0.3, "z": 0.9, "depth_m": 2.1}},
            },
        )
        client.publish_state(world_state, reason="perception_test")

        context = ToolCallContext(
            goal=GoalContext("Find the fridge"),
            subgoal=Subgoal(text="find the fridge", kind="search", target="fridge"),
            world_state=world_state,
            payload={"target": "fridge"},
        )
        scene = client.execute_tool("perceive_scene", context)
        self.assertEqual(scene["status"], "succeeded")
        self.assertTrue(scene["details"]["annotations"])

        grounded = client.execute_tool("ground_object_3d", context)
        self.assertEqual(grounded["status"], "succeeded")
        self.assertEqual(grounded["details"]["best_match"]["label"], "fridge")

        waypoint = client.execute_tool("set_waypoint_from_object", context)
        self.assertEqual(waypoint["status"], "succeeded")
        self.assertEqual(waypoint["recommended_action_id"], "go_to_pose")

    def test_offload_perception_proxy_uses_rgbd_service(self) -> None:
        perception_service = PerceptionService(PerceptionServiceConfig(host="127.0.0.1", port=0))
        perception_service.start_in_thread()
        offload_server = OffloadServer(
            OffloadServerConfig(
                host="127.0.0.1",
                port=0,
                perception_service_url=f"http://{perception_service.host}:{perception_service.port}",
            )
        )
        offload_server.start_in_thread()
        try:
            client = OffloadClient(
                f"http://{offload_server.host}:{offload_server.port}",
                brain_name="rgbd-proxy-brain",
            )
            client.register()
            world_state = WorldState(
                current_task="Find the fridge handle",
                current_pose="kitchen",
                metadata={
                    "sensors": {
                        "depth": {
                            "data": [
                                [0.0, 0.0, 0.0, 0.0],
                                [0.0, 1.8, 1.9, 0.0],
                                [0.0, 1.85, 1.95, 0.0],
                                [0.0, 0.0, 0.0, 0.0],
                            ],
                            "intrinsics": [
                                [100.0, 0.0, 1.5],
                                [0.0, 100.0, 1.5],
                                [0.0, 0.0, 1.0],
                            ],
                            "pose_mat": [
                                [1.0, 0.0, 0.0, 0.5],
                                [0.0, 1.0, 0.0, 0.1],
                                [0.0, 0.0, 1.0, 0.2],
                                [0.0, 0.0, 0.0, 1.0],
                            ],
                        }
                    },
                    "perception": {
                        "annotations": [
                            {
                                "label": "fridge handle",
                                "confidence": 0.93,
                                "bbox_2d": {"x": 1, "y": 1, "w": 2, "h": 2},
                                "mask_id": "handle_mask",
                            }
                        ]
                    },
                },
            )
            client.publish_state(world_state, reason="rgbd_proxy_test")
            context = ToolCallContext(
                goal=GoalContext("Find the fridge handle"),
                subgoal=Subgoal(text="find the fridge handle", kind="align", target="fridge handle"),
                world_state=world_state,
                payload={"target": "fridge handle"},
            )
            grounding = client.execute_tool("ground_object_3d", context)
            self.assertEqual(grounding["status"], "succeeded")
            centroid = grounding["details"]["best_match"]["centroid_3d"]
            self.assertEqual(centroid["frame"], "map")
            self.assertGreater(centroid["x"], 0.45)
            self.assertGreater(grounding["details"]["best_match"]["depth_m"], 1.7)
        finally:
            offload_server.shutdown()
            perception_service.shutdown()


if __name__ == "__main__":
    unittest.main()
