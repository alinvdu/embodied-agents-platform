# Copyright 2026 Alin Vasile Dumitru
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import subprocess
import time
from typing import Any

IMPORT_ERROR: Exception | None = None
try:
    import rclpy
    from geometry_msgs.msg import PoseStamped, Quaternion
    from rclpy.action.graph import get_action_server_names_and_types_by_node
    from rclpy.action import ActionClient
    from rclpy.node import Node
except Exception as exc:  # pragma: no cover - runtime guard.
    IMPORT_ERROR = exc
    rclpy = None
    PoseStamped = None
    Quaternion = None
    ActionClient = None
    Node = object


def quaternion_from_yaw(yaw: float) -> Quaternion:
    message = Quaternion()
    message.x = 0.0
    message.y = 0.0
    message.z = math.sin(yaw / 2.0)
    message.w = math.cos(yaw / 2.0)
    return message


def _require_runtime_dependencies() -> None:
    if IMPORT_ERROR is not None:
        raise RuntimeError(
            "The Nav2 goal client requires `rclpy`, `geometry_msgs`, and "
            "`nav2_msgs` in a ROS 2 compatible Python environment."
        ) from IMPORT_ERROR


@dataclass(frozen=True)
class GoalRequest:
    x: float
    y: float
    yaw: float
    frame_id: str = "map"
    planner_id: str = ""
    controller_id: str = ""
    behavior_tree: str = ""


class Nav2GoalClient(Node):
    def __init__(
        self,
        *,
        allow_multiple_servers: bool = False,
        server_timeout_s: float = 10.0,
    ) -> None:
        _require_runtime_dependencies()
        super().__init__("xlerobot_nav2_goal_client")
        try:
            from nav2_msgs.action import ComputePathToPose, NavigateToPose
        except ImportError as exc:  # pragma: no cover - runtime guard.
            raise RuntimeError(
                "nav2_msgs is not importable in the current Python environment. "
                "Install `ros-$ROS_DISTRO-navigation2` and use a Python environment "
                "that can see ROS 2 system packages."
            ) from exc

        self._compute_path_action = ComputePathToPose
        self._navigate_action = NavigateToPose
        self.allow_multiple_servers = allow_multiple_servers
        self.server_timeout_s = server_timeout_s
        self.compute_path_client = ActionClient(self, ComputePathToPose, "compute_path_to_pose")
        self.navigate_to_pose_client = ActionClient(self, NavigateToPose, "navigate_to_pose")

    def _fully_qualified_node_name(self, name: str, namespace: str) -> str:
        if not namespace or namespace == "/":
            return f"/{name}"
        return f"{namespace.rstrip('/')}/{name}"

    def _action_servers(self, action_name: str) -> list[str]:
        normalized_action = action_name if action_name.startswith("/") else f"/{action_name}"
        servers: set[str] = set()
        for node_name, namespace in self.get_node_names_and_namespaces():
            try:
                action_servers = get_action_server_names_and_types_by_node(self, node_name, namespace)
            except Exception:
                continue
            for advertised_name, _types in action_servers:
                normalized_advertised = (
                    advertised_name
                    if advertised_name.startswith("/")
                    else f"/{advertised_name}"
                )
                if normalized_advertised == normalized_action:
                    servers.add(self._fully_qualified_node_name(node_name, namespace))
                    break
        return sorted(servers)

    def _ensure_action_server_health(self, action_name: str) -> None:
        servers: list[str] = []
        normalized_action = action_name if action_name.startswith("/") else f"/{action_name}"
        if not self.allow_multiple_servers:
            for _attempt in range(10):
                rclpy.spin_once(self, timeout_sec=0.1)
                servers = self._action_servers(action_name)
                if len(servers) > 1:
                    break
                time.sleep(0.05)
            if len(servers) <= 1:
                servers = self._action_servers_via_cli(normalized_action) or servers
        if len(servers) <= 1 or self.allow_multiple_servers:
            return
        joined = ", ".join(servers)
        raise RuntimeError(
            f"Expected exactly one action server for `{action_name}`, found {len(servers)}: {joined}. "
            "Stop duplicate Nav2 launches and retry."
        )

    def _action_servers_via_cli(self, action_name: str) -> list[str] | None:
        try:
            completed = subprocess.run(
                ["ros2", "action", "info", action_name],
                check=False,
                capture_output=True,
                text=True,
                timeout=5.0,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if completed.returncode != 0:
            return None

        servers: list[str] = []
        in_action_servers = False
        for raw_line in completed.stdout.splitlines():
            line = raw_line.rstrip()
            if line.startswith("Action servers:"):
                in_action_servers = True
                continue
            if line.startswith("Action clients:"):
                in_action_servers = False
                continue
            if not in_action_servers:
                continue
            stripped = line.strip()
            if stripped:
                servers.append(stripped)
        return servers

    def _build_pose(self, goal: GoalRequest) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = goal.frame_id
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = goal.x
        pose.pose.position.y = goal.y
        pose.pose.position.z = 0.0
        pose.pose.orientation = quaternion_from_yaw(goal.yaw)
        return pose

    def compute_path(self, goal: GoalRequest) -> Any:
        if not self.compute_path_client.wait_for_server(timeout_sec=self.server_timeout_s):
            raise RuntimeError(
                "`compute_path_to_pose` action server did not appear within "
                f"{self.server_timeout_s:.1f} seconds."
            )
        self._ensure_action_server_health("compute_path_to_pose")

        request = self._compute_path_action.Goal()
        request.goal = self._build_pose(goal)
        if hasattr(request, "planner_id"):
            request.planner_id = goal.planner_id
        future = self.compute_path_client.send_goal_async(request)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError("Nav2 rejected the ComputePathToPose goal.")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        return result_future.result()

    def navigate_to_pose(self, goal: GoalRequest) -> Any:
        if not self.navigate_to_pose_client.wait_for_server(timeout_sec=self.server_timeout_s):
            raise RuntimeError(
                "`navigate_to_pose` action server did not appear within "
                f"{self.server_timeout_s:.1f} seconds."
            )
        self._ensure_action_server_health("navigate_to_pose")

        request = self._navigate_action.Goal()
        request.pose = self._build_pose(goal)
        if hasattr(request, "behavior_tree") and goal.behavior_tree:
            request.behavior_tree = goal.behavior_tree
        future = self.navigate_to_pose_client.send_goal_async(request)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError("Nav2 rejected the NavigateToPose goal.")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        return result_future.result()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Send real Nav2 `ComputePathToPose` and `NavigateToPose` goals against "
            "the ManiSkill ROS bridge."
        )
    )
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--frame-id", default="map")
    parser.add_argument("--planner-id", default="")
    parser.add_argument("--behavior-tree", default="")
    parser.add_argument("--compute-path-first", action="store_true")
    parser.add_argument("--server-timeout-s", type=float, default=10.0)
    parser.add_argument("--allow-multiple-servers", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _require_runtime_dependencies()
    rclpy.init()
    node = Nav2GoalClient(
        allow_multiple_servers=args.allow_multiple_servers,
        server_timeout_s=args.server_timeout_s,
    )
    goal = GoalRequest(
        x=args.x,
        y=args.y,
        yaw=args.yaw,
        frame_id=args.frame_id,
        planner_id=args.planner_id,
        behavior_tree=args.behavior_tree,
    )
    try:
        if args.compute_path_first:
            path_result = node.compute_path(goal)
            path = getattr(getattr(path_result, "result", None), "path", None)
            path_len = len(getattr(path, "poses", [])) if path is not None else 0
            node.get_logger().info(f"ComputePathToPose returned {path_len} poses.")
            if path_len == 0:
                node.get_logger().warning(
                    "ComputePathToPose returned 0 poses. The planner does not yet have a valid path. "
                    "Let SLAM build more map coverage, confirm the goal is inside mapped free space, "
                    "and make sure only one Nav2 stack is running."
                )

        nav_result = node.navigate_to_pose(goal)
        status = getattr(nav_result, "status", None)
        node.get_logger().info(f"NavigateToPose finished with status={status}")
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
