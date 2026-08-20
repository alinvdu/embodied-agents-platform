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
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import time
from typing import Any
from urllib import request, error
from urllib.parse import urljoin

IMPORT_ERROR: Exception | None = None
try:
    import rclpy
    from action_msgs.msg import GoalStatus
    from geometry_msgs.msg import PoseStamped, Quaternion, Twist
    from nav_msgs.msg import Odometry
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from sensor_msgs.msg import LaserScan
except Exception as exc:  # pragma: no cover - runtime guard.
    IMPORT_ERROR = exc
    rclpy = None
    GoalStatus = None
    PoseStamped = None
    Quaternion = None
    Twist = None
    Odometry = None
    ActionClient = None
    Node = object
    LaserScan = None


@dataclass(frozen=True)
class OdomPose2D:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class SmokeStep:
    name: str
    translation_m: float = 0.0
    yaw_delta_rad: float = 0.0


@dataclass(frozen=True)
class StepResult:
    name: str
    start: OdomPose2D
    goal: OdomPose2D
    end: OdomPose2D
    nav_status: int | None
    nav_status_label: str
    distance_error_m: float
    yaw_error_rad: float
    path_pose_count: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "start": self.start.__dict__,
            "goal": self.goal.__dict__,
            "end": self.end.__dict__,
            "nav_status": self.nav_status,
            "nav_status_label": self.nav_status_label,
            "distance_error_m": round(self.distance_error_m, 4),
            "yaw_error_deg": round(math.degrees(self.yaw_error_rad), 3),
            "path_pose_count": self.path_pose_count,
        }


DEFAULT_STEPS = (
    SmokeStep("forward_10cm", translation_m=0.10),
    SmokeStep("rotate_left_10deg", yaw_delta_rad=math.radians(10.0)),
    SmokeStep("rotate_right_10deg", yaw_delta_rad=math.radians(-10.0)),
)


def require_runtime_dependencies() -> None:
    if IMPORT_ERROR is not None:
        raise RuntimeError(
            "The real Nav2 smoke test requires ROS 2 Python packages: `rclpy`, "
            "`nav2_msgs`, `action_msgs`, `geometry_msgs`, `nav_msgs`, and `sensor_msgs`."
        ) from IMPORT_ERROR


def angle_wrap(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quaternion_xyzw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw: float) -> Any:
    msg = Quaternion()
    msg.x = 0.0
    msg.y = 0.0
    msg.z = math.sin(yaw / 2.0)
    msg.w = math.cos(yaw / 2.0)
    return msg


def odom_pose_from_message(message: Any) -> OdomPose2D:
    pose = message.pose.pose
    q = pose.orientation
    return OdomPose2D(
        x=float(pose.position.x),
        y=float(pose.position.y),
        yaw=yaw_from_quaternion_xyzw(float(q.x), float(q.y), float(q.z), float(q.w)),
    )


def relative_goal(start: OdomPose2D, step: SmokeStep) -> OdomPose2D:
    return OdomPose2D(
        x=start.x + step.translation_m * math.cos(start.yaw),
        y=start.y + step.translation_m * math.sin(start.yaw),
        yaw=angle_wrap(start.yaw + step.yaw_delta_rad),
    )


def distance_error_m(actual: OdomPose2D, goal: OdomPose2D) -> float:
    return math.hypot(actual.x - goal.x, actual.y - goal.y)


def yaw_error_rad(actual: OdomPose2D, goal: OdomPose2D) -> float:
    return abs(angle_wrap(actual.yaw - goal.yaw))


def goal_status_label(status: int | None) -> str:
    if GoalStatus is None:
        return f"status_{status}"
    mapping = {
        GoalStatus.STATUS_UNKNOWN: "unknown",
        GoalStatus.STATUS_ACCEPTED: "accepted",
        GoalStatus.STATUS_EXECUTING: "executing",
        GoalStatus.STATUS_CANCELING: "canceling",
        GoalStatus.STATUS_SUCCEEDED: "succeeded",
        GoalStatus.STATUS_CANCELED: "canceled",
        GoalStatus.STATUS_ABORTED: "aborted",
    }
    return mapping.get(status, f"status_{status}")


class RealNav2SmokeTest(Node):
    def __init__(
        self,
        *,
        frame_id: str,
        odom_topic: str,
        scan_topic: str,
        cmd_vel_topic: str,
        planner_id: str,
        behavior_tree: str,
        server_timeout_s: float,
    ) -> None:
        require_runtime_dependencies()
        super().__init__("xlerobot_real_nav2_smoke_test")
        try:
            from nav2_msgs.action import ComputePathToPose, NavigateToPose
        except ImportError as exc:  # pragma: no cover - runtime guard.
            raise RuntimeError("`nav2_msgs` is not importable in the current ROS 2 Python environment.") from exc

        self.frame_id = frame_id
        self.planner_id = planner_id
        self.behavior_tree = behavior_tree
        self.server_timeout_s = server_timeout_s
        self.compute_path_action = ComputePathToPose
        self.navigate_action = NavigateToPose
        self.compute_path_client = ActionClient(self, ComputePathToPose, "compute_path_to_pose")
        self.navigate_client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.cmd_vel_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.latest_odom: Any | None = None
        self.latest_scan: Any | None = None
        self.create_subscription(Odometry, odom_topic, self._on_odom, 10)
        self.create_subscription(LaserScan, scan_topic, self._on_scan, 10)

    def _on_odom(self, message: Any) -> None:
        self.latest_odom = message

    def _on_scan(self, message: Any) -> None:
        self.latest_scan = message

    def wait_until_ready(self, *, timeout_s: float, require_scan: bool) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.latest_odom is not None and (not require_scan or self.latest_scan is not None):
                return
        missing = []
        if self.latest_odom is None:
            missing.append("odom")
        if require_scan and self.latest_scan is None:
            missing.append("scan")
        raise RuntimeError(f"Timed out waiting for required ROS inputs: {', '.join(missing)}")

    def current_pose(self) -> OdomPose2D:
        if self.latest_odom is None:
            raise RuntimeError("No odometry sample has been received yet.")
        return odom_pose_from_message(self.latest_odom)

    def stop_robot(self) -> None:
        self.cmd_vel_pub.publish(Twist())

    def build_pose_stamped(self, goal: OdomPose2D) -> Any:
        pose = PoseStamped()
        pose.header.frame_id = self.frame_id
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = goal.x
        pose.pose.position.y = goal.y
        pose.pose.position.z = 0.0
        pose.pose.orientation = quaternion_from_yaw(goal.yaw)
        return pose

    def compute_path(self, goal: OdomPose2D) -> int:
        if not self.compute_path_client.wait_for_server(timeout_sec=self.server_timeout_s):
            raise RuntimeError("Nav2 `compute_path_to_pose` action server is not available.")
        request_msg = self.compute_path_action.Goal()
        request_msg.goal = self.build_pose_stamped(goal)
        if hasattr(request_msg, "planner_id"):
            request_msg.planner_id = self.planner_id
        future = self.compute_path_client.send_goal_async(request_msg)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.server_timeout_s)
        if not future.done():
            raise RuntimeError("Timed out waiting for Nav2 to accept the ComputePathToPose goal.")
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError("Nav2 rejected the ComputePathToPose goal.")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=self.server_timeout_s)
        if not result_future.done():
            raise RuntimeError("Timed out waiting for Nav2 ComputePathToPose result.")
        result = result_future.result()
        path = getattr(getattr(result, "result", None), "path", None)
        return len(getattr(path, "poses", [])) if path is not None else 0

    def navigate_to(self, goal: OdomPose2D, *, timeout_s: float) -> int | None:
        if not self.navigate_client.wait_for_server(timeout_sec=self.server_timeout_s):
            raise RuntimeError("Nav2 `navigate_to_pose` action server is not available.")
        request_msg = self.navigate_action.Goal()
        request_msg.pose = self.build_pose_stamped(goal)
        if hasattr(request_msg, "behavior_tree") and self.behavior_tree:
            request_msg.behavior_tree = self.behavior_tree
        future = self.navigate_client.send_goal_async(request_msg)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.server_timeout_s)
        if not future.done():
            raise RuntimeError("Timed out waiting for Nav2 to accept the NavigateToPose goal.")
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError("Nav2 rejected the NavigateToPose goal.")
        result_future = goal_handle.get_result_async()
        start = time.monotonic()
        while not result_future.done():
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.monotonic() - start > timeout_s:
                cancel_future = goal_handle.cancel_goal_async()
                rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=2.0)
                self.stop_robot()
                raise RuntimeError(f"NavigateToPose timed out after {timeout_s:.1f}s and was canceled.")
        result = result_future.result()
        return getattr(result, "status", None)

    def run_steps(
        self,
        steps: tuple[SmokeStep, ...],
        *,
        nav_timeout_s: float,
        settle_s: float,
        require_path: bool,
        max_translation_error_m: float,
        max_yaw_error_rad: float,
    ) -> list[StepResult]:
        results: list[StepResult] = []
        for step in steps:
            start = self.current_pose()
            goal = relative_goal(start, step)
            self.get_logger().info(
                f"{step.name}: start=({start.x:.3f}, {start.y:.3f}, {math.degrees(start.yaw):.1f}deg) "
                f"goal=({goal.x:.3f}, {goal.y:.3f}, {math.degrees(goal.yaw):.1f}deg)"
            )
            path_pose_count = self.compute_path(goal)
            if require_path and path_pose_count <= 0:
                raise RuntimeError(f"{step.name}: Nav2 returned an empty path.")
            status = self.navigate_to(goal, timeout_s=nav_timeout_s)
            time.sleep(settle_s)
            rclpy.spin_once(self, timeout_sec=0.1)
            end = self.current_pose()
            d_error = distance_error_m(end, goal)
            y_error = yaw_error_rad(end, goal)
            result = StepResult(
                name=step.name,
                start=start,
                goal=goal,
                end=end,
                nav_status=status,
                nav_status_label=goal_status_label(status),
                distance_error_m=d_error,
                yaw_error_rad=y_error,
                path_pose_count=path_pose_count,
            )
            results.append(result)
            self.get_logger().info(json.dumps(result.to_dict(), sort_keys=True))
            if status != GoalStatus.STATUS_SUCCEEDED:
                raise RuntimeError(f"{step.name}: Nav2 finished with {result.nav_status_label}.")
            if abs(step.translation_m) > 1e-6 and d_error > max_translation_error_m:
                raise RuntimeError(
                    f"{step.name}: odometry translation error {d_error:.3f}m exceeds "
                    f"{max_translation_error_m:.3f}m."
                )
            if abs(step.yaw_delta_rad) > 1e-6 and y_error > max_yaw_error_rad:
                raise RuntimeError(
                    f"{step.name}: odometry yaw error {math.degrees(y_error):.2f}deg exceeds "
                    f"{math.degrees(max_yaw_error_rad):.2f}deg."
                )
        self.stop_robot()
        return results


def check_robot_brain_health(base_url: str, *, timeout_s: float = 2.0) -> dict[str, Any]:
    with request.urlopen(urljoin(base_url.rstrip("/") + "/", "health"), timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end real XLeRobot smoke test: tiny Nav2 goals -> /cmd_vel -> "
            "ROS bridge -> robot brain, validated with RGB-D odometry on /odom."
        )
    )
    parser.add_argument("--robot-brain-url", default=None, help="Optional robot brain URL for a /health check.")
    parser.add_argument("--frame-id", default="odom", help="Goal frame for tiny relative goals.")
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--scan-topic", default="/scan")
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--planner-id", default="")
    parser.add_argument("--behavior-tree", default="")
    parser.add_argument("--server-timeout-s", type=float, default=10.0)
    parser.add_argument("--ready-timeout-s", type=float, default=20.0)
    parser.add_argument("--nav-timeout-s", type=float, default=20.0)
    parser.add_argument("--settle-s", type=float, default=0.5)
    parser.add_argument("--forward-m", type=float, default=0.10)
    parser.add_argument("--turn-deg", type=float, default=10.0)
    parser.add_argument("--max-translation-error-m", type=float, default=0.06)
    parser.add_argument("--max-yaw-error-deg", type=float, default=6.0)
    parser.add_argument("--no-require-scan", action="store_true")
    parser.add_argument("--allow-empty-path", action="store_true")
    parser.add_argument("--serve", action="store_true", help="Run as an offload HTTP service waiting for robot-brain test triggers.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8895)
    return parser


def steps_from_args(args: argparse.Namespace) -> tuple[SmokeStep, ...]:
    turn = math.radians(args.turn_deg)
    return (
        SmokeStep("forward", translation_m=float(args.forward_m)),
        SmokeStep("rotate_left", yaw_delta_rad=turn),
        SmokeStep("rotate_right", yaw_delta_rad=-turn),
    )


def run_smoke_test(args: argparse.Namespace) -> dict[str, Any]:
    require_runtime_dependencies()
    health = None
    if args.robot_brain_url:
        health = check_robot_brain_health(args.robot_brain_url)
        if not health.get("motion_enabled", False):
            raise RuntimeError("Robot brain reports motion_enabled=false. Restart it with --allow-motion-commands.")

    owns_rclpy = False
    if not rclpy.ok():
        rclpy.init()
        owns_rclpy = True
    node = RealNav2SmokeTest(
        frame_id=args.frame_id,
        odom_topic=args.odom_topic,
        scan_topic=args.scan_topic,
        cmd_vel_topic=args.cmd_vel_topic,
        planner_id=args.planner_id,
        behavior_tree=args.behavior_tree,
        server_timeout_s=args.server_timeout_s,
    )
    try:
        node.wait_until_ready(timeout_s=args.ready_timeout_s, require_scan=not args.no_require_scan)
        results = node.run_steps(
            steps_from_args(args),
            nav_timeout_s=args.nav_timeout_s,
            settle_s=args.settle_s,
            require_path=not args.allow_empty_path,
            max_translation_error_m=args.max_translation_error_m,
            max_yaw_error_rad=math.radians(args.max_yaw_error_deg),
        )
        return {
            "ok": True,
            "robot_brain_health": health,
            "results": [item.to_dict() for item in results],
        }
    finally:
        node.stop_robot()
        node.destroy_node()
        if owns_rclpy and rclpy.ok():
            rclpy.shutdown()


def args_with_overrides(base_args: argparse.Namespace, payload: dict[str, Any]) -> argparse.Namespace:
    values = vars(base_args).copy()
    allowed = {
        "robot_brain_url",
        "frame_id",
        "odom_topic",
        "scan_topic",
        "cmd_vel_topic",
        "planner_id",
        "behavior_tree",
        "server_timeout_s",
        "ready_timeout_s",
        "nav_timeout_s",
        "settle_s",
        "forward_m",
        "turn_deg",
        "max_translation_error_m",
        "max_yaw_error_deg",
        "no_require_scan",
        "allow_empty_path",
    }
    for key, value in payload.items():
        if key in allowed:
            values[key] = value
    return argparse.Namespace(**values)


def serve_smoke_tests(args: argparse.Namespace) -> int:
    require_runtime_dependencies()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path.rstrip("/") == "/health":
                self._send_json({"ok": True, "service": "real_nav2_smoke_test"})
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if self.path.rstrip("/") != "/run":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                payload = self._read_json()
                result = run_smoke_test(args_with_overrides(args, payload))
                self._send_json(result)
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_GATEWAY)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def _send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
            body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Real Nav2 smoke service ready: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.serve:
        return serve_smoke_tests(args)
    result = run_smoke_test(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
