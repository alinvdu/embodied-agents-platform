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
import json
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from xlerobot_agent.exploration import Pose2D
from xlerobot_playground.ros_nav2_runtime import (
    ActionClient,
    Buffer,
    ComputePathToPose,
    ConnectivityException,
    DurabilityPolicy,
    ExtrapolationException,
    GoalStatus,
    LaserScan,
    LookupException,
    Node,
    OccupancyGrid,
    PoseStamped,
    QoSProfile,
    ReliabilityPolicy,
    RosOccupancyMap,
    RosTime,
    TransformBroadcaster,
    TransformListener,
    TransformStamped,
    quaternion_from_yaw,
    require_runtime_dependencies,
    rclpy,
    ros_goal_status_label,
    yaw_from_quaternion_xyzw,
)

try:
    from rosgraph_msgs.msg import Clock
except Exception:  # pragma: no cover - ROS is optional for non-runtime imports.
    Clock = None


@dataclass(frozen=True)
class RosNav2RouterConfig:
    map_topic: str = "/map"
    scan_topic: str = "/scan"
    map_frame: str = "map"
    odom_frame: str = "odom"
    base_frame: str = "base_link"
    server_timeout_s: float = 10.0
    ready_timeout_s: float = 20.0
    allow_multiple_action_servers: bool = False
    publish_clock: bool = True
    publish_external_state_tf: bool = True
    fake_free_map: bool = False
    fake_map_size_m: float = 2.0
    fake_map_resolution_m: float = 0.02


def serialize_pose(pose: Pose2D | None) -> dict[str, Any] | None:
    return None if pose is None else pose.to_dict()


def build_fake_free_map(*, size_m: float, resolution_m: float) -> RosOccupancyMap:
    if size_m <= 0.0:
        raise ValueError("fake map size must be positive")
    if resolution_m <= 0.0:
        raise ValueError("fake map resolution must be positive")
    cells = max(1, int(round(size_m / resolution_m)))
    return RosOccupancyMap(
        resolution=float(resolution_m),
        width=cells,
        height=cells,
        origin_x=-0.5 * cells * resolution_m,
        origin_y=-0.5 * cells * resolution_m,
        data=tuple(0 for _ in range(cells * cells)),
    )


def pose_from_payload(payload: Any) -> Pose2D | None:
    if not isinstance(payload, dict):
        return None
    try:
        return Pose2D(
            float(payload.get("x", 0.0)),
            float(payload.get("y", 0.0)),
            float(payload.get("yaw", 0.0)),
        )
    except Exception:
        return None


def serialize_map(occupancy_map: RosOccupancyMap | None) -> dict[str, Any] | None:
    if occupancy_map is None:
        return None
    return {
        "resolution": float(occupancy_map.resolution),
        "width": int(occupancy_map.width),
        "height": int(occupancy_map.height),
        "origin_x": float(occupancy_map.origin_x),
        "origin_y": float(occupancy_map.origin_y),
        "data": [int(item) for item in occupancy_map.data],
    }


def map_from_payload(payload: Any) -> RosOccupancyMap | None:
    if not isinstance(payload, dict):
        return None
    try:
        return RosOccupancyMap(
            resolution=float(payload["resolution"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
            origin_x=float(payload["origin_x"]),
            origin_y=float(payload["origin_y"]),
            data=tuple(int(item) for item in payload.get("data", [])),
        )
    except Exception:
        return None


def serialize_scan_observation(observation: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(observation, dict):
        return None
    pose = observation.get("pose")
    return {
        "frame_id": str(observation.get("frame_id", "")),
        "reference_frame": str(observation.get("reference_frame", "")),
        "pose": serialize_pose(pose if isinstance(pose, Pose2D) else None),
        "range_min": float(observation.get("range_min", 0.05) or 0.05),
        "range_max": float(observation.get("range_max", 0.0) or 0.0),
        "angle_min": float(observation.get("angle_min", 0.0) or 0.0),
        "angle_increment": float(observation.get("angle_increment", 0.0) or 0.0),
        "ranges": [float(item) for item in observation.get("ranges", ())],
    }


def scan_observation_from_payload(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    pose = pose_from_payload(payload.get("pose"))
    ranges = payload.get("ranges", [])
    if pose is None or not isinstance(ranges, list):
        return None
    try:
        return {
            "frame_id": str(payload.get("frame_id", "")),
            "reference_frame": str(payload.get("reference_frame", "")),
            "pose": pose,
            "range_min": float(payload.get("range_min", 0.05) or 0.05),
            "range_max": float(payload.get("range_max", 0.0) or 0.0),
            "angle_min": float(payload.get("angle_min", 0.0) or 0.0),
            "angle_increment": float(payload.get("angle_increment", 0.0) or 0.0),
            "ranges": tuple(float(item) for item in ranges),
        }
    except Exception:
        return None


class RosNav2RouterNode(Node):
    def __init__(self, config: RosNav2RouterConfig) -> None:
        require_runtime_dependencies()
        if config.publish_clock and Clock is None:
            raise RuntimeError("rosgraph_msgs.msg.Clock is required when router clock publishing is enabled.")
        super().__init__("xlerobot_ros_nav2_router")
        self.config = config
        self._clock_start_wall_s = time.monotonic()
        self._ros_lock = threading.RLock()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)
        self.tf_broadcaster = TransformBroadcaster(self)
        map_qos = QoSProfile(depth=1)
        map_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        map_qos.reliability = ReliabilityPolicy.RELIABLE
        self._map_pub = self.create_publisher(OccupancyGrid, config.map_topic, map_qos)
        self._scan_pub = self.create_publisher(LaserScan, config.scan_topic, 10)
        self._clock_pub = self.create_publisher(Clock, "/clock", 10) if config.publish_clock else None
        self._compute_path_client = ActionClient(self, ComputePathToPose, "compute_path_to_pose")
        self.latest_map: RosOccupancyMap | None = (
            build_fake_free_map(size_m=config.fake_map_size_m, resolution_m=config.fake_map_resolution_m)
            if config.fake_free_map
            else None
        )
        self.latest_pose: Pose2D = Pose2D(0.0, 0.0, 0.0)
        self.received_external_state = False
        self.latest_scan_observation: dict[str, Any] | None = None
        self.latest_scan_stats: dict[str, Any] | None = None
        self.latest_image_data_url: str | None = None
        self.create_timer(0.1, self._publish_state)

    def spin_until_ready(self, *, timeout_s: float | None = None) -> None:
        deadline = time.time() + (timeout_s if timeout_s is not None else self.config.ready_timeout_s)
        while time.time() < deadline:
            with self._ros_lock:
                rclpy.spin_once(self, timeout_sec=0.1)
            if self.received_external_state:
                return
        raise RuntimeError("Timed out waiting for external brain state to initialize the router pose.")

    def current_pose(self) -> Pose2D:
        return Pose2D(self.latest_pose.x, self.latest_pose.y, self.latest_pose.yaw)

    def current_pose_for_robot(self) -> Pose2D:
        try:
            transform = self.tf_buffer.lookup_transform(self.config.odom_frame, self.config.base_frame, RosTime())
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            return Pose2D(
                float(translation.x),
                float(translation.y),
                yaw_from_quaternion_xyzw(rotation.x, rotation.y, rotation.z, rotation.w),
            )
        except (LookupException, ConnectivityException, ExtrapolationException):
            pass
        return self.current_pose()

    def current_pose_in_frame(self, frame_id: str) -> Pose2D | None:
        if frame_id == self.config.map_frame or frame_id == self.config.odom_frame:
            return self.current_pose()
        if frame_id == self.config.base_frame:
            return Pose2D(0.0, 0.0, 0.0)
        try:
            transform = self.tf_buffer.lookup_transform(frame_id, self.config.base_frame, RosTime())
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return Pose2D(
            float(translation.x),
            float(translation.y),
            yaw_from_quaternion_xyzw(rotation.x, rotation.y, rotation.z, rotation.w),
        )

    def _spin_future_until_complete(self, future: Any, *, timeout_s: float, label: str) -> None:
        deadline = time.monotonic() + timeout_s
        while not future.done():
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0.0:
                raise RuntimeError(f"Timed out waiting for Nav2 {label} after {timeout_s:.1f}s.")
            rclpy.spin_once(self, timeout_sec=min(0.05, remaining_s))

    def update_state(
        self,
        *,
        occupancy_map: RosOccupancyMap | None = None,
        pose: Pose2D,
        scan_observation: dict[str, Any] | None = None,
        image_data_url: str | None = None,
    ) -> None:
        if occupancy_map is not None:
            self.latest_map = occupancy_map
        self.latest_pose = pose
        self.received_external_state = True
        self.latest_image_data_url = image_data_url or self.latest_image_data_url
        if scan_observation is not None:
            self.latest_scan_observation = scan_observation
            ranges = list(scan_observation.get("ranges", ()))
            finite_count = sum(1 for item in ranges if isinstance(item, (int, float)))
            self.latest_scan_stats = {
                "frame_id": str(scan_observation.get("frame_id", self.config.base_frame)),
                "beam_count": len(ranges),
                "finite_beam_count": finite_count,
                "valid_beam_count": finite_count,
                "range_min": round(float(scan_observation.get("range_min", 0.05) or 0.05), 3),
                "range_max": round(float(scan_observation.get("range_max", 0.0) or 0.0), 3),
                "angle_min": round(float(scan_observation.get("angle_min", 0.0) or 0.0), 3),
                "angle_increment": round(float(scan_observation.get("angle_increment", 0.0) or 0.0), 6),
            }
        self._publish_state()

    def compute_path(self, *, goal_pose: Pose2D, planner_id: str = "") -> tuple[int, list[Pose2D], str]:
        with self._ros_lock:
            if not self._compute_path_client.wait_for_server(timeout_sec=self.config.server_timeout_s):
                raise RuntimeError("`compute_path_to_pose` action server did not appear in time.")
            request = ComputePathToPose.Goal()
            request.goal = self._build_pose(goal_pose)
            if planner_id:
                request.planner_id = planner_id
            request.use_start = False
            future = self._compute_path_client.send_goal_async(request)
            self._spin_future_until_complete(future, timeout_s=self.config.server_timeout_s, label="goal response")
            goal_handle = future.result()
            if goal_handle is None or not goal_handle.accepted:
                raise RuntimeError("Nav2 rejected the ComputePathToPose goal.")
            result_future = goal_handle.get_result_async()
            self._spin_future_until_complete(
                result_future,
                timeout_s=max(self.config.server_timeout_s, 30.0),
                label="ComputePathToPose result",
            )
            outcome = result_future.result()
        path = getattr(getattr(outcome, "result", None), "path", None)
        poses: list[Pose2D] = []
        if path is not None:
            for pose_stamped in getattr(path, "poses", []):
                pose = pose_stamped.pose
                poses.append(
                    Pose2D(
                        float(pose.position.x),
                        float(pose.position.y),
                        yaw_from_quaternion_xyzw(
                            pose.orientation.x,
                            pose.orientation.y,
                            pose.orientation.z,
                            pose.orientation.w,
                        ),
                    )
                )
        status = int(getattr(outcome, "status", GoalStatus.STATUS_UNKNOWN))
        return status, poses, ros_goal_status_label(status)

    def snapshot(self, *, include_map: bool = True) -> dict[str, Any]:
        return {
            "module": "ros_nav2_router",
            "latest_map": serialize_map(self.latest_map) if include_map else None,
            "latest_map_summary": self._map_summary(),
            "latest_pose": serialize_pose(self.latest_pose),
            "current_pose": serialize_pose(self.current_pose_for_robot()),
            "received_external_state": self.received_external_state,
            "latest_scan": self.latest_scan_stats,
            "latest_image_data_url": self.latest_image_data_url if include_map else None,
        }

    def close(self) -> None:
        self.destroy_node()

    def _publish_state(self) -> None:
        with self._ros_lock:
            self._publish_clock()
            if self.latest_map is not None:
                self._map_pub.publish(self._occupancy_grid_message(self.latest_map))
            self._publish_transforms()
            if self.latest_scan_observation is not None:
                self._scan_pub.publish(self._laser_scan_message(self.latest_scan_observation))

    def _map_summary(self) -> dict[str, Any] | None:
        if self.latest_map is None:
            return None
        data = self.latest_map.data
        return {
            "resolution": float(self.latest_map.resolution),
            "width": int(self.latest_map.width),
            "height": int(self.latest_map.height),
            "origin_x": float(self.latest_map.origin_x),
            "origin_y": float(self.latest_map.origin_y),
            "free_cells": sum(1 for item in data if item == 0),
            "occupied_cells": sum(1 for item in data if item > 0),
            "unknown_cells": sum(1 for item in data if item < 0),
        }

    def _ros_now_msg(self):
        seconds_float = 1.0 + max(0.0, time.monotonic() - self._clock_start_wall_s)
        seconds = int(seconds_float)
        nanoseconds = int((seconds_float - seconds) * 1e9)
        msg = Clock()
        msg.clock.sec = seconds
        msg.clock.nanosec = nanoseconds
        return msg.clock

    def _publish_clock(self) -> None:
        if self._clock_pub is None:
            return
        msg = Clock()
        msg.clock = self._ros_now_msg()
        self._clock_pub.publish(msg)

    def _publish_transforms(self) -> None:
        if not self.config.publish_external_state_tf and not self.config.fake_free_map:
            return
        map_to_odom = TransformStamped()
        map_to_odom.header.stamp = self._ros_now_msg()
        map_to_odom.header.frame_id = self.config.map_frame
        map_to_odom.child_frame_id = self.config.odom_frame
        map_to_odom.transform.rotation = quaternion_from_yaw(0.0)
        self.tf_broadcaster.sendTransform(map_to_odom)
        if not self.config.publish_external_state_tf:
            return
        odom_to_base = TransformStamped()
        odom_to_base.header.stamp = self._ros_now_msg()
        odom_to_base.header.frame_id = self.config.odom_frame
        odom_to_base.child_frame_id = self.config.base_frame
        odom_to_base.transform.translation.x = float(self.latest_pose.x)
        odom_to_base.transform.translation.y = float(self.latest_pose.y)
        odom_to_base.transform.translation.z = 0.0
        odom_to_base.transform.rotation = quaternion_from_yaw(float(self.latest_pose.yaw))
        self.tf_broadcaster.sendTransform(odom_to_base)

    def _occupancy_grid_message(self, occupancy_map: RosOccupancyMap) -> OccupancyGrid:
        message = OccupancyGrid()
        message.header.stamp = self._ros_now_msg()
        message.header.frame_id = self.config.map_frame
        message.info.map_load_time = message.header.stamp
        message.info.resolution = float(occupancy_map.resolution)
        message.info.width = int(occupancy_map.width)
        message.info.height = int(occupancy_map.height)
        message.info.origin.position.x = float(occupancy_map.origin_x)
        message.info.origin.position.y = float(occupancy_map.origin_y)
        message.info.origin.position.z = 0.0
        message.info.origin.orientation = quaternion_from_yaw(0.0)
        message.data = [int(item) for item in occupancy_map.data]
        return message

    def _laser_scan_message(self, observation: dict[str, Any]) -> LaserScan:
        message = LaserScan()
        message.header.stamp = self._ros_now_msg()
        message.header.frame_id = self.config.base_frame
        message.angle_min = float(observation.get("angle_min", 0.0) or 0.0)
        message.angle_increment = float(observation.get("angle_increment", 0.0) or 0.0)
        ranges = [float(item) for item in observation.get("ranges", ())]
        if ranges:
            message.angle_max = message.angle_min + message.angle_increment * max(len(ranges) - 1, 0)
        else:
            message.angle_max = message.angle_min
        message.range_min = float(observation.get("range_min", 0.05) or 0.05)
        message.range_max = float(observation.get("range_max", 0.0) or 0.0)
        message.ranges = ranges
        return message

    def _build_pose(self, goal: Pose2D) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = self.config.map_frame
        pose.header.stamp = self._ros_now_msg()
        pose.pose.position.x = goal.x
        pose.pose.position.y = goal.y
        pose.pose.position.z = 0.0
        pose.pose.orientation = quaternion_from_yaw(goal.yaw)
        return pose


class RemoteNav2RouterClient:
    def __init__(self, base_url: str, *, timeout_s: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def update_state(
        self,
        *,
        occupancy_map: RosOccupancyMap | None = None,
        pose: Pose2D,
        scan_observation: dict[str, Any] | None = None,
        image_data_url: str | None = None,
    ) -> dict[str, Any]:
        return self._request_json(
            "/api/router/update_state",
            {
                "occupancy_map": serialize_map(occupancy_map),
                "pose": pose.to_dict(),
                "scan_observation": serialize_scan_observation(scan_observation),
                "image_data_url": image_data_url,
            },
            timeout_s=min(self.timeout_s, 5.0),
        )

    def compute_path(self, *, goal_pose: Pose2D, planner_id: str = "") -> tuple[int, list[Pose2D], str]:
        payload = self._request_json(
            "/api/router/compute_path",
            {"goal_pose": goal_pose.to_dict(), "planner_id": planner_id},
            timeout_s=max(self.timeout_s, 120.0),
        )
        poses = [pose_from_payload(item) for item in payload.get("path_poses", [])]
        return (
            int(payload.get("status", GoalStatus.STATUS_UNKNOWN)),
            [item for item in poses if item is not None],
            str(payload.get("status_label", "unknown")),
        )

    def snapshot(self) -> dict[str, Any]:
        return self._request_json("/api/state", None, method="GET")

    def _request_json(
        self,
        path: str,
        payload: dict[str, Any] | None,
        *,
        method: str = "POST",
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = None if payload is None or method == "GET" else json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"} if data is not None else {}
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout_s or self.timeout_s) as response:
                return json.loads(response.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        except TimeoutError as exc:
            raise RuntimeError(f"HTTP request to {url} timed out") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"HTTP request to {url} failed: {exc}") from exc


class RosNav2RouterServer:
    def __init__(self, config: RosNav2RouterConfig, *, host: str = "127.0.0.1", port: int = 8891) -> None:
        require_runtime_dependencies()
        self._owns_rclpy = False
        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_rclpy = True
        self.node = RosNav2RouterNode(config)
        self._lock = threading.RLock()
        self._server = ThreadingHTTPServer((host, port), self._build_handler())
        self.host, self.port = self._server.server_address
        self._spin_thread = threading.Thread(target=self._spin_loop, name="ros_nav2_router_spin", daemon=True)
        self._spin_thread.start()

    def _spin_loop(self) -> None:
        while True:
            try:
                if self.node._ros_lock.acquire(timeout=0.01):
                    try:
                        rclpy.spin_once(self.node, timeout_sec=0.01)
                    finally:
                        self.node._ros_lock.release()
                time.sleep(0.005)
            except Exception:
                return

    def _build_handler(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path.rstrip("/") == "/api/health":
                    self._send_json({"status": "ok"})
                    return
                if self.path.rstrip("/") == "/api/state":
                    self._send_json(outer.node.snapshot())
                    return
                if self.path.rstrip("/") == "/api/router/current_pose":
                    self._send_json({"pose": serialize_pose(outer.node.current_pose_for_robot())})
                    return
                self.send_error(HTTPStatus.NOT_FOUND)

            def do_POST(self) -> None:
                try:
                    path = self.path.rstrip("/")
                    payload = self._read_json()
                    if path == "/api/router/update_state":
                        with outer._lock:
                            occupancy_map = map_from_payload(payload.get("occupancy_map"))
                            pose = pose_from_payload(payload.get("pose"))
                            if pose is None:
                                raise ValueError("pose is required")
                            outer.node.update_state(
                                occupancy_map=occupancy_map,
                                pose=pose,
                                scan_observation=scan_observation_from_payload(payload.get("scan_observation")),
                                image_data_url=payload.get("image_data_url"),
                            )
                            self._send_json({"status": "ok", "state": outer.node.snapshot(include_map=False)})
                            return
                    if path in {"/api/router/compute_path", "/api/router/compute_path_to_pose"}:
                        goal_pose = pose_from_payload(payload.get("goal_pose"))
                        if goal_pose is None:
                            raise ValueError("goal_pose is required")
                        planner_id = str(payload.get("planner_id", ""))
                    else:
                        self.send_error(HTTPStatus.NOT_FOUND)
                        return
                    status, path_poses, status_label = outer.node.compute_path(
                        goal_pose=goal_pose,
                        planner_id=planner_id,
                    )
                    self._send_json(
                        {
                            "status": status,
                            "status_label": status_label,
                            "path_poses": [pose.to_dict() for pose in path_poses],
                            "state": outer.node.snapshot(include_map=False),
                        }
                    )
                    return
                except Exception as exc:
                    self._send_json(
                        {
                            "status": "error",
                            "error": str(exc),
                            "state": outer.node.snapshot(include_map=False),
                        },
                        status=HTTPStatus.BAD_GATEWAY,
                    )

            def _read_json(self) -> dict[str, Any]:
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length <= 0:
                    return {}
                return json.loads(self.rfile.read(content_length).decode("utf-8"))

            def _send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:
                return

        return Handler

    def serve_forever(self) -> None:
        self._server.serve_forever()

    def shutdown(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self.node.close()
        if self._owns_rclpy and rclpy.ok():
            rclpy.shutdown()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HTTP wrapper around ROS/Nav2 ComputePathToPose for XLeRobot.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8891)
    parser.add_argument("--map-topic", default="/map")
    parser.add_argument("--scan-topic", default="/scan")
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--odom-frame", default="odom")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--server-timeout-s", type=float, default=10.0)
    parser.add_argument("--ready-timeout-s", type=float, default=20.0)
    parser.add_argument("--allow-multiple-action-servers", action="store_true")
    parser.add_argument("--publish-clock", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--publish-external-state-tf",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use only for simulator/external-state mode. Real robot mode should leave this disabled.",
    )
    parser.add_argument(
        "--fake-free-map",
        action="store_true",
        help=(
            "Publish a small all-free /map and identity map->odom TF from this router. "
            "Useful for tiny real-robot smoke tests before SLAM is stable."
        ),
    )
    parser.add_argument("--fake-map-size-m", type=float, default=2.0)
    parser.add_argument("--fake-map-resolution-m", type=float, default=0.02)
    return parser


def config_from_args(args: argparse.Namespace) -> RosNav2RouterConfig:
    return RosNav2RouterConfig(
        map_topic=args.map_topic,
        scan_topic=args.scan_topic,
        map_frame=args.map_frame,
        odom_frame=args.odom_frame,
        base_frame=args.base_frame,
        server_timeout_s=args.server_timeout_s,
        ready_timeout_s=args.ready_timeout_s,
        allow_multiple_action_servers=args.allow_multiple_action_servers,
        publish_clock=args.publish_clock,
        publish_external_state_tf=args.publish_external_state_tf,
        fake_free_map=args.fake_free_map,
        fake_map_size_m=args.fake_map_size_m,
        fake_map_resolution_m=args.fake_map_resolution_m,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    server = RosNav2RouterServer(config_from_args(args), host=args.host, port=args.port)
    print(f"ROS Nav2 router ready: http://{server.host}:{server.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
