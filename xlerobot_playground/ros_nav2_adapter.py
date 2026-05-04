from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from xlerobot_agent.exploration import Pose2D
from xlerobot_playground.ros_nav2_runtime import RosExplorationRuntime, RosOccupancyMap, RosRuntimeConfig


def serialize_pose(pose: Pose2D | None) -> dict[str, Any] | None:
    return None if pose is None else pose.to_dict()


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
        data = payload.get("data", [])
        return RosOccupancyMap(
            resolution=float(payload["resolution"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
            origin_x=float(payload["origin_x"]),
            origin_y=float(payload["origin_y"]),
            data=tuple(int(item) for item in data),
        )
    except Exception:
        return None


def serialize_scan_observation(observation: dict[str, Any]) -> dict[str, Any]:
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


class RemoteRosExplorationRuntime:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 30.0,
        turn_scan_mode: str = "camera_pan",
        robot_brain_url: str | None = "http://127.0.0.1:8765",
        camera_pan_action_key: str = "head_motor_1.pos",
        camera_pan_settle_s: float = 0.5,
        camera_pan_step_deg: float = 60.0,
        camera_pan_compute_s: float = 2.0,
        camera_pan_sample_count: int = 12,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.turn_scan_mode = turn_scan_mode
        self.robot_brain_url = robot_brain_url
        self.camera_pan_action_key = camera_pan_action_key
        self.camera_pan_settle_s = camera_pan_settle_s
        self.camera_pan_step_deg = camera_pan_step_deg
        self.camera_pan_compute_s = camera_pan_compute_s
        self.camera_pan_sample_count = camera_pan_sample_count
        self.latest_map: RosOccupancyMap | None = None
        self.latest_map_stamp_s: float = 0.0
        self.latest_scan_stats: dict[str, Any] | None = None
        self.latest_point_cloud_stats: dict[str, Any] | None = None
        self.latest_image_data_url: str | None = None

    def spin_until_ready(self, *, timeout_s: float | None = None) -> None:
        self._request_json("/api/runtime/wait_ready", {"timeout_s": timeout_s})

    def spin_for(self, duration_s: float) -> None:
        self._request_json("/api/runtime/spin_for", {"duration_s": duration_s})

    def current_pose(self) -> Pose2D | None:
        payload = self._request_json("/api/runtime/current_pose", {"frame_id": None})
        return pose_from_payload(payload.get("pose"))

    def current_pose_in_frame(self, frame_id: str) -> Pose2D | None:
        payload = self._request_json("/api/runtime/current_pose", {"frame_id": frame_id})
        return pose_from_payload(payload.get("pose"))

    def compute_path(self, *, goal_pose: Pose2D, planner_id: str = "") -> tuple[int, list[Pose2D], Any]:
        payload = self._request_json(
            "/api/runtime/compute_path",
            {"goal_pose": goal_pose.to_dict(), "planner_id": planner_id},
        )
        poses = [pose_from_payload(item) for item in payload.get("path_poses", [])]
        return int(payload.get("status", 0)), [item for item in poses if item is not None], payload.get("raw_result", {})

    def navigate_to_pose(
        self,
        *,
        goal_pose: Pose2D,
        behavior_tree: str = "",
        should_cancel: Any | None = None,
    ) -> Any:
        del should_cancel
        payload = self._request_json(
            "/api/runtime/navigate_to_pose",
            {"goal_pose": goal_pose.to_dict(), "behavior_tree": behavior_tree},
            timeout_s=max(self.timeout_s, 300.0),
        )
        return payload.get("outcome", {}), list(payload.get("feedback_samples", []))

    def perform_turnaround_scan(
        self,
        *,
        reason: str,
        should_cancel: Any | None = None,
    ) -> dict[str, Any]:
        del should_cancel
        payload = self._request_json(
            "/api/runtime/turnaround_scan",
            {
                "reason": reason,
                "turn_scan_mode": self.turn_scan_mode,
                "robot_brain_url": self.robot_brain_url,
                "camera_pan_action_key": self.camera_pan_action_key,
                "camera_pan_settle_s": self.camera_pan_settle_s,
                "camera_pan_step_deg": self.camera_pan_step_deg,
                "camera_pan_compute_s": self.camera_pan_compute_s,
                "camera_pan_sample_count": self.camera_pan_sample_count,
            },
            timeout_s=max(self.timeout_s, 300.0),
        )
        event = dict(payload.get("event", {}))
        event["observations"] = [
            item
            for item in (
                scan_observation_from_payload(observation)
                for observation in payload.get("observations", [])
            )
            if item is not None
        ]
        event["observation_stop_index"] = int(payload.get("observation_stop_index", 0))
        return event

    def scan_observation_count(self) -> int:
        payload = self._request_json("/api/runtime/scan_count", {}, method="GET")
        return int(payload.get("count", 0))

    def drain_scan_observations(self, since_index: int) -> tuple[list[dict[str, Any]], int]:
        payload = self._request_json("/api/runtime/drain_scan_observations", {"since_index": since_index})
        observations = [
            item
            for item in (
                scan_observation_from_payload(observation)
                for observation in payload.get("observations", [])
            )
            if item is not None
        ]
        return observations, int(payload.get("stop_index", since_index))

    def publish_navigation_map(
        self,
        occupancy_map: RosOccupancyMap,
        *,
        map_to_odom: Pose2D | None = None,
    ) -> None:
        self._request_json(
            "/api/runtime/publish_navigation_map",
            {
                "occupancy_map": serialize_map(occupancy_map),
                "map_to_odom": serialize_pose(map_to_odom),
            },
        )

    def snapshot(self) -> dict[str, Any]:
        return self._request_json("/api/state", {}, method="GET")

    def close(self) -> None:
        return

    def _request_json(
        self,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        method: str = "POST",
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        if method == "GET":
            if payload:
                query = urllib.parse.urlencode(payload)
                url = f"{url}?{query}"
            data = None
            headers: dict[str, str] = {}
        else:
            data = None if payload is None else json.dumps(payload).encode("utf-8")
            headers = {"Content-Type": "application/json"} if payload is not None else {}
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout_s or self.timeout_s) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        result = json.loads(body or "{}")
        self._refresh_cache(result.get("runtime_state"))
        return result

    def _refresh_cache(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        self.latest_map = map_from_payload(payload.get("latest_map"))
        self.latest_map_stamp_s = float(payload.get("latest_map_stamp_s", self.latest_map_stamp_s) or 0.0)
        latest_scan = payload.get("latest_scan")
        self.latest_scan_stats = dict(latest_scan) if isinstance(latest_scan, dict) else None
        latest_point_cloud = payload.get("latest_point_cloud")
        self.latest_point_cloud_stats = dict(latest_point_cloud) if isinstance(latest_point_cloud, dict) else None
        image_data = payload.get("latest_image_data_url")
        self.latest_image_data_url = str(image_data) if isinstance(image_data, str) and image_data else None


class RosNav2AdapterServer:
    def __init__(self, config: RosRuntimeConfig, *, host: str = "127.0.0.1", port: int = 8891) -> None:
        self.runtime = RosExplorationRuntime(config)
        self._lock = threading.RLock()
        self._server = ThreadingHTTPServer((host, port), self._build_handler())
        self.host, self.port = self._server.server_address

    def _build_handler(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                parsed = urllib.parse.urlparse(self.path)
                path = parsed.path.rstrip("/")
                query = urllib.parse.parse_qs(parsed.query)
                if path == "/api/health":
                    self._send_json({"status": "ok"})
                    return
                if path == "/api/state":
                    with outer._lock:
                        outer.runtime.spin_for(0.05)
                        self._send_json(outer._state_payload())
                    return
                if path == "/api/runtime/scan_count":
                    with outer._lock:
                        outer.runtime.spin_for(0.05)
                        self._send_json(
                            {
                                "count": len(outer.runtime.scan_observations),
                                "runtime_state": outer._runtime_state_payload(),
                            }
                        )
                    return
                if path == "/api/runtime/current_pose":
                    frame_id = query.get("frame_id", [None])[0]
                    with outer._lock:
                        outer.runtime.spin_for(0.05)
                        pose = outer.runtime.current_pose() if not frame_id or frame_id == "None" else outer.runtime.current_pose_in_frame(frame_id)
                        self._send_json({"pose": serialize_pose(pose), "runtime_state": outer._runtime_state_payload()})
                    return
                self.send_error(HTTPStatus.NOT_FOUND)

            def do_POST(self) -> None:
                parsed = urllib.parse.urlparse(self.path)
                path = parsed.path.rstrip("/")
                payload = self._read_json()
                with outer._lock:
                    if path == "/api/runtime/wait_ready":
                        outer.runtime.spin_until_ready(timeout_s=payload.get("timeout_s"))
                        self._send_json({"status": "ok", "runtime_state": outer._runtime_state_payload()})
                        return
                    if path == "/api/runtime/spin_for":
                        outer.runtime.spin_for(float(payload.get("duration_s", 0.0) or 0.0))
                        self._send_json({"status": "ok", "runtime_state": outer._runtime_state_payload()})
                        return
                    if path == "/api/runtime/current_pose":
                        frame_id = payload.get("frame_id")
                        pose = outer.runtime.current_pose() if not frame_id else outer.runtime.current_pose_in_frame(str(frame_id))
                        self._send_json({"pose": serialize_pose(pose), "runtime_state": outer._runtime_state_payload()})
                        return
                    if path == "/api/runtime/compute_path":
                        goal_pose = pose_from_payload(payload.get("goal_pose"))
                        if goal_pose is None:
                            raise ValueError("goal_pose is required")
                        status, path_poses, raw_result = outer.runtime.compute_path(
                            goal_pose=goal_pose,
                            planner_id=str(payload.get("planner_id", "")),
                        )
                        self._send_json(
                            {
                                "status": status,
                                "path_poses": [pose.to_dict() for pose in path_poses],
                                "raw_result": {"status": status},
                                "runtime_state": outer._runtime_state_payload(),
                            }
                        )
                        return
                    if path == "/api/runtime/navigate_to_pose":
                        goal_pose = pose_from_payload(payload.get("goal_pose"))
                        if goal_pose is None:
                            raise ValueError("goal_pose is required")
                        outer.runtime.set_point_cloud_map_updates_enabled(False)
                        try:
                            outcome, feedback_samples = outer.runtime.navigate_to_pose(
                                goal_pose=goal_pose,
                                behavior_tree=str(payload.get("behavior_tree", "")),
                            )
                        finally:
                            outer.runtime.set_point_cloud_map_updates_enabled(True)
                        self._send_json(
                            {
                                "outcome": {"status": int(getattr(outcome, "status", 0))},
                                "feedback_samples": feedback_samples,
                                "runtime_state": outer._runtime_state_payload(),
                            }
                        )
                        return
                    if path == "/api/runtime/turnaround_scan":
                        result = outer.runtime.perform_turnaround_scan(
                            reason=str(payload.get("reason", "turnaround_scan")),
                            turn_scan_mode=payload.get("turn_scan_mode"),
                            robot_brain_url=payload.get("robot_brain_url"),
                            camera_pan_action_key=payload.get("camera_pan_action_key"),
                            camera_pan_settle_s=payload.get("camera_pan_settle_s"),
                            camera_pan_step_deg=payload.get("camera_pan_step_deg"),
                            camera_pan_compute_s=payload.get("camera_pan_compute_s"),
                            camera_pan_sample_count=payload.get("camera_pan_sample_count"),
                        )
                        self._send_json(
                            {
                                "event": {key: value for key, value in result.items() if key != "observations"},
                                "observations": [
                                    serialize_scan_observation(observation)
                                    for observation in result.get("observations", [])
                                ],
                                "observation_stop_index": int(result.get("observation_stop_index", len(outer.runtime.scan_observations))),
                                "runtime_state": outer._runtime_state_payload(),
                            }
                        )
                        return
                    if path == "/api/runtime/drain_scan_observations":
                        observations, stop_index = outer.runtime.drain_scan_observations(
                            int(payload.get("since_index", 0) or 0)
                        )
                        self._send_json(
                            {
                                "observations": [serialize_scan_observation(item) for item in observations],
                                "stop_index": stop_index,
                                "runtime_state": outer._runtime_state_payload(),
                            }
                        )
                        return
                    if path == "/api/runtime/publish_navigation_map":
                        occupancy_map = map_from_payload(payload.get("occupancy_map"))
                        if occupancy_map is None:
                            raise ValueError("occupancy_map is required")
                        outer.runtime.publish_navigation_map(
                            occupancy_map,
                            map_to_odom=pose_from_payload(payload.get("map_to_odom")),
                        )
                        self._send_json({"status": "ok", "runtime_state": outer._runtime_state_payload()})
                        return
                self.send_error(HTTPStatus.NOT_FOUND)

            def log_message(self, format: str, *args: Any) -> None:
                return

            def _read_json(self) -> dict[str, Any]:
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length <= 0:
                    return {}
                raw = self.rfile.read(content_length)
                if not raw:
                    return {}
                return json.loads(raw.decode("utf-8"))

            def _send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return Handler

    def _runtime_state_payload(self) -> dict[str, Any]:
        return {
            "latest_map": serialize_map(self.runtime.latest_map),
            "latest_map_stamp_s": float(self.runtime.latest_map_stamp_s),
            "latest_scan": self.runtime.latest_scan_stats,
            "latest_point_cloud": self.runtime.latest_point_cloud_stats,
            "latest_image_data_url": self.runtime.latest_image_data_url,
        }

    def _state_payload(self) -> dict[str, Any]:
        return {
            **self.runtime.snapshot(),
            "runtime_state": self._runtime_state_payload(),
        }

    def serve_forever(self) -> None:
        self._server.serve_forever()

    def start_in_thread(self) -> threading.Thread:
        thread = threading.Thread(target=self.serve_forever, name="ros_nav2_adapter_http", daemon=True)
        thread.start()
        return thread

    def shutdown(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self.runtime.close()
