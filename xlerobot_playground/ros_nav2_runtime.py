from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
import json
import math
import subprocess
import threading
import time
from typing import Any, Callable, Iterable
from urllib import error, request

import numpy as np

from xlerobot_agent.exploration import Pose2D

IMPORT_ERROR: Exception | None = None
PIL_IMPORT_ERROR: Exception | None = None
MAP_UPDATE_IMPORT_ERROR: Exception | None = None
try:
    from PIL import Image as PILImage
except Exception as exc:  # pragma: no cover - optional runtime dependency.
    PIL_IMPORT_ERROR = exc
    PILImage = None

try:
    import rclpy
    from action_msgs.msg import GoalStatus
    from builtin_interfaces.msg import Duration as DurationMsg
    from geometry_msgs.msg import PoseStamped, Quaternion, TransformStamped, Twist
    from nav_msgs.msg import OccupancyGrid
    from nav2_msgs.action import ComputePathToPose, NavigateToPose, Spin
    from rclpy.action import ActionClient
    from rclpy.action.graph import get_action_server_names_and_types_by_node
    from rclpy.node import Node
    from rclpy.qos import (
        DurabilityPolicy,
        QoSProfile,
        ReliabilityPolicy,
        qos_profile_sensor_data,
    )
    from rclpy.time import Time as RosTime
    from sensor_msgs.msg import Image, Imu, LaserScan, PointCloud2
    from std_msgs.msg import Bool
    from tf2_ros import Buffer, TransformBroadcaster, TransformListener
    from tf2_ros import ConnectivityException, ExtrapolationException, LookupException
except Exception as exc:  # pragma: no cover - runtime guard.
    IMPORT_ERROR = exc
    rclpy = None
    GoalStatus = None
    DurationMsg = None
    PoseStamped = None
    Quaternion = None
    TransformStamped = None
    Twist = None
    OccupancyGrid = None
    ComputePathToPose = None
    NavigateToPose = None
    Spin = None
    ActionClient = None
    Node = object
    DurabilityPolicy = None
    QoSProfile = None
    ReliabilityPolicy = None
    qos_profile_sensor_data = None
    RosTime = None
    Image = None
    Imu = None
    LaserScan = None
    PointCloud2 = None
    Bool = None
    Buffer = None
    TransformBroadcaster = None
    TransformListener = None
    ConnectivityException = Exception
    ExtrapolationException = Exception
    LookupException = Exception

try:
    from map_msgs.msg import OccupancyGridUpdate
except Exception as exc:  # pragma: no cover - optional ROS runtime dependency.
    MAP_UPDATE_IMPORT_ERROR = exc
    OccupancyGridUpdate = None


def require_runtime_dependencies() -> None:
    if IMPORT_ERROR is not None:
        raise RuntimeError(
            "ROS/Nav2 exploration mode requires `rclpy`, `nav2_msgs`, `sensor_msgs`, "
            "`nav_msgs`, and `tf2_ros` in the active ROS 2 Python environment."
        ) from IMPORT_ERROR


def quaternion_from_yaw(yaw: float) -> Quaternion:
    message = Quaternion()
    message.x = 0.0
    message.y = 0.0
    message.z = math.sin(yaw / 2.0)
    message.w = math.cos(yaw / 2.0)
    return message


def yaw_from_quaternion_xyzw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _quaternion_rotation_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-9:
        return np.eye(3, dtype=np.float32)
    x /= norm
    y /= norm
    z /= norm
    w /= norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _point_cloud2_xyz_array(message: Any) -> np.ndarray:
    fields = {str(field.name): field for field in getattr(message, "fields", [])}
    if not all(name in fields for name in ("x", "y", "z")):
        return np.empty((0, 3), dtype=np.float32)
    point_step = int(getattr(message, "point_step", 0) or 0)
    if point_step <= 0:
        return np.empty((0, 3), dtype=np.float32)
    data = bytes(getattr(message, "data", b""))
    point_count = int(getattr(message, "width", 0) or 0) * int(getattr(message, "height", 0) or 0)
    if point_count <= 0 or len(data) < point_count * point_step:
        return np.empty((0, 3), dtype=np.float32)
    endian = ">" if bool(getattr(message, "is_bigendian", False)) else "<"
    dtype = np.dtype(
        {
            "names": ["x", "y", "z"],
            "formats": [f"{endian}f4", f"{endian}f4", f"{endian}f4"],
            "offsets": [
                int(fields["x"].offset),
                int(fields["y"].offset),
                int(fields["z"].offset),
            ],
            "itemsize": point_step,
        }
    )
    structured = np.frombuffer(data, dtype=dtype, count=point_count)
    return np.column_stack((structured["x"], structured["y"], structured["z"])).astype(np.float32, copy=False)


def ros_goal_status_label(status: int | None) -> str:
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


def remaining_turn_delta_rad(*, desired_total_yaw_rad: float, achieved_total_yaw_rad: float) -> float:
    remaining = max(abs(float(desired_total_yaw_rad)) - abs(float(achieved_total_yaw_rad)), 0.0)
    return math.copysign(remaining, float(desired_total_yaw_rad) if abs(float(desired_total_yaw_rad)) > 1e-9 else 1.0)


def compute_turn_command(
    *,
    requested_angular_rad_s: float,
    target_yaw_rad: float | None,
    feedback_yaw_rad: float | None,
    minimum_command_rad_s: float = 0.12,
    slowdown_zone_rad: float = math.radians(50.0),
    stop_tolerance_rad: float = math.radians(2.0),
) -> tuple[float, bool]:
    command_direction = 1.0 if requested_angular_rad_s >= 0.0 else -1.0
    max_command_speed = abs(float(requested_angular_rad_s))
    if max_command_speed <= 1e-6:
        return 0.0, True
    if target_yaw_rad is None or feedback_yaw_rad is None:
        return command_direction * max_command_speed, False
    remaining_yaw_rad = max(abs(float(target_yaw_rad)) - abs(float(feedback_yaw_rad)), 0.0)
    if remaining_yaw_rad <= max(float(stop_tolerance_rad), 0.0):
        return 0.0, True
    if remaining_yaw_rad < max(float(slowdown_zone_rad), 1e-6):
        scaled_speed = max_command_speed * (remaining_yaw_rad / max(float(slowdown_zone_rad), 1e-6))
        commanded_speed = max(minimum_command_rad_s, scaled_speed)
    else:
        commanded_speed = max_command_speed
    commanded_speed = min(commanded_speed, max_command_speed)
    return command_direction * commanded_speed, False


def _unwrap_yaw_sequence(yaws: Iterable[float]) -> list[float]:
    sequence = list(yaws)
    if not sequence:
        return []
    unwrapped = [float(sequence[0])]
    previous = float(sequence[0])
    for value in sequence[1:]:
        current = float(value)
        delta = math.atan2(math.sin(current - previous), math.cos(current - previous))
        unwrapped.append(unwrapped[-1] + delta)
        previous = current
    return unwrapped


def _select_turnaround_scan_observations(
    observations: list[dict[str, Any]],
    *,
    sample_count: int,
) -> list[dict[str, Any]]:
    if len(observations) <= sample_count:
        return list(observations)
    poses = [item.get("pose") for item in observations]
    if not all(isinstance(pose, Pose2D) for pose in poses):
        stride = max(len(observations) // max(sample_count, 1), 1)
        return observations[::stride][:sample_count]
    yaws = _unwrap_yaw_sequence([float(pose.yaw) for pose in poses if isinstance(pose, Pose2D)])
    if len(yaws) != len(observations):
        stride = max(len(observations) // max(sample_count, 1), 1)
        return observations[::stride][:sample_count]
    span = yaws[-1] - yaws[0]
    if abs(span) < math.radians(45.0):
        stride = max(len(observations) // max(sample_count, 1), 1)
        return observations[::stride][:sample_count]
    direction = 1.0 if span >= 0.0 else -1.0
    usable_span = min(abs(span), math.tau)
    start_yaw = yaws[0]
    if sample_count <= 1:
        targets = [start_yaw]
    else:
        targets = [
            start_yaw + direction * (usable_span * index / (sample_count - 1))
            for index in range(sample_count)
        ]
    chosen_indices: set[int] = set()
    for target in targets:
        best_index = min(
            range(len(yaws)),
            key=lambda index: (abs(yaws[index] - target), abs(index - len(yaws) // 2)),
        )
        chosen_indices.add(best_index)
    if 0 not in chosen_indices:
        chosen_indices.add(0)
    if len(observations) - 1 not in chosen_indices:
        chosen_indices.add(len(observations) - 1)
    ordered = sorted(chosen_indices)
    if len(ordered) > sample_count:
        stride = max(len(ordered) / float(sample_count), 1.0)
        compacted = [ordered[min(int(round(index * stride)), len(ordered) - 1)] for index in range(sample_count)]
        ordered = sorted(set(compacted))
    return [observations[index] for index in ordered]


@dataclass(frozen=True)
class RosRuntimeConfig:
    map_topic: str = "/map"
    map_updates_topic: str | None = None
    scan_topic: str = "/scan"
    point_cloud_topic: str = "/camera/head/points"
    point_cloud_update_map_enabled_topic: str = "/camera/head/points/update_map_enabled"
    scan_active_topic: str = "/xlerobot/scan_active"
    rgb_topic: str = "/camera/head/image_raw"
    imu_topic: str = "/imu/filtered_yaw"
    cmd_vel_topic: str = "/cmd_vel"
    map_frame: str = "map"
    odom_frame: str = "odom"
    base_frame: str = "base_link"
    server_timeout_s: float = 10.0
    ready_timeout_s: float = 20.0
    turn_scan_radians: float = math.tau
    turn_scan_timeout_s: float = 45.0
    turn_scan_settle_s: float = 1.0
    manual_spin_angular_speed_rad_s: float = 0.25
    manual_spin_publish_hz: float = 20.0
    manual_spin_direction_sign: float = -1.0
    turn_scan_mode: str = "camera_pan"
    robot_brain_url: str | None = "http://127.0.0.1:8765"
    camera_pan_action_key: str = "head_motor_1.pos"
    camera_pan_settle_s: float = 0.5
    camera_pan_step_deg: float = 60.0
    camera_pan_compute_s: float = 2.0
    camera_pan_sample_count: int = 12
    allow_multiple_action_servers: bool = False
    publish_internal_navigation_map: bool = True
    navigation_map_source: str = "fused_scan"
    fuse_external_projected_map_snapshots: bool = False
    log_map_summaries: bool = False


@dataclass(frozen=True)
class RosOccupancyMap:
    resolution: float
    width: int
    height: int
    origin_x: float
    origin_y: float
    data: tuple[int, ...]

    def in_bounds(self, cell_x: int, cell_y: int) -> bool:
        return 0 <= cell_x < self.width and 0 <= cell_y < self.height

    def value(self, cell_x: int, cell_y: int) -> int:
        if not self.in_bounds(cell_x, cell_y):
            return 100
        return int(self.data[cell_y * self.width + cell_x])

    def is_unknown(self, cell_x: int, cell_y: int) -> bool:
        return self.value(cell_x, cell_y) < 0

    def is_free(self, cell_x: int, cell_y: int) -> bool:
        return self.value(cell_x, cell_y) == 0

    def is_occupied(self, cell_x: int, cell_y: int) -> bool:
        value = self.value(cell_x, cell_y)
        return value > 50 or value == 100

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        return (
            int(math.floor((x - self.origin_x) / self.resolution)),
            int(math.floor((y - self.origin_y) / self.resolution)),
        )

    def cell_to_pose(self, cell_x: int, cell_y: int, *, yaw: float = 0.0) -> Pose2D:
        return Pose2D(
            self.origin_x + (cell_x + 0.5) * self.resolution,
            self.origin_y + (cell_y + 0.5) * self.resolution,
            yaw,
        )

    def bounds(self) -> dict[str, float]:
        return {
            "min_x": round(self.origin_x, 3),
            "max_x": round(self.origin_x + self.width * self.resolution, 3),
            "min_y": round(self.origin_y, 3),
            "max_y": round(self.origin_y + self.height * self.resolution, 3),
        }


def default_map_updates_topic(map_topic: str) -> str:
    topic = str(map_topic or "/map").rstrip("/")
    if not topic:
        topic = "/map"
    return f"{topic}_updates"


def apply_occupancy_grid_update(
    occupancy_map: RosOccupancyMap,
    *,
    update_x: int,
    update_y: int,
    update_width: int,
    update_height: int,
    update_data: Iterable[int],
) -> RosOccupancyMap:
    width = int(update_width)
    height = int(update_height)
    if width <= 0 or height <= 0:
        return occupancy_map
    patch = tuple(int(item) for item in update_data)
    if len(patch) < width * height:
        return occupancy_map
    data = list(occupancy_map.data)
    for patch_y in range(height):
        dst_y = int(update_y) + patch_y
        if not (0 <= dst_y < int(occupancy_map.height)):
            continue
        for patch_x in range(width):
            dst_x = int(update_x) + patch_x
            if not (0 <= dst_x < int(occupancy_map.width)):
                continue
            src_index = patch_y * width + patch_x
            dst_index = dst_y * int(occupancy_map.width) + dst_x
            data[dst_index] = int(patch[src_index])
    return RosOccupancyMap(
        resolution=float(occupancy_map.resolution),
        width=int(occupancy_map.width),
        height=int(occupancy_map.height),
        origin_x=float(occupancy_map.origin_x),
        origin_y=float(occupancy_map.origin_y),
        data=tuple(data),
    )


def fuse_projected_maps(
    maps: Iterable[RosOccupancyMap],
    *,
    free_weight: float = -0.25,
    occupied_weight: float = 1.0,
    free_threshold: float = -0.5,
    occupied_threshold: float = 0.75,
) -> RosOccupancyMap | None:
    snapshots = [item for item in maps if item is not None and item.width > 0 and item.height > 0]
    if not snapshots:
        return None
    resolution = float(snapshots[0].resolution)
    if resolution <= 0.0:
        return None
    min_x = min(int(math.floor(item.origin_x / resolution)) for item in snapshots)
    min_y = min(int(math.floor(item.origin_y / resolution)) for item in snapshots)
    max_x = max(int(math.floor(item.origin_x / resolution)) + int(item.width) - 1 for item in snapshots)
    max_y = max(int(math.floor(item.origin_y / resolution)) + int(item.height) - 1 for item in snapshots)
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    evidence: dict[tuple[int, int], float] = {}
    for occupancy_map in snapshots:
        map_origin_cell_x = int(math.floor(float(occupancy_map.origin_x) / resolution))
        map_origin_cell_y = int(math.floor(float(occupancy_map.origin_y) / resolution))
        for y in range(int(occupancy_map.height)):
            for x in range(int(occupancy_map.width)):
                value = occupancy_map.value(x, y)
                if value < 0:
                    continue
                cell = (map_origin_cell_x + x, map_origin_cell_y + y)
                if value > 50:
                    evidence[cell] = max(evidence.get(cell, 0.0) + occupied_weight, occupied_weight)
                elif value == 0:
                    evidence[cell] = evidence.get(cell, 0.0) + free_weight
    data = [-1] * (width * height)
    for (cell_x, cell_y), score in evidence.items():
        local_x = cell_x - min_x
        local_y = cell_y - min_y
        if not (0 <= local_x < width and 0 <= local_y < height):
            continue
        if score >= occupied_threshold:
            data[local_y * width + local_x] = 100
        elif score <= free_threshold:
            data[local_y * width + local_x] = 0
    return RosOccupancyMap(
        resolution=resolution,
        width=width,
        height=height,
        origin_x=min_x * resolution,
        origin_y=min_y * resolution,
        data=tuple(data),
    )


class RosExplorationRuntime(Node):
    def __init__(self, config: RosRuntimeConfig) -> None:
        require_runtime_dependencies()
        super().__init__("xlerobot_ros_exploration_runtime")
        self.config = config
        self._spin_lock = threading.RLock()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.latest_map: RosOccupancyMap | None = None
        self.latest_map_stamp_s: float = 0.0
        self.latest_map_header_frame_id: str = ""
        self._last_map_log_s: float = 0.0
        self._last_map_update_log_s: float = 0.0
        self.latest_scan: LaserScan | None = None
        self.latest_scan_stats: dict[str, Any] | None = None
        self.latest_point_cloud_stats: dict[str, Any] | None = None
        self.latest_imu_msg: Imu | None = None
        self._latest_imu_orientation_yaw_rad: float | None = None
        self._latest_imu_orientation_unwrapped_yaw_rad: float | None = None
        self._scan_sensor_yaw_offset_rad: float | None = None
        self._use_turn_feedback_for_scan_pose = False
        self.scan_observations: list[dict[str, Any]] = []
        self.point_cloud_observations: list[dict[str, Any]] = []
        self.latest_image_msg: Image | None = None
        self.latest_image_data_url: str | None = None
        self._nav_goal_history: list[dict[str, Any]] = []
        self._nav_plan_history: list[dict[str, Any]] = []
        self._nav_scan_history: list[dict[str, Any]] = []
        self._cmd_vel_pub = self.create_publisher(Twist, config.cmd_vel_topic, 10)
        self._point_cloud_update_map_enabled_pub = self.create_publisher(
            Bool,
            config.point_cloud_update_map_enabled_topic,
            10,
        )
        self._scan_active_pub = self.create_publisher(Bool, config.scan_active_topic, 10)
        map_qos = QoSProfile(depth=1)
        map_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        map_qos.reliability = ReliabilityPolicy.RELIABLE
        self._map_pub = self.create_publisher(OccupancyGrid, config.map_topic, map_qos)
        self._compute_path_client = ActionClient(self, ComputePathToPose, "compute_path_to_pose")
        self._navigate_to_pose_client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self._spin_client = ActionClient(self, Spin, "spin")
        self.create_subscription(OccupancyGrid, config.map_topic, self._on_map, map_qos)
        self._map_updates_topic = config.map_updates_topic or default_map_updates_topic(config.map_topic)
        if OccupancyGridUpdate is not None:
            map_update_qos = QoSProfile(depth=20)
            map_update_qos.reliability = ReliabilityPolicy.RELIABLE
            self.create_subscription(OccupancyGridUpdate, self._map_updates_topic, self._on_map_update, map_update_qos)
        elif MAP_UPDATE_IMPORT_ERROR is not None:
            self.get_logger().warning(
                f"map_msgs OccupancyGridUpdate is unavailable; `{self._map_updates_topic}` will not be consumed. "
                f"Only full maps from `{config.map_topic}` will update the UI map."
            )
        self.create_subscription(LaserScan, config.scan_topic, self._on_scan, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, config.point_cloud_topic, self._on_point_cloud, qos_profile_sensor_data)
        self.create_subscription(Image, config.rgb_topic, self._on_rgb, qos_profile_sensor_data)
        self.create_subscription(Imu, config.imu_topic, self._on_imu, qos_profile_sensor_data)
        self._published_navigation_map: RosOccupancyMap | None = None
        self._map_to_odom = Pose2D(0.0, 0.0, 0.0)
        self._publish_timer = self.create_timer(0.2, self._publish_internal_navigation_state)

    def _spin_once(self, *, timeout_sec: float) -> None:
        with self._spin_lock:
            rclpy.spin_once(self, timeout_sec=timeout_sec)

    def _spin_until_future_complete(self, future: Any) -> None:
        with self._spin_lock:
            rclpy.spin_until_future_complete(self, future)

    def _on_map(self, message: OccupancyGrid) -> None:
        self.latest_map = RosOccupancyMap(
            resolution=float(message.info.resolution),
            width=int(message.info.width),
            height=int(message.info.height),
            origin_x=float(message.info.origin.position.x),
            origin_y=float(message.info.origin.position.y),
            data=tuple(int(item) for item in message.data),
        )
        self.latest_map_stamp_s = time.time()
        self.latest_map_header_frame_id = str(message.header.frame_id)
        now = time.time()
        if self.config.log_map_summaries and now - self._last_map_log_s >= 2.0:
            self._last_map_log_s = now
            print(f"[ros_nav2_runtime] received map topic={self.config.map_topic} summary={self.latest_map_summary()}")

    def _on_map_update(self, message: Any) -> None:
        if self.latest_map is None:
            return
        self.latest_map = apply_occupancy_grid_update(
            self.latest_map,
            update_x=int(message.x),
            update_y=int(message.y),
            update_width=int(message.width),
            update_height=int(message.height),
            update_data=message.data,
        )
        self.latest_map_stamp_s = time.time()
        header_frame_id = str(getattr(message.header, "frame_id", "") or "")
        if header_frame_id:
            self.latest_map_header_frame_id = header_frame_id
        now = time.time()
        if self.config.log_map_summaries and now - self._last_map_update_log_s >= 2.0:
            self._last_map_update_log_s = now
            print(
                "[ros_nav2_runtime] applied map update "
                f"topic={self._map_updates_topic} "
                f"rect=({int(message.x)},{int(message.y)},{int(message.width)},{int(message.height)}) "
                f"summary={self.latest_map_summary()}"
            )

    def _on_scan(self, message: LaserScan) -> None:
        self.latest_scan = message
        ranges = np.asarray(message.ranges, dtype=np.float32)
        finite = np.isfinite(ranges)
        valid = finite & (ranges >= float(message.range_min)) & (ranges <= float(message.range_max))
        max_like = valid & (ranges >= float(message.range_max) * 0.999)
        self.latest_scan_stats = {
            "frame_id": message.header.frame_id,
            "beam_count": int(ranges.size),
            "valid_beam_count": int(np.count_nonzero(valid)),
            "finite_beam_count": int(np.count_nonzero(finite)),
            "max_range_beam_count": int(np.count_nonzero(max_like)),
            "range_min": round(float(message.range_min), 3),
            "range_max": round(float(message.range_max), 3),
            "angle_min": round(float(message.angle_min), 3),
            "angle_max": round(float(message.angle_max), 3),
            "angle_increment": round(float(message.angle_increment), 6),
        }
        reference_frame = self.config.odom_frame if self.config.publish_internal_navigation_map else self.config.map_frame
        sensor_pose = self.lookup_pose(reference_frame, message.header.frame_id)
        if sensor_pose is not None:
            sensor_pose = self._scan_pose_with_turn_feedback(sensor_pose)
            self.scan_observations.append(
                {
                    "frame_id": str(message.header.frame_id),
                    "pose": sensor_pose,
                    "reference_frame": reference_frame,
                    "range_min": float(message.range_min),
                    "range_max": float(message.range_max),
                    "angle_min": float(message.angle_min),
                    "angle_increment": float(message.angle_increment),
                    "ranges": tuple(float(item) for item in message.ranges),
                }
            )
            if len(self.scan_observations) > 4096:
                self.scan_observations = self.scan_observations[-2048:]

    def _on_point_cloud(self, message: PointCloud2) -> None:
        points = _point_cloud2_xyz_array(message)
        finite = np.isfinite(points).all(axis=1) if points.size else np.zeros((0,), dtype=bool)
        reference_frame = self.config.odom_frame if self.config.publish_internal_navigation_map else self.config.map_frame
        transform = self._lookup_transform_xyz_quat(reference_frame, message.header.frame_id)
        transformed_points = np.empty((0, 3), dtype=np.float32)
        sensor_origin = None
        if transform is not None and points.size:
            translation, quaternion = transform
            rotation = _quaternion_rotation_matrix(*quaternion)
            transformed_points = (points @ rotation.T + translation.reshape(1, 3)).astype(np.float32, copy=False)
            sensor_origin = translation
        elif transform is not None:
            translation, _quaternion = transform
            sensor_origin = translation
        self.latest_point_cloud_stats = {
            "frame_id": message.header.frame_id,
            "point_count": int(points.shape[0]),
            "finite_point_count": int(np.count_nonzero(finite)),
            "width": int(message.width),
            "height": int(message.height),
            "point_step": int(message.point_step),
            "reference_frame": reference_frame,
            "tf_ready": transform is not None,
        }
        if transform is None or sensor_origin is None:
            return
        self.point_cloud_observations.append(
            {
                "frame_id": str(message.header.frame_id),
                "reference_frame": reference_frame,
                "sensor_origin_xyz": tuple(float(item) for item in sensor_origin),
                "points_xyz": transformed_points,
                "point_count": int(transformed_points.shape[0]),
            }
        )
        if len(self.point_cloud_observations) > 1024:
            self.point_cloud_observations = self.point_cloud_observations[-512:]

    def _on_rgb(self, message: Image) -> None:
        self.latest_image_msg = message
        encoded = image_message_to_data_url(message)
        if encoded:
            self.latest_image_data_url = encoded

    def _on_imu(self, message: Imu) -> None:
        self.latest_imu_msg = message
        covariance = getattr(message, "orientation_covariance", None)
        if covariance is None or len(covariance) < 1 or float(covariance[0]) < 0.0:
            return
        orientation = message.orientation
        yaw_rad = yaw_from_quaternion_xyzw(
            float(orientation.x),
            float(orientation.y),
            float(orientation.z),
            float(orientation.w),
        )
        if self._latest_imu_orientation_yaw_rad is None:
            self._latest_imu_orientation_unwrapped_yaw_rad = yaw_rad
        else:
            assert self._latest_imu_orientation_unwrapped_yaw_rad is not None
            self._latest_imu_orientation_unwrapped_yaw_rad += math.atan2(
                math.sin(yaw_rad - self._latest_imu_orientation_yaw_rad),
                math.cos(yaw_rad - self._latest_imu_orientation_yaw_rad),
            )
        self._latest_imu_orientation_yaw_rad = yaw_rad

    def _current_turn_feedback(self) -> tuple[str, float] | tuple[None, None]:
        if self._latest_imu_orientation_unwrapped_yaw_rad is not None:
            return "imu", float(self._latest_imu_orientation_unwrapped_yaw_rad)
        pose = self.current_pose_in_frame(self.config.odom_frame)
        if pose is not None:
            return self.config.odom_frame, float(pose.yaw)
        return None, None

    def _scan_pose_with_turn_feedback(self, sensor_pose: Pose2D) -> Pose2D:
        if not self.config.publish_internal_navigation_map:
            return sensor_pose
        if not self._use_turn_feedback_for_scan_pose:
            return sensor_pose
        _feedback_frame, feedback_yaw = self._current_turn_feedback()
        if feedback_yaw is None:
            return sensor_pose
        if self._scan_sensor_yaw_offset_rad is None:
            self._scan_sensor_yaw_offset_rad = sensor_pose.yaw - feedback_yaw
        return Pose2D(
            float(sensor_pose.x),
            float(sensor_pose.y),
            float(feedback_yaw + self._scan_sensor_yaw_offset_rad),
        )

    def spin_until_ready(self, *, timeout_s: float | None = None) -> None:
        deadline = time.time() + (timeout_s if timeout_s is not None else self.config.ready_timeout_s)
        while time.time() < deadline:
            self._spin_once(timeout_sec=0.1)
            if self.config.publish_internal_navigation_map:
                if self.current_pose_in_frame(self.config.odom_frame) is not None:
                    return
            elif self.latest_map is not None and self.current_pose() is not None:
                return
            time.sleep(0.05)
        raise RuntimeError(
            (
                f"Timed out waiting for `{self.config.odom_frame}->{self.config.base_frame}` pose."
                if self.config.publish_internal_navigation_map
                else f"Timed out waiting for `{self.config.map_topic}` and `{self.config.map_frame}->{self.config.base_frame}` pose."
            )
        )

    def current_pose(self) -> Pose2D | None:
        return self.lookup_pose(self.config.map_frame, self.config.base_frame)

    def current_pose_in_frame(self, frame_id: str) -> Pose2D | None:
        return self.lookup_pose(frame_id, self.config.base_frame)

    def lookup_pose(self, target_frame: str, source_frame: str) -> Pose2D | None:
        transform = self._lookup_transform_xyz_quat(target_frame, source_frame)
        if transform is None:
            return None
        translation, rotation = transform
        return Pose2D(
            float(translation[0]),
            float(translation[1]),
            yaw_from_quaternion_xyzw(rotation[0], rotation[1], rotation[2], rotation[3]),
        )

    def _lookup_transform_xyz_quat(self, target_frame: str, source_frame: str) -> tuple[np.ndarray, tuple[float, float, float, float]] | None:
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                RosTime(),
            )
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return (
            np.asarray([float(translation.x), float(translation.y), float(translation.z)], dtype=np.float32),
            (float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w)),
        )

    def spin_for(self, duration_s: float) -> None:
        deadline = time.time() + duration_s
        while time.time() < deadline:
            self._spin_once(timeout_sec=0.05)

    def wait_for_map_update(self, *, after_stamp_s: float, timeout_s: float = 2.0) -> bool:
        deadline = time.time() + max(float(timeout_s), 0.0)
        while time.time() < deadline:
            self._spin_once(timeout_sec=0.05)
            if self.latest_map is not None and self.latest_map_stamp_s > after_stamp_s:
                return True
        return False

    def latest_map_summary(self) -> dict[str, Any] | None:
        occupancy_map = self.latest_map
        if occupancy_map is None:
            return None
        return self._occupancy_map_summary(
            occupancy_map,
            frame_id=self.latest_map_header_frame_id,
            stamp_s=self.latest_map_stamp_s,
        )

    def _occupancy_map_summary(
        self,
        occupancy_map: RosOccupancyMap,
        *,
        frame_id: str,
        stamp_s: float,
    ) -> dict[str, Any]:
        data = occupancy_map.data
        return {
            "frame_id": frame_id,
            "resolution": round(float(occupancy_map.resolution), 4),
            "width": int(occupancy_map.width),
            "height": int(occupancy_map.height),
            "origin_x": round(float(occupancy_map.origin_x), 3),
            "origin_y": round(float(occupancy_map.origin_y), 3),
            "free_cells": sum(1 for item in data if int(item) == 0),
            "occupied_cells": sum(1 for item in data if int(item) > 50),
            "unknown_cells": sum(1 for item in data if int(item) < 0),
            "stamp_age_s": round(max(time.time() - float(stamp_s), 0.0), 3),
        }

    def hold_stop_until_stable(
        self,
        *,
        duration_s: float,
        yaw_stable_tolerance_rad: float = math.radians(0.6),
        min_stable_cycles: int = 3,
    ) -> dict[str, Any]:
        deadline = time.time() + max(float(duration_s), 0.0)
        _feedback_frame, previous_yaw = self._current_turn_feedback()
        stable_cycles = 0
        observed_yaw_delta = 0.0
        while time.time() < deadline:
            self._cmd_vel_pub.publish(Twist())
            self._spin_once(timeout_sec=0.05)
            feedback_frame, current_yaw = self._current_turn_feedback()
            if current_yaw is None or previous_yaw is None:
                time.sleep(0.05)
                continue
            delta = math.atan2(
                math.sin(current_yaw - previous_yaw),
                math.cos(current_yaw - previous_yaw),
            )
            observed_yaw_delta += delta
            previous_yaw = current_yaw
            if abs(delta) <= yaw_stable_tolerance_rad:
                stable_cycles += 1
                if stable_cycles >= max(int(min_stable_cycles), 1):
                    break
            else:
                stable_cycles = 0
            time.sleep(0.05)
        return {
            "stable": stable_cycles >= max(int(min_stable_cycles), 1),
            "stable_cycles": stable_cycles,
            "observed_yaw_delta_rad": observed_yaw_delta,
        }

    def scan_observation_count(self) -> int:
        return len(self.scan_observations)

    def point_cloud_observation_count(self) -> int:
        return len(self.point_cloud_observations)

    def set_point_cloud_map_updates_enabled(self, enabled: bool) -> None:
        message = Bool()
        message.data = bool(enabled)
        self._point_cloud_update_map_enabled_pub.publish(message)
        self._spin_once(timeout_sec=0.0)

    def set_scan_active(self, active: bool) -> None:
        message = Bool()
        message.data = bool(active)
        self._scan_active_pub.publish(message)
        self._spin_once(timeout_sec=0.0)

    def _set_scan_active_if_available(self, active: bool) -> None:
        self.set_scan_active(active)

    def drain_scan_observations(self, since_index: int) -> tuple[list[dict[str, Any]], int]:
        self.spin_for(0.05)
        stop_index = len(self.scan_observations)
        if since_index < 0:
            since_index = 0
        if since_index >= stop_index:
            return [], stop_index
        return list(self.scan_observations[since_index:stop_index]), stop_index

    def drain_point_cloud_observations(self, since_index: int) -> tuple[list[dict[str, Any]], int]:
        self.spin_for(0.05)
        stop_index = len(self.point_cloud_observations)
        if since_index < 0:
            since_index = 0
        if since_index >= stop_index:
            return [], stop_index
        return list(self.point_cloud_observations[since_index:stop_index]), stop_index

    def compute_path(
        self,
        *,
        goal_pose: Pose2D,
        planner_id: str = "",
    ) -> tuple[int, list[Pose2D], Any]:
        if not self._compute_path_client.wait_for_server(timeout_sec=self.config.server_timeout_s):
            raise RuntimeError("`compute_path_to_pose` action server did not appear in time.")
        self._ensure_action_server_health("compute_path_to_pose")
        request = ComputePathToPose.Goal()
        request.goal = self._build_pose(goal_pose)
        if planner_id:
            request.planner_id = planner_id
        request.use_start = False
        future = self._compute_path_client.send_goal_async(request)
        self._spin_until_future_complete(future)
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError("Nav2 rejected the ComputePathToPose goal.")
        result_future = goal_handle.get_result_async()
        self._spin_until_future_complete(result_future)
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
        return int(getattr(outcome, "status", GoalStatus.STATUS_UNKNOWN)), poses, outcome

    def navigate_to_pose(
        self,
        *,
        goal_pose: Pose2D,
        behavior_tree: str = "",
        should_cancel: Callable[[], bool] | None = None,
        feedback_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> Any:
        if not self._navigate_to_pose_client.wait_for_server(timeout_sec=self.config.server_timeout_s):
            raise RuntimeError("`navigate_to_pose` action server did not appear in time.")
        self._ensure_action_server_health("navigate_to_pose")
        feedback_samples: list[dict[str, Any]] = []

        def _feedback(message: Any) -> None:
            feedback = getattr(message, "feedback", None)
            current_pose = self.current_pose()
            sample = {
                "navigation_time_s": _duration_to_seconds(getattr(feedback, "navigation_time", None)),
                "estimated_time_remaining_s": _duration_to_seconds(
                    getattr(feedback, "estimated_time_remaining", None)
                ),
                "distance_remaining_m": float(getattr(feedback, "distance_remaining", 0.0)),
                "number_of_recoveries": int(getattr(feedback, "number_of_recoveries", 0)),
                "current_pose": None if current_pose is None else current_pose.to_dict(),
            }
            feedback_samples.append(sample)
            if feedback_callback is not None:
                feedback_callback(sample)

        request = NavigateToPose.Goal()
        request.pose = self._build_pose(goal_pose)
        if behavior_tree:
            request.behavior_tree = behavior_tree
        future = self._navigate_to_pose_client.send_goal_async(request, feedback_callback=_feedback)
        self._spin_until_future_complete(future)
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError("Nav2 rejected the NavigateToPose goal.")
        result_future = goal_handle.get_result_async()
        cancel_requested = False
        while not result_future.done():
            self._spin_once(timeout_sec=0.1)
            if should_cancel is not None and should_cancel() and not cancel_requested:
                cancel_requested = True
                cancel_future = goal_handle.cancel_goal_async()
                self._spin_until_future_complete(cancel_future)
        outcome = result_future.result()
        return outcome, feedback_samples

    def perform_turnaround_scan(
        self,
        *,
        reason: str,
        should_cancel: Callable[[], bool] | None = None,
        turn_scan_mode: str | None = None,
        robot_brain_url: str | None = None,
        camera_pan_action_key: str | None = None,
        camera_pan_settle_s: float | None = None,
        camera_pan_step_deg: float | None = None,
        camera_pan_compute_s: float | None = None,
        camera_pan_sample_count: int | None = None,
    ) -> dict[str, Any]:
        start_time = time.time()
        start_pose = self.current_pose()
        observation_start_index = len(self.scan_observations)
        mode = str(turn_scan_mode or self.config.turn_scan_mode)
        sample_count = max(int(camera_pan_sample_count or self.config.camera_pan_sample_count), 2)
        configured_pan_step_deg = float(getattr(self.config, "camera_pan_step_deg", 60.0))
        configured_pan_compute_s = float(getattr(self.config, "camera_pan_compute_s", 2.0))
        pan_step_deg = float(configured_pan_step_deg if camera_pan_step_deg is None else camera_pan_step_deg)
        pan_compute_s = float(configured_pan_compute_s if camera_pan_compute_s is None else camera_pan_compute_s)
        event = {
            "reason": reason,
            "mode": mode,
            "target_yaw_rad": round(self.config.turn_scan_radians, 3),
            "sample_count": sample_count,
            "camera_pan_step_deg": round(pan_step_deg, 3),
            "camera_pan_compute_s": round(max(pan_compute_s, 0.0), 3),
        }
        if mode == "camera_pan":
            return self._perform_camera_pan_scan(
                reason=reason,
                should_cancel=should_cancel,
                start_time=start_time,
                start_pose=start_pose,
                observation_start_index=observation_start_index,
                sample_count=sample_count,
                event=event,
                robot_brain_url=robot_brain_url,
                camera_pan_action_key=camera_pan_action_key,
                camera_pan_settle_s=camera_pan_settle_s,
                camera_pan_step_deg=pan_step_deg,
                camera_pan_compute_s=pan_compute_s,
            )
        if mode != "robot_spin":
            raise ValueError(f"Unsupported turn scan mode: {mode!r}")
        self._set_scan_active_if_available(True)
        self._use_turn_feedback_for_scan_pose = True
        try:
            spin_event = self._manual_spin(should_cancel=should_cancel)
            settle_result = self.hold_stop_until_stable(duration_s=self.config.turn_scan_settle_s)
            raw_observations, observation_stop_index = self.drain_scan_observations(observation_start_index)
        finally:
            self._use_turn_feedback_for_scan_pose = False
            self._set_scan_active_if_available(False)
        observations = _select_turnaround_scan_observations(raw_observations, sample_count=sample_count)
        end_pose = self.current_pose()
        event["elapsed_s"] = round(time.time() - start_time, 3)
        event["spin_completed"] = bool(spin_event.get("spin_completed", False))
        event["spin_stop_reason"] = spin_event.get("spin_stop_reason", "unknown")
        event["spin_feedback_frame"] = spin_event.get("spin_feedback_frame", self.config.odom_frame)
        event["actual_unwrapped_yaw_delta_rad"] = round(
            float(spin_event.get("actual_unwrapped_yaw_delta_rad", 0.0) or 0.0),
            3,
        )
        event["spin_command_angular_speed_rad_s"] = round(
            float(spin_event.get("spin_command_angular_speed_rad_s", self.config.manual_spin_angular_speed_rad_s)),
            3,
        )
        event["spin_timeout_s"] = round(float(spin_event.get("spin_timeout_s", self.config.turn_scan_timeout_s)), 3)
        event["settle_stable"] = bool(settle_result.get("stable", False))
        event["settle_observed_yaw_delta_rad"] = round(float(settle_result.get("observed_yaw_delta_rad", 0.0)), 3)
        event["captured_observation_count"] = len(observations)
        event["raw_observation_count"] = len(raw_observations)
        if start_pose is not None:
            event["start_pose"] = start_pose.to_dict()
        if end_pose is not None:
            event["end_pose"] = end_pose.to_dict()
        if start_pose is not None and end_pose is not None:
            yaw_delta = math.atan2(
                math.sin(end_pose.yaw - start_pose.yaw),
                math.cos(end_pose.yaw - start_pose.yaw),
            )
            event["wrapped_yaw_delta_rad"] = round(yaw_delta, 3)
            event["note"] = (
                "A full 360 degree spin wraps back near the start yaw, so wrapped_yaw_delta_rad may be near 0."
            )
        response = dict(event)
        response["observations"] = observations
        response["observation_stop_index"] = observation_stop_index
        self._nav_scan_history.append(event)
        return response

    def _perform_camera_pan_scan(
        self,
        *,
        reason: str,
        should_cancel: Callable[[], bool] | None,
        start_time: float,
        start_pose: Pose2D | None,
        observation_start_index: int,
        sample_count: int,
        event: dict[str, Any],
        robot_brain_url: str | None = None,
        camera_pan_action_key: str | None = None,
        camera_pan_settle_s: float | None = None,
        camera_pan_step_deg: float | None = None,
        camera_pan_compute_s: float | None = None,
    ) -> dict[str, Any]:
        effective_robot_brain_url = robot_brain_url or self.config.robot_brain_url
        if not effective_robot_brain_url:
            raise RuntimeError("Camera-pan scan requires robot_brain_url; use turn_scan_mode='robot_spin' for base rotation.")
        map_start_stamp_s = float(self.latest_map_stamp_s)
        configured_pan_step_deg = float(getattr(self.config, "camera_pan_step_deg", 60.0))
        configured_pan_compute_s = float(getattr(self.config, "camera_pan_compute_s", 2.0))
        pan_step_deg = abs(float(configured_pan_step_deg if camera_pan_step_deg is None else camera_pan_step_deg))
        if pan_step_deg <= 0.0:
            pan_step_deg = 60.0
        pan_compute_s = max(float(configured_pan_compute_s if camera_pan_compute_s is None else camera_pan_compute_s), 0.0)
        positive_deg: list[float] = []
        angle_deg = 0.0
        while angle_deg < 180.0:
            positive_deg.append(angle_deg)
            angle_deg += pan_step_deg
        positive_deg.append(180.0)
        negative_deg: list[float] = [0.0]
        angle_deg = -pan_step_deg
        while angle_deg > -180.0:
            negative_deg.append(angle_deg)
            angle_deg -= pan_step_deg
        scan_angles = [math.radians(item) for item in positive_deg + negative_deg]
        observations: list[dict[str, Any]] = []
        command_events: list[dict[str, Any]] = []
        settled_sample_events: list[dict[str, Any]] = []
        projected_map_snapshots: list[RosOccupancyMap] = []
        fused_projected_map: RosOccupancyMap | None = None
        fuse_external_projected_maps = bool(getattr(self.config, "fuse_external_projected_map_snapshots", False))
        try:
            if hasattr(self, "_set_scan_active_if_available"):
                self._set_scan_active_if_available(True)
            for pan_rad in scan_angles:
                if should_cancel is not None and should_cancel():
                    event["scan_stop_reason"] = "canceled"
                    break
                point_cloud_start_index = len(self.point_cloud_observations)
                map_before_command_s = float(self.latest_map_stamp_s)
                command_events.append(
                    self._command_camera_pan(
                        pan_rad,
                        robot_brain_url=effective_robot_brain_url,
                        action_key=camera_pan_action_key,
                        settle_s=camera_pan_settle_s,
                    )
                )
                point_cloud_observation = self._wait_for_next_point_cloud_observation(
                    point_cloud_start_index,
                    timeout_s=max(pan_compute_s + 2.0, 3.0),
                )
                self.spin_for(pan_compute_s)
                map_updated = self.latest_map_stamp_s > map_before_command_s
                settled_sample_events.append(
                    {
                        "pan_deg": round(math.degrees(pan_rad), 1),
                        "fresh_point_cloud": point_cloud_observation is not None,
                        "map_updated": bool(map_updated),
                        "compute_s": round(pan_compute_s, 3),
                    }
                )
                observation = self._capture_settled_scan_observation(settle_s=0.0)
                if observation is not None:
                    observation["camera_pan_rad"] = pan_rad
                    observations.append(observation)
                if (
                    fuse_external_projected_maps
                    and not self.config.publish_internal_navigation_map
                    and self.latest_map is not None
                    and self.latest_map_header_frame_id == self.config.map_frame
                ):
                    projected_map_snapshots.append(self.latest_map)
                elif (
                    not self.config.publish_internal_navigation_map
                    and self.latest_map is not None
                    and self.latest_map_header_frame_id == self.config.map_frame
                ):
                    event["external_projected_map_seen"] = True
        finally:
            try:
                command_events.append(
                    self._command_camera_pan(
                        0.0,
                        robot_brain_url=effective_robot_brain_url,
                        action_key=camera_pan_action_key,
                        settle_s=camera_pan_settle_s,
                    )
                )
            except Exception as exc:
                event["restore_error"] = str(exc)
            finally:
                if hasattr(self, "_set_scan_active_if_available"):
                    self._set_scan_active_if_available(False)

        raw_observations, observation_stop_index = self.drain_scan_observations(observation_start_index)
        if not self.config.publish_internal_navigation_map:
            event["external_map_updated_after_scan"] = self.wait_for_map_update(
                after_stamp_s=map_start_stamp_s,
                timeout_s=max(float(self.config.turn_scan_settle_s), 2.0),
            )
            event["external_map_summary"] = self.latest_map_summary()
            if (
                fuse_external_projected_maps
                and self.latest_map is not None
                and self.latest_map_header_frame_id == self.config.map_frame
            ):
                projected_map_snapshots.append(self.latest_map)
            elif (
                self.latest_map is not None
                and self.latest_map_header_frame_id == self.config.map_frame
            ):
                event["external_projected_map_seen"] = True
            if fuse_external_projected_maps:
                fused_projected_map = fuse_projected_maps(projected_map_snapshots)
                if fused_projected_map is not None:
                    response_map_summary = self._occupancy_map_summary(
                        fused_projected_map,
                        frame_id=self.config.map_frame,
                        stamp_s=time.time(),
                    )
                    event["fused_projected_map_summary"] = response_map_summary
                else:
                    event["fused_projected_map_summary"] = None
            else:
                fused_projected_map = None
                event["fused_projected_map_summary"] = None
                event["fused_projected_map_enabled"] = False
        end_pose = self.current_pose()
        event["elapsed_s"] = round(time.time() - start_time, 3)
        event["captured_observation_count"] = len(observations)
        event["raw_observation_count"] = len(raw_observations)
        event["camera_pan_command_count"] = len(command_events)
        event["camera_pan_settled_sample_count"] = len(settled_sample_events)
        event["camera_pan_settled_samples"] = settled_sample_events
        event["camera_pan_commanded_deg"] = [
            round(math.degrees(float(item.get("pan_rad", 0.0))), 1)
            for item in command_events
            if isinstance(item, dict) and "pan_rad" in item
        ]
        event["captured_pose_yaw_deg"] = [
            round(math.degrees(float(item["pose"].yaw)), 1)
            for item in observations
            if isinstance(item.get("pose"), Pose2D)
        ]
        event["camera_pan_action_key"] = camera_pan_action_key or self.config.camera_pan_action_key
        event["scan_stop_reason"] = event.get("scan_stop_reason", "completed")
        if start_pose is not None:
            event["start_pose"] = start_pose.to_dict()
        if end_pose is not None:
            event["end_pose"] = end_pose.to_dict()
        if start_pose is not None and end_pose is not None:
            yaw_delta = math.atan2(
                math.sin(end_pose.yaw - start_pose.yaw),
                math.cos(end_pose.yaw - start_pose.yaw),
            )
            event["wrapped_yaw_delta_rad"] = round(yaw_delta, 3)
            event["note"] = "Camera-pan scan keeps the robot base fixed; yaw change should remain near 0."
        response = dict(event)
        response["observations"] = observations
        response["observation_stop_index"] = observation_stop_index
        if not self.config.publish_internal_navigation_map:
            response["fused_projected_map"] = fused_projected_map
        self._nav_scan_history.append(event)
        return response

    def _command_camera_pan(
        self,
        pan_rad: float,
        *,
        robot_brain_url: str | None = None,
        action_key: str | None = None,
        settle_s: float | None = None,
    ) -> dict[str, Any]:
        payload = {
            "pan_rad": float(pan_rad),
            "action_key": action_key or self.config.camera_pan_action_key,
            "settle_s": float(self.config.camera_pan_settle_s if settle_s is None else settle_s),
        }
        url = f"{str(robot_brain_url or self.config.robot_brain_url).rstrip('/')}/camera/head/pan"
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with request.urlopen(req, timeout=max(float(self.config.server_timeout_s), 1.0)) as response:
                body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Camera pan command failed: HTTP {exc.code}: {detail}") from exc
        result = json.loads(body or "{}")
        if not bool(result.get("succeeded", False)):
            raise RuntimeError(f"Camera pan command failed: {result.get('message', 'unknown error')}")
        return {
            "pan_rad": float(pan_rad),
            "response": result,
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "module": "ros_nav2",
            "map_topic": self.config.map_topic,
            "map_updates_topic": self._map_updates_topic,
            "scan_topic": self.config.scan_topic,
            "point_cloud_topic": self.config.point_cloud_topic,
            "rgb_topic": self.config.rgb_topic,
            "navigation_map_source": self.config.navigation_map_source
            if self.config.publish_internal_navigation_map
            else "external",
            "goals": list(self._nav_goal_history),
            "plans": list(self._nav_plan_history),
            "turn_scans": list(self._nav_scan_history),
            "latest_scan": self.latest_scan_stats,
            "latest_point_cloud": self.latest_point_cloud_stats,
            "latest_map": self.latest_map_summary(),
        }

    def record_goal(self, payload: dict[str, Any]) -> None:
        self._nav_goal_history.append(payload)

    def record_plan(self, payload: dict[str, Any]) -> None:
        self._nav_plan_history.append(payload)

    def publish_navigation_map(
        self,
        occupancy_map: RosOccupancyMap,
        *,
        map_to_odom: Pose2D | None = None,
    ) -> None:
        self._published_navigation_map = occupancy_map
        if map_to_odom is not None:
            self._map_to_odom = map_to_odom
        self.latest_map = occupancy_map
        self.latest_map_stamp_s = time.time()
        self._publish_internal_navigation_state()

    def close(self) -> None:
        self.destroy_node()

    def _publish_internal_navigation_state(self) -> None:
        if not self.config.publish_internal_navigation_map:
            return
        self._publish_map_to_odom_transform()
        if self._published_navigation_map is None:
            return
        self._map_pub.publish(self._occupancy_grid_message(self._published_navigation_map))

    def _publish_map_to_odom_transform(self) -> None:
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self.config.map_frame
        transform.child_frame_id = self.config.odom_frame
        transform.transform.translation.x = float(self._map_to_odom.x)
        transform.transform.translation.y = float(self._map_to_odom.y)
        transform.transform.translation.z = 0.0
        transform.transform.rotation = quaternion_from_yaw(float(self._map_to_odom.yaw))
        self.tf_broadcaster.sendTransform(transform)

    def _occupancy_grid_message(self, occupancy_map: RosOccupancyMap) -> OccupancyGrid:
        message = OccupancyGrid()
        message.header.stamp = self.get_clock().now().to_msg()
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

    def _manual_spin(self, *, should_cancel: Callable[[], bool] | None = None) -> dict[str, Any]:
        return self._spin_by_delta(float(self.config.turn_scan_radians), should_cancel=should_cancel)

    def _spin_by_delta(
        self,
        target_yaw_rad: float,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        twist = Twist()
        target_direction = 1.0 if target_yaw_rad >= 0.0 else -1.0
        command_direction = target_direction * (-1.0 if float(self.config.manual_spin_direction_sign) < 0.0 else 1.0)
        max_command_speed = abs(float(self.config.manual_spin_angular_speed_rad_s))
        requested_angular_rad_s = command_direction * max_command_speed
        fallback_duration_s = abs(target_yaw_rad) / max(max_command_speed, 1e-6)
        step_s = 1.0 / max(self.config.manual_spin_publish_hz, 1e-6)
        timeout_s = max(float(self.config.turn_scan_timeout_s), fallback_duration_s * 3.0, fallback_duration_s + 5.0)
        deadline = time.time() + timeout_s
        start_time = time.time()
        feedback_frame, start_yaw = self._current_turn_feedback()
        feedback_yaw_rad = 0.0 if start_yaw is not None else None
        used_feedback = start_yaw is not None
        timed_fallback_deadline = start_time + fallback_duration_s
        last_feedback_yaw = start_yaw
        last_relative_yaw = 0.0
        while time.time() < deadline:
            if should_cancel is not None and should_cancel():
                stop_reason = "canceled"
                break
            twist.angular.z, target_reached = compute_turn_command(
                requested_angular_rad_s=requested_angular_rad_s,
                target_yaw_rad=target_yaw_rad,
                feedback_yaw_rad=feedback_yaw_rad,
                minimum_command_rad_s=min(max_command_speed, 0.12),
            )
            if target_reached:
                stop_reason = "target_yaw_reached"
                break
            self._cmd_vel_pub.publish(twist)
            self._spin_once(timeout_sec=0.0)
            feedback_frame, current_yaw = self._current_turn_feedback()
            if current_yaw is not None and start_yaw is not None:
                relative_yaw = math.atan2(
                    math.sin(current_yaw - start_yaw),
                    math.cos(current_yaw - start_yaw),
                )
                if last_feedback_yaw is not None:
                    unwrapped_delta = math.atan2(
                        math.sin(current_yaw - last_feedback_yaw),
                        math.cos(current_yaw - last_feedback_yaw),
                    )
                    last_relative_yaw += unwrapped_delta
                else:
                    last_relative_yaw = relative_yaw
                last_feedback_yaw = current_yaw
                feedback_yaw_rad = last_relative_yaw
                used_feedback = True
            elif not used_feedback and time.time() >= timed_fallback_deadline:
                stop_reason = "time_fallback_elapsed"
                break
            time.sleep(step_s)
        else:
            stop_reason = "timeout"
        self._cmd_vel_pub.publish(Twist())
        stop_hold = self.hold_stop_until_stable(duration_s=max(step_s * 6.0, 0.4), min_stable_cycles=2)
        if used_feedback:
            last_relative_yaw += float(stop_hold.get("observed_yaw_delta_rad", 0.0))
        completed = bool(
            (used_feedback and abs(last_relative_yaw) >= max(abs(target_yaw_rad) - math.radians(2.0), 0.0))
            or (not used_feedback and stop_reason == "time_fallback_elapsed")
        )
        return {
            "spin_completed": completed,
            "spin_stop_reason": stop_reason,
            "spin_feedback_frame": feedback_frame if used_feedback else "time_fallback",
            "actual_unwrapped_yaw_delta_rad": round(last_relative_yaw, 3) if used_feedback else None,
            "spin_command_angular_speed_rad_s": round(command_direction * max_command_speed, 3),
            "spin_timeout_s": round(timeout_s, 3),
        }

    def _capture_settled_scan_observation(self, *, settle_s: float | None = None) -> dict[str, Any] | None:
        reference_frame = self.config.odom_frame if self.config.publish_internal_navigation_map else self.config.map_frame
        capture_start = len(self.scan_observations)
        self.hold_stop_until_stable(duration_s=self.config.turn_scan_settle_s if settle_s is None else settle_s)
        observation = self._wait_for_next_scan_observation(capture_start)
        if observation is not None:
            return observation
        if self.scan_observations:
            latest = self.scan_observations[-1]
            if str(latest.get("reference_frame", "")) == reference_frame:
                return dict(latest)
        return None

    def _wait_for_next_scan_observation(self, after_index: int, *, timeout_s: float = 2.0) -> dict[str, Any] | None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            self._spin_once(timeout_sec=0.05)
            if len(self.scan_observations) > after_index:
                return dict(self.scan_observations[-1])
        return None

    def _wait_for_next_point_cloud_observation(
        self,
        after_index: int,
        *,
        timeout_s: float = 3.0,
    ) -> dict[str, Any] | None:
        deadline = time.time() + max(float(timeout_s), 0.0)
        while time.time() < deadline:
            self._spin_once(timeout_sec=0.05)
            if len(self.point_cloud_observations) > after_index:
                return dict(self.point_cloud_observations[-1])
        return None

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
                    advertised_name if advertised_name.startswith("/") else f"/{advertised_name}"
                )
                if normalized_advertised != normalized_action:
                    continue
                if not namespace or namespace == "/":
                    servers.add(f"/{node_name}")
                else:
                    servers.add(f"{namespace.rstrip('/')}/{node_name}")
                break
        return sorted(servers)

    def _action_servers_via_cli(self, action_name: str) -> list[str] | None:
        normalized_action = action_name if action_name.startswith("/") else f"/{action_name}"
        try:
            completed = subprocess.run(
                ["ros2", "action", "info", normalized_action],
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
        in_servers = False
        for raw_line in completed.stdout.splitlines():
            line = raw_line.rstrip()
            if line.startswith("Action servers:"):
                in_servers = True
                continue
            if line.startswith("Action clients:"):
                in_servers = False
                continue
            if in_servers and line.strip():
                servers.append(line.strip())
        return servers

    def _ensure_action_server_health(self, action_name: str) -> None:
        if self.config.allow_multiple_action_servers:
            return
        servers: list[str] = []
        for _attempt in range(10):
            self._spin_once(timeout_sec=0.05)
            servers = self._action_servers(action_name)
            if len(servers) > 1:
                break
            time.sleep(0.05)
        if len(servers) <= 1:
            servers = self._action_servers_via_cli(action_name) or servers
        if len(servers) <= 1:
            return
        raise RuntimeError(
            f"Expected exactly one action server for `{action_name}`, found {len(servers)}: {', '.join(servers)}."
        )

    def _build_pose(self, pose: Pose2D) -> PoseStamped:
        stamped = PoseStamped()
        stamped.header.frame_id = self.config.map_frame
        # A zero stamp asks TF/Nav2 to use the latest available transform. This
        # avoids aborting goals when the goal stamp is a few milliseconds newer
        # than the newest odom/base transform in the TF cache.
        stamped.header.stamp = RosTime().to_msg()
        stamped.pose.position.x = float(pose.x)
        stamped.pose.position.y = float(pose.y)
        stamped.pose.position.z = 0.0
        stamped.pose.orientation = quaternion_from_yaw(float(pose.yaw))
        return stamped


def image_message_to_data_url(message: Image) -> str | None:
    if PILImage is None:
        return None
    if str(message.encoding).lower() not in {"rgb8", "bgr8"}:
        return None
    channels = 3
    expected_bytes = int(message.height) * int(message.width) * channels
    if len(message.data) < expected_bytes:
        return None
    array = np.frombuffer(message.data, dtype=np.uint8, count=expected_bytes).reshape(
        int(message.height),
        int(message.width),
        channels,
    )
    if str(message.encoding).lower() == "bgr8":
        array = array[..., ::-1]
    image = PILImage.fromarray(array, mode="RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def path_length_m(poses: list[Pose2D]) -> float:
    if len(poses) < 2:
        return 0.0
    total = 0.0
    for previous, nxt in zip(poses, poses[1:]):
        total += math.dist((previous.x, previous.y), (nxt.x, nxt.y))
    return total


def seconds_since(stamp_s: float) -> float:
    if stamp_s <= 0:
        return 1e9
    return max(time.time() - stamp_s, 0.0)


def _seconds_to_duration(value_s: float) -> DurationMsg:
    value_s = max(float(value_s), 0.0)
    seconds = int(value_s)
    nanoseconds = int((value_s - seconds) * 1e9)
    return DurationMsg(sec=seconds, nanosec=nanoseconds)


def _duration_to_seconds(value: Any) -> float | None:
    if value is None:
        return None
    sec = getattr(value, "sec", None)
    nanosec = getattr(value, "nanosec", None)
    if sec is None or nanosec is None:
        return None
    return float(sec) + float(nanosec) / 1e9
