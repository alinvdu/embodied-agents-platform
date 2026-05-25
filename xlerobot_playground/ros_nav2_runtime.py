from __future__ import annotations

import base64
from collections import deque
from dataclasses import dataclass
from io import BytesIO
import json
import math
import subprocess
import threading
import time
from typing import Any, Callable, Iterable
from urllib import error, request

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is expected on ROS hosts.
    np = None  # type: ignore[assignment]

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
    from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Quaternion, TransformStamped, Twist
    from nav_msgs.msg import OccupancyGrid
    from nav2_msgs.action import ComputePathToPose, NavigateToPose, Spin
    from rcl_interfaces.msg import Log
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
    from sensor_msgs.msg import CameraInfo, Image, Imu, LaserScan, PointCloud2
    from std_msgs.msg import Bool
    from std_srvs.srv import Empty
    from tf2_ros import Buffer, TransformBroadcaster, TransformListener
    from tf2_ros import ConnectivityException, ExtrapolationException, LookupException
except Exception as exc:  # pragma: no cover - runtime guard.
    IMPORT_ERROR = exc
    rclpy = None
    GoalStatus = None
    DurationMsg = None
    PoseStamped = None
    PoseWithCovarianceStamped = None
    Quaternion = None
    TransformStamped = None
    Twist = None
    OccupancyGrid = None
    ComputePathToPose = None
    NavigateToPose = None
    Spin = None
    Log = None
    ActionClient = None
    Node = object
    DurabilityPolicy = None
    QoSProfile = None
    ReliabilityPolicy = None
    qos_profile_sensor_data = None
    RosTime = None
    CameraInfo = None
    Image = None
    Imu = None
    LaserScan = None
    PointCloud2 = None
    Bool = None
    Empty = None
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


def scan_active_qos_profile() -> Any:
    qos = QoSProfile(depth=1)
    qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
    qos.reliability = ReliabilityPolicy.RELIABLE
    return qos


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


def compose_pose_2d(first: Pose2D, second: Pose2D) -> Pose2D:
    cos_yaw = math.cos(first.yaw)
    sin_yaw = math.sin(first.yaw)
    return Pose2D(
        first.x + cos_yaw * second.x - sin_yaw * second.y,
        first.y + sin_yaw * second.x + cos_yaw * second.y,
        first.yaw + second.yaw,
    )


def inverse_pose_2d(pose: Pose2D) -> Pose2D:
    cos_yaw = math.cos(pose.yaw)
    sin_yaw = math.sin(pose.yaw)
    return Pose2D(
        -cos_yaw * pose.x - sin_yaw * pose.y,
        sin_yaw * pose.x - cos_yaw * pose.y,
        -pose.yaw,
    )


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


def _depth_image_to_meters_array(message: Any) -> np.ndarray:
    if np is None:
        return np.empty((0, 0), dtype=np.float32)
    encoding = str(getattr(message, "encoding", "") or "").lower()
    height = int(getattr(message, "height", 0) or 0)
    width = int(getattr(message, "width", 0) or 0)
    if height <= 0 or width <= 0:
        return np.empty((0, 0), dtype=np.float32)
    if encoding in {"mono16", "16uc1", "uint16"}:
        bytes_per_pixel = 2
        dtype = np.dtype(">u2" if bool(getattr(message, "is_bigendian", False)) else "<u2")
        scale = 0.001
    elif encoding in {"32fc1", "float32"}:
        bytes_per_pixel = 4
        dtype = np.dtype(">f4" if bool(getattr(message, "is_bigendian", False)) else "<f4")
        scale = 1.0
    else:
        return np.empty((0, 0), dtype=np.float32)
    row_bytes = width * bytes_per_pixel
    step = int(getattr(message, "step", 0) or row_bytes)
    if step < row_bytes:
        return np.empty((0, 0), dtype=np.float32)
    data = bytes(getattr(message, "data", b""))
    expected = height * step
    if len(data) < expected:
        return np.empty((0, 0), dtype=np.float32)
    rows = np.frombuffer(data, dtype=np.uint8, count=expected).reshape(height, step)
    packed = np.ascontiguousarray(rows[:, :row_bytes])
    depth = np.frombuffer(packed.tobytes(), dtype=dtype, count=height * width).reshape(height, width)
    return depth.astype(np.float32, copy=False) * float(scale)


def _camera_info_intrinsics(message: Any) -> dict[str, Any] | None:
    try:
        width = int(getattr(message, "width", 0) or 0)
        height = int(getattr(message, "height", 0) or 0)
        k = list(getattr(message, "k", []) or [])
        p = list(getattr(message, "p", []) or [])
        if len(k) >= 6 and float(k[0]) > 0.0 and float(k[4]) > 0.0:
            fx = float(k[0])
            fy = float(k[4])
            cx = float(k[2])
            cy = float(k[5])
        elif len(p) >= 7 and float(p[0]) > 0.0 and float(p[5]) > 0.0:
            fx = float(p[0])
            fy = float(p[5])
            cx = float(p[2])
            cy = float(p[6])
        else:
            return None
    except Exception:
        return None
    if min(width, height) <= 0 or min(fx, fy) <= 0.0:
        return None
    return {
        "width": width,
        "height": height,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "frame_id": str(getattr(getattr(message, "header", None), "frame_id", "") or ""),
        "stamp_s": time.time(),
    }


def _fallback_camera_intrinsics(
    *,
    width: int,
    height: int,
    frame_id: str,
    horizontal_fov_deg: float,
) -> dict[str, Any] | None:
    try:
        width = int(width)
        height = int(height)
        horizontal_fov_rad = math.radians(float(horizontal_fov_deg))
    except Exception:
        return None
    if width <= 1 or height <= 1 or horizontal_fov_rad <= 0.0:
        return None
    fx = width / (2.0 * math.tan(horizontal_fov_rad / 2.0))
    if fx <= 0.0 or not math.isfinite(fx):
        return None
    return {
        "width": width,
        "height": height,
        "fx": fx,
        "fy": fx,
        "cx": width / 2.0,
        "cy": height / 2.0,
        "frame_id": frame_id,
        "stamp_s": time.time(),
        "source": "fallback_horizontal_fov",
        "horizontal_fov_deg": float(horizontal_fov_deg),
    }


def _snapshot_stamp_s(snapshot: Any) -> float:
    if isinstance(snapshot, dict):
        try:
            return float(snapshot.get("stamp_s", 0.0) or 0.0)
        except Exception:
            return 0.0
    return 0.0


def _scaled_intrinsics_for_image(intrinsics: dict[str, Any], *, width: int, height: int) -> tuple[float, float, float, float]:
    source_width = max(float(intrinsics.get("width") or width), 1.0)
    source_height = max(float(intrinsics.get("height") or height), 1.0)
    scale_x = float(width) / source_width
    scale_y = float(height) / source_height
    return (
        float(intrinsics["fx"]) * scale_x,
        float(intrinsics["fy"]) * scale_y,
        float(intrinsics["cx"]) * scale_x,
        float(intrinsics["cy"]) * scale_y,
    )


def _project_depth_pixels_to_camera_link(
    *,
    u: np.ndarray,
    v: np.ndarray,
    depth_m: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    optical_x = (u.astype(np.float32, copy=False) - float(cx)) * depth_m / max(float(fx), 1e-6)
    optical_y = (v.astype(np.float32, copy=False) - float(cy)) * depth_m / max(float(fy), 1e-6)
    optical_z = depth_m.astype(np.float32, copy=False)
    return np.column_stack((optical_z, -optical_x, -optical_y)).astype(np.float32, copy=False)


def _depth_image_to_sampled_camera_link_points(
    *,
    depth_m: np.ndarray,
    valid: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    max_points: int,
) -> np.ndarray:
    rows, cols = np.nonzero(valid)
    if rows.size <= 0:
        return np.empty((0, 3), dtype=np.float32)
    limit = max(int(max_points), 1)
    if rows.size > limit:
        stride = max(int(math.ceil(rows.size / float(limit))), 1)
        rows = rows[::stride]
        cols = cols[::stride]
    z = depth_m[rows, cols].astype(np.float32, copy=False)
    return _project_depth_pixels_to_camera_link(
        u=cols.astype(np.float32),
        v=rows.astype(np.float32),
        depth_m=z,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )


def _bbox_xyxy(value: Any) -> tuple[float, float, float, float] | None:
    if isinstance(value, dict):
        value = value.get("bbox_xyxy") or value.get("box") or value.get("bbox")
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x0, y0, x1, y1 = (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
    except Exception:
        return None
    if not all(math.isfinite(item) for item in (x0, y0, x1, y1)):
        return None
    left, right = sorted((x0, x1))
    top, bottom = sorted((y0, y1))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _scaled_bbox_window(
    *,
    bbox_xyxy: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    cloud_width: int,
    cloud_height: int,
    inner_ratio: float = 0.65,
) -> tuple[int, int, int, int]:
    image_width = max(int(image_width), 1)
    image_height = max(int(image_height), 1)
    cloud_width = max(int(cloud_width), 1)
    cloud_height = max(int(cloud_height), 1)
    left, top, right, bottom = bbox_xyxy
    scale_x = float(cloud_width) / float(image_width)
    scale_y = float(cloud_height) / float(image_height)
    left *= scale_x
    right *= scale_x
    top *= scale_y
    bottom *= scale_y
    ratio = clamp(float(inner_ratio), 0.2, 1.0)
    center_x = (left + right) * 0.5
    center_y = (top + bottom) * 0.5
    half_w = max((right - left) * ratio * 0.5, 1.0)
    half_h = max((bottom - top) * ratio * 0.5, 1.0)
    x0 = int(max(math.floor(center_x - half_w), 0))
    y0 = int(max(math.floor(center_y - half_h), 0))
    x1 = int(min(math.ceil(center_x + half_w), cloud_width - 1))
    y1 = int(min(math.ceil(center_y + half_h), cloud_height - 1))
    if x1 <= x0:
        x1 = min(x0 + 1, cloud_width - 1)
    if y1 <= y0:
        y1 = min(y0 + 1, cloud_height - 1)
    return x0, y0, x1, y1


def _point_dict(point: np.ndarray) -> dict[str, float]:
    return {
        "x": round(float(point[0]), 4),
        "y": round(float(point[1]), 4),
        "z": round(float(point[2]), 4),
    }


def _safe_forward_step_from_points(
    points_base: np.ndarray,
    *,
    target_forward_m: float,
    target_max_m: float,
    max_step_m: float,
    robot_width_m: float,
    clearance_m: float,
    collision_height_min_m: float,
    collision_height_max_m: float,
) -> dict[str, Any]:
    desired_step = min(max(float(max_step_m), 0.0), max(float(target_forward_m) - float(target_max_m), 0.0))
    corridor_half_width = max(float(robot_width_m) * 0.5 + float(clearance_m), 0.05)
    if desired_step <= 1e-3:
        return {
            "safe": True,
            "safe_forward_step_m": 0.0,
            "desired_forward_step_m": round(desired_step, 3),
            "reason": "Object is already inside the configured approach range.",
            "corridor_half_width_m": round(corridor_half_width, 3),
        }
    finite = np.isfinite(points_base).all(axis=1) if points_base.size else np.zeros((0,), dtype=bool)
    if not np.any(finite):
        return {
            "safe": False,
            "safe_forward_step_m": 0.0,
            "desired_forward_step_m": round(desired_step, 3),
            "reason": "No valid RGB-D points are available for corridor safety.",
            "corridor_half_width_m": round(corridor_half_width, 3),
        }
    points = points_base[finite]
    obstacle_mask = (
        (points[:, 0] > 0.05)
        & (points[:, 0] < max(desired_step + 0.15, 0.12))
        & (np.abs(points[:, 1]) <= corridor_half_width)
        & (points[:, 2] >= float(collision_height_min_m))
        & (points[:, 2] <= float(collision_height_max_m))
    )
    if np.any(obstacle_mask):
        nearest = float(np.min(points[obstacle_mask, 0]))
        return {
            "safe": False,
            "safe_forward_step_m": 0.0,
            "desired_forward_step_m": round(desired_step, 3),
            "nearest_blocker_forward_m": round(nearest, 3),
            "reason": "RGB-D corridor check found an obstacle before the requested forward step.",
            "corridor_half_width_m": round(corridor_half_width, 3),
        }
    return {
        "safe": True,
        "safe_forward_step_m": round(desired_step, 3),
        "desired_forward_step_m": round(desired_step, 3),
        "reason": "RGB-D corridor is clear for the requested small forward step.",
        "corridor_half_width_m": round(corridor_half_width, 3),
    }


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


def wrapped_yaw_delta_rad(current: float, previous: float) -> float:
    return math.atan2(math.sin(float(current) - float(previous)), math.cos(float(current) - float(previous)))


def clamp(value: float, low: float, high: float) -> float:
    return min(max(float(value), float(low)), float(high))


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
    relocalization_map_topic: str = "/relocalization_projected_map"
    relocalization_reset_service: str = "/relocalization_octomap_server/reset"
    scan_topic: str = "/scan"
    point_cloud_topic: str = "/camera/head/points"
    point_cloud_update_map_enabled_topic: str = "/camera/head/points/update_map_enabled"
    scan_active_topic: str = "/xlerobot/scan_active"
    nav_active_topic: str = "/xlerobot/nav_active"
    local_rotation_active_topic: str = "/xlerobot/local_rotation_active"
    scan_active_release_delay_s: float = 3.0
    rgb_topic: str = "/camera/head/image_raw"
    depth_topic: str = "/camera/head/depth/image_raw"
    camera_info_topic: str = "/camera/head/camera_info"
    rgbd_update_timeout_s: float = 0.7
    rgbd_fallback_horizontal_fov_deg: float = 64.0
    imu_topic: str = "/imu/filtered_yaw"
    cmd_vel_topic: str = "/cmd_vel"
    initial_pose_topic: str = "/initialpose"
    odom_reset_topic: str = "/xlerobot/odom/set_pose"
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
    manual_spin_direction_sign: float = 1.0
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
        self.latest_relocalization_map: RosOccupancyMap | None = None
        self.latest_relocalization_map_stamp_s: float = 0.0
        self.latest_relocalization_map_header_frame_id: str = ""
        self._last_map_log_s: float = 0.0
        self._last_map_update_log_s: float = 0.0
        self.latest_scan: LaserScan | None = None
        self.latest_scan_stats: dict[str, Any] | None = None
        self.latest_point_cloud_stats: dict[str, Any] | None = None
        self.latest_point_cloud_snapshot: dict[str, Any] | None = None
        self.latest_imu_msg: Imu | None = None
        self._latest_imu_orientation_yaw_rad: float | None = None
        self._latest_imu_orientation_unwrapped_yaw_rad: float | None = None
        self._scan_sensor_yaw_offset_rad: float | None = None
        self._use_turn_feedback_for_scan_pose = False
        self.scan_observations: list[dict[str, Any]] = []
        self.point_cloud_observations: list[dict[str, Any]] = []
        self.latest_image_msg: Image | None = None
        self.latest_image_data_url: str | None = None
        self.latest_depth_msg: Image | None = None
        self.latest_depth_stats: dict[str, Any] | None = None
        self.latest_depth_snapshot: dict[str, Any] | None = None
        self.latest_camera_info_msg: CameraInfo | None = None
        self.latest_camera_info_snapshot: dict[str, Any] | None = None
        self._nav_goal_history: list[dict[str, Any]] = []
        self._nav_plan_history: list[dict[str, Any]] = []
        self._nav_scan_history: list[dict[str, Any]] = []
        self._nav2_log_events: deque[dict[str, Any]] = deque(maxlen=300)
        self._published_navigation_map: RosOccupancyMap | None = None
        self._map_to_odom = Pose2D(0.0, 0.0, 0.0)
        self._publish_map_to_odom_enabled = True
        self._force_publish_navigation_state = False
        self._cmd_vel_pub = self.create_publisher(Twist, config.cmd_vel_topic, 10)
        self._initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped, config.initial_pose_topic, 10)
        self._odom_reset_pose_pub = self.create_publisher(PoseWithCovarianceStamped, config.odom_reset_topic, 10)
        self._point_cloud_update_map_enabled_pub = self.create_publisher(
            Bool,
            config.point_cloud_update_map_enabled_topic,
            10,
        )
        self._scan_active_pub = self.create_publisher(Bool, config.scan_active_topic, scan_active_qos_profile())
        self._nav_active_pub = self.create_publisher(Bool, config.nav_active_topic, scan_active_qos_profile())
        self._local_rotation_active_pub = self.create_publisher(
            Bool,
            config.local_rotation_active_topic,
            scan_active_qos_profile(),
        )
        self._scan_active_heartbeat_enabled = False
        self._last_scan_active_heartbeat_s = 0.0
        self.set_scan_active(False)
        self.set_nav_active(False)
        self.set_local_rotation_active(False)
        map_qos = QoSProfile(depth=1)
        map_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        map_qos.reliability = ReliabilityPolicy.RELIABLE
        self._map_pub = self.create_publisher(OccupancyGrid, config.map_topic, map_qos)
        self._compute_path_client = ActionClient(self, ComputePathToPose, "compute_path_to_pose")
        self._navigate_to_pose_client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self._spin_client = ActionClient(self, Spin, "spin")
        self.create_subscription(OccupancyGrid, config.map_topic, self._on_map, map_qos)
        if config.relocalization_map_topic:
            self.create_subscription(
                OccupancyGrid,
                config.relocalization_map_topic,
                self._on_relocalization_map,
                map_qos,
            )
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
        self.create_subscription(Image, config.depth_topic, self._on_depth, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, config.camera_info_topic, self._on_camera_info, qos_profile_sensor_data)
        self.create_subscription(Imu, config.imu_topic, self._on_imu, qos_profile_sensor_data)
        if Log is not None:
            self.create_subscription(Log, "/rosout", self._on_rosout, 50)
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

    def _on_relocalization_map(self, message: OccupancyGrid) -> None:
        self.latest_relocalization_map = RosOccupancyMap(
            resolution=float(message.info.resolution),
            width=int(message.info.width),
            height=int(message.info.height),
            origin_x=float(message.info.origin.position.x),
            origin_y=float(message.info.origin.position.y),
            data=tuple(int(item) for item in message.data),
        )
        self.latest_relocalization_map_stamp_s = time.time()
        self.latest_relocalization_map_header_frame_id = str(message.header.frame_id)

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
        if transform is not None:
            self.latest_point_cloud_snapshot = {
                "frame_id": str(message.header.frame_id),
                "reference_frame": reference_frame,
                "width": int(message.width),
                "height": int(message.height),
                "stamp_s": time.time(),
                "points_camera": points.copy(),
                "points_reference": transformed_points.copy(),
                "sensor_origin_xyz": None
                if sensor_origin is None
                else tuple(float(item) for item in sensor_origin),
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

    def _on_depth(self, message: Image) -> None:
        self.latest_depth_msg = message
        depth_m = _depth_image_to_meters_array(message)
        valid = np.isfinite(depth_m) & (depth_m > 0.0) if depth_m.size else np.zeros((0,), dtype=bool)
        self.latest_depth_stats = {
            "frame_id": str(message.header.frame_id),
            "width": int(message.width),
            "height": int(message.height),
            "encoding": str(message.encoding),
            "valid_depth_count": int(np.count_nonzero(valid)),
            "stamp_s": time.time(),
        }
        if depth_m.size:
            self.latest_depth_snapshot = {
                "frame_id": str(message.header.frame_id),
                "width": int(message.width),
                "height": int(message.height),
                "encoding": str(message.encoding),
                "stamp_s": time.time(),
                "depth_m": depth_m.copy(),
            }

    def _on_camera_info(self, message: CameraInfo) -> None:
        self.latest_camera_info_msg = message
        intrinsics = _camera_info_intrinsics(message)
        if intrinsics is not None:
            self.latest_camera_info_snapshot = intrinsics

    def _on_rosout(self, message: Any) -> None:
        node = str(getattr(message, "name", "") or "")
        text = str(getattr(message, "msg", "") or "")
        haystack = f"{node} {text}".lower()
        interesting = (
            "nav2" in haystack
            or "controller_server" in haystack
            or "planner_server" in haystack
            or "bt_navigator" in haystack
            or "recover" in haystack
            or "progress" in haystack
        )
        if not interesting:
            return
        self._nav2_log_events.append(
            {
                "wall_s": round(time.time(), 3),
                "node": node,
                "level": int(getattr(message, "level", 0) or 0),
                "message": text[:500],
            }
        )

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

    def recent_nav2_log_events(self, *, since_wall_s: float, limit: int = 20) -> list[dict[str, Any]]:
        events = [
            event for event in list(self._nav2_log_events)
            if float(event.get("wall_s", 0.0) or 0.0) >= float(since_wall_s)
        ]
        return events[-max(int(limit), 1):]

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

    def spin_until_odom_pose(self, *, timeout_s: float | None = None) -> Pose2D:
        deadline = time.time() + (timeout_s if timeout_s is not None else self.config.ready_timeout_s)
        while time.time() < deadline:
            self._spin_once(timeout_sec=0.1)
            pose = self.current_pose_in_frame(self.config.odom_frame)
            if pose is not None:
                return pose
            time.sleep(0.05)
        raise RuntimeError(f"Timed out waiting for `{self.config.odom_frame}->{self.config.base_frame}` pose.")

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

    def estimate_detection_geometry(self, payload: dict[str, Any]) -> dict[str, Any]:
        previous_depth_stamp_s = _snapshot_stamp_s(self.latest_depth_snapshot)
        previous_camera_info_stamp_s = _snapshot_stamp_s(self.latest_camera_info_snapshot)
        try:
            settle_s = clamp(float(payload.get("settle_s", 0.05) or 0.05), 0.0, 0.5)
        except Exception:
            settle_s = 0.05
        if settle_s > 0.0:
            self.spin_for(settle_s)
        try:
            rgbd_wait_s = clamp(
                float(payload.get("rgbd_update_timeout_s", self.config.rgbd_update_timeout_s) or 0.0),
                0.0,
                5.0,
            )
        except Exception:
            rgbd_wait_s = self.config.rgbd_update_timeout_s
        if rgbd_wait_s > 0.0:
            self.wait_for_rgbd_update(
                after_depth_stamp_s=previous_depth_stamp_s,
                after_camera_info_stamp_s=previous_camera_info_stamp_s,
                timeout_s=rgbd_wait_s,
            )
        bbox = _bbox_xyxy(payload.get("bbox_xyxy") or payload.get("bbox") or payload.get("detection"))
        if bbox is None:
            return {"status": "rejected", "reason": "bbox_xyxy is required to estimate object geometry."}
        depth_result = self._estimate_detection_geometry_from_depth(payload, bbox)
        if depth_result.get("status") != "unavailable":
            return depth_result
        if bool(payload.get("disable_point_cloud_fallback") or payload.get("require_depth_image")):
            return {
                **depth_result,
                "point_cloud_fallback": "disabled",
                "reason": depth_result.get("reason") or "Depth-image grounding is unavailable.",
            }
        snapshot = self.latest_point_cloud_snapshot
        if not isinstance(snapshot, dict):
            return {
                "status": "unavailable",
                "reason": (
                    f"{depth_result.get('reason', 'No aligned RGB-D depth image is available yet.')} "
                    "Point-cloud fallback is also unavailable."
                ),
                "depth_image": depth_result.get("depth_image"),
                "camera_info": depth_result.get("camera_info"),
            }
        cloud_width = int(snapshot.get("width", 0) or 0)
        cloud_height = int(snapshot.get("height", 0) or 0)
        if cloud_width <= 1 or cloud_height <= 1:
            return {
                "status": "unavailable",
                "reason": "The latest RGB-D point cloud is not organized, so bbox depth cannot be solved.",
                "depth_image": depth_result.get("depth_image"),
                "camera_info": depth_result.get("camera_info"),
                "point_cloud": {
                    "width": cloud_width,
                    "height": cloud_height,
                    "frame_id": snapshot.get("frame_id"),
                },
            }
        points_camera = snapshot.get("points_camera")
        if not hasattr(points_camera, "shape") or int(points_camera.shape[0]) != cloud_width * cloud_height:
            return {
                "status": "unavailable",
                "reason": "The latest RGB-D point cloud does not match its organized dimensions.",
                "point_cloud": {
                    "width": cloud_width,
                    "height": cloud_height,
                    "frame_id": snapshot.get("frame_id"),
                },
            }
        image_width = int(payload.get("image_width") or payload.get("width") or cloud_width)
        image_height = int(payload.get("image_height") or payload.get("height") or cloud_height)
        inner_ratio = clamp(float(payload.get("bbox_sample_inner_ratio", 0.65) or 0.65), 0.2, 1.0)
        x0, y0, x1, y1 = _scaled_bbox_window(
            bbox_xyxy=bbox,
            image_width=image_width,
            image_height=image_height,
            cloud_width=cloud_width,
            cloud_height=cloud_height,
            inner_ratio=inner_ratio,
        )
        frame_id = str(snapshot.get("frame_id") or "")
        base_transform = self._lookup_transform_xyz_quat(self.config.base_frame, frame_id)
        map_transform = self._lookup_transform_xyz_quat(self.config.map_frame, frame_id)
        if base_transform is None:
            return {
                "status": "unavailable",
                "reason": f"Could not transform RGB-D points from `{frame_id}` to `{self.config.base_frame}`.",
                "bbox_xyxy": [round(item, 3) for item in bbox],
            }
        base_translation, base_quaternion = base_transform
        base_rotation = _quaternion_rotation_matrix(*base_quaternion)
        points_base = (points_camera @ base_rotation.T + base_translation.reshape(1, 3)).astype(np.float32, copy=False)
        points_map = None
        if map_transform is not None:
            map_translation, map_quaternion = map_transform
            map_rotation = _quaternion_rotation_matrix(*map_quaternion)
            points_map = (points_camera @ map_rotation.T + map_translation.reshape(1, 3)).astype(np.float32, copy=False)

        base_grid = points_base.reshape((cloud_height, cloud_width, 3))
        sample_base = base_grid[y0 : y1 + 1, x0 : x1 + 1, :].reshape((-1, 3))
        finite = np.isfinite(sample_base).all(axis=1) if sample_base.size else np.zeros((0,), dtype=bool)
        max_depth_m = clamp(float(payload.get("max_depth_m", 4.0) or 4.0), 0.2, 12.0)
        valid = finite & (sample_base[:, 0] > 0.05) & (sample_base[:, 0] <= max_depth_m)
        min_points = max(int(payload.get("min_valid_points", 12) or 12), 1)
        if int(np.count_nonzero(valid)) < min_points and inner_ratio < 1.0:
            x0, y0, x1, y1 = _scaled_bbox_window(
                bbox_xyxy=bbox,
                image_width=image_width,
                image_height=image_height,
                cloud_width=cloud_width,
                cloud_height=cloud_height,
                inner_ratio=1.0,
            )
            sample_base = base_grid[y0 : y1 + 1, x0 : x1 + 1, :].reshape((-1, 3))
            finite = np.isfinite(sample_base).all(axis=1) if sample_base.size else np.zeros((0,), dtype=bool)
            valid = finite & (sample_base[:, 0] > 0.05) & (sample_base[:, 0] <= max_depth_m)
        valid_count = int(np.count_nonzero(valid))
        if valid_count < min_points:
            return {
                "status": "not_found",
                "reason": f"Only {valid_count} valid depth samples were found inside the detection bbox.",
                "bbox_xyxy": [round(item, 3) for item in bbox],
                "sample_window": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                "point_cloud": {"width": cloud_width, "height": cloud_height, "frame_id": frame_id},
                "valid_sample_count": valid_count,
            }

        object_base = np.median(sample_base[valid], axis=0)
        object_map = None
        if points_map is not None:
            map_grid = points_map.reshape((cloud_height, cloud_width, 3))
            sample_map = map_grid[y0 : y1 + 1, x0 : x1 + 1, :].reshape((-1, 3))
            if sample_map.shape[0] == sample_base.shape[0]:
                object_map = np.median(sample_map[valid], axis=0)
        forward_m = float(object_base[0])
        lateral_m = float(object_base[1])
        vertical_m = float(object_base[2])
        range_m = math.sqrt(forward_m * forward_m + lateral_m * lateral_m + vertical_m * vertical_m)
        target_max_m = clamp(float(payload.get("target_max_m", 0.45) or 0.45), 0.1, 2.0)
        safety = _safe_forward_step_from_points(
            points_base,
            target_forward_m=forward_m,
            target_max_m=target_max_m,
            max_step_m=clamp(float(payload.get("max_step_m", 0.08) or 0.08), 0.0, 0.35),
            robot_width_m=clamp(float(payload.get("robot_width_m", 0.459) or 0.459), 0.1, 1.2),
            clearance_m=clamp(float(payload.get("clearance_m", 0.06) or 0.06), 0.0, 0.5),
            collision_height_min_m=float(payload.get("collision_height_min_m", -0.05) or -0.05),
            collision_height_max_m=float(payload.get("collision_height_max_m", 0.85) or 0.85),
        )
        current_pose = self.current_pose()
        return {
            "status": "succeeded",
            "reason": "Detection bbox was grounded with the latest organized RGB-D point cloud fallback.",
            "geometry_source": "organized_point_cloud",
            "bbox_xyxy": [round(item, 3) for item in bbox],
            "image_size": {"width": image_width, "height": image_height},
            "point_cloud": {
                "width": cloud_width,
                "height": cloud_height,
                "frame_id": frame_id,
                "reference_frame": snapshot.get("reference_frame"),
                "age_s": round(max(time.time() - float(snapshot.get("stamp_s", time.time())), 0.0), 3),
            },
            "sample_window": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
            "valid_sample_count": valid_count,
            "estimated_pose_base": _point_dict(object_base),
            "estimated_pose_map": None if object_map is None else _point_dict(object_map),
            "current_pose": None if current_pose is None else current_pose.to_dict(),
            "distance_m": round(range_m, 3),
            "forward_m": round(forward_m, 3),
            "lateral_m": round(lateral_m, 3),
            "vertical_m": round(vertical_m, 3),
            "bearing_error_deg": round(math.degrees(math.atan2(lateral_m, max(forward_m, 1e-6))), 2),
            "safety": safety,
        }

    def _estimate_detection_geometry_from_depth(
        self,
        payload: dict[str, Any],
        bbox: tuple[float, float, float, float],
    ) -> dict[str, Any]:
        depth_snapshot = self.latest_depth_snapshot
        if not isinstance(depth_snapshot, dict):
            return {
                "status": "unavailable",
                "reason": "No latest aligned RGB-D depth image is available yet.",
            }
        depth_m = depth_snapshot.get("depth_m")
        if not hasattr(depth_m, "shape") or len(depth_m.shape) != 2:
            return {
                "status": "unavailable",
                "reason": "Latest depth image could not be decoded.",
                "depth_image": {
                    "width": depth_snapshot.get("width"),
                    "height": depth_snapshot.get("height"),
                    "frame_id": depth_snapshot.get("frame_id"),
                    "encoding": depth_snapshot.get("encoding"),
                },
            }
        depth_height, depth_width = int(depth_m.shape[0]), int(depth_m.shape[1])
        frame_id = str(depth_snapshot.get("frame_id") or "")
        camera_info = self.latest_camera_info_snapshot
        if not isinstance(camera_info, dict):
            camera_info = _fallback_camera_intrinsics(
                width=depth_width,
                height=depth_height,
                frame_id=frame_id,
                horizontal_fov_deg=self.config.rgbd_fallback_horizontal_fov_deg,
            )
        if not isinstance(camera_info, dict):
            return {
                "status": "unavailable",
                "reason": (
                    "No latest camera_info intrinsics are available for RGB-D bbox grounding, "
                    "and fallback camera FOV is disabled."
                ),
                "depth_image": {
                    "width": depth_snapshot.get("width"),
                    "height": depth_snapshot.get("height"),
                    "frame_id": depth_snapshot.get("frame_id"),
                    "encoding": depth_snapshot.get("encoding"),
                },
            }
        try:
            max_depth_age_s = clamp(float(payload.get("max_depth_age_s", 2.0) or 2.0), 0.1, 10.0)
        except Exception:
            max_depth_age_s = 2.0
        depth_age_s = max(time.time() - float(depth_snapshot.get("stamp_s", 0.0) or 0.0), 0.0)
        if depth_age_s > max_depth_age_s:
            return {
                "status": "unavailable",
                "reason": f"Latest depth image is stale ({depth_age_s:.2f}s old).",
                "depth_image": {
                    "width": depth_snapshot.get("width"),
                    "height": depth_snapshot.get("height"),
                    "frame_id": depth_snapshot.get("frame_id"),
                    "encoding": depth_snapshot.get("encoding"),
                    "age_s": round(depth_age_s, 3),
                },
                "camera_info": {
                    "frame_id": camera_info.get("frame_id"),
                    "age_s": round(max(time.time() - float(camera_info.get("stamp_s", time.time())), 0.0), 3),
                },
            }
        if depth_width <= 1 or depth_height <= 1:
            return {"status": "unavailable", "reason": "Latest depth image is empty."}
        frame_id = str(depth_snapshot.get("frame_id") or camera_info.get("frame_id") or "")
        base_transform = self._lookup_transform_xyz_quat(self.config.base_frame, frame_id)
        map_transform = self._lookup_transform_xyz_quat(self.config.map_frame, frame_id)
        if base_transform is None:
            return {
                "status": "unavailable",
                "reason": f"Could not transform RGB-D depth points from `{frame_id}` to `{self.config.base_frame}`.",
                "bbox_xyxy": [round(item, 3) for item in bbox],
                "depth_image": {"width": depth_width, "height": depth_height, "frame_id": frame_id},
            }
        image_width = int(payload.get("image_width") or payload.get("width") or depth_width)
        image_height = int(payload.get("image_height") or payload.get("height") or depth_height)
        inner_ratio = clamp(float(payload.get("bbox_sample_inner_ratio", 0.65) or 0.65), 0.2, 1.0)
        x0, y0, x1, y1 = _scaled_bbox_window(
            bbox_xyxy=bbox,
            image_width=image_width,
            image_height=image_height,
            cloud_width=depth_width,
            cloud_height=depth_height,
            inner_ratio=inner_ratio,
        )
        max_depth_m = clamp(float(payload.get("max_depth_m", 4.0) or 4.0), 0.2, 12.0)
        min_depth_m = clamp(float(payload.get("min_depth_m", 0.05) or 0.05), 0.01, 2.0)
        min_points = max(int(payload.get("min_valid_points", 12) or 12), 1)
        sample_depth = depth_m[y0 : y1 + 1, x0 : x1 + 1]
        valid = np.isfinite(sample_depth) & (sample_depth >= min_depth_m) & (sample_depth <= max_depth_m)
        if int(np.count_nonzero(valid)) < min_points and inner_ratio < 1.0:
            x0, y0, x1, y1 = _scaled_bbox_window(
                bbox_xyxy=bbox,
                image_width=image_width,
                image_height=image_height,
                cloud_width=depth_width,
                cloud_height=depth_height,
                inner_ratio=1.0,
            )
            sample_depth = depth_m[y0 : y1 + 1, x0 : x1 + 1]
            valid = np.isfinite(sample_depth) & (sample_depth >= min_depth_m) & (sample_depth <= max_depth_m)
        valid_count = int(np.count_nonzero(valid))
        if valid_count < min_points:
            return {
                "status": "not_found",
                "reason": f"Only {valid_count} valid depth samples were found inside the detection bbox.",
                "geometry_source": "depth_image",
                "bbox_xyxy": [round(item, 3) for item in bbox],
                "sample_window": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                "depth_image": {"width": depth_width, "height": depth_height, "frame_id": frame_id},
                "valid_sample_count": valid_count,
            }

        fx, fy, cx, cy = _scaled_intrinsics_for_image(camera_info, width=depth_width, height=depth_height)
        rows, cols = np.nonzero(valid)
        u = cols.astype(np.float32) + float(x0)
        v = rows.astype(np.float32) + float(y0)
        z = sample_depth[valid].astype(np.float32, copy=False)
        sample_camera = _project_depth_pixels_to_camera_link(u=u, v=v, depth_m=z, fx=fx, fy=fy, cx=cx, cy=cy)
        base_translation, base_quaternion = base_transform
        base_rotation = _quaternion_rotation_matrix(*base_quaternion)
        sample_base = (sample_camera @ base_rotation.T + base_translation.reshape(1, 3)).astype(np.float32, copy=False)
        object_base = np.median(sample_base, axis=0)
        object_map = None
        if map_transform is not None:
            map_translation, map_quaternion = map_transform
            map_rotation = _quaternion_rotation_matrix(*map_quaternion)
            object_map = np.median(
                (sample_camera @ map_rotation.T + map_translation.reshape(1, 3)).astype(np.float32, copy=False),
                axis=0,
            )

        depth_valid = np.isfinite(depth_m) & (depth_m >= min_depth_m) & (depth_m <= max_depth_m)
        safety_points_camera = _depth_image_to_sampled_camera_link_points(
            depth_m=depth_m,
            valid=depth_valid,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            max_points=int(payload.get("max_safety_depth_points", 50000) or 50000),
        )
        safety_points_base = (
            (safety_points_camera @ base_rotation.T + base_translation.reshape(1, 3)).astype(np.float32, copy=False)
            if safety_points_camera.size
            else sample_base
        )
        forward_m = float(object_base[0])
        lateral_m = float(object_base[1])
        vertical_m = float(object_base[2])
        range_m = math.sqrt(forward_m * forward_m + lateral_m * lateral_m + vertical_m * vertical_m)
        target_max_m = clamp(float(payload.get("target_max_m", 0.45) or 0.45), 0.1, 2.0)
        safety = _safe_forward_step_from_points(
            safety_points_base,
            target_forward_m=forward_m,
            target_max_m=target_max_m,
            max_step_m=clamp(float(payload.get("max_step_m", 0.08) or 0.08), 0.0, 0.35),
            robot_width_m=clamp(float(payload.get("robot_width_m", 0.459) or 0.459), 0.1, 1.2),
            clearance_m=clamp(float(payload.get("clearance_m", 0.06) or 0.06), 0.0, 0.5),
            collision_height_min_m=float(payload.get("collision_height_min_m", -0.05) or -0.05),
            collision_height_max_m=float(payload.get("collision_height_max_m", 0.85) or 0.85),
        )
        current_pose = self.current_pose()
        return {
            "status": "succeeded",
            "reason": "Detection bbox was grounded with the latest aligned RGB-D depth image.",
            "geometry_source": "depth_image",
            "bbox_xyxy": [round(item, 3) for item in bbox],
            "image_size": {"width": image_width, "height": image_height},
            "depth_image": {
                "width": depth_width,
                "height": depth_height,
                "frame_id": frame_id,
                "encoding": depth_snapshot.get("encoding"),
                "age_s": round(max(time.time() - float(depth_snapshot.get("stamp_s", time.time())), 0.0), 3),
            },
            "camera_info": {
                "width": int(camera_info.get("width", 0) or 0),
                "height": int(camera_info.get("height", 0) or 0),
                "fx": round(float(camera_info.get("fx", 0.0) or 0.0), 3),
                "fy": round(float(camera_info.get("fy", 0.0) or 0.0), 3),
                "cx": round(float(camera_info.get("cx", 0.0) or 0.0), 3),
                "cy": round(float(camera_info.get("cy", 0.0) or 0.0), 3),
                "source": camera_info.get("source", "camera_info_topic"),
                "horizontal_fov_deg": camera_info.get("horizontal_fov_deg"),
            },
            "sample_window": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
            "valid_sample_count": valid_count,
            "estimated_pose_base": _point_dict(object_base),
            "estimated_pose_map": None if object_map is None else _point_dict(object_map),
            "current_pose": None if current_pose is None else current_pose.to_dict(),
            "distance_m": round(range_m, 3),
            "forward_m": round(forward_m, 3),
            "lateral_m": round(lateral_m, 3),
            "vertical_m": round(vertical_m, 3),
            "bearing_error_deg": round(math.degrees(math.atan2(lateral_m, max(forward_m, 1e-6))), 2),
            "safety": safety,
        }

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

    def wait_for_rgbd_update(
        self,
        *,
        after_depth_stamp_s: float = 0.0,
        after_camera_info_stamp_s: float = 0.0,
        timeout_s: float = 0.7,
    ) -> bool:
        deadline = time.time() + max(float(timeout_s), 0.0)
        while time.time() < deadline:
            self._spin_once(timeout_sec=min(0.05, max(deadline - time.time(), 0.0)))
            depth_stamp_s = _snapshot_stamp_s(self.latest_depth_snapshot)
            camera_info_stamp_s = _snapshot_stamp_s(self.latest_camera_info_snapshot)
            if (
                isinstance(self.latest_depth_snapshot, dict)
                and depth_stamp_s > after_depth_stamp_s
                and (
                    isinstance(self.latest_camera_info_snapshot, dict)
                    or self.config.rgbd_fallback_horizontal_fov_deg > 0.0
                )
                and (
                    after_camera_info_stamp_s <= 0.0
                    or camera_info_stamp_s > 0.0
                    or self.config.rgbd_fallback_horizontal_fov_deg > 0.0
                )
            ):
                return True
        return (
            isinstance(self.latest_depth_snapshot, dict)
            and _snapshot_stamp_s(self.latest_depth_snapshot) > after_depth_stamp_s
            and (
                isinstance(self.latest_camera_info_snapshot, dict)
                or self.config.rgbd_fallback_horizontal_fov_deg > 0.0
            )
        )

    def wait_for_map_update(self, *, after_stamp_s: float, timeout_s: float = 2.0) -> bool:
        deadline = time.time() + max(float(timeout_s), 0.0)
        while time.time() < deadline:
            self._spin_once(timeout_sec=0.05)
            if self.latest_map is not None and self.latest_map_stamp_s > after_stamp_s:
                return True
        return False

    def wait_for_relocalization_map_update(self, *, after_stamp_s: float, timeout_s: float = 5.0) -> bool:
        deadline = time.time() + max(float(timeout_s), 0.0)
        while time.time() < deadline:
            self._spin_once(timeout_sec=0.05)
            if (
                self.latest_relocalization_map is not None
                and self.latest_relocalization_map_stamp_s > after_stamp_s
            ):
                return True
        return False

    def reset_relocalization_map(self, *, timeout_s: float = 2.0) -> dict[str, Any]:
        service = str(getattr(self.config, "relocalization_reset_service", "") or "")
        if not service:
            return {"status": "unavailable", "reason": "No relocalization reset service configured."}
        if Empty is None:
            return {"status": "unavailable", "reason": "std_srvs/Empty is unavailable in this environment."}
        client = self.create_client(Empty, service)
        if not client.wait_for_service(timeout_sec=max(float(timeout_s), 0.0)):
            return {"status": "unavailable", "reason": f"Relocalization reset service `{service}` is unavailable."}
        future = client.call_async(Empty.Request())
        deadline = time.time() + max(float(timeout_s), 0.0)
        while not future.done():
            remaining = deadline - time.time()
            if remaining <= 0.0:
                return {"status": "timeout", "reason": f"Timed out calling `{service}`."}
            self._spin_once(timeout_sec=min(0.05, remaining))
        try:
            future.result()
        except Exception as exc:
            return {"status": "failed", "reason": f"Relocalization reset service `{service}` failed: {exc}"}
        return {"status": "ok", "service": service}

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

    def set_nav_active(self, active: bool) -> None:
        message = Bool()
        message.data = bool(active)
        self._nav_active_pub.publish(message)
        self._spin_once(timeout_sec=0.0)

    def set_local_rotation_active(self, active: bool) -> None:
        message = Bool()
        message.data = bool(active)
        self._local_rotation_active_pub.publish(message)
        self._spin_once(timeout_sec=0.0)

    def _set_scan_active_if_available(self, active: bool) -> None:
        self.set_scan_active(active)

    def _refresh_scan_active_heartbeat(self) -> None:
        if not self._scan_active_heartbeat_enabled:
            return
        now_s = time.time()
        if now_s - self._last_scan_active_heartbeat_s < 1.0:
            return
        self._last_scan_active_heartbeat_s = now_s
        self._set_scan_active_if_available(True)

    def _release_scan_active_after_delay(self) -> None:
        delay_s = max(float(getattr(self.config, "scan_active_release_delay_s", 0.0)), 0.0)
        if delay_s > 0.0:
            time.sleep(delay_s)
        self._set_scan_active_if_available(False)

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
        self.set_nav_active(True)
        try:
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
        finally:
            self.set_nav_active(False)

    def rotate_by(
        self,
        *,
        delta_yaw_rad: float,
        reason: str = "",
        should_cancel: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        start_pose = self.current_pose()
        if abs(float(delta_yaw_rad)) > math.pi + 1e-6:
            return self._local_motion_result(
                primitive="rotate_by",
                status="rejected",
                reason="Local rotate_by is bounded to 180 degrees. Use a smaller rotation or relocalization scan.",
                start_pose=start_pose,
                end_pose=start_pose,
                extra={
                    "requested_delta_yaw_deg": round(math.degrees(float(delta_yaw_rad)), 2),
                    "request_reason": reason,
                },
            )
        self.set_point_cloud_map_updates_enabled(False)
        self.set_nav_active(True)
        self.set_local_rotation_active(True)
        try:
            event = self._spin_by_delta(delta_yaw_rad, should_cancel=should_cancel)
        finally:
            self._cmd_vel_pub.publish(Twist())
            self.set_local_rotation_active(False)
            self.set_point_cloud_map_updates_enabled(False)
            self.set_nav_active(False)
        end_pose = self.current_pose()
        stop_reason = str(event.get("spin_stop_reason") or "rotation completed")
        return self._local_motion_result(
            primitive="rotate_by",
            status="succeeded" if bool(event.get("spin_completed", False)) else "failed",
            reason=stop_reason,
            start_pose=start_pose,
            end_pose=end_pose,
            extra={
                "requested_delta_yaw_deg": round(math.degrees(float(delta_yaw_rad)), 2),
                "request_reason": reason,
                "spin": event,
            },
        )

    def rotate_towards_point(
        self,
        *,
        x: float,
        y: float,
        reason: str = "",
        should_cancel: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        pose = self.current_pose()
        if pose is None:
            return {
                "primitive": "rotate_towards_point",
                "status": "failed",
                "reason": "Current robot pose is unavailable.",
                "target_point": {"x": float(x), "y": float(y)},
            }
        target_bearing = math.atan2(float(y) - pose.y, float(x) - pose.x)
        delta_yaw = wrapped_yaw_delta_rad(target_bearing, pose.yaw)
        result = self.rotate_by(delta_yaw_rad=delta_yaw, reason=reason or "rotate toward target point", should_cancel=should_cancel)
        result["primitive"] = "rotate_towards_point"
        result["target_point"] = {"x": round(float(x), 3), "y": round(float(y), 3)}
        result["target_bearing_deg"] = round(math.degrees(target_bearing), 2)
        result["requested_delta_yaw_deg"] = round(math.degrees(delta_yaw), 2)
        return result

    def micro_adjust_to_pose(
        self,
        *,
        target_pose: Pose2D,
        max_distance_m: float = 0.5,
        reason: str = "",
        should_cancel: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        start_pose = self.current_pose()
        if start_pose is None:
            return {
                "primitive": "micro_adjust_to_pose",
                "status": "failed",
                "reason": "Current robot pose is unavailable.",
                "target_pose": target_pose.to_dict(),
            }
        distance = math.hypot(target_pose.x - start_pose.x, target_pose.y - start_pose.y)
        max_distance_m = clamp(max_distance_m, 0.05, 1.0)
        if distance > max_distance_m:
            return self._local_motion_result(
                primitive="micro_adjust_to_pose",
                status="rejected",
                reason=f"Target is {distance:.2f} m away, beyond max micro-adjust distance {max_distance_m:.2f} m.",
                start_pose=start_pose,
                end_pose=start_pose,
                extra={
                    "target_pose": target_pose.to_dict(),
                    "distance_to_target_m": round(distance, 3),
                    "max_distance_m": round(max_distance_m, 3),
                },
            )

        events: list[dict[str, Any]] = []
        self.set_point_cloud_map_updates_enabled(False)
        self.set_nav_active(True)
        try:
            self.set_local_rotation_active(True)
            try:
                orient_event = self._rotate_toward_pose_xy(target_pose, should_cancel=should_cancel)
            finally:
                self.set_local_rotation_active(False)
            if orient_event is not None:
                events.append({"phase": "face_target", **orient_event})
            drive_event = self._drive_straight_towards_pose(target_pose, should_cancel=should_cancel)
            events.append({"phase": "translate", **drive_event})
            current_pose = self.current_pose()
            if current_pose is not None:
                final_delta = wrapped_yaw_delta_rad(target_pose.yaw, current_pose.yaw)
                if abs(final_delta) > math.radians(3.0):
                    self.set_local_rotation_active(True)
                    try:
                        events.append({"phase": "final_yaw", **self._spin_by_delta(final_delta, should_cancel=should_cancel)})
                    finally:
                        self.set_local_rotation_active(False)
        finally:
            self._cmd_vel_pub.publish(Twist())
            self.set_local_rotation_active(False)
            self.set_point_cloud_map_updates_enabled(False)
            self.set_nav_active(False)

        end_pose = self.current_pose()
        remaining = (
            math.hypot(target_pose.x - end_pose.x, target_pose.y - end_pose.y)
            if end_pose is not None
            else None
        )
        status = "succeeded" if remaining is not None and remaining <= 0.08 else "partial"
        return self._local_motion_result(
            primitive="micro_adjust_to_pose",
            status=status,
            reason=reason or ("micro adjustment completed" if status == "succeeded" else "micro adjustment ended before reaching tolerance"),
            start_pose=start_pose,
            end_pose=end_pose,
            extra={
                "target_pose": target_pose.to_dict(),
                "distance_to_target_start_m": round(distance, 3),
                "distance_remaining_m": None if remaining is None else round(remaining, 3),
                "max_distance_m": round(max_distance_m, 3),
                "events": events,
            },
        )

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
        self._scan_active_heartbeat_enabled = True
        self._last_scan_active_heartbeat_s = 0.0
        self._use_turn_feedback_for_scan_pose = True
        try:
            spin_event = self._manual_spin(should_cancel=should_cancel)
            settle_result = self.hold_stop_until_stable(duration_s=self.config.turn_scan_settle_s)
            raw_observations, observation_stop_index = self.drain_scan_observations(observation_start_index)
        finally:
            self._use_turn_feedback_for_scan_pose = False
            self._scan_active_heartbeat_enabled = False
            self._release_scan_active_after_delay()
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
                self._scan_active_heartbeat_enabled = True
                self._last_scan_active_heartbeat_s = 0.0
            for pan_rad in scan_angles:
                if hasattr(self, "_refresh_scan_active_heartbeat"):
                    self._refresh_scan_active_heartbeat()
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
                if hasattr(self, "_refresh_scan_active_heartbeat"):
                    self._refresh_scan_active_heartbeat()
                self.spin_for(pan_compute_s)
                if hasattr(self, "_refresh_scan_active_heartbeat"):
                    self._refresh_scan_active_heartbeat()
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
                self._scan_active_heartbeat_enabled = False
                if hasattr(self, "_set_scan_active_if_available"):
                    if hasattr(self, "_release_scan_active_after_delay"):
                        self._release_scan_active_after_delay()
                    else:
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
        try:
            self._spin_once(timeout_sec=0.0)
        except Exception:
            pass
        return {
            "module": "ros_nav2",
            "map_topic": self.config.map_topic,
            "map_updates_topic": self._map_updates_topic,
            "scan_topic": self.config.scan_topic,
            "point_cloud_topic": self.config.point_cloud_topic,
            "rgb_topic": self.config.rgb_topic,
            "depth_topic": self.config.depth_topic,
            "camera_info_topic": self.config.camera_info_topic,
            "navigation_map_source": self.config.navigation_map_source
            if self.config.publish_internal_navigation_map
            else "external",
            "goals": list(self._nav_goal_history),
            "plans": list(self._nav_plan_history),
            "turn_scans": list(self._nav_scan_history),
            "latest_scan": self.latest_scan_stats,
            "latest_point_cloud": self.latest_point_cloud_stats,
            "latest_depth_image": self.latest_depth_stats,
            "latest_camera_info": None
            if self.latest_camera_info_snapshot is None
            else {
                key: self.latest_camera_info_snapshot.get(key)
                for key in ("frame_id", "width", "height", "fx", "fy", "cx", "cy", "source", "horizontal_fov_deg")
            },
            "latest_map": self.latest_map_summary(),
            "latest_relocalization_map": None
            if self.latest_relocalization_map is None
            else self._occupancy_map_summary(
                self.latest_relocalization_map,
                frame_id=self.latest_relocalization_map_header_frame_id,
                stamp_s=self.latest_relocalization_map_stamp_s,
            ),
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
        force_publish: bool = False,
        publish_map_to_odom: bool = True,
    ) -> None:
        self._published_navigation_map = occupancy_map
        self._publish_map_to_odom_enabled = bool(publish_map_to_odom)
        if map_to_odom is not None:
            self._map_to_odom = map_to_odom
            self._publish_map_to_odom_enabled = True
        if force_publish:
            self._force_publish_navigation_state = True
        self.latest_map = occupancy_map
        self.latest_map_stamp_s = time.time()
        self._publish_internal_navigation_state()

    def publishes_map_to_odom(self) -> bool:
        return bool(
            self._publish_map_to_odom_enabled
            and (self.config.publish_internal_navigation_map or self._force_publish_navigation_state)
        )

    def reset_odom_pose(
        self,
        pose: Pose2D,
        *,
        publish_count: int = 3,
        subscriber_timeout_s: float = 1.0,
        covariance_xy_m2: float = 0.05,
        covariance_yaw_rad2: float = 0.10,
    ) -> dict[str, Any]:
        deadline = time.time() + max(float(subscriber_timeout_s), 0.0)
        subscriber_count = int(self._odom_reset_pose_pub.get_subscription_count())
        while subscriber_count <= 0 and time.time() < deadline:
            self._spin_once(timeout_sec=0.05)
            subscriber_count = int(self._odom_reset_pose_pub.get_subscription_count())
        if subscriber_count <= 0:
            return {
                "status": "unavailable",
                "reason": f"No subscribers are listening on odom reset topic `{self.config.odom_reset_topic}`.",
                "odom_reset_topic": self.config.odom_reset_topic,
                "subscriber_count": subscriber_count,
            }
        for _ in range(max(int(publish_count), 1)):
            message = PoseWithCovarianceStamped()
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = self.config.odom_frame
            message.pose.pose.position.x = float(pose.x)
            message.pose.pose.position.y = float(pose.y)
            message.pose.pose.position.z = 0.0
            message.pose.pose.orientation = quaternion_from_yaw(float(pose.yaw))
            message.pose.covariance[0] = float(covariance_xy_m2)
            message.pose.covariance[7] = float(covariance_xy_m2)
            message.pose.covariance[35] = float(covariance_yaw_rad2)
            self._odom_reset_pose_pub.publish(message)
            self._spin_once(timeout_sec=0.05)
        return {
            "status": "ok",
            "odom_pose": pose.to_dict(),
            "odom_reset_topic": self.config.odom_reset_topic,
            "subscriber_count": subscriber_count,
        }

    def set_initial_pose(
        self,
        pose: Pose2D,
        *,
        publish_count: int = 3,
        covariance_xy_m2: float = 0.25,
        covariance_yaw_rad2: float = 0.06853892326654787,
    ) -> dict[str, Any]:
        odom_pose = self.spin_until_odom_pose(timeout_s=self.config.ready_timeout_s)
        map_to_odom = compose_pose_2d(pose, inverse_pose_2d(odom_pose))
        self._map_to_odom = map_to_odom
        self._publish_map_to_odom_enabled = True
        for _ in range(max(int(publish_count), 1)):
            message = PoseWithCovarianceStamped()
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = self.config.map_frame
            message.pose.pose.position.x = float(pose.x)
            message.pose.pose.position.y = float(pose.y)
            message.pose.pose.position.z = 0.0
            message.pose.pose.orientation = quaternion_from_yaw(float(pose.yaw))
            message.pose.covariance[0] = float(covariance_xy_m2)
            message.pose.covariance[7] = float(covariance_xy_m2)
            message.pose.covariance[35] = float(covariance_yaw_rad2)
            self._initial_pose_pub.publish(message)
            self._publish_map_to_odom_transform()
            self._spin_once(timeout_sec=0.05)
        return {
            "status": "ok",
            "initial_pose": pose.to_dict(),
            "odom_pose": odom_pose.to_dict(),
            "map_to_odom": map_to_odom.to_dict(),
            "initial_pose_topic": self.config.initial_pose_topic,
        }

    def close(self) -> None:
        try:
            self._scan_active_heartbeat_enabled = False
            self.set_scan_active(False)
            self.set_nav_active(False)
            self.set_local_rotation_active(False)
        except Exception:
            pass
        self.destroy_node()

    def _publish_internal_navigation_state(self) -> None:
        if not self.config.publish_internal_navigation_map and not self._force_publish_navigation_state:
            return
        if self._publish_map_to_odom_enabled:
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
            self._refresh_scan_active_heartbeat()
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

    def _rotate_toward_pose_xy(
        self,
        target_pose: Pose2D,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> dict[str, Any] | None:
        pose = self.current_pose()
        if pose is None:
            return None
        bearing = math.atan2(target_pose.y - pose.y, target_pose.x - pose.x)
        delta = wrapped_yaw_delta_rad(bearing, pose.yaw)
        if abs(delta) <= math.radians(4.0):
            return {
                "spin_completed": True,
                "spin_stop_reason": "already_facing_target",
                "requested_delta_yaw_deg": round(math.degrees(delta), 2),
            }
        event = self._spin_by_delta(delta, should_cancel=should_cancel)
        event["requested_delta_yaw_deg"] = round(math.degrees(delta), 2)
        return event

    def _drive_straight_towards_pose(
        self,
        target_pose: Pose2D,
        *,
        should_cancel: Callable[[], bool] | None = None,
        tolerance_m: float = 0.05,
    ) -> dict[str, Any]:
        step_s = 1.0 / max(self.config.manual_spin_publish_hz, 1e-6)
        max_speed_m_s = 0.08
        initial_pose = self.current_pose()
        initial_distance = (
            math.hypot(target_pose.x - initial_pose.x, target_pose.y - initial_pose.y)
            if initial_pose is not None
            else 0.0
        )
        timeout_s = max(2.0, min(12.0, initial_distance / max_speed_m_s + 4.0))
        deadline = time.time() + timeout_s
        samples: list[dict[str, Any]] = []
        stop_reason = "timeout"
        while time.time() < deadline:
            if should_cancel is not None and should_cancel():
                stop_reason = "canceled"
                break
            pose = self.current_pose()
            if pose is None:
                stop_reason = "pose_unavailable"
                break
            distance = math.hypot(target_pose.x - pose.x, target_pose.y - pose.y)
            if distance <= tolerance_m:
                stop_reason = "target_reached"
                break
            bearing = math.atan2(target_pose.y - pose.y, target_pose.x - pose.x)
            bearing_error = wrapped_yaw_delta_rad(bearing, pose.yaw)
            if abs(bearing_error) > math.radians(12.0):
                self._cmd_vel_pub.publish(Twist())
                correction = self._spin_by_delta(bearing_error, should_cancel=should_cancel)
                samples.append(
                    {
                        "event": "bearing_correction",
                        "remaining_distance_m": round(distance, 3),
                        "bearing_error_deg": round(math.degrees(bearing_error), 2),
                        "spin_stop_reason": correction.get("spin_stop_reason"),
                    }
                )
                continue
            twist = Twist()
            twist.linear.x = min(max_speed_m_s, max(0.025, distance * 0.35))
            self._cmd_vel_pub.publish(twist)
            self._spin_once(timeout_sec=0.0)
            if len(samples) < 20:
                samples.append(
                    {
                        "event": "drive",
                        "remaining_distance_m": round(distance, 3),
                        "linear_x_m_s": round(float(twist.linear.x), 3),
                        "bearing_error_deg": round(math.degrees(bearing_error), 2),
                    }
                )
            time.sleep(step_s)
        self._cmd_vel_pub.publish(Twist())
        self.hold_stop_until_stable(duration_s=max(step_s * 4.0, 0.25), min_stable_cycles=2)
        final_pose = self.current_pose()
        final_distance = (
            math.hypot(target_pose.x - final_pose.x, target_pose.y - final_pose.y)
            if final_pose is not None
            else None
        )
        return {
            "stop_reason": stop_reason,
            "timeout_s": round(timeout_s, 3),
            "distance_start_m": round(initial_distance, 3),
            "distance_remaining_m": None if final_distance is None else round(final_distance, 3),
            "samples": samples,
        }

    def _local_motion_result(
        self,
        *,
        primitive: str,
        status: str,
        reason: str,
        start_pose: Pose2D | None,
        end_pose: Pose2D | None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "primitive": primitive,
            "status": status,
            "reason": reason,
            "start_pose": None if start_pose is None else start_pose.to_dict(),
            "end_pose": None if end_pose is None else end_pose.to_dict(),
            "actual_pose_delta_m": (
                None if start_pose is None or end_pose is None else round(math.hypot(end_pose.x - start_pose.x, end_pose.y - start_pose.y), 3)
            ),
            "actual_yaw_delta_deg": (
                None if start_pose is None or end_pose is None else round(math.degrees(wrapped_yaw_delta_rad(end_pose.yaw, start_pose.yaw)), 2)
            ),
        }
        if extra:
            payload.update(extra)
        return payload

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
    if PILImage is None or np is None:
        return None
    if str(message.encoding).lower() not in {"rgb8", "bgr8"}:
        return None
    channels = 3
    height = int(message.height)
    width = int(message.width)
    row_bytes = width * channels
    step = int(getattr(message, "step", 0) or row_bytes)
    if height <= 0 or width <= 0 or step < row_bytes:
        return None
    expected_bytes = height * step
    if len(message.data) < expected_bytes:
        return None
    rows = np.frombuffer(message.data, dtype=np.uint8, count=expected_bytes).reshape(height, step)
    array = rows[:, :row_bytes].reshape(height, width, channels)
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
