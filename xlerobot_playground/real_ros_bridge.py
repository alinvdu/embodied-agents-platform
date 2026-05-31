from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct
import threading
import time
from typing import Any, Protocol, Sequence
from urllib import error, request
from urllib.parse import urljoin

from multido_xlerobot.bootstrap import resolve_xlerobot_repo_root
from xlerobot_playground.imu_transport import build_websocket_url, parse_imu_json
from xlerobot_playground.real_exploration_runtime import (
    RealXLeRobotDirectRuntime,
    RealXLeRobotRuntimeConfig,
)
from xlerobot_playground.rgbd_transport import POINT_CLOUD_FORMAT_XYZ_FLOAT32, unpack_rgbd_frame

try:
    import aiohttp
except Exception as exc:  # pragma: no cover - runtime dependency guard.
    aiohttp = None
    AIOHTTP_IMPORT_ERROR: Exception | None = exc
else:
    AIOHTTP_IMPORT_ERROR = None

IMPORT_ERROR: Exception | None = None
try:
    import rclpy
    from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    from geometry_msgs.msg import Quaternion, TransformStamped, Twist
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from sensor_msgs.msg import CameraInfo, Image, Imu, LaserScan, PointCloud2, PointField
    from std_msgs.msg import Bool, Float32
    from tf2_ros import TransformBroadcaster
except Exception as exc:  # pragma: no cover - exercised as a runtime guard.
    IMPORT_ERROR = exc
    rclpy = None
    MutuallyExclusiveCallbackGroup = None
    MultiThreadedExecutor = None
    Quaternion = None
    TransformStamped = None
    Twist = None
    Odometry = None
    Node = object
    CameraInfo = None
    Image = None
    Imu = None
    LaserScan = None
    PointCloud2 = None
    PointField = None
    Bool = None
    Float32 = None
    TransformBroadcaster = None


DEFAULT_ORBBEC_HORIZONTAL_FOV_RAD = 1.20


@dataclass(frozen=True)
class OrbbecFilesystemConfig:
    output_dir: Path = Path("artifacts/orbbec_rgbd")
    rgb_filename: str = "latest.ppm"
    depth_filename: str = "latest_depth.pgm"
    imu_filename: str = "latest_imu.json"
    horizontal_fov_rad: float = DEFAULT_ORBBEC_HORIZONTAL_FOV_RAD


@dataclass(frozen=True)
class RealRosBridgeConfig:
    repo_root: str = str(resolve_xlerobot_repo_root())
    robot_kind: str = "xlerobot_2wheels"
    port1: str = "/dev/tty.usbmodem5B140330101"
    port2: str = "/dev/tty.usbmodem5B140332271"
    fps: int = 30
    use_degrees: bool = False
    allow_motion_commands: bool = False
    max_linear_m_s: float = 0.05
    max_angular_rad_s: float = 0.20
    cmd_vel_topic: str = "/cmd_vel"
    odom_source: str = "none"
    odom_topic: str = "/odom"
    scan_topic: str = "/scan"
    imu_topic: str = "/imu"
    base_frame: str = "base_link"
    odom_frame: str = "odom"
    head_camera_frame: str = "head_camera_link"
    head_points_topic: str = "/camera/head/points"
    head_points_mode: str = "continuous"
    head_points_settled_delay_s: float = 0.20
    head_points_stale_tolerance_s: float = 0.10
    head_points_update_map_enabled_topic: str = "/camera/head/points/update_map_enabled"
    head_points_update_map_while_base_moving: bool = False
    head_points_only_during_scan: bool = False
    scan_active_topic: str = "/xlerobot/scan_active"
    head_laser_frame: str = "head_laser"
    camera_x_m: float = 0.24
    camera_y_m: float = 0.0
    camera_z_m: float = 0.35
    camera_yaw_rad: float = 0.0
    camera_pitch_rad: float = 0.0
    camera_pitch_topic: str = "/camera/head/pitch_rad"
    camera_pan_topic: str = "/camera/head/pan_rad"
    camera_pose_poll_period_s: float = 0.2
    publish_head_camera: bool = True
    publish_imu: bool = True
    publish_rate_hz: float = 30.0
    imu_publish_rate_hz: float = 200.0
    imu_ws_path: str = "/ws/imu"
    imu_ws_reconnect_delay_s: float = 1.0
    cmd_vel_timeout_s: float = 0.5
    laser_min_range_m: float = 0.05
    laser_max_range_m: float = 6.0
    scan_band_height_px: int = 12
    laser_fill_no_return: bool = False
    orbbec: OrbbecFilesystemConfig = OrbbecFilesystemConfig()
    robot_brain_url: str | None = None


@dataclass(frozen=True)
class RgbdFrame:
    rgb: bytes | None
    rgb_width: int | None
    rgb_height: int | None
    depth_mm: tuple[tuple[int, ...], ...] | None
    depth_width: int | None
    depth_height: int | None
    imu_sample: dict[str, Any] | None
    timestamp_s: float
    frame_index: int | None = None
    depth_be: bytes | None = None
    metadata: dict[str, Any] | None = None
    point_cloud_format: int = 0
    point_cloud_points: bytes | None = None
    point_cloud_count: int = 0
    point_cloud_stride: int = 0
    point_cloud_units: str | None = None


class RgbdSource(Protocol):
    def capture(self) -> RgbdFrame:
        ...


class VelocityRuntime(Protocol):
    def drive_velocity(self, *, linear_m_s: float, angular_rad_s: float) -> Any:
        ...

    def stop(self) -> Any:
        ...

    def close(self) -> None:
        ...


def require_ros_dependencies() -> None:
    if IMPORT_ERROR is not None:
        raise RuntimeError(
            "The real XLeRobot ROS bridge requires `rclpy`, `geometry_msgs`, "
            "`sensor_msgs`, `nav_msgs`, and `tf2_ros` in the active ROS 2 Python environment."
        ) from IMPORT_ERROR


def require_aiohttp() -> None:
    if AIOHTTP_IMPORT_ERROR is not None:
        raise RuntimeError(
            "The websocket IMU bridge requires `aiohttp`. Install it with "
            "`python -m pip install aiohttp` in the active environment."
        ) from AIOHTTP_IMPORT_ERROR


def yaw_to_quaternion_xyzw(yaw_rad: float) -> tuple[float, float, float, float]:
    half = yaw_rad / 2.0
    return 0.0, 0.0, math.sin(half), math.cos(half)


def rpy_to_quaternion_xyzw(roll_rad: float, pitch_rad: float, yaw_rad: float) -> tuple[float, float, float, float]:
    cr = math.cos(roll_rad * 0.5)
    sr = math.sin(roll_rad * 0.5)
    cp = math.cos(pitch_rad * 0.5)
    sp = math.sin(pitch_rad * 0.5)
    cy = math.cos(yaw_rad * 0.5)
    sy = math.sin(yaw_rad * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def imu_ros_timestamp_s(imu_sample: dict[str, Any]) -> float | None:
    if imu_sample.get("has_accel", True) and "accel_timestamp_us" in imu_sample:
        return float(imu_sample["accel_timestamp_us"]) / 1_000_000.0
    if "timestamp_s" in imu_sample:
        return float(imu_sample["timestamp_s"])
    if "timestamp_us" in imu_sample:
        return float(imu_sample["timestamp_us"]) / 1_000_000.0
    if "system_timestamp_us" in imu_sample:
        return float(imu_sample["system_timestamp_us"]) / 1_000_000.0
    if "device_timestamp_us" in imu_sample:
        return float(imu_sample["device_timestamp_us"]) / 1_000_000.0
    if "gyro_timestamp_us" in imu_sample:
        return float(imu_sample["gyro_timestamp_us"]) / 1_000_000.0
    return None


def twist_to_base_velocity(message: Any) -> tuple[float, float]:
    return float(message.linear.x), float(message.angular.z)


def synthesize_scan_from_depth_rows(
    depth_mm: Sequence[Sequence[int | float]],
    *,
    horizontal_fov_rad: float,
    band_height_px: int,
    range_min_m: float,
    range_max_m: float,
    fill_no_return: bool = True,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    height = len(depth_mm)
    if height <= 0:
        raise ValueError("Expected a non-empty depth image.")
    width = len(depth_mm[0])
    if width <= 0:
        raise ValueError("Expected a non-empty depth image.")
    if any(len(row) != width for row in depth_mm):
        raise ValueError("Expected a rectangular depth image.")

    band_half = max(1, int(band_height_px) // 2)
    center = height // 2
    row_start = max(0, center - band_half)
    row_stop = min(height, center + band_half)
    fx = width / (2.0 * math.tan(horizontal_fov_rad / 2.0))
    cx = width / 2.0
    ranges = [math.inf for _ in range(width)]
    for column in range(width):
        image_angle = -math.atan2((column - cx) / max(fx, 1e-6), 1.0)
        best = math.inf
        for row in range(row_start, row_stop):
            raw_depth = float(depth_mm[row][column])
            if not math.isfinite(raw_depth) or raw_depth <= 0.0:
                continue
            planar_range_m = (raw_depth / 1000.0) / max(math.cos(image_angle), 1e-3)
            if range_min_m <= planar_range_m <= range_max_m:
                best = min(best, planar_range_m)
        scan_index = width - 1 - column
        ranges[scan_index] = best
    if fill_no_return:
        ranges = [range_max_m if math.isinf(value) else value for value in ranges]
    if width == 1:
        angles = [0.0]
    else:
        angles = [
            -horizontal_fov_rad / 2.0 + (horizontal_fov_rad * index / (width - 1))
            for index in range(width)
        ]
    return tuple(float(value) for value in ranges), tuple(float(value) for value in angles)


def synthesize_scan_from_depth_be(
    depth_be: bytes,
    *,
    width: int,
    height: int,
    horizontal_fov_rad: float,
    band_height_px: int,
    range_min_m: float,
    range_max_m: float,
    fill_no_return: bool = True,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if width <= 0 or height <= 0:
        raise ValueError("Expected a non-empty depth image.")
    expected_size = width * height * 2
    if len(depth_be) < expected_size:
        raise ValueError(f"Depth payload is truncated: expected {expected_size} bytes, got {len(depth_be)}.")

    band_half = max(1, int(band_height_px) // 2)
    center = height // 2
    row_start = max(0, center - band_half)
    row_stop = min(height, center + band_half)
    fx = width / (2.0 * math.tan(horizontal_fov_rad / 2.0))
    cx = width / 2.0
    payload = memoryview(depth_be)
    ranges = [math.inf for _ in range(width)]
    for column in range(width):
        image_angle = -math.atan2((column - cx) / max(fx, 1e-6), 1.0)
        cos_angle = max(math.cos(image_angle), 1e-3)
        best = math.inf
        for row in range(row_start, row_stop):
            offset = (row * width + column) * 2
            raw_depth = int.from_bytes(payload[offset : offset + 2], "big")
            if raw_depth <= 0:
                continue
            planar_range_m = (float(raw_depth) / 1000.0) / cos_angle
            if range_min_m <= planar_range_m <= range_max_m:
                best = min(best, planar_range_m)
        scan_index = width - 1 - column
        ranges[scan_index] = best
    if fill_no_return:
        ranges = [range_max_m if math.isinf(value) else value for value in ranges]
    if width == 1:
        angles = [0.0]
    else:
        angles = [
            -horizontal_fov_rad / 2.0 + (horizontal_fov_rad * index / (width - 1))
            for index in range(width)
        ]
    return tuple(float(value) for value in ranges), tuple(float(value) for value in angles)


def _read_binary_pnm(path: Path) -> tuple[str, int, int, int, bytes]:
    return _parse_binary_pnm(path.read_bytes())


def _parse_binary_pnm(data: bytes) -> tuple[str, int, int, int, bytes]:
    index = 0
    tokens: list[bytes] = []
    while len(tokens) < 4:
        while index < len(data) and chr(data[index]).isspace():
            index += 1
        if index < len(data) and data[index] == ord("#"):
            while index < len(data) and data[index] not in (10, 13):
                index += 1
            continue
        start = index
        while index < len(data) and not chr(data[index]).isspace():
            index += 1
        tokens.append(data[start:index])
    while index < len(data) and chr(data[index]).isspace():
        index += 1
    magic = tokens[0].decode("ascii")
    width = int(tokens[1])
    height = int(tokens[2])
    max_value = int(tokens[3])
    return magic, width, height, max_value, data[index:]


def read_depth_pgm_mm(path: Path) -> tuple[tuple[tuple[int, ...], ...], int, int]:
    return parse_depth_pgm_mm(path.read_bytes())


def parse_depth_pgm_mm(data: bytes) -> tuple[tuple[tuple[int, ...], ...], int, int]:
    magic, width, height, max_value, payload = _parse_binary_pnm(data)
    if magic != "P5":
        raise ValueError(f"Expected binary PGM P5 depth image, got {magic}.")
    bytes_per_sample = 1 if max_value < 256 else 2
    expected = width * height * bytes_per_sample
    if len(payload) < expected:
        raise ValueError(f"Depth image is truncated: expected {expected} bytes, got {len(payload)}.")
    rows: list[tuple[int, ...]] = []
    offset = 0
    for _ in range(height):
        row: list[int] = []
        for _ in range(width):
            if bytes_per_sample == 1:
                value = payload[offset]
                offset += 1
            else:
                value = int.from_bytes(payload[offset : offset + 2], "big")
                offset += 2
            row.append(value)
        rows.append(tuple(row))
    return tuple(rows), width, height


def read_rgb_ppm(path: Path) -> tuple[bytes, int, int]:
    return parse_rgb_ppm(path.read_bytes())


def parse_rgb_ppm(data: bytes) -> tuple[bytes, int, int]:
    magic, width, height, _max_value, payload = _parse_binary_pnm(data)
    if magic != "P6":
        raise ValueError(f"Expected binary PPM P6 RGB image, got {magic}.")
    expected = width * height * 3
    if len(payload) < expected:
        raise ValueError(f"RGB image is truncated: expected {expected} bytes, got {len(payload)}.")
    return payload[:expected], width, height


def read_imu_json(path: Path) -> dict[str, Any] | None:
    return parse_imu_json(path.read_bytes())


class OrbbecFilesystemRgbdSource:
    """Reads frames produced by the native Orbbec RGB-D sidecar.

    The current repository sidecar is still RGB-only. This reader already names
    the file contract for the next native step: `latest.ppm` plus a 16-bit
    millimetre `latest_depth.pgm`.
    """

    def __init__(self, config: OrbbecFilesystemConfig) -> None:
        self.config = config

    def capture(self) -> RgbdFrame:
        output_dir = self.config.output_dir.expanduser().resolve()
        rgb = None
        rgb_width = None
        rgb_height = None
        depth = None
        depth_width = None
        depth_height = None
        imu_sample = None
        rgb_path = output_dir / self.config.rgb_filename
        depth_path = output_dir / self.config.depth_filename
        imu_path = output_dir / self.config.imu_filename
        if rgb_path.exists():
            rgb, rgb_width, rgb_height = read_rgb_ppm(rgb_path)
        if depth_path.exists():
            depth, depth_width, depth_height = read_depth_pgm_mm(depth_path)
        if imu_path.exists():
            try:
                imu_sample = read_imu_json(imu_path)
            except Exception:
                imu_sample = None
        return RgbdFrame(
            rgb=rgb,
            rgb_width=rgb_width,
            rgb_height=rgb_height,
            depth_mm=depth,
            depth_width=depth_width,
            depth_height=depth_height,
            imu_sample=imu_sample,
            timestamp_s=time.time(),
        )


class RobotBrainClient:
    def __init__(self, base_url: str, *, timeout_s: float = 1.0) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout_s = timeout_s

    def get_bytes(self, path: str) -> bytes:
        with request.urlopen(urljoin(self.base_url, path.lstrip("/")), timeout=self.timeout_s) as response:
            return response.read()

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            urljoin(self.base_url, path.lstrip("/")),
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout_s) as response:
            data = response.read()
        if not data:
            return {}
        return json.loads(data.decode("utf-8"))

    def get_json(self, path: str) -> dict[str, Any]:
        data = self.get_bytes(path)
        if not data:
            return {}
        return json.loads(data.decode("utf-8"))

    def websocket_url(self, path: str) -> str:
        return build_websocket_url(self.base_url, path)


class RobotBrainRgbdSource:
    def __init__(self, client: RobotBrainClient) -> None:
        self.client = client

    def capture(self) -> RgbdFrame:
        try:
            frame = unpack_rgbd_frame(self.client.get_bytes("/rgbd"))
            return RgbdFrame(
                rgb=frame.rgb,
                rgb_width=frame.rgb_width,
                rgb_height=frame.rgb_height,
                depth_mm=None,
                depth_width=frame.depth_width,
                depth_height=frame.depth_height,
                imu_sample=None,
                timestamp_s=frame.timestamp_s,
                frame_index=frame.frame_index,
                depth_be=frame.depth_be,
                metadata=frame.metadata,
                point_cloud_format=frame.point_cloud_format,
                point_cloud_points=frame.point_cloud_points,
                point_cloud_count=frame.point_cloud_count,
                point_cloud_stride=frame.point_cloud_stride,
                point_cloud_units=frame.point_cloud_units,
            )
        except Exception:
            pass
        rgb = None
        rgb_width = None
        rgb_height = None
        depth = None
        depth_width = None
        depth_height = None
        try:
            rgb, rgb_width, rgb_height = parse_rgb_ppm(self.client.get_bytes("/rgb"))
        except Exception:
            pass
        try:
            depth, depth_width, depth_height = parse_depth_pgm_mm(self.client.get_bytes("/depth"))
        except Exception:
            pass
        return RgbdFrame(
            rgb=rgb,
            rgb_width=rgb_width,
            rgb_height=rgb_height,
            depth_mm=depth,
            depth_width=depth_width,
            depth_height=depth_height,
            imu_sample=None,
            timestamp_s=time.time(),
            metadata=None,
        )


class RobotBrainVelocityRuntime:
    def __init__(self, client: RobotBrainClient) -> None:
        self.client = client

    def drive_velocity(self, *, linear_m_s: float, angular_rad_s: float) -> dict[str, Any]:
        return self.client.post_json(
            "/cmd_vel",
            {"linear_m_s": float(linear_m_s), "angular_rad_s": float(angular_rad_s)},
        )

    def stop(self) -> dict[str, Any]:
        return self.client.post_json("/stop", {})

    def close(self) -> None:
        try:
            self.stop()
        except Exception:
            pass


def _motion_result_error(response: Any) -> str | None:
    if not isinstance(response, dict):
        return None
    succeeded = response.get("succeeded")
    if succeeded is None or bool(succeeded):
        return None
    message = str(response.get("message", "")).strip()
    metadata = response.get("metadata")
    if message and metadata:
        return f"{message} metadata={metadata}"
    if message:
        return message
    return json.dumps(response, sort_keys=True)


class RealXLeRobotRosBridge(Node):
    def __init__(
        self,
        config: RealRosBridgeConfig,
        *,
        runtime: VelocityRuntime | None = None,
        rgbd_source: RgbdSource | None = None,
    ) -> None:
        require_ros_dependencies()
        super().__init__("xlerobot_real_ros_bridge")
        self.config = config
        brain_client = RobotBrainClient(config.robot_brain_url) if config.robot_brain_url else None
        self.brain_client = brain_client
        if runtime is not None:
            self.runtime = runtime
        elif brain_client is not None:
            self.runtime = RobotBrainVelocityRuntime(brain_client)
        else:
            runtime_config = RealXLeRobotRuntimeConfig(
                repo_root=config.repo_root,
                robot_kind=config.robot_kind,
                port1=config.port1,
                port2=config.port2,
                fps=config.fps,
                use_degrees=config.use_degrees,
                allow_motion_commands=config.allow_motion_commands,
                max_linear_m_s=config.max_linear_m_s,
                max_angular_rad_s=config.max_angular_rad_s,
            )
            self.runtime = RealXLeRobotDirectRuntime(runtime_config)
        if rgbd_source is not None:
            self.rgbd_source = rgbd_source
        elif brain_client is not None:
            self.rgbd_source = RobotBrainRgbdSource(brain_client)
        else:
            self.rgbd_source = OrbbecFilesystemRgbdSource(config.orbbec)
        self.tf_broadcaster = TransformBroadcaster(self)
        self._latest_twist = Twist()
        self._latest_cmd_stamp = 0.0
        self._last_step_stamp = time.monotonic()
        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0
        self._last_linear = 0.0
        self._last_angular = 0.0
        self._last_motion_error = ""
        self._last_imu_timestamp_s: float | None = None
        self._max_imu_timestamp_s: float | None = None
        self._last_accel_frame_index: int | None = None
        self._last_gyro_frame_index: int | None = None
        self._imu_duplicate_timestamps = 0
        self._imu_stale_timestamps = 0
        self._imu_inferred_drop_count = 0
        self._imu_received_count = 0
        self._imu_published_count = 0
        self._imu_log_started_at = time.monotonic()
        self._imu_log_received_at = 0
        self._imu_log_published_at = 0
        self._imu_stream_stop = threading.Event()
        self._imu_stream_thread: threading.Thread | None = None
        self._cmd_vel_callback_group = MutuallyExclusiveCallbackGroup()
        self._step_callback_group = MutuallyExclusiveCallbackGroup()
        self._imu_callback_group = MutuallyExclusiveCallbackGroup()

        self.create_subscription(
            Twist,
            config.cmd_vel_topic,
            self._on_cmd_vel,
            10,
            callback_group=self._cmd_vel_callback_group,
        )
        self.odom_publisher = None
        if config.odom_source == "commanded":
            self.odom_publisher = self.create_publisher(Odometry, config.odom_topic, 10)
        self.scan_publisher = self.create_publisher(LaserScan, config.scan_topic, 10)
        self.imu_publisher = self.create_publisher(Imu, config.imu_topic, 10) if config.publish_imu else None
        self.head_rgb_publisher = None
        self.head_depth_publisher = None
        self.head_camera_info_publisher = None
        self.head_points_publisher = None
        self.camera_pitch_publisher = self.create_publisher(Float32, config.camera_pitch_topic, 10)
        self.camera_pan_publisher = self.create_publisher(Float32, config.camera_pan_topic, 10)
        self._camera_pitch_rad = float(config.camera_pitch_rad)
        self._camera_pan_rad = 0.0
        self._camera_pose_updated_s: float | None = None
        self._camera_pose_received_s: float | None = None
        self._camera_pose_moving = False
        self._last_camera_pose_poll_s = 0.0
        self._base_motion_active = False
        self._head_points_update_map_enabled = True
        self._scan_active = False
        self._last_head_points_skip_reason = ""
        if config.publish_head_camera:
            self.head_rgb_publisher = self.create_publisher(Image, "/camera/head/image_raw", 10)
            self.head_depth_publisher = self.create_publisher(Image, "/camera/head/depth/image_raw", 10)
            self.head_camera_info_publisher = self.create_publisher(CameraInfo, "/camera/head/camera_info", 10)
            self.head_points_publisher = self.create_publisher(PointCloud2, config.head_points_topic, 10)
            self.create_subscription(
                Bool,
                config.head_points_update_map_enabled_topic,
                self._on_head_points_update_map_enabled,
                10,
            )
            self.create_subscription(
                Bool,
                config.scan_active_topic,
                self._on_scan_active,
                10,
            )
        self.timer = self.create_timer(
            1.0 / max(config.publish_rate_hz, 1e-6),
            self.step,
            callback_group=self._step_callback_group,
        )
        self.imu_timer = None
        if config.publish_imu and brain_client is not None:
            require_aiohttp()
            self._start_imu_stream_thread(brain_client.websocket_url(config.imu_ws_path))
        elif config.publish_imu:
            self.imu_timer = self.create_timer(
                1.0 / max(config.imu_publish_rate_hz, 1e-6),
                self._poll_and_publish_imu,
                callback_group=self._imu_callback_group,
            )
        self.get_logger().info(
            "Real bridge ready: "
            f"robot={config.robot_kind} port1={config.port1} port2={config.port2} "
            f"cmd_vel={config.cmd_vel_topic} odom={config.odom_topic} scan={config.scan_topic} "
            f"odom_source={config.odom_source} motion_enabled={config.allow_motion_commands} "
            f"publish_imu={config.publish_imu} "
            f"robot_brain_url={config.robot_brain_url or 'local'}"
        )

    def _on_cmd_vel(self, message: Any) -> None:
        self._latest_twist = message
        self._latest_cmd_stamp = time.monotonic()

    def _active_velocity(self) -> tuple[float, float]:
        if self._latest_cmd_stamp <= 0.0:
            return 0.0, 0.0
        if time.monotonic() - self._latest_cmd_stamp > self.config.cmd_vel_timeout_s:
            return 0.0, 0.0
        return twist_to_base_velocity(self._latest_twist)

    def step(self) -> None:
        now = time.monotonic()
        dt = max(0.0, now - self._last_step_stamp)
        self._last_step_stamp = now
        self._poll_camera_pose(now_s=now)
        linear, angular = self._active_velocity()
        self._update_base_motion_state(linear=linear, angular=angular, now_s=now)
        motion_sent = self._drive_or_stop(linear=linear, angular=angular)
        if motion_sent:
            self._integrate_commanded_odom(linear=linear, angular=angular, dt=dt)
        frame = self.rgbd_source.capture()
        stamp = self.get_clock().now().to_msg()
        self._publish_transforms(stamp=stamp, linear=linear, angular=angular)
        self._publish_scan(frame=frame, stamp=stamp)
        self._publish_head_images(frame=frame, stamp=stamp)
        self._publish_head_points(frame=frame, stamp=stamp)

    def _poll_camera_pose(self, *, now_s: float) -> None:
        if self.brain_client is not None and now_s - self._last_camera_pose_poll_s >= self.config.camera_pose_poll_period_s:
            self._last_camera_pose_poll_s = now_s
            try:
                state = self.brain_client.get_json("/camera/head/pose")
                if "pitch_rad" in state:
                    self._camera_pitch_rad = float(state["pitch_rad"])
                if "pan_rad" in state:
                    self._camera_pan_rad = float(state["pan_rad"])
                if "updated_s" in state:
                    self._camera_pose_updated_s = float(state["updated_s"])
                self._camera_pose_moving = bool(state.get("moving", False))
                self._camera_pose_received_s = now_s
            except Exception as exc:
                self.get_logger().warning(f"Failed to fetch camera head pose: {_format_runtime_error(exc)}")
        pitch_msg = Float32()
        pitch_msg.data = float(self._camera_pitch_rad)
        self.camera_pitch_publisher.publish(pitch_msg)
        pan_msg = Float32()
        pan_msg.data = float(self._camera_pan_rad)
        self.camera_pan_publisher.publish(pan_msg)

    def _update_base_motion_state(self, *, linear: float, angular: float, now_s: float) -> None:
        _ = now_s
        self._base_motion_active = abs(float(linear)) > 1e-5 or abs(float(angular)) > 1e-5

    def _on_head_points_update_map_enabled(self, message: Any) -> None:
        enabled = bool(message.data)
        if enabled == self._head_points_update_map_enabled:
            return
        self._head_points_update_map_enabled = enabled
        state = "enabled" if enabled else "disabled"
        self.get_logger().info(f"PointCloud2 map updates {state} by {self.config.head_points_update_map_enabled_topic}.")

    def _on_scan_active(self, message: Any) -> None:
        active = bool(message.data)
        if active == self._scan_active:
            return
        self._scan_active = active
        state = "active" if active else "inactive"
        self.get_logger().info(f"Scan-active settled PointCloud2 gating is now {state}.")
        if not active:
            self._last_head_points_skip_reason = ""

    def _start_imu_stream_thread(self, websocket_url: str) -> None:
        self._imu_stream_thread = threading.Thread(
            target=self._run_imu_stream_thread,
            args=(websocket_url,),
            name="robot-brain-imu-ws",
            daemon=True,
        )
        self._imu_stream_thread.start()

    def _run_imu_stream_thread(self, websocket_url: str) -> None:
        asyncio.run(self._imu_stream_loop(websocket_url))

    async def _imu_stream_loop(self, websocket_url: str) -> None:
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=5.0, sock_read=None)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while not self._imu_stream_stop.is_set():
                try:
                    self.get_logger().info(f"Connecting IMU websocket: {websocket_url}")
                    async with session.ws_connect(websocket_url, heartbeat=20.0) as ws:
                        self.get_logger().info(f"IMU websocket connected: {websocket_url}")
                        while not self._imu_stream_stop.is_set():
                            try:
                                message = await ws.receive(timeout=1.0)
                            except asyncio.TimeoutError:
                                continue
                            if message.type == aiohttp.WSMsgType.TEXT:
                                payload = message.data
                            elif message.type == aiohttp.WSMsgType.BINARY:
                                payload = message.data
                            elif message.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
                                raise RuntimeError("IMU websocket closed")
                            elif message.type == aiohttp.WSMsgType.ERROR:
                                raise ws.exception() or RuntimeError("IMU websocket error")
                            else:
                                continue
                            try:
                                sample = parse_imu_json(payload)
                            except Exception as exc:
                                self.get_logger().warning(f"Failed to decode IMU websocket payload: {exc}")
                                continue
                            if sample is None:
                                continue
                            self._handle_streamed_imu_sample(sample)
                except Exception as exc:
                    if self._imu_stream_stop.is_set():
                        break
                    self.get_logger().warning(
                        "IMU websocket disconnected: "
                        f"{_format_runtime_error(exc)}; retrying in {self.config.imu_ws_reconnect_delay_s:.1f}s"
                    )
                    await asyncio.sleep(max(self.config.imu_ws_reconnect_delay_s, 0.1))

    def _handle_streamed_imu_sample(self, imu_sample: dict[str, Any]) -> None:
        timestamp_s = imu_ros_timestamp_s(imu_sample)
        if timestamp_s is None:
            timestamp_s = time.time()
        self._imu_received_count += 1
        if self._last_imu_timestamp_s is not None and timestamp_s == self._last_imu_timestamp_s:
            self._imu_duplicate_timestamps += 1
            return
        if self._max_imu_timestamp_s is not None and timestamp_s < self._max_imu_timestamp_s:
            self._imu_stale_timestamps += 1
            return
        self._last_imu_timestamp_s = timestamp_s
        self._max_imu_timestamp_s = timestamp_s if self._max_imu_timestamp_s is None else max(
            self._max_imu_timestamp_s,
            timestamp_s,
        )
        self._track_imu_frame_gap(imu_sample, "accel_frame_index", "_last_accel_frame_index")
        self._track_imu_frame_gap(imu_sample, "gyro_frame_index", "_last_gyro_frame_index")
        self._publish_imu_sample(imu_sample=imu_sample)
        self._imu_published_count += 1
        self._log_imu_stream_stats()

    def _track_imu_frame_gap(self, imu_sample: dict[str, Any], key: str, attr_name: str) -> None:
        current = imu_sample.get(key)
        if current is None:
            return
        current_index = int(current)
        previous = getattr(self, attr_name)
        if previous is not None and current_index > previous + 1:
            self._imu_inferred_drop_count += current_index - previous - 1
        setattr(self, attr_name, current_index)

    def _log_imu_stream_stats(self) -> None:
        if self._imu_published_count <= 0 or self._imu_published_count % 200 != 0:
            return
        now = time.monotonic()
        dt = max(now - self._imu_log_started_at, 1e-6)
        received_delta = self._imu_received_count - self._imu_log_received_at
        published_delta = self._imu_published_count - self._imu_log_published_at
        self.get_logger().info(
            "IMU stream "
            f"rx~={received_delta / dt:.1f}Hz publish~={published_delta / dt:.1f}Hz "
            f"duplicates={self._imu_duplicate_timestamps} stale={self._imu_stale_timestamps} "
            f"inferred_drops={self._imu_inferred_drop_count}"
        )
        self._imu_log_started_at = now
        self._imu_log_received_at = self._imu_received_count
        self._imu_log_published_at = self._imu_published_count

    def _capture_imu_sample(self) -> dict[str, Any] | None:
        imu_path = self.config.orbbec.output_dir.expanduser().resolve() / self.config.orbbec.imu_filename
        if not imu_path.exists():
            return None
        try:
            return read_imu_json(imu_path)
        except Exception:
            return None

    def _poll_and_publish_imu(self) -> None:
        imu_sample = self._capture_imu_sample()
        if imu_sample is None:
            return
        timestamp_s = imu_ros_timestamp_s(imu_sample)
        if timestamp_s is None:
            timestamp_s = time.time()
        if self._last_imu_timestamp_s is not None and timestamp_s <= self._last_imu_timestamp_s:
            return
        self._last_imu_timestamp_s = timestamp_s
        self._publish_imu_sample(imu_sample=imu_sample)

    def _drive_or_stop(self, *, linear: float, angular: float) -> bool:
        try:
            if abs(linear) <= 1e-6 and abs(angular) <= 1e-6:
                if abs(self._last_linear) > 1e-6 or abs(self._last_angular) > 1e-6:
                    response = self.runtime.stop()
                    motion_error = _motion_result_error(response)
                    if motion_error is not None:
                        raise RuntimeError(f"Stop rejected by runtime: {motion_error}")
            else:
                response = self.runtime.drive_velocity(linear_m_s=linear, angular_rad_s=angular)
                motion_error = _motion_result_error(response)
                if motion_error is not None:
                    raise RuntimeError(f"Motion rejected by runtime: {motion_error}")
        except Exception as exc:
            message = _format_runtime_error(exc)
            if message != self._last_motion_error:
                self.get_logger().error(f"Motion command failed: {message}")
                self._last_motion_error = message
            self._last_linear = 0.0
            self._last_angular = 0.0
            return False
        self._last_motion_error = ""
        self._last_linear = linear
        self._last_angular = angular
        return True

    def _integrate_commanded_odom(self, *, linear: float, angular: float, dt: float) -> None:
        if dt <= 0.0:
            return
        self._yaw = _angle_wrap(self._yaw + angular * dt)
        self._x += linear * math.cos(self._yaw) * dt
        self._y += linear * math.sin(self._yaw) * dt

    def _publish_transforms(self, *, stamp: Any, linear: float, angular: float) -> None:
        qx, qy, qz, qw = yaw_to_quaternion_xyzw(self._yaw)
        transforms: list[Any] = []
        if self.config.odom_source == "commanded":
            if self.odom_publisher is not None:
                odom = Odometry()
                odom.header.stamp = stamp
                odom.header.frame_id = self.config.odom_frame
                odom.child_frame_id = self.config.base_frame
                odom.pose.pose.position.x = self._x
                odom.pose.pose.position.y = self._y
                odom.pose.pose.position.z = 0.0
                odom.pose.pose.orientation = _quaternion_msg(qx, qy, qz, qw)
                odom.twist.twist.linear.x = linear
                odom.twist.twist.angular.z = angular
                self.odom_publisher.publish(odom)

            odom_tf = TransformStamped()
            odom_tf.header.stamp = stamp
            odom_tf.header.frame_id = self.config.odom_frame
            odom_tf.child_frame_id = self.config.base_frame
            odom_tf.transform.translation.x = self._x
            odom_tf.transform.translation.y = self._y
            odom_tf.transform.translation.z = 0.0
            odom_tf.transform.rotation = _quaternion_msg(qx, qy, qz, qw)
            transforms.append(odom_tf)

        camera_tf = TransformStamped()
        camera_tf.header.stamp = stamp
        camera_tf.header.frame_id = self.config.base_frame
        camera_tf.child_frame_id = self.config.head_camera_frame
        camera_tf.transform.translation.x = self.config.camera_x_m
        camera_tf.transform.translation.y = self.config.camera_y_m
        camera_tf.transform.translation.z = self.config.camera_z_m
        cx, cy, cz, cw = rpy_to_quaternion_xyzw(
            0.0,
            self._camera_pitch_rad,
            self.config.camera_yaw_rad + self._camera_pan_rad,
        )
        camera_tf.transform.rotation = _quaternion_msg(cx, cy, cz, cw)

        laser_tf = TransformStamped()
        laser_tf.header.stamp = stamp
        laser_tf.header.frame_id = self.config.head_camera_frame
        laser_tf.child_frame_id = self.config.head_laser_frame
        laser_tf.transform.translation.x = 0.0
        laser_tf.transform.translation.y = 0.0
        laser_tf.transform.translation.z = 0.0
        laser_tf.transform.rotation = _quaternion_msg(0.0, 0.0, 0.0, 1.0)
        transforms.extend([camera_tf, laser_tf])
        self.tf_broadcaster.sendTransform(transforms)

    def _publish_imu_sample(self, *, imu_sample: dict[str, Any]) -> None:
        if self.imu_publisher is None:
            return
        msg = Imu()
        imu_timestamp_s = imu_ros_timestamp_s(imu_sample)
        if imu_timestamp_s is None:
            stamp = self.get_clock().now().to_msg()
            msg.header.stamp = stamp
        else:
            timestamp_s = float(imu_timestamp_s)
            sec = int(math.floor(timestamp_s))
            nanosec = int(round((timestamp_s - sec) * 1_000_000_000.0))
            if nanosec >= 1_000_000_000:
                sec += 1
                nanosec -= 1_000_000_000
            msg.header.stamp.sec = sec
            msg.header.stamp.nanosec = nanosec
        msg.header.frame_id = self.config.head_camera_frame
        msg.orientation_covariance[0] = -1.0
        angular = imu_sample.get("angular_velocity_rad_s", {})
        linear = imu_sample.get("linear_acceleration_m_s2", {})
        msg.angular_velocity.x = float(angular.get("x", 0.0))
        msg.angular_velocity.y = float(angular.get("y", 0.0))
        msg.angular_velocity.z = float(angular.get("z", 0.0))
        msg.linear_acceleration.x = float(linear.get("x", 0.0))
        msg.linear_acceleration.y = float(linear.get("y", 0.0))
        msg.linear_acceleration.z = float(linear.get("z", 0.0))
        msg.angular_velocity_covariance[0] = 0.01
        msg.angular_velocity_covariance[4] = 0.01
        msg.angular_velocity_covariance[8] = 0.01
        msg.linear_acceleration_covariance[0] = 0.1
        msg.linear_acceleration_covariance[4] = 0.1
        msg.linear_acceleration_covariance[8] = 0.1
        orientation = imu_sample.get("orientation_xyzw")
        if isinstance(orientation, dict):
            msg.orientation = _quaternion_msg(
                float(orientation.get("x", 0.0)),
                float(orientation.get("y", 0.0)),
                float(orientation.get("z", 0.0)),
                float(orientation.get("w", 1.0)),
            )
            msg.orientation_covariance[0] = 0.05
            msg.orientation_covariance[4] = 0.05
            msg.orientation_covariance[8] = 0.05
        self.imu_publisher.publish(msg)

    def _publish_scan(self, *, frame: RgbdFrame, stamp: Any) -> None:
        if frame.depth_be is not None and frame.depth_width is not None and frame.depth_height is not None:
            ranges, angles = synthesize_scan_from_depth_be(
                frame.depth_be,
                width=int(frame.depth_width),
                height=int(frame.depth_height),
                horizontal_fov_rad=self.config.orbbec.horizontal_fov_rad,
                band_height_px=self.config.scan_band_height_px,
                range_min_m=self.config.laser_min_range_m,
                range_max_m=self.config.laser_max_range_m,
                fill_no_return=self.config.laser_fill_no_return,
            )
        elif frame.depth_mm is not None:
            ranges, angles = synthesize_scan_from_depth_rows(
                frame.depth_mm,
                horizontal_fov_rad=self.config.orbbec.horizontal_fov_rad,
                band_height_px=self.config.scan_band_height_px,
                range_min_m=self.config.laser_min_range_m,
                range_max_m=self.config.laser_max_range_m,
                fill_no_return=self.config.laser_fill_no_return,
            )
        else:
            return
        msg = LaserScan()
        msg.header.stamp = stamp
        msg.header.frame_id = self.config.head_laser_frame
        msg.angle_min = angles[0]
        msg.angle_max = angles[-1]
        msg.angle_increment = angles[1] - angles[0] if len(angles) > 1 else 0.0
        msg.time_increment = 0.0
        msg.scan_time = 1.0 / max(self.config.publish_rate_hz, 1e-6)
        msg.range_min = self.config.laser_min_range_m
        msg.range_max = self.config.laser_max_range_m
        msg.ranges = list(ranges)
        self.scan_publisher.publish(msg)

    def _publish_head_images(self, *, frame: RgbdFrame, stamp: Any) -> None:
        if self.head_rgb_publisher is None or self.head_depth_publisher is None or self.head_camera_info_publisher is None:
            return
        if frame.rgb is not None and frame.rgb_width is not None and frame.rgb_height is not None:
            rgb_msg = Image()
            rgb_msg.header.stamp = stamp
            rgb_msg.header.frame_id = self.config.head_camera_frame
            rgb_msg.height = int(frame.rgb_height)
            rgb_msg.width = int(frame.rgb_width)
            rgb_msg.encoding = "rgb8"
            rgb_msg.is_bigendian = False
            rgb_msg.step = int(frame.rgb_width) * 3
            rgb_msg.data = frame.rgb
            self.head_rgb_publisher.publish(rgb_msg)
        if frame.depth_be is not None and frame.depth_width is not None and frame.depth_height is not None:
            depth_msg = Image()
            depth_msg.header.stamp = stamp
            depth_msg.header.frame_id = self.config.head_camera_frame
            depth_msg.height = int(frame.depth_height)
            depth_msg.width = int(frame.depth_width)
            depth_msg.encoding = "mono16"
            depth_msg.is_bigendian = True
            depth_msg.step = int(frame.depth_width) * 2
            depth_msg.data = frame.depth_be
            self.head_depth_publisher.publish(depth_msg)
        elif frame.depth_mm is not None and frame.depth_width is not None and frame.depth_height is not None:
            depth_msg = Image()
            depth_msg.header.stamp = stamp
            depth_msg.header.frame_id = self.config.head_camera_frame
            depth_msg.height = int(frame.depth_height)
            depth_msg.width = int(frame.depth_width)
            depth_msg.encoding = "mono16"
            depth_msg.is_bigendian = True
            depth_msg.step = int(frame.depth_width) * 2
            depth_msg.data = b"".join(
                int(value).to_bytes(2, "big", signed=False)
                for row in frame.depth_mm
                for value in row
            )
            self.head_depth_publisher.publish(depth_msg)
        width = frame.depth_width or frame.rgb_width
        height = frame.depth_height or frame.rgb_height
        if width is None or height is None:
            return
        camera_info = None
        if frame.metadata:
            camera_info = _build_camera_info_from_metadata(
                frame_id=self.config.head_camera_frame,
                width=int(width),
                height=int(height),
                metadata=frame.metadata,
            )
        if camera_info is None:
            camera_info = _build_camera_info(
                frame_id=self.config.head_camera_frame,
                width=int(width),
                height=int(height),
                horizontal_fov_rad=self.config.orbbec.horizontal_fov_rad,
            )
        camera_info.header.stamp = stamp
        self.head_camera_info_publisher.publish(camera_info)

    def _publish_head_points(self, *, frame: RgbdFrame, stamp: Any) -> None:
        if self.head_points_publisher is None:
            return
        if not self._head_points_publish_allowed(frame):
            return
        if (
            frame.point_cloud_format != POINT_CLOUD_FORMAT_XYZ_FLOAT32
            or frame.point_cloud_points is None
            or frame.point_cloud_count <= 0
            or frame.point_cloud_stride != 12
        ):
            return
        msg = PointCloud2()
        msg.header.stamp = stamp
        msg.header.frame_id = self.config.head_camera_frame
        msg.height = 1
        msg.width = int(frame.point_cloud_count)
        msg.fields = [
            _point_field("x", 0),
            _point_field("y", 4),
            _point_field("z", 8),
        ]
        msg.is_bigendian = False
        msg.point_step = int(frame.point_cloud_stride)
        msg.row_step = msg.point_step * msg.width
        msg.data = _orbbec_optical_xyz_to_ros_camera_link(frame.point_cloud_points, count=msg.width)
        msg.is_dense = False
        self.head_points_publisher.publish(msg)

    def _head_points_publish_allowed(self, frame: RgbdFrame) -> bool:
        if not self._head_points_update_map_enabled:
            self._log_head_points_skip_once("map updates disabled")
            return False
        if self._base_motion_active and not self.config.head_points_update_map_while_base_moving:
            self._log_head_points_skip_once("base moving")
            return False
        if self.config.head_points_only_during_scan and not self._scan_active:
            self._log_head_points_skip_once("scan inactive")
            return False
        if self.config.head_points_mode == "continuous":
            return True
        if self.config.head_points_mode != "settled":
            return True
        if not self._scan_active:
            self._last_head_points_skip_reason = ""
            return True
        if self._camera_pose_moving:
            self._log_head_points_skip_once("head moving")
            return False
        pose_updated_s = self._camera_pose_updated_s
        if pose_updated_s is None:
            self._log_head_points_skip_once("camera pose not ready")
            return False
        pose_received_s = self._camera_pose_received_s
        if pose_received_s is None:
            self._log_head_points_skip_once("camera pose not ready")
            return False
        age_s = time.monotonic() - pose_received_s
        if age_s < max(float(self.config.head_points_settled_delay_s), 0.0):
            self._log_head_points_skip_once("waiting for settled pose/frame")
            return False
        frame_timestamp_s = float(frame.timestamp_s)
        stale_tolerance_s = max(float(self.config.head_points_stale_tolerance_s), 0.0)
        if frame_timestamp_s + stale_tolerance_s < pose_updated_s:
            self._log_head_points_skip_once("discarding pre-settle point cloud")
            return False
        if self._last_head_points_skip_reason:
            self.get_logger().info(
                "Resuming settled /camera/head/points publication "
                f"pan_deg={math.degrees(self._camera_pan_rad):.1f} "
                f"pitch_deg={math.degrees(self._camera_pitch_rad):.1f}"
            )
            self._last_head_points_skip_reason = ""
        return True

    def _log_head_points_skip_once(self, reason: str) -> None:
        if reason == self._last_head_points_skip_reason:
            return
        self._last_head_points_skip_reason = reason
        self.get_logger().info(f"Suppressing /camera/head/points in settled mode: {reason}.")

    def close(self) -> None:
        try:
            self._imu_stream_stop.set()
            if self._imu_stream_thread is not None:
                self._imu_stream_thread.join(timeout=max(2.0, self.config.imu_ws_reconnect_delay_s + 1.0))
            try:
                self.runtime.stop()
            except Exception as exc:
                self.get_logger().warning(f"Failed to send stop during bridge shutdown: {_format_runtime_error(exc)}")
            finally:
                self.runtime.close()
        finally:
            self.destroy_node()


def _quaternion_msg(x: float, y: float, z: float, w: float) -> Any:
    msg = Quaternion()
    msg.x = float(x)
    msg.y = float(y)
    msg.z = float(z)
    msg.w = float(w)
    return msg


def _build_camera_info(*, frame_id: str, width: int, height: int, horizontal_fov_rad: float) -> Any:
    fx = width / (2.0 * math.tan(horizontal_fov_rad / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    msg = CameraInfo()
    msg.header.frame_id = frame_id
    msg.width = width
    msg.height = height
    msg.distortion_model = "plumb_bob"
    msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return msg


def _point_field(name: str, offset: int) -> Any:
    field = PointField()
    field.name = name
    field.offset = int(offset)
    field.datatype = PointField.FLOAT32
    field.count = 1
    return field


def _orbbec_optical_xyz_to_ros_camera_link(data: bytes, *, count: int) -> bytes:
    expected = int(count) * 12
    if len(data) < expected:
        raise ValueError(f"Point cloud payload is truncated: expected {expected} bytes, got {len(data)}.")
    converted = bytearray(expected)
    for index in range(int(count)):
        offset = index * 12
        optical_x, optical_y, optical_z = struct.unpack_from("<fff", data, offset)
        struct.pack_into("<fff", converted, offset, optical_z, -optical_x, -optical_y)
    return bytes(converted)


def _build_camera_info_from_metadata(
    *,
    frame_id: str,
    width: int,
    height: int,
    metadata: dict[str, Any],
) -> Any | None:
    intrinsics = metadata.get("camera_intrinsics")
    if not isinstance(intrinsics, dict):
        return None
    try:
        source_width = float(intrinsics.get("width") or width)
        source_height = float(intrinsics.get("height") or height)
        if source_width <= 0.0 or source_height <= 0.0:
            return None
        scale_x = float(width) / source_width
        scale_y = float(height) / source_height
        fx = float(intrinsics["fx"]) * scale_x
        fy = float(intrinsics["fy"]) * scale_y
        cx = float(intrinsics["cx"]) * scale_x
        cy = float(intrinsics["cy"]) * scale_y
    except (KeyError, TypeError, ValueError):
        return None
    msg = CameraInfo()
    msg.header.frame_id = frame_id
    msg.width = int(width)
    msg.height = int(height)
    msg.distortion_model = "plumb_bob"
    msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return msg


def _format_runtime_error(exc: Exception) -> str:
    if isinstance(exc, error.HTTPError):
        detail = str(exc.reason)
        try:
            body = exc.read().decode("utf-8", errors="replace").strip()
        except Exception:
            body = ""
        if body:
            detail = body
        return f"HTTP {exc.code}: {detail}"
    return str(exc)


def _angle_wrap(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bridge the real two-wheel XLeRobot into ROS 2/Nav2 topics.")
    parser.add_argument("--repo-root", default=str(resolve_xlerobot_repo_root()))
    parser.add_argument("--robot-kind", choices=("xlerobot", "xlerobot_2wheels"), default="xlerobot_2wheels")
    parser.add_argument("--port1", default="/dev/tty.usbmodem5B140330101")
    parser.add_argument("--port2", default="/dev/tty.usbmodem5B140332271")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--use-degrees", action="store_true")
    parser.add_argument("--allow-motion-commands", action="store_true")
    parser.add_argument("--max-linear-m-s", type=float, default=0.05)
    parser.add_argument("--max-angular-rad-s", type=float, default=0.20)
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument(
        "--odom-source",
        choices=("none", "commanded"),
        default="none",
        help=(
            "Use `none` for real exploration with camera/RGB-D odometry published by another node. "
            "Use `commanded` only for wheels-raised smoke tests."
        ),
    )
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--scan-topic", default="/scan")
    parser.add_argument("--imu-topic", default="/imu")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--odom-frame", default="odom")
    parser.add_argument("--head-camera-frame", default="head_camera_link")
    parser.add_argument("--head-points-topic", default="/camera/head/points")
    parser.add_argument(
        "--head-points-mode",
        choices=("continuous", "settled"),
        default="continuous",
        help=(
            "Use `continuous` for RViz/debug streaming. Use `settled` for OctoMap camera-pan scans: "
            "while scan-active is true, PointCloud2 is suppressed while the head is moving and only "
            "resumes after the pose settles."
        ),
    )
    parser.add_argument(
        "--head-points-settled-delay-s",
        type=float,
        default=0.20,
        help="Extra delay after robot_brain reports a settled head pose before publishing PointCloud2 in settled mode.",
    )
    parser.add_argument(
        "--head-points-stale-tolerance-s",
        type=float,
        default=0.10,
        help="Accept PointCloud2 frames this many seconds older than the settled camera pose timestamp.",
    )
    parser.add_argument(
        "--head-points-update-map-enabled-topic",
        default="/camera/head/points/update_map_enabled",
        help=(
            "std_msgs/Bool topic controlling whether /camera/head/points is published into the map pipeline. "
            "False pauses only PointCloud2 publication; odom, TF, images, IMU, and cmd_vel remain live."
        ),
    )
    parser.add_argument(
        "--head-points-update-map-while-base-moving",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow /camera/head/points publication while nonzero /cmd_vel is active. "
            "Default false keeps OctoMap from integrating point clouds during base motion "
            "while live motion mapping is still experimental."
        ),
    )
    parser.add_argument(
        "--head-points-only-during-scan",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Publish /camera/head/points only while scan-active is true. Use this for OctoMap validation "
            "runs where the map should change only during intentional scan windows."
        ),
    )
    parser.add_argument(
        "--scan-active-topic",
        default="/xlerobot/scan_active",
        help=(
            "std_msgs/Bool scan lifecycle topic. In settled head-points mode, pose-settle suppression "
            "is applied only while this topic is true."
        ),
    )
    parser.add_argument("--head-laser-frame", default="head_laser")
    parser.add_argument("--camera-x-m", type=float, default=0.24)
    parser.add_argument("--camera-y-m", type=float, default=0.0)
    parser.add_argument("--camera-z-m", type=float, default=0.35)
    parser.add_argument("--camera-yaw-rad", type=float, default=0.0)
    parser.add_argument("--camera-pitch-rad", type=float, default=0.0)
    parser.add_argument("--camera-pitch-topic", default="/camera/head/pitch_rad")
    parser.add_argument("--camera-pan-topic", default="/camera/head/pan_rad")
    parser.add_argument("--camera-pose-poll-period-s", type=float, default=0.2)
    parser.add_argument("--publish-head-camera", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--publish-imu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Publish raw /imu from the Orbbec stream. Disable in wheel-odometry-only mode.",
    )
    parser.add_argument("--publish-rate-hz", type=float, default=30.0)
    parser.add_argument(
        "--imu-publish-rate-hz",
        type=float,
        default=200.0,
        help="Local filesystem IMU poll rate. Remote robot_brain IMU uses websocket push instead.",
    )
    parser.add_argument("--imu-ws-path", default="/ws/imu")
    parser.add_argument("--imu-ws-reconnect-delay-s", type=float, default=1.0)
    parser.add_argument("--cmd-vel-timeout-s", type=float, default=0.5)
    parser.add_argument("--laser-min-range-m", type=float, default=0.05)
    parser.add_argument("--laser-max-range-m", type=float, default=6.0)
    parser.add_argument("--scan-band-height-px", type=int, default=12)
    parser.add_argument("--laser-fill-no-return", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--orbbec-output-dir", default="artifacts/orbbec_rgbd")
    parser.add_argument("--orbbec-rgb-filename", default="latest.ppm")
    parser.add_argument("--orbbec-depth-filename", default="latest_depth.pgm")
    parser.add_argument(
        "--orbbec-imu-filename",
        default="latest_imu.json",
        help="JSON file that carries the latest Orbbec IMU sample. Default uses latest_imu.json.",
    )
    parser.add_argument("--orbbec-horizontal-fov-rad", type=float, default=DEFAULT_ORBBEC_HORIZONTAL_FOV_RAD)
    parser.add_argument(
        "--robot-brain-url",
        default=None,
        help=(
            "HTTP URL for the non-ROS robot brain agent. When set, the ROS bridge "
            "fetches /rgb and /depth from that agent and forwards /cmd_vel to it."
        ),
    )
    return parser


def config_from_args(args: argparse.Namespace) -> RealRosBridgeConfig:
    return RealRosBridgeConfig(
        repo_root=args.repo_root,
        robot_kind=args.robot_kind,
        port1=args.port1,
        port2=args.port2,
        fps=args.fps,
        use_degrees=args.use_degrees,
        allow_motion_commands=args.allow_motion_commands,
        max_linear_m_s=args.max_linear_m_s,
        max_angular_rad_s=args.max_angular_rad_s,
        cmd_vel_topic=args.cmd_vel_topic,
        odom_source=args.odom_source,
        odom_topic=args.odom_topic,
        scan_topic=args.scan_topic,
        imu_topic=args.imu_topic,
        base_frame=args.base_frame,
        odom_frame=args.odom_frame,
        head_camera_frame=args.head_camera_frame,
        head_points_topic=args.head_points_topic,
        head_points_mode=args.head_points_mode,
        head_points_settled_delay_s=args.head_points_settled_delay_s,
        head_points_stale_tolerance_s=args.head_points_stale_tolerance_s,
        head_points_update_map_enabled_topic=args.head_points_update_map_enabled_topic,
        head_points_update_map_while_base_moving=args.head_points_update_map_while_base_moving,
        head_points_only_during_scan=args.head_points_only_during_scan,
        scan_active_topic=args.scan_active_topic,
        head_laser_frame=args.head_laser_frame,
        camera_x_m=args.camera_x_m,
        camera_y_m=args.camera_y_m,
        camera_z_m=args.camera_z_m,
        camera_yaw_rad=args.camera_yaw_rad,
        camera_pitch_rad=args.camera_pitch_rad,
        camera_pitch_topic=args.camera_pitch_topic,
        camera_pan_topic=args.camera_pan_topic,
        camera_pose_poll_period_s=args.camera_pose_poll_period_s,
        publish_head_camera=args.publish_head_camera,
        publish_imu=args.publish_imu,
        publish_rate_hz=args.publish_rate_hz,
        imu_publish_rate_hz=args.imu_publish_rate_hz,
        imu_ws_path=args.imu_ws_path,
        imu_ws_reconnect_delay_s=args.imu_ws_reconnect_delay_s,
        cmd_vel_timeout_s=args.cmd_vel_timeout_s,
        laser_min_range_m=args.laser_min_range_m,
        laser_max_range_m=args.laser_max_range_m,
        scan_band_height_px=args.scan_band_height_px,
        laser_fill_no_return=args.laser_fill_no_return,
        orbbec=OrbbecFilesystemConfig(
            output_dir=Path(args.orbbec_output_dir),
            rgb_filename=args.orbbec_rgb_filename,
            depth_filename=args.orbbec_depth_filename,
            imu_filename=args.orbbec_imu_filename,
            horizontal_fov_rad=args.orbbec_horizontal_fov_rad,
        ),
        robot_brain_url=args.robot_brain_url,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require_ros_dependencies()
    if args.robot_brain_url and args.publish_imu:
        require_aiohttp()
    rclpy.init()
    bridge = RealXLeRobotRosBridge(config_from_args(args))
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(bridge)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        bridge.close()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
