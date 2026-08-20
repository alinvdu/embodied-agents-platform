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
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from multido_xlerobot.bootstrap import resolve_xlerobot_repo_root
from multido_xlerobot.maniskill import bootstrap_xlerobot_maniskill

IMPORT_ERROR: Exception | None = None
try:
    import gymnasium as gym
    import rclpy
    from geometry_msgs.msg import Quaternion, TransformStamped, Twist
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import CameraInfo, Image, LaserScan
    from tf2_ros import TransformBroadcaster
except Exception as exc:  # pragma: no cover - exercised as a runtime guard.
    IMPORT_ERROR = exc
    gym = None
    rclpy = None
    Quaternion = None
    TransformStamped = None
    Twist = None
    Odometry = None
    Node = object
    Clock = None
    CameraInfo = None
    Image = None
    LaserScan = None
    TransformBroadcaster = None


HEAD_CAMERA_UID = "fetch_head"
HEAD_CAMERA_FOV_RAD = 1.6
HEAD_CAMERA_WIDTH = 256
HEAD_CAMERA_HEIGHT = 256
IDENTITY_QUATERNION_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


@dataclass(frozen=True)
class BridgeConfig:
    repo_root: str
    env_id: str = "SceneManipulation-v1"
    robot_uid: str = "xlerobot"
    control_mode: str = "pd_joint_delta_pos_dual_arm"
    render_mode: str | None = "rgb_array"
    shader: str = "default"
    sim_backend: str = "auto"
    num_envs: int = 1
    force_reload: bool = False
    cmd_vel_topic: str = "/cmd_vel"
    odom_topic: str = "/odom"
    scan_topic: str = "/scan"
    base_frame: str = "base_link"
    odom_frame: str = "odom"
    map_frame: str = "map"
    head_camera_frame: str = "head_camera_link"
    head_laser_frame: str = "head_laser"
    publish_head_camera: bool = True
    publish_rate_hz: float | None = None
    cmd_vel_timeout_s: float = 0.5
    linear_cmd_gain: float | None = None
    angular_cmd_gain: float | None = None
    laser_min_range_m: float = 0.05
    laser_max_range_m: float = 10.0
    scan_band_height_px: int = 12
    laser_fill_no_return: bool = True
    build_config_idx: int | None = None
    spawn_x: float | None = None
    spawn_y: float | None = None
    spawn_yaw: float = 0.0
    ros_base_yaw_offset_rad: float | None = None
    max_steps: int | None = None
    max_episode_steps: int | None = None
    realtime_factor: float = 1.0
    auto_reset: bool = False


def _require_runtime_dependencies() -> None:
    if IMPORT_ERROR is not None:
        raise RuntimeError(
            "The ManiSkill ROS bridge requires `rclpy`, `sensor_msgs`, `nav_msgs`, "
            "`tf2_ros`, and `gymnasium` in the active "
            "Python environment."
        ) from IMPORT_ERROR


def quaternion_wxyz_to_xyzw(quaternion: np.ndarray) -> Quaternion:
    msg = Quaternion()
    msg.x = float(quaternion[1])
    msg.y = float(quaternion[2])
    msg.z = float(quaternion[3])
    msg.w = float(quaternion[0])
    return msg


def normalize_quaternion_wxyz(quaternion: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quaternion))
    if norm <= 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quaternion / norm


def quaternion_inverse_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quaternion_wxyz(quaternion)
    return np.array([w, -x, -y, -z], dtype=np.float64)


def quaternion_multiply_wxyz(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = lhs
    rw, rx, ry, rz = rhs
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def rotate_vector_wxyz(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    pure = np.array([0.0, vector[0], vector[1], vector[2]], dtype=np.float64)
    rotated = quaternion_multiply_wxyz(
        quaternion_multiply_wxyz(quaternion, pure),
        quaternion_inverse_wxyz(quaternion),
    )
    return rotated[1:]


def relative_pose_wxyz(
    origin_position: np.ndarray,
    origin_quaternion: np.ndarray,
    current_position: np.ndarray,
    current_quaternion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    inverse_origin = quaternion_inverse_wxyz(origin_quaternion)
    translation = rotate_vector_wxyz(inverse_origin, current_position - origin_position)
    rotation = quaternion_multiply_wxyz(inverse_origin, current_quaternion)
    return translation, normalize_quaternion_wxyz(rotation)


def quaternion_to_yaw(quaternion: np.ndarray) -> float:
    w, x, y, z = quaternion
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def angle_wrap(angle: float) -> float:
    wrapped = (angle + math.pi) % (2.0 * math.pi) - math.pi
    if wrapped == -math.pi:
        return math.pi
    return wrapped


def quaternion_from_rotation_matrix(matrix: np.ndarray) -> np.ndarray:
    trace = float(matrix[0, 0] + matrix[1, 1] + matrix[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (matrix[2, 1] - matrix[1, 2]) / s
        y = (matrix[0, 2] - matrix[2, 0]) / s
        z = (matrix[1, 0] - matrix[0, 1]) / s
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / s
        x = 0.25 * s
        y = (matrix[0, 1] + matrix[1, 0]) / s
        z = (matrix[0, 2] + matrix[2, 0]) / s
    elif matrix[1, 1] > matrix[2, 2]:
        s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / s
        x = (matrix[0, 1] + matrix[1, 0]) / s
        y = 0.25 * s
        z = (matrix[1, 2] + matrix[2, 1]) / s
    else:
        s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / s
        x = (matrix[0, 2] + matrix[2, 0]) / s
        y = (matrix[1, 2] + matrix[2, 1]) / s
        z = 0.25 * s
    return normalize_quaternion_wxyz(np.array([w, x, y, z], dtype=np.float64))


def quaternion_from_yaw_wxyz(yaw: float) -> np.ndarray:
    half = yaw / 2.0
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


def build_camera_info(
    *,
    frame_id: str,
    width: int,
    height: int,
    horizontal_fov_rad: float,
) -> CameraInfo:
    fx = width / (2.0 * math.tan(horizontal_fov_rad / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0

    msg = CameraInfo()
    msg.header.frame_id = frame_id
    msg.width = width
    msg.height = height
    msg.distortion_model = "plumb_bob"
    msg.k = [
        fx,
        0.0,
        cx,
        0.0,
        fy,
        cy,
        0.0,
        0.0,
        1.0,
    ]
    msg.p = [
        fx,
        0.0,
        cx,
        0.0,
        0.0,
        fy,
        cy,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ]
    return msg


def image_message_from_array(array: np.ndarray, *, encoding: str, frame_id: str, sec: int, nanosec: int) -> Image:
    contiguous = np.ascontiguousarray(array)
    msg = Image()
    msg.header.stamp.sec = sec
    msg.header.stamp.nanosec = nanosec
    msg.header.frame_id = frame_id
    msg.height = int(contiguous.shape[0])
    msg.width = int(contiguous.shape[1])
    msg.encoding = encoding
    msg.is_bigendian = contiguous.dtype.byteorder == ">" or (
        contiguous.dtype.byteorder == "=" and sys.byteorder == "big"
    )
    if contiguous.ndim == 2:
        channels = 1
    else:
        channels = int(contiguous.shape[2])
    msg.step = int(contiguous.shape[1] * contiguous.dtype.itemsize * channels)
    msg.data = contiguous.tobytes()
    return msg


def synthesize_scan_from_depth(
    depth_mm: np.ndarray,
    *,
    horizontal_fov_rad: float,
    band_height_px: int,
    range_min_m: float,
    range_max_m: float,
    fill_no_return: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if depth_mm.ndim != 2:
        raise ValueError(f"Expected a 2D depth image, got shape {depth_mm.shape}")

    height, width = depth_mm.shape
    band_half = max(1, band_height_px // 2)
    center = height // 2
    row_start = max(0, center - band_half)
    row_stop = min(height, center + band_half)
    band = depth_mm[row_start:row_stop, :].astype(np.float32)
    valid = np.isfinite(band) & (band > 0.0)

    fx = width / (2.0 * math.tan(horizontal_fov_rad / 2.0))
    cx = width / 2.0
    image_columns = np.arange(width, dtype=np.float32)
    image_angles = -np.arctan2((image_columns - cx) / max(fx, 1e-6), 1.0)

    # ROS LaserScan consumers expect monotonically increasing angles. Image columns
    # are left->right while positive planar yaw is left of robot-forward, so publish
    # beam bins from image right to image left.
    angles = np.linspace(-horizontal_fov_rad / 2.0, horizontal_fov_rad / 2.0, width, dtype=np.float32)
    ranges = np.full(width, np.inf, dtype=np.float32)
    for column in range(width):
        column_depths_mm = band[:, column][valid[:, column]]
        if column_depths_mm.size == 0:
            continue
        depth_m = column_depths_mm.astype(np.float32) / 1000.0
        planar_range_m = depth_m / max(float(math.cos(float(image_angles[column]))), 1e-3)
        usable = planar_range_m[
            np.isfinite(planar_range_m)
            & (planar_range_m >= range_min_m)
            & (planar_range_m <= range_max_m)
        ]
        if usable.size == 0:
            continue
        scan_index = width - 1 - column
        ranges[scan_index] = float(np.min(usable))
    if fill_no_return:
        ranges = np.where(np.isinf(ranges), range_max_m, ranges)
    return ranges.astype(np.float32), angles


class ManiSkillRosBridge(Node):
    def __init__(self, config: BridgeConfig):
        _require_runtime_dependencies()
        super().__init__("xlerobot_maniskill_ros_bridge")
        self.config = config
        self.tf_broadcaster = TransformBroadcaster(self)
        self._latest_twist = Twist()
        self._latest_cmd_stamp = time.monotonic()
        self._sim_time_s = 0.0
        self._step_index = 0
        self._origin_position: np.ndarray | None = None
        self._origin_quaternion: np.ndarray | None = None
        self._last_odom_position: np.ndarray | None = None
        self._last_odom_yaw: float | None = None

        self.create_subscription(Twist, config.cmd_vel_topic, self._on_cmd_vel, 10)
        self.odom_publisher = self.create_publisher(Odometry, config.odom_topic, 10)
        self.scan_publisher = self.create_publisher(LaserScan, config.scan_topic, 10)
        self.clock_publisher = self.create_publisher(Clock, "/clock", 10)
        self.head_rgb_publisher = None
        self.head_depth_publisher = None
        self.head_camera_info_publisher = None
        if config.publish_head_camera:
            self.head_rgb_publisher = self.create_publisher(Image, "/camera/head/image_raw", 10)
            self.head_depth_publisher = self.create_publisher(Image, "/camera/head/depth/image_raw", 10)
            self.head_camera_info_publisher = self.create_publisher(CameraInfo, "/camera/head/camera_info", 10)

        self.camera_info = build_camera_info(
            frame_id=config.head_camera_frame,
            width=HEAD_CAMERA_WIDTH,
            height=HEAD_CAMERA_HEIGHT,
            horizontal_fov_rad=HEAD_CAMERA_FOV_RAD,
        )
        # The ManiSkill head camera is mounted with identity pose on `head_camera_link`.
        # Its frame already behaves like a planar navigation sensor frame for yaw projection,
        # so publishing an extra optical->laser rotation here introduces a false yaw bias.
        self._head_laser_quaternion = IDENTITY_QUATERNION_WXYZ.copy()
        yaw_offset = config.ros_base_yaw_offset_rad
        if yaw_offset is None:
            yaw_offset = math.pi if config.robot_uid == "xlerobot" else 0.0
        self._ros_base_yaw_offset_rad = float(yaw_offset)
        self._ros_base_rotation = quaternion_from_yaw_wxyz(self._ros_base_yaw_offset_rad)
        linear_cmd_gain = config.linear_cmd_gain
        angular_cmd_gain = config.angular_cmd_gain
        if linear_cmd_gain is None:
            linear_cmd_gain = 1.5 if config.robot_uid == "xlerobot" else 1.0
        if angular_cmd_gain is None:
            angular_cmd_gain = 0.32 if config.robot_uid == "xlerobot" else 1.0
        self._linear_cmd_gain = float(linear_cmd_gain)
        self._angular_cmd_gain = float(angular_cmd_gain)
        self._origin_ros_quaternion: np.ndarray | None = None

        bootstrap_xlerobot_maniskill(config.repo_root, force_reload=config.force_reload)
        env_kwargs: dict[str, Any] = {
            "obs_mode": "sensor_data",
            "control_mode": config.control_mode,
            "render_mode": config.render_mode,
            "sensor_configs": {"shader_pack": config.shader},
            "human_render_camera_configs": {"shader_pack": config.shader},
            "viewer_camera_configs": {"shader_pack": config.shader},
            "robot_uids": config.robot_uid,
            "num_envs": config.num_envs,
            "sim_backend": config.sim_backend,
            "enable_shadow": True,
            "parallel_in_single_scene": False,
        }
        if config.max_episode_steps is not None:
            env_kwargs["max_episode_steps"] = config.max_episode_steps
        self.env = gym.make(config.env_id, **env_kwargs)
        self._reset_environment()
        self.get_logger().info(
            "Bridge ready: "
            f"cmd_vel={config.cmd_vel_topic} "
            f"odom={config.odom_topic} "
            f"scan={config.scan_topic} "
            f"env={config.env_id} "
            f"robot={config.robot_uid} "
            f"ros_base_yaw_offset={self._ros_base_yaw_offset_rad:.3f} "
            f"linear_cmd_gain={self._linear_cmd_gain:.3f} "
            f"angular_cmd_gain={self._angular_cmd_gain:.3f} "
            f"control_dt={self.control_dt_s:.3f}"
        )

    @property
    def control_dt_s(self) -> float:
        return float(getattr(self.env.unwrapped, "control_timestep", 0.05))

    def _reset_environment(self) -> None:
        reset_options: dict[str, Any] = {"reconfigure": True}
        if self.config.build_config_idx is not None:
            reset_options["build_config_idxs"] = self.config.build_config_idx
        self.env.reset(seed=2022, options=reset_options)
        self.robot = self.env.unwrapped.agent.robot
        self.base_link = self.env.unwrapped.agent.base_link
        self.head_link = self.env.unwrapped.agent.head_camera_link
        self.scene = self.env.unwrapped.scene
        self.action = np.zeros(self.env.action_space.shape, dtype=np.float32)
        if self.config.spawn_x is not None and self.config.spawn_y is not None:
            import sapien

            self.robot.set_pose(
                sapien.Pose(
                    [self.config.spawn_x, self.config.spawn_y, 0.02],
                    quaternion_from_yaw_wxyz(self.config.spawn_yaw).tolist(),
                )
            )
        self._origin_position = self._tensor_to_numpy(self.base_link.pose.p)
        self._origin_quaternion = self._tensor_to_numpy(self.base_link.pose.q)
        self._origin_ros_quaternion = quaternion_multiply_wxyz(
            normalize_quaternion_wxyz(self._origin_quaternion),
            self._ros_base_rotation,
        )
        self._last_odom_position = np.zeros(3, dtype=np.float64)
        self._last_odom_yaw = 0.0
        self._latest_twist = Twist()
        self._latest_cmd_stamp = time.monotonic()
        self._publish_once()

    def _tensor_to_numpy(self, value: Any) -> np.ndarray:
        array = value.cpu().numpy() if hasattr(value, "cpu") else np.asarray(value)
        return np.asarray(array, dtype=np.float64).squeeze()

    def _on_cmd_vel(self, message: Twist) -> None:
        self._latest_twist = message
        self._latest_cmd_stamp = time.monotonic()

    def _active_twist(self) -> Twist:
        if time.monotonic() - self._latest_cmd_stamp > self.config.cmd_vel_timeout_s:
            return Twist()
        return self._latest_twist

    def _apply_cmd_vel(self) -> None:
        twist = self._active_twist()
        self.action.fill(0.0)
        linear_command_ros = np.array(
            [
                float(twist.linear.x) * self._linear_cmd_gain,
                float(twist.linear.y) * self._linear_cmd_gain,
                0.0,
            ],
            dtype=np.float64,
        )
        linear_command_sim = rotate_vector_wxyz(self._ros_base_rotation, linear_command_ros)
        self.action[0] = float(np.clip(linear_command_sim[0], -1.0, 1.0))
        self.action[1] = float(np.clip(twist.angular.z * self._angular_cmd_gain, -math.pi, math.pi))

    def _ros_time(self) -> tuple[int, int]:
        seconds = int(self._sim_time_s)
        nanoseconds = int((self._sim_time_s - seconds) * 1e9)
        return seconds, nanoseconds

    def _make_header(self, frame_id: str) -> tuple[int, int, str]:
        seconds, nanoseconds = self._ros_time()
        return seconds, nanoseconds, frame_id

    def _publish_clock(self) -> None:
        seconds, nanoseconds = self._ros_time()
        msg = Clock()
        msg.clock.sec = seconds
        msg.clock.nanosec = nanoseconds
        self.clock_publisher.publish(msg)

    def _publish_transforms_and_odom(self) -> None:
        assert self._origin_position is not None
        assert self._origin_quaternion is not None
        assert self._origin_ros_quaternion is not None
        assert self._last_odom_position is not None
        assert self._last_odom_yaw is not None

        base_position = self._tensor_to_numpy(self.base_link.pose.p)
        base_quaternion = normalize_quaternion_wxyz(self._tensor_to_numpy(self.base_link.pose.q))
        ros_base_quaternion = quaternion_multiply_wxyz(base_quaternion, self._ros_base_rotation)
        odom_position, odom_quaternion = relative_pose_wxyz(
            self._origin_position,
            self._origin_ros_quaternion,
            base_position,
            ros_base_quaternion,
        )
        current_yaw = quaternion_to_yaw(odom_quaternion)

        head_position = self._tensor_to_numpy(self.head_link.pose.p)
        head_quaternion = normalize_quaternion_wxyz(self._tensor_to_numpy(self.head_link.pose.q))
        head_relative_position, head_relative_quaternion = relative_pose_wxyz(
            base_position,
            ros_base_quaternion,
            head_position,
            head_quaternion,
        )

        linear_x = (odom_position[0] - self._last_odom_position[0]) / self.control_dt_s
        linear_y = (odom_position[1] - self._last_odom_position[1]) / self.control_dt_s
        angular_z = angle_wrap(current_yaw - self._last_odom_yaw) / self.control_dt_s
        self._last_odom_position = odom_position
        self._last_odom_yaw = current_yaw

        seconds, nanoseconds, _ = self._make_header(self.config.odom_frame)
        odom = Odometry()
        odom.header.stamp.sec = seconds
        odom.header.stamp.nanosec = nanoseconds
        odom.header.frame_id = self.config.odom_frame
        odom.child_frame_id = self.config.base_frame
        odom.pose.pose.position.x = float(odom_position[0])
        odom.pose.pose.position.y = float(odom_position[1])
        odom.pose.pose.position.z = float(odom_position[2])
        odom.pose.pose.orientation = quaternion_wxyz_to_xyzw(odom_quaternion)
        odom.twist.twist.linear.x = float(linear_x)
        odom.twist.twist.linear.y = float(linear_y)
        odom.twist.twist.angular.z = float(angular_z)
        self.odom_publisher.publish(odom)

        transforms: list[TransformStamped] = []

        odom_tf = TransformStamped()
        odom_tf.header.stamp.sec = seconds
        odom_tf.header.stamp.nanosec = nanoseconds
        odom_tf.header.frame_id = self.config.odom_frame
        odom_tf.child_frame_id = self.config.base_frame
        odom_tf.transform.translation.x = float(odom_position[0])
        odom_tf.transform.translation.y = float(odom_position[1])
        odom_tf.transform.translation.z = float(odom_position[2])
        odom_tf.transform.rotation = quaternion_wxyz_to_xyzw(odom_quaternion)
        transforms.append(odom_tf)

        camera_tf = TransformStamped()
        camera_tf.header.stamp.sec = seconds
        camera_tf.header.stamp.nanosec = nanoseconds
        camera_tf.header.frame_id = self.config.base_frame
        camera_tf.child_frame_id = self.config.head_camera_frame
        camera_tf.transform.translation.x = float(head_relative_position[0])
        camera_tf.transform.translation.y = float(head_relative_position[1])
        camera_tf.transform.translation.z = float(head_relative_position[2])
        camera_tf.transform.rotation = quaternion_wxyz_to_xyzw(head_relative_quaternion)
        transforms.append(camera_tf)

        laser_tf = TransformStamped()
        laser_tf.header.stamp.sec = seconds
        laser_tf.header.stamp.nanosec = nanoseconds
        laser_tf.header.frame_id = self.config.head_camera_frame
        laser_tf.child_frame_id = self.config.head_laser_frame
        laser_tf.transform.translation.x = 0.0
        laser_tf.transform.translation.y = 0.0
        laser_tf.transform.translation.z = 0.0
        laser_tf.transform.rotation = quaternion_wxyz_to_xyzw(self._head_laser_quaternion)
        transforms.append(laser_tf)

        self.tf_broadcaster.sendTransform(transforms)

    def _capture_head_camera(self) -> dict[str, np.ndarray]:
        self.scene.update_render(update_sensors=True, update_human_render_cameras=False)
        camera = self.scene.sensors[HEAD_CAMERA_UID]
        camera.capture()
        data = camera.get_obs(
            rgb=self.config.publish_head_camera,
            depth=True,
            position=False,
            segmentation=False,
        )
        converted: dict[str, np.ndarray] = {}
        for key, value in data.items():
            array = value.cpu().numpy() if hasattr(value, "cpu") else np.asarray(value)
            array = np.asarray(array).squeeze()
            converted[key] = array
        return converted

    def _publish_scan(self, head_data: dict[str, np.ndarray]) -> None:
        depth_image = head_data["depth"]
        if depth_image.ndim == 3:
            depth_image = depth_image[..., 0]
        ranges, angles = synthesize_scan_from_depth(
            depth_image,
            horizontal_fov_rad=HEAD_CAMERA_FOV_RAD,
            band_height_px=self.config.scan_band_height_px,
            range_min_m=self.config.laser_min_range_m,
            range_max_m=self.config.laser_max_range_m,
            fill_no_return=self.config.laser_fill_no_return,
        )

        seconds, nanoseconds, _ = self._make_header(self.config.head_laser_frame)
        msg = LaserScan()
        msg.header.stamp.sec = seconds
        msg.header.stamp.nanosec = nanoseconds
        msg.header.frame_id = self.config.head_laser_frame
        msg.angle_min = float(angles[0])
        msg.angle_max = float(angles[-1])
        msg.angle_increment = float(angles[1] - angles[0]) if len(angles) > 1 else 0.0
        msg.time_increment = 0.0
        msg.scan_time = self.control_dt_s
        msg.range_min = self.config.laser_min_range_m
        msg.range_max = self.config.laser_max_range_m
        msg.ranges = ranges.tolist()
        self.scan_publisher.publish(msg)

    def _publish_head_images(self, head_data: dict[str, np.ndarray]) -> None:
        if self.head_rgb_publisher is None or self.head_depth_publisher is None or self.head_camera_info_publisher is None:
            return

        seconds, nanoseconds, _ = self._make_header(self.config.head_camera_frame)
        stamp_sec = seconds
        stamp_nanosec = nanoseconds

        rgb = head_data.get("rgb")
        if rgb is not None:
            if rgb.ndim == 3 and rgb.shape[-1] == 3:
                rgb_msg = image_message_from_array(
                    rgb.astype(np.uint8),
                    encoding="rgb8",
                    frame_id=self.config.head_camera_frame,
                    sec=stamp_sec,
                    nanosec=stamp_nanosec,
                )
                self.head_rgb_publisher.publish(rgb_msg)

        depth = head_data.get("depth")
        if depth is not None:
            depth_2d = depth[..., 0] if depth.ndim == 3 else depth
            depth_msg = image_message_from_array(
                depth_2d.astype(np.uint16),
                encoding="mono16",
                frame_id=self.config.head_camera_frame,
                sec=stamp_sec,
                nanosec=stamp_nanosec,
            )
            self.head_depth_publisher.publish(depth_msg)

        camera_info = CameraInfo()
        camera_info.header.stamp.sec = stamp_sec
        camera_info.header.stamp.nanosec = stamp_nanosec
        camera_info.header.frame_id = self.config.head_camera_frame
        camera_info.width = self.camera_info.width
        camera_info.height = self.camera_info.height
        camera_info.distortion_model = self.camera_info.distortion_model
        camera_info.k = list(self.camera_info.k)
        camera_info.p = list(self.camera_info.p)
        self.head_camera_info_publisher.publish(camera_info)

    def _publish_once(self) -> None:
        self._publish_clock()
        self._publish_transforms_and_odom()
        head_data = self._capture_head_camera()
        self._publish_scan(head_data)
        self._publish_head_images(head_data)

    def step(self) -> bool:
        rclpy.spin_once(self, timeout_sec=0.0)
        self._apply_cmd_vel()
        _, _, terminated, truncated, _ = self.env.step(self.action)
        self._sim_time_s += self.control_dt_s
        self._step_index += 1
        self._publish_once()
        if self.config.render_mode == "human":
            self.env.render()

        if bool(np.asarray(terminated).any()) or bool(np.asarray(truncated).any()):
            if self.config.auto_reset:
                self.get_logger().warning("Environment ended, resetting bridge environment.")
                self._reset_environment()
                return True
            self.get_logger().warning("Environment ended and auto-reset is disabled. Exiting bridge.")
            return False

        if self.config.max_steps is not None and self._step_index >= self.config.max_steps:
            self.get_logger().info(f"Reached max-steps={self.config.max_steps}, exiting bridge.")
            return False

        if self.config.realtime_factor > 0.0:
            time.sleep(self.control_dt_s / self.config.realtime_factor)
        return True

    def close(self) -> None:
        try:
            self.env.close()
        finally:
            self.destroy_node()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bridge the XLeRobot ManiSkill simulation into ROS 2 topics that "
            "Nav2 and slam_toolbox can consume directly."
        )
    )
    parser.add_argument("--repo-root", default=str(resolve_xlerobot_repo_root()))
    parser.add_argument("--env-id", default="SceneManipulation-v1")
    parser.add_argument("--robot-uid", default="xlerobot")
    parser.add_argument("--control-mode", default="pd_joint_delta_pos_dual_arm")
    parser.add_argument("--render-mode", default="rgb_array")
    parser.add_argument("--shader", default="default")
    parser.add_argument("--sim-backend", default="auto")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--force-reload", action="store_true")
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--scan-topic", default="/scan")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--odom-frame", default="odom")
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--head-camera-frame", default="head_camera_link")
    parser.add_argument("--head-laser-frame", default="head_laser")
    parser.add_argument("--publish-head-camera", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cmd-vel-timeout-s", type=float, default=0.5)
    parser.add_argument("--linear-cmd-gain", type=float, default=None)
    parser.add_argument("--angular-cmd-gain", type=float, default=None)
    parser.add_argument("--laser-min-range-m", type=float, default=0.05)
    parser.add_argument("--laser-max-range-m", type=float, default=10.0)
    parser.add_argument("--scan-band-height-px", type=int, default=12)
    parser.add_argument("--laser-fill-no-return", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--build-config-idx", type=int, default=None)
    parser.add_argument("--spawn-x", type=float, default=None)
    parser.add_argument("--spawn-y", type=float, default=None)
    parser.add_argument("--spawn-yaw", type=float, default=0.0)
    parser.add_argument("--ros-base-yaw-offset-rad", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--realtime-factor", type=float, default=1.0)
    parser.add_argument("--auto-reset", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _require_runtime_dependencies()
    rclpy.init()
    bridge = ManiSkillRosBridge(
        BridgeConfig(
            repo_root=args.repo_root,
            env_id=args.env_id,
            robot_uid=args.robot_uid,
            control_mode=args.control_mode,
            render_mode=args.render_mode,
            shader=args.shader,
            sim_backend=args.sim_backend,
            num_envs=args.num_envs,
            force_reload=args.force_reload,
            cmd_vel_topic=args.cmd_vel_topic,
            odom_topic=args.odom_topic,
            scan_topic=args.scan_topic,
            base_frame=args.base_frame,
            odom_frame=args.odom_frame,
            map_frame=args.map_frame,
            head_camera_frame=args.head_camera_frame,
            head_laser_frame=args.head_laser_frame,
            publish_head_camera=args.publish_head_camera,
            cmd_vel_timeout_s=args.cmd_vel_timeout_s,
            linear_cmd_gain=args.linear_cmd_gain,
            angular_cmd_gain=args.angular_cmd_gain,
            laser_min_range_m=args.laser_min_range_m,
            laser_max_range_m=args.laser_max_range_m,
            scan_band_height_px=args.scan_band_height_px,
            laser_fill_no_return=args.laser_fill_no_return,
            build_config_idx=args.build_config_idx,
            spawn_x=args.spawn_x,
            spawn_y=args.spawn_y,
            spawn_yaw=args.spawn_yaw,
            ros_base_yaw_offset_rad=args.ros_base_yaw_offset_rad,
            max_steps=args.max_steps,
            max_episode_steps=args.max_episode_steps,
            realtime_factor=args.realtime_factor,
            auto_reset=args.auto_reset,
        )
    )
    try:
        while rclpy.ok():
            if not bridge.step():
                break
    finally:
        bridge.close()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
