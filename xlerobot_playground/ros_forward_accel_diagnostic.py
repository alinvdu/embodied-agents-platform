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
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

from xlerobot_playground.rgbd_visual_odometry import imu_to_base_components

try:
    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.time import Time as RosTime
    from sensor_msgs.msg import Imu
    from tf2_ros import Buffer, ConnectivityException, ExtrapolationException, LookupException, TransformListener
except Exception as exc:  # pragma: no cover - runtime guard for non-ROS test envs.
    IMPORT_ERROR: Exception | None = exc
    rclpy = None
    Twist = None
    Odometry = None
    Imu = None
    Node = object
    RosTime = None
    Buffer = None
    TransformListener = None
    ConnectivityException = Exception
    ExtrapolationException = Exception
    LookupException = Exception
else:
    IMPORT_ERROR = None


def require_ros() -> None:
    if IMPORT_ERROR is not None:
        raise RuntimeError("ROS forward accelerometer diagnostic requires ROS 2 Python packages.") from IMPORT_ERROR


def yaw_from_quaternion_xyzw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def tilt_from_acceleration(*, accel_x_m_s2: float, accel_y_m_s2: float, accel_z_m_s2: float) -> tuple[float, float]:
    lateral_norm = math.sqrt(float(accel_y_m_s2) ** 2 + float(accel_z_m_s2) ** 2)
    pitch_rad = math.atan2(float(accel_x_m_s2), max(lateral_norm, 1e-9))
    roll_rad = math.atan2(-float(accel_y_m_s2), float(accel_z_m_s2))
    return roll_rad, pitch_rad


def gravity_components_from_tilt(*, roll_rad: float, pitch_rad: float, gravity_m_s2: float) -> tuple[float, float, float]:
    cos_roll = math.cos(float(roll_rad))
    sin_roll = math.sin(float(roll_rad))
    cos_pitch = math.cos(float(pitch_rad))
    sin_pitch = math.sin(float(pitch_rad))
    return (
        float(gravity_m_s2) * sin_pitch,
        -float(gravity_m_s2) * sin_roll * cos_pitch,
        float(gravity_m_s2) * cos_roll * cos_pitch,
    )


def update_tilt_estimate(
    *,
    previous_roll_rad: float | None,
    previous_pitch_rad: float | None,
    gyro_roll_rate_rad_s: float,
    gyro_pitch_rate_rad_s: float,
    accel_x_m_s2: float,
    accel_y_m_s2: float,
    accel_z_m_s2: float,
    dt_s: float,
    accel_correction_alpha: float,
) -> tuple[float, float]:
    accel_roll_rad, accel_pitch_rad = tilt_from_acceleration(
        accel_x_m_s2=accel_x_m_s2,
        accel_y_m_s2=accel_y_m_s2,
        accel_z_m_s2=accel_z_m_s2,
    )
    if previous_roll_rad is None or previous_pitch_rad is None or dt_s <= 1e-6:
        return accel_roll_rad, accel_pitch_rad
    alpha = max(0.0, min(float(accel_correction_alpha), 1.0))
    predicted_roll_rad = float(previous_roll_rad) + float(gyro_roll_rate_rad_s) * float(dt_s)
    predicted_pitch_rad = float(previous_pitch_rad) + float(gyro_pitch_rate_rad_s) * float(dt_s)
    return (
        (1.0 - alpha) * predicted_roll_rad + alpha * accel_roll_rad,
        (1.0 - alpha) * predicted_pitch_rad + alpha * accel_pitch_rad,
    )


def tilt_correction_alpha_for_motion(
    *,
    stationary_alpha: float,
    commanded_linear_m_s: float,
    moving_alpha: float = 0.0,
) -> float:
    if abs(float(commanded_linear_m_s)) > 1e-6:
        return max(0.0, min(float(moving_alpha), 1.0))
    return max(0.0, min(float(stationary_alpha), 1.0))

def stationary_state_from_imu(
    *,
    accel_x_m_s2: float,
    accel_y_m_s2: float,
    accel_z_m_s2: float,
    gyro_roll_rate_rad_s: float,
    gyro_pitch_rate_rad_s: float,
    gyro_yaw_rate_rad_s: float,
    gravity_m_s2: float,
    accel_norm_tolerance_m_s2: float,
    gyro_norm_tolerance_rad_s: float,
) -> tuple[bool, float, float]:
    accel_norm_m_s2 = math.sqrt(
        float(accel_x_m_s2) ** 2 + float(accel_y_m_s2) ** 2 + float(accel_z_m_s2) ** 2
    )
    gyro_norm_rad_s = math.sqrt(
        float(gyro_roll_rate_rad_s) ** 2
        + float(gyro_pitch_rate_rad_s) ** 2
        + float(gyro_yaw_rate_rad_s) ** 2
    )
    is_stationary = (
        abs(accel_norm_m_s2 - float(gravity_m_s2)) <= max(float(accel_norm_tolerance_m_s2), 0.0)
        and gyro_norm_rad_s <= max(float(gyro_norm_tolerance_rad_s), 0.0)
    )
    return is_stationary, accel_norm_m_s2, gyro_norm_rad_s


def integrate_acceleration_step(
    *,
    distance_m: float,
    velocity_m_s: float,
    acceleration_m_s2: float,
    dt_s: float,
    acceleration_deadband_m_s2: float,
    velocity_damping_per_s: float,
    max_velocity_m_s: float,
) -> tuple[float, float, float]:
    dt_s = max(float(dt_s), 0.0)
    if dt_s <= 0.0:
        return float(distance_m), float(velocity_m_s), 0.0
    corrected_accel = float(acceleration_m_s2)
    if abs(corrected_accel) < max(float(acceleration_deadband_m_s2), 0.0):
        corrected_accel = 0.0
    damping = max(0.0, 1.0 - max(float(velocity_damping_per_s), 0.0) * dt_s)
    next_velocity = (float(velocity_m_s) + corrected_accel * dt_s) * damping
    max_velocity_m_s = max(float(max_velocity_m_s), 1e-6)
    next_velocity = clamp(next_velocity, -max_velocity_m_s, max_velocity_m_s)
    next_distance = float(distance_m) + (float(velocity_m_s) + next_velocity) * 0.5 * dt_s
    return next_distance, next_velocity, corrected_accel


def forward_displacement_m(
    *,
    start_x_m: float,
    start_y_m: float,
    start_yaw_rad: float,
    current_x_m: float,
    current_y_m: float,
) -> tuple[float, float]:
    dx = float(current_x_m) - float(start_x_m)
    dy = float(current_y_m) - float(start_y_m)
    cos_yaw = math.cos(float(start_yaw_rad))
    sin_yaw = math.sin(float(start_yaw_rad))
    forward_m = dx * cos_yaw + dy * sin_yaw
    lateral_m = -dx * sin_yaw + dy * cos_yaw
    return forward_m, lateral_m


def target_signed_progress_m(
    *,
    sample: dict[str, Any],
    target_source: str,
    accelerometer_distance_m: float,
    target_direction_sign: float,
) -> float | None:
    normalized_source = str(target_source).strip().lower()
    if normalized_source == "accelerometer":
        return float(accelerometer_distance_m) * float(target_direction_sign)
    distance_key = f"{normalized_source}_forward_distance_m"
    if distance_key not in sample:
        return None
    return float(sample[distance_key]) * float(target_direction_sign)


def _format_progress_value(value: Any, *, suffix: str = "", precision: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{precision}f}{suffix}"
    except (TypeError, ValueError):
        return "n/a"


def format_progress_log_line(
    sample: dict[str, Any],
    *,
    target_source: str,
    target_distance_m: float | None,
) -> str:
    normalized_target_source = str(target_source).strip().lower()
    target_progress = sample.get("target_signed_progress_m")
    if target_progress is None:
        target_progress = sample.get(f"{normalized_target_source}_forward_distance_m")
    target_label = f"target({normalized_target_source})={_format_progress_value(target_progress, suffix='m')}"
    if target_distance_m is not None:
        target_label += f"/{abs(float(target_distance_m)):.3f}m"
    if "target_remaining_distance_m" in sample:
        target_label += f" remaining={_format_progress_value(sample.get('target_remaining_distance_m'), suffix='m')}"
    elif "target_unavailable_reason" in sample:
        target_label += f" unavailable={sample['target_unavailable_reason']}"
    return (
        f"[forward_accel_diag] progress t={_format_progress_value(sample.get('t_s'), suffix='s', precision=2)} "
        f"{target_label} "
        f"tf={_format_progress_value(sample.get('tf_forward_distance_m'), suffix='m')} "
        f"odom={_format_progress_value(sample.get('odom_forward_distance_m'), suffix='m')} "
        f"raw_accel={_format_progress_value(sample.get('imu_estimated_forward_distance_m'), suffix='m')} "
        f"accel_vel={_format_progress_value(sample.get('imu_estimated_forward_velocity_m_s'), suffix='m/s')} "
        f"cmd={_format_progress_value(sample.get('cmd_linear_m_s'), suffix='m/s')} "
        f"imu_age={_format_progress_value(sample.get('imu_wall_age_s'), suffix='s')}"
    )


class ForwardAccelDiagnosticNode(Node):
    def __init__(
        self,
        *,
        odom_frame: str,
        base_frame: str,
        cmd_vel_topic: str,
        odom_topic: str,
        imu_topic: str,
        imu_frame_convention: str,
        sample_hz: float,
        accel_bias_calibration_s: float,
        accel_bias_min_samples: int,
        acceleration_deadband_m_s2: float,
        velocity_damping_per_s: float,
        max_estimated_velocity_m_s: float,
        tilt_accel_correction_alpha: float,
        tilt_accel_correction_alpha_when_moving: float,
        gravity_m_s2: float,
        accel_stationary_tolerance_m_s2: float,
        gyro_stationary_tolerance_rad_s: float,
        enable_zupt: bool,
        allow_zupt_while_commanded_motion: bool,
        max_imu_staleness_s: float,
    ) -> None:
        super().__init__("xlerobot_forward_accel_diagnostic")
        self.odom_frame = odom_frame
        self.base_frame = base_frame
        self.imu_frame_convention = str(imu_frame_convention).strip().lower()
        self.sample_dt_s = 1.0 / max(float(sample_hz), 1e-6)
        self.accel_bias_calibration_s = max(float(accel_bias_calibration_s), 0.0)
        self.accel_bias_min_samples = max(int(accel_bias_min_samples), 1)
        self.acceleration_deadband_m_s2 = max(float(acceleration_deadband_m_s2), 0.0)
        self.velocity_damping_per_s = max(float(velocity_damping_per_s), 0.0)
        self.max_estimated_velocity_m_s = max(float(max_estimated_velocity_m_s), 1e-6)
        self.tilt_accel_correction_alpha = max(0.0, min(float(tilt_accel_correction_alpha), 1.0))
        self.tilt_accel_correction_alpha_when_moving = max(
            0.0, min(float(tilt_accel_correction_alpha_when_moving), 1.0)
        )
        self.gravity_m_s2 = max(float(gravity_m_s2), 1e-6)
        self.accel_stationary_tolerance_m_s2 = max(float(accel_stationary_tolerance_m_s2), 0.0)
        self.gyro_stationary_tolerance_rad_s = max(float(gyro_stationary_tolerance_rad_s), 0.0)
        self.enable_zupt = bool(enable_zupt)
        self.allow_zupt_while_commanded_motion = bool(allow_zupt_while_commanded_motion)
        self.max_imu_staleness_s = max(float(max_imu_staleness_s), 0.0)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)
        self.cmd_vel_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.latest_odom_pose: dict[str, float] | None = None
        self.latest_imu_sample: dict[str, float] | None = None
        self.accel_bias_forward_m_s2 = 0.0
        self.accel_bias_lateral_m_s2 = 0.0
        self.accel_bias_vertical_m_s2 = 0.0
        self.gyro_bias_roll_rad_s = 0.0
        self.gyro_bias_pitch_rad_s = 0.0
        self.gyro_bias_yaw_rad_s = 0.0
        self._run_bias_applied = False
        self._run_active = False
        self._run_started_wall_s: float | None = None
        self._integration_last_stamp_s: float | None = None
        self._integration_first_stamp_s: float | None = None
        self._integration_last_wall_s: float | None = None
        self._integration_sample_count = 0
        self._integration_rejected_nonmonotonic_count = 0
        self._estimated_distance_m = 0.0
        self._estimated_velocity_m_s = 0.0
        self._estimated_roll_rad: float | None = None
        self._estimated_pitch_rad: float | None = None
        self._latest_integration_sample: dict[str, Any] | None = None
        self._commanded_linear_m_s = 0.0
        self.create_subscription(Odometry, odom_topic, self._on_odom, 10)
        self.create_subscription(Imu, imu_topic, self._on_imu, 200)

    def _reset_run_state(self, *, accel_bias: dict[str, float] | None) -> None:
        self._run_bias_applied = accel_bias is not None
        self._run_active = True
        self._run_started_wall_s = time.monotonic()
        self._integration_last_stamp_s = None
        self._integration_first_stamp_s = None
        self._integration_last_wall_s = None
        self._integration_sample_count = 0
        self._integration_rejected_nonmonotonic_count = 0
        self._estimated_distance_m = 0.0
        self._estimated_velocity_m_s = 0.0
        self._estimated_roll_rad = None if accel_bias is None else float(accel_bias["tilt_roll_rad"])
        self._estimated_pitch_rad = None if accel_bias is None else float(accel_bias["tilt_pitch_rad"])
        self._latest_integration_sample = None
        self._commanded_linear_m_s = 0.0

    def _integrate_imu_sample(self, imu_sample: dict[str, float]) -> None:
        if not self._run_active:
            return
        imu_stamp_s = float(imu_sample["timestamp_s"])
        if self._integration_last_stamp_s is not None and imu_stamp_s <= self._integration_last_stamp_s:
            self._integration_rejected_nonmonotonic_count += 1
            return
        dt_s = (
            self.sample_dt_s
            if self._integration_last_stamp_s is None
            else max(0.0, imu_stamp_s - self._integration_last_stamp_s)
        )
        self._integration_last_stamp_s = imu_stamp_s
        if self._integration_first_stamp_s is None:
            self._integration_first_stamp_s = imu_stamp_s
        self._integration_last_wall_s = float(imu_sample.get("received_monotonic_s", time.monotonic()))
        self._integration_sample_count += 1
        raw_forward_m_s2 = float(imu_sample["base_linear_acceleration_x_m_s2"])
        raw_lateral_m_s2 = float(imu_sample["base_linear_acceleration_y_m_s2"])
        raw_vertical_m_s2 = float(imu_sample["base_linear_acceleration_z_m_s2"])
        raw_gyro_roll_rad_s = float(imu_sample["base_angular_velocity_x_rad_s"])
        raw_gyro_pitch_rad_s = float(imu_sample["base_angular_velocity_y_rad_s"])
        raw_gyro_yaw_rad_s = float(imu_sample["base_angular_velocity_z_rad_s"])
        corrected_gyro_roll_rad_s = raw_gyro_roll_rad_s - self.gyro_bias_roll_rad_s
        corrected_gyro_pitch_rad_s = raw_gyro_pitch_rad_s - self.gyro_bias_pitch_rad_s
        corrected_gyro_yaw_rad_s = raw_gyro_yaw_rad_s - self.gyro_bias_yaw_rad_s
        is_stationary, accel_norm_m_s2, gyro_norm_rad_s = stationary_state_from_imu(
            accel_x_m_s2=raw_forward_m_s2,
            accel_y_m_s2=raw_lateral_m_s2,
            accel_z_m_s2=raw_vertical_m_s2,
            gyro_roll_rate_rad_s=corrected_gyro_roll_rad_s,
            gyro_pitch_rate_rad_s=corrected_gyro_pitch_rad_s,
            gyro_yaw_rate_rad_s=corrected_gyro_yaw_rad_s,
            gravity_m_s2=self.gravity_m_s2,
            accel_norm_tolerance_m_s2=self.accel_stationary_tolerance_m_s2,
            gyro_norm_tolerance_rad_s=self.gyro_stationary_tolerance_rad_s,
        )
        tilt_alpha = (
            tilt_correction_alpha_for_motion(
                stationary_alpha=self.tilt_accel_correction_alpha,
                commanded_linear_m_s=self._commanded_linear_m_s,
                moving_alpha=self.tilt_accel_correction_alpha_when_moving,
            )
            if is_stationary
            else self.tilt_accel_correction_alpha_when_moving
        )
        self._estimated_roll_rad, self._estimated_pitch_rad = update_tilt_estimate(
            previous_roll_rad=self._estimated_roll_rad,
            previous_pitch_rad=self._estimated_pitch_rad,
            gyro_roll_rate_rad_s=corrected_gyro_roll_rad_s,
            gyro_pitch_rate_rad_s=corrected_gyro_pitch_rad_s,
            accel_x_m_s2=raw_forward_m_s2,
            accel_y_m_s2=raw_lateral_m_s2,
            accel_z_m_s2=raw_vertical_m_s2,
            dt_s=dt_s,
            accel_correction_alpha=tilt_alpha,
        )
        gravity_forward_m_s2, gravity_lateral_m_s2, gravity_vertical_m_s2 = gravity_components_from_tilt(
            roll_rad=self._estimated_roll_rad,
            pitch_rad=self._estimated_pitch_rad,
            gravity_m_s2=self.gravity_m_s2,
        )
        compensated_forward_m_s2 = raw_forward_m_s2 - gravity_forward_m_s2
        compensated_lateral_m_s2 = raw_lateral_m_s2 - gravity_lateral_m_s2
        compensated_vertical_m_s2 = raw_vertical_m_s2 - gravity_vertical_m_s2
        corrected_forward_m_s2 = compensated_forward_m_s2 - self.accel_bias_forward_m_s2
        zupt_applied = self.enable_zupt and is_stationary and (
            self.allow_zupt_while_commanded_motion or abs(self._commanded_linear_m_s) <= 1e-6
        )
        if zupt_applied:
            self._estimated_velocity_m_s = 0.0
            used_forward_m_s2 = 0.0
        else:
            (
                self._estimated_distance_m,
                self._estimated_velocity_m_s,
                used_forward_m_s2,
            ) = integrate_acceleration_step(
                distance_m=self._estimated_distance_m,
                velocity_m_s=self._estimated_velocity_m_s,
                acceleration_m_s2=corrected_forward_m_s2,
                dt_s=dt_s,
                acceleration_deadband_m_s2=self.acceleration_deadband_m_s2,
                velocity_damping_per_s=self.velocity_damping_per_s,
                max_velocity_m_s=self.max_estimated_velocity_m_s,
            )
        self._latest_integration_sample = {
            "imu_available": True,
            "imu_timestamp_s": imu_stamp_s,
            "imu_dt_s": dt_s,
            "imu_integration_sample_count": self._integration_sample_count,
            "imu_integration_first_timestamp_s": self._integration_first_stamp_s,
            "imu_integration_rejected_nonmonotonic_count": self._integration_rejected_nonmonotonic_count,
            "imu_raw_linear_acceleration_x_m_s2": float(imu_sample["linear_acceleration_x_m_s2"]),
            "imu_raw_linear_acceleration_y_m_s2": float(imu_sample["linear_acceleration_y_m_s2"]),
            "imu_raw_linear_acceleration_z_m_s2": float(imu_sample["linear_acceleration_z_m_s2"]),
            "imu_base_linear_acceleration_x_m_s2": raw_forward_m_s2,
            "imu_base_linear_acceleration_y_m_s2": raw_lateral_m_s2,
            "imu_base_linear_acceleration_z_m_s2": raw_vertical_m_s2,
            "imu_base_angular_velocity_roll_rad_s": raw_gyro_roll_rad_s,
            "imu_base_angular_velocity_pitch_rad_s": raw_gyro_pitch_rad_s,
            "imu_base_angular_velocity_yaw_rad_s": raw_gyro_yaw_rad_s,
            "imu_corrected_gyro_roll_rad_s": corrected_gyro_roll_rad_s,
            "imu_corrected_gyro_pitch_rad_s": corrected_gyro_pitch_rad_s,
            "imu_corrected_gyro_yaw_rad_s": corrected_gyro_yaw_rad_s,
            "imu_estimated_roll_rad": self._estimated_roll_rad,
            "imu_estimated_pitch_rad": self._estimated_pitch_rad,
            "imu_tilt_accel_correction_alpha": tilt_alpha,
            "imu_stationary": is_stationary,
            "imu_zupt_applied": zupt_applied,
            "imu_commanded_linear_m_s": self._commanded_linear_m_s,
            "imu_accel_norm_m_s2": accel_norm_m_s2,
            "imu_gyro_norm_rad_s": gyro_norm_rad_s,
            "imu_gravity_forward_m_s2": gravity_forward_m_s2,
            "imu_gravity_lateral_m_s2": gravity_lateral_m_s2,
            "imu_gravity_vertical_m_s2": gravity_vertical_m_s2,
            "imu_compensated_forward_acceleration_m_s2": compensated_forward_m_s2,
            "imu_compensated_lateral_acceleration_m_s2": compensated_lateral_m_s2,
            "imu_compensated_vertical_acceleration_m_s2": compensated_vertical_m_s2,
            "imu_corrected_forward_acceleration_m_s2": corrected_forward_m_s2,
            "imu_used_forward_acceleration_m_s2": used_forward_m_s2,
            "imu_estimated_forward_velocity_m_s": self._estimated_velocity_m_s,
            "imu_estimated_forward_distance_m": self._estimated_distance_m,
        }

    def _on_odom(self, message: Any) -> None:
        position = message.pose.pose.position
        rotation = message.pose.pose.orientation
        self.latest_odom_pose = {
            "x": float(position.x),
            "y": float(position.y),
            "yaw_rad": yaw_from_quaternion_xyzw(rotation.x, rotation.y, rotation.z, rotation.w),
        }

    def _on_imu(self, message: Any) -> None:
        stamp = getattr(message, "header", None)
        stamp_s = time.time()
        if stamp is not None:
            stamp_s = float(getattr(stamp.stamp, "sec", 0)) + float(getattr(stamp.stamp, "nanosec", 0)) / 1_000_000_000.0
        linear_x = float(message.linear_acceleration.x)
        linear_y = float(message.linear_acceleration.y)
        linear_z = float(message.linear_acceleration.z)
        base_x, base_y, base_z = imu_to_base_components(
            frame_convention=self.imu_frame_convention,
            x=linear_x,
            y=linear_y,
            z=linear_z,
        )
        angular_x = float(message.angular_velocity.x)
        angular_y = float(message.angular_velocity.y)
        angular_z = float(message.angular_velocity.z)
        base_gyro_x, base_gyro_y, base_gyro_z = imu_to_base_components(
            frame_convention=self.imu_frame_convention,
            x=angular_x,
            y=angular_y,
            z=angular_z,
        )
        self.latest_imu_sample = {
            "timestamp_s": stamp_s,
            "received_monotonic_s": time.monotonic(),
            "linear_acceleration_x_m_s2": linear_x,
            "linear_acceleration_y_m_s2": linear_y,
            "linear_acceleration_z_m_s2": linear_z,
            "angular_velocity_x_rad_s": angular_x,
            "angular_velocity_y_rad_s": angular_y,
            "angular_velocity_z_rad_s": angular_z,
            "base_linear_acceleration_x_m_s2": float(base_x),
            "base_linear_acceleration_y_m_s2": float(base_y),
            "base_linear_acceleration_z_m_s2": float(base_z),
            "base_angular_velocity_x_rad_s": float(base_gyro_x),
            "base_angular_velocity_y_rad_s": float(base_gyro_y),
            "base_angular_velocity_z_rad_s": float(base_gyro_z),
        }
        self._integrate_imu_sample(self.latest_imu_sample)

    def imu_staleness_s(self, *, now_s: float | None = None) -> float | None:
        if self._integration_last_wall_s is None:
            return None
        if now_s is None:
            now_s = time.monotonic()
        return max(float(now_s) - self._integration_last_wall_s, 0.0)

    def imu_is_fresh(self, *, now_s: float | None = None) -> bool:
        if self.max_imu_staleness_s <= 1e-9:
            return self._integration_sample_count > 0
        age_s = self.imu_staleness_s(now_s=now_s)
        return age_s is not None and age_s <= self.max_imu_staleness_s

    def lookup_pose(self) -> dict[str, float] | None:
        try:
            transform = self.tf_buffer.lookup_transform(self.odom_frame, self.base_frame, RosTime())
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return {
            "x": float(translation.x),
            "y": float(translation.y),
            "yaw_rad": yaw_from_quaternion_xyzw(rotation.x, rotation.y, rotation.z, rotation.w),
        }

    def publish_forward(self, linear_m_s: float) -> None:
        self._commanded_linear_m_s = float(linear_m_s)
        twist = Twist()
        twist.linear.x = float(linear_m_s)
        self.cmd_vel_pub.publish(twist)

    def stop(self) -> None:
        self._commanded_linear_m_s = 0.0
        self.cmd_vel_pub.publish(Twist())

    def calibrate_accel_bias(self) -> dict[str, float] | None:
        print(
            f"[forward_accel_diag] Starting accel bias calibration for {self.accel_bias_calibration_s:.2f}s "
            f"(min_samples={self.accel_bias_min_samples})",
            flush=True,
        )
        if self.accel_bias_calibration_s <= 1e-6:
            print("[forward_accel_diag] Accel bias calibration disabled; using zero bias.", flush=True)
            return {
                "bias_forward_m_s2": self.accel_bias_forward_m_s2,
                "bias_lateral_m_s2": self.accel_bias_lateral_m_s2,
                "bias_vertical_m_s2": self.accel_bias_vertical_m_s2,
                "gyro_bias_roll_rad_s": self.gyro_bias_roll_rad_s,
                "gyro_bias_pitch_rad_s": self.gyro_bias_pitch_rad_s,
                "gyro_bias_yaw_rad_s": self.gyro_bias_yaw_rad_s,
                "tilt_roll_rad": 0.0,
                "tilt_pitch_rad": 0.0,
                "sample_count": 0,
                "elapsed_s": 0.0,
            }
        start = time.time()
        deadline = start + self.accel_bias_calibration_s
        sum_forward = 0.0
        sum_lateral = 0.0
        sum_vertical = 0.0
        sum_gyro_roll = 0.0
        sum_gyro_pitch = 0.0
        sum_gyro_yaw = 0.0
        sum_roll_rad = 0.0
        sum_pitch_rad = 0.0
        count = 0
        last_timestamp_s: float | None = None
        estimated_roll_rad: float | None = None
        estimated_pitch_rad: float | None = None
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=min(self.sample_dt_s, 0.05))
            imu_sample = self.latest_imu_sample
            if imu_sample is None:
                continue
            timestamp_s = float(imu_sample["timestamp_s"])
            if last_timestamp_s is not None and timestamp_s <= last_timestamp_s:
                continue
            dt_s = self.sample_dt_s if last_timestamp_s is None else max(0.0, timestamp_s - last_timestamp_s)
            raw_base_forward = float(imu_sample["base_linear_acceleration_x_m_s2"])
            raw_base_lateral = float(imu_sample["base_linear_acceleration_y_m_s2"])
            raw_base_vertical = float(imu_sample["base_linear_acceleration_z_m_s2"])
            raw_base_gyro_roll = float(imu_sample["base_angular_velocity_x_rad_s"])
            raw_base_gyro_pitch = float(imu_sample["base_angular_velocity_y_rad_s"])
            raw_base_gyro_yaw = float(imu_sample["base_angular_velocity_z_rad_s"])
            estimated_roll_rad, estimated_pitch_rad = update_tilt_estimate(
                previous_roll_rad=estimated_roll_rad,
                previous_pitch_rad=estimated_pitch_rad,
                gyro_roll_rate_rad_s=raw_base_gyro_roll,
                gyro_pitch_rate_rad_s=raw_base_gyro_pitch,
                accel_x_m_s2=raw_base_forward,
                accel_y_m_s2=raw_base_lateral,
                accel_z_m_s2=raw_base_vertical,
                dt_s=dt_s,
                accel_correction_alpha=self.tilt_accel_correction_alpha,
            )
            gravity_forward, gravity_lateral, gravity_vertical = gravity_components_from_tilt(
                roll_rad=estimated_roll_rad,
                pitch_rad=estimated_pitch_rad,
                gravity_m_s2=self.gravity_m_s2,
            )
            last_timestamp_s = timestamp_s
            sum_forward += raw_base_forward - gravity_forward
            sum_lateral += raw_base_lateral - gravity_lateral
            sum_vertical += raw_base_vertical - gravity_vertical
            sum_gyro_roll += raw_base_gyro_roll
            sum_gyro_pitch += raw_base_gyro_pitch
            sum_gyro_yaw += raw_base_gyro_yaw
            sum_roll_rad += estimated_roll_rad
            sum_pitch_rad += estimated_pitch_rad
            count += 1
        if count < self.accel_bias_min_samples:
            print(
                f"[forward_accel_diag] Accel bias calibration failed: collected {count} samples, "
                f"needed at least {self.accel_bias_min_samples}. Continuing without bias subtraction.",
                flush=True,
            )
            return None
        self.accel_bias_forward_m_s2 = sum_forward / count
        self.accel_bias_lateral_m_s2 = sum_lateral / count
        self.accel_bias_vertical_m_s2 = sum_vertical / count
        self.gyro_bias_roll_rad_s = sum_gyro_roll / count
        self.gyro_bias_pitch_rad_s = sum_gyro_pitch / count
        self.gyro_bias_yaw_rad_s = sum_gyro_yaw / count
        result = {
            "bias_forward_m_s2": self.accel_bias_forward_m_s2,
            "bias_lateral_m_s2": self.accel_bias_lateral_m_s2,
            "bias_vertical_m_s2": self.accel_bias_vertical_m_s2,
            "gyro_bias_roll_rad_s": self.gyro_bias_roll_rad_s,
            "gyro_bias_pitch_rad_s": self.gyro_bias_pitch_rad_s,
            "gyro_bias_yaw_rad_s": self.gyro_bias_yaw_rad_s,
            "tilt_roll_rad": sum_roll_rad / count,
            "tilt_pitch_rad": sum_pitch_rad / count,
            "sample_count": count,
            "elapsed_s": round(time.time() - start, 3),
        }
        print(
            "[forward_accel_diag] Accel bias calibration complete: "
            f"samples={count} elapsed_s={result['elapsed_s']} "
            f"forward={self.accel_bias_forward_m_s2:.6f} "
            f"lateral={self.accel_bias_lateral_m_s2:.6f} "
            f"vertical={self.accel_bias_vertical_m_s2:.6f} "
            f"gyro_roll={self.gyro_bias_roll_rad_s:.6f} "
            f"gyro_pitch={self.gyro_bias_pitch_rad_s:.6f} "
            f"gyro_yaw={self.gyro_bias_yaw_rad_s:.6f} "
            f"tilt_roll={result['tilt_roll_rad']:.6f} "
            f"tilt_pitch={result['tilt_pitch_rad']:.6f}",
            flush=True,
        )
        return result

    def collect(
        self,
        *,
        duration_s: float,
        linear_m_s: float,
        send_motion: bool,
        target_distance_m: float | None = None,
        target_source: str = "tf",
        progress_log_period_s: float = 1.0,
    ) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        accel_bias = self.calibrate_accel_bias()
        if accel_bias is None:
            print("[forward_accel_diag] Running without accel bias correction.", flush=True)
        else:
            print("[forward_accel_diag] Running with accel bias correction enabled.", flush=True)
        self._reset_run_state(accel_bias=accel_bias)
        self._commanded_linear_m_s = float(linear_m_s) if send_motion else 0.0
        target_abs_distance_m = abs(float(target_distance_m)) if target_distance_m is not None else None
        normalized_target_source = str(target_source).strip().lower()
        if normalized_target_source not in {"tf", "odom", "accelerometer"}:
            raise ValueError(f"Unsupported target source: {target_source}")
        if not self.enable_zupt:
            print("[forward_accel_diag] Pure integration mode: ZUPT disabled.", flush=True)
        elif self.allow_zupt_while_commanded_motion:
            print("[forward_accel_diag] ZUPT enabled for all stationary samples.", flush=True)
        else:
            print("[forward_accel_diag] ZUPT enabled only when commanded motion is zero.", flush=True)
        if target_abs_distance_m is not None:
            print(
                f"[forward_accel_diag] Target stop source: {normalized_target_source} "
                f"target={target_abs_distance_m:.3f} m.",
                flush=True,
            )
        start = time.time()
        start_monotonic = time.monotonic()
        deadline = start + max(float(duration_s), 0.0)
        progress_log_period_s = max(float(progress_log_period_s), 0.0)
        next_progress_log_s = start if progress_log_period_s > 1e-9 else math.inf
        tf_start_pose: dict[str, float] | None = None
        odom_start_pose: dict[str, float] | None = None
        stop_reason = "duration_timeout"
        next_sample_s = start
        command_integrated_distance_m = 0.0
        last_command_integral_s = start_monotonic
        target_direction_sign = 1.0 if (target_distance_m is None or float(target_distance_m) >= 0.0) else -1.0
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.001)
            now = time.time()
            now_monotonic = time.monotonic()
            if now < next_sample_s:
                continue
            next_sample_s += self.sample_dt_s
            pose = self.lookup_pose()
            odom_pose = self.latest_odom_pose
            if tf_start_pose is None and pose is not None:
                tf_start_pose = dict(pose)
            if odom_start_pose is None and odom_pose is not None:
                odom_start_pose = dict(odom_pose)
            sample: dict[str, Any] = {
                "t_s": round(now - start, 4),
            }
            if accel_bias is not None:
                sample.update(
                    {
                        "accel_bias_forward_m_s2": accel_bias["bias_forward_m_s2"],
                        "accel_bias_lateral_m_s2": accel_bias["bias_lateral_m_s2"],
                        "accel_bias_vertical_m_s2": accel_bias["bias_vertical_m_s2"],
                        "gyro_bias_roll_rad_s": accel_bias["gyro_bias_roll_rad_s"],
                        "gyro_bias_pitch_rad_s": accel_bias["gyro_bias_pitch_rad_s"],
                        "gyro_bias_yaw_rad_s": accel_bias["gyro_bias_yaw_rad_s"],
                    }
                )
            if pose is None:
                sample["tf_pose_available"] = False
            else:
                sample.update(
                    {
                        "tf_pose_available": True,
                        "tf_x_m": float(pose["x"]),
                        "tf_y_m": float(pose["y"]),
                        "tf_yaw_rad": float(pose["yaw_rad"]),
                    }
                )
                if tf_start_pose is not None:
                    tf_forward_m, tf_lateral_m = forward_displacement_m(
                        start_x_m=tf_start_pose["x"],
                        start_y_m=tf_start_pose["y"],
                        start_yaw_rad=tf_start_pose["yaw_rad"],
                        current_x_m=pose["x"],
                        current_y_m=pose["y"],
                    )
                    sample["tf_forward_distance_m"] = tf_forward_m
                    sample["tf_lateral_distance_m"] = tf_lateral_m
            if odom_pose is None:
                sample["odom_pose_available"] = False
            else:
                sample.update(
                    {
                        "odom_pose_available": True,
                        "odom_x_m": float(odom_pose["x"]),
                        "odom_y_m": float(odom_pose["y"]),
                        "odom_yaw_rad": float(odom_pose["yaw_rad"]),
                    }
                )
                if odom_start_pose is not None:
                    odom_forward_m, odom_lateral_m = forward_displacement_m(
                        start_x_m=odom_start_pose["x"],
                        start_y_m=odom_start_pose["y"],
                        start_yaw_rad=odom_start_pose["yaw_rad"],
                        current_x_m=odom_pose["x"],
                        current_y_m=odom_pose["y"],
                    )
                    sample["odom_forward_distance_m"] = odom_forward_m
                    sample["odom_lateral_distance_m"] = odom_lateral_m
            if self._latest_integration_sample is None:
                sample["imu_available"] = False
            else:
                sample.update(self._latest_integration_sample)
                imu_age_s = self.imu_staleness_s(now_s=now_monotonic)
                if imu_age_s is not None:
                    sample["imu_wall_age_s"] = imu_age_s
                    sample["imu_stale"] = (
                        self.max_imu_staleness_s > 1e-9 and imu_age_s > self.max_imu_staleness_s
                    )
            command_linear_m_s = 0.0
            target_reached = False
            if send_motion:
                imu_fresh = self.imu_is_fresh(now_s=now_monotonic)
                if not imu_fresh and self.max_imu_staleness_s > 1e-9:
                    if self._integration_sample_count <= 0:
                        elapsed_without_imu_s = now_monotonic - start_monotonic
                        sample["motion_blocked_reason"] = "imu_unavailable"
                        if elapsed_without_imu_s >= self.max_imu_staleness_s:
                            sample["cmd_linear_m_s"] = 0.0
                            stop_reason = "imu_unavailable"
                            self.stop()
                            samples.append(sample)
                            break
                    else:
                        sample["motion_blocked_reason"] = "imu_stale"
                        sample["cmd_linear_m_s"] = 0.0
                        stop_reason = "imu_stale_timeout"
                        self.stop()
                        samples.append(sample)
                        break
                else:
                    command_linear_m_s = float(linear_m_s)
                    if target_abs_distance_m is not None:
                        sample["target_source"] = normalized_target_source
                        signed_progress_m = target_signed_progress_m(
                            sample=sample,
                            target_source=normalized_target_source,
                            accelerometer_distance_m=self._estimated_distance_m,
                            target_direction_sign=target_direction_sign,
                        )
                        if signed_progress_m is None:
                            sample["target_unavailable_reason"] = f"{normalized_target_source}_distance_unavailable"
                        if signed_progress_m is not None:
                            remaining_m = max(target_abs_distance_m - max(signed_progress_m, 0.0), 0.0)
                            sample["target_signed_progress_m"] = signed_progress_m
                            sample["target_remaining_distance_m"] = remaining_m
                            if signed_progress_m >= target_abs_distance_m:
                                command_linear_m_s = 0.0
                                target_reached = True
                self.publish_forward(command_linear_m_s)
            command_dt_s = max(0.0, now_monotonic - last_command_integral_s)
            last_command_integral_s = now_monotonic
            command_integrated_distance_m += command_linear_m_s * command_dt_s
            sample["cmd_linear_m_s"] = command_linear_m_s
            sample["cmd_integrated_forward_distance_m"] = command_integrated_distance_m
            if now >= next_progress_log_s:
                print(
                    format_progress_log_line(
                        sample,
                        target_source=normalized_target_source,
                        target_distance_m=target_abs_distance_m,
                    ),
                    flush=True,
                )
                while next_progress_log_s <= now:
                    next_progress_log_s += progress_log_period_s
            samples.append(sample)
            if target_abs_distance_m is not None and target_reached:
                stop_reason = f"target_{normalized_target_source}_distance_reached"
                break
        self._run_active = False
        self.stop()
        for _ in range(5):
            rclpy.spin_once(self, timeout_sec=0.02)
            self.stop()
        if samples:
            samples[-1]["stop_reason"] = stop_reason
        return samples


def _summarize_pose_source(samples: list[dict[str, Any]], *, prefix: str) -> dict[str, Any]:
    distance_key = f"{prefix}_forward_distance_m"
    lateral_key = f"{prefix}_lateral_distance_m"
    valid = [sample for sample in samples if distance_key in sample]
    if not valid:
        return {
            "valid_pose_count": 0,
            "message": f"No {prefix} pose samples were available.",
        }
    start = valid[0]
    end = valid[-1]
    return {
        "valid_pose_count": len(valid),
        "elapsed_s": round(max(float(end["t_s"]) - float(start["t_s"]), 1e-6), 3),
        "forward_distance_m": round(float(end[distance_key]), 4),
        "lateral_distance_m": round(float(end[lateral_key]), 4),
    }


def _summarize_accel(samples: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [sample for sample in samples if sample.get("imu_available")]
    if not valid:
        return {
            "valid_sample_count": 0,
            "message": "No IMU samples were available.",
        }
    end = valid[-1]
    peak_used_accel = max(abs(float(sample.get("imu_used_forward_acceleration_m_s2", 0.0))) for sample in valid)
    peak_velocity = max(abs(float(sample.get("imu_estimated_forward_velocity_m_s", 0.0))) for sample in valid)
    stationary_sample_count = sum(1 for sample in valid if sample.get("imu_stationary"))
    zupt_sample_count = sum(1 for sample in valid if sample.get("imu_zupt_applied"))
    stale_sample_count = sum(1 for sample in valid if sample.get("imu_stale"))
    observed_rate_hz: float | None = None
    integrated_sample_count: int | None = None
    integration_rejected_count: int | None = None
    max_wall_age_s: float | None = None
    wall_ages = [float(sample["imu_wall_age_s"]) for sample in valid if "imu_wall_age_s" in sample]
    if wall_ages:
        max_wall_age_s = max(wall_ages)
    if "imu_integration_sample_count" in end:
        integrated_sample_count = int(end.get("imu_integration_sample_count", 0))
        integration_rejected_count = int(end.get("imu_integration_rejected_nonmonotonic_count", 0))
        first_stamp_s = float(end.get("imu_integration_first_timestamp_s", end.get("imu_timestamp_s", 0.0)))
        last_stamp_s = float(end.get("imu_timestamp_s", 0.0))
        elapsed_stamp_s = last_stamp_s - first_stamp_s
        if integrated_sample_count >= 2 and elapsed_stamp_s > 1e-6:
            observed_rate_hz = (integrated_sample_count - 1) / elapsed_stamp_s
    if len(valid) >= 2:
        first_stamp_s = float(valid[0].get("imu_timestamp_s", 0.0))
        last_stamp_s = float(valid[-1].get("imu_timestamp_s", 0.0))
        elapsed_stamp_s = last_stamp_s - first_stamp_s
        if observed_rate_hz is None and elapsed_stamp_s > 1e-6:
            observed_rate_hz = (len(valid) - 1) / elapsed_stamp_s
    result = {
        "valid_sample_count": len(valid),
        "logged_sample_count": len(valid),
        "elapsed_s": round(max(float(end["t_s"]) - float(valid[0]["t_s"]), 1e-6), 3),
        "reported_distance_m": round(float(end.get("imu_estimated_forward_distance_m", 0.0)), 4),
        "reported_velocity_m_s": round(float(end.get("imu_estimated_forward_velocity_m_s", 0.0)), 4),
        "peak_used_forward_acceleration_m_s2": round(peak_used_accel, 4),
        "peak_estimated_velocity_m_s": round(peak_velocity, 4),
        "stationary_sample_count": stationary_sample_count,
        "moving_sample_count": len(valid) - stationary_sample_count,
        "stationary_fraction": round(stationary_sample_count / len(valid), 4),
        "zupt_applied_sample_count": zupt_sample_count,
        "zupt_applied_fraction": round(zupt_sample_count / len(valid), 4),
        "stale_sample_count": stale_sample_count,
        "stale_fraction": round(stale_sample_count / len(valid), 4),
        "observed_imu_rate_hz": None if observed_rate_hz is None else round(observed_rate_hz, 3),
    }
    if integrated_sample_count is not None:
        result["integrated_sample_count"] = integrated_sample_count
    if integration_rejected_count is not None:
        result["integration_rejected_nonmonotonic_count"] = integration_rejected_count
    if max_wall_age_s is not None:
        result["max_imu_wall_age_s"] = round(max_wall_age_s, 3)
    return result


def _summarize_command(samples: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [sample for sample in samples if "cmd_integrated_forward_distance_m" in sample]
    if not valid:
        return {"reported_distance_m": 0.0, "valid_sample_count": 0}
    moving = [sample for sample in valid if abs(float(sample.get("cmd_linear_m_s", 0.0))) > 1e-9]
    return {
        "reported_distance_m": round(float(valid[-1]["cmd_integrated_forward_distance_m"]), 4),
        "valid_sample_count": len(valid),
        "moving_sample_count": len(moving),
    }


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "sample_count": len(samples),
        "stop_reason": samples[-1].get("stop_reason") if samples else "no_samples",
        "command": _summarize_command(samples),
        "accelerometer": _summarize_accel(samples),
        "tf": _summarize_pose_source(samples, prefix="tf"),
        "odom_topic": _summarize_pose_source(samples, prefix="odom"),
        "accelerometer_bias_applied": bool(samples and "accel_bias_forward_m_s2" in samples[0]),
        "tilt_compensation_applied": any("imu_gravity_forward_m_s2" in sample for sample in samples),
    }
    target_sources = [str(sample["target_source"]) for sample in samples if "target_source" in sample]
    if target_sources:
        summary["target_source"] = target_sources[-1]
    if samples and "accel_bias_forward_m_s2" in samples[0]:
        summary["accelerometer_bias"] = {
            "forward_m_s2": round(float(samples[0]["accel_bias_forward_m_s2"]), 6),
            "lateral_m_s2": round(float(samples[0]["accel_bias_lateral_m_s2"]), 6),
            "vertical_m_s2": round(float(samples[0]["accel_bias_vertical_m_s2"]), 6),
            "gyro_roll_rad_s": round(float(samples[0]["gyro_bias_roll_rad_s"]), 6),
            "gyro_pitch_rad_s": round(float(samples[0]["gyro_bias_pitch_rad_s"]), 6),
            "gyro_yaw_rad_s": round(float(samples[0]["gyro_bias_yaw_rad_s"]), 6),
        }
    return summary


def write_csv(path: Path, samples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for sample in samples for key in sample})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(samples)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Drive forward and estimate distance from raw accelerometer integration while logging TF and /odom for comparison."
    )
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument(
        "--sample-hz",
        type=float,
        default=50.0,
        help="Snapshot/logging and command publish cadence. IMU integration runs at IMU callback rate.",
    )
    parser.add_argument("--accel-bias-calibration-s", type=float, default=3.0)
    parser.add_argument("--accel-bias-min-samples", type=int, default=50)
    parser.add_argument("--acceleration-deadband-m-s2", type=float, default=0.0)
    parser.add_argument("--velocity-damping-per-s", type=float, default=0.0)
    parser.add_argument("--max-estimated-velocity-m-s", type=float, default=0.75)
    parser.add_argument("--tilt-accel-correction-alpha", type=float, default=0.02)
    parser.add_argument("--tilt-accel-correction-alpha-when-moving", type=float, default=0.0)
    parser.add_argument("--gravity-m-s2", type=float, default=9.80665)
    parser.add_argument("--accel-stationary-tolerance-m-s2", type=float, default=0.10)
    parser.add_argument("--gyro-stationary-tolerance-rad-s", type=float, default=0.05)
    parser.add_argument(
        "--enable-zupt",
        action="store_true",
        help="Enable zero-velocity resets on stationary-classified IMU samples.",
    )
    parser.add_argument(
        "--zupt-while-commanded-motion",
        action="store_true",
        help="Keep zero-velocity resets enabled even while a non-zero forward command is active.",
    )
    parser.add_argument(
        "--max-imu-staleness-s",
        type=float,
        default=0.5,
        help="When sending motion, stop immediately if no integrated IMU callback has arrived within this wall-clock age. Use 0 to disable.",
    )
    parser.add_argument("--odom-frame", default="odom")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--imu-topic", default="/imu")
    parser.add_argument(
        "--imu-frame-convention",
        choices=("camera_optical", "base_link"),
        default="camera_optical",
        help="Interpret raw IMU vectors in camera optical coordinates or already-rotated base_link coordinates.",
    )
    parser.add_argument("--linear-m-s", type=float, default=0.03)
    parser.add_argument(
        "--target-distance-m",
        type=float,
        default=0.45,
        help="Stop early when the selected target source reaches this absolute distance.",
    )
    parser.add_argument(
        "--target-source",
        choices=("tf", "odom", "accelerometer"),
        default="tf",
        help="Distance source used for target stopping. The accelerometer source is raw double integration and is diagnostic-only.",
    )
    parser.add_argument(
        "--progress-log-period-s",
        type=float,
        default=1.0,
        help="Print live distance progress at this interval. Use 0 to disable progress logging.",
    )
    parser.add_argument(
        "--send-motion",
        action="store_true",
        help="Actually publish forward /cmd_vel. Without this flag the script only records accelerometer, TF, and /odom.",
    )
    parser.add_argument("--csv-out", default="artifacts/diagnostics/forward_accel_diagnostic.csv")
    parser.add_argument("--json-out", default="artifacts/diagnostics/forward_accel_diagnostic_summary.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require_ros()
    rclpy.init()
    node = ForwardAccelDiagnosticNode(
        odom_frame=args.odom_frame,
        base_frame=args.base_frame,
        cmd_vel_topic=args.cmd_vel_topic,
        odom_topic=args.odom_topic,
        imu_topic=args.imu_topic,
        imu_frame_convention=args.imu_frame_convention,
        sample_hz=args.sample_hz,
        accel_bias_calibration_s=args.accel_bias_calibration_s,
        accel_bias_min_samples=args.accel_bias_min_samples,
        acceleration_deadband_m_s2=args.acceleration_deadband_m_s2,
        velocity_damping_per_s=args.velocity_damping_per_s,
        max_estimated_velocity_m_s=args.max_estimated_velocity_m_s,
        tilt_accel_correction_alpha=args.tilt_accel_correction_alpha,
        tilt_accel_correction_alpha_when_moving=args.tilt_accel_correction_alpha_when_moving,
        gravity_m_s2=args.gravity_m_s2,
        accel_stationary_tolerance_m_s2=args.accel_stationary_tolerance_m_s2,
        gyro_stationary_tolerance_rad_s=args.gyro_stationary_tolerance_rad_s,
        enable_zupt=args.enable_zupt,
        allow_zupt_while_commanded_motion=args.zupt_while_commanded_motion,
        max_imu_staleness_s=args.max_imu_staleness_s,
    )
    try:
        samples = node.collect(
            duration_s=args.duration_s,
            linear_m_s=args.linear_m_s,
            send_motion=args.send_motion,
            target_distance_m=args.target_distance_m,
            target_source=args.target_source,
            progress_log_period_s=args.progress_log_period_s,
        )
        summary = summarize(samples)
        write_csv(Path(args.csv_out).expanduser(), samples)
        json_path = Path(args.json_out).expanduser()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
        accel_distance_m = summary.get("accelerometer", {}).get("reported_distance_m")
        if accel_distance_m is not None:
            print(f"Raw accelerometer-only distance: {accel_distance_m} m")
        print(f"Wrote samples: {Path(args.csv_out).expanduser()}")
        print(f"Wrote summary: {json_path}")
        return 0 if (
            summary.get("accelerometer", {}).get("valid_sample_count", 0)
            or summary.get("tf", {}).get("valid_pose_count", 0)
            or summary.get("odom_topic", {}).get("valid_pose_count", 0)
        ) else 2
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
