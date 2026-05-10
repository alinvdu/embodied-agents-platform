from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

try:
    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.time import Time as RosTime
    from sensor_msgs.msg import Imu
    from std_msgs.msg import Float32
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
    Float32 = None
    ConnectivityException = Exception
    ExtrapolationException = Exception
    LookupException = Exception
else:
    IMPORT_ERROR = None


def require_ros() -> None:
    if IMPORT_ERROR is not None:
        raise RuntimeError("ROS rotation diagnostic requires ROS 2 Python packages.") from IMPORT_ERROR


def http_json(method: str, url: str, payload: dict[str, Any] | None = None, *, timeout_s: float = 5.0) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib_request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib_request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read()
    except urllib_error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {method} {url} failed with {exc.code}: {body}") from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"HTTP {method} {url} failed: {exc}") from exc
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def yaw_from_quaternion_xyzw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def angle_delta(current: float, previous: float) -> float:
    return math.atan2(math.sin(current - previous), math.cos(current - previous))


def imu_value_from_source(*, source: str, gyro_x: float, gyro_y: float, gyro_z: float) -> float:
    normalized = str(source).strip().lower()
    if normalized == "x":
        return gyro_x
    if normalized == "y":
        return gyro_y
    if normalized == "z":
        return gyro_z
    if normalized in {"robot_yaw", "optical_yaw"}:
        # Gemini 2 publishes raw sensor data in camera optical coordinates:
        # optical X right, Y down, Z forward. For an aligned robot body,
        # base_link yaw rate (around Z up) maps to -optical Y.
        return -gyro_y
    raise ValueError(f"Unsupported IMU source: {source}")


def feedback_unwrapped_yaw_rad(sample: dict[str, Any], *, source: str) -> float | None:
    normalized = str(source).strip().lower()
    if normalized == "imu" and sample.get("imu_orientation_available"):
        value = sample.get("imu_orientation_unwrapped_yaw_rad")
        return float(value) if value is not None else None
    key = f"{normalized}_unwrapped_yaw_rad"
    value = sample.get(key)
    return float(value) if value is not None else None


def compute_control_angular_velocity(
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


class RotationDiagnosticNode(Node):
    def __init__(
        self,
        *,
        odom_frame: str,
        base_frame: str,
        cmd_vel_topic: str,
        odom_topic: str,
        imu_topic: str,
        camera_pan_topic: str,
        imu_axis: str,
        sample_hz: float,
        imu_bias_calibration_s: float,
        imu_bias_min_samples: int,
    ) -> None:
        super().__init__("xlerobot_rotation_diagnostic")
        self.odom_frame = odom_frame
        self.base_frame = base_frame
        self.imu_axis = str(imu_axis).strip().lower()
        self.sample_dt_s = 1.0 / max(float(sample_hz), 1e-6)
        self.imu_bias_calibration_s = max(float(imu_bias_calibration_s), 0.0)
        self.imu_bias_min_samples = max(int(imu_bias_min_samples), 1)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)
        self.cmd_vel_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.latest_odom_pose: dict[str, float] | None = None
        self.latest_imu_sample: dict[str, float] | None = None
        self.latest_camera_pan_rad: float | None = None
        self.imu_bias_x_rad_s = 0.0
        self.imu_bias_y_rad_s = 0.0
        self.imu_bias_z_rad_s = 0.0
        self.create_subscription(Odometry, odom_topic, self._on_odom, 10)
        self.create_subscription(Imu, imu_topic, self._on_imu, 50)
        self.create_subscription(Float32, camera_pan_topic, self._on_camera_pan, 10)

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
        orientation_yaw_rad = None
        covariance = getattr(message, "orientation_covariance", None)
        if covariance is not None and len(covariance) >= 1 and float(covariance[0]) >= 0.0:
            orientation = message.orientation
            orientation_yaw_rad = yaw_from_quaternion_xyzw(
                float(orientation.x),
                float(orientation.y),
                float(orientation.z),
                float(orientation.w),
            )
        self.latest_imu_sample = {
            "timestamp_s": stamp_s,
            "angular_velocity_x_rad_s": float(message.angular_velocity.x),
            "angular_velocity_y_rad_s": float(message.angular_velocity.y),
            "angular_velocity_z_rad_s": float(message.angular_velocity.z),
            "linear_acceleration_x_m_s2": float(message.linear_acceleration.x),
            "linear_acceleration_y_m_s2": float(message.linear_acceleration.y),
            "linear_acceleration_z_m_s2": float(message.linear_acceleration.z),
            "orientation_yaw_rad": orientation_yaw_rad,
        }

    def _on_camera_pan(self, message: Any) -> None:
        self.latest_camera_pan_rad = float(message.data)

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

    def publish_spin(self, angular_rad_s: float) -> None:
        twist = Twist()
        twist.angular.z = float(angular_rad_s)
        self.cmd_vel_pub.publish(twist)

    def stop(self) -> None:
        self.cmd_vel_pub.publish(Twist())

    def command_head_pan(
        self,
        *,
        robot_brain_url: str,
        pan_deg: float,
        settle_s: float,
        action_key: str | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "pan_deg": float(pan_deg),
            "settle_s": float(settle_s),
        }
        if action_key:
            payload["action_key"] = action_key
        url = f"{robot_brain_url.rstrip('/')}/camera/head/pan"
        return http_json("POST", url, payload, timeout_s=max(float(settle_s) + 5.0, 5.0))

    def read_head_pose(self, *, robot_brain_url: str) -> dict[str, Any] | None:
        try:
            return http_json("GET", f"{robot_brain_url.rstrip('/')}/camera/head/pose", None, timeout_s=2.0)
        except RuntimeError:
            return None

    def collect_head_pan(
        self,
        *,
        robot_brain_url: str,
        pan_deg: float,
        settle_s: float,
        action_key: str | None,
        duration_s: float,
    ) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        start = time.time()
        command_response: dict[str, Any] | None = None
        command_error: str | None = None
        for _ in range(max(int(self.sample_dt_s * 20), 3)):
            rclpy.spin_once(self, timeout_sec=0.02)
        initial_head_pose = self.read_head_pose(robot_brain_url=robot_brain_url)
        initial_sample: dict[str, Any] = {
            "t_s": round(time.time() - start, 4),
            "requested_head_pan_deg": float(pan_deg),
            "command_succeeded": False,
            "command_error": None,
            "pre_command_sample": True,
        }
        if self.latest_camera_pan_rad is None:
            initial_sample["head_pan_topic_available"] = False
        else:
            initial_sample.update(
                {
                    "head_pan_topic_available": True,
                    "head_pan_topic_rad": self.latest_camera_pan_rad,
                    "head_pan_topic_deg": math.degrees(self.latest_camera_pan_rad),
                }
            )
        if isinstance(initial_head_pose, dict):
            initial_sample.update(
                {
                    "head_pose_available": True,
                    "head_pose_pan_rad": float(initial_head_pose.get("pan_rad", 0.0)),
                    "head_pose_pan_deg": float(initial_head_pose.get("pan_deg", 0.0)),
                    "head_pose_moving": bool(initial_head_pose.get("moving", False)),
                }
            )
        else:
            initial_sample["head_pose_available"] = False
        samples.append(initial_sample)
        try:
            command_response = self.command_head_pan(
                robot_brain_url=robot_brain_url,
                pan_deg=pan_deg,
                settle_s=settle_s,
                action_key=action_key,
            )
        except RuntimeError as exc:
            command_error = str(exc)
        deadline = start + max(float(duration_s), float(settle_s) + 1.0)
        while time.time() < deadline:
            now = time.time()
            rclpy.spin_once(self, timeout_sec=0.0)
            head_pose = self.read_head_pose(robot_brain_url=robot_brain_url)
            sample: dict[str, Any] = {
                "t_s": round(now - start, 4),
                "requested_head_pan_deg": float(pan_deg),
                "command_succeeded": bool(command_response and command_response.get("succeeded")),
                "command_error": command_error,
            }
            if self.latest_camera_pan_rad is None:
                sample["head_pan_topic_available"] = False
            else:
                sample.update(
                    {
                        "head_pan_topic_available": True,
                        "head_pan_topic_rad": self.latest_camera_pan_rad,
                        "head_pan_topic_deg": math.degrees(self.latest_camera_pan_rad),
                    }
                )
            if isinstance(head_pose, dict):
                sample.update(
                    {
                        "head_pose_available": True,
                        "head_pose_pan_rad": float(head_pose.get("pan_rad", 0.0)),
                        "head_pose_pan_deg": float(head_pose.get("pan_deg", 0.0)),
                        "head_pose_moving": bool(head_pose.get("moving", False)),
                    }
                )
            else:
                sample["head_pose_available"] = False
            samples.append(sample)
            time.sleep(self.sample_dt_s)
        if samples:
            samples[-1]["stop_reason"] = "head_pan_command_error" if command_error else "duration_timeout"
            if command_response is not None:
                samples[-1]["command_response"] = json.dumps(command_response, sort_keys=True)
        return samples

    def calibrate_imu_bias(self) -> dict[str, float] | None:
        if self.imu_bias_calibration_s <= 1e-6:
            return {
                "bias_x_rad_s": self.imu_bias_x_rad_s,
                "bias_y_rad_s": self.imu_bias_y_rad_s,
                "bias_z_rad_s": self.imu_bias_z_rad_s,
                "sample_count": 0,
                "elapsed_s": 0.0,
            }
        start = time.time()
        deadline = start + self.imu_bias_calibration_s
        sum_x = 0.0
        sum_y = 0.0
        sum_z = 0.0
        count = 0
        last_timestamp_s: float | None = None
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=min(self.sample_dt_s, 0.05))
            imu_sample = self.latest_imu_sample
            if imu_sample is None:
                continue
            timestamp_s = float(imu_sample["timestamp_s"])
            if last_timestamp_s is not None and timestamp_s <= last_timestamp_s:
                continue
            last_timestamp_s = timestamp_s
            sum_x += float(imu_sample["angular_velocity_x_rad_s"])
            sum_y += float(imu_sample["angular_velocity_y_rad_s"])
            sum_z += float(imu_sample["angular_velocity_z_rad_s"])
            count += 1
        if count < self.imu_bias_min_samples:
            return None
        self.imu_bias_x_rad_s = sum_x / count
        self.imu_bias_y_rad_s = sum_y / count
        self.imu_bias_z_rad_s = sum_z / count
        return {
            "bias_x_rad_s": self.imu_bias_x_rad_s,
            "bias_y_rad_s": self.imu_bias_y_rad_s,
            "bias_z_rad_s": self.imu_bias_z_rad_s,
            "sample_count": count,
            "elapsed_s": round(time.time() - start, 3),
        }

    def collect(
        self,
        *,
        duration_s: float,
        angular_rad_s: float,
        send_motion: bool,
        target_yaw_rad: float | None = None,
        target_source: str = "tf",
    ) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        imu_bias = self.calibrate_imu_bias()
        start = time.time()
        deadline = start + max(float(duration_s), 0.0)
        target_abs_yaw_rad = abs(float(target_yaw_rad)) if target_yaw_rad is not None else None
        previous_tf_yaw: float | None = None
        previous_odom_yaw: float | None = None
        previous_imu_stamp_s: float | None = None
        previous_imu_orientation_yaw: float | None = None
        unwrapped_tf_yaw = 0.0
        unwrapped_odom_yaw = 0.0
        unwrapped_imu_yaw = 0.0
        unwrapped_imu_orientation_yaw = 0.0
        stop_reason = "duration_timeout"
        while time.time() < deadline:
            now = time.time()
            rclpy.spin_once(self, timeout_sec=0.0)
            pose = self.lookup_pose()
            odom_pose = self.latest_odom_pose
            imu_sample = self.latest_imu_sample
            sample: dict[str, Any] = {
                "t_s": round(now - start, 4),
            }
            if imu_bias is not None:
                sample.update(
                    {
                        "imu_bias_x_rad_s": imu_bias["bias_x_rad_s"],
                        "imu_bias_y_rad_s": imu_bias["bias_y_rad_s"],
                        "imu_bias_z_rad_s": imu_bias["bias_z_rad_s"],
                    }
                )
            if pose is None:
                sample["tf_pose_available"] = False
            else:
                yaw = float(pose["yaw_rad"])
                if previous_tf_yaw is not None:
                    unwrapped_tf_yaw += angle_delta(yaw, previous_tf_yaw)
                previous_tf_yaw = yaw
                sample.update(
                    {
                        "tf_pose_available": True,
                        "tf_x_m": pose["x"],
                        "tf_y_m": pose["y"],
                        "tf_yaw_rad": yaw,
                        "tf_yaw_deg": math.degrees(yaw),
                        "tf_unwrapped_yaw_rad": unwrapped_tf_yaw,
                        "tf_unwrapped_yaw_deg": math.degrees(unwrapped_tf_yaw),
                    }
                )
            if odom_pose is None:
                sample["odom_pose_available"] = False
            else:
                odom_yaw = float(odom_pose["yaw_rad"])
                if previous_odom_yaw is not None:
                    unwrapped_odom_yaw += angle_delta(odom_yaw, previous_odom_yaw)
                previous_odom_yaw = odom_yaw
                sample.update(
                    {
                        "odom_pose_available": True,
                        "odom_x_m": odom_pose["x"],
                        "odom_y_m": odom_pose["y"],
                        "odom_yaw_rad": odom_yaw,
                        "odom_yaw_deg": math.degrees(odom_yaw),
                        "odom_unwrapped_yaw_rad": unwrapped_odom_yaw,
                        "odom_unwrapped_yaw_deg": math.degrees(unwrapped_odom_yaw),
                    }
                )
            if imu_sample is None:
                sample["imu_available"] = False
            else:
                imu_stamp_s = float(imu_sample["timestamp_s"])
                raw_gyro_x = float(imu_sample["angular_velocity_x_rad_s"])
                raw_gyro_y = float(imu_sample["angular_velocity_y_rad_s"])
                raw_gyro_z = float(imu_sample["angular_velocity_z_rad_s"])
                orientation_yaw_rad = imu_sample.get("orientation_yaw_rad")
                gyro_x = raw_gyro_x - self.imu_bias_x_rad_s
                gyro_y = raw_gyro_y - self.imu_bias_y_rad_s
                gyro_z = raw_gyro_z - self.imu_bias_z_rad_s
                gyro_axis_value = imu_value_from_source(
                    source=self.imu_axis,
                    gyro_x=gyro_x,
                    gyro_y=gyro_y,
                    gyro_z=gyro_z,
                )
                imu_dt_s = self.sample_dt_s if previous_imu_stamp_s is None else max(0.0, imu_stamp_s - previous_imu_stamp_s)
                previous_imu_stamp_s = imu_stamp_s
                unwrapped_imu_yaw += gyro_axis_value * imu_dt_s
                orientation_available = orientation_yaw_rad is not None
                if orientation_available:
                    orientation_yaw_value = float(orientation_yaw_rad)
                    if previous_imu_orientation_yaw is not None:
                        unwrapped_imu_orientation_yaw += angle_delta(orientation_yaw_value, previous_imu_orientation_yaw)
                    previous_imu_orientation_yaw = orientation_yaw_value
                sample.update(
                    {
                        "imu_available": True,
                        "imu_timestamp_s": imu_stamp_s,
                        "imu_dt_s": imu_dt_s,
                        "imu_axis": self.imu_axis,
                        "imu_raw_angular_velocity_x_rad_s": raw_gyro_x,
                        "imu_raw_angular_velocity_y_rad_s": raw_gyro_y,
                        "imu_raw_angular_velocity_z_rad_s": raw_gyro_z,
                        "imu_angular_velocity_x_rad_s": gyro_x,
                        "imu_angular_velocity_y_rad_s": gyro_y,
                        "imu_angular_velocity_z_rad_s": gyro_z,
                        "imu_angular_velocity_axis_rad_s": gyro_axis_value,
                        "imu_linear_acceleration_x_m_s2": float(imu_sample["linear_acceleration_x_m_s2"]),
                        "imu_linear_acceleration_y_m_s2": float(imu_sample["linear_acceleration_y_m_s2"]),
                        "imu_linear_acceleration_z_m_s2": float(imu_sample["linear_acceleration_z_m_s2"]),
                        "imu_unwrapped_yaw_rad": unwrapped_imu_yaw,
                        "imu_unwrapped_yaw_deg": math.degrees(unwrapped_imu_yaw),
                        "imu_orientation_available": orientation_available,
                        "imu_orientation_yaw_rad": float(orientation_yaw_rad) if orientation_available else None,
                        "imu_orientation_yaw_deg": math.degrees(float(orientation_yaw_rad)) if orientation_available else None,
                        "imu_orientation_unwrapped_yaw_rad": unwrapped_imu_orientation_yaw,
                        "imu_orientation_unwrapped_yaw_deg": math.degrees(unwrapped_imu_orientation_yaw),
                    }
                )
            feedback_yaw_rad = feedback_unwrapped_yaw_rad(sample, source=target_source)
            command_angular_rad_s = 0.0
            target_reached = False
            if send_motion:
                command_angular_rad_s, target_reached = compute_control_angular_velocity(
                    requested_angular_rad_s=angular_rad_s,
                    target_yaw_rad=target_yaw_rad,
                    feedback_yaw_rad=feedback_yaw_rad,
                )
                self.publish_spin(command_angular_rad_s)
            sample["cmd_angular_rad_s"] = command_angular_rad_s
            if feedback_yaw_rad is not None:
                sample["target_feedback_unwrapped_yaw_rad"] = feedback_yaw_rad
                sample["target_feedback_unwrapped_yaw_deg"] = math.degrees(feedback_yaw_rad)
                if target_abs_yaw_rad is not None:
                    sample["target_feedback_remaining_yaw_rad"] = max(target_abs_yaw_rad - abs(feedback_yaw_rad), 0.0)
                    sample["target_feedback_remaining_yaw_deg"] = math.degrees(
                        sample["target_feedback_remaining_yaw_rad"]
                    )
            samples.append(sample)
            if target_abs_yaw_rad is not None and target_reached:
                stop_reason = f"target_{target_source}_yaw_reached"
                break
            time.sleep(self.sample_dt_s)
        self.stop()
        for _ in range(5):
            rclpy.spin_once(self, timeout_sec=0.02)
            self.stop()
        if samples:
            samples[-1]["stop_reason"] = stop_reason
        return samples


def _summarize_source(samples: list[dict[str, Any]], *, prefix: str) -> dict[str, Any]:
    valid = [sample for sample in samples if sample.get(f"{prefix}_pose_available")]
    if not valid:
        return {
            "valid_pose_count": 0,
            "message": f"No {prefix} pose samples were available.",
        }
    start = valid[0]
    end = valid[-1]
    elapsed_s = max(float(end["t_s"]) - float(start["t_s"]), 1e-6)
    yaw_delta_rad = float(end[f"{prefix}_unwrapped_yaw_rad"]) - float(start[f"{prefix}_unwrapped_yaw_rad"])
    drift_m = math.hypot(float(end[f"{prefix}_x_m"]) - float(start[f"{prefix}_x_m"]), float(end[f"{prefix}_y_m"]) - float(start[f"{prefix}_y_m"]))
    return {
        "valid_pose_count": len(valid),
        "elapsed_s": round(elapsed_s, 3),
        "unwrapped_yaw_delta_rad": round(yaw_delta_rad, 4),
        "unwrapped_yaw_delta_deg": round(math.degrees(yaw_delta_rad), 2),
        "mean_yaw_rate_rad_s": round(yaw_delta_rad / elapsed_s, 4),
        "translation_drift_m": round(drift_m, 4),
        "start": {
            "x_m": start[f"{prefix}_x_m"],
            "y_m": start[f"{prefix}_y_m"],
            "yaw_deg": start[f"{prefix}_yaw_deg"],
        },
        "end": {
            "x_m": end[f"{prefix}_x_m"],
            "y_m": end[f"{prefix}_y_m"],
            "yaw_deg": end[f"{prefix}_yaw_deg"],
        },
    }


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if samples and "requested_head_pan_deg" in samples[0]:
        return summarize_head_pan(samples)
    summary = {
        "sample_count": len(samples),
        "stop_reason": samples[-1].get("stop_reason") if samples else "no_samples",
        "tf": _summarize_source(samples, prefix="tf"),
        "odom_topic": _summarize_source(samples, prefix="odom"),
        "imu": _summarize_imu(samples),
    }
    if samples and "imu_bias_x_rad_s" in samples[0]:
        summary["imu_bias"] = {
            "bias_x_rad_s": round(float(samples[0]["imu_bias_x_rad_s"]), 6),
            "bias_y_rad_s": round(float(samples[0]["imu_bias_y_rad_s"]), 6),
            "bias_z_rad_s": round(float(samples[0]["imu_bias_z_rad_s"]), 6),
        }
    return summary


def _summarize_imu(samples: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [sample for sample in samples if sample.get("imu_available")]
    if not valid:
        return {
            "valid_sample_count": 0,
            "message": "No IMU samples were available.",
        }
    start = valid[0]
    end = valid[-1]
    elapsed_s = max(float(end["t_s"]) - float(start["t_s"]), 1e-6)
    yaw_delta_rad = float(end["imu_unwrapped_yaw_rad"]) - float(start["imu_unwrapped_yaw_rad"])
    yaw_delta_deg = float(end["imu_unwrapped_yaw_deg"]) - float(start["imu_unwrapped_yaw_deg"])
    summary = {
        "valid_sample_count": len(valid),
        "elapsed_s": round(elapsed_s, 3),
        "reported_turn_deg": round(yaw_delta_deg, 2),
        "reported_turn_rad": round(yaw_delta_rad, 4),
        "unwrapped_yaw_delta_rad": round(yaw_delta_rad, 4),
        "unwrapped_yaw_delta_deg": round(yaw_delta_deg, 2),
        "mean_yaw_rate_rad_s": round(yaw_delta_rad / elapsed_s, 4),
        "start": {
            "yaw_deg": start["imu_unwrapped_yaw_deg"],
            "axis": start["imu_axis"],
            "angular_velocity_axis_rad_s": start["imu_angular_velocity_axis_rad_s"],
            "angular_velocity_x_rad_s": start["imu_angular_velocity_x_rad_s"],
            "angular_velocity_y_rad_s": start["imu_angular_velocity_y_rad_s"],
            "angular_velocity_z_rad_s": start["imu_angular_velocity_z_rad_s"],
        },
        "end": {
            "yaw_deg": end["imu_unwrapped_yaw_deg"],
            "axis": end["imu_axis"],
            "angular_velocity_axis_rad_s": end["imu_angular_velocity_axis_rad_s"],
            "angular_velocity_x_rad_s": end["imu_angular_velocity_x_rad_s"],
            "angular_velocity_y_rad_s": end["imu_angular_velocity_y_rad_s"],
            "angular_velocity_z_rad_s": end["imu_angular_velocity_z_rad_s"],
        },
    }
    orientation_valid = [sample for sample in valid if sample.get("imu_orientation_available")]
    if orientation_valid:
        start_orientation = orientation_valid[0]
        end_orientation = orientation_valid[-1]
        orientation_delta_rad = float(end_orientation["imu_orientation_unwrapped_yaw_rad"]) - float(
            start_orientation["imu_orientation_unwrapped_yaw_rad"]
        )
        orientation_delta_deg = float(end_orientation["imu_orientation_unwrapped_yaw_deg"]) - float(
            start_orientation["imu_orientation_unwrapped_yaw_deg"]
        )
        summary["orientation"] = {
            "valid_sample_count": len(orientation_valid),
            "reported_turn_deg": round(orientation_delta_deg, 2),
            "reported_turn_rad": round(orientation_delta_rad, 4),
            "start_yaw_deg": round(float(start_orientation["imu_orientation_yaw_deg"]), 2),
            "end_yaw_deg": round(float(end_orientation["imu_orientation_yaw_deg"]), 2),
        }
        summary["reported_turn_deg"] = round(orientation_delta_deg, 2)
        summary["reported_turn_rad"] = round(orientation_delta_rad, 4)
    return summary


def summarize_head_pan(samples: list[dict[str, Any]]) -> dict[str, Any]:
    topic_valid = [sample for sample in samples if sample.get("head_pan_topic_available")]
    pose_valid = [sample for sample in samples if sample.get("head_pose_available")]
    requested_deg = float(samples[0].get("requested_head_pan_deg", 0.0)) if samples else 0.0
    summary: dict[str, Any] = {
        "sample_count": len(samples),
        "mode": "head_pan",
        "stop_reason": samples[-1].get("stop_reason") if samples else "no_samples",
        "requested_head_pan_deg": round(requested_deg, 2),
    }
    if topic_valid:
        start = topic_valid[0]
        end = topic_valid[-1]
        start_deg = float(start["head_pan_topic_deg"])
        end_deg = float(end["head_pan_topic_deg"])
        summary["topic"] = {
            "valid_sample_count": len(topic_valid),
            "start_deg": round(start_deg, 2),
            "end_deg": round(end_deg, 2),
            "delta_deg": round(end_deg - start_deg, 2),
            "target_error_deg": round(end_deg - requested_deg, 2),
        }
    else:
        summary["topic"] = {
            "valid_sample_count": 0,
            "message": "No /camera/head/pan_rad samples were available.",
        }
    if pose_valid:
        start_pose = pose_valid[0]
        end_pose = pose_valid[-1]
        start_pose_deg = float(start_pose["head_pose_pan_deg"])
        end_pose_deg = float(end_pose["head_pose_pan_deg"])
        summary["robot_brain_pose"] = {
            "valid_sample_count": len(pose_valid),
            "start_deg": round(start_pose_deg, 2),
            "end_deg": round(end_pose_deg, 2),
            "delta_deg": round(end_pose_deg - start_pose_deg, 2),
            "target_error_deg": round(end_pose_deg - requested_deg, 2),
            "moving": bool(end_pose.get("head_pose_moving", False)),
        }
    else:
        summary["robot_brain_pose"] = {
            "valid_sample_count": 0,
            "message": "Robot brain /camera/head/pose was not available.",
        }
    command_error = next((sample.get("command_error") for sample in samples if sample.get("command_error")), None)
    if command_error:
        summary["command_error"] = command_error
    return summary


def write_csv(path: Path, samples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for sample in samples for key in sample})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(samples)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record odometry yaw during passive or commanded in-place rotation.")
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--sample-hz", type=float, default=10.0)
    parser.add_argument("--imu-bias-calibration-s", type=float, default=2.0)
    parser.add_argument("--imu-bias-min-samples", type=int, default=20)
    parser.add_argument("--odom-frame", default="odom")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--imu-topic", default="/imu")
    parser.add_argument("--camera-pan-topic", default="/camera/head/pan_rad")
    parser.add_argument(
        "--imu-axis",
        choices=("x", "y", "z", "robot_yaw", "optical_yaw"),
        default="robot_yaw",
        help="Raw IMU axis or derived robot yaw source. For an aligned Gemini 2 camera, use robot_yaw.",
    )
    parser.add_argument("--angular-rad-s", type=float, default=0.10)
    parser.add_argument(
        "--target-yaw-deg",
        type=float,
        default=None,
        help="Stop early when the selected pose source reports this absolute yaw delta. --duration-s remains the safety timeout.",
    )
    parser.add_argument(
        "--target-source",
        choices=("tf", "odom", "imu"),
        default="tf",
        help="Source used by --target-yaw-deg. Use 'tf' for odom->base_link, 'odom' for /odom, or 'imu' for integrated selected IMU source.",
    )
    parser.add_argument(
        "--send-motion",
        action="store_true",
        help="Actually publish angular /cmd_vel. Without this flag the script only records TF.",
    )
    parser.add_argument(
        "--head-pan-deg",
        type=float,
        default=None,
        help="Command the camera head pan motor to this absolute angle and record reported head pan. This does not move the base.",
    )
    parser.add_argument("--head-pan-settle-s", type=float, default=1.0)
    parser.add_argument("--head-pan-action-key", default=None)
    parser.add_argument("--robot-brain-url", default="http://127.0.0.1:8765")
    parser.add_argument("--csv-out", default="artifacts/diagnostics/rotation_diagnostic.csv")
    parser.add_argument("--json-out", default="artifacts/diagnostics/rotation_diagnostic_summary.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require_ros()
    rclpy.init()
    node = RotationDiagnosticNode(
        odom_frame=args.odom_frame,
        base_frame=args.base_frame,
        cmd_vel_topic=args.cmd_vel_topic,
        odom_topic=args.odom_topic,
        imu_topic=args.imu_topic,
        camera_pan_topic=args.camera_pan_topic,
        imu_axis=args.imu_axis,
        sample_hz=args.sample_hz,
        imu_bias_calibration_s=args.imu_bias_calibration_s,
        imu_bias_min_samples=args.imu_bias_min_samples,
    )
    try:
        if args.head_pan_deg is not None:
            samples = node.collect_head_pan(
                robot_brain_url=args.robot_brain_url,
                pan_deg=args.head_pan_deg,
                settle_s=args.head_pan_settle_s,
                action_key=args.head_pan_action_key,
                duration_s=args.duration_s,
            )
        else:
            samples = node.collect(
                duration_s=args.duration_s,
                angular_rad_s=args.angular_rad_s,
                send_motion=args.send_motion,
                target_yaw_rad=math.radians(args.target_yaw_deg) if args.target_yaw_deg is not None else None,
                target_source=args.target_source,
            )
        summary = summarize(samples)
        write_csv(Path(args.csv_out).expanduser(), samples)
        json_path = Path(args.json_out).expanduser()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
        imu_reported_turn_deg = summary.get("imu", {}).get("reported_turn_deg")
        if imu_reported_turn_deg is not None:
            print(f"Reported IMU turn: {imu_reported_turn_deg} deg")
        head_pan_topic = summary.get("topic", {}).get("end_deg")
        if head_pan_topic is not None:
            print(f"Reported head pan topic: {head_pan_topic} deg")
        print(f"Wrote samples: {Path(args.csv_out).expanduser()}")
        print(f"Wrote summary: {json_path}")
        if args.head_pan_deg is not None:
            return 0 if summary.get("topic", {}).get("valid_sample_count", 0) or summary.get(
                "robot_brain_pose", {}
            ).get("valid_sample_count", 0) else 2
        return 0 if (
            summary.get("tf", {}).get("valid_pose_count", 0)
            or summary.get("odom_topic", {}).get("valid_pose_count", 0)
            or summary.get("imu", {}).get("valid_sample_count", 0)
        ) else 2
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
