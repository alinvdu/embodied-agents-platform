from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from typing import Any

IMPORT_ERROR: Exception | None = None
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Imu
    from std_msgs.msg import Float32
except Exception as exc:  # pragma: no cover - runtime guard.
    IMPORT_ERROR = exc
    rclpy = None
    Node = object
    Imu = None
    Float32 = None


@dataclass(frozen=True)
class ImuYawFilterConfig:
    imu_topic: str = "/imu"
    output_topic: str = "/imu/filtered_yaw"
    camera_pitch_topic: str = "/camera/head/pitch_rad"
    input_frame_convention: str = "camera_optical"
    output_frame: str = "base_link"
    yaw_source: str = "gyro_y"
    compensate_camera_pitch: bool = True
    min_pitch_compensation_cos: float = 0.25
    bias_calibration_s: float = 2.0
    bias_min_samples: int = 20
    yaw_rate_lowpass_alpha: float = 0.2


def require_ros() -> None:
    if IMPORT_ERROR is not None:
        raise RuntimeError("IMU yaw filter requires ROS 2 Python packages.") from IMPORT_ERROR


def stamp_to_seconds(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) / 1_000_000_000.0


def yaw_to_quaternion_xyzw(yaw_rad: float) -> tuple[float, float, float, float]:
    half = yaw_rad / 2.0
    return 0.0, 0.0, math.sin(half), math.cos(half)


def imu_to_base_components(
    *,
    frame_convention: str,
    x: float,
    y: float,
    z: float,
) -> tuple[float, float, float]:
    normalized = str(frame_convention).strip().lower()
    if normalized == "base_link":
        return x, y, z
    if normalized == "camera_optical":
        # ROS base_link: X forward, Y left, Z up
        # Camera optical: X right, Y down, Z forward
        return z, -x, -y
    raise ValueError(f"Unsupported IMU frame convention: {frame_convention}")


def yaw_rate_from_source(
    *,
    yaw_source: str,
    raw_x: float,
    raw_y: float,
    raw_z: float,
    base_x: float,
    base_y: float,
    base_z: float,
) -> float:
    normalized = str(yaw_source).strip().lower()
    if normalized == "gyro_y":
        return raw_y
    if normalized == "gyro_neg_y":
        return -raw_y
    if normalized == "gyro_z":
        return raw_z
    if normalized == "base_link_z":
        return base_z
    raise ValueError(f"Unsupported yaw source: {yaw_source}")


def compensate_yaw_rate_for_pitch(
    yaw_rate_rad_s: float,
    *,
    yaw_source: str,
    pitch_rad: float,
    enabled: bool,
    min_cos: float,
) -> float:
    if not enabled:
        return yaw_rate_rad_s
    normalized = str(yaw_source).strip().lower()
    if normalized not in {"gyro_y", "gyro_neg_y"}:
        return yaw_rate_rad_s
    cos_pitch = abs(math.cos(float(pitch_rad)))
    clamped_cos = max(cos_pitch, max(float(min_cos), 1e-6))
    return yaw_rate_rad_s / clamped_cos


def angle_wrap(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


class ImuYawFilterNode(Node):
    def __init__(self, config: ImuYawFilterConfig) -> None:
        require_ros()
        super().__init__("xlerobot_imu_yaw_filter")
        self.config = config
        self.publisher = self.create_publisher(Imu, config.output_topic, 50)
        self.create_subscription(Imu, config.imu_topic, self._on_imu, 50)
        self.create_subscription(Float32, config.camera_pitch_topic, self._on_camera_pitch, 10)
        self._camera_pitch_rad = 0.0
        self._bias_started_s: float | None = None
        self._last_stamp_s: float | None = None
        self._sample_count = 0
        self._sum_gyro_x = 0.0
        self._sum_gyro_y = 0.0
        self._sum_gyro_z = 0.0
        self._bias_x = 0.0
        self._bias_y = 0.0
        self._bias_z = 0.0
        self._bias_ready = config.bias_calibration_s <= 1e-6
        self._filtered_yaw_rate_rad_s = 0.0
        self._yaw_rad = 0.0
        self.get_logger().info(
            f"IMU yaw filter ready: input={config.imu_topic} output={config.output_topic} "
            f"frame_convention={config.input_frame_convention} yaw_source={config.yaw_source} "
            f"camera_pitch_topic={config.camera_pitch_topic} "
            f"compensate_camera_pitch={config.compensate_camera_pitch}"
        )

    def _on_camera_pitch(self, message: Any) -> None:
        self._camera_pitch_rad = float(message.data)

    def _on_imu(self, message: Any) -> None:
        if message.header.stamp is None:
            return
        stamp_s = stamp_to_seconds(message.header.stamp)
        base_accel_x, base_accel_y, base_accel_z = imu_to_base_components(
            frame_convention=self.config.input_frame_convention,
            x=float(message.linear_acceleration.x),
            y=float(message.linear_acceleration.y),
            z=float(message.linear_acceleration.z),
        )
        raw_gyro_x = float(message.angular_velocity.x)
        raw_gyro_y = float(message.angular_velocity.y)
        raw_gyro_z = float(message.angular_velocity.z)
        base_gyro_x, base_gyro_y, base_gyro_z = imu_to_base_components(
            frame_convention=self.config.input_frame_convention,
            x=raw_gyro_x,
            y=raw_gyro_y,
            z=raw_gyro_z,
        )

        if not self._bias_ready:
            if self._bias_started_s is None:
                self._bias_started_s = stamp_s
            if self._last_stamp_s is not None and stamp_s <= self._last_stamp_s:
                return
            self._last_stamp_s = stamp_s
            self._sum_gyro_x += raw_gyro_x
            self._sum_gyro_y += raw_gyro_y
            self._sum_gyro_z += raw_gyro_z
            self._sample_count += 1
            elapsed_s = stamp_s - self._bias_started_s
            if elapsed_s >= self.config.bias_calibration_s and self._sample_count >= self.config.bias_min_samples:
                self._bias_x = self._sum_gyro_x / self._sample_count
                self._bias_y = self._sum_gyro_y / self._sample_count
                self._bias_z = self._sum_gyro_z / self._sample_count
                self._bias_ready = True
                self._last_stamp_s = stamp_s
                self.get_logger().info(
                    "IMU yaw bias calibrated: "
                    f"x={self._bias_x:.6f} y={self._bias_y:.6f} z={self._bias_z:.6f} "
                    f"from {self._sample_count} samples over {elapsed_s:.2f}s"
                )
            return

        if self._last_stamp_s is None:
            self._last_stamp_s = stamp_s
            return
        dt = max(0.0, min(stamp_s - self._last_stamp_s, 0.5))
        self._last_stamp_s = stamp_s
        if dt <= 1e-6:
            return

        yaw_rate_rad_s = yaw_rate_from_source(
            yaw_source=self.config.yaw_source,
            raw_x=raw_gyro_x - self._bias_x,
            raw_y=raw_gyro_y - self._bias_y,
            raw_z=raw_gyro_z - self._bias_z,
            base_x=base_gyro_x - self._bias_x,
            base_y=base_gyro_y - self._bias_y,
            base_z=base_gyro_z - self._bias_z,
        )
        yaw_rate_rad_s = compensate_yaw_rate_for_pitch(
            yaw_rate_rad_s,
            yaw_source=self.config.yaw_source,
            pitch_rad=self._camera_pitch_rad,
            enabled=self.config.compensate_camera_pitch,
            min_cos=self.config.min_pitch_compensation_cos,
        )
        alpha = max(0.0, min(self.config.yaw_rate_lowpass_alpha, 1.0))
        self._filtered_yaw_rate_rad_s = (
            alpha * yaw_rate_rad_s + (1.0 - alpha) * self._filtered_yaw_rate_rad_s
        )
        self._yaw_rad = angle_wrap(self._yaw_rad + self._filtered_yaw_rate_rad_s * dt)

        filtered = Imu()
        filtered.header = message.header
        filtered.header.frame_id = self.config.output_frame
        qx, qy, qz, qw = yaw_to_quaternion_xyzw(self._yaw_rad)
        filtered.orientation.x = qx
        filtered.orientation.y = qy
        filtered.orientation.z = qz
        filtered.orientation.w = qw
        filtered.orientation_covariance[0] = 0.05
        filtered.orientation_covariance[4] = 0.05
        filtered.orientation_covariance[8] = 0.10
        filtered.angular_velocity.x = 0.0
        filtered.angular_velocity.y = 0.0
        filtered.angular_velocity.z = self._filtered_yaw_rate_rad_s
        filtered.angular_velocity_covariance[0] = 0.02
        filtered.angular_velocity_covariance[4] = 0.02
        filtered.angular_velocity_covariance[8] = 0.02
        filtered.linear_acceleration.x = base_accel_x
        filtered.linear_acceleration.y = base_accel_y
        filtered.linear_acceleration.z = base_accel_z
        filtered.linear_acceleration_covariance[0] = 0.1
        filtered.linear_acceleration_covariance[4] = 0.1
        filtered.linear_acceleration_covariance[8] = 0.1
        self.publisher.publish(filtered)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bias-correct and low-pass Gemini 2 IMU into a yaw-oriented IMU topic for 2D robot use."
    )
    parser.add_argument("--imu-topic", default="/imu")
    parser.add_argument("--output-topic", default="/imu/filtered_yaw")
    parser.add_argument("--camera-pitch-topic", default="/camera/head/pitch_rad")
    parser.add_argument(
        "--input-frame-convention",
        choices=("camera_optical", "base_link"),
        default="camera_optical",
    )
    parser.add_argument(
        "--yaw-source",
        choices=("gyro_y", "gyro_neg_y", "gyro_z", "base_link_z"),
        default="gyro_y",
        help="Yaw-rate source. For Gemini 2 Viewer-style integration, use gyro_y.",
    )
    parser.add_argument("--output-frame", default="base_link")
    parser.add_argument("--compensate-camera-pitch", dest="compensate_camera_pitch", action="store_true", default=True)
    parser.add_argument("--no-compensate-camera-pitch", dest="compensate_camera_pitch", action="store_false")
    parser.add_argument("--min-pitch-compensation-cos", type=float, default=0.25)
    parser.add_argument("--bias-calibration-s", type=float, default=2.0)
    parser.add_argument("--bias-min-samples", type=int, default=20)
    parser.add_argument("--yaw-rate-lowpass-alpha", type=float, default=0.2)
    return parser


def config_from_args(args: argparse.Namespace) -> ImuYawFilterConfig:
    return ImuYawFilterConfig(
        imu_topic=args.imu_topic,
        output_topic=args.output_topic,
        camera_pitch_topic=args.camera_pitch_topic,
        input_frame_convention=args.input_frame_convention,
        yaw_source=args.yaw_source,
        compensate_camera_pitch=args.compensate_camera_pitch,
        min_pitch_compensation_cos=args.min_pitch_compensation_cos,
        output_frame=args.output_frame,
        bias_calibration_s=args.bias_calibration_s,
        bias_min_samples=args.bias_min_samples,
        yaw_rate_lowpass_alpha=args.yaw_rate_lowpass_alpha,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require_ros()
    rclpy.init()
    node = ImuYawFilterNode(config_from_args(args))
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
