from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import time
from typing import Any
from urllib import request
from urllib.parse import urljoin

IMPORT_ERROR: Exception | None = None
try:
    import rclpy
    from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion, TransformStamped
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from std_msgs.msg import Bool
    from tf2_ros import TransformBroadcaster
except Exception as exc:  # pragma: no cover - runtime guard.
    IMPORT_ERROR = exc
    rclpy = None
    PoseWithCovarianceStamped = None
    Quaternion = None
    TransformStamped = None
    Odometry = None
    Node = object
    Bool = None
    TransformBroadcaster = None


@dataclass(frozen=True)
class PlanarPose:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class WheelOdometryConfig:
    robot_brain_url: str = "http://127.0.0.1:8765"
    wheel_state_path: str = "/wheel_state"
    odom_topic: str = "/odom"
    odom_reset_topic: str = "/xlerobot/odom/set_pose"
    odom_frame: str = "odom"
    base_frame: str = "base_link"
    nav_active_topic: str = "/xlerobot/nav_active"
    odom_requires_nav_active: bool = False
    publish_rate_hz: float = 50.0
    http_timeout_s: float = 2.0
    encoder_ticks_per_revolution: float = 4096.0
    wheel_radius_m: float = 0.05
    wheel_track_width_m: float = 0.25
    base_link_x_from_wheel_axle_m: float = 0.0
    base_link_y_from_wheel_axle_m: float = 0.0
    left_wheel_motor: str = "base_left_wheel"
    right_wheel_motor: str = "base_right_wheel"
    left_wheel_position_sign: float = -1.0
    right_wheel_position_sign: float = 1.0
    max_sample_dt_s: float = 0.5
    pose_covariance_xy: float = 0.02
    pose_covariance_yaw: float = 0.05


@dataclass(frozen=True)
class WheelStateSample:
    timestamp_s: float | None
    left_position_ticks: float
    right_position_ticks: float
    left_velocity_raw: float | None = None
    right_velocity_raw: float | None = None


@dataclass(frozen=True)
class WheelOdometryStep:
    pose: PlanarPose
    forward_m: float
    yaw_delta_rad: float
    left_distance_m: float
    right_distance_m: float


def require_runtime_dependencies() -> None:
    if IMPORT_ERROR is not None:
        raise RuntimeError(
            "Wheel odometry requires ROS 2 Python packages: `rclpy`, "
            "`geometry_msgs`, `nav_msgs`, `std_msgs`, and `tf2_ros`."
        ) from IMPORT_ERROR


def angle_wrap(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def yaw_to_quaternion_xyzw(yaw_rad: float) -> tuple[float, float, float, float]:
    half = yaw_rad / 2.0
    return 0.0, 0.0, math.sin(half), math.cos(half)


def yaw_from_quaternion_xyzw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def unwrap_encoder_delta_ticks(
    current_ticks: float,
    previous_ticks: float,
    *,
    ticks_per_revolution: float,
) -> float:
    delta = float(current_ticks) - float(previous_ticks)
    ticks_per_revolution = float(ticks_per_revolution)
    if ticks_per_revolution <= 0.0:
        return delta
    half_turn = ticks_per_revolution / 2.0
    if delta > half_turn:
        delta -= ticks_per_revolution
    elif delta < -half_turn:
        delta += ticks_per_revolution
    return delta


def wheel_ticks_to_distance_m(
    delta_ticks: float,
    *,
    ticks_per_revolution: float,
    wheel_radius_m: float,
    position_sign: float,
) -> float:
    if ticks_per_revolution <= 0.0:
        raise ValueError("ticks_per_revolution must be positive.")
    wheel_circumference_m = 2.0 * math.pi * float(wheel_radius_m)
    return float(position_sign) * float(delta_ticks) / float(ticks_per_revolution) * wheel_circumference_m


def integrate_differential_drive(
    pose: PlanarPose,
    *,
    left_distance_m: float,
    right_distance_m: float,
    wheel_track_width_m: float,
) -> WheelOdometryStep:
    wheel_track_width_m = float(wheel_track_width_m)
    if wheel_track_width_m <= 0.0:
        raise ValueError("wheel_track_width_m must be positive.")
    forward_m = (float(left_distance_m) + float(right_distance_m)) / 2.0
    yaw_delta_rad = (float(right_distance_m) - float(left_distance_m)) / wheel_track_width_m
    mid_yaw = pose.yaw + yaw_delta_rad / 2.0
    new_pose = PlanarPose(
        x=pose.x + forward_m * math.cos(mid_yaw),
        y=pose.y + forward_m * math.sin(mid_yaw),
        yaw=angle_wrap(pose.yaw + yaw_delta_rad),
    )
    return WheelOdometryStep(
        pose=new_pose,
        forward_m=forward_m,
        yaw_delta_rad=yaw_delta_rad,
        left_distance_m=float(left_distance_m),
        right_distance_m=float(right_distance_m),
    )


def apply_planar_offset(pose: PlanarPose, *, x_offset_m: float, y_offset_m: float) -> PlanarPose:
    """Return the pose of a frame offset from `pose` by a body-frame x/y vector."""
    x_offset_m = float(x_offset_m)
    y_offset_m = float(y_offset_m)
    cos_yaw = math.cos(pose.yaw)
    sin_yaw = math.sin(pose.yaw)
    return PlanarPose(
        x=pose.x + cos_yaw * x_offset_m - sin_yaw * y_offset_m,
        y=pose.y + sin_yaw * x_offset_m + cos_yaw * y_offset_m,
        yaw=pose.yaw,
    )


def remove_planar_offset(pose: PlanarPose, *, x_offset_m: float, y_offset_m: float) -> PlanarPose:
    """Return the pose of the parent frame when `pose` is an offset child frame."""
    x_offset_m = float(x_offset_m)
    y_offset_m = float(y_offset_m)
    cos_yaw = math.cos(pose.yaw)
    sin_yaw = math.sin(pose.yaw)
    return PlanarPose(
        x=pose.x - (cos_yaw * x_offset_m - sin_yaw * y_offset_m),
        y=pose.y - (sin_yaw * x_offset_m + cos_yaw * y_offset_m),
        yaw=pose.yaw,
    )


def parse_wheel_state_sample(
    payload: dict[str, Any],
    *,
    left_wheel_motor: str,
    right_wheel_motor: str,
) -> WheelStateSample:
    positions = payload.get("positions_raw")
    if not isinstance(positions, dict):
        raise ValueError("Wheel state payload is missing `positions_raw`.")
    if left_wheel_motor not in positions or right_wheel_motor not in positions:
        raise ValueError(
            "Wheel state payload does not include both configured wheel motors: "
            f"{left_wheel_motor!r}, {right_wheel_motor!r}."
        )
    velocities = payload.get("velocities_raw")
    if not isinstance(velocities, dict):
        velocities = {}
    timestamp_s = payload.get("timestamp_s")
    return WheelStateSample(
        timestamp_s=None if timestamp_s is None else float(timestamp_s),
        left_position_ticks=float(positions[left_wheel_motor]),
        right_position_ticks=float(positions[right_wheel_motor]),
        left_velocity_raw=None if left_wheel_motor not in velocities else float(velocities[left_wheel_motor]),
        right_velocity_raw=None if right_wheel_motor not in velocities else float(velocities[right_wheel_motor]),
    )


def integrate_wheel_state_delta(
    pose: PlanarPose,
    *,
    previous_sample: WheelStateSample,
    current_sample: WheelStateSample,
    config: WheelOdometryConfig,
) -> WheelOdometryStep:
    left_delta_ticks = unwrap_encoder_delta_ticks(
        current_sample.left_position_ticks,
        previous_sample.left_position_ticks,
        ticks_per_revolution=config.encoder_ticks_per_revolution,
    )
    right_delta_ticks = unwrap_encoder_delta_ticks(
        current_sample.right_position_ticks,
        previous_sample.right_position_ticks,
        ticks_per_revolution=config.encoder_ticks_per_revolution,
    )
    left_distance_m = wheel_ticks_to_distance_m(
        left_delta_ticks,
        ticks_per_revolution=config.encoder_ticks_per_revolution,
        wheel_radius_m=config.wheel_radius_m,
        position_sign=config.left_wheel_position_sign,
    )
    right_distance_m = wheel_ticks_to_distance_m(
        right_delta_ticks,
        ticks_per_revolution=config.encoder_ticks_per_revolution,
        wheel_radius_m=config.wheel_radius_m,
        position_sign=config.right_wheel_position_sign,
    )
    return integrate_differential_drive(
        pose,
        left_distance_m=left_distance_m,
        right_distance_m=right_distance_m,
        wheel_track_width_m=config.wheel_track_width_m,
    )


class RobotBrainWheelStateClient:
    def __init__(self, base_url: str, *, path: str = "/wheel_state", timeout_s: float = 2.0) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.path = path
        self.timeout_s = float(timeout_s)

    def get_json(self) -> dict[str, Any]:
        with request.urlopen(urljoin(self.base_url, self.path.lstrip("/")), timeout=self.timeout_s) as response:
            data = response.read()
        if not data:
            return {}
        return json.loads(data.decode("utf-8"))


class WheelOdometryNode(Node):
    def __init__(
        self,
        config: WheelOdometryConfig,
        *,
        client: RobotBrainWheelStateClient | None = None,
    ) -> None:
        require_runtime_dependencies()
        super().__init__("xlerobot_wheel_odometry")
        self.config = config
        self.client = client or RobotBrainWheelStateClient(
            config.robot_brain_url,
            path=config.wheel_state_path,
            timeout_s=config.http_timeout_s,
        )
        self.axle_pose = PlanarPose(0.0, 0.0, 0.0)
        self._previous_sample: WheelStateSample | None = None
        self._previous_sample_monotonic_s: float | None = None
        self._latest_forward_m_s = 0.0
        self._latest_yaw_rate_rad_s = 0.0
        self._nav_active = False
        self._last_error_log_s = 0.0
        self.odom_publisher = self.create_publisher(Odometry, config.odom_topic, 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.create_subscription(PoseWithCovarianceStamped, config.odom_reset_topic, self._on_odom_reset_pose, 10)
        self.create_subscription(Bool, config.nav_active_topic, self._on_nav_active, 10)
        self.timer = self.create_timer(1.0 / max(config.publish_rate_hz, 1e-6), self.step)
        self.get_logger().info(
            "Wheel odometry ready: "
            f"brain={config.robot_brain_url.rstrip('/')}{config.wheel_state_path} "
            f"odom={config.odom_topic} frame={config.odom_frame}->{config.base_frame} "
            f"wheel_radius={config.wheel_radius_m:.3f}m track={config.wheel_track_width_m:.3f}m "
            f"base_from_axle=({config.base_link_x_from_wheel_axle_m:.3f},"
            f"{config.base_link_y_from_wheel_axle_m:.3f})m "
            f"ticks_per_rev={config.encoder_ticks_per_revolution:.1f}"
        )

    def _on_nav_active(self, message: Any) -> None:
        self._nav_active = bool(message.data)

    def _on_odom_reset_pose(self, message: Any) -> None:
        pose = message.pose.pose
        yaw = yaw_from_quaternion_xyzw(
            float(pose.orientation.x),
            float(pose.orientation.y),
            float(pose.orientation.z),
            float(pose.orientation.w),
        )
        base_pose = PlanarPose(float(pose.position.x), float(pose.position.y), angle_wrap(yaw))
        self.axle_pose = self._axle_pose_from_base_pose(base_pose)
        self._previous_sample = None
        self._previous_sample_monotonic_s = None
        self._latest_forward_m_s = 0.0
        self._latest_yaw_rate_rad_s = 0.0
        self._publish_odom(self.get_clock().now().to_msg())
        self.get_logger().info(
            "Applied wheel odom pose reset "
            f"topic={self.config.odom_reset_topic} frame={message.header.frame_id or '<empty>'} "
            f"base_x={base_pose.x:.3f} base_y={base_pose.y:.3f} "
            f"axle_x={self.axle_pose.x:.3f} axle_y={self.axle_pose.y:.3f} "
            f"yaw_deg={math.degrees(self.axle_pose.yaw):.1f}"
        )

    def step(self) -> None:
        stamp = self.get_clock().now().to_msg()
        try:
            self._poll_and_integrate()
        except Exception as exc:
            now_s = time.monotonic()
            if now_s - self._last_error_log_s >= 2.0:
                self._last_error_log_s = now_s
                self.get_logger().warning(f"Wheel odometry poll failed: {exc}")
        self._publish_odom(stamp)

    def _poll_and_integrate(self) -> None:
        sample = parse_wheel_state_sample(
            self.client.get_json(),
            left_wheel_motor=self.config.left_wheel_motor,
            right_wheel_motor=self.config.right_wheel_motor,
        )
        now_s = time.monotonic()
        previous = self._previous_sample
        previous_time_s = self._previous_sample_monotonic_s
        self._previous_sample = sample
        self._previous_sample_monotonic_s = now_s
        if previous is None or previous_time_s is None:
            self._latest_forward_m_s = 0.0
            self._latest_yaw_rate_rad_s = 0.0
            return
        dt = max(now_s - previous_time_s, 0.0)
        if dt <= 1e-6 or dt > self.config.max_sample_dt_s:
            self._latest_forward_m_s = 0.0
            self._latest_yaw_rate_rad_s = 0.0
            return
        if self.config.odom_requires_nav_active and not self._nav_active:
            self._latest_forward_m_s = 0.0
            self._latest_yaw_rate_rad_s = 0.0
            return
        step = integrate_wheel_state_delta(
            self.axle_pose,
            previous_sample=previous,
            current_sample=sample,
            config=self.config,
        )
        self.axle_pose = step.pose
        self._latest_forward_m_s = step.forward_m / dt
        self._latest_yaw_rate_rad_s = step.yaw_delta_rad / dt

    def _base_pose_from_axle_pose(self, axle_pose: PlanarPose) -> PlanarPose:
        return apply_planar_offset(
            axle_pose,
            x_offset_m=self.config.base_link_x_from_wheel_axle_m,
            y_offset_m=self.config.base_link_y_from_wheel_axle_m,
        )

    def _axle_pose_from_base_pose(self, base_pose: PlanarPose) -> PlanarPose:
        return remove_planar_offset(
            base_pose,
            x_offset_m=self.config.base_link_x_from_wheel_axle_m,
            y_offset_m=self.config.base_link_y_from_wheel_axle_m,
        )

    def _base_twist_from_axle_twist(self) -> tuple[float, float, float]:
        yaw_rate_rad_s = float(self._latest_yaw_rate_rad_s)
        linear_x_m_s = float(self._latest_forward_m_s) - yaw_rate_rad_s * float(
            self.config.base_link_y_from_wheel_axle_m
        )
        linear_y_m_s = yaw_rate_rad_s * float(self.config.base_link_x_from_wheel_axle_m)
        return linear_x_m_s, linear_y_m_s, yaw_rate_rad_s

    def _publish_odom(self, stamp: Any) -> None:
        base_pose = self._base_pose_from_axle_pose(self.axle_pose)
        linear_x_m_s, linear_y_m_s, yaw_rate_rad_s = self._base_twist_from_axle_twist()
        qx, qy, qz, qw = yaw_to_quaternion_xyzw(base_pose.yaw)
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.config.odom_frame
        odom.child_frame_id = self.config.base_frame
        odom.pose.pose.position.x = base_pose.x
        odom.pose.pose.position.y = base_pose.y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation = _quaternion_msg(qx, qy, qz, qw)
        odom.pose.covariance[0] = float(self.config.pose_covariance_xy)
        odom.pose.covariance[7] = float(self.config.pose_covariance_xy)
        odom.pose.covariance[35] = float(self.config.pose_covariance_yaw)
        odom.twist.twist.linear.x = linear_x_m_s
        odom.twist.twist.linear.y = linear_y_m_s
        odom.twist.twist.angular.z = yaw_rate_rad_s
        self.odom_publisher.publish(odom)

        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = self.config.odom_frame
        transform.child_frame_id = self.config.base_frame
        transform.transform.translation.x = base_pose.x
        transform.transform.translation.y = base_pose.y
        transform.transform.translation.z = 0.0
        transform.transform.rotation = _quaternion_msg(qx, qy, qz, qw)
        self.tf_broadcaster.sendTransform(transform)


def _quaternion_msg(x: float, y: float, z: float, w: float) -> Any:
    msg = Quaternion()
    msg.x = float(x)
    msg.y = float(y)
    msg.z = float(z)
    msg.w = float(w)
    return msg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Wheel-encoder odometry for XLeRobot differential-drive base."
    )
    parser.add_argument("--robot-brain-url", default="http://127.0.0.1:8765")
    parser.add_argument("--wheel-state-path", default="/wheel_state")
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--odom-reset-topic", default="/xlerobot/odom/set_pose")
    parser.add_argument("--odom-frame", default="odom")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--nav-active-topic", default="/xlerobot/nav_active")
    parser.add_argument(
        "--odom-requires-nav-active",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only integrate wheel deltas while nav_active is true. Default false keeps local/manual wheel motion visible.",
    )
    parser.add_argument("--publish-rate-hz", type=float, default=50.0)
    parser.add_argument("--http-timeout-s", type=float, default=2.0)
    parser.add_argument("--encoder-ticks-per-revolution", type=float, default=4096.0)
    parser.add_argument("--wheel-radius-m", type=float, default=0.0604)
    parser.add_argument("--wheel-track-width-m", "--wheelbase-m", dest="wheel_track_width_m", type=float, default=0.535)
    parser.add_argument(
        "--base-link-x-from-wheel-axle-m",
        type=float,
        default=0.0,
        help=(
            "Body-frame x offset from the driven wheel axle midpoint to base_link. "
            "Positive means base_link is forward of the axle. Keep 0.0 for Nav2 axle-centered wheel odom; "
            "model cart body extent with the Nav2 footprint instead."
        ),
    )
    parser.add_argument(
        "--base-link-y-from-wheel-axle-m",
        type=float,
        default=0.0,
        help="Body-frame y offset from the driven wheel axle midpoint to base_link.",
    )
    parser.add_argument("--left-wheel-motor", default="base_left_wheel")
    parser.add_argument("--right-wheel-motor", default="base_right_wheel")
    parser.add_argument("--left-wheel-position-sign", type=float, choices=(-1.0, 1.0), default=-1.0)
    parser.add_argument("--right-wheel-position-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument("--max-sample-dt-s", type=float, default=0.5)
    parser.add_argument("--pose-covariance-xy", type=float, default=0.02)
    parser.add_argument("--pose-covariance-yaw", type=float, default=0.05)
    return parser


def config_from_args(args: argparse.Namespace) -> WheelOdometryConfig:
    return WheelOdometryConfig(
        robot_brain_url=args.robot_brain_url,
        wheel_state_path=args.wheel_state_path,
        odom_topic=args.odom_topic,
        odom_reset_topic=args.odom_reset_topic,
        odom_frame=args.odom_frame,
        base_frame=args.base_frame,
        nav_active_topic=args.nav_active_topic,
        odom_requires_nav_active=args.odom_requires_nav_active,
        publish_rate_hz=args.publish_rate_hz,
        http_timeout_s=args.http_timeout_s,
        encoder_ticks_per_revolution=args.encoder_ticks_per_revolution,
        wheel_radius_m=args.wheel_radius_m,
        wheel_track_width_m=args.wheel_track_width_m,
        base_link_x_from_wheel_axle_m=args.base_link_x_from_wheel_axle_m,
        base_link_y_from_wheel_axle_m=args.base_link_y_from_wheel_axle_m,
        left_wheel_motor=args.left_wheel_motor,
        right_wheel_motor=args.right_wheel_motor,
        left_wheel_position_sign=args.left_wheel_position_sign,
        right_wheel_position_sign=args.right_wheel_position_sign,
        max_sample_dt_s=args.max_sample_dt_s,
        pose_covariance_xy=args.pose_covariance_xy,
        pose_covariance_yaw=args.pose_covariance_yaw,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require_runtime_dependencies()
    rclpy.init()
    node = WheelOdometryNode(config_from_args(args))
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
