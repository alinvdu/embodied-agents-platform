from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from typing import Any

IMPORT_ERROR: Exception | None = None
CV_IMPORT_ERROR: Exception | None = None
try:
    import numpy as np
except Exception as exc:  # pragma: no cover - runtime guard.
    CV_IMPORT_ERROR = exc
    np = None

try:
    import cv2
except Exception as exc:  # pragma: no cover - runtime guard.
    CV_IMPORT_ERROR = exc
    cv2 = None

try:
    import rclpy
    from geometry_msgs.msg import Quaternion, TransformStamped
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from sensor_msgs.msg import CameraInfo, Image, Imu
    from std_msgs.msg import Bool, Float32
    from tf2_ros import TransformBroadcaster
except Exception as exc:  # pragma: no cover - runtime guard.
    IMPORT_ERROR = exc
    rclpy = None
    Quaternion = None
    TransformStamped = None
    Odometry = None
    Node = object
    CameraInfo = None
    Image = None
    Imu = None
    Bool = None
    Float32 = None
    TransformBroadcaster = None


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class RgbdVoConfig:
    rgb_topic: str = "/camera/head/image_raw"
    depth_topic: str = "/camera/head/depth/image_raw"
    camera_info_topic: str = "/camera/head/camera_info"
    imu_topic: str = "/imu"
    odom_topic: str = "/odom"
    odom_frame: str = "odom"
    base_frame: str = "base_link"
    camera_frame: str = "head_camera_link"
    camera_pitch_topic: str = "/camera/head/pitch_rad"
    camera_pan_topic: str = "/camera/head/pan_rad"
    scan_active_topic: str = "/xlerobot/scan_active"
    freeze_orientation_during_scan: bool = True
    head_pan_freeze_threshold_rad: float = math.radians(1.0)
    head_pitch_freeze_threshold_rad: float = math.radians(1.0)
    publish_rate_hz: float = 30.0
    min_depth_m: float = 0.15
    max_depth_m: float = 4.0
    min_matches: int = 20
    min_inliers: int = 12
    max_translation_step_m: float = 0.25
    min_translation_update_m: float = 0.005
    jitter_threshold: bool = True
    max_yaw_step_rad: float = math.radians(30.0)
    imu_stale_after_s: float = 0.5
    imu_frame_convention: str = "camera_optical"
    odom_yaw_sign: float = 1.0
    imu_bias_calibration_s: float = 2.0
    imu_bias_min_samples: int = 20
    camera_x_m: float = 0.0
    camera_y_m: float = 0.0
    camera_yaw_rad: float = 0.0
    camera_pitch_rad: float = 0.0


@dataclass(frozen=True)
class RgbdVoDiagnostics:
    accepted: int = 0
    rejected: int = 0
    total_translation_m: float = 0.0
    total_abs_translation_m: float = 0.0
    last_translation_m: float = 0.0
    min_abs_translation_m: float | None = None
    max_abs_translation_m: float = 0.0
    last_matches: int = 0
    last_inliers: int = 0
    missing_descriptors: int = 0
    too_few_matches: int = 0
    too_few_depth_points: int = 0
    too_few_inliers: int = 0
    translation_step_too_large: int = 0
    translation_step_too_small: int = 0
    yaw_step_too_large: int = 0
    exceptions: int = 0


@dataclass(frozen=True)
class PlanarPose:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class VisualOdomFrame:
    gray: Any
    depth_m: Any
    stamp: Any
    intrinsics: CameraIntrinsics


@dataclass(frozen=True)
class VisualOdomEstimate:
    translation_m: float
    camera_translation_x_m: float
    camera_translation_y_m: float
    camera_translation_z_m: float
    yaw_rad: float
    matches: int
    inliers: int


@dataclass(frozen=True)
class VisualOdomRejection:
    reason: str
    matches: int = 0
    object_points: int = 0
    inliers: int = 0


def require_runtime_dependencies() -> None:
    if IMPORT_ERROR is not None:
        raise RuntimeError(
            "RGB-D visual odometry requires ROS 2 Python packages: `rclpy`, "
            "`sensor_msgs`, `nav_msgs`, `geometry_msgs`, and `tf2_ros`."
        ) from IMPORT_ERROR
    if CV_IMPORT_ERROR is not None or cv2 is None or np is None:
        raise RuntimeError("RGB-D visual odometry requires `opencv-python`/`cv2` and `numpy`.") from CV_IMPORT_ERROR


def angle_wrap(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def yaw_to_quaternion_xyzw(yaw_rad: float) -> tuple[float, float, float, float]:
    half = yaw_rad / 2.0
    return 0.0, 0.0, math.sin(half), math.cos(half)


def yaw_from_quaternion_xyzw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def compose_planar(pose: PlanarPose, delta_forward_m: float, delta_yaw_rad: float) -> PlanarPose:
    mid_yaw = pose.yaw + delta_yaw_rad * 0.5
    return PlanarPose(
        x=pose.x + delta_forward_m * math.cos(mid_yaw),
        y=pose.y + delta_forward_m * math.sin(mid_yaw),
        yaw=angle_wrap(pose.yaw + delta_yaw_rad),
    )


def compose_planar_local(pose: PlanarPose, *, delta_x_m: float, delta_y_m: float, delta_yaw_rad: float) -> PlanarPose:
    mid_yaw = pose.yaw + delta_yaw_rad * 0.5
    cos_yaw = math.cos(mid_yaw)
    sin_yaw = math.sin(mid_yaw)
    world_dx = delta_x_m * cos_yaw - delta_y_m * sin_yaw
    world_dy = delta_x_m * sin_yaw + delta_y_m * cos_yaw
    return PlanarPose(
        x=pose.x + world_dx,
        y=pose.y + world_dy,
        yaw=angle_wrap(pose.yaw + delta_yaw_rad),
    )


def camera_optical_translation_to_base_planar(
    *,
    camera_x_m: float,
    camera_y_m: float,
    camera_z_m: float,
    pitch_rad: float,
) -> tuple[float, float]:
    base_forward_m = camera_z_m * math.cos(pitch_rad) - camera_y_m * math.sin(pitch_rad)
    base_left_m = -camera_x_m
    return base_forward_m, base_left_m


def stamp_to_seconds(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) / 1_000_000_000.0


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


def intrinsics_from_camera_info(message: Any) -> CameraIntrinsics:
    return CameraIntrinsics(
        fx=float(message.k[0]),
        fy=float(message.k[4]),
        cx=float(message.k[2]),
        cy=float(message.k[5]),
    )


def image_to_array(message: Any) -> Any:
    require_runtime_dependencies()
    height = int(message.height)
    width = int(message.width)
    encoding = str(message.encoding).lower()
    data = np.frombuffer(bytes(message.data), dtype=np.uint8)
    if encoding in ("rgb8", "bgr8"):
        array = data.reshape((height, width, 3))
        if encoding == "rgb8":
            return cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        return cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    if encoding in ("mono8", "8uc1"):
        return data.reshape((height, width))
    raise ValueError(f"Unsupported RGB image encoding for VO: {message.encoding}")


def depth_to_meters(message: Any) -> Any:
    require_runtime_dependencies()
    height = int(message.height)
    width = int(message.width)
    encoding = str(message.encoding).lower()
    if encoding in ("mono16", "16uc1"):
        dtype = ">u2" if bool(message.is_bigendian) else "<u2"
        raw = np.frombuffer(bytes(message.data), dtype=np.dtype(dtype)).reshape((height, width))
        return raw.astype(np.float32) / 1000.0
    if encoding == "32fc1":
        dtype = ">f4" if bool(message.is_bigendian) else "<f4"
        return np.frombuffer(bytes(message.data), dtype=np.dtype(dtype)).reshape((height, width)).astype(np.float32)
    raise ValueError(f"Unsupported depth image encoding for VO: {message.encoding}")


class FeatureRgbdOdometry:
    def __init__(self, config: RgbdVoConfig) -> None:
        require_runtime_dependencies()
        self.config = config
        self.detector = cv2.ORB_create(nfeatures=800)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def estimate(self, previous: VisualOdomFrame, current: VisualOdomFrame) -> VisualOdomEstimate | VisualOdomRejection:
        prev_keypoints, prev_descriptors = self.detector.detectAndCompute(previous.gray, None)
        curr_keypoints, curr_descriptors = self.detector.detectAndCompute(current.gray, None)
        if prev_descriptors is None or curr_descriptors is None:
            return VisualOdomRejection("missing_descriptors")
        matches = sorted(self.matcher.match(prev_descriptors, curr_descriptors), key=lambda item: item.distance)
        if len(matches) < self.config.min_matches:
            return VisualOdomRejection("too_few_matches", matches=len(matches))

        object_points: list[list[float]] = []
        image_points: list[list[float]] = []
        intrinsics = previous.intrinsics
        for match in matches[: min(len(matches), 200)]:
            prev_u, prev_v = prev_keypoints[match.queryIdx].pt
            curr_u, curr_v = curr_keypoints[match.trainIdx].pt
            u = int(round(prev_u))
            v = int(round(prev_v))
            if v < 0 or u < 0 or v >= previous.depth_m.shape[0] or u >= previous.depth_m.shape[1]:
                continue
            z = float(previous.depth_m[v, u])
            if not math.isfinite(z) or z < self.config.min_depth_m or z > self.config.max_depth_m:
                continue
            x = (prev_u - intrinsics.cx) * z / intrinsics.fx
            y = (prev_v - intrinsics.cy) * z / intrinsics.fy
            object_points.append([x, y, z])
            image_points.append([curr_u, curr_v])

        if len(object_points) < self.config.min_matches:
            return VisualOdomRejection(
                "too_few_depth_points",
                matches=len(matches),
                object_points=len(object_points),
            )

        camera_matrix = np.array(
            [
                [current.intrinsics.fx, 0.0, current.intrinsics.cx],
                [0.0, current.intrinsics.fy, current.intrinsics.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            np.asarray(object_points, dtype=np.float32),
            np.asarray(image_points, dtype=np.float32),
            camera_matrix,
            None,
            iterationsCount=80,
            reprojectionError=5.0,
            confidence=0.99,
        )
        inlier_count = 0 if inliers is None else int(len(inliers))
        if not success or inlier_count < self.config.min_inliers:
            return VisualOdomRejection(
                "too_few_inliers",
                matches=len(matches),
                object_points=len(object_points),
                inliers=inlier_count,
            )

        rotation, _ = cv2.Rodrigues(rvec)
        # solvePnP returns previous-camera coordinates expressed in the current
        # camera frame. Invert it to get current camera motion in previous frame.
        prev_from_curr = rotation.T
        prev_t_curr = -prev_from_curr @ tvec.reshape(3)
        camera_forward_m = float(prev_t_curr[2])
        camera_yaw_rad = float(math.atan2(prev_from_curr[0, 2], prev_from_curr[2, 2]))
        if abs(camera_forward_m) > self.config.max_translation_step_m:
            return VisualOdomRejection(
                "translation_step_too_large",
                matches=len(matches),
                object_points=len(object_points),
                inliers=inlier_count,
            )
        if self.config.jitter_threshold and abs(camera_forward_m) < self.config.min_translation_update_m:
            return VisualOdomRejection(
                "translation_step_too_small",
                matches=len(matches),
                object_points=len(object_points),
                inliers=inlier_count,
            )
        if abs(camera_yaw_rad) > self.config.max_yaw_step_rad:
            return VisualOdomRejection(
                "yaw_step_too_large",
                matches=len(matches),
                object_points=len(object_points),
                inliers=inlier_count,
            )
        return VisualOdomEstimate(
            translation_m=camera_forward_m,
            camera_translation_x_m=float(prev_t_curr[0]),
            camera_translation_y_m=float(prev_t_curr[1]),
            camera_translation_z_m=float(prev_t_curr[2]),
            yaw_rad=camera_yaw_rad,
            matches=len(matches),
            inliers=inlier_count,
        )


class RgbdVisualOdometryNode(Node):
    def __init__(self, config: RgbdVoConfig) -> None:
        require_runtime_dependencies()
        super().__init__("xlerobot_rgbd_visual_odometry")
        self.config = config
        self.estimator = FeatureRgbdOdometry(config)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.odom_publisher = self.create_publisher(Odometry, config.odom_topic, 10)
        self.latest_rgb: Any | None = None
        self.latest_depth: Any | None = None
        self.latest_imu: Any | None = None
        self.intrinsics: CameraIntrinsics | None = None
        self.camera_pitch_rad = float(config.camera_pitch_rad)
        self.camera_pan_rad = 0.0
        self._last_camera_pitch_rad = self.camera_pitch_rad
        self._last_camera_pan_rad = self.camera_pan_rad
        self._last_head_motion_s: float | None = None
        self._scan_orientation_frozen = False
        self.previous_frame: VisualOdomFrame | None = None
        self.pose = PlanarPose(0.0, 0.0, 0.0)
        self.planar_velocity_x_m_s = 0.0
        self.planar_velocity_y_m_s = 0.0
        self._last_prediction_stamp_s: float | None = None
        self._imu_bias_started_s: float | None = None
        self._imu_bias_last_stamp_s: float | None = None
        self._imu_bias_sample_count = 0
        self._imu_bias_sum_x = 0.0
        self._imu_bias_sum_y = 0.0
        self._imu_bias_sum_z = 0.0
        self._imu_bias_ready = self.config.imu_bias_calibration_s <= 1e-6
        self._imu_bias_x_rad_s = 0.0
        self._imu_bias_y_rad_s = 0.0
        self._imu_bias_z_rad_s = 0.0
        self._latest_imu_stamp_s: float | None = None
        self._latest_imu_orientation_yaw_rad: float | None = None
        self._latest_imu_orientation_unwrapped_yaw_rad: float | None = None
        self._imu_orientation_origin_yaw_rad: float | None = None
        self._latest_imu_received_s: float | None = None
        self.accepted_updates = 0
        self.rejected_updates = 0
        self._diagnostics = RgbdVoDiagnostics()
        self._last_diagnostics_log_s: float | None = None
        self._last_scan_freeze_log_s: float | None = None
        self.create_subscription(Image, config.rgb_topic, self._on_rgb, 10)
        self.create_subscription(Image, config.depth_topic, self._on_depth, 10)
        self.create_subscription(CameraInfo, config.camera_info_topic, self._on_camera_info, 10)
        self.create_subscription(Float32, config.camera_pitch_topic, self._on_camera_pitch, 10)
        self.create_subscription(Float32, config.camera_pan_topic, self._on_camera_pan, 10)
        self.create_subscription(Bool, config.scan_active_topic, self._on_scan_active, 10)
        self.create_subscription(Imu, config.imu_topic, self._on_imu, 50)
        self.create_timer(1.0 / max(config.publish_rate_hz, 1e-6), self.step)
        self.get_logger().info(
            "RGB-D visual odometry ready: "
            f"rgb={config.rgb_topic} depth={config.depth_topic} camera_info={config.camera_info_topic} "
            f"camera_pitch={config.camera_pitch_topic} camera_pan={config.camera_pan_topic} "
            f"scan_active={config.scan_active_topic} "
            f"freeze_orientation_during_scan={config.freeze_orientation_during_scan} "
            f"imu={config.imu_topic} odom={config.odom_topic}"
        )

    def _record_estimate(self, estimate: VisualOdomEstimate) -> None:
        self._diagnostics = RgbdVoDiagnostics(
            accepted=self._diagnostics.accepted + 1,
            rejected=self._diagnostics.rejected,
            total_translation_m=self._diagnostics.total_translation_m + estimate.translation_m,
            total_abs_translation_m=self._diagnostics.total_abs_translation_m + abs(estimate.translation_m),
            last_translation_m=estimate.translation_m,
            min_abs_translation_m=(
                abs(estimate.translation_m)
                if self._diagnostics.min_abs_translation_m is None
                else min(self._diagnostics.min_abs_translation_m, abs(estimate.translation_m))
            ),
            max_abs_translation_m=max(self._diagnostics.max_abs_translation_m, abs(estimate.translation_m)),
            last_matches=estimate.matches,
            last_inliers=estimate.inliers,
            missing_descriptors=self._diagnostics.missing_descriptors,
            too_few_matches=self._diagnostics.too_few_matches,
            too_few_depth_points=self._diagnostics.too_few_depth_points,
            too_few_inliers=self._diagnostics.too_few_inliers,
            translation_step_too_large=self._diagnostics.translation_step_too_large,
            translation_step_too_small=self._diagnostics.translation_step_too_small,
            yaw_step_too_large=self._diagnostics.yaw_step_too_large,
            exceptions=self._diagnostics.exceptions,
        )

    def _record_rejection(self, rejection: VisualOdomRejection) -> None:
        counts = {
            "missing_descriptors": self._diagnostics.missing_descriptors,
            "too_few_matches": self._diagnostics.too_few_matches,
            "too_few_depth_points": self._diagnostics.too_few_depth_points,
            "too_few_inliers": self._diagnostics.too_few_inliers,
            "translation_step_too_large": self._diagnostics.translation_step_too_large,
            "translation_step_too_small": self._diagnostics.translation_step_too_small,
            "yaw_step_too_large": self._diagnostics.yaw_step_too_large,
        }
        if rejection.reason in counts:
            counts[rejection.reason] += 1
        self._diagnostics = RgbdVoDiagnostics(
            accepted=self._diagnostics.accepted,
            rejected=self._diagnostics.rejected + 1,
            total_translation_m=self._diagnostics.total_translation_m,
            total_abs_translation_m=self._diagnostics.total_abs_translation_m,
            last_translation_m=self._diagnostics.last_translation_m,
            min_abs_translation_m=self._diagnostics.min_abs_translation_m,
            max_abs_translation_m=self._diagnostics.max_abs_translation_m,
            last_matches=rejection.matches,
            last_inliers=rejection.inliers,
            missing_descriptors=counts["missing_descriptors"],
            too_few_matches=counts["too_few_matches"],
            too_few_depth_points=counts["too_few_depth_points"],
            too_few_inliers=counts["too_few_inliers"],
            translation_step_too_large=counts["translation_step_too_large"],
            translation_step_too_small=counts["translation_step_too_small"],
            yaw_step_too_large=counts["yaw_step_too_large"],
            exceptions=self._diagnostics.exceptions,
        )

    def _record_exception(self) -> None:
        self._diagnostics = RgbdVoDiagnostics(
            accepted=self._diagnostics.accepted,
            rejected=self._diagnostics.rejected + 1,
            total_translation_m=self._diagnostics.total_translation_m,
            total_abs_translation_m=self._diagnostics.total_abs_translation_m,
            last_translation_m=self._diagnostics.last_translation_m,
            min_abs_translation_m=self._diagnostics.min_abs_translation_m,
            max_abs_translation_m=self._diagnostics.max_abs_translation_m,
            last_matches=self._diagnostics.last_matches,
            last_inliers=self._diagnostics.last_inliers,
            missing_descriptors=self._diagnostics.missing_descriptors,
            too_few_matches=self._diagnostics.too_few_matches,
            too_few_depth_points=self._diagnostics.too_few_depth_points,
            too_few_inliers=self._diagnostics.too_few_inliers,
            translation_step_too_large=self._diagnostics.translation_step_too_large,
            translation_step_too_small=self._diagnostics.translation_step_too_small,
            yaw_step_too_large=self._diagnostics.yaw_step_too_large,
            exceptions=self._diagnostics.exceptions + 1,
        )

    def _log_diagnostics(self, *, now_s: float) -> None:
        if self._last_diagnostics_log_s is not None and now_s - self._last_diagnostics_log_s < 2.0:
            return
        self._last_diagnostics_log_s = now_s
        diagnostics = self._diagnostics
        accepted = max(diagnostics.accepted, 1)
        min_abs_dx = 0.0 if diagnostics.min_abs_translation_m is None else diagnostics.min_abs_translation_m
        self.get_logger().info(
            "RGB-D VO stats: "
            f"accepted={diagnostics.accepted} rejected={diagnostics.rejected} "
            f"pose_x={self.pose.x:.3f} pose_y={self.pose.y:.3f} yaw_deg={math.degrees(self.pose.yaw):.1f} "
            f"sum_dx={diagnostics.total_translation_m:.3f} "
            f"mean_abs_dx={diagnostics.total_abs_translation_m / accepted:.4f} "
            f"min_abs_dx={min_abs_dx:.4f} max_abs_dx={diagnostics.max_abs_translation_m:.4f} "
            f"last_dx={diagnostics.last_translation_m:.4f} "
            f"last_matches={diagnostics.last_matches} last_inliers={diagnostics.last_inliers} "
            "rejects="
            f"desc:{diagnostics.missing_descriptors},"
            f"matches:{diagnostics.too_few_matches},"
            f"depth:{diagnostics.too_few_depth_points},"
            f"inliers:{diagnostics.too_few_inliers},"
            f"big_step:{diagnostics.translation_step_too_large},"
            f"small_step:{diagnostics.translation_step_too_small},"
            f"yaw:{diagnostics.yaw_step_too_large},"
            f"exceptions:{diagnostics.exceptions}"
        )

    def _on_rgb(self, message: Any) -> None:
        self.latest_rgb = message

    def _on_depth(self, message: Any) -> None:
        self.latest_depth = message

    def _on_camera_info(self, message: Any) -> None:
        self.intrinsics = intrinsics_from_camera_info(message)

    def _on_camera_pitch(self, message: Any) -> None:
        pitch_rad = float(message.data)
        if abs(pitch_rad - self._last_camera_pitch_rad) >= self.config.head_pitch_freeze_threshold_rad:
            self._last_head_motion_s = self.get_clock().now().nanoseconds / 1_000_000_000.0
        self.camera_pitch_rad = pitch_rad
        self._last_camera_pitch_rad = pitch_rad

    def _on_camera_pan(self, message: Any) -> None:
        pan_rad = float(message.data)
        if abs(pan_rad - self._last_camera_pan_rad) >= self.config.head_pan_freeze_threshold_rad:
            self._last_head_motion_s = self.get_clock().now().nanoseconds / 1_000_000_000.0
        self.camera_pan_rad = pan_rad
        self._last_camera_pan_rad = pan_rad

    def _on_scan_active(self, message: Any) -> None:
        active = bool(message.data)
        if active == self._scan_orientation_frozen:
            return
        self._scan_orientation_frozen = active
        self.previous_frame = None
        self._last_prediction_stamp_s = None
        if active:
            self.get_logger().info("Freezing odom orientation while scan_active is true; translation updates remain enabled.")
        else:
            self._reset_imu_origin_to_current_pose()
            self.get_logger().info("Resuming odom orientation after scan_active returned false; reset VO keyframe and IMU yaw origin.")

    def _on_imu(self, message: Any) -> None:
        self.latest_imu = message
        self._latest_imu_received_s = self.get_clock().now().nanoseconds / 1_000_000_000.0
        if message.header.stamp is None:
            return
        stamp_s = stamp_to_seconds(message.header.stamp)
        self._latest_imu_stamp_s = stamp_s
        covariance = getattr(message, "orientation_covariance", None)
        if covariance is not None and len(covariance) >= 1 and float(covariance[0]) >= 0.0:
            orientation = message.orientation
            yaw_rad = yaw_from_quaternion_xyzw(
                float(orientation.x),
                float(orientation.y),
                float(orientation.z),
                float(orientation.w),
            )
            if self._latest_imu_orientation_yaw_rad is None:
                self._latest_imu_orientation_unwrapped_yaw_rad = yaw_rad
                self._imu_orientation_origin_yaw_rad = yaw_rad
            else:
                assert self._latest_imu_orientation_unwrapped_yaw_rad is not None
                self._latest_imu_orientation_unwrapped_yaw_rad += math.atan2(
                    math.sin(yaw_rad - self._latest_imu_orientation_yaw_rad),
                    math.cos(yaw_rad - self._latest_imu_orientation_yaw_rad),
                )
            self._latest_imu_orientation_yaw_rad = yaw_rad
        if self._imu_bias_ready:
            return
        if self._imu_bias_started_s is None:
            self._imu_bias_started_s = stamp_s
        if self._imu_bias_last_stamp_s is not None and stamp_s <= self._imu_bias_last_stamp_s:
            return
        self._imu_bias_last_stamp_s = stamp_s
        self._imu_bias_sum_x += float(message.angular_velocity.x)
        self._imu_bias_sum_y += float(message.angular_velocity.y)
        self._imu_bias_sum_z += float(message.angular_velocity.z)
        self._imu_bias_sample_count += 1
        elapsed_s = stamp_s - self._imu_bias_started_s
        if (
            elapsed_s >= self.config.imu_bias_calibration_s
            and self._imu_bias_sample_count >= self.config.imu_bias_min_samples
        ):
            self._imu_bias_x_rad_s = self._imu_bias_sum_x / self._imu_bias_sample_count
            self._imu_bias_y_rad_s = self._imu_bias_sum_y / self._imu_bias_sample_count
            self._imu_bias_z_rad_s = self._imu_bias_sum_z / self._imu_bias_sample_count
            self._imu_bias_ready = True
            self.get_logger().info(
                "IMU gyro bias calibrated: "
                f"x={self._imu_bias_x_rad_s:.6f} y={self._imu_bias_y_rad_s:.6f} z={self._imu_bias_z_rad_s:.6f} "
                f"from {self._imu_bias_sample_count} samples over {elapsed_s:.2f}s"
            )

    def _latest_imu_age_s(self) -> float | None:
        if self._latest_imu_received_s is None:
            return None
        now_s = self.get_clock().now().nanoseconds / 1_000_000_000.0
        return max(now_s - self._latest_imu_received_s, 0.0)

    def _relative_imu_yaw_rad(self) -> float | None:
        age_s = self._latest_imu_age_s()
        if age_s is None or age_s > self.config.imu_stale_after_s:
            return None
        if self._latest_imu_orientation_unwrapped_yaw_rad is None or self._imu_orientation_origin_yaw_rad is None:
            return None
        return float(getattr(self.config, "odom_yaw_sign", 1.0)) * (
            self._latest_imu_orientation_unwrapped_yaw_rad - self._imu_orientation_origin_yaw_rad
        )

    def _predict_yaw_from_imu(self, *, stamp_s: float) -> float:
        if not self._imu_bias_ready:
            self._last_prediction_stamp_s = stamp_s
            return 0.0
        if self.latest_imu is None or self.latest_imu.header.stamp is None:
            self._last_prediction_stamp_s = stamp_s
            return 0.0
        age_s = self._latest_imu_age_s()
        if age_s is None or age_s > self.config.imu_stale_after_s:
            self._last_prediction_stamp_s = stamp_s
            return 0.0
        previous_stamp_s = self._last_prediction_stamp_s
        self._last_prediction_stamp_s = stamp_s
        if previous_stamp_s is None:
            return 0.0
        dt = max(0.0, min(stamp_s - previous_stamp_s, 0.5))
        if dt <= 1e-6:
            return 0.0
        _, _, yaw_rate_rad_s = imu_to_base_components(
            frame_convention=self.config.imu_frame_convention,
            x=float(self.latest_imu.angular_velocity.x) - self._imu_bias_x_rad_s,
            y=float(self.latest_imu.angular_velocity.y) - self._imu_bias_y_rad_s,
            z=float(self.latest_imu.angular_velocity.z) - self._imu_bias_z_rad_s,
        )
        return float(getattr(self.config, "odom_yaw_sign", 1.0)) * yaw_rate_rad_s * dt

    def _orientation_freeze_active(self) -> bool:
        return bool(self.config.freeze_orientation_during_scan and self._scan_orientation_frozen)

    def _log_scan_orientation_freeze(self, *, now_s: float) -> None:
        if self._last_scan_freeze_log_s is not None and now_s - self._last_scan_freeze_log_s < 1.0:
            return
        self._last_scan_freeze_log_s = now_s
        self.get_logger().info(
            "RGB-D VO orientation freeze active: "
            f"scan_active={self._scan_orientation_frozen} "
            f"pan_deg={math.degrees(self.camera_pan_rad):.3f} "
            f"pose_x={self.pose.x:.3f} pose_y={self.pose.y:.3f} yaw_deg={math.degrees(self.pose.yaw):.1f}"
        )

    def _reset_imu_origin_to_current_pose(self) -> None:
        if self._latest_imu_orientation_unwrapped_yaw_rad is not None:
            self._imu_orientation_origin_yaw_rad = self._latest_imu_orientation_unwrapped_yaw_rad - self.pose.yaw
        self._last_prediction_stamp_s = None

    def _imu_delta_yaw_rad(self, *, predicted_yaw_rad: float, absolute_imu_yaw_rad: float | None) -> float:
        if absolute_imu_yaw_rad is not None:
            return angle_wrap(absolute_imu_yaw_rad - self.pose.yaw)
        if self.latest_imu is not None:
            return predicted_yaw_rad
        return 0.0

    def step(self) -> None:
        stamp = self.get_clock().now().to_msg()
        stamp_s = stamp_to_seconds(stamp)
        orientation_frozen = self._orientation_freeze_active()
        if orientation_frozen:
            self._last_prediction_stamp_s = stamp_s
            self._log_scan_orientation_freeze(now_s=self.get_clock().now().nanoseconds / 1_000_000_000.0)
        predicted_yaw_rad = self._predict_yaw_from_imu(stamp_s=stamp_s)
        absolute_imu_yaw_rad = self._relative_imu_yaw_rad()
        imu_yaw_rad = self._imu_delta_yaw_rad(
            predicted_yaw_rad=predicted_yaw_rad,
            absolute_imu_yaw_rad=absolute_imu_yaw_rad,
        )
        if orientation_frozen:
            imu_yaw_rad = 0.0
        imu_yaw_applied = False
        if self.latest_rgb is not None and self.latest_depth is not None and self.intrinsics is not None:
            try:
                frame = VisualOdomFrame(
                    gray=image_to_array(self.latest_rgb),
                    depth_m=depth_to_meters(self.latest_depth),
                    stamp=self.latest_rgb.header.stamp,
                    intrinsics=self.intrinsics,
                )
                stamp_s = stamp_to_seconds(frame.stamp)
                predicted_yaw_rad = self._predict_yaw_from_imu(stamp_s=stamp_s)
                absolute_imu_yaw_rad = self._relative_imu_yaw_rad()
                imu_yaw_rad = self._imu_delta_yaw_rad(
                    predicted_yaw_rad=predicted_yaw_rad,
                    absolute_imu_yaw_rad=absolute_imu_yaw_rad,
                )
                if orientation_frozen:
                    imu_yaw_rad = 0.0
                if self.previous_frame is not None:
                    estimate = self.estimator.estimate(self.previous_frame, frame)
                    if isinstance(estimate, VisualOdomEstimate):
                        base_forward_m, base_left_m = camera_optical_translation_to_base_planar(
                            camera_x_m=estimate.camera_translation_x_m,
                            camera_y_m=estimate.camera_translation_y_m,
                            camera_z_m=estimate.camera_translation_z_m,
                            pitch_rad=self.camera_pitch_rad,
                        )
                        self.pose = compose_planar_local(
                            self.pose,
                            delta_x_m=base_forward_m,
                            delta_y_m=base_left_m,
                            delta_yaw_rad=imu_yaw_rad,
                        )
                        dt = max(stamp_s - stamp_to_seconds(self.previous_frame.stamp), 1e-6)
                        self.planar_velocity_x_m_s = base_forward_m / dt
                        self.planar_velocity_y_m_s = base_left_m / dt
                        self.accepted_updates += 1
                        self._record_estimate(estimate)
                        self.previous_frame = frame
                        imu_yaw_applied = True
                    else:
                        self.rejected_updates += 1
                        self._record_rejection(estimate)
                else:
                    self.previous_frame = frame
                stamp = frame.stamp
            except Exception as exc:
                self.rejected_updates += 1
                self._record_exception()
                self.get_logger().warning(f"RGB-D VO update rejected: {exc}")
        if not imu_yaw_applied and abs(imu_yaw_rad) > 1e-6:
            self.pose = compose_planar_local(
                self.pose,
                delta_x_m=0.0,
                delta_y_m=0.0,
                delta_yaw_rad=imu_yaw_rad,
            )
            self.planar_velocity_x_m_s = 0.0
            self.planar_velocity_y_m_s = 0.0
        self._log_diagnostics(now_s=self.get_clock().now().nanoseconds / 1_000_000_000.0)
        self._publish_odom(stamp)

    def _publish_odom(self, stamp: Any) -> None:
        qx, qy, qz, qw = yaw_to_quaternion_xyzw(self.pose.yaw)
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.config.odom_frame
        odom.child_frame_id = self.config.base_frame
        odom.pose.pose.position.x = self.pose.x
        odom.pose.pose.position.y = self.pose.y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation = _quaternion_msg(qx, qy, qz, qw)
        odom.pose.covariance[0] = 0.05
        odom.pose.covariance[7] = 0.05
        odom.pose.covariance[35] = 0.10
        odom.twist.twist.linear.x = float(self.planar_velocity_x_m_s)
        odom.twist.twist.linear.y = float(self.planar_velocity_y_m_s)
        if self._orientation_freeze_active():
            odom.twist.twist.angular.z = 0.0
        elif self.latest_imu is not None and self._imu_bias_ready:
            _, _, yaw_rate_rad_s = imu_to_base_components(
                frame_convention=self.config.imu_frame_convention,
                x=float(self.latest_imu.angular_velocity.x) - self._imu_bias_x_rad_s,
                y=float(self.latest_imu.angular_velocity.y) - self._imu_bias_y_rad_s,
                z=float(self.latest_imu.angular_velocity.z) - self._imu_bias_z_rad_s,
            )
            odom.twist.twist.angular.z = float(getattr(self.config, "odom_yaw_sign", 1.0)) * yaw_rate_rad_s
        self.odom_publisher.publish(odom)

        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = self.config.odom_frame
        transform.child_frame_id = self.config.base_frame
        transform.transform.translation.x = self.pose.x
        transform.transform.translation.y = self.pose.y
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
        description="Experimental feature-based RGB-D visual odometry for the real XLeRobot."
    )
    parser.add_argument("--rgb-topic", default="/camera/head/image_raw")
    parser.add_argument("--depth-topic", default="/camera/head/depth/image_raw")
    parser.add_argument("--camera-info-topic", default="/camera/head/camera_info")
    parser.add_argument("--imu-topic", default="/imu")
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--odom-frame", default="odom")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--camera-frame", default="head_camera_link")
    parser.add_argument("--camera-pitch-topic", default="/camera/head/pitch_rad")
    parser.add_argument("--camera-pan-topic", default="/camera/head/pan_rad")
    parser.add_argument(
        "--scan-active-topic",
        default="/xlerobot/scan_active",
        help="Bool topic that is true only while an intentional scan is active.",
    )
    parser.add_argument(
        "--freeze-during-head-motion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Compatibility alias for --freeze-orientation-during-scan. This no longer freezes "
            "position from camera pan alone."
        ),
    )
    parser.add_argument(
        "--freeze-orientation-during-scan",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Freeze odom yaw only while --scan-active-topic publishes true; RGB-D translation remains enabled.",
    )
    parser.add_argument(
        "--head-motion-freeze-settle-s",
        type=float,
        default=0.75,
        help="Deprecated compatibility option; scan-active events now control orientation freeze.",
    )
    parser.add_argument("--head-pan-freeze-threshold-deg", type=float, default=1.0)
    parser.add_argument("--head-pitch-freeze-threshold-deg", type=float, default=1.0)
    parser.add_argument("--publish-rate-hz", type=float, default=30.0)
    parser.add_argument("--min-depth-m", type=float, default=0.15)
    parser.add_argument("--max-depth-m", type=float, default=4.0)
    parser.add_argument("--min-matches", type=int, default=20)
    parser.add_argument("--min-inliers", type=int, default=12)
    parser.add_argument("--max-translation-step-m", type=float, default=0.25)
    parser.add_argument(
        "--min-translation-update-m",
        type=float,
        default=0.005,
        help="Keep the current RGB-D keyframe until estimated forward motion reaches this threshold.",
    )
    parser.add_argument(
        "--jitter-threshold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reject visual odometry updates whose estimated forward motion is below "
            "`--min-translation-update-m`. Use `--no-jitter-threshold` to accept tiny "
            "PnP translations while debugging slow real Nav2 motion."
        ),
    )
    parser.add_argument("--max-yaw-step-deg", type=float, default=30.0)
    parser.add_argument("--imu-stale-after-s", type=float, default=0.5)
    parser.add_argument("--imu-bias-calibration-s", type=float, default=2.0)
    parser.add_argument("--imu-bias-min-samples", type=int, default=20)
    parser.add_argument(
        "--imu-frame-convention",
        choices=("camera_optical", "base_link"),
        default="camera_optical",
        help="Interpret raw IMU vectors in camera optical coordinates or already-rotated base_link coordinates.",
    )
    parser.add_argument(
        "--odom-yaw-sign",
        type=float,
        choices=(-1.0, 1.0),
        default=1.0,
        help=(
            "Sign applied to IMU yaw before publishing odom->base_link. Use -1 if physical right turns "
            "make odometry yaw increase instead of decrease."
        ),
    )
    parser.add_argument("--camera-x-m", type=float, default=0.0)
    parser.add_argument("--camera-y-m", type=float, default=0.0)
    parser.add_argument("--camera-yaw-rad", type=float, default=0.0)
    parser.add_argument("--camera-pitch-rad", type=float, default=0.0)
    parser.add_argument("--camera-pitch-deg", type=float, default=None)
    return parser


def config_from_args(args: argparse.Namespace) -> RgbdVoConfig:
    return RgbdVoConfig(
        rgb_topic=args.rgb_topic,
        depth_topic=args.depth_topic,
        camera_info_topic=args.camera_info_topic,
        imu_topic=args.imu_topic,
        odom_topic=args.odom_topic,
        odom_frame=args.odom_frame,
        base_frame=args.base_frame,
        camera_frame=args.camera_frame,
        camera_pitch_topic=args.camera_pitch_topic,
        camera_pan_topic=args.camera_pan_topic,
        scan_active_topic=args.scan_active_topic,
        freeze_orientation_during_scan=(
            bool(args.freeze_during_head_motion)
            if args.freeze_orientation_during_scan is None
            else bool(args.freeze_orientation_during_scan)
        ),
        head_pan_freeze_threshold_rad=math.radians(args.head_pan_freeze_threshold_deg),
        head_pitch_freeze_threshold_rad=math.radians(args.head_pitch_freeze_threshold_deg),
        publish_rate_hz=args.publish_rate_hz,
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
        min_matches=args.min_matches,
        min_inliers=args.min_inliers,
        max_translation_step_m=args.max_translation_step_m,
        min_translation_update_m=args.min_translation_update_m,
        jitter_threshold=args.jitter_threshold,
        max_yaw_step_rad=math.radians(args.max_yaw_step_deg),
        imu_stale_after_s=args.imu_stale_after_s,
        imu_bias_calibration_s=args.imu_bias_calibration_s,
        imu_bias_min_samples=args.imu_bias_min_samples,
        imu_frame_convention=args.imu_frame_convention,
        odom_yaw_sign=args.odom_yaw_sign,
        camera_x_m=args.camera_x_m,
        camera_y_m=args.camera_y_m,
        camera_yaw_rad=args.camera_yaw_rad,
        camera_pitch_rad=(
            args.camera_pitch_rad if args.camera_pitch_deg is None else math.radians(args.camera_pitch_deg)
        ),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require_runtime_dependencies()
    rclpy.init()
    node = RgbdVisualOdometryNode(config_from_args(args))
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
