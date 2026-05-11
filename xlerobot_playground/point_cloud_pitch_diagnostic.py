from __future__ import annotations

import argparse
import math
import time
from typing import Any

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import PointCloud2
    from tf2_ros import Buffer, ConnectivityException, ExtrapolationException, LookupException, TransformListener
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only outside ROS envs
    rclpy = None
    Node = object
    qos_profile_sensor_data = None
    PointCloud2 = object
    Buffer = TransformListener = None
    ConnectivityException = ExtrapolationException = LookupException = Exception
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from xlerobot_playground.ros_nav2_runtime import _point_cloud2_xyz_array, _quaternion_rotation_matrix


class PointCloudPitchDiagnostic(Node):
    def __init__(
        self,
        *,
        topic: str,
        reference_frame: str,
        min_range_m: float,
        max_range_m: float,
        low_percentile: float,
        candidate_margin_m: float,
        min_points: int,
        max_samples: int,
        duration_s: float,
    ) -> None:
        super().__init__("xlerobot_point_cloud_pitch_diagnostic")
        self.reference_frame = reference_frame
        self.min_range_m = float(min_range_m)
        self.max_range_m = float(max_range_m)
        self.low_percentile = float(low_percentile)
        self.candidate_margin_m = float(candidate_margin_m)
        self.min_points = int(min_points)
        self.max_samples = int(max_samples)
        self.deadline_s = time.monotonic() + max(float(duration_s), 0.1)
        self.samples: list[dict[str, float]] = []
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(PointCloud2, topic, self._on_point_cloud, qos_profile_sensor_data)
        self.get_logger().info(
            f"Listening for {topic}; fitting floor plane in {reference_frame}. "
            "Keep robot still and point the camera at visible floor."
        )

    @property
    def done(self) -> bool:
        return len(self.samples) >= self.max_samples or time.monotonic() >= self.deadline_s

    def _on_point_cloud(self, message: Any) -> None:
        points = _point_cloud2_xyz_array(message)
        if points.size == 0:
            return
        transformed = self._transform_points(points, str(message.header.frame_id))
        if transformed is None:
            return
        fit = fit_floor_pitch_error(
            transformed,
            min_range_m=self.min_range_m,
            max_range_m=self.max_range_m,
            low_percentile=self.low_percentile,
            candidate_margin_m=self.candidate_margin_m,
            min_points=self.min_points,
        )
        if fit is None:
            return
        self.samples.append(fit)
        self.get_logger().info(
            "sample "
            f"{len(self.samples)}/{self.max_samples}: "
            f"pitch_error_deg={fit['pitch_error_deg']:.3f} "
            f"roll_error_deg={fit['roll_error_deg']:.3f} "
            f"floor_points={fit['floor_points']:.0f}"
        )

    def _transform_points(self, points: np.ndarray, source_frame: str) -> np.ndarray | None:
        try:
            transform = self.tf_buffer.lookup_transform(self.reference_frame, source_frame, rclpy.time.Time())
        except (ConnectivityException, ExtrapolationException, LookupException) as exc:
            self.get_logger().warning(f"TF not ready {self.reference_frame} <- {source_frame}: {exc}")
            return None
        translation = np.array(
            [
                float(transform.transform.translation.x),
                float(transform.transform.translation.y),
                float(transform.transform.translation.z),
            ],
            dtype=np.float32,
        )
        rotation = _quaternion_rotation_matrix(
            float(transform.transform.rotation.x),
            float(transform.transform.rotation.y),
            float(transform.transform.rotation.z),
            float(transform.transform.rotation.w),
        )
        return (points @ rotation.T + translation.reshape(1, 3)).astype(np.float32, copy=False)


def fit_floor_pitch_error(
    points: np.ndarray,
    *,
    min_range_m: float,
    max_range_m: float,
    low_percentile: float,
    candidate_margin_m: float,
    min_points: int,
) -> dict[str, float] | None:
    finite = np.isfinite(points).all(axis=1)
    if not np.any(finite):
        return None
    finite_points = points[finite]
    ranges = np.linalg.norm(finite_points[:, :2], axis=1)
    range_mask = (ranges >= min_range_m) & (ranges <= max_range_m)
    ranged = finite_points[range_mask]
    if ranged.shape[0] < max(min_points, 3):
        return None
    low_z = float(np.percentile(ranged[:, 2], min(max(low_percentile, 5.0), 80.0)))
    floor_points = ranged[ranged[:, 2] <= low_z + max(candidate_margin_m, 0.02)]
    if floor_points.shape[0] < max(min_points, 3):
        return None
    design = np.column_stack(
        [
            floor_points[:, 0].astype(np.float64),
            floor_points[:, 1].astype(np.float64),
            np.ones((floor_points.shape[0],), dtype=np.float64),
        ]
    )
    target = floor_points[:, 2].astype(np.float64)
    try:
        a, b, c = np.linalg.lstsq(design, target, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    return {
        "pitch_error_deg": math.degrees(math.atan(float(a))),
        "roll_error_deg": math.degrees(math.atan(float(b))),
        "plane_a": float(a),
        "plane_b": float(b),
        "plane_c": float(c),
        "floor_points": float(floor_points.shape[0]),
        "total_points": float(points.shape[0]),
    }


def summarize(samples: list[dict[str, float]]) -> dict[str, float]:
    if not samples:
        return {}
    keys = ["pitch_error_deg", "roll_error_deg", "plane_a", "plane_b", "plane_c", "floor_points"]
    return {key: float(np.median([sample[key] for sample in samples])) for key in keys}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate camera pitch offset from the live floor point cloud.")
    parser.add_argument("--topic", default="/camera/head/points")
    parser.add_argument("--reference-frame", default="base_link")
    parser.add_argument("--old-offset-deg", type=float, default=None)
    parser.add_argument("--min-range-m", type=float, default=0.4)
    parser.add_argument("--max-range-m", type=float, default=3.0)
    parser.add_argument("--low-percentile", type=float, default=35.0)
    parser.add_argument("--candidate-margin-m", type=float, default=0.12)
    parser.add_argument("--min-points", type=int, default=200)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--duration-s", type=float, default=8.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "point_cloud_pitch_diagnostic requires ROS 2 Python packages: "
            "rclpy, sensor_msgs, and tf2_ros."
        ) from _IMPORT_ERROR
    args = build_parser().parse_args(argv)
    rclpy.init()
    node = PointCloudPitchDiagnostic(
        topic=args.topic,
        reference_frame=args.reference_frame,
        min_range_m=args.min_range_m,
        max_range_m=args.max_range_m,
        low_percentile=args.low_percentile,
        candidate_margin_m=args.candidate_margin_m,
        min_points=args.min_points,
        max_samples=args.samples,
        duration_s=args.duration_s,
    )
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.2)
        summary = summarize(node.samples)
        if not summary:
            print("No usable floor plane samples collected.")
            return 2
        pitch_error = summary["pitch_error_deg"]
        print(f"median_pitch_error_deg={pitch_error:.3f}")
        print(f"median_roll_error_deg={summary['roll_error_deg']:.3f}")
        print(f"median_floor_points={summary['floor_points']:.0f}")
        if args.old_offset_deg is not None:
            print(f"try_offset_plus_error_deg={args.old_offset_deg + pitch_error:.3f}")
            print(f"try_offset_minus_error_deg={args.old_offset_deg - pitch_error:.3f}")
            print("Use the candidate that makes the next run's median_pitch_error_deg closest to 0.")
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
