import struct
from types import SimpleNamespace
import unittest

from xlerobot_playground import ros_nav2_runtime


class RosNav2RuntimeDepthTest(unittest.TestCase):
    def setUp(self) -> None:
        if ros_nav2_runtime.np is None:
            self.skipTest("numpy is unavailable")

    def test_depth_image_to_meters_array_reads_big_endian_mono16(self) -> None:
        data = b"".join(int(value).to_bytes(2, "big") for value in (1000, 1500, 2000, 0))
        message = SimpleNamespace(
            encoding="mono16",
            height=2,
            width=2,
            step=4,
            is_bigendian=True,
            data=data,
        )

        depth = ros_nav2_runtime._depth_image_to_meters_array(message)

        self.assertEqual(depth.shape, (2, 2))
        self.assertAlmostEqual(float(depth[0, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(depth[0, 1]), 1.5, places=5)
        self.assertAlmostEqual(float(depth[1, 0]), 2.0, places=5)
        self.assertAlmostEqual(float(depth[1, 1]), 0.0, places=5)

    def test_depth_image_to_meters_array_reads_float32(self) -> None:
        data = b"".join(struct.pack("<f", value) for value in (0.4, 0.5, 0.6, 0.7))
        message = SimpleNamespace(
            encoding="32FC1",
            height=2,
            width=2,
            step=8,
            is_bigendian=False,
            data=data,
        )

        depth = ros_nav2_runtime._depth_image_to_meters_array(message)

        self.assertEqual(depth.shape, (2, 2))
        self.assertAlmostEqual(float(depth[0, 0]), 0.4, places=5)
        self.assertAlmostEqual(float(depth[1, 1]), 0.7, places=5)

    def test_project_depth_pixels_converts_optical_to_ros_camera_link(self) -> None:
        np = ros_nav2_runtime.np
        points = ros_nav2_runtime._project_depth_pixels_to_camera_link(
            u=np.asarray([320.0, 420.0], dtype=np.float32),
            v=np.asarray([240.0, 140.0], dtype=np.float32),
            depth_m=np.asarray([1.0, 2.0], dtype=np.float32),
            fx=500.0,
            fy=500.0,
            cx=320.0,
            cy=240.0,
        )

        self.assertEqual(points.shape, (2, 3))
        self.assertAlmostEqual(float(points[0, 0]), 1.0)
        self.assertAlmostEqual(float(points[0, 1]), -0.0)
        self.assertAlmostEqual(float(points[0, 2]), -0.0)
        self.assertAlmostEqual(float(points[1, 0]), 2.0)
        self.assertAlmostEqual(float(points[1, 1]), -0.4)
        self.assertAlmostEqual(float(points[1, 2]), 0.4)

    def test_snapshot_stamp_s_handles_missing_or_invalid_values(self) -> None:
        self.assertEqual(ros_nav2_runtime._snapshot_stamp_s(None), 0.0)
        self.assertEqual(ros_nav2_runtime._snapshot_stamp_s({}), 0.0)
        self.assertEqual(ros_nav2_runtime._snapshot_stamp_s({"stamp_s": "bad"}), 0.0)
        self.assertEqual(ros_nav2_runtime._snapshot_stamp_s({"stamp_s": 123.5}), 123.5)

    def test_fallback_camera_intrinsics_from_horizontal_fov(self) -> None:
        intrinsics = ros_nav2_runtime._fallback_camera_intrinsics(
            width=640,
            height=480,
            frame_id="head_camera_link",
            horizontal_fov_deg=64.0,
        )

        self.assertIsNotNone(intrinsics)
        assert intrinsics is not None
        self.assertEqual(intrinsics["source"], "fallback_horizontal_fov")
        self.assertEqual(intrinsics["frame_id"], "head_camera_link")
        self.assertEqual(intrinsics["width"], 640)
        self.assertAlmostEqual(float(intrinsics["fx"]), 512.1, delta=5.0)
        self.assertAlmostEqual(float(intrinsics["fy"]), float(intrinsics["fx"]), places=5)
        self.assertEqual(intrinsics["cx"], 320.0)
        self.assertEqual(intrinsics["cy"], 240.0)

    def test_detection_geometry_uses_depth_image_with_fallback_intrinsics(self) -> None:
        np = ros_nav2_runtime.np

        class Pose:
            def to_dict(self):
                return {"x": 0.0, "y": 0.0, "yaw": 0.0}

        class FakeRuntime:
            def __init__(self) -> None:
                self.latest_depth_snapshot = {
                    "frame_id": "head_camera_link",
                    "width": 640,
                    "height": 480,
                    "encoding": "mono16",
                    "stamp_s": ros_nav2_runtime.time.time(),
                    "depth_m": np.full((480, 640), 0.8, dtype=np.float32),
                }
                self.latest_camera_info_snapshot = None
                self.config = SimpleNamespace(
                    rgbd_fallback_horizontal_fov_deg=64.0,
                    base_frame="base_link",
                    map_frame="map",
                )

            def _lookup_transform_xyz_quat(self, target_frame, source_frame):
                return np.asarray([0.0, 0.0, 0.0], dtype=np.float32), (0.0, 0.0, 0.0, 1.0)

            def current_pose(self):
                return Pose()

        result = ros_nav2_runtime.RosExplorationRuntime._estimate_detection_geometry_from_depth(
            FakeRuntime(),
            {
                "image_width": 640,
                "image_height": 480,
                "target_max_m": 0.45,
                "max_step_m": 0.08,
            },
            (300.0, 200.0, 340.0, 320.0),
        )

        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["geometry_source"], "depth_image")
        self.assertEqual(result["camera_info"]["source"], "fallback_horizontal_fov")
        self.assertAlmostEqual(result["forward_m"], 0.8, delta=0.02)
        self.assertGreater(result["valid_sample_count"], 12)

    def test_detection_geometry_can_disable_point_cloud_fallback(self) -> None:
        class FakeRuntime:
            def __init__(self) -> None:
                self.latest_depth_snapshot = None
                self.latest_camera_info_snapshot = None
                self.latest_point_cloud_snapshot = {
                    "width": 152708,
                    "height": 1,
                    "frame_id": "head_camera_link",
                }
                self.config = SimpleNamespace(
                    rgbd_update_timeout_s=0.0,
                    rgbd_fallback_horizontal_fov_deg=64.0,
                )

            def spin_for(self, duration_s: float) -> None:
                return None

            def wait_for_rgbd_update(self, **kwargs) -> bool:
                return False

        result = ros_nav2_runtime.RosExplorationRuntime.estimate_detection_geometry(
            FakeRuntime(),
            {
                "bbox_xyxy": [300, 200, 340, 320],
                "rgbd_update_timeout_s": 0.0,
                "require_depth_image": True,
                "disable_point_cloud_fallback": True,
            },
        )

        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["point_cloud_fallback"], "disabled")
        self.assertNotIn("point_cloud", result)
        self.assertNotIn("not organized", result["reason"])


if __name__ == "__main__":
    unittest.main()
