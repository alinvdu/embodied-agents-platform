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


if __name__ == "__main__":
    unittest.main()
