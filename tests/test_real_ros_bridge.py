from __future__ import annotations

from io import BytesIO
import math
from pathlib import Path
import struct
import tempfile
import threading
import time
import unittest
from urllib.error import HTTPError

import xlerobot_playground.real_ros_bridge as real_ros_bridge
from xlerobot_playground.real_ros_bridge import (
    OrbbecFilesystemConfig,
    OrbbecFilesystemRgbdSource,
    RealRosBridgeConfig,
    RobotBrainRgbdSource,
    _build_camera_info_from_metadata,
    _motion_result_error,
    _orbbec_optical_xyz_to_ros_camera_link,
    _point_field,
    build_parser,
    config_from_args,
    _format_runtime_error,
    imu_ros_timestamp_s,
    parse_imu_json,
    parse_depth_pgm_mm,
    parse_rgb_ppm,
    read_depth_pgm_mm,
    synthesize_scan_from_depth_be,
    synthesize_scan_from_depth_rows,
    twist_to_base_velocity,
    yaw_to_quaternion_xyzw,
)
from xlerobot_playground.rgbd_transport import pack_rgbd_frame
from xlerobot_playground.rgbd_transport import POINT_CLOUD_FORMAT_XYZ_FLOAT32


class _Vector:
    def __init__(self, *, x: float = 0.0, z: float = 0.0) -> None:
        self.x = x
        self.z = z


class _Twist:
    def __init__(self, *, linear: float = 0.04, angular: float = 0.12) -> None:
        self.linear = _Vector(x=linear)
        self.angular = _Vector(z=angular)


class _MotionRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[str, float, float]] = []
        self.stopped = threading.Event()

    def drive_velocity(self, *, linear_m_s: float, angular_rad_s: float) -> dict[str, object]:
        self.calls.append(("drive", linear_m_s, angular_rad_s))
        return {"succeeded": True}

    def stop(self) -> dict[str, object]:
        self.calls.append(("stop", 0.0, 0.0))
        self.stopped.set()
        return {"succeeded": True}


class _BlockingRgbdSource:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def capture(self) -> _Frame:
        self.entered.set()
        self.release.wait(timeout=2.0)
        return _Frame(time.time())


class _FakeBrainClient:
    def __init__(self, *, paired: bool = False) -> None:
        self.requested_paths: list[str] = []
        self.payloads = {
            "/rgb": b"P6\n1 1\n255\nabc",
            "/depth": b"P5\n1 1\n65535\n" + (1234).to_bytes(2, "big"),
        }
        if paired:
            self.payloads["/rgbd"] = pack_rgbd_frame(
                frame_index=7,
                timestamp_us=1_250_000,
                rgb=b"abc",
                rgb_width=1,
                rgb_height=1,
                depth_be=(1234).to_bytes(2, "big"),
                depth_width=1,
                depth_height=1,
                metadata={
                    "camera_intrinsics": {
                        "fx": 500.0,
                        "fy": 510.0,
                        "cx": 300.0,
                        "cy": 200.0,
                        "width": 640,
                        "height": 480,
                    }
                },
                point_cloud_format=POINT_CLOUD_FORMAT_XYZ_FLOAT32,
                point_cloud_points=struct.pack("<fff", 0.1, 0.2, 0.3),
                point_cloud_count=1,
                point_cloud_stride=12,
            )

    def get_bytes(self, path: str) -> bytes:
        self.requested_paths.append(path)
        return self.payloads[path]


class _Frame:
    def __init__(self, timestamp_s: float) -> None:
        self.timestamp_s = timestamp_s


class _ClockInstant:
    def to_msg(self) -> object:
        return object()


class _Clock:
    def now(self) -> _ClockInstant:
        return _ClockInstant()


class _Logger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(message)


class RealRosBridgeTests(unittest.TestCase):
    def test_parser_defaults_match_two_wheel_robot_ports(self) -> None:
        args = build_parser().parse_args([])
        config = config_from_args(args)

        self.assertEqual(config.robot_kind, "xlerobot_2wheels")
        self.assertEqual(config.port1, "/dev/tty.usbmodem5B140330101")
        self.assertEqual(config.port2, "/dev/tty.usbmodem5B140332271")
        self.assertFalse(config.allow_motion_commands)
        self.assertEqual(config.odom_source, "none")
        self.assertEqual(config.max_linear_m_s, 0.05)
        self.assertEqual(config.max_angular_rad_s, 0.20)
        self.assertEqual(config.motion_command_rate_hz, 50.0)
        self.assertEqual(config.camera_z_m, 0.35)
        self.assertEqual(config.camera_pan_topic, "/camera/head/pan_rad")
        self.assertEqual(config.scan_active_topic, "/xlerobot/scan_active")
        self.assertTrue(config.publish_imu)

    def test_camera_mount_arguments_are_configurable(self) -> None:
        args = build_parser().parse_args(
            ["--camera-x-m", "0.04", "--camera-y-m", "-0.01", "--camera-z-m", "0.32", "--camera-yaw-rad", "0.1"]
        )
        config = config_from_args(args)

        self.assertEqual(config.camera_x_m, 0.04)
        self.assertEqual(config.camera_y_m, -0.01)
        self.assertEqual(config.camera_z_m, 0.32)
        self.assertEqual(config.camera_yaw_rad, 0.1)

    def test_commanded_odom_is_explicit_smoke_test_mode(self) -> None:
        args = build_parser().parse_args(["--odom-source", "commanded"])
        config = config_from_args(args)

        self.assertEqual(config.odom_source, "commanded")

    def test_settled_head_points_do_not_wait_for_pose_outside_scan(self) -> None:
        bridge = real_ros_bridge.RealXLeRobotRosBridge.__new__(real_ros_bridge.RealXLeRobotRosBridge)
        logger = _Logger()
        bridge.get_logger = lambda: logger
        bridge.config = RealRosBridgeConfig(head_points_mode="settled", head_points_settled_delay_s=10.0)
        bridge._head_points_update_map_enabled = True
        bridge._base_motion_active = False
        bridge._scan_active = False
        bridge._camera_pose_moving = True
        bridge._camera_pose_updated_s = None
        bridge._camera_pose_received_s = None
        bridge._last_head_points_skip_reason = "waiting for settled pose/frame"

        self.assertTrue(bridge._head_points_publish_allowed(_Frame(time.monotonic())))
        self.assertEqual(bridge._last_head_points_skip_reason, "")
        self.assertEqual(logger.messages, [])

    def test_head_points_can_be_limited_to_scan_windows(self) -> None:
        bridge = real_ros_bridge.RealXLeRobotRosBridge.__new__(real_ros_bridge.RealXLeRobotRosBridge)
        logger = _Logger()
        bridge.get_logger = lambda: logger
        bridge.config = RealRosBridgeConfig(head_points_only_during_scan=True)
        bridge._head_points_update_map_enabled = True
        bridge._base_motion_active = False
        bridge._scan_active = False
        bridge._last_head_points_skip_reason = ""

        self.assertFalse(bridge._head_points_publish_allowed(_Frame(time.monotonic())))
        self.assertEqual(bridge._last_head_points_skip_reason, "scan inactive")
        self.assertEqual(logger.messages, ["Suppressing /camera/head/points in settled mode: scan inactive."])

    def test_settled_head_points_wait_for_pose_during_scan(self) -> None:
        now_s = time.monotonic()
        bridge = real_ros_bridge.RealXLeRobotRosBridge.__new__(real_ros_bridge.RealXLeRobotRosBridge)
        logger = _Logger()
        bridge.get_logger = lambda: logger
        bridge.config = RealRosBridgeConfig(head_points_mode="settled", head_points_settled_delay_s=10.0)
        bridge._head_points_update_map_enabled = True
        bridge._base_motion_active = False
        bridge._scan_active = True
        bridge._camera_pose_moving = False
        bridge._camera_pose_updated_s = now_s
        bridge._camera_pose_received_s = now_s
        bridge._last_head_points_skip_reason = ""

        self.assertFalse(bridge._head_points_publish_allowed(_Frame(now_s)))
        self.assertEqual(bridge._last_head_points_skip_reason, "waiting for settled pose/frame")
        self.assertEqual(
            logger.messages,
            ["Suppressing /camera/head/points in settled mode: waiting for settled pose/frame."],
        )

    def test_robot_brain_url_selects_remote_hardware_endpoint(self) -> None:
        args = build_parser().parse_args(
            [
                "--robot-brain-url",
                "http://robot-brain.local:8765",
                "--imu-ws-path",
                "/ws/imu",
                "--imu-ws-reconnect-delay-s",
                "0.5",
            ]
        )
        config = config_from_args(args)

        self.assertEqual(config.robot_brain_url, "http://robot-brain.local:8765")
        self.assertEqual(config.imu_topic, "/imu")
        self.assertEqual(config.imu_ws_path, "/ws/imu")
        self.assertEqual(config.imu_ws_reconnect_delay_s, 0.5)

    def test_parser_can_disable_raw_imu_for_wheel_odometry_mode(self) -> None:
        args = build_parser().parse_args(["--no-publish-imu"])
        config = config_from_args(args)

        self.assertFalse(config.publish_imu)

    def test_runtime_error_formatter_includes_robot_brain_http_body(self) -> None:
        exc = HTTPError(
            url="http://robot-brain.local:8765/cmd_vel",
            code=500,
            msg="Internal Server Error",
            hdrs={},
            fp=BytesIO(b"(6, 'Device not configured')"),
        )

        self.assertEqual(_format_runtime_error(exc), "HTTP 500: (6, 'Device not configured')")

    def test_motion_result_error_reports_rejected_remote_command(self) -> None:
        self.assertEqual(
            _motion_result_error(
                {
                    "succeeded": False,
                    "message": "Real motion commands are disabled.",
                    "metadata": {"requested_angular_rad_s": 0.3},
                }
            ),
            "Real motion commands are disabled. metadata={'requested_angular_rad_s': 0.3}",
        )
        self.assertIsNone(_motion_result_error({"succeeded": True, "message": "ok"}))

    def test_twist_to_base_velocity_uses_forward_and_yaw_only(self) -> None:
        self.assertEqual(twist_to_base_velocity(_Twist()), (0.04, 0.12))

    def test_explicit_stop_reaches_hardware_while_rgbd_capture_is_blocked(self) -> None:
        bridge = real_ros_bridge.RealXLeRobotRosBridge.__new__(real_ros_bridge.RealXLeRobotRosBridge)
        bridge.config = RealRosBridgeConfig(cmd_vel_timeout_s=0.5)
        bridge.runtime = _MotionRuntime()
        bridge.rgbd_source = _BlockingRgbdSource()
        bridge._latest_twist = _Twist(angular=0.2)
        bridge._latest_cmd_stamp = time.monotonic()
        bridge._velocity_lock = threading.Lock()
        bridge._motion_lock = threading.Lock()
        bridge._last_motion_step_stamp = time.monotonic()
        bridge._last_linear = 0.0
        bridge._last_angular = 0.2
        bridge._last_motion_error = ""
        bridge._base_motion_active = True
        bridge._x = 0.0
        bridge._y = 0.0
        bridge._yaw = 0.0
        bridge._poll_camera_pose = lambda *, now_s: None
        bridge.get_clock = lambda: _Clock()
        bridge._publish_transforms = lambda **_kwargs: None
        bridge._publish_scan = lambda **_kwargs: None
        bridge._publish_head_images = lambda **_kwargs: None
        bridge._publish_head_points = lambda **_kwargs: None

        camera_thread = threading.Thread(target=bridge.step)
        camera_thread.start()
        self.assertTrue(bridge.rgbd_source.entered.wait(timeout=1.0))

        bridge._on_cmd_vel(_Twist(linear=0.0, angular=0.0))

        self.assertTrue(bridge.runtime.stopped.wait(timeout=0.2))
        self.assertTrue(camera_thread.is_alive())
        self.assertEqual(bridge.runtime.calls[-1][0], "stop")
        bridge.rgbd_source.release.set()
        camera_thread.join(timeout=1.0)
        self.assertFalse(camera_thread.is_alive())

    def test_motion_watchdog_stops_stale_command_without_sensor_step(self) -> None:
        bridge = real_ros_bridge.RealXLeRobotRosBridge.__new__(real_ros_bridge.RealXLeRobotRosBridge)
        bridge.config = RealRosBridgeConfig(cmd_vel_timeout_s=0.5)
        bridge.runtime = _MotionRuntime()
        bridge._latest_twist = _Twist(angular=0.2)
        bridge._latest_cmd_stamp = time.monotonic() - 1.0
        bridge._velocity_lock = threading.Lock()
        bridge._motion_lock = threading.Lock()
        bridge._last_motion_step_stamp = time.monotonic()
        bridge._last_linear = 0.0
        bridge._last_angular = 0.2
        bridge._last_motion_error = ""
        bridge._base_motion_active = True
        bridge._x = 0.0
        bridge._y = 0.0
        bridge._yaw = 0.0

        bridge.motion_step()

        self.assertEqual(bridge.runtime.calls, [("stop", 0.0, 0.0)])
        self.assertFalse(bridge._base_motion_active)

    def test_yaw_to_quaternion(self) -> None:
        _x, _y, z, w = yaw_to_quaternion_xyzw(0.0)

        self.assertEqual(z, 0.0)
        self.assertEqual(w, 1.0)

    def test_depth_rows_convert_to_scan_ranges(self) -> None:
        depth = tuple(tuple(1000 for _ in range(5)) for _ in range(7))

        ranges, angles = synthesize_scan_from_depth_rows(
            depth,
            horizontal_fov_rad=1.0,
            band_height_px=3,
            range_min_m=0.05,
            range_max_m=4.0,
        )

        self.assertEqual(len(ranges), 5)
        self.assertEqual(len(angles), 5)
        self.assertTrue(all(value >= 1.0 for value in ranges[1:4]))
        self.assertLess(angles[0], 0.0)
        self.assertGreater(angles[-1], 0.0)

    def test_depth_rows_fill_no_return_beams(self) -> None:
        depth = tuple(
            tuple(1000 if column == 2 else 0 for column in range(5))
            for _row in range(7)
        )

        ranges, _angles = synthesize_scan_from_depth_rows(
            depth,
            horizontal_fov_rad=1.0,
            band_height_px=3,
            range_min_m=0.05,
            range_max_m=4.0,
        )

        self.assertEqual(ranges[0], 4.0)
        self.assertLess(ranges[2], 1.1)
        self.assertEqual(ranges[-1], 4.0)

    def test_depth_rows_can_leave_no_return_beams_infinite(self) -> None:
        depth = tuple(
            tuple(1000 if column == 2 else 0 for column in range(5))
            for _row in range(7)
        )

        ranges, _angles = synthesize_scan_from_depth_rows(
            depth,
            horizontal_fov_rad=1.0,
            band_height_px=3,
            range_min_m=0.05,
            range_max_m=4.0,
            fill_no_return=False,
        )

        self.assertTrue(math.isinf(ranges[0]))
        self.assertLess(ranges[2], 1.1)
        self.assertTrue(math.isinf(ranges[-1]))

    def test_real_bridge_defaults_do_not_clear_missing_depth(self) -> None:
        config = RealRosBridgeConfig()

        self.assertFalse(config.laser_fill_no_return)

    def test_reads_16_bit_depth_pgm_as_millimetres(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "latest_depth.pgm"
            path.write_bytes(
                b"P5\n2 2\n65535\n"
                + (1000).to_bytes(2, "big")
                + (2000).to_bytes(2, "big")
                + (0).to_bytes(2, "big")
                + (65535).to_bytes(2, "big")
            )

            depth, width, height = read_depth_pgm_mm(path)

        self.assertEqual((width, height), (2, 2))
        self.assertEqual(depth[0], (1000, 2000))
        self.assertEqual(depth[1], (0, 65535))

    def test_parse_pnm_payloads_from_robot_brain(self) -> None:
        rgb, rgb_width, rgb_height = parse_rgb_ppm(b"P6\n1 1\n255\nabc")
        depth, depth_width, depth_height = parse_depth_pgm_mm(
            b"P5\n1 1\n65535\n" + (1234).to_bytes(2, "big")
        )

        self.assertEqual(rgb, b"abc")
        self.assertEqual((rgb_width, rgb_height), (1, 1))
        self.assertEqual(depth, ((1234,),))
        self.assertEqual((depth_width, depth_height), (1, 1))

    def test_parse_imu_json_reads_nested_metadata_contract(self) -> None:
        sample = parse_imu_json(
            b'{"imu":{"angular_velocity_rad_s":{"x":0.1,"y":0.2,"z":0.3},"linear_acceleration_m_s2":{"x":1.0,"y":2.0,"z":3.0},"system_timestamp_us":1234567}}'
        )

        self.assertEqual(sample["angular_velocity_rad_s"]["z"], 0.3)
        self.assertEqual(sample["linear_acceleration_m_s2"]["x"], 1.0)
        self.assertAlmostEqual(sample["timestamp_s"], 1.234567)

    def test_imu_ros_timestamp_prefers_accel_frame_time(self) -> None:
        sample = parse_imu_json(
            b'{"timestamp_s":2.0,"has_accel":true,"has_gyro":true,'
            b'"accel_timestamp_us":1500000,"gyro_timestamp_us":2000000,'
            b'"angular_velocity_rad_s":{"x":0.1,"y":0.2,"z":0.3},'
            b'"linear_acceleration_m_s2":{"x":1.0,"y":2.0,"z":3.0}}'
        )

        self.assertAlmostEqual(imu_ros_timestamp_s(sample), 1.5)

    def test_imu_ros_timestamp_keeps_gyro_time_for_gyro_only_samples(self) -> None:
        sample = parse_imu_json(
            b'{"timestamp_s":2.0,"has_accel":false,"has_gyro":true,'
            b'"gyro_timestamp_us":2000000,'
            b'"angular_velocity_rad_s":{"x":0.1,"y":0.2,"z":0.3}}'
        )

        self.assertAlmostEqual(imu_ros_timestamp_s(sample), 2.0)

    def test_filesystem_source_can_return_rgb_without_depth(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            (output_dir / "latest.ppm").write_bytes(b"P6\n1 1\n255\nabc")
            source = OrbbecFilesystemRgbdSource(OrbbecFilesystemConfig(output_dir=output_dir))

            frame = source.capture()

        self.assertEqual(frame.rgb, b"abc")
        self.assertEqual(frame.rgb_width, 1)
        self.assertEqual(frame.rgb_height, 1)
        self.assertIsNone(frame.depth_mm)

    def test_robot_brain_rgbd_source_reads_remote_pnm_payloads(self) -> None:
        client = _FakeBrainClient()
        source = RobotBrainRgbdSource(client)

        frame = source.capture()

        self.assertEqual(frame.rgb, b"abc")
        self.assertEqual(frame.rgb_width, 1)
        self.assertEqual(frame.depth_mm, ((1234,),))
        self.assertEqual(frame.depth_width, 1)
        self.assertIsNone(frame.imu_sample)
        self.assertEqual(client.requested_paths, ["/rgbd", "/rgb", "/depth"])

    def test_robot_brain_rgbd_source_prefers_paired_payload(self) -> None:
        client = _FakeBrainClient(paired=True)
        source = RobotBrainRgbdSource(client)

        frame = source.capture()

        self.assertEqual(frame.rgb, b"abc")
        self.assertEqual(frame.rgb_width, 1)
        self.assertIsNone(frame.depth_mm)
        self.assertEqual(frame.depth_be, (1234).to_bytes(2, "big"))
        self.assertEqual(frame.depth_width, 1)
        self.assertEqual(frame.metadata["camera_intrinsics"]["fy"], 510.0)
        self.assertEqual(frame.point_cloud_format, POINT_CLOUD_FORMAT_XYZ_FLOAT32)
        self.assertEqual(frame.point_cloud_count, 1)
        self.assertEqual(frame.point_cloud_stride, 12)
        self.assertEqual(frame.point_cloud_points, struct.pack("<fff", 0.1, 0.2, 0.3))
        self.assertEqual(frame.frame_index, 7)
        self.assertAlmostEqual(frame.timestamp_s, 1.25)
        self.assertEqual(client.requested_paths, ["/rgbd"])

    def test_point_field_uses_ros_float32_layout(self) -> None:
        class _PointField:
            FLOAT32 = 7

            def __init__(self) -> None:
                self.name = ""
                self.offset = 0
                self.datatype = 0
                self.count = 0

        original_point_field = real_ros_bridge.PointField
        real_ros_bridge.PointField = _PointField
        try:
            field = _point_field("z", 8)
        finally:
            real_ros_bridge.PointField = original_point_field

        self.assertEqual(field.name, "z")
        self.assertEqual(field.offset, 8)
        self.assertEqual(field.datatype, _PointField.FLOAT32)
        self.assertEqual(field.count, 1)

    def test_orbbec_optical_points_convert_to_ros_camera_link_axes(self) -> None:
        converted = _orbbec_optical_xyz_to_ros_camera_link(
            struct.pack("<fff", 1.0, 2.0, 3.0),
            count=1,
        )

        self.assertEqual(struct.unpack("<fff", converted), (3.0, -1.0, -2.0))

    def test_camera_info_from_metadata_scales_intrinsics(self) -> None:
        class _Header:
            frame_id = ""

        class _CameraInfo:
            def __init__(self) -> None:
                self.header = _Header()
                self.width = 0
                self.height = 0
                self.distortion_model = ""
                self.k = []
                self.p = []

        original_camera_info = real_ros_bridge.CameraInfo
        real_ros_bridge.CameraInfo = _CameraInfo
        try:
            msg = _build_camera_info_from_metadata(
                frame_id="camera",
                width=320,
                height=240,
                metadata={
                    "camera_intrinsics": {
                        "fx": 500.0,
                        "fy": 520.0,
                        "cx": 300.0,
                        "cy": 220.0,
                        "width": 640,
                        "height": 480,
                    }
                },
            )
        finally:
            real_ros_bridge.CameraInfo = original_camera_info

        self.assertIsNotNone(msg)
        self.assertEqual(msg.width, 320)
        self.assertEqual(msg.height, 240)
        self.assertAlmostEqual(msg.k[0], 250.0)
        self.assertAlmostEqual(msg.k[4], 260.0)
        self.assertAlmostEqual(msg.k[2], 150.0)
        self.assertAlmostEqual(msg.k[5], 110.0)

    def test_synthesize_scan_from_depth_be_matches_row_path(self) -> None:
        rows = (
            (1000, 0, 2000),
            (1500, 1200, 0),
            (0, 1800, 2500),
        )
        depth_be = b"".join(value.to_bytes(2, "big") for row in rows for value in row)

        from_rows = synthesize_scan_from_depth_rows(
            rows,
            horizontal_fov_rad=1.0,
            band_height_px=2,
            range_min_m=0.05,
            range_max_m=6.0,
        )
        from_bytes = synthesize_scan_from_depth_be(
            depth_be,
            width=3,
            height=3,
            horizontal_fov_rad=1.0,
            band_height_px=2,
            range_min_m=0.05,
            range_max_m=6.0,
        )

        self.assertEqual(from_bytes, from_rows)


if __name__ == "__main__":
    unittest.main()
