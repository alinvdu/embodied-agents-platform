from __future__ import annotations

import math
import unittest

from xlerobot_playground.rgbd_visual_odometry import (
    PlanarPose,
    RgbdVisualOdometryNode,
    angle_wrap,
    build_parser,
    camera_optical_translation_to_base_planar,
    compose_planar,
    config_from_args,
    yaw_from_quaternion_xyzw,
    yaw_to_quaternion_xyzw,
)


class RgbdVisualOdometryHelperTests(unittest.TestCase):
    def test_compose_planar_forward(self) -> None:
        pose = compose_planar(PlanarPose(1.0, 2.0, math.pi / 2.0), 0.10, 0.0)

        self.assertAlmostEqual(pose.x, 1.0)
        self.assertAlmostEqual(pose.y, 2.10)
        self.assertAlmostEqual(pose.yaw, math.pi / 2.0)

    def test_compose_planar_wraps_yaw(self) -> None:
        pose = compose_planar(PlanarPose(0.0, 0.0, math.radians(175.0)), 0.0, math.radians(20.0))

        self.assertAlmostEqual(pose.yaw, math.radians(-165.0))

    def test_yaw_quaternion_identity(self) -> None:
        x, y, z, w = yaw_to_quaternion_xyzw(0.0)

        self.assertEqual((x, y, z), (0.0, 0.0, 0.0))
        self.assertEqual(w, 1.0)

    def test_yaw_from_quaternion_round_trip(self) -> None:
        x, y, z, w = yaw_to_quaternion_xyzw(math.radians(90.0))

        self.assertAlmostEqual(yaw_from_quaternion_xyzw(x, y, z, w), math.radians(90.0))

    def test_parser_config_converts_degrees(self) -> None:
        args = build_parser().parse_args(["--max-yaw-step-deg", "15", "--min-yaw-update-deg", "1.5", "--min-matches", "8"])
        config = config_from_args(args)

        self.assertEqual(config.min_matches, 8)
        self.assertAlmostEqual(config.max_yaw_step_rad, math.radians(15.0))
        self.assertAlmostEqual(config.min_yaw_update_rad, math.radians(1.5))

    def test_parser_can_disable_jitter_threshold(self) -> None:
        default_config = config_from_args(build_parser().parse_args([]))
        debug_config = config_from_args(build_parser().parse_args(["--no-jitter-threshold"]))

        self.assertTrue(default_config.jitter_threshold)
        self.assertFalse(debug_config.jitter_threshold)

    def test_parser_uses_scan_active_topic_for_odom_freeze(self) -> None:
        config = config_from_args(
            build_parser().parse_args(
                [
                    "--scan-active-topic",
                    "/scan/is_active",
                    "--nav-active-topic",
                    "/nav/is_active",
                    "--scan-active-stale-timeout-s",
                    "6.5",
                    "--no-freeze-orientation-during-scan",
                    "--no-odom-requires-nav-active",
                ]
            )
        )

        self.assertEqual(config.scan_active_topic, "/scan/is_active")
        self.assertEqual(config.nav_active_topic, "/nav/is_active")
        self.assertEqual(config.scan_active_stale_timeout_s, 6.5)
        self.assertTrue(config.freeze_odom_during_scan)
        self.assertFalse(config.freeze_orientation_during_scan)
        self.assertFalse(config.odom_requires_nav_active)

    def test_compat_freeze_flag_maps_to_scan_orientation_freeze(self) -> None:
        config = config_from_args(build_parser().parse_args(["--no-freeze-during-head-motion"]))

        self.assertFalse(config.freeze_orientation_during_scan)

    def test_translation_freeze_defaults_to_scan_active(self) -> None:
        node = object.__new__(RgbdVisualOdometryNode)
        node.config = type("Config", (), {"freeze_odom_during_scan": True, "scan_active_stale_timeout_s": 0.0})()
        node._scan_orientation_frozen = False
        node._scan_active_last_true_s = None

        self.assertFalse(node._translation_freeze_active())

        node._scan_orientation_frozen = True
        self.assertTrue(node._translation_freeze_active())

    def test_odom_freezes_when_nav_inactive_is_required(self) -> None:
        node = object.__new__(RgbdVisualOdometryNode)
        node.config = type(
            "Config",
            (),
            {
                "freeze_odom_during_scan": False,
                "freeze_orientation_during_scan": False,
                "scan_active_stale_timeout_s": 0.0,
                "odom_requires_nav_active": True,
            },
        )()
        node._scan_orientation_frozen = False
        node._scan_active_last_true_s = None
        node._nav_active = False

        self.assertTrue(node._translation_freeze_active())
        self.assertTrue(node._orientation_freeze_active())

        node._nav_active = True

        self.assertFalse(node._translation_freeze_active())
        self.assertFalse(node._orientation_freeze_active())

    def test_angle_wrap(self) -> None:
        self.assertAlmostEqual(angle_wrap(math.radians(181.0)), math.radians(-179.0))

    def test_camera_pitch_projects_optical_translation_to_base(self) -> None:
        forward, left = camera_optical_translation_to_base_planar(
            camera_x_m=0.0,
            camera_y_m=0.0,
            camera_z_m=1.0,
            pitch_rad=math.radians(30.0),
        )

        self.assertAlmostEqual(forward, math.cos(math.radians(30.0)))
        self.assertAlmostEqual(left, 0.0)

        forward, left = camera_optical_translation_to_base_planar(
            camera_x_m=0.1,
            camera_y_m=0.2,
            camera_z_m=1.0,
            pitch_rad=0.0,
        )

        self.assertAlmostEqual(forward, 1.0)
        self.assertAlmostEqual(left, -0.1)

    def test_imu_arrival_age_is_independent_of_header_stamp(self) -> None:
        class _ClockTime:
            nanoseconds = 10_000_000_000

        class _Clock:
            def now(self) -> _ClockTime:
                return _ClockTime()

        node = object.__new__(RgbdVisualOdometryNode)
        node.config = type("Config", (), {"imu_stale_after_s": 0.5})()
        node.get_clock = lambda: _Clock()
        node._latest_imu_received_s = 9.8
        node._latest_imu_orientation_unwrapped_yaw_rad = math.radians(45.0)
        node._imu_orientation_origin_yaw_rad = math.radians(5.0)

        self.assertAlmostEqual(node._relative_imu_yaw_rad(), math.radians(40.0))

        node._latest_imu_received_s = 9.0
        self.assertIsNone(node._relative_imu_yaw_rad())

    def test_imu_delta_yaw_uses_absolute_yaw_when_available(self) -> None:
        node = object.__new__(RgbdVisualOdometryNode)
        node.pose = PlanarPose(0.0, 0.0, math.radians(10.0))
        node.latest_imu = None

        delta = node._imu_delta_yaw_rad(
            predicted_yaw_rad=math.radians(2.0),
            absolute_imu_yaw_rad=math.radians(25.0),
        )

        self.assertAlmostEqual(delta, math.radians(15.0))

    def test_imu_delta_yaw_uses_prediction_when_absolute_yaw_missing(self) -> None:
        node = object.__new__(RgbdVisualOdometryNode)
        node.pose = PlanarPose(0.0, 0.0, 0.0)
        node.latest_imu = object()

        delta = node._imu_delta_yaw_rad(
            predicted_yaw_rad=math.radians(-3.0),
            absolute_imu_yaw_rad=None,
        )

        self.assertAlmostEqual(delta, math.radians(-3.0))

    def test_trusted_filtered_yaw_sets_pose_yaw_directly(self) -> None:
        node = object.__new__(RgbdVisualOdometryNode)
        node.config = type("Config", (), {"min_yaw_update_rad": 0.0})()
        node.pose = PlanarPose(1.0, 2.0, math.radians(5.0))

        applied = node._apply_trusted_imu_yaw(
            absolute_imu_yaw_rad=math.radians(45.0),
            orientation_frozen=False,
        )

        self.assertTrue(applied)
        self.assertAlmostEqual(node.pose.x, 1.0)
        self.assertAlmostEqual(node.pose.y, 2.0)
        self.assertAlmostEqual(node.pose.yaw, math.radians(45.0))

    def test_trusted_filtered_yaw_does_not_override_scan_freeze(self) -> None:
        node = object.__new__(RgbdVisualOdometryNode)
        node.config = type("Config", (), {"min_yaw_update_rad": 0.0})()
        node.pose = PlanarPose(1.0, 2.0, math.radians(5.0))

        applied = node._apply_trusted_imu_yaw(
            absolute_imu_yaw_rad=math.radians(45.0),
            orientation_frozen=True,
        )

        self.assertFalse(applied)
        self.assertAlmostEqual(node.pose.yaw, math.radians(5.0))

    def test_imu_origin_reset_respects_yaw_sign(self) -> None:
        node = object.__new__(RgbdVisualOdometryNode)
        node.config = type("Config", (), {"odom_yaw_sign": -1.0})()
        node.pose = PlanarPose(0.0, 0.0, math.radians(15.0))
        node._latest_imu_orientation_unwrapped_yaw_rad = math.radians(30.0)
        node._last_prediction_stamp_s = 123.0
        node._pending_predicted_yaw_rad = math.radians(1.0)

        node._reset_imu_origin_to_current_pose()

        self.assertAlmostEqual(node._imu_orientation_origin_yaw_rad, math.radians(45.0))
        self.assertIsNone(node._last_prediction_stamp_s)
        self.assertAlmostEqual(node._pending_predicted_yaw_rad, 0.0)

    def test_trusted_filtered_yaw_respects_min_yaw_threshold(self) -> None:
        node = object.__new__(RgbdVisualOdometryNode)
        node.config = type("Config", (), {"min_yaw_update_rad": math.radians(2.0)})()
        node.pose = PlanarPose(1.0, 2.0, math.radians(5.0))

        applied = node._apply_trusted_imu_yaw(
            absolute_imu_yaw_rad=math.radians(6.0),
            orientation_frozen=False,
        )

        self.assertFalse(applied)
        self.assertAlmostEqual(node.pose.yaw, math.radians(5.0))

    def test_predicted_yaw_threshold_discards_small_updates(self) -> None:
        node = object.__new__(RgbdVisualOdometryNode)
        node.config = type("Config", (), {"min_yaw_update_rad": math.radians(2.0)})()
        node._pending_predicted_yaw_rad = math.radians(0.25)

        first = node._filter_imu_yaw_delta(math.radians(0.75), absolute_imu_yaw_rad=None)
        second = node._filter_imu_yaw_delta(math.radians(0.75), absolute_imu_yaw_rad=None)
        third = node._filter_imu_yaw_delta(math.radians(0.75), absolute_imu_yaw_rad=None)
        fourth = node._filter_imu_yaw_delta(math.radians(2.25), absolute_imu_yaw_rad=None)

        self.assertAlmostEqual(first, 0.0)
        self.assertAlmostEqual(second, 0.0)
        self.assertAlmostEqual(third, 0.0)
        self.assertAlmostEqual(fourth, math.radians(2.25))
        self.assertAlmostEqual(node._pending_predicted_yaw_rad, 0.0)

    def test_orientation_freeze_depends_on_scan_event_not_pan(self) -> None:
        node = object.__new__(RgbdVisualOdometryNode)
        node.config = type("Config", (), {"freeze_orientation_during_scan": True, "freeze_odom_during_scan": False, "scan_active_stale_timeout_s": 0.0})()
        node._scan_orientation_frozen = False
        node._scan_active_last_true_s = None
        node.camera_pan_rad = math.radians(90.0)

        self.assertFalse(node._orientation_freeze_active())

        node._scan_orientation_frozen = True
        self.assertTrue(node._orientation_freeze_active())

    def test_pan_callback_only_updates_camera_state(self) -> None:
        node = object.__new__(RgbdVisualOdometryNode)
        node.camera_pan_rad = 0.0
        node._last_camera_pan_rad = 0.0

        node._on_camera_pan(type("Message", (), {"data": math.radians(45.0)})())

        self.assertAlmostEqual(node.camera_pan_rad, math.radians(45.0))
        self.assertAlmostEqual(node._last_camera_pan_rad, math.radians(45.0))
        self.assertFalse(hasattr(node, "_last_head_motion_s"))


if __name__ == "__main__":
    unittest.main()
