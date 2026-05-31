from __future__ import annotations

import math
import unittest

from xlerobot_playground.wheel_odometry import (
    PlanarPose,
    WheelOdometryConfig,
    WheelStateSample,
    apply_planar_offset,
    build_parser,
    config_from_args,
    integrate_differential_drive,
    integrate_wheel_state_delta,
    parse_wheel_state_sample,
    remove_planar_offset,
    unwrap_encoder_delta_ticks,
    wheel_ticks_to_distance_m,
)


class WheelOdometryTests(unittest.TestCase):
    def test_parser_defaults_match_xlerobot_two_wheel_base(self) -> None:
        args = build_parser().parse_args([])
        config = config_from_args(args)

        self.assertEqual(config.robot_brain_url, "http://127.0.0.1:8765")
        self.assertEqual(config.wheel_state_transport, "websocket")
        self.assertEqual(config.odom_topic, "/odom")
        self.assertEqual(config.odom_reset_topic, "/xlerobot/odom/set_pose")
        self.assertEqual(config.left_wheel_motor, "base_left_wheel")
        self.assertEqual(config.right_wheel_motor, "base_right_wheel")
        self.assertEqual(config.left_wheel_position_sign, -1.0)
        self.assertEqual(config.right_wheel_position_sign, 1.0)
        self.assertFalse(config.odom_requires_nav_active)
        self.assertEqual(config.publish_rate_hz, 100.0)
        self.assertEqual(config.http_timeout_s, 2.0)
        self.assertEqual(config.wheel_state_ws_path, "/ws/wheel_state")
        self.assertEqual(config.wheel_state_ws_queue_size, 4096)
        self.assertEqual(config.base_link_x_from_wheel_axle_m, 0.0)
        self.assertEqual(config.base_link_y_from_wheel_axle_m, 0.0)

    def test_parser_can_use_http_wheel_state_transport(self) -> None:
        args = build_parser().parse_args(
            [
                "--wheel-state-transport",
                "http",
                "--wheel-state-ws-path",
                "/custom/ws/wheels",
                "--wheel-state-ws-reconnect-delay-s",
                "0.25",
            ]
        )
        config = config_from_args(args)

        self.assertEqual(config.wheel_state_transport, "http")
        self.assertEqual(config.wheel_state_ws_path, "/custom/ws/wheels")
        self.assertEqual(config.wheel_state_ws_reconnect_delay_s, 0.25)

    def test_parser_accepts_base_link_offset_from_wheel_axle(self) -> None:
        args = build_parser().parse_args(
            [
                "--base-link-x-from-wheel-axle-m",
                "0.196",
                "--base-link-y-from-wheel-axle-m",
                "0.01",
            ]
        )
        config = config_from_args(args)

        self.assertEqual(config.base_link_x_from_wheel_axle_m, 0.196)
        self.assertEqual(config.base_link_y_from_wheel_axle_m, 0.01)

    def test_unwrap_encoder_delta_handles_single_turn_wrap(self) -> None:
        self.assertEqual(
            unwrap_encoder_delta_ticks(5, 4090, ticks_per_revolution=4096),
            11,
        )
        self.assertEqual(
            unwrap_encoder_delta_ticks(4090, 5, ticks_per_revolution=4096),
            -11,
        )

    def test_wheel_tick_distance_applies_motor_mount_sign(self) -> None:
        distance = wheel_ticks_to_distance_m(
            -4096,
            ticks_per_revolution=4096,
            wheel_radius_m=0.05,
            position_sign=-1.0,
        )

        self.assertAlmostEqual(distance, 2.0 * math.pi * 0.05)

    def test_forward_encoder_deltas_move_pose_forward(self) -> None:
        config = WheelOdometryConfig(wheel_radius_m=0.05, wheel_track_width_m=0.25)
        previous = WheelStateSample(timestamp_s=0.0, left_position_ticks=0, right_position_ticks=0)
        current = WheelStateSample(timestamp_s=1.0, left_position_ticks=-1024, right_position_ticks=1024)

        step = integrate_wheel_state_delta(
            PlanarPose(0.0, 0.0, 0.0),
            previous_sample=previous,
            current_sample=current,
            config=config,
        )

        self.assertAlmostEqual(step.pose.x, 0.5 * math.pi * 0.05)
        self.assertAlmostEqual(step.pose.y, 0.0)
        self.assertAlmostEqual(step.pose.yaw, 0.0)

    def test_opposite_wheel_distances_rotate_axle_in_place(self) -> None:
        step = integrate_differential_drive(
            PlanarPose(1.0, 2.0, 0.0),
            left_distance_m=-0.05,
            right_distance_m=0.05,
            wheel_track_width_m=0.25,
        )

        self.assertAlmostEqual(step.forward_m, 0.0)
        self.assertAlmostEqual(step.yaw_delta_rad, 0.4)
        self.assertAlmostEqual(step.pose.x, 1.0)
        self.assertAlmostEqual(step.pose.y, 2.0)
        self.assertAlmostEqual(step.pose.yaw, 0.4)

    def test_base_link_offset_moves_during_axle_centered_rotation(self) -> None:
        axle_pose = PlanarPose(0.0, 0.0, math.pi / 2.0)

        base_pose = apply_planar_offset(
            axle_pose,
            x_offset_m=0.2,
            y_offset_m=0.0,
        )

        self.assertAlmostEqual(base_pose.x, 0.0, places=6)
        self.assertAlmostEqual(base_pose.y, 0.2, places=6)
        self.assertAlmostEqual(base_pose.yaw, math.pi / 2.0)

    def test_base_link_offset_round_trips_to_axle_pose(self) -> None:
        axle_pose = PlanarPose(1.0, 2.0, math.radians(30.0))
        base_pose = apply_planar_offset(
            axle_pose,
            x_offset_m=0.2,
            y_offset_m=-0.03,
        )

        recovered = remove_planar_offset(
            base_pose,
            x_offset_m=0.2,
            y_offset_m=-0.03,
        )

        self.assertAlmostEqual(recovered.x, axle_pose.x)
        self.assertAlmostEqual(recovered.y, axle_pose.y)
        self.assertAlmostEqual(recovered.yaw, axle_pose.yaw)

    def test_parse_wheel_state_sample_uses_configured_motor_names(self) -> None:
        sample = parse_wheel_state_sample(
            {
                "timestamp_s": 123.4,
                "positions_raw": {"left": -10, "right": 12},
                "velocities_raw": {"left": -3, "right": 4},
            },
            left_wheel_motor="left",
            right_wheel_motor="right",
        )

        self.assertEqual(sample.timestamp_s, 123.4)
        self.assertEqual(sample.left_position_ticks, -10)
        self.assertEqual(sample.right_position_ticks, 12)
        self.assertEqual(sample.left_velocity_raw, -3)
        self.assertEqual(sample.right_velocity_raw, 4)


if __name__ == "__main__":
    unittest.main()
