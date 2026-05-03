from __future__ import annotations

import unittest

from xlerobot_playground.real_nav2_config import build_parser


class RealNav2ConfigTests(unittest.TestCase):
    def test_parser_defaults_are_real_robot_safe(self) -> None:
        args = build_parser().parse_args([])

        self.assertEqual(args.scan_topic, "/scan")
        self.assertEqual(args.map_frame, "map")
        self.assertEqual(args.odom_frame, "odom")
        self.assertEqual(args.base_frame, "base_link")
        self.assertEqual(args.max_linear_velocity, 0.03)
        self.assertEqual(args.max_angular_velocity, 0.18)
        self.assertEqual(args.min_linear_velocity_threshold, 0.01)
        self.assertEqual(args.min_angular_velocity_threshold, 0.02)
        self.assertEqual(args.min_speed_theta, 0.02)
        self.assertEqual(args.rotate_to_goal_slowing_factor, 1.0)
        self.assertEqual(args.progress_required_movement_radius, 0.05)
        self.assertEqual(args.progress_movement_time_allowance_s, 25.0)
        self.assertEqual(args.xy_goal_tolerance_m, 0.18)
        self.assertEqual(args.yaw_goal_tolerance_rad, 3.14)


if __name__ == "__main__":
    unittest.main()
