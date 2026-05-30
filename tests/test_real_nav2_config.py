from __future__ import annotations

import unittest

from xlerobot_playground.nav2_params import asymmetric_rectangular_footprint
from xlerobot_playground.real_nav2_config import build_parser


class RealNav2ConfigTests(unittest.TestCase):
    def test_asymmetric_footprint_places_base_link_near_rear_axle(self) -> None:
        self.assertEqual(
            asymmetric_rectangular_footprint(front_m=0.45, rear_m=0.08, width_m=0.54),
            "[[0.45, 0.27], [0.45, -0.27], [-0.08, -0.27], [-0.08, 0.27]]",
        )

    def test_parser_defaults_are_real_robot_safe(self) -> None:
        args = build_parser().parse_args([])

        self.assertEqual(args.scan_topic, "/scan")
        self.assertEqual(args.map_frame, "map")
        self.assertEqual(args.odom_frame, "odom")
        self.assertEqual(args.base_frame, "base_link")
        self.assertIsNone(args.robot_footprint_front_m)
        self.assertIsNone(args.robot_footprint_rear_m)
        self.assertEqual(args.max_linear_velocity, 0.03)
        self.assertEqual(args.max_angular_velocity, 0.18)
        self.assertEqual(args.min_linear_velocity_threshold, 0.01)
        self.assertEqual(args.min_angular_velocity_threshold, 0.02)
        self.assertEqual(args.min_speed_theta, 0.02)
        self.assertEqual(args.trans_stopped_velocity, 0.01)
        self.assertEqual(args.follow_path_xy_goal_tolerance_m, 0.10)
        self.assertEqual(args.rotate_to_goal_slowing_factor, 1.0)
        self.assertEqual(args.path_align_scale, 4.0)
        self.assertEqual(args.goal_align_scale, 0.0)
        self.assertEqual(args.oscillation_reset_dist_m, 0.01)
        self.assertEqual(args.oscillation_reset_angle_rad, 0.05)
        self.assertEqual(args.oscillation_reset_time_s, 5.0)
        self.assertEqual(args.goal_dist_scale, 8.0)
        self.assertEqual(args.rotate_to_goal_scale, 0.0)
        self.assertEqual(args.inflation_radius_m, 0.08)
        self.assertEqual(args.inflation_cost_scaling_factor, 4.0)
        self.assertEqual(args.progress_required_movement_radius, 0.01)
        self.assertEqual(args.progress_movement_time_allowance_s, 60.0)
        self.assertEqual(args.xy_goal_tolerance_m, 0.18)
        self.assertEqual(args.yaw_goal_tolerance_rad, 3.14)


if __name__ == "__main__":
    unittest.main()
