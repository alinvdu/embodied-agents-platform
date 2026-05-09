from __future__ import annotations

import unittest

import numpy as np

from xlerobot_playground.maniskill_ros_bridge import build_parser, synthesize_scan_from_depth
from xlerobot_playground.nav2_params import patch_nav2_params, rectangular_footprint, render_slam_toolbox_params


class Nav2BridgeTests(unittest.TestCase):
    def test_bridge_parser_accepts_nav2_relevant_topics(self) -> None:
        args = build_parser().parse_args(
            [
                "--env-id",
                "ReplicaCAD_SceneManipulation-v1",
                "--cmd-vel-topic",
                "/robot/cmd_vel",
                "--scan-topic",
                "/robot/scan",
                "--linear-cmd-gain",
                "1.5",
                "--angular-cmd-gain",
                "0.32",
                "--laser-max-range-m",
                "8.5",
                "--max-episode-steps",
                "1500",
                "--build-config-idx",
                "2",
                "--spawn-x",
                "-0.2",
                "--spawn-y",
                "-1.0",
                "--spawn-yaw",
                "1.57",
                "--ros-base-yaw-offset-rad",
                "3.14159",
                "--no-publish-head-camera",
                "--no-laser-fill-no-return",
            ]
        )
        self.assertEqual(args.env_id, "ReplicaCAD_SceneManipulation-v1")
        self.assertEqual(args.cmd_vel_topic, "/robot/cmd_vel")
        self.assertEqual(args.scan_topic, "/robot/scan")
        self.assertEqual(args.linear_cmd_gain, 1.5)
        self.assertEqual(args.angular_cmd_gain, 0.32)
        self.assertEqual(args.laser_max_range_m, 8.5)
        self.assertEqual(args.max_episode_steps, 1500)
        self.assertEqual(args.build_config_idx, 2)
        self.assertEqual(args.spawn_x, -0.2)
        self.assertEqual(args.spawn_y, -1.0)
        self.assertEqual(args.spawn_yaw, 1.57)
        self.assertEqual(args.ros_base_yaw_offset_rad, 3.14159)
        self.assertFalse(args.publish_head_camera)
        self.assertEqual(args.scan_band_height_px, 12)
        self.assertFalse(args.laser_fill_no_return)

    def test_depth_band_is_converted_into_laser_ranges(self) -> None:
        depth_mm = [[1000] * 5 for _ in range(7)]
        ranges, angles = synthesize_scan_from_depth(
            depth_mm=np.asarray(depth_mm, dtype="int16"),
            horizontal_fov_rad=1.0,
            band_height_px=3,
            range_min_m=0.05,
            range_max_m=10.0,
        )
        self.assertEqual(len(ranges), 5)
        self.assertEqual(len(angles), 5)
        self.assertTrue(all(value >= 1.0 for value in ranges[1:4]))
        self.assertLess(float(angles[0]), 0.0)
        self.assertGreater(float(angles[-1]), 0.0)
        self.assertGreater(float(angles[1] - angles[0]), 0.0)

    def test_depth_scan_fills_no_return_beams_for_clearing(self) -> None:
        depth_mm = np.zeros((9, 5), dtype="int16")
        depth_mm[:, 2] = 1000
        ranges, _ = synthesize_scan_from_depth(
            depth_mm=depth_mm,
            horizontal_fov_rad=1.0,
            band_height_px=5,
            range_min_m=0.05,
            range_max_m=4.0,
        )
        self.assertEqual(float(ranges[0]), 4.0)
        self.assertLess(float(ranges[2]), 1.1)
        self.assertEqual(float(ranges[-1]), 4.0)

    def test_nav2_params_are_patched_to_bridge_topics(self) -> None:
        base = {
            "global_costmap": {
                "global_costmap": {
                    "ros__parameters": {
                        "plugins": ["static_layer", "obstacle_layer", "inflation_layer"],
                        "obstacle_layer": {},
                        "inflation_layer": {},
                    }
                }
            },
            "local_costmap": {
                "local_costmap": {
                    "ros__parameters": {
                        "plugins": ["voxel_layer", "inflation_layer"],
                        "voxel_layer": {},
                        "inflation_layer": {},
                    }
                }
            },
            "bt_navigator": {"ros__parameters": {}},
            "amcl": {"ros__parameters": {}},
            "behavior_server": {"ros__parameters": {}},
            "controller_server": {
                "ros__parameters": {
                    "FollowPath": {
                        "max_vel_x": 0.26,
                        "max_speed_xy": 0.26,
                        "max_vel_theta": 1.0,
                        "trans_stopped_velocity": 0.25,
                        "xy_goal_tolerance": 0.25,
                        "critics": ["RotateToGoal", "Oscillation", "GoalAlign", "PathAlign", "PathDist", "GoalDist"],
                        "PathAlign.scale": 32.0,
                        "GoalAlign.scale": 24.0,
                        "RotateToGoal.scale": 32.0,
                        "RotateToGoal.slowing_factor": 5.0,
                    }
                }
            },
            "planner_server": {"ros__parameters": {}},
            "velocity_smoother": {
                "ros__parameters": {
                    "max_velocity": [0.26, 0.0, 1.0],
                    "min_velocity": [-0.26, 0.0, -1.0],
                }
            },
        }
        patched = patch_nav2_params(
            base,
            scan_topic="/bridge/scan",
            base_frame="base_link",
            odom_frame="odom",
            map_frame="map",
            robot_radius=0.3,
            footprint=rectangular_footprint(length_m=0.3913, width_m=0.459),
        )

        local_costmap = patched["local_costmap"]["local_costmap"]["ros__parameters"]
        self.assertEqual(local_costmap["plugins"], ["static_layer", "inflation_layer"])
        self.assertNotIn("voxel_layer", local_costmap)
        self.assertNotIn("obstacle_layer", local_costmap)
        self.assertEqual(local_costmap["static_layer"]["map_topic"], "/projected_map")
        self.assertTrue(local_costmap["static_layer"]["subscribe_to_updates"])
        self.assertFalse(local_costmap["static_layer"]["map_subscribe_transient_local"])
        self.assertEqual(
            patched["global_costmap"]["global_costmap"]["ros__parameters"]["static_layer"]["map_topic"],
            "/projected_map",
        )
        self.assertNotIn("obstacle_layer", patched["global_costmap"]["global_costmap"]["ros__parameters"]["plugins"])
        self.assertNotIn("obstacle_layer", patched["global_costmap"]["global_costmap"]["ros__parameters"])
        self.assertEqual(
            patched["global_costmap"]["global_costmap"]["ros__parameters"]["footprint"],
            "[[0.1956, 0.2295], [0.1956, -0.2295], [-0.1956, -0.2295], [-0.1956, 0.2295]]",
        )
        self.assertNotIn("robot_radius", patched["global_costmap"]["global_costmap"]["ros__parameters"])
        self.assertNotIn("robot_radius", patched["local_costmap"]["local_costmap"]["ros__parameters"])
        self.assertIn("inflation_layer", patched["global_costmap"]["global_costmap"]["ros__parameters"]["plugins"])
        self.assertEqual(
            patched["global_costmap"]["global_costmap"]["ros__parameters"]["inflation_layer"]["inflation_radius"],
            0.08,
        )
        self.assertEqual(
            patched["global_costmap"]["global_costmap"]["ros__parameters"]["inflation_layer"][
                "cost_scaling_factor"
            ],
            4.0,
        )
        self.assertIn("inflation_layer", patched["local_costmap"]["local_costmap"]["ros__parameters"]["plugins"])
        self.assertEqual(
            patched["local_costmap"]["local_costmap"]["ros__parameters"]["inflation_layer"]["inflation_radius"],
            0.08,
        )
        self.assertEqual(
            patched["local_costmap"]["local_costmap"]["ros__parameters"]["inflation_layer"]["cost_scaling_factor"],
            4.0,
        )
        self.assertEqual(patched["global_costmap"]["global_costmap"]["ros__parameters"]["footprint_padding"], 0.0)
        self.assertEqual(patched["local_costmap"]["local_costmap"]["ros__parameters"]["global_frame"], "odom")
        self.assertEqual(patched["bt_navigator"]["ros__parameters"]["odom_topic"], "/odom")
        self.assertEqual(patched["amcl"]["ros__parameters"]["base_frame_id"], "base_link")
        self.assertFalse(patched["amcl"]["ros__parameters"]["tf_broadcast"])
        self.assertEqual(
            patched["controller_server"]["ros__parameters"]["FollowPath"]["max_vel_x"],
            0.65,
        )
        self.assertEqual(
            patched["controller_server"]["ros__parameters"]["FollowPath"]["max_vel_theta"],
            0.45,
        )
        self.assertEqual(
            patched["controller_server"]["ros__parameters"]["min_x_velocity_threshold"],
            0.01,
        )
        self.assertEqual(
            patched["controller_server"]["ros__parameters"]["min_theta_velocity_threshold"],
            0.02,
        )
        self.assertEqual(
            patched["controller_server"]["ros__parameters"]["FollowPath"]["min_speed_theta"],
            0.02,
        )
        self.assertEqual(
            patched["controller_server"]["ros__parameters"]["FollowPath"]["trans_stopped_velocity"],
            0.01,
        )
        self.assertEqual(
            patched["controller_server"]["ros__parameters"]["FollowPath"]["xy_goal_tolerance"],
            0.10,
        )
        self.assertNotIn(
            "RotateToGoal",
            patched["controller_server"]["ros__parameters"]["FollowPath"]["critics"],
        )
        self.assertEqual(
            patched["velocity_smoother"]["ros__parameters"]["max_velocity"][0],
            0.65,
        )
        self.assertEqual(
            patched["velocity_smoother"]["ros__parameters"]["max_velocity"][2],
            0.45,
        )
        self.assertEqual(
            patched["velocity_smoother"]["ros__parameters"]["deadband_velocity"],
            [0.01, 0.0, 0.02],
        )
        self.assertNotIn(
            "RotateToGoal.scale",
            patched["controller_server"]["ros__parameters"]["FollowPath"],
        )
        self.assertEqual(
            patched["controller_server"]["ros__parameters"]["progress_checker"]["required_movement_radius"],
            0.05,
        )
        self.assertEqual(
            patched["controller_server"]["ros__parameters"]["progress_checker"]["movement_time_allowance"],
            25.0,
        )
        self.assertEqual(
            patched["controller_server"]["ros__parameters"]["goal_checker_plugins"],
            ["goal_checker"],
        )
        self.assertEqual(
            patched["controller_server"]["ros__parameters"]["goal_checker"]["plugin"],
            "nav2_controller::PositionGoalChecker",
        )
        self.assertEqual(
            patched["controller_server"]["ros__parameters"]["goal_checker"]["xy_goal_tolerance"],
            0.18,
        )
        self.assertTrue(patched["controller_server"]["ros__parameters"]["goal_checker"]["stateful"])

    def test_nav2_params_can_keep_local_scan_obstacles_when_explicitly_enabled(self) -> None:
        base = {
            "global_costmap": {
                "global_costmap": {"ros__parameters": {"plugins": ["static_layer"]}}
            },
            "local_costmap": {
                "local_costmap": {
                    "ros__parameters": {
                        "plugins": ["voxel_layer"],
                        "voxel_layer": {},
                    }
                }
            },
        }
        patched = patch_nav2_params(
            base,
            scan_topic="/bridge/scan",
            enable_local_scan_obstacles=True,
        )

        local_scan = patched["local_costmap"]["local_costmap"]["ros__parameters"]["voxel_layer"]["scan"]
        self.assertEqual(local_scan["topic"], "/bridge/scan")
        self.assertTrue(local_scan["inf_is_valid"])
        self.assertEqual(local_scan["observation_persistence"], 0.35)

    def test_nav2_params_can_remove_heading_alignment_critics(self) -> None:
        base = {
            "controller_server": {
                "ros__parameters": {
                    "FollowPath": {
                        "critics": ["RotateToGoal", "GoalAlign", "PathAlign", "PathDist", "GoalDist"],
                        "RotateToGoal.scale": 8.0,
                        "RotateToGoal.slowing_factor": 3.0,
                        "GoalAlign.scale": 12.0,
                        "PathAlign.scale": 16.0,
                    }
                }
            }
        }

        patched = patch_nav2_params(
            base,
            rotate_to_goal_scale=0.0,
            goal_align_scale=0.0,
            path_align_scale=0.0,
        )

        follow_path = patched["controller_server"]["ros__parameters"]["FollowPath"]
        self.assertEqual(follow_path["critics"], ["PathDist", "GoalDist"])
        self.assertNotIn("RotateToGoal.scale", follow_path)
        self.assertNotIn("GoalAlign.scale", follow_path)
        self.assertNotIn("PathAlign.scale", follow_path)

    def test_slam_toolbox_params_use_standard_frames(self) -> None:
        params = render_slam_toolbox_params()
        slam = params["slam_toolbox"]["ros__parameters"]
        self.assertEqual(slam["scan_topic"], "/scan")
        self.assertEqual(slam["map_frame"], "map")
        self.assertEqual(slam["odom_frame"], "odom")
        self.assertEqual(slam["base_frame"], "base_link")


if __name__ == "__main__":
    unittest.main()
