from __future__ import annotations

import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from xlerobot_agent.exploration import Pose2D
from xlerobot_playground.ros_nav2_runtime import (
    RosRuntimeConfig,
    RosExplorationRuntime,
    RosOccupancyMap,
    apply_occupancy_grid_update,
    check_rear_axle_rotation_sweep_occupancy,
    compute_turn_command,
    default_map_updates_topic,
    fuse_projected_maps,
    rear_axle_pose_from_base_pose,
    rear_axle_rotation_sweep_base_poses,
    rear_drive_camera_center_shift,
    remaining_turn_delta_rad,
)


class RosNav2RuntimeTests(unittest.TestCase):
    def test_local_spin_uses_nav2_yaw_sign_by_default(self) -> None:
        self.assertEqual(RosRuntimeConfig().manual_spin_direction_sign, 1.0)
        self.assertEqual(RosRuntimeConfig().local_rotation_active_topic, "/xlerobot/local_rotation_active")
        self.assertFalse(RosRuntimeConfig().local_rotation_safety_enabled)
        self.assertAlmostEqual(RosRuntimeConfig().base_link_x_from_wheel_axle_m, 0.0)
        self.assertGreaterEqual(RosRuntimeConfig().rgbd_update_timeout_s, 0.5)
        self.assertAlmostEqual(RosRuntimeConfig().rgbd_fallback_horizontal_fov_deg, 64.0)
        self.assertAlmostEqual(RosRuntimeConfig().camera_center_forward_m, 0.24)

    def test_rear_drive_camera_center_shift_reports_unavoidable_arc(self) -> None:
        shift = rear_drive_camera_center_shift(
            delta_yaw_rad=math.radians(12.0),
            camera_forward_m=0.24,
        )

        self.assertAlmostEqual(shift["distance_m"], 0.0502, places=4)
        self.assertGreater(shift["lateral_m"], 0.0)

    def test_compute_turn_command_stops_at_target(self) -> None:
        command, done = compute_turn_command(
            requested_angular_rad_s=0.3,
            target_yaw_rad=math.radians(90.0),
            feedback_yaw_rad=math.radians(89.5),
        )

        self.assertEqual(command, 0.0)
        self.assertTrue(done)

    def test_remaining_turn_delta_catches_up_segment_shortfall(self) -> None:
        self.assertAlmostEqual(
            remaining_turn_delta_rad(
                desired_total_yaw_rad=math.radians(60.0),
                achieved_total_yaw_rad=math.radians(28.0),
            ),
            math.radians(32.0),
        )

    def test_remaining_turn_delta_clamps_when_total_target_already_met(self) -> None:
        self.assertEqual(
            remaining_turn_delta_rad(
                desired_total_yaw_rad=math.radians(30.0),
                achieved_total_yaw_rad=math.radians(31.0),
            ),
            0.0,
        )

    def test_rear_axle_rotation_sweep_moves_base_link_in_arc(self) -> None:
        start = Pose2D(0.2, 0.0, 0.0)

        axle = rear_axle_pose_from_base_pose(start, x_offset_m=0.2, y_offset_m=0.0)
        poses = rear_axle_rotation_sweep_base_poses(
            start_base_pose=start,
            delta_yaw_rad=math.pi / 2.0,
            x_offset_m=0.2,
            y_offset_m=0.0,
            sample_step_rad=math.radians(10.0),
        )

        self.assertAlmostEqual(axle.x, 0.0)
        self.assertAlmostEqual(axle.y, 0.0)
        self.assertAlmostEqual(poses[-1].x, 0.0, places=6)
        self.assertAlmostEqual(poses[-1].y, 0.2, places=6)
        self.assertAlmostEqual(poses[-1].yaw, math.pi / 2.0)

    def test_rear_axle_rotation_sweep_occupancy_blocks_future_body_arc(self) -> None:
        occupancy_map = RosOccupancyMap(
            resolution=0.05,
            width=24,
            height=24,
            origin_x=-0.5,
            origin_y=-0.5,
            data=tuple(0 for _ in range(24 * 24)),
        )
        blocker_x, blocker_y = occupancy_map.world_to_cell(0.0, 0.2)
        data = list(occupancy_map.data)
        data[blocker_y * occupancy_map.width + blocker_x] = 100
        blocked_map = RosOccupancyMap(
            resolution=occupancy_map.resolution,
            width=occupancy_map.width,
            height=occupancy_map.height,
            origin_x=occupancy_map.origin_x,
            origin_y=occupancy_map.origin_y,
            data=tuple(data),
        )

        result = check_rear_axle_rotation_sweep_occupancy(
            blocked_map,
            start_base_pose=Pose2D(0.2, 0.0, 0.0),
            delta_yaw_rad=math.pi / 2.0,
            robot_length_m=0.16,
            robot_width_m=0.16,
            base_link_x_from_wheel_axle_m=0.2,
            base_link_y_from_wheel_axle_m=0.0,
            safety_padding_m=0.0,
            yaw_sample_step_rad=math.radians(5.0),
        )

        self.assertFalse(result["safe"])
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["first_blocker_cell"], {"x": blocker_x, "y": blocker_y})

    def test_rear_axle_rotation_sweep_occupancy_allows_clear_map(self) -> None:
        occupancy_map = RosOccupancyMap(
            resolution=0.05,
            width=24,
            height=24,
            origin_x=-0.5,
            origin_y=-0.5,
            data=tuple(0 for _ in range(24 * 24)),
        )

        result = check_rear_axle_rotation_sweep_occupancy(
            occupancy_map,
            start_base_pose=Pose2D(0.2, 0.0, 0.0),
            delta_yaw_rad=math.pi / 2.0,
            robot_length_m=0.16,
            robot_width_m=0.16,
            base_link_x_from_wheel_axle_m=0.2,
            base_link_y_from_wheel_axle_m=0.0,
            safety_padding_m=0.0,
            yaw_sample_step_rad=math.radians(5.0),
        )

        self.assertTrue(result["safe"])
        self.assertEqual(result["status"], "clear")

    def test_scan_active_release_waits_configured_delay(self) -> None:
        class FakeRuntime:
            def __init__(self) -> None:
                self.config = SimpleNamespace(scan_active_release_delay_s=3.0)
                self.events: list[bool] = []

            def _set_scan_active_if_available(self, active: bool) -> None:
                self.events.append(active)

        runtime = FakeRuntime()
        with patch("xlerobot_playground.ros_nav2_runtime.time.sleep") as sleep:
            RosExplorationRuntime._release_scan_active_after_delay(runtime)

        sleep.assert_called_once_with(3.0)
        self.assertEqual(runtime.events, [False])

    def test_camera_pan_scan_captures_outward_sweeps_only_and_restores_center(self) -> None:
        class FakeRuntime:
            def __init__(self) -> None:
                self.config = SimpleNamespace(
                    robot_brain_url="http://brain.local:8765",
                    camera_pan_action_key="head_motor_1.pos",
                    camera_pan_settle_s=0.0,
                    camera_pan_step_deg=60.0,
                    camera_pan_compute_s=0.0,
                    turn_scan_settle_s=0.0,
                    server_timeout_s=1.0,
                    publish_internal_navigation_map=False,
                    map_frame="base_link",
                )
                self.commands: list[float] = []
                self.hold_count = 0
                self.scan_observations = [{"pose": Pose2D(0.0, 0.0, 99.0)} for _ in range(99)]
                self.point_cloud_observations = []
                self._nav_scan_history = []
                self.latest_map = None
                self.latest_map_stamp_s = 0.0
                self.latest_map_header_frame_id = "base_link"

            def current_pose(self) -> Pose2D:
                return Pose2D(1.0, 2.0, 0.25)

            def _command_camera_pan(self, pan_rad: float, **_kwargs):
                self.commands.append(pan_rad)
                return {"pan_rad": pan_rad}

            def _capture_settled_scan_observation(self, **_kwargs):
                return {
                    "pose": Pose2D(1.0, 2.0, self.commands[-1]),
                    "ranges": (1.0,),
                    "angle_min": 0.0,
                    "angle_increment": 0.1,
                    "range_min": 0.05,
                    "range_max": 4.0,
                }

            def hold_stop_until_stable(self, *, duration_s: float):
                self.hold_count += 1
                return {"stable": True, "duration_s": duration_s}

            def spin_for(self, duration_s: float):
                return None

            def _wait_for_next_point_cloud_observation(self, after_index: int, *, timeout_s: float = 3.0):
                return None

            def drain_scan_observations(self, since_index: int):
                return list(self.scan_observations[since_index:]), len(self.scan_observations)

            def wait_for_map_update(self, *, after_stamp_s: float, timeout_s: float = 2.0) -> bool:
                return False

            def latest_map_summary(self):
                return None

            def _occupancy_map_summary(self, occupancy_map, *, frame_id: str, stamp_s: float):
                return {"frame_id": frame_id, "width": occupancy_map.width, "height": occupancy_map.height}

        runtime = FakeRuntime()

        result = RosExplorationRuntime._perform_camera_pan_scan(
            runtime,
            reason="test",
            should_cancel=None,
            start_time=0.0,
            start_pose=Pose2D(1.0, 2.0, 0.25),
            observation_start_index=10,
            sample_count=6,
            event={"reason": "test", "mode": "camera_pan", "sample_count": 6},
        )

        expected = [
            0.0,
            math.radians(60.0),
            math.radians(120.0),
            math.pi,
            0.0,
            math.radians(-60.0),
            math.radians(-120.0),
            0.0,
        ]
        self.assertEqual(len(runtime.commands), len(expected))
        for actual, expected_value in zip(runtime.commands, expected):
            self.assertAlmostEqual(actual, expected_value)
        self.assertEqual(len(result["observations"]), 7)
        self.assertEqual(result["camera_pan_settled_sample_count"], 7)
        self.assertEqual(result["observation_stop_index"], 99)
        self.assertEqual(result["raw_observation_count"], 89)
        self.assertEqual(result["scan_stop_reason"], "completed")

    def test_fuse_projected_maps_preserves_occupied_over_free(self) -> None:
        first = RosOccupancyMap(
            resolution=1.0,
            width=2,
            height=1,
            origin_x=0.0,
            origin_y=0.0,
            data=(100, -1),
        )
        second = RosOccupancyMap(
            resolution=1.0,
            width=2,
            height=1,
            origin_x=0.0,
            origin_y=0.0,
            data=(0, 0),
        )

        fused = fuse_projected_maps([first, second])

        self.assertIsNotNone(fused)
        assert fused is not None
        self.assertEqual(fused.data, (100, -1))

    def test_fuse_projected_maps_accumulates_repeated_free_observations(self) -> None:
        first = RosOccupancyMap(
            resolution=1.0,
            width=1,
            height=1,
            origin_x=0.0,
            origin_y=0.0,
            data=(0,),
        )
        second = RosOccupancyMap(
            resolution=1.0,
            width=1,
            height=1,
            origin_x=0.0,
            origin_y=0.0,
            data=(0,),
        )

        fused = fuse_projected_maps([first, second])

        self.assertIsNotNone(fused)
        assert fused is not None
        self.assertEqual(fused.data, (0,))

    def test_apply_occupancy_grid_update_patches_existing_map(self) -> None:
        occupancy_map = RosOccupancyMap(
            resolution=1.0,
            width=4,
            height=3,
            origin_x=0.0,
            origin_y=0.0,
            data=(-1,) * 12,
        )

        updated = apply_occupancy_grid_update(
            occupancy_map,
            update_x=1,
            update_y=1,
            update_width=2,
            update_height=2,
            update_data=(0, 100, 100, 0),
        )

        self.assertEqual(
            updated.data,
            (
                -1,
                -1,
                -1,
                -1,
                -1,
                0,
                100,
                -1,
                -1,
                100,
                0,
                -1,
            ),
        )

    def test_default_map_updates_topic_matches_rviz_projected_map_convention(self) -> None:
        self.assertEqual(default_map_updates_topic("/projected_map"), "/projected_map_updates")

    def test_camera_pan_scan_pose_keeps_tf_head_yaw(self) -> None:
        class FakeRuntime:
            def __init__(self) -> None:
                self.config = SimpleNamespace(publish_internal_navigation_map=True)
                self._use_turn_feedback_for_scan_pose = False
                self._scan_sensor_yaw_offset_rad = None

            def _current_turn_feedback(self):
                return "imu", math.radians(10.0)

        sensor_pose = Pose2D(1.0, 2.0, math.radians(75.0))

        adjusted = RosExplorationRuntime._scan_pose_with_turn_feedback(FakeRuntime(), sensor_pose)

        self.assertEqual(adjusted, sensor_pose)

    def test_robot_spin_scan_pose_can_use_imu_yaw_feedback(self) -> None:
        class FakeRuntime:
            def __init__(self) -> None:
                self.config = SimpleNamespace(publish_internal_navigation_map=True)
                self._use_turn_feedback_for_scan_pose = True
                self._scan_sensor_yaw_offset_rad = None
                self.feedback_yaw = math.radians(10.0)

            def _current_turn_feedback(self):
                return "imu", self.feedback_yaw

        runtime = FakeRuntime()

        first = RosExplorationRuntime._scan_pose_with_turn_feedback(
            runtime,
            Pose2D(1.0, 2.0, math.radians(75.0)),
        )
        runtime.feedback_yaw = math.radians(30.0)
        second = RosExplorationRuntime._scan_pose_with_turn_feedback(
            runtime,
            Pose2D(1.0, 2.0, math.radians(75.0)),
        )

        self.assertAlmostEqual(first.yaw, math.radians(75.0))
        self.assertAlmostEqual(second.yaw, math.radians(95.0))

    def test_unknown_turn_scan_mode_is_rejected(self) -> None:
        class FakeRuntime:
            config = SimpleNamespace(
                turn_scan_mode="bad",
                camera_pan_sample_count=12,
                turn_scan_radians=math.tau,
            )
            scan_observations = []

            def current_pose(self):
                return None

        with self.assertRaises(ValueError):
            RosExplorationRuntime.perform_turnaround_scan(FakeRuntime(), reason="test")


if __name__ == "__main__":
    unittest.main()
