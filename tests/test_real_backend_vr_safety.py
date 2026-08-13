import math
import unittest
from types import SimpleNamespace

from xlerobot_playground.real_backend import (
    BaseSmoother,
    VrArmTuning,
    VrEpisodeBoundaryState,
    _VR_ARM_CLUTCH_RELEASE_HOLD_FRAMES,
    _force_vr_base_stop,
    _initial_vr_episode_boundary,
    _release_episode_boundary_hold,
    _so101_forward_kinematics,
    _so101_inverse_kinematics_with_limits,
    _update_yawed_vr_arm_target,
    build_parser,
)


def _arm_tuning() -> VrArmTuning:
    return VrArmTuning(
        ik_mode="yawed",
        vertical_sign=1.0,
        y_gain=1.4,
        z_gain=1.0,
        ik_alpha=0.25,
        yawed_forward_gain=1.0,
        yawed_lateral_gain=0.30,
        yawed_pan_sign=1.0,
        yawed_pan_limit=120.0,
        yawed_pan_step_limit=3.0,
        shoulder_lift_min=-108.0,
        shoulder_lift_max=96.0,
        elbow_flex_min=-115.0,
        elbow_flex_max=106.0,
        enforce_joint_limits=True,
        debug=False,
        debug_hz=2.0,
    )


class RealBackendVRSafetyTests(unittest.TestCase):
    def test_recording_starts_with_action_ready_hold(self) -> None:
        fixed_modes, boundary = _initial_vr_episode_boundary(object())

        self.assertEqual(
            fixed_modes,
            {"left": "episode_hold", "right": "episode_hold"},
        )
        self.assertEqual(boundary.phase, "await_start")
        self.assertEqual(boundary.held_sides, {"left", "right"})

        fixed_modes, boundary = _initial_vr_episode_boundary(None)
        self.assertEqual(fixed_modes, {})
        self.assertEqual(boundary.phase, "idle")
        self.assertEqual(boundary.held_sides, set())

    def test_action_ready_pose_round_trips_through_runtime_kinematics(self) -> None:
        kinematics = SimpleNamespace(l1=0.1159, l2=0.1350)

        x, y = _so101_forward_kinematics(kinematics, -44.1708, 35.0)
        shoulder, elbow = _so101_inverse_kinematics_with_limits(
            kinematics,
            x,
            y,
            _arm_tuning(),
        )

        self.assertAlmostEqual(shoulder, -44.1708, places=5)
        self.assertAlmostEqual(elbow, 35.0, places=5)

    def test_episode_release_syncs_to_observed_pose_and_rebaselines(self) -> None:
        arm = SimpleNamespace(
            target_positions={},
            kinematics=SimpleNamespace(l1=0.1159, l2=0.1350),
            prev_vr_pos=(1.0, 2.0, 3.0),
            prev_wrist_flex=10.0,
            prev_wrist_roll=20.0,
            ik_fixed_pose_paused=True,
        )
        vr_teleop = SimpleNamespace(right_arm=arm)
        state = VrEpisodeBoundaryState(phase="await_start", held_sides={"right"})
        fixed_modes = {"right": "episode_hold"}
        motion_states = {}
        targets = {
            "right_arm_shoulder_pan.pos": -4.8316,
            "right_arm_shoulder_lift.pos": -44.1708,
            "right_arm_elbow_flex.pos": 35.0,
            "right_arm_wrist_flex.pos": 36.2061,
            "right_arm_wrist_roll.pos": 0.1709,
            "right_arm_gripper.pos": 0.9466,
        }
        observed = {
            **targets,
            "right_arm_shoulder_lift.pos": -45.0,
            "right_arm_elbow_flex.pos": 34.0,
        }

        _release_episode_boundary_hold(
            state,
            fixed_modes,
            motion_states,
            vr_teleop,
            observed,
            targets,
        )

        self.assertEqual(arm.target_positions["shoulder_lift"], -45.0)
        self.assertEqual(arm.target_positions["elbow_flex"], 34.0)
        self.assertEqual(
            arm.ik_clutch_rebaseline_frames,
            _VR_ARM_CLUTCH_RELEASE_HOLD_FRAMES,
        )
        self.assertFalse(arm.ik_fixed_pose_paused)
        self.assertFalse(hasattr(arm, "prev_vr_pos"))
        self.assertEqual(state.phase, "idle")
        self.assertEqual(state.held_sides, set())
        self.assertEqual(fixed_modes, {})

    def test_yawed_lateral_pan_change_is_bounded_without_backlog(self) -> None:
        tuning = _arm_tuning()
        initial_pan = -4.8316
        radial = 0.075
        arm = SimpleNamespace(
            current_forward=radial * math.cos(math.radians(initial_pan)),
            current_lateral=radial * math.sin(math.radians(initial_pan)),
            current_height=0.127,
            current_x=radial,
            current_y=0.127,
            target_positions={"shoulder_pan": initial_pan},
        )

        _update_yawed_vr_arm_target(
            arm,
            tuning,
            delta_x=0.02,
            delta_y=0.0,
            delta_z=0.0,
        )

        new_pan = arm.target_positions["shoulder_pan"]
        self.assertLessEqual(abs(new_pan - initial_pan), tuning.yawed_pan_step_limit)
        reconstructed_pan = math.degrees(math.atan2(arm.current_lateral, arm.current_forward))
        self.assertAlmostEqual(reconstructed_pan, new_pan, places=6)

    def test_recording_base_stop_is_immediate(self) -> None:
        smoother = BaseSmoother(
            max_linear=0.25,
            max_angular=75.0,
            linear_accel=0.9,
            angular_accel=240.0,
            deadzone=0.14,
            curve=1.5,
            x_vel=0.2,
            theta_vel=60.0,
            last_t=123.0,
        )

        action = _force_vr_base_stop(
            {"right_arm_shoulder_pan.pos": -5.0, "x.vel": 0.2, "theta.vel": 60.0},
            smoother,
        )

        self.assertEqual(action["x.vel"], 0.0)
        self.assertEqual(action["theta.vel"], 0.0)
        self.assertEqual(action["right_arm_shoulder_pan.pos"], -5.0)
        self.assertEqual(smoother.x_vel, 0.0)
        self.assertEqual(smoother.theta_vel, 0.0)
        self.assertIsNone(smoother.last_t)

    def test_safe_yawed_and_recording_defaults(self) -> None:
        args = build_parser().parse_args(["manipulate"])

        self.assertEqual(args.vr_action_ready_shoulder_delta, 55.0)
        self.assertEqual(args.vr_action_ready_elbow_delta, -65.0)
        self.assertEqual(args.vr_arm_yawed_lateral_gain, 0.30)
        self.assertEqual(args.vr_arm_yawed_pan_step_limit, 3.0)
        self.assertEqual(args.vr_base_max_linear, 0.12)
        self.assertEqual(args.vr_base_max_angular, 35.0)
        self.assertFalse(args.allow_vr_base_while_recording)


if __name__ == "__main__":
    unittest.main()
