from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from xlerobot_playground.robot_brain_agent import (
    RobotBrainAgent,
    RobotBrainAgentConfig,
    build_parser,
    config_from_args,
)
from xlerobot_playground.rgbd_transport import pack_rgbd_frame


class FakeResult:
    def __init__(self, succeeded=True, message="ok", metadata=None) -> None:
        self.succeeded = succeeded
        self.message = message
        self.metadata = metadata or {}


class FakeRuntime:
    def __init__(self) -> None:
        self.velocity_calls = []
        self.stop_calls = 0
        self.close_calls = 0
        self.connected = False
        self.robot = self
        self.actions = []

    def connect(self) -> None:
        self.connected = True

    def send_action(self, action):
        self.actions.append(dict(action))
        return dict(action)

    def drive_velocity(self, *, linear_m_s: float, angular_rad_s: float):
        self.velocity_calls.append((linear_m_s, angular_rad_s))
        return FakeResult(metadata={"sent": True})

    def stop(self):
        self.stop_calls += 1
        return FakeResult(message="stopped")

    def close(self) -> None:
        self.close_calls += 1


class RobotBrainAgentTests(unittest.TestCase):
    def test_parser_defaults_match_robot_brain_deployment(self) -> None:
        args = build_parser().parse_args([])
        config = config_from_args(args)

        self.assertEqual(config.robot_kind, "xlerobot_2wheels")
        self.assertEqual(config.port1, "/dev/tty.usbmodem5B140330101")
        self.assertEqual(config.port2, "/dev/tty.usbmodem5B140332271")
        self.assertFalse(config.allow_motion_commands)
        self.assertEqual(config.port, 8765)
        self.assertFalse(config.debug_motion)
        self.assertTrue(config.use_degrees)
        self.assertEqual(config.calibration_prompt_response, "")
        self.assertEqual(config.imu_udp_host, "127.0.0.1")
        self.assertEqual(config.imu_udp_port, 8766)
        self.assertEqual(config.camera_max_frame_bytes, 16 * 1024 * 1024)
        self.assertEqual(config.camera_log_every, 30)
        self.assertEqual(config.camera_pan_action_key, "head_motor_1.pos")
        self.assertEqual(config.camera_pan_action_sign, 1.0)
        self.assertEqual(config.base_angular_action_sign, 1.0)

    def test_parser_accepts_debug_motion(self) -> None:
        args = build_parser().parse_args(["--debug-motion"])
        config = config_from_args(args)

        self.assertTrue(config.debug_motion)

    def test_parser_accepts_interactive_calibration(self) -> None:
        args = build_parser().parse_args(["--interactive-calibration"])
        config = config_from_args(args)

        self.assertIsNone(config.calibration_prompt_response)

    def test_agent_forwards_velocity_to_runtime(self) -> None:
        runtime = FakeRuntime()
        agent = RobotBrainAgent(RobotBrainAgentConfig(), runtime=runtime)

        response = agent.velocity(linear_m_s=0.02, angular_rad_s=0.08)

        self.assertTrue(response["succeeded"])
        self.assertEqual(runtime.velocity_calls, [(0.02, 0.08)])
        self.assertEqual(response["metadata"], {"sent": True})

    def test_agent_can_flip_base_angular_action_sign(self) -> None:
        runtime = FakeRuntime()
        agent = RobotBrainAgent(RobotBrainAgentConfig(base_angular_action_sign=-1.0), runtime=runtime)

        response = agent.velocity(linear_m_s=0.0, angular_rad_s=0.08)

        self.assertTrue(response["succeeded"])
        self.assertEqual(runtime.velocity_calls, [(0.0, -0.08)])

    def test_agent_commands_camera_pitch_and_updates_state(self) -> None:
        runtime = FakeRuntime()
        agent = RobotBrainAgent(
            RobotBrainAgentConfig(
                allow_motion_commands=True,
                camera_pitch_action_key="head_tilt.pos",
                camera_pitch_settle_s=0.0,
            ),
            runtime=runtime,
        )

        response = agent.pitch_camera(pitch_rad=0.5)

        self.assertTrue(response["succeeded"])
        self.assertEqual(runtime.actions, [{"head_tilt.pos": 0.5 * 180.0 / 3.141592653589793}])
        self.assertAlmostEqual(agent.camera_state()["pitch_rad"], 0.5)

    def test_agent_applies_camera_pitch_motor_offset_without_changing_state(self) -> None:
        runtime = FakeRuntime()
        agent = RobotBrainAgent(
            RobotBrainAgentConfig(
                allow_motion_commands=True,
                camera_pitch_action_key="head_motor_2.pos",
                camera_pitch_action_offset_deg=-25.0,
                camera_pitch_settle_s=0.0,
            ),
            runtime=runtime,
        )

        response = agent.pitch_camera(pitch_rad=0.0)

        self.assertTrue(response["succeeded"])
        self.assertEqual(runtime.actions, [{"head_motor_2.pos": -25.0}])
        self.assertAlmostEqual(agent.camera_state()["pitch_rad"], 0.0)

    def test_agent_commands_camera_pan_and_updates_state(self) -> None:
        runtime = FakeRuntime()
        agent = RobotBrainAgent(
            RobotBrainAgentConfig(
                allow_motion_commands=True,
                camera_pan_action_key="head_motor_1.pos",
                camera_pan_settle_s=0.0,
            ),
            runtime=runtime,
        )

        response = agent.pan_camera(pan_rad=-0.5)

        self.assertTrue(response["succeeded"])
        self.assertEqual(runtime.actions, [{"head_motor_1.pos": -0.5 * 180.0 / 3.141592653589793}])
        self.assertAlmostEqual(agent.camera_state()["pan_rad"], -0.5)
        self.assertAlmostEqual(agent.camera_state()["pitch_rad"], 0.0)

    def test_agent_can_invert_camera_pan_motor_action_without_inverting_pose(self) -> None:
        runtime = FakeRuntime()
        agent = RobotBrainAgent(
            RobotBrainAgentConfig(
                allow_motion_commands=True,
                camera_pan_action_key="head_motor_1.pos",
                camera_pan_action_sign=-1.0,
                camera_pan_settle_s=0.0,
            ),
            runtime=runtime,
        )

        response = agent.pan_camera(pan_rad=0.5)

        self.assertTrue(response["succeeded"])
        self.assertEqual(runtime.actions, [{"head_motor_1.pos": -0.5 * 180.0 / 3.141592653589793}])
        self.assertAlmostEqual(agent.camera_state()["pan_rad"], 0.5)

    def test_agent_rejects_degree_pan_when_robot_is_not_in_degree_mode(self) -> None:
        runtime = FakeRuntime()
        agent = RobotBrainAgent(
            RobotBrainAgentConfig(
                use_degrees=False,
                allow_motion_commands=True,
                camera_pan_action_key="head_motor_1.pos",
                camera_pan_settle_s=0.0,
            ),
            runtime=runtime,
        )

        response = agent.pan_camera(pan_rad=1.0)

        self.assertFalse(response["succeeded"])
        self.assertIn("--use-degrees", response["message"])
        self.assertEqual(runtime.actions, [])

    def test_agent_can_send_normalized_pan_when_degree_mode_is_disabled(self) -> None:
        runtime = FakeRuntime()
        agent = RobotBrainAgent(
            RobotBrainAgentConfig(
                use_degrees=False,
                allow_motion_commands=True,
                camera_pan_action_key="head_motor_1.pos",
                camera_pan_action_units="normalized",
                camera_pan_settle_s=0.0,
            ),
            runtime=runtime,
        )

        response = agent.pan_camera(pan_rad=3.141592653589793)

        self.assertTrue(response["succeeded"])
        self.assertEqual(runtime.actions, [{"head_motor_1.pos": 100.0}])

    def test_agent_serves_expected_orbbec_file_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = RobotBrainAgent(
                RobotBrainAgentConfig(orbbec_output_dir=Path(tmpdir)),
                runtime=FakeRuntime(),
            )

            self.assertEqual(agent.file_path("/rgb"), Path(tmpdir) / "latest.ppm")
            self.assertEqual(agent.file_path("/depth"), Path(tmpdir) / "latest_depth.pgm")
            self.assertEqual(agent.file_path("/metadata"), Path(tmpdir) / "latest.json")
            self.assertIsNone(agent.file_path("/imu"))
            self.assertIsNone(agent.file_path("/missing"))

    def test_agent_keeps_latest_imu_in_memory(self) -> None:
        agent = RobotBrainAgent(RobotBrainAgentConfig(), runtime=FakeRuntime())

        agent.ingest_imu_datagram(
            b'{"timestamp_s":1.25,"angular_velocity_rad_s":{"x":0.1,"y":0.2,"z":0.3},"linear_acceleration_m_s2":{"x":1.0,"y":2.0,"z":3.0},"gyro_frame_index":7}'
        )

        snapshot = agent.imu_snapshot()
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot["angular_velocity_rad_s"]["z"], 0.3)
        self.assertEqual(snapshot["gyro_frame_index"], 7)
        stats = agent.imu_stream.stats()
        self.assertTrue(stats["ready"])
        self.assertEqual(stats["received_count"], 1)
        self.assertEqual(stats["latest_timestamp_s"], 1.25)
        self.assertIsNotNone(stats["age_s"])

    def test_agent_keeps_latest_rgbd_in_memory(self) -> None:
        agent = RobotBrainAgent(RobotBrainAgentConfig(), runtime=FakeRuntime())

        frame = agent.ingest_rgbd_payload(
            pack_rgbd_frame(
                frame_index=3,
                timestamp_us=2_500_000,
                rgb=b"abc",
                rgb_width=1,
                rgb_height=1,
                depth_be=(1234).to_bytes(2, "big"),
                depth_width=1,
                depth_height=1,
            )
        )

        self.assertEqual(frame.frame_index, 3)
        self.assertEqual(agent.rgbd_stream.rgb_ppm(), b"P6\n1 1\n255\nabc")
        self.assertEqual(agent.rgbd_stream.depth_pgm(), b"P5\n1 1\n65535\n" + (1234).to_bytes(2, "big"))
        self.assertIn(b'"frame_index": 3', agent.rgbd_stream.metadata_json())
        stats = agent.rgbd_stream.stats()
        self.assertTrue(stats["ready"])
        self.assertEqual(stats["received_count"], 1)
        self.assertEqual(stats["frame_index"], 3)
        self.assertEqual(stats["rgb"], {"width": 1, "height": 1})
        self.assertEqual(stats["depth"], {"width": 1, "height": 1})


if __name__ == "__main__":
    unittest.main()
