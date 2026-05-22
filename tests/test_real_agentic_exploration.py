from __future__ import annotations

import unittest
from unittest.mock import patch

from xlerobot_agent.exploration import ExplorationBackend, ExplorationBackendConfig
from xlerobot_playground.real_agentic_exploration import build_parser, translated_args
from xlerobot_playground.sim_exploration_backend import RosExplorationSession, SimExplorationConfig


class RealAgenticExplorationTests(unittest.TestCase):
    def test_defaults_translate_to_ros_nav2_real_exploration(self) -> None:
        args = build_parser().parse_args([])

        translated = translated_args(args)

        self.assertIn("--nav2-mode", translated)
        self.assertEqual(translated[translated.index("--nav2-mode") + 1], "ros")
        self.assertIn("--serve-review-ui", translated)
        self.assertIn("--wait-for-ui-start", translated)
        self.assertIn("--ros-navigation-map-source", translated)
        self.assertEqual(translated[translated.index("--ros-navigation-map-source") + 1], "fused_scan")
        self.assertEqual(translated[translated.index("--ros-imu-topic") + 1], "/imu/filtered_yaw")
        self.assertEqual(translated[translated.index("--source") + 1], "real_xlerobot")
        self.assertIn("--memory-root", translated)
        self.assertEqual(translated[translated.index("--memory-root") + 1], "./artifacts/memories")
        self.assertEqual(translated[translated.index("--ros-manual-spin-angular-speed-rad-s") + 1], "0.3")
        self.assertEqual(translated[translated.index("--ros-manual-spin-direction-sign") + 1], "1.0")
        self.assertEqual(translated[translated.index("--ros-turn-scan-mode") + 1], "camera_pan")
        self.assertEqual(translated[translated.index("--camera-pan-action-key") + 1], "head_motor_1.pos")
        self.assertEqual(translated[translated.index("--ros-scan-active-topic") + 1], "/xlerobot/scan_active")
        self.assertEqual(translated[translated.index("--ros-nav-active-topic") + 1], "/xlerobot/nav_active")
        self.assertEqual(translated[translated.index("--ros-local-rotation-active-topic") + 1], "/xlerobot/local_rotation_active")
        self.assertEqual(translated[translated.index("--ros-scan-active-release-delay-s") + 1], "3.0")
        self.assertIn("navigate_to_pose_replanning_no_recovery.xml", translated[translated.index("--nav2-behavior-tree") + 1])
        self.assertIn("--no-pause-for-operator-approval", translated)

    def test_explicit_llm_and_ui_options_are_preserved(self) -> None:
        args = build_parser().parse_args(
            [
                "--llm-provider",
                "openai",
                "--llm-model",
                "gpt-test",
                "--llm-api-key",
                "secret",
                "--no-serve-review-ui",
                "--memory-root",
                "/tmp/robot42-memories",
                "--review-host",
                "127.0.0.1",
                "--review-port",
                "8899",
            ]
        )

        translated = translated_args(args)

        self.assertEqual(translated[translated.index("--llm-provider") + 1], "openai")
        self.assertEqual(translated[translated.index("--llm-model") + 1], "gpt-test")
        self.assertEqual(translated[translated.index("--llm-api-key") + 1], "secret")
        self.assertEqual(translated[translated.index("--memory-root") + 1], "/tmp/robot42-memories")
        self.assertIn("--no-serve-review-ui", translated)
        self.assertEqual(translated[translated.index("--review-host") + 1], "127.0.0.1")
        self.assertEqual(translated[translated.index("--review-port") + 1], "8899")

    def test_pause_for_operator_approval_is_translated(self) -> None:
        args = build_parser().parse_args(["--pause-for-operator-approval"])

        translated = translated_args(args)

        self.assertIn("--pause-for-operator-approval", translated)

    def test_stop_after_initial_scan_is_translated(self) -> None:
        args = build_parser().parse_args(["--stop-after-initial-scan"])

        translated = translated_args(args)

        self.assertIn("--stop-after-initial-scan", translated)

    def test_fused_point_cloud_map_source_is_translated(self) -> None:
        args = build_parser().parse_args(
            [
                "--ros-navigation-map-source",
                "fused_point_cloud",
                "--ros-point-cloud-topic",
                "/camera/head/points",
                "--point-cloud-robot-clearance-height-m",
                "1.5",
            ]
        )

        translated = translated_args(args)

        self.assertEqual(translated[translated.index("--ros-navigation-map-source") + 1], "fused_point_cloud")
        self.assertEqual(translated[translated.index("--ros-point-cloud-topic") + 1], "/camera/head/points")
        self.assertEqual(translated[translated.index("--point-cloud-robot-clearance-height-m") + 1], "1.5")

    def test_ros_session_initializes_scan_fusion_state_before_first_scan(self) -> None:
        class FakeRuntime:
            latest_map = None

            def scan_observation_count(self) -> int:
                return 3

        config = SimExplorationConfig(
            repo_root=".",
            persist_path="/tmp/robot42-test-map.json",
            ros_adapter_url="http://127.0.0.1:8891",
        )
        backend = ExplorationBackend(ExplorationBackendConfig(mode="sim"))

        with patch(
            "xlerobot_playground.sim_exploration_backend.RemoteRosExplorationRuntime",
            return_value=FakeRuntime(),
        ):
            session = RosExplorationSession(config, backend, "task_1")

        self.assertEqual(session.scan_known_cells, {})
        self.assertEqual(session.scan_occupancy_evidence, {})
        self.assertEqual(session.scan_range_edge_cells, set())
        self.assertEqual(session.scan_map_resolution, config.occupancy_resolution)
        self.assertEqual(session.scan_observation_index, 3)

    def test_fused_point_cloud_rejects_remote_ros_adapter(self) -> None:
        config = SimExplorationConfig(
            repo_root=".",
            persist_path="/tmp/robot42-test-map.json",
            ros_adapter_url="http://127.0.0.1:8891",
            ros_navigation_map_source="fused_point_cloud",
        )
        backend = ExplorationBackend(ExplorationBackendConfig(mode="sim"))

        with self.assertRaisesRegex(RuntimeError, "requires the local ROS runtime"):
            RosExplorationSession(config, backend, "task_1")


if __name__ == "__main__":
    unittest.main()
