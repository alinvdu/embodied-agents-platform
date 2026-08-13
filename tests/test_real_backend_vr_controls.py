import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from xlerobot_playground.real_backend import (
    _ACTION_READY_ELBOW_DELTA,
    _ACTION_READY_SHOULDER_DELTA,
    RecordingSession,
    VRRecordingControls,
    VrVideoDisplayConfig,
    build_parser,
    _create_dataset,
    _decide_vr_recording_action,
    _finalize_recording,
    _map_vr_events_to_recording_controls,
    _orbbec_vr_overlay_js,
    _save_episode,
)


class _FakeDataset:
    def __init__(self, *, buffered_frames: int) -> None:
        self.episode_buffer = {"size": buffered_frames}
        self.meta = SimpleNamespace(total_episodes=12)
        self.finalized = False

    def save_episode(self) -> None:
        self.meta.total_episodes += 1
        self.episode_buffer["size"] = 0

    def clear_episode_buffer(self) -> None:
        self.episode_buffer["size"] = 0

    def finalize(self) -> None:
        self.finalized = True


class RealBackendVRControlTests(unittest.TestCase):
    def test_resume_dataset_root_is_resolved_before_vr_changes_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            previous_cwd = Path.cwd()
            os.chdir(temp_dir)
            try:
                relative_root = Path("datasets/test_dataset")
                (relative_root / "meta").mkdir(parents=True)
                (relative_root / "meta/info.json").touch()
                expected_root = relative_root.resolve()
                marker = object()
                with patch(
                    "lerobot.datasets.lerobot_dataset.LeRobotDataset.resume",
                    return_value=marker,
                ) as resume:
                    result = _create_dataset(
                        SimpleNamespace(),
                        dataset_id="local/test_dataset",
                        dataset_root=str(relative_root),
                        fps=30,
                        use_videos=True,
                        resume=True,
                    )
            finally:
                os.chdir(previous_cwd)

        self.assertIs(result, marker)
        resume.assert_called_once_with("local/test_dataset", root=expected_root)

    def test_action_ready_default_is_farther_back(self) -> None:
        args = build_parser().parse_args(["manipulate"])

        self.assertEqual(_ACTION_READY_ELBOW_DELTA, -65.0)
        self.assertEqual(_ACTION_READY_SHOULDER_DELTA, 55.0)
        self.assertEqual(args.vr_action_ready_elbow_delta, -65.0)
        self.assertEqual(args.vr_action_ready_shoulder_delta, 55.0)

    def test_legacy_exit_early_event_does_not_start_recording(self) -> None:
        controls = _map_vr_events_to_recording_controls({"exit_early": True})
        decision = _decide_vr_recording_action(False, controls)
        self.assertEqual(decision.start_recording, False)
        self.assertEqual(decision.save_episode, False)

    def test_legacy_exit_early_event_does_not_save_episode(self) -> None:
        controls = _map_vr_events_to_recording_controls({"exit_early": True})
        decision = _decide_vr_recording_action(True, controls)
        self.assertEqual(decision.start_recording, False)
        self.assertEqual(decision.save_episode, False)

    def test_legacy_rerecord_event_does_not_discard_episode(self) -> None:
        controls = _map_vr_events_to_recording_controls(
            {
                "exit_early": True,
                "rerecord_episode": True,
            }
        )
        decision = _decide_vr_recording_action(True, controls)
        self.assertEqual(decision.save_episode, False)
        self.assertEqual(decision.discard_episode, False)

    def test_legacy_stop_event_does_not_finish_session(self) -> None:
        controls = _map_vr_events_to_recording_controls(
            {
                "exit_early": True,
                "stop_recording": True,
            }
        )
        decision = _decide_vr_recording_action(True, controls)
        self.assertEqual(decision.save_episode, False)
        self.assertEqual(decision.quit_session, False)

    def test_reset_position_maps_to_reset_robot(self) -> None:
        controls = _map_vr_events_to_recording_controls({"reset_position": True})
        self.assertEqual(
            controls,
            VRRecordingControls(reset_robot=True),
        )

    def test_successful_save_increments_session_episode_count(self) -> None:
        recording = RecordingSession(
            dataset=_FakeDataset(buffered_frames=30),
            task="test",
            active=True,
        )

        saved = _save_episode(recording)

        self.assertTrue(saved)
        self.assertEqual(recording.session_episode_count, 1)
        self.assertFalse(recording.active)

    def test_empty_save_does_not_increment_session_episode_count(self) -> None:
        recording = RecordingSession(
            dataset=_FakeDataset(buffered_frames=0),
            task="test",
            active=True,
        )

        saved = _save_episode(recording)

        self.assertFalse(saved)
        self.assertEqual(recording.session_episode_count, 0)
        self.assertFalse(recording.active)

    def test_explicit_save_does_not_depend_on_action_phase(self) -> None:
        dataset = _FakeDataset(buffered_frames=30)
        recording = RecordingSession(
            dataset=dataset,
            task="test",
            active=True,
        )

        saved = _save_episode(recording)

        self.assertTrue(saved)
        self.assertFalse(recording.active)
        self.assertEqual(recording.session_episode_count, 1)
        self.assertEqual(dataset.meta.total_episodes, 13)
        self.assertEqual(dataset.episode_buffer["size"], 0)

    def test_finalize_discards_active_unsaved_episode(self) -> None:
        dataset = _FakeDataset(buffered_frames=30)
        recording = RecordingSession(
            dataset=dataset,
            task="test",
            active=True,
        )

        _finalize_recording(recording)

        self.assertFalse(recording.active)
        self.assertEqual(recording.session_episode_count, 0)
        self.assertEqual(dataset.meta.total_episodes, 12)
        self.assertEqual(dataset.episode_buffer["size"], 0)
        self.assertTrue(dataset.finalized)

    def test_vr_overlay_displays_session_episode_count(self) -> None:
        script = _orbbec_vr_overlay_js(
            include_orbbec=True,
            video_display=VrVideoDisplayConfig(
                wrist_gain=1.0,
                wrist_gamma=1.0,
                wrist_bias=0.0,
                orbbec_gain=1.0,
                orbbec_gamma=1.0,
                orbbec_bias=0.0,
            ),
            video_streams_enabled=True,
        )

        self.assertIn("session_episode_count", script)
        self.assertIn("Episodes saved this session", script)


if __name__ == "__main__":
    unittest.main()
