from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from xlerobot_agent.vla_policy import RIGHT_ARM_ACTION_NAMES
from xlerobot_agent.vla_worker import (
    VLAWorkerConfig,
    VLAWorkerStartError,
    VLAWorkerSupervisor,
)


class VLAWorkerTests(unittest.TestCase):
    def _config(self, root: Path, **overrides) -> VLAWorkerConfig:
        values = {
            "policy_path": root / "unused-policy",
            "dataset_repo_id": "test/dataset",
            "dataset_root": root / "unused-dataset",
            "task": "put the bottle in the basket",
            "backend": "mock",
            "startup_timeout_s": 3.0,
            "prediction_timeout_s": 3.0,
            "shutdown_timeout_s": 1.0,
            "log_path": root / "worker.log",
        }
        values.update(overrides)
        return VLAWorkerConfig(**values)

    def test_spawn_worker_reports_ready_and_predicts_without_hardware(self) -> None:
        with TemporaryDirectory() as tmp:
            supervisor = VLAWorkerSupervisor(self._config(Path(tmp), mock_chunk_size=4))
            self.assertEqual(supervisor.start_method, "spawn")
            try:
                supervisor.spawn()
                self.assertEqual(supervisor.state, "starting")
                ready = supervisor.wait_until_ready()
                parent_pid = __import__("os").getpid()
                self.assertNotEqual(ready.worker_pid, parent_pid)
                self.assertEqual(ready.action_names, RIGHT_ARM_ACTION_NAMES)
                self.assertEqual(ready.required_image_keys, ())
                prediction = supervisor.predict(
                    {"observation.state": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
                )
                self.assertEqual(len(prediction.actions), 4)
                self.assertEqual(prediction.actions[0][RIGHT_ARM_ACTION_NAMES[-1]], 6.0)
                self.assertEqual(supervisor.state, "ready")
            finally:
                supervisor.stop()
            self.assertEqual(supervisor.state, "stopped")

    def test_worker_load_failure_is_structured_and_process_is_stopped(self) -> None:
        with TemporaryDirectory() as tmp:
            supervisor = VLAWorkerSupervisor(self._config(Path(tmp), mock_fail_load=True))
            try:
                with self.assertRaisesRegex(VLAWorkerStartError, "mock VLA load failure"):
                    supervisor.ensure_ready()
                self.assertEqual(supervisor.state, "failed")
                self.assertIsNone(supervisor.worker_pid)
            finally:
                supervisor.stop()

    def test_startup_timeout_terminates_worker(self) -> None:
        with TemporaryDirectory() as tmp:
            supervisor = VLAWorkerSupervisor(
                self._config(
                    Path(tmp),
                    mock_load_delay_s=1.0,
                    startup_timeout_s=0.05,
                )
            )
            try:
                with self.assertRaisesRegex(VLAWorkerStartError, "timed out"):
                    supervisor.ensure_ready()
                self.assertEqual(supervisor.state, "failed")
                self.assertIsNone(supervisor.worker_pid)
            finally:
                supervisor.stop()


if __name__ == "__main__":
    unittest.main()
