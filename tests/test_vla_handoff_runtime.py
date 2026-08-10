from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import threading
import unittest

import numpy as np

from xlerobot_agent.vla_policy import RIGHT_ARM_ACTION_NAMES
from xlerobot_agent.vla_worker import VLAWorkerPrediction, VLAWorkerReady
from xlerobot_playground.vla_handoff_runtime import VLAHandoffConfig, VLAHandoffRuntime


class _FakeRobot:
    def __init__(self) -> None:
        self.actions: list[dict[str, float]] = []
        self.observation = {
            **{name: float(index) for index, name in enumerate(RIGHT_ARM_ACTION_NAMES)},
            "right_wrist": np.full((2, 3, 3), 7, dtype=np.uint8),
        }

    def get_observation(self, *, use_camera: bool = True):
        observation = dict(self.observation)
        if not use_camera:
            observation.pop("right_wrist", None)
        return observation

    def send_action(self, action):
        copied = {key: float(value) for key, value in action.items()}
        self.actions.append(copied)
        self.observation.update(copied)
        return copied


class _FakeRobotRuntime:
    def __init__(self) -> None:
        self.robot = _FakeRobot()
        self.connect_calls = 0

    def connect(self) -> None:
        self.connect_calls += 1


class _FakeWorker:
    def __init__(self, config) -> None:
        self.config = config
        self.spawned = False
        self.stopped = False
        self.observations = []

    def spawn(self) -> None:
        self.spawned = True

    def wait_until_ready(self, timeout_s=None):
        return VLAWorkerReady(
            worker_pid=42,
            policy_type="smolvla",
            action_names=RIGHT_ARM_ACTION_NAMES,
            required_image_keys=("observation.images.right_wrist", "observation.images.head"),
            chunk_size=2,
            n_action_steps=2,
            load_duration_s=0.1,
        )

    def reset_policy(self) -> None:
        return

    def predict(self, observation):
        self.observations.append(observation)
        actions = tuple(
            {name: float(index + offset) for index, name in enumerate(RIGHT_ARM_ACTION_NAMES)}
            for offset in (1.0, 2.0)
        )
        return VLAWorkerPrediction("request", actions, 0.01)

    def stop(self) -> None:
        self.stopped = True

    def status_snapshot(self):
        return {"state": "ready" if self.spawned and not self.stopped else "stopped"}


class _ReleaseWorker(_FakeWorker):
    def wait_until_ready(self, timeout_s=None):
        return VLAWorkerReady(
            worker_pid=42,
            policy_type="smolvla",
            action_names=RIGHT_ARM_ACTION_NAMES,
            required_image_keys=("observation.images.right_wrist", "observation.images.head"),
            chunk_size=9,
            n_action_steps=9,
            load_duration_s=0.1,
        )

    def predict(self, observation):
        self.observations.append(observation)
        gripper_values = (35.0, 35.0, 35.0, 5.0, 5.0, 5.0, 40.0, 40.0, 40.0)
        actions = []
        for gripper in gripper_values:
            action = {name: float(index) for index, name in enumerate(RIGHT_ARM_ACTION_NAMES)}
            action["right_arm_gripper.pos"] = gripper
            actions.append(action)
        return VLAWorkerPrediction("release", tuple(actions), 0.01)


class VLAHandoffRuntimeTests(unittest.TestCase):
    def test_worker_is_created_only_when_handoff_runs(self) -> None:
        robot_runtime = _FakeRobotRuntime()
        workers = []

        def factory(config):
            worker = _FakeWorker(config)
            workers.append(worker)
            return worker

        runtime = VLAHandoffRuntime(
            config=VLAHandoffConfig(
                policy_path=Path("policy"),
                dataset_repo_id="owner/dataset",
                dataset_root=Path("dataset"),
                task="put bottle in basket",
                duration_s=0.004,
                fps=1000.0,
                action_steps=2,
                startup_pose=False,
            ),
            robot_runtime=robot_runtime,
            motion_lock=threading.Lock(),
            head_frame_provider=lambda: SimpleNamespace(
                rgb=bytes(range(18)),
                rgb_width=3,
                rgb_height=2,
            ),
            worker_factory=factory,
        )

        self.assertEqual(workers, [])
        result = runtime.run()

        self.assertEqual(result["status"], "timed_out")
        self.assertGreater(result["actions_sent"], 0)
        self.assertEqual(len(workers), 1)
        self.assertTrue(workers[0].spawned)
        self.assertTrue(workers[0].stopped)
        self.assertFalse(runtime.active)
        self.assertFalse(runtime.status()["awaiting_verification"])
        self.assertEqual(runtime.status()["phase"], "idle")
        observation = workers[0].observations[0]
        self.assertEqual(observation["observation.state"].shape, (6,))
        self.assertEqual(observation["observation.images.right_wrist"].shape, (2, 3, 3))
        self.assertEqual(observation["observation.images.head"].shape, (2, 3, 3))
        policy_actions = [
            action
            for action in robot_runtime.robot.actions
            if not set(action).issubset({"x.vel", "theta.vel"})
        ]
        self.assertTrue(policy_actions)
        self.assertTrue(all(set(action).issubset(RIGHT_ARM_ACTION_NAMES) for action in policy_actions))

    def test_second_open_stops_actions_and_returns_wrist_evidence(self) -> None:
        robot_runtime = _FakeRobotRuntime()
        workers = []

        def factory(config):
            worker = _ReleaseWorker(config)
            workers.append(worker)
            return worker

        runtime = VLAHandoffRuntime(
            config=VLAHandoffConfig(
                policy_path=Path("policy"),
                dataset_repo_id="owner/dataset",
                dataset_root=Path("dataset"),
                task="put bottle in basket",
                duration_s=1.0,
                fps=1000.0,
                action_steps=9,
                startup_pose=False,
                release_transition_samples=2,
                release_observed_open_samples=1,
                release_settle_s=0.0,
                release_capture_count=2,
                release_capture_interval_s=0.0,
            ),
            robot_runtime=robot_runtime,
            motion_lock=threading.Lock(),
            head_frame_provider=lambda: SimpleNamespace(
                rgb=bytes(range(18)),
                rgb_width=3,
                rgb_height=2,
            ),
            worker_factory=factory,
        )

        result = runtime.run()

        self.assertEqual(result["status"], "release_detected")
        self.assertEqual(result["actions_sent"], 8)
        self.assertEqual(result["release_detection"]["action_index"], 8)
        self.assertEqual(result["release_wrist_image_count"], 2)
        self.assertTrue(all(image.startswith("data:image/jpeg;base64,") for image in result["release_wrist_images"]))
        self.assertNotIn("release_wrist_images", runtime.status()["last_result"])
        self.assertTrue(workers[0].stopped)
        self.assertTrue(runtime.active)
        self.assertTrue(runtime.status()["awaiting_verification"])
        self.assertEqual(runtime.status()["phase"], "awaiting_verification")

        stow = runtime.stow()
        self.assertEqual(stow["status"], "succeeded")
        self.assertAlmostEqual(robot_runtime.robot.observation["right_arm_gripper.pos"], 0.9466)
        self.assertFalse(runtime.active)
        self.assertFalse(runtime.status()["awaiting_verification"])
        self.assertEqual(runtime.status()["phase"], "idle")

    def test_stow_requires_release_awaiting_verification(self) -> None:
        runtime = VLAHandoffRuntime(
            config=VLAHandoffConfig(
                policy_path=Path("policy"),
                dataset_repo_id="owner/dataset",
                dataset_root=Path("dataset"),
                task="put bottle in basket",
                startup_pose=False,
            ),
            robot_runtime=_FakeRobotRuntime(),
            motion_lock=threading.Lock(),
            head_frame_provider=lambda: None,
        )

        result = runtime.stow()

        self.assertEqual(result["status"], "blocked")
        self.assertIn("release awaiting", result["reason"])

    def test_missing_head_camera_fails_and_stops_worker(self) -> None:
        robot_runtime = _FakeRobotRuntime()
        workers = []

        def factory(config):
            worker = _FakeWorker(config)
            workers.append(worker)
            return worker

        runtime = VLAHandoffRuntime(
            config=VLAHandoffConfig(
                policy_path=Path("policy"),
                dataset_repo_id="owner/dataset",
                dataset_root=Path("dataset"),
                task="task",
                duration_s=0.01,
                startup_pose=False,
                camera_ready_timeout_s=0.001,
            ),
            robot_runtime=robot_runtime,
            motion_lock=threading.Lock(),
            head_frame_provider=lambda: None,
            worker_factory=factory,
        )

        result = runtime.run()

        self.assertEqual(result["status"], "failed")
        self.assertIn("Orbbec head RGB frame", result["reason"])
        self.assertEqual(result["actions_sent"], 0)
        self.assertTrue(workers[0].stopped)


if __name__ == "__main__":
    unittest.main()
