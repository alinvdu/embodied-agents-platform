import time
import unittest
from collections import deque
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from scripts.run_xlerobot_act_policy import build_parser as build_act_parser
from scripts.test_xlerobot_action_ready_pose import _action_ready_pose
from scripts.run_xlerobot_smolvla_policy import (
    RIGHT_ARM_ACTION_NAMES,
    _AsyncChunkPredictor,
    _camera_rename_map,
    _clamp_action,
    _configure_act_action_steps,
    _control_scope,
    _controlled_arm_sides,
    _merge_action_chunk,
    _policy_action_queue_depth,
    _policy_camera_rename_map,
    _run_startup_pose,
    _send_stop,
    _validate_policy_camera_contract,
    _validate_policy_dataset_contract,
    _validate_policy_type,
)


class _FakeRobot:
    def __init__(self) -> None:
        self.actions = []
        self.observation = {
            f"{side}_arm_{joint}.pos": 0.0
            for side in ("left", "right")
            for joint in (
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
                "gripper",
            )
        }
        self.observation["head_motor_1.pos"] = 15.0

    def get_observation(self):
        return dict(self.observation)

    def send_action(self, action):
        self.actions.append(dict(action))


class SmolVLAInferenceRunnerTests(unittest.TestCase):
    def test_new_action_ready_pose_is_ten_units_farther_back(self) -> None:
        targets = _action_ready_pose(
            ("right",),
            elbow_delta=-80.0,
            shoulder_delta=70.0,
            wrist_delta=-40.0,
        )

        self.assertAlmostEqual(targets["right_arm_shoulder_lift.pos"], -29.1708)
        self.assertAlmostEqual(targets["right_arm_elbow_flex.pos"], 20.0)
        self.assertAlmostEqual(targets["right_arm_wrist_flex.pos"], 36.2061)

    def test_right_arm_contract_and_scope(self) -> None:
        features = {
            "action": {"names": list(RIGHT_ARM_ACTION_NAMES)},
            "observation.state": {"names": list(RIGHT_ARM_ACTION_NAMES)},
        }
        cfg = SimpleNamespace(
            output_features={"action": SimpleNamespace(shape=(6,))},
            input_features={"observation.state": SimpleNamespace(shape=(6,))},
        )

        _validate_policy_dataset_contract(cfg, features)

        self.assertEqual(_controlled_arm_sides(list(RIGHT_ARM_ACTION_NAMES)), ("right",))
        self.assertEqual(
            _control_scope(list(RIGHT_ARM_ACTION_NAMES)),
            "right arm only (no left arm, head motors, or base)",
        )

    def test_right_arm_contract_rejects_full_body_dataset(self) -> None:
        features = {
            "action": {"names": [*RIGHT_ARM_ACTION_NAMES, "x.vel", "theta.vel"]},
            "observation.state": {"names": [*RIGHT_ARM_ACTION_NAMES, "x.vel", "theta.vel"]},
        }
        cfg = SimpleNamespace(
            output_features={"action": SimpleNamespace(shape=(6,))},
            input_features={"observation.state": SimpleNamespace(shape=(6,))},
        )

        with self.assertRaisesRegex(RuntimeError, "expects 6 actions"):
            _validate_policy_dataset_contract(cfg, features)

    def test_startup_pose_moves_both_arms_for_right_only_policy(self) -> None:
        robot = _FakeRobot()

        _run_startup_pose(
            robot=robot,
            stow_wait_s=0.0,
            action_ready_elbow_delta=-80.0,
            action_ready_shoulder_delta=80.0,
            action_ready_wrist_delta=-40.0,
            steps_per_stage=1,
            stage_delay_s=0.0,
        )

        self.assertTrue(robot.actions)
        self.assertEqual(robot.actions[0], {"head_motor_1.pos": 0.0})
        moved_sides = {
            key.split("_", maxsplit=1)[0]
            for action in robot.actions
            for key in action
            if key.startswith(("left_arm_", "right_arm_"))
        }
        self.assertEqual(moved_sides, {"left", "right"})
        self.assertEqual(
            set(robot.actions[-1]),
            {"left_arm_shoulder_lift.pos", "right_arm_shoulder_lift.pos"},
        )

    def test_stop_does_not_send_base_action_for_right_arm_policy(self) -> None:
        robot = _FakeRobot()

        _send_stop(robot, list(RIGHT_ARM_ACTION_NAMES))

        self.assertEqual(robot.actions, [])

    def test_stop_only_sends_base_outputs_declared_by_policy(self) -> None:
        robot = _FakeRobot()

        _send_stop(robot, [*RIGHT_ARM_ACTION_NAMES, "x.vel"])

        self.assertEqual(robot.actions, [{"x.vel": 0.0}])

    def test_camera_rename_map_compacts_missing_camera_slots(self) -> None:
        features = {
            "observation.images.head": {"dtype": "video"},
            "observation.images.right_wrist": {"dtype": "video"},
            "observation.state": {"dtype": "float32"},
        }

        self.assertEqual(
            _camera_rename_map(features),
            {
                "observation.images.head": "observation.images.camera1",
                "observation.images.right_wrist": "observation.images.camera2",
            },
        )

    def test_act_saved_empty_camera_rename_map_is_preserved(self) -> None:
        features = {
            "observation.images.head": {"dtype": "video"},
            "observation.images.right_wrist": {"dtype": "video"},
        }
        with TemporaryDirectory() as tmp_dir:
            policy_path = Path(tmp_dir)
            (policy_path / "policy_preprocessor.json").write_text(
                """
                {
                  "steps": [
                    {
                      "registry_name": "rename_observations_processor",
                      "config": {"rename_map": {}}
                    }
                  ]
                }
                """
            )

            camera_rename = _policy_camera_rename_map(
                policy_path,
                features,
                policy_input_features={
                    "observation.images.head": object(),
                    "observation.images.right_wrist": object(),
                },
            )

        self.assertEqual(camera_rename, {})
        _validate_policy_camera_contract(
            SimpleNamespace(
                input_features={
                    "observation.images.head": object(),
                    "observation.images.right_wrist": object(),
                }
            ),
            features,
            camera_rename,
        )

    def test_smolvla_allows_unused_declared_camera_slot(self) -> None:
        features = {
            "observation.images.head": {"dtype": "video"},
            "observation.images.right_wrist": {"dtype": "video"},
        }
        camera_rename = {
            "observation.images.head": "observation.images.camera1",
            "observation.images.right_wrist": "observation.images.camera2",
        }

        _validate_policy_camera_contract(
            SimpleNamespace(
                type="smolvla",
                input_features={
                    "observation.images.camera1": object(),
                    "observation.images.camera2": object(),
                    "observation.images.camera3": object(),
                },
            ),
            features,
            camera_rename,
        )

    def test_act_rejects_unused_declared_camera_slot(self) -> None:
        features = {
            "observation.images.head": {"dtype": "video"},
            "observation.images.right_wrist": {"dtype": "video"},
        }
        camera_rename = {
            "observation.images.head": "observation.images.camera1",
            "observation.images.right_wrist": "observation.images.camera2",
        }

        with self.assertRaisesRegex(RuntimeError, "camera inputs do not match"):
            _validate_policy_camera_contract(
                SimpleNamespace(
                    type="act",
                    input_features={
                        "observation.images.camera1": object(),
                        "observation.images.camera2": object(),
                        "observation.images.camera3": object(),
                    },
                ),
                features,
                camera_rename,
            )

    def test_act_runner_defaults_to_native_policy_queue(self) -> None:
        args = build_act_parser().parse_args([])

        self.assertTrue(args.policy_path.endswith("outputs/train/pretrained_act"))
        self.assertFalse(args.async_inference)
        self.assertEqual(args.policy_warmup_runs, 1)
        self.assertEqual(args.act_action_steps, 25)
        self.assertEqual(args.duration_s, 20.0)
        self.assertEqual(args.max_joint_delta, 100.0)
        self.assertEqual(args.max_gripper_delta, 100.0)
        self.assertEqual(args.startup_action_ready_shoulder_delta, 55.0)
        self.assertTrue(args.startup_pose)

    def test_act_action_horizon_override_resets_policy_queue(self) -> None:
        reset_calls = []
        policy = SimpleNamespace(
            config=SimpleNamespace(type="act", chunk_size=50, n_action_steps=10),
            reset=lambda: reset_calls.append(True),
        )

        _configure_act_action_steps(policy, 25)

        self.assertEqual(policy.config.n_action_steps, 25)
        self.assertEqual(reset_calls, [True])

    def test_policy_type_validation_rejects_wrong_runner(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "requires an? 'act' checkpoint"):
            _validate_policy_type(SimpleNamespace(type="smolvla"), "act")

    def test_action_queue_depth_supports_act(self) -> None:
        policy = SimpleNamespace(_action_queue=deque([1, 2, 3]))

        self.assertEqual(_policy_action_queue_depth(policy, "action"), 3)

    def test_clamp_uses_separate_gripper_delta(self) -> None:
        action = {
            "right_arm_shoulder_pan.pos": 20.0,
            "right_arm_gripper.pos": 0.0,
            "x.vel": 1.0,
        }
        observation = {
            "right_arm_shoulder_pan.pos": 0.0,
            "right_arm_gripper.pos": 40.0,
        }

        clamped = _clamp_action(
            action,
            observation,
            max_joint_delta=4.0,
            max_gripper_delta=12.0,
            max_base_x_vel=0.0,
            max_base_theta_vel=0.0,
        )

        self.assertEqual(clamped["right_arm_shoulder_pan.pos"], 4.0)
        self.assertEqual(clamped["right_arm_gripper.pos"], 28.0)
        self.assertEqual(clamped["x.vel"], 0.0)

    def test_merge_drops_stale_actions_and_blends_overlap(self) -> None:
        existing = {
            2: {"joint.pos": 10.0},
            3: {"joint.pos": 10.0},
        }
        actions = [
            {"joint.pos": 1.0},
            {"joint.pos": 20.0},
            {"joint.pos": 30.0},
        ]

        merged, stale = _merge_action_chunk(
            existing,
            request_step=1,
            actions=actions,
            current_step=2,
            new_action_weight=0.7,
        )

        self.assertEqual(stale, 1)
        self.assertAlmostEqual(merged[2]["joint.pos"], 17.0)
        self.assertAlmostEqual(merged[3]["joint.pos"], 24.0)

    def test_async_predictor_copies_observation_and_returns_result(self) -> None:
        def predict(observation):
            time.sleep(0.01)
            return [{"joint.pos": observation["state"][0]}]

        predictor = _AsyncChunkPredictor(predict)
        state = [3.0]
        try:
            self.assertTrue(predictor.submit(7, {"state": state}))
            state[0] = 99.0
            deadline = time.monotonic() + 1.0
            result = None
            while result is None and time.monotonic() < deadline:
                result = predictor.poll()
                time.sleep(0.005)
        finally:
            predictor.close()

        self.assertIsNotNone(result)
        self.assertIsNone(result.error)
        self.assertEqual(result.request_step, 7)
        self.assertEqual(result.actions, [{"joint.pos": 3.0}])


if __name__ == "__main__":
    unittest.main()
