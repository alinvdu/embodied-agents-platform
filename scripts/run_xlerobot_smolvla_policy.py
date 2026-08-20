#!/usr/bin/env python
# Copyright 2026 Alin Vasile Dumitru
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import json
import math
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multido_xlerobot import XLeRobotInterface
from multido_xlerobot.bootstrap import bootstrap_xlerobot, resolve_xlerobot_repo_root
from xlerobot_agent.vla_policy import (
    RIGHT_ARM_ACTION_NAMES,
    camera_rename_map as _camera_rename_map,
    load_policy_stack as _load_policy_stack,
    policy_camera_rename_map as _policy_camera_rename_map,
    predict_action_chunk as _predict_action_chunk,
    unrename_action as _unrename_action,
    validate_policy_camera_contract as _validate_policy_camera_contract,
    validate_policy_dataset_contract as _validate_policy_dataset_contract,
    validate_policy_type as _validate_policy_type,
)
from xlerobot_playground.real_backend import (
    _ACTION_READY_ELBOW_DELTA,
    _ACTION_READY_SHOULDER_DELTA,
    _ACTION_READY_WRIST_DELTA,
    OrbbecRgbConfig,
    _NAV_STOW_ARM_POSE,
    _augment_recording_observation,
    _build_camera_configs,
    _connect_robot,
    _get_robot_observation,
    _get_robot_observation_best_effort,
    _start_orbbec_rgb_sidecar,
    _stop_orbbec_rgb_sidecar,
    RecordingSession,
)


DEFAULT_POLICY_PATH = REPO_ROOT / "outputs/train/pretrained_model_100k_right_only"
DEFAULT_DATASET_REPO_ID = "alindumitru/robot42_grab_to_basket_right_arm_v0"
DEFAULT_DATASET_ROOT = REPO_ROOT / "datasets/robot42_grab_to_basket_right_arm_v0"
DEFAULT_TASK = "Grab the Tabasco sauce bottle and put it in the robot basket."
BASE_ACTION_NAMES = ("x.vel", "theta.vel")


def build_parser(
    *,
    policy_label: str = "SmolVLA",
    default_policy_path: Path = DEFAULT_POLICY_PATH,
    default_duration_s: float = 12.0,
    default_async_inference: bool = True,
    default_policy_warmup_runs: int = 0,
    default_act_action_steps: int | None = None,
    default_max_joint_delta: float = 4.0,
    default_max_gripper_delta: float = 12.0,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            f"Run a trained {policy_label} grab-to-basket policy on the real XLeRobot. "
            "Start the robot in the same ACTION_READY pose used for data collection."
        )
    )
    parser.add_argument("--policy-path", default=str(default_policy_path))
    parser.add_argument("--dataset-repo-id", default=DEFAULT_DATASET_REPO_ID)
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--repo-root", default=str(resolve_xlerobot_repo_root()))
    parser.add_argument("--robot-kind", choices=("xlerobot", "xlerobot_2wheels"), default="xlerobot_2wheels")
    parser.add_argument("--port1", default="/dev/ttyACM0")
    parser.add_argument("--port2", default="/dev/ttyACM1")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--duration-s", type=float, default=default_duration_s)
    parser.add_argument("--warmup-s", type=float, default=3.0)
    parser.add_argument(
        "--policy-warmup-runs",
        type=int,
        default=default_policy_warmup_runs,
        help="Run this many throwaway model predictions before the timed rollout.",
    )
    parser.add_argument(
        "--act-action-steps",
        type=int,
        default=default_act_action_steps,
        help="Number of actions from each ACT chunk to execute before observing and replanning.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-degrees", action="store_true", default=True)
    parser.add_argument("--no-use-degrees", action="store_false", dest="use_degrees")
    parser.add_argument("--manual-calibration-prompt", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print predicted actions without sending them.")
    parser.add_argument(
        "--display-data",
        action="store_true",
        help="Show live policy observations and actions in LeRobot's Rerun viewer.",
    )
    parser.add_argument(
        "--display-compressed-images",
        action="store_true",
        help="JPEG-compress camera frames before sending them to Rerun.",
    )
    parser.add_argument(
        "--startup-pose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the NAV_STOW -> ACTION_READY startup motion before policy rollout.",
    )
    parser.add_argument(
        "--startup-pose-in-dry-run",
        action="store_true",
        help="Also run the physical startup motion when --dry-run is set.",
    )
    parser.add_argument(
        "--startup-head-pan-deg",
        type=float,
        default=0.0,
        help="Absolute head pan angle set before the arm startup motion.",
    )
    parser.add_argument("--startup-nav-stow-wait-s", type=float, default=5.0)
    parser.add_argument(
        "--startup-action-ready-elbow-delta",
        type=float,
        default=_ACTION_READY_ELBOW_DELTA,
    )
    parser.add_argument(
        "--startup-action-ready-shoulder-delta",
        type=float,
        default=_ACTION_READY_SHOULDER_DELTA,
    )
    parser.add_argument(
        "--startup-action-ready-wrist-delta",
        type=float,
        default=_ACTION_READY_WRIST_DELTA,
    )
    parser.add_argument("--startup-pose-steps", type=int, default=40)
    parser.add_argument("--startup-pose-stage-delay-s", type=float, default=0.02)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument(
        "--async-inference",
        action=argparse.BooleanOptionalAction,
        default=default_async_inference,
        help="Predict action chunks in a background thread so the robot loop can keep its target FPS.",
    )
    parser.add_argument(
        "--async-chunk-threshold",
        type=float,
        default=0.6,
        help="Request another chunk when the queued fraction falls to this value (0 to 1).",
    )
    parser.add_argument(
        "--async-new-action-weight",
        type=float,
        default=0.7,
        help="Weight assigned to a new prediction when overlapping chunks are merged (0 to 1).",
    )
    parser.add_argument(
        "--max-joint-delta",
        type=float,
        default=default_max_joint_delta,
        help="Clamp non-gripper position jumps from the current observation, in joint units.",
    )
    parser.add_argument(
        "--max-gripper-delta",
        type=float,
        default=default_max_gripper_delta,
        help="Clamp gripper position jumps separately so grasp commands are not limited by arm safety.",
    )
    parser.add_argument("--max-base-x-vel", type=float, default=0.0)
    parser.add_argument("--max-base-theta-vel", type=float, default=0.0)
    parser.add_argument(
        "--camera",
        action="append",
        default=[],
        metavar="NAME=DRIVER:SOURCE",
        help="Wrist camera config, same syntax as real_backend.py. Example: left_wrist=opencv:0",
    )
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--orbbec-capture-bin", default=None)
    parser.add_argument("--orbbec-no-launch", action="store_true")
    parser.add_argument("--orbbec-output-dir", default="artifacts/orbbec_rgb")
    parser.add_argument("--orbbec-width", type=int, default=640)
    parser.add_argument("--orbbec-height", type=int, default=480)
    parser.add_argument("--orbbec-fps", type=int, default=30)
    parser.add_argument("--orbbec-timeout-ms", type=int, default=1000)
    parser.add_argument(
        "--camera-ready-timeout-s",
        type=float,
        default=10.0,
        help="Seconds to wait for all dataset-required camera observations before rollout.",
    )
    parser.add_argument(
        "--missing-camera-grace-s",
        type=float,
        default=2.0,
        help="Abort rollout if required camera observations are missing for this many seconds.",
    )
    parser.add_argument(
        "--camera-cache-max-age-s",
        type=float,
        default=1.0,
        help="Reuse the latest good required camera frame for this many seconds if a frame is transiently missing.",
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    expected_policy_type: str = "smolvla",
    policy_label: str = "SmolVLA",
    default_policy_path: Path = DEFAULT_POLICY_PATH,
    default_duration_s: float = 12.0,
    default_async_inference: bool = True,
    default_policy_warmup_runs: int = 0,
    default_act_action_steps: int | None = None,
    default_max_joint_delta: float = 4.0,
    default_max_gripper_delta: float = 12.0,
    rerun_session_name: str = "smolvla_inference",
) -> int:
    args = build_parser(
        policy_label=policy_label,
        default_policy_path=default_policy_path,
        default_duration_s=default_duration_s,
        default_async_inference=default_async_inference,
        default_policy_warmup_runs=default_policy_warmup_runs,
        default_act_action_steps=default_act_action_steps,
        default_max_joint_delta=default_max_joint_delta,
        default_max_gripper_delta=default_max_gripper_delta,
    ).parse_args(argv)
    policy_path = Path(args.policy_path).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy path does not exist: {policy_path}")

    print(f"Policy: {policy_path}")
    print(f"Task: {args.task}")
    print(f"Dataset root: {dataset_root}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'SEND ACTIONS'}")

    bootstrap_xlerobot(args.repo_root)
    interface = XLeRobotInterface(args.repo_root)
    if args.robot_kind == "xlerobot_2wheels":
        config_cls, robot_cls = interface.robot_2wheels_classes()
    else:
        config_cls, robot_cls = interface.robot_classes()
    robot = robot_cls(
        config_cls(
            port1=args.port1,
            port2=args.port2,
            cameras=_build_camera_configs(
                args.camera,
                width=args.camera_width,
                height=args.camera_height,
                fps=args.camera_fps,
            ),
            use_degrees=args.use_degrees,
        )
    )

    orbbec_rgb = OrbbecRgbConfig(
        enabled=True,
        launch_capture=not args.orbbec_no_launch,
        capture_bin=Path(
            args.orbbec_capture_bin
            or REPO_ROOT / "build" / "orbbec_rgb_test" / "orbbec_rgb_test"
        ).expanduser().resolve(),
        output_dir=Path(args.orbbec_output_dir).expanduser().resolve(),
        width=args.orbbec_width,
        height=args.orbbec_height,
        fps=args.orbbec_fps,
        timeout_ms=args.orbbec_timeout_ms,
        log_every=0,
    )

    policy, preprocessor, postprocessor, ds_meta = _load_policy_stack(
        policy_path=policy_path,
        dataset_repo_id=args.dataset_repo_id,
        dataset_root=dataset_root,
        device=args.device,
        expected_policy_type=expected_policy_type,
    )
    action_names = list(ds_meta.features["action"]["names"])
    policy_type = str(
        getattr(policy.config, "type", getattr(policy, "name", "unknown"))
    ).lower()
    _configure_act_action_steps(policy, args.act_action_steps)
    print(f"Action names: {action_names}")
    print(f"Control scope: {_control_scope(action_names)}")
    _print_policy_execution_config(policy)
    if policy_type == "act" and args.async_inference:
        print(
            "ACT uses its native receding-horizon action queue; "
            "disabling generic overlapping-chunk asynchronous inference."
        )
        args.async_inference = False

    orbbec_process = _start_orbbec_rgb_sidecar(orbbec_rgb)
    recording_stub = RecordingSession(
        dataset=None,
        task=args.task,
        orbbec_output_dir=orbbec_rgb.output_dir,
    )
    rerun_initialized = False
    try:
        if args.display_data:
            from lerobot.utils.visualization_utils import init_rerun

            init_rerun(session_name=rerun_session_name)
            rerun_initialized = True
            print("Rerun viewer started for live policy observations and actions.")
        _connect_robot(robot, auto_restore_calibration=not args.manual_calibration_prompt)
        if args.startup_pose and (not args.dry_run or args.startup_pose_in_dry_run):
            _run_startup_pose(
                robot=robot,
                head_pan_deg=args.startup_head_pan_deg,
                stow_wait_s=args.startup_nav_stow_wait_s,
                action_ready_elbow_delta=args.startup_action_ready_elbow_delta,
                action_ready_shoulder_delta=args.startup_action_ready_shoulder_delta,
                action_ready_wrist_delta=args.startup_action_ready_wrist_delta,
                steps_per_stage=args.startup_pose_steps,
                stage_delay_s=args.startup_pose_stage_delay_s,
            )
        elif args.startup_pose and args.dry_run:
            print("Dry run: skipping physical NAV_STOW -> ACTION_READY startup motion.")
        _wait_for_required_observations(
            robot=robot,
            recording_stub=recording_stub,
            ds_features=ds_meta.features,
            timeout_s=args.camera_ready_timeout_s,
        )
        _reset_policy_stack(policy, preprocessor, postprocessor)
        if args.policy_warmup_runs > 0:
            _warmup_policy_stack(
                robot=robot,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                ds_features=ds_meta.features,
                recording_stub=recording_stub,
                task=args.task,
                robot_type=getattr(robot, "name", args.robot_kind),
                runs=args.policy_warmup_runs,
            )
            _reset_policy_stack(policy, preprocessor, postprocessor)
        if args.warmup_s > 0:
            print(f"Starting rollout in {args.warmup_s:.1f}s. Keep e-stop/disable within reach.")
            time.sleep(args.warmup_s)
        _run_rollout(
            robot=robot,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            ds_features=ds_meta.features,
            recording_stub=recording_stub,
            task=args.task,
            robot_type=getattr(robot, "name", args.robot_kind),
            fps=args.fps,
            duration_s=args.duration_s,
            dry_run=args.dry_run,
            print_every=args.print_every,
            action_names=action_names,
            async_inference=args.async_inference,
            async_chunk_threshold=args.async_chunk_threshold,
            async_new_action_weight=args.async_new_action_weight,
            max_joint_delta=args.max_joint_delta,
            max_gripper_delta=args.max_gripper_delta,
            max_base_x_vel=args.max_base_x_vel,
            max_base_theta_vel=args.max_base_theta_vel,
            missing_camera_grace_s=args.missing_camera_grace_s,
            camera_cache_max_age_s=args.camera_cache_max_age_s,
            display_data=args.display_data,
            display_compressed_images=args.display_compressed_images,
        )
    finally:
        _send_stop(robot, action_names)
        _stop_orbbec_rgb_sidecar(orbbec_process)
        try:
            robot.disconnect()
        except Exception:
            pass
        if rerun_initialized:
            import rerun as rr

            rr.rerun_shutdown()
    return 0


def _wait_for_required_observations(
    *,
    robot: Any,
    recording_stub: RecordingSession,
    ds_features: dict[str, dict],
    timeout_s: float,
) -> None:
    required_images = _required_dataset_image_keys(ds_features)
    if not required_images:
        return

    print(f"Waiting for required camera observations: {', '.join(required_images)}")
    deadline = time.monotonic() + max(0.0, timeout_s)
    last_missing: list[str] = []
    while True:
        camera_obs, camera_error = _get_robot_observation_best_effort(robot)
        if camera_error is not None:
            print(f"Camera observation warning while waiting: {camera_error}")
        obs = _augment_recording_observation(recording_stub, camera_obs)
        missing = _missing_required_image_keys(ds_features, obs)
        if not missing:
            print("All required camera observations are available.")
            return
        last_missing = missing
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "Required camera observations are missing before rollout: "
                f"{', '.join(last_missing)}. For Orbbec/head, make sure "
                f"{recording_stub.orbbec_output_dir / 'latest.ppm' if recording_stub.orbbec_output_dir else 'latest.ppm'} "
                "is being updated."
            )
        time.sleep(0.1)


def _reset_policy_stack(policy: Any, preprocessor: Any, postprocessor: Any) -> None:
    policy.reset()
    for processor in (preprocessor, postprocessor):
        reset = getattr(processor, "reset", None)
        if callable(reset):
            reset()


def _warmup_policy_stack(
    *,
    robot: Any,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    ds_features: dict[str, dict],
    recording_stub: RecordingSession,
    task: str,
    robot_type: str,
    runs: int,
) -> None:
    try:
        from lerobot.datasets.feature_utils import build_dataset_frame
    except ImportError:
        from lerobot.datasets.utils import build_dataset_frame
    from lerobot.utils.constants import OBS_STR
    from lerobot.utils.control_utils import predict_action

    device = next(policy.parameters()).device
    runs = max(0, int(runs))
    if runs == 0:
        return

    print(f"Warming up policy with {runs} throwaway prediction(s); no actions will be sent.")
    for index in range(runs):
        _reset_policy_stack(policy, preprocessor, postprocessor)
        camera_obs, camera_error = _get_robot_observation_best_effort(robot)
        if camera_error is not None:
            raise RuntimeError("Could not obtain a complete observation for policy warmup.") from camera_error
        obs = _augment_recording_observation(recording_stub, camera_obs)
        obs = _observation_with_cached_required_images(
            ds_features,
            obs,
            {},
            max_age_s=0.0,
        )
        missing_images = _missing_required_image_keys(ds_features, obs)
        if missing_images:
            raise RuntimeError(
                "Required camera observations disappeared during policy warmup: "
                f"{', '.join(missing_images)}"
            )
        observation_frame = build_dataset_frame(ds_features, obs, prefix=OBS_STR)
        started_at = time.perf_counter()
        predict_action(
            observation=observation_frame,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=bool(getattr(policy.config, "use_amp", False)),
            task=task,
            robot_type=robot_type,
        )
        elapsed_ms = (time.perf_counter() - started_at) * 1_000.0
        print(f"Policy warmup {index + 1}/{runs} complete in {elapsed_ms:.1f} ms.")


def _configure_act_action_steps(policy: Any, action_steps: int | None) -> None:
    policy_type = str(
        getattr(policy.config, "type", getattr(policy, "name", "unknown"))
    ).lower()
    if action_steps is None:
        return
    if policy_type != "act":
        raise RuntimeError("--act-action-steps can only be used with an ACT checkpoint.")

    chunk_size = int(getattr(policy.config, "chunk_size", 1))
    action_steps = int(action_steps)
    if not 1 <= action_steps <= chunk_size:
        raise ValueError(
            f"--act-action-steps must be between 1 and the ACT chunk size ({chunk_size})."
        )

    saved_action_steps = int(getattr(policy.config, "n_action_steps", 1))
    policy.config.n_action_steps = action_steps
    policy.reset()
    if saved_action_steps != action_steps:
        print(
            "ACT action horizon override: "
            f"checkpoint={saved_action_steps}, runtime={action_steps}."
        )


def _print_policy_execution_config(policy: Any) -> None:
    policy_type = str(getattr(policy.config, "type", getattr(policy, "name", "unknown"))).lower()
    chunk_size = int(getattr(policy.config, "chunk_size", 1))
    if policy_type == "act":
        n_action_steps = int(getattr(policy.config, "n_action_steps", 1))
        temporal_ensemble_coeff = getattr(policy.config, "temporal_ensemble_coeff", None)
        if temporal_ensemble_coeff is None:
            print(
                "ACT execution: "
                f"predict {chunk_size} actions, execute {n_action_steps}, then observe and replan."
            )
        else:
            print(
                "ACT execution: "
                f"predict {chunk_size} actions at every step with temporal ensemble coefficient "
                f"{temporal_ensemble_coeff}."
            )
        return
    print(f"{policy_type} execution: action chunk size {chunk_size}.")


def _run_startup_pose(
    *,
    robot: Any,
    stow_wait_s: float,
    action_ready_elbow_delta: float,
    action_ready_shoulder_delta: float,
    action_ready_wrist_delta: float,
    steps_per_stage: int,
    stage_delay_s: float,
    head_pan_deg: float | None = 0.0,
) -> None:
    startup_sides = ("left", "right")

    if head_pan_deg is not None:
        print(f"Centering head pan at {head_pan_deg:.1f} degrees.")
        _move_to_joint_targets(
            robot,
            {"head_motor_1.pos": float(head_pan_deg)},
            steps=steps_per_stage,
            delay_s=stage_delay_s,
        )

    stow_targets = {
        f"{side}_arm_{joint}.pos": value
        for side in startup_sides
        for joint, value in _NAV_STOW_ARM_POSE.items()
    }
    print("Moving both arms to NAV_STOW; policy control remains dataset-defined.")
    _move_to_joint_targets(robot, stow_targets, steps=steps_per_stage, delay_s=stage_delay_s)

    if stow_wait_s > 0:
        print(f"Waiting {stow_wait_s:.1f}s before moving to ACTION_READY.")
        time.sleep(stow_wait_s)

    print(
        "Moving to ACTION_READY: "
        f"elbow delta {action_ready_elbow_delta:+.1f}, "
        f"shoulder delta {action_ready_shoulder_delta:+.1f}, "
        f"wrist delta {action_ready_wrist_delta:+.1f}."
    )
    obs = _get_robot_observation(robot, use_camera=False)
    elbow_targets: dict[str, float] = {}
    for side in startup_sides:
        elbow_key = f"{side}_arm_elbow_flex.pos"
        wrist_key = f"{side}_arm_wrist_flex.pos"
        if elbow_key in obs:
            elbow_targets[elbow_key] = float(obs[elbow_key]) + action_ready_elbow_delta
        if action_ready_wrist_delta and wrist_key in obs:
            elbow_targets[wrist_key] = float(obs[wrist_key]) + action_ready_wrist_delta
    _move_to_joint_targets(robot, elbow_targets, steps=steps_per_stage, delay_s=stage_delay_s)

    obs = _get_robot_observation(robot, use_camera=False)
    shoulder_targets: dict[str, float] = {}
    for side in startup_sides:
        shoulder_key = f"{side}_arm_shoulder_lift.pos"
        if shoulder_key in obs:
            shoulder_targets[shoulder_key] = float(obs[shoulder_key]) + action_ready_shoulder_delta
    _move_to_joint_targets(robot, shoulder_targets, steps=steps_per_stage, delay_s=stage_delay_s)
    print("Startup pose complete: robot is at ACTION_READY.")


def _move_to_joint_targets(
    robot: Any,
    targets: dict[str, float],
    *,
    steps: int,
    delay_s: float,
) -> None:
    if not targets:
        return
    obs = _get_robot_observation(robot, use_camera=False)
    starts = {key: float(obs[key]) for key in targets if key in obs}
    if not starts:
        return
    steps = max(1, steps)
    for idx in range(1, steps + 1):
        ratio = idx / steps
        action = {
            key: start + (float(targets[key]) - start) * ratio
            for key, start in starts.items()
        }
        robot.send_action(action)
        if delay_s > 0:
            time.sleep(delay_s)


def _controlled_arm_sides(action_names: list[str]) -> tuple[str, ...]:
    return tuple(
        side
        for side in ("left", "right")
        if any(name.startswith(f"{side}_arm_") for name in action_names)
    )


def _control_scope(action_names: list[str]) -> str:
    if tuple(action_names) == RIGHT_ARM_ACTION_NAMES:
        return "right arm only (no left arm, head motors, or base)"
    return "dataset-defined: " + ", ".join(action_names)


def _run_rollout(
    *,
    robot: Any,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    ds_features: dict[str, dict],
    recording_stub: RecordingSession,
    task: str,
    robot_type: str,
    fps: float,
    duration_s: float,
    dry_run: bool,
    print_every: int,
    action_names: list[str],
    async_inference: bool,
    async_chunk_threshold: float,
    async_new_action_weight: float,
    max_joint_delta: float,
    max_gripper_delta: float,
    max_base_x_vel: float,
    max_base_theta_vel: float,
    missing_camera_grace_s: float,
    camera_cache_max_age_s: float,
    display_data: bool,
    display_compressed_images: bool,
) -> None:
    if async_inference:
        _run_rollout_async(
            robot=robot,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            ds_features=ds_features,
            recording_stub=recording_stub,
            task=task,
            robot_type=robot_type,
            fps=fps,
            duration_s=duration_s,
            dry_run=dry_run,
            print_every=print_every,
            action_names=action_names,
            async_chunk_threshold=async_chunk_threshold,
            async_new_action_weight=async_new_action_weight,
            max_joint_delta=max_joint_delta,
            max_gripper_delta=max_gripper_delta,
            max_base_x_vel=max_base_x_vel,
            max_base_theta_vel=max_base_theta_vel,
            missing_camera_grace_s=missing_camera_grace_s,
            camera_cache_max_age_s=camera_cache_max_age_s,
            display_data=display_data,
            display_compressed_images=display_compressed_images,
        )
        return

    from lerobot.policies.utils import make_robot_action
    from lerobot.utils.control_utils import predict_action
    try:
        from lerobot.datasets.feature_utils import build_dataset_frame
    except ImportError:
        from lerobot.datasets.utils import build_dataset_frame
    from lerobot.utils.constants import ACTION, OBS_STR

    log_rerun_data = None
    rr = None
    if display_data:
        import rerun as rr
        from lerobot.utils.visualization_utils import log_rerun_data

    device = next(policy.parameters()).device
    period_s = 1.0 / max(1.0, fps)
    deadline = time.monotonic() + max(0.0, duration_s)
    step = 0
    previous_step_started_at: float | None = None
    missing_since: float | None = None
    image_cache: dict[str, tuple[Any, float]] = {}
    while time.monotonic() < deadline:
        loop_t = time.perf_counter()
        actual_fps = (
            None
            if previous_step_started_at is None
            else 1.0 / max(loop_t - previous_step_started_at, 1e-9)
        )
        previous_step_started_at = loop_t
        camera_obs, camera_error = _get_robot_observation_best_effort(robot)
        if camera_error is not None:
            print(f"Camera observation warning: {camera_error}")
        obs = _augment_recording_observation(recording_stub, camera_obs)
        obs = _observation_with_cached_required_images(
            ds_features,
            obs,
            image_cache,
            max_age_s=camera_cache_max_age_s,
        )
        missing_images = _missing_required_image_keys(ds_features, obs)
        if missing_images:
            now = time.monotonic()
            if missing_since is None:
                missing_since = now
                print(f"Missing required camera observation(s), holding rollout: {', '.join(missing_images)}")
            if now - missing_since > missing_camera_grace_s:
                raise RuntimeError(
                    "Required camera observations stayed missing during rollout: "
                    f"{', '.join(missing_images)}"
                )
            elapsed = time.perf_counter() - loop_t
            if elapsed < period_s:
                time.sleep(period_s - elapsed)
            continue
        missing_since = None
        observation_frame = build_dataset_frame(ds_features, obs, prefix=OBS_STR)
        queued_actions_before = _policy_action_queue_depth(policy, ACTION)
        chunk_refill = queued_actions_before == 0
        inference_started_at = time.perf_counter()
        action_tensor = predict_action(
            observation=observation_frame,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=bool(getattr(policy.config, "use_amp", False)),
            task=task,
            robot_type=robot_type,
        )
        inference_s = time.perf_counter() - inference_started_at
        raw_action = make_robot_action(action_tensor, ds_features)
        raw_action = _unrename_action(raw_action, action_names)
        sent_action = _clamp_action(
            raw_action,
            obs,
            max_joint_delta=max_joint_delta,
            max_gripper_delta=max_gripper_delta,
            max_base_x_vel=max_base_x_vel,
            max_base_theta_vel=max_base_theta_vel,
        )
        if log_rerun_data is not None:
            action_diagnostics = {
                **{f"raw/{key}": value for key, value in raw_action.items()},
                **{f"sent/{key}": value for key, value in sent_action.items()},
                **{
                    f"clamp_delta/{key}": sent_action[key] - value
                    for key, value in raw_action.items()
                    if key in sent_action
                },
            }
            log_rerun_data(
                observation=obs,
                action=action_diagnostics,
                compress_images=display_compressed_images,
            )
            rr.log("diagnostics/inference_ms", rr.Scalars(inference_s * 1_000.0))
            rr.log("diagnostics/chunk_refill", rr.Scalars(float(chunk_refill)))
            rr.log("diagnostics/queued_actions_before", rr.Scalars(float(queued_actions_before)))
            if actual_fps is not None:
                rr.log("diagnostics/actual_fps", rr.Scalars(actual_fps))
        clamp_deltas = {
            key: abs(sent_action[key] - value)
            for key, value in raw_action.items()
            if key in sent_action and abs(sent_action[key] - value) > 1e-6
        }
        if chunk_refill:
            max_clamp_delta = max(clamp_deltas.values(), default=0.0)
            gripper_key = "right_arm_gripper.pos"
            gripper_observed = float(obs[gripper_key]) if gripper_key in obs else float("nan")
            gripper_raw = raw_action.get(gripper_key, float("nan"))
            gripper_sent = sent_action.get(gripper_key, float("nan"))
            print(
                f"chunk_refill step={step} inference_ms={inference_s * 1_000.0:.1f} "
                f"clamped_actions={len(clamp_deltas)} max_clamp_delta={max_clamp_delta:.2f} "
                f"right_gripper(observed={gripper_observed:.2f}, raw={gripper_raw:.2f}, sent={gripper_sent:.2f})"
            )
        if print_every > 0 and step % print_every == 0:
            print(f"step={step} action={json.dumps(_round_floats(sent_action), sort_keys=True)}")
        if not dry_run:
            robot.send_action(sent_action)
        step += 1
        elapsed = time.perf_counter() - loop_t
        if elapsed < period_s:
                time.sleep(period_s - elapsed)


def _policy_action_queue_depth(policy: Any, action_key: str) -> int:
    action_queue = getattr(policy, "_action_queue", None)
    if action_queue is not None:
        return len(action_queue)
    action_queue = getattr(policy, "_queues", {}).get(action_key)
    if action_queue is not None:
        return len(action_queue)
    return -1


@dataclass(frozen=True)
class _ChunkRequest:
    request_step: int
    observation: dict[str, Any]
    submitted_at: float


@dataclass(frozen=True)
class _ChunkResult:
    request_step: int
    actions: list[dict[str, float]]
    inference_s: float
    end_to_end_s: float
    error: BaseException | None = None


class _AsyncChunkPredictor:
    def __init__(self, predict_chunk: Any) -> None:
        self._predict_chunk = predict_chunk
        self._requests: queue.Queue[_ChunkRequest | None] = queue.Queue(maxsize=1)
        self._results: queue.Queue[_ChunkResult] = queue.Queue()
        self._busy = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="smolvla-chunk-predictor",
            daemon=True,
        )
        self._thread.start()

    @property
    def busy(self) -> bool:
        return self._busy.is_set()

    def submit(self, request_step: int, observation: dict[str, Any]) -> bool:
        if self.busy or self._stop.is_set():
            return False
        request = _ChunkRequest(
            request_step=request_step,
            observation=_copy_policy_observation(observation),
            submitted_at=time.perf_counter(),
        )
        self._busy.set()
        try:
            self._requests.put_nowait(request)
        except queue.Full:
            self._busy.clear()
            return False
        return True

    def poll(self) -> _ChunkResult | None:
        try:
            return self._results.get_nowait()
        except queue.Empty:
            return None

    def close(self, timeout_s: float = 5.0) -> None:
        self._stop.set()
        try:
            self._requests.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=max(0.0, timeout_s))
        if self._thread.is_alive():
            print("Warning: asynchronous policy worker did not stop before timeout.")

    def _run(self) -> None:
        while not self._stop.is_set():
            request = self._requests.get()
            if request is None:
                return
            inference_started_at = time.perf_counter()
            try:
                actions = self._predict_chunk(request.observation)
                inference_s = time.perf_counter() - inference_started_at
                result = _ChunkResult(
                    request_step=request.request_step,
                    actions=actions,
                    inference_s=inference_s,
                    end_to_end_s=time.perf_counter() - request.submitted_at,
                )
            except BaseException as exc:
                inference_s = time.perf_counter() - inference_started_at
                result = _ChunkResult(
                    request_step=request.request_step,
                    actions=[],
                    inference_s=inference_s,
                    end_to_end_s=time.perf_counter() - request.submitted_at,
                    error=exc,
                )
            self._results.put(result)
            self._busy.clear()


def _run_rollout_async(
    *,
    robot: Any,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    ds_features: dict[str, dict],
    recording_stub: RecordingSession,
    task: str,
    robot_type: str,
    fps: float,
    duration_s: float,
    dry_run: bool,
    print_every: int,
    action_names: list[str],
    async_chunk_threshold: float,
    async_new_action_weight: float,
    max_joint_delta: float,
    max_gripper_delta: float,
    max_base_x_vel: float,
    max_base_theta_vel: float,
    missing_camera_grace_s: float,
    camera_cache_max_age_s: float,
    display_data: bool,
    display_compressed_images: bool,
) -> None:
    try:
        from lerobot.datasets.feature_utils import build_dataset_frame
    except ImportError:
        from lerobot.datasets.utils import build_dataset_frame
    from lerobot.utils.constants import OBS_STR

    if duration_s <= 0:
        return

    log_rerun_data = None
    rr = None
    if display_data:
        import rerun as rr
        from lerobot.utils.visualization_utils import log_rerun_data

    device = next(policy.parameters()).device
    use_amp = bool(getattr(policy.config, "use_amp", False))
    chunk_size = max(1, int(getattr(policy.config, "chunk_size", 50)))
    threshold_fraction = _clamp(float(async_chunk_threshold), 0.0, 1.0)
    new_action_weight = _clamp(float(async_new_action_weight), 0.0, 1.0)
    request_threshold = max(1, int(round(chunk_size * threshold_fraction)))
    period_s = 1.0 / max(1.0, fps)
    predictor = _AsyncChunkPredictor(
        lambda observation: _predict_action_chunk(
            observation=observation,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=use_amp,
            task=task,
            robot_type=robot_type,
            ds_features=ds_features,
            action_names=action_names,
        )
    )

    print(
        "Asynchronous inference enabled: "
        f"chunk_size={chunk_size}, request_threshold={request_threshold}, "
        f"new_action_weight={new_action_weight:.2f}."
    )
    timed_actions: dict[int, dict[str, float]] = {}
    image_cache: dict[str, tuple[Any, float]] = {}
    previous_step_started_at: float | None = None
    missing_since: float | None = None
    rollout_started_at: float | None = None
    rollout_deadline: float | None = None
    initial_request_submitted = False
    last_raw_action: dict[str, float] | None = None
    step = 0
    chunks_received = 0
    stale_actions_total = 0
    queue_underruns = 0

    try:
        while rollout_deadline is None or time.monotonic() < rollout_deadline:
            loop_t = time.perf_counter()
            actual_fps = (
                None
                if previous_step_started_at is None
                else 1.0 / max(loop_t - previous_step_started_at, 1e-9)
            )
            previous_step_started_at = loop_t
            camera_obs, camera_error = _get_robot_observation_best_effort(robot)
            if camera_error is not None:
                print(f"Camera observation warning: {camera_error}")
            obs = _augment_recording_observation(recording_stub, camera_obs)
            obs = _observation_with_cached_required_images(
                ds_features,
                obs,
                image_cache,
                max_age_s=camera_cache_max_age_s,
            )
            missing_images = _missing_required_image_keys(ds_features, obs)
            if missing_images:
                now = time.monotonic()
                if missing_since is None:
                    missing_since = now
                    print(
                        "Missing required camera observation(s), holding rollout: "
                        f"{', '.join(missing_images)}"
                    )
                if now - missing_since > missing_camera_grace_s:
                    raise RuntimeError(
                        "Required camera observations stayed missing during rollout: "
                        f"{', '.join(missing_images)}"
                    )
                _sleep_for_period(loop_t, period_s)
                continue
            missing_since = None
            observation_frame = build_dataset_frame(ds_features, obs, prefix=OBS_STR)

            chunk_result = predictor.poll()
            stale_actions_dropped = 0
            if chunk_result is not None:
                if chunk_result.error is not None:
                    raise RuntimeError(
                        f"Asynchronous action prediction failed at request step {chunk_result.request_step}."
                    ) from chunk_result.error
                timed_actions, stale_actions_dropped = _merge_action_chunk(
                    timed_actions,
                    request_step=chunk_result.request_step,
                    actions=chunk_result.actions,
                    current_step=step,
                    new_action_weight=new_action_weight,
                )
                chunk_size = max(1, len(chunk_result.actions))
                request_threshold = max(1, int(round(chunk_size * threshold_fraction)))
                chunks_received += 1
                stale_actions_total += stale_actions_dropped

            if not initial_request_submitted:
                if predictor.submit(step, observation_frame):
                    initial_request_submitted = True
                    print("Initial action chunk requested; holding the robot until it is ready.")

            raw_action = timed_actions.pop(step, None)
            queue_underrun = raw_action is None
            if queue_underrun:
                if last_raw_action is None:
                    _sleep_for_period(loop_t, period_s)
                    continue
                raw_action = dict(last_raw_action)
                queue_underruns += 1
            else:
                last_raw_action = dict(raw_action)

            if rollout_started_at is None:
                rollout_started_at = time.monotonic()
                rollout_deadline = rollout_started_at + duration_s
                previous_step_started_at = None
                print("Initial action chunk ready; starting timed policy rollout.")

            sent_action = _clamp_action(
                raw_action,
                obs,
                max_joint_delta=max_joint_delta,
                max_gripper_delta=max_gripper_delta,
                max_base_x_vel=max_base_x_vel,
                max_base_theta_vel=max_base_theta_vel,
            )
            queue_depth = len(timed_actions)
            request_submitted = False
            if queue_depth <= request_threshold and not predictor.busy:
                request_submitted = predictor.submit(step, observation_frame)

            if log_rerun_data is not None:
                action_diagnostics = {
                    **{f"raw/{key}": value for key, value in raw_action.items()},
                    **{f"sent/{key}": value for key, value in sent_action.items()},
                    **{
                        f"clamp_delta/{key}": sent_action[key] - value
                        for key, value in raw_action.items()
                        if key in sent_action
                    },
                }
                log_rerun_data(
                    observation=obs,
                    action=action_diagnostics,
                    compress_images=display_compressed_images,
                )
                rr.log("diagnostics/queue_depth", rr.Scalars(float(queue_depth)))
                rr.log("diagnostics/queue_underrun", rr.Scalars(float(queue_underrun)))
                rr.log("diagnostics/stale_actions_dropped", rr.Scalars(float(stale_actions_dropped)))
                rr.log("diagnostics/chunk_request", rr.Scalars(float(request_submitted)))
                if actual_fps is not None:
                    rr.log("diagnostics/actual_fps", rr.Scalars(actual_fps))
                if chunk_result is not None:
                    rr.log(
                        "diagnostics/chunk_inference_ms",
                        rr.Scalars(chunk_result.inference_s * 1_000.0),
                    )
                    rr.log(
                        "diagnostics/chunk_end_to_end_ms",
                        rr.Scalars(chunk_result.end_to_end_s * 1_000.0),
                    )

            if chunk_result is not None:
                clamp_deltas = {
                    key: abs(sent_action[key] - value)
                    for key, value in raw_action.items()
                    if key in sent_action and abs(sent_action[key] - value) > 1e-6
                }
                gripper_key = "right_arm_gripper.pos"
                gripper_observed = float(obs[gripper_key]) if gripper_key in obs else float("nan")
                print(
                    f"chunk_result request_step={chunk_result.request_step} received_step={step} "
                    f"inference_ms={chunk_result.inference_s * 1_000.0:.1f} "
                    f"stale_actions={stale_actions_dropped} queue_depth={queue_depth} "
                    f"clamped_actions={len(clamp_deltas)} "
                    f"right_gripper(observed={gripper_observed:.2f}, "
                    f"raw={raw_action.get(gripper_key, float('nan')):.2f}, "
                    f"sent={sent_action.get(gripper_key, float('nan')):.2f})"
                )
            if queue_underrun and print_every > 0 and step % print_every == 0:
                print(f"queue_underrun step={step}; holding the previous action target.")
            if print_every > 0 and step % print_every == 0:
                print(f"step={step} action={json.dumps(_round_floats(sent_action), sort_keys=True)}")
            if not dry_run:
                robot.send_action(sent_action)
            step += 1
            _sleep_for_period(loop_t, period_s)
    finally:
        predictor.close()
        if rollout_started_at is not None:
            elapsed_s = max(time.monotonic() - rollout_started_at, 1e-9)
            print(
                f"async_rollout_summary steps={step} elapsed_s={elapsed_s:.2f} "
                f"effective_fps={step / elapsed_s:.2f} chunks={chunks_received} "
                f"stale_actions={stale_actions_total} queue_underruns={queue_underruns}"
            )


def _merge_action_chunk(
    existing: dict[int, dict[str, float]],
    *,
    request_step: int,
    actions: list[dict[str, float]],
    current_step: int,
    new_action_weight: float,
) -> tuple[dict[int, dict[str, float]], int]:
    weight = _clamp(float(new_action_weight), 0.0, 1.0)
    merged: dict[int, dict[str, float]] = {}
    stale_actions = 0
    for offset, new_action in enumerate(actions):
        target_step = request_step + offset
        if target_step < current_step:
            stale_actions += 1
            continue
        old_action = existing.get(target_step)
        if old_action is None:
            merged[target_step] = dict(new_action)
            continue
        merged[target_step] = {
            key: (1.0 - weight) * float(old_action.get(key, value)) + weight * float(value)
            for key, value in new_action.items()
        }
    return merged, stale_actions


def _copy_policy_observation(observation: dict[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, value in observation.items():
        value_copy = getattr(value, "copy", None)
        if callable(value_copy):
            try:
                copied[key] = value_copy()
                continue
            except Exception:
                pass
        copied[key] = value
    return copied


def _sleep_for_period(loop_started_at: float, period_s: float) -> None:
    elapsed = time.perf_counter() - loop_started_at
    if elapsed < period_s:
        time.sleep(period_s - elapsed)


def _required_dataset_image_keys(ds_features: dict[str, dict]) -> list[str]:
    prefix = "observation.images."
    return [
        key.removeprefix(prefix)
        for key, feature in ds_features.items()
        if key.startswith(prefix) and feature.get("dtype") in {"image", "video"}
    ]


def _missing_required_image_keys(ds_features: dict[str, dict], obs: dict[str, Any]) -> list[str]:
    return [key for key in _required_dataset_image_keys(ds_features) if key not in obs]


def _observation_with_cached_required_images(
    ds_features: dict[str, dict],
    obs: dict[str, Any],
    image_cache: dict[str, tuple[Any, float]],
    *,
    max_age_s: float,
) -> dict[str, Any]:
    now = time.monotonic()
    augmented = dict(obs)
    for key in _required_dataset_image_keys(ds_features):
        if key in augmented:
            writable_value = _copy_observation_value(augmented[key])
            augmented[key] = writable_value
            image_cache[key] = (writable_value, now)
            continue
        cached = image_cache.get(key)
        if cached is None:
            continue
        value, captured_at = cached
        if now - captured_at <= max_age_s:
            augmented[key] = value
    return augmented


def _copy_observation_value(value: Any) -> Any:
    try:
        import numpy as np

        return np.asarray(value).copy()
    except Exception:
        return value


def _clamp_action(
    action: dict[str, float],
    obs: dict[str, Any],
    *,
    max_joint_delta: float,
    max_gripper_delta: float,
    max_base_x_vel: float,
    max_base_theta_vel: float,
) -> dict[str, float]:
    clamped = {}
    for key, value in action.items():
        value = float(value)
        if key == "x.vel":
            value = _clamp(value, -max_base_x_vel, max_base_x_vel)
        elif key == "theta.vel":
            value = _clamp(value, -max_base_theta_vel, max_base_theta_vel)
        elif key.endswith(".pos") and key in obs:
            current = float(obs[key])
            max_delta = max_gripper_delta if key.endswith("_gripper.pos") else max_joint_delta
            if math.isfinite(max_delta) and max_delta >= 0:
                value = _clamp(value, current - max_delta, current + max_delta)
        clamped[key] = value
    return clamped


def _clamp(value: float, low: float, high: float) -> float:
    if high < low:
        low, high = high, low
    return max(low, min(high, value))


def _round_floats(action: dict[str, float]) -> dict[str, float]:
    return {key: round(float(value), 4) for key, value in action.items()}


def _send_stop(robot: Any, action_names: list[str]) -> None:
    stop_action = {name: 0.0 for name in BASE_ACTION_NAMES if name in action_names}
    if not stop_action:
        return
    try:
        robot.send_action(stop_action)
    except Exception:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
