#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multido_xlerobot import XLeRobotInterface
from multido_xlerobot.bootstrap import bootstrap_xlerobot, resolve_xlerobot_repo_root
from xlerobot_playground.real_backend import (
    OrbbecRgbConfig,
    _augment_recording_observation,
    _build_camera_configs,
    _connect_robot,
    _get_robot_observation,
    _get_robot_observation_best_effort,
    _start_orbbec_rgb_sidecar,
    _stop_orbbec_rgb_sidecar,
    RecordingSession,
)


DEFAULT_POLICY_PATH = (
    REPO_ROOT
    / "outputs/train/smolvla_xlerobot_grab_to_basket_v0_cuda_b4_s16k/checkpoints/016000/pretrained_model"
)
DEFAULT_DATASET_ROOT = Path(
    "/home/alin/.cache/huggingface/lerobot/alindumitru/robot42_grab_to_basket_v0"
)
DEFAULT_TASK = "Grab the Tabasco sauce bottle and put it in the robot basket."
CAMERA_RENAME = {
    "observation.images.head": "observation.images.camera1",
    "observation.images.left_wrist": "observation.images.camera2",
    "observation.images.right_wrist": "observation.images.camera3",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a trained SmolVLA grab-to-basket policy on the real XLeRobot. "
            "Start the robot in the same ACTION_READY pose used for data collection."
        )
    )
    parser.add_argument("--policy-path", default=str(DEFAULT_POLICY_PATH))
    parser.add_argument("--dataset-repo-id", default="alindumitru/robot42_grab_to_basket_v0")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--repo-root", default=str(resolve_xlerobot_repo_root()))
    parser.add_argument("--robot-kind", choices=("xlerobot", "xlerobot_2wheels"), default="xlerobot_2wheels")
    parser.add_argument("--port1", default="/dev/ttyACM0")
    parser.add_argument("--port2", default="/dev/ttyACM1")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--duration-s", type=float, default=12.0)
    parser.add_argument("--warmup-s", type=float, default=3.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-degrees", action="store_true", default=True)
    parser.add_argument("--no-use-degrees", action="store_false", dest="use_degrees")
    parser.add_argument("--manual-calibration-prompt", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print predicted actions without sending them.")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument(
        "--max-joint-delta",
        type=float,
        default=12.0,
        help="Clamp absolute position action jumps from current observation, in joint units.",
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
    )
    action_names = list(ds_meta.features["action"]["names"])
    print(f"Action names: {action_names}")

    orbbec_process = _start_orbbec_rgb_sidecar(orbbec_rgb)
    recording_stub = RecordingSession(
        dataset=None,
        task=args.task,
        orbbec_output_dir=orbbec_rgb.output_dir,
    )
    try:
        _connect_robot(robot, auto_restore_calibration=not args.manual_calibration_prompt)
        policy.reset()
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
            max_joint_delta=args.max_joint_delta,
            max_base_x_vel=args.max_base_x_vel,
            max_base_theta_vel=args.max_base_theta_vel,
        )
    finally:
        _send_stop(robot)
        _stop_orbbec_rgb_sidecar(orbbec_process)
        try:
            robot.disconnect()
        except Exception:
            pass
    return 0


def _load_policy_stack(
    *,
    policy_path: Path,
    dataset_repo_id: str,
    dataset_root: Path,
    device: str,
) -> tuple[Any, Any, Any, Any]:
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false.")

    ds_meta = LeRobotDatasetMetadata(dataset_repo_id, root=dataset_root)
    cfg = PreTrainedConfig.from_pretrained(policy_path)
    cfg.pretrained_path = str(policy_path)
    cfg.device = device
    policy = make_policy(cfg, ds_meta=ds_meta, rename_map=CAMERA_RENAME)
    preprocessor, postprocessor = make_pre_post_processors(
        cfg,
        pretrained_path=str(policy_path),
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": CAMERA_RENAME},
        },
    )
    policy.eval()
    return policy, preprocessor, postprocessor, ds_meta


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
    max_joint_delta: float,
    max_base_x_vel: float,
    max_base_theta_vel: float,
) -> None:
    import torch
    from lerobot.policies.utils import make_robot_action
    from lerobot.utils.control_utils import predict_action
    from lerobot.datasets.utils import build_dataset_frame
    from lerobot.utils.constants import OBS_STR

    device = next(policy.parameters()).device
    period_s = 1.0 / max(1.0, fps)
    deadline = time.monotonic() + max(0.0, duration_s)
    step = 0
    while time.monotonic() < deadline:
        loop_t = time.perf_counter()
        camera_obs, camera_error = _get_robot_observation_best_effort(robot)
        if camera_error is not None:
            print(f"Camera observation warning: {camera_error}")
        obs = _augment_recording_observation(recording_stub, camera_obs)
        observation_frame = build_dataset_frame(ds_features, obs, prefix=OBS_STR)
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
        action = make_robot_action(action_tensor, ds_features)
        action = _unrename_action(action, action_names)
        action = _clamp_action(
            action,
            obs,
            max_joint_delta=max_joint_delta,
            max_base_x_vel=max_base_x_vel,
            max_base_theta_vel=max_base_theta_vel,
        )
        if print_every > 0 and step % print_every == 0:
            print(f"step={step} action={json.dumps(_round_floats(action), sort_keys=True)}")
        if not dry_run:
            robot.send_action(action)
        step += 1
        elapsed = time.perf_counter() - loop_t
        if elapsed < period_s:
            time.sleep(period_s - elapsed)


def _unrename_action(action: dict[str, float], action_names: list[str]) -> dict[str, float]:
    return {name: float(action[name]) for name in action_names if name in action}


def _clamp_action(
    action: dict[str, float],
    obs: dict[str, Any],
    *,
    max_joint_delta: float,
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
        elif key.endswith(".pos") and key in obs and math.isfinite(max_joint_delta) and max_joint_delta >= 0:
            current = float(obs[key])
            value = _clamp(value, current - max_joint_delta, current + max_joint_delta)
        clamped[key] = value
    return clamped


def _clamp(value: float, low: float, high: float) -> float:
    if high < low:
        low, high = high, low
    return max(low, min(high, value))


def _round_floats(action: dict[str, float]) -> dict[str, float]:
    return {key: round(float(value), 4) for key, value in action.items()}


def _send_stop(robot: Any) -> None:
    try:
        robot.send_action({"x.vel": 0.0, "theta.vel": 0.0})
    except Exception:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
