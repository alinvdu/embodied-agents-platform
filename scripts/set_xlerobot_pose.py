#!/usr/bin/env python3
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
import builtins
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multido_xlerobot import XLeRobotInterface
from multido_xlerobot.bootstrap import resolve_xlerobot_repo_root


ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
HEAD_JOINTS = ("head_motor_1", "head_motor_2")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Move XLeRobot joints to requested angles, then print the current pose "
            "and reusable VR basket target flags."
        )
    )
    parser.add_argument("--repo-root", default=str(resolve_xlerobot_repo_root()))
    parser.add_argument("--robot-kind", choices=("xlerobot", "xlerobot_2wheels"), default="xlerobot_2wheels")
    parser.add_argument("--port1", default="/dev/tty.usbmodem5B140330101")
    parser.add_argument("--port2", default="/dev/tty.usbmodem5B140332271")
    parser.add_argument("--use-degrees", action="store_true")
    parser.add_argument("--manual-calibration-prompt", action="store_true")
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        metavar="JOINT=VALUE",
        help=(
            "Joint target, repeatable. Examples: "
            "`right_arm_elbow_flex.pos=20`, `right:elbow_flex=20`, `head_motor_1=0`."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=60,
        help="Interpolation steps used to reach the requested targets.",
    )
    parser.add_argument(
        "--delay-s",
        type=float,
        default=0.02,
        help="Delay between interpolation steps.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Connect and validate targets, but do not move the robot.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Number of decimal places to print.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    interface = XLeRobotInterface(Path(args.repo_root).expanduser().resolve())
    if args.robot_kind == "xlerobot_2wheels":
        config_cls, robot_cls = interface.robot_2wheels_classes()
    else:
        config_cls, robot_cls = interface.robot_classes()

    robot = robot_cls(
        config_cls(
            port1=args.port1,
            port2=args.port2,
            cameras={},
            use_degrees=args.use_degrees,
        )
    )

    _connect_robot(robot, auto_restore_calibration=not args.manual_calibration_prompt)
    try:
        obs = robot.get_observation(use_camera=False)
        try:
            targets = _parse_targets(args.target)
            _validate_targets(targets, obs)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if targets:
            _print_targets(targets)
            if args.dry_run:
                print("\nDry run: targets validated; robot was not moved.")
            else:
                _move_to_joint_targets(robot, obs, targets, steps=args.steps, delay_s=args.delay_s)
                obs = robot.get_observation(use_camera=False)
        else:
            print("\nNo --target values supplied; reading current pose only.")
    finally:
        robot.disconnect()

    _print_pose(obs, precision=args.precision)
    return 0


def _connect_robot(robot: Any, *, auto_restore_calibration: bool) -> None:
    if not auto_restore_calibration:
        robot.connect()
        return

    original_input = builtins.input

    def auto_input(prompt: str = "") -> str:
        if "restore calibration from file" in prompt:
            print(prompt)
            return ""
        return original_input(prompt)

    try:
        builtins.input = auto_input
        robot.connect()
    finally:
        builtins.input = original_input


def _parse_targets(raw_targets: list[str]) -> dict[str, float]:
    targets: dict[str, float] = {}
    for raw in raw_targets:
        if "=" not in raw:
            raise ValueError(f"Invalid --target `{raw}`. Use JOINT=VALUE.")
        raw_key, raw_value = raw.split("=", 1)
        key = _normalize_joint_key(raw_key.strip())
        try:
            targets[key] = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"Invalid value for --target `{raw}`.") from exc
    return targets


def _normalize_joint_key(raw_key: str) -> str:
    key = raw_key.strip()
    if ":" in key:
        side, joint = key.split(":", 1)
        side = side.strip()
        joint = joint.strip()
        if side in {"left", "right"} and joint in ARM_JOINTS:
            key = f"{side}_arm_{joint}"
    elif key.startswith(("left_", "right_")) and "_arm_" not in key:
        side, joint = key.split("_", 1)
        if joint in ARM_JOINTS:
            key = f"{side}_arm_{joint}"

    if not key.endswith(".pos") and (key in HEAD_JOINTS or key.startswith(("left_arm_", "right_arm_"))):
        key = f"{key}.pos"
    return key


def _validate_targets(targets: dict[str, float], obs: dict[str, Any]) -> None:
    missing = [key for key in targets if key not in obs]
    if missing:
        available = sorted(key for key in obs if key.endswith(".pos"))
        lines = "\n".join(f"  {key}" for key in missing)
        raise ValueError(f"Target joint(s) not found in robot observation:\n{lines}\n\nAvailable position keys include:\n{available}")


def _move_to_joint_targets(
    robot: Any,
    obs: dict[str, Any],
    targets: dict[str, float],
    *,
    steps: int,
    delay_s: float,
) -> None:
    starts = {key: float(obs[key]) for key in targets}
    steps = max(1, steps)
    print(f"\nMoving {len(targets)} joint(s) over {steps} steps...")
    for idx in range(1, steps + 1):
        ratio = idx / steps
        action = {
            key: start + (float(targets[key]) - start) * ratio
            for key, start in starts.items()
        }
        robot.send_action(action)
        if delay_s > 0:
            time.sleep(delay_s)
    print("Move complete.")


def _print_targets(targets: dict[str, float]) -> None:
    print("\nRequested targets:")
    for key, value in sorted(targets.items()):
        print(f"  {key}: {value:.4f}")


def _print_pose(obs: dict[str, Any], *, precision: int) -> None:
    fmt = f"{{:.{precision}f}}"
    print("\nCurrent arm pose:")
    for side in ("left", "right"):
        print(f"\n{side}:")
        for joint in ARM_JOINTS:
            key = f"{side}_arm_{joint}.pos"
            value = obs.get(key)
            print(f"  {key}: {_format(value, fmt)}")

    print("\nHead:")
    for joint in HEAD_JOINTS:
        key = f"{joint}.pos"
        value = obs.get(key)
        print(f"  {key}: {_format(value, fmt)}")

    print("\nVR basket target flags:")
    for side in ("left", "right"):
        print(f"\n{side}:")
        for joint in ARM_JOINTS:
            key = f"{side}_arm_{joint}.pos"
            value = obs.get(key)
            if value is None:
                continue
            print(f"  --vr-basket-target {key}={_format(value, fmt)} \\")


def _format(value: Any, fmt: str) -> str:
    if value is None:
        return "missing"
    try:
        return fmt.format(float(value))
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
