#!/usr/bin/env python3
from __future__ import annotations

import argparse
import builtins
import sys
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
        description="Read the current XLeRobot joint pose and print reusable VR NAV_STOW flags."
    )
    parser.add_argument("--repo-root", default=str(resolve_xlerobot_repo_root()))
    parser.add_argument("--robot-kind", choices=("xlerobot", "xlerobot_2wheels"), default="xlerobot_2wheels")
    parser.add_argument("--port1", default="/dev/tty.usbmodem5B140330101")
    parser.add_argument("--port2", default="/dev/tty.usbmodem5B140332271")
    parser.add_argument("--use-degrees", action="store_true")
    parser.add_argument("--manual-calibration-prompt", action="store_true")
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

    print("\nNAV_STOW flags:")
    for side in ("left", "right"):
        for joint in ARM_JOINTS:
            key = f"{side}_arm_{joint}.pos"
            value = obs.get(key)
            if value is None:
                continue
            flag = f"--vr-nav-stow-{side}-{joint.replace('_', '-')}"
            print(f"  {flag} {_format(value, fmt)} \\")


def _format(value: Any, fmt: str) -> str:
    if value is None:
        return "missing"
    try:
        return fmt.format(float(value))
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
