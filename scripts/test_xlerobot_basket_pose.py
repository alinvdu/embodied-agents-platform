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
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multido_xlerobot import XLeRobotInterface
from multido_xlerobot.bootstrap import resolve_xlerobot_repo_root
from xlerobot_playground.real_backend import (
    _ACTION_READY_ELBOW_DELTA,
    _ACTION_READY_SHOULDER_DELTA,
    _ACTION_READY_WRIST_DELTA,
    _NAV_STOW_ARM_POSE,
    _RIGHT_BASKET_PATH_CLEARANCE,
    _RIGHT_BASKET_PATH_OVER_BASKET,
    _VR_BASKET_POSE_DEFAULTS,
    _connect_robot,
    _get_robot_observation,
    _move_to_joint_targets,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Move the right arm through the captured safe path to the basket placement pose, "
            "hold for inspection, then reverse the path and stow."
        )
    )
    parser.add_argument("--repo-root", default=str(resolve_xlerobot_repo_root()))
    parser.add_argument("--robot-kind", choices=("xlerobot", "xlerobot_2wheels"), default="xlerobot_2wheels")
    parser.add_argument("--port1", default="/dev/tty.usbmodem5B140330101")
    parser.add_argument("--port2", default="/dev/tty.usbmodem5B140332271")
    parser.add_argument("--use-degrees", action="store_true")
    parser.add_argument("--manual-calibration-prompt", action="store_true")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--delay-s", type=float, default=0.02)
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

    stow = _right_targets(_NAV_STOW_ARM_POSE)
    action_ready_pose = dict(_NAV_STOW_ARM_POSE)
    action_ready_pose["shoulder_lift"] += _ACTION_READY_SHOULDER_DELTA
    action_ready_pose["elbow_flex"] += _ACTION_READY_ELBOW_DELTA
    action_ready_pose["wrist_flex"] += _ACTION_READY_WRIST_DELTA
    action_ready = _right_targets(action_ready_pose)
    clearance = dict(_RIGHT_BASKET_PATH_CLEARANCE)
    over_basket = dict(_RIGHT_BASKET_PATH_OVER_BASKET)
    basket = {
        key: float(value)
        for key, value in _VR_BASKET_POSE_DEFAULTS.items()
        if key.startswith("right_arm_")
    }

    _print_pose("Final basket placement target", basket)
    _connect_robot(robot, auto_restore_calibration=not args.manual_calibration_prompt)
    try:
        input("\nKeep the e-stop within reach. Press ENTER to move the right arm to NAV_STOW: ")
        _move(robot, stow, args)

        input("NAV_STOW reached. Press ENTER to move to ACTION_READY: ")
        _move_action_ready(robot, action_ready, args)

        input("ACTION_READY reached. Press ENTER to follow the captured path to the basket: ")
        for label, targets in (
            ("clearance", clearance),
            ("over basket", over_basket),
            ("basket placement", basket),
        ):
            print(f"Moving to {label}.")
            _move(robot, targets, args)

        observed = _get_robot_observation(robot, use_camera=False)
        _print_pose("Observed basket placement pose", observed)
        _disable_right_gripper_torque(robot)
        input(
            "\nThe arm is holding the basket placement pose, but the right gripper is torque-free. "
            "Open and close it by hand to test objects. Remove the object, then press ENTER to "
            "re-enable the gripper at its current position and return to NAV_STOW: "
        )
        _enable_right_gripper_at_current_position(robot)

        for label, targets in (
            ("over basket", over_basket),
            ("clearance", clearance),
            ("ACTION_READY", _without_gripper(action_ready)),
        ):
            print(f"Returning through {label}.")
            _move(robot, targets, args)
        _move_nav_stow(robot, _without_gripper(stow), args)
        print("NAV_STOW restored.")
    finally:
        robot.disconnect()
    return 0


def _right_targets(pose: dict[str, float]) -> dict[str, float]:
    return {f"right_arm_{joint}.pos": float(value) for joint, value in pose.items()}


def _without_gripper(targets: dict[str, float]) -> dict[str, float]:
    return {key: value for key, value in targets.items() if not key.endswith("_gripper.pos")}


def _disable_right_gripper_torque(robot: Any) -> None:
    robot.bus2.disable_torque("right_arm_gripper", num_retry=5)
    print("Right gripper torque disabled. Shoulder, elbow, and wrist torque remain enabled.")


def _enable_right_gripper_at_current_position(robot: Any) -> None:
    motor = "right_arm_gripper"
    current_position = float(robot.bus2.read("Present_Position", motor, num_retry=5))
    robot.bus2.write("Goal_Position", motor, current_position, num_retry=5)
    robot.bus2.enable_torque(motor, num_retry=5)
    print(f"Right gripper torque re-enabled at its current position ({current_position:.4f}).")


def _move(robot: Any, targets: dict[str, float], args: argparse.Namespace) -> None:
    _move_to_joint_targets(
        robot,
        targets,
        steps=max(1, args.steps),
        delay_s=max(0.0, args.delay_s),
    )


def _move_action_ready(robot: Any, targets: dict[str, float], args: argparse.Namespace) -> None:
    _move(
        robot,
        {
            key: value
            for key, value in targets.items()
            if key.endswith(("elbow_flex.pos", "wrist_flex.pos"))
        },
        args,
    )
    _move(
        robot,
        {key: value for key, value in targets.items() if key.endswith("shoulder_lift.pos")},
        args,
    )
    _move(robot, targets, args)


def _move_nav_stow(robot: Any, targets: dict[str, float], args: argparse.Namespace) -> None:
    _move(
        robot,
        {key: value for key, value in targets.items() if key.endswith("shoulder_lift.pos")},
        args,
    )
    _move(
        robot,
        {
            key: value
            for key, value in targets.items()
            if key.endswith(("elbow_flex.pos", "wrist_flex.pos"))
        },
        args,
    )
    _move(robot, targets, args)


def _print_pose(label: str, values: dict[str, Any]) -> None:
    print(f"\n{label}:")
    for key in sorted(key for key in values if key.startswith("right_arm_") and key.endswith(".pos")):
        print(f"  {key}: {float(values[key]):.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
