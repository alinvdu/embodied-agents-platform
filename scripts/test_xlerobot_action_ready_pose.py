#!/usr/bin/env python3
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
    _connect_robot,
    _get_robot_observation,
    _move_to_joint_targets,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Test the staged NAV_STOW -> ACTION_READY pose without starting VR, "
            "then return to NAV_STOW after inspection."
        )
    )
    parser.add_argument("--repo-root", default=str(resolve_xlerobot_repo_root()))
    parser.add_argument(
        "--robot-kind",
        choices=("xlerobot", "xlerobot_2wheels"),
        default="xlerobot_2wheels",
    )
    parser.add_argument("--port1", default="/dev/tty.usbmodem5B140330101")
    parser.add_argument("--port2", default="/dev/tty.usbmodem5B140332271")
    parser.add_argument("--use-degrees", action="store_true")
    parser.add_argument("--manual-calibration-prompt", action="store_true")
    parser.add_argument(
        "--shoulder-delta",
        type=float,
        default=_ACTION_READY_SHOULDER_DELTA,
        help="Shoulder lift offset from NAV_STOW. Smaller values move ACTION_READY farther back.",
    )
    parser.add_argument(
        "--elbow-delta",
        type=float,
        default=_ACTION_READY_ELBOW_DELTA,
    )
    parser.add_argument(
        "--wrist-delta",
        type=float,
        default=_ACTION_READY_WRIST_DELTA,
    )
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--delay-s", type=float, default=0.02)
    parser.add_argument(
        "--side",
        choices=("left", "right", "both"),
        default="both",
        help="Arm side to test. Recording and inference normally use both.",
    )
    parser.add_argument(
        "--leave-in-action-ready",
        action="store_true",
        help="Do not return the tested arm(s) to NAV_STOW before disconnecting.",
    )
    parser.add_argument(
        "--skip-confirmation",
        action="store_true",
        help="Start the NAV_STOW and ACTION_READY motions without the initial confirmation prompts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the candidate pose without connecting to or moving the robot.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    sides = ("left", "right") if args.side == "both" else (args.side,)
    stow_targets = _arm_pose_targets(sides, _NAV_STOW_ARM_POSE)
    action_ready_targets = _action_ready_pose(
        sides,
        elbow_delta=args.elbow_delta,
        shoulder_delta=args.shoulder_delta,
        wrist_delta=args.wrist_delta,
    )
    _print_candidate(action_ready_targets)
    if args.dry_run:
        return 0

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
        if not args.skip_confirmation:
            input(
                "\nKeep the e-stop within reach. Press ENTER to move the selected arm(s) "
                "to NAV_STOW, or Ctrl+C to cancel: "
            )
        _move_to_joint_targets(
            robot,
            stow_targets,
            steps=max(1, args.steps),
            delay_s=max(0.0, args.delay_s),
        )

        if not args.skip_confirmation:
            input(
                "NAV_STOW reached. Press ENTER to move to the candidate ACTION_READY pose, "
                "or Ctrl+C to cancel: "
            )
        _move_action_ready_stages(
            robot,
            action_ready_targets,
            steps=max(1, args.steps),
            delay_s=max(0.0, args.delay_s),
        )
        observed = _get_robot_observation(robot, use_camera=False)
        _print_observed(observed, sides)

        if args.leave_in_action_ready:
            print("\nLeaving the selected arm(s) in ACTION_READY.")
            return 0

        input(
            "\nInspect the pose. Press ENTER to return the selected arm(s) to NAV_STOW: "
        )
        _move_nav_stow_stages(
            robot,
            stow_targets,
            steps=max(1, args.steps),
            delay_s=max(0.0, args.delay_s),
        )
        print("NAV_STOW restored.")
    finally:
        robot.disconnect()
    return 0


def _arm_pose_targets(
    sides: tuple[str, ...],
    pose: dict[str, float],
) -> dict[str, float]:
    return {
        f"{side}_arm_{joint}.pos": float(value)
        for side in sides
        for joint, value in pose.items()
    }


def _action_ready_pose(
    sides: tuple[str, ...],
    *,
    elbow_delta: float,
    shoulder_delta: float,
    wrist_delta: float,
) -> dict[str, float]:
    pose = dict(_NAV_STOW_ARM_POSE)
    pose["elbow_flex"] += float(elbow_delta)
    pose["shoulder_lift"] += float(shoulder_delta)
    pose["wrist_flex"] += float(wrist_delta)
    return _arm_pose_targets(sides, pose)


def _move_action_ready_stages(
    robot: Any,
    targets: dict[str, float],
    *,
    steps: int,
    delay_s: float,
) -> None:
    elbow_wrist = {
        key: value
        for key, value in targets.items()
        if key.endswith(("elbow_flex.pos", "wrist_flex.pos"))
    }
    shoulders = {
        key: value
        for key, value in targets.items()
        if key.endswith("shoulder_lift.pos")
    }
    _move_to_joint_targets(robot, elbow_wrist, steps=steps, delay_s=delay_s)
    _move_to_joint_targets(robot, shoulders, steps=steps, delay_s=delay_s)


def _move_nav_stow_stages(
    robot: Any,
    targets: dict[str, float],
    *,
    steps: int,
    delay_s: float,
) -> None:
    shoulders = {
        key: value
        for key, value in targets.items()
        if key.endswith("shoulder_lift.pos")
    }
    elbow_wrist = {
        key: value
        for key, value in targets.items()
        if key.endswith(("elbow_flex.pos", "wrist_flex.pos"))
    }
    _move_to_joint_targets(robot, shoulders, steps=steps, delay_s=delay_s)
    _move_to_joint_targets(robot, elbow_wrist, steps=steps, delay_s=delay_s)
    _move_to_joint_targets(robot, targets, steps=steps, delay_s=delay_s)


def _print_candidate(targets: dict[str, float]) -> None:
    print("Candidate ACTION_READY targets:")
    for key, value in sorted(targets.items()):
        print(f"  {key}: {value:.4f}")


def _print_observed(observation: dict[str, Any], sides: tuple[str, ...]) -> None:
    print("\nObserved ACTION_READY pose:")
    for side in sides:
        print(f"  {side}:")
        for joint in _NAV_STOW_ARM_POSE:
            key = f"{side}_arm_{joint}.pos"
            if key in observation:
                print(f"    {key}: {float(observation[key]):.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
