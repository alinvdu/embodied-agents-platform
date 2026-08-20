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
import time

from multido_xlerobot.bootstrap import resolve_xlerobot_repo_root
from xlerobot_playground.real_exploration_runtime import RealXLeRobotDirectRuntime, RealXLeRobotRuntimeConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Direct low-speed XLeRobot motor probe without ROS or HTTP.")
    parser.add_argument("--repo-root", default=str(resolve_xlerobot_repo_root()))
    parser.add_argument("--robot-kind", choices=("xlerobot", "xlerobot_2wheels"), default="xlerobot_2wheels")
    parser.add_argument("--port1", default="/dev/tty.usbmodem5B140330101")
    parser.add_argument("--port2", default="/dev/tty.usbmodem5B140332271")
    parser.add_argument("--use-degrees", action="store_true")
    parser.add_argument("--max-linear-m-s", type=float, default=0.03)
    parser.add_argument("--max-angular-rad-s", type=float, default=0.10)
    parser.add_argument("--linear-m-s", type=float, default=0.0)
    parser.add_argument("--angular-rad-s", type=float, default=0.0)
    parser.add_argument("--hold-s", type=float, default=0.5)
    parser.add_argument("--skip-motion", action="store_true", help="Only connect and send zero stop.")
    parser.add_argument(
        "--calibration-prompt-response",
        default="",
        help="Automatic calibration prompt response. Default empty response restores calibration from file.",
    )
    parser.add_argument("--interactive-calibration", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runtime = RealXLeRobotDirectRuntime(
        RealXLeRobotRuntimeConfig(
            repo_root=args.repo_root,
            robot_kind=args.robot_kind,
            port1=args.port1,
            port2=args.port2,
            use_degrees=args.use_degrees,
            allow_motion_commands=True,
            max_linear_m_s=args.max_linear_m_s,
            max_angular_rad_s=args.max_angular_rad_s,
            debug_motion=True,
            calibration_prompt_response=None if args.interactive_calibration else args.calibration_prompt_response,
        )
    )
    try:
        print("probe: connect", flush=True)
        runtime.connect()
        print("probe: zero velocity", flush=True)
        runtime.drive_velocity(linear_m_s=0.0, angular_rad_s=0.0)
        if not args.skip_motion:
            print(f"probe: command linear={args.linear_m_s} angular={args.angular_rad_s}", flush=True)
            runtime.drive_velocity(linear_m_s=args.linear_m_s, angular_rad_s=args.angular_rad_s)
            time.sleep(max(0.0, args.hold_s))
        print("probe: final zero velocity", flush=True)
        runtime.stop()
        print("probe: ok", flush=True)
        return 0
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
