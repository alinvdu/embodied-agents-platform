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
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xlerobot_playground.launcher import default_sim_python_bin, exec_python_module, resolve_repo_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Launch the ManiSkill-to-ROS bridge so slam_toolbox and Nav2 can "
            "consume the XLeRobot simulator directly."
        )
    )
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--sim-python-bin", default=None)
    parser.add_argument("--env-id", default="SceneManipulation-v1")
    parser.add_argument("--robot-uid", default="xlerobot")
    parser.add_argument("--control-mode", default="pd_joint_delta_pos_dual_arm")
    parser.add_argument("--render-mode", default="rgb_array")
    parser.add_argument("--shader", default="default")
    parser.add_argument("--sim-backend", default="auto")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--force-reload", action="store_true")
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--scan-topic", default="/scan")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--odom-frame", default="odom")
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--head-camera-frame", default="head_camera_link")
    parser.add_argument("--head-laser-frame", default="head_laser")
    parser.add_argument("--publish-head-camera", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cmd-vel-timeout-s", type=float, default=0.5)
    parser.add_argument("--linear-cmd-gain", type=float, default=None)
    parser.add_argument("--angular-cmd-gain", type=float, default=None)
    parser.add_argument("--laser-min-range-m", type=float, default=0.05)
    parser.add_argument("--laser-max-range-m", type=float, default=10.0)
    parser.add_argument("--scan-band-height-px", type=int, default=12)
    parser.add_argument("--laser-fill-no-return", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--build-config-idx", type=int, default=None)
    parser.add_argument("--spawn-x", type=float, default=None)
    parser.add_argument("--spawn-y", type=float, default=None)
    parser.add_argument("--spawn-yaw", type=float, default=0.0)
    parser.add_argument("--ros-base-yaw-offset-rad", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--realtime-factor", type=float, default=1.0)
    parser.add_argument("--auto-reset", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = resolve_repo_root(args.repo_root)
    python_bin = args.sim_python_bin or default_sim_python_bin(REPO_ROOT)

    backend_args = [
        "--repo-root",
        str(repo_root),
        "--env-id",
        args.env_id,
        "--robot-uid",
        args.robot_uid,
        "--control-mode",
        args.control_mode,
        "--render-mode",
        args.render_mode,
        "--shader",
        args.shader,
        "--sim-backend",
        args.sim_backend,
        "--num-envs",
        str(args.num_envs),
        "--cmd-vel-topic",
        args.cmd_vel_topic,
        "--odom-topic",
        args.odom_topic,
        "--scan-topic",
        args.scan_topic,
        "--base-frame",
        args.base_frame,
        "--odom-frame",
        args.odom_frame,
        "--map-frame",
        args.map_frame,
        "--head-camera-frame",
        args.head_camera_frame,
        "--head-laser-frame",
        args.head_laser_frame,
        "--cmd-vel-timeout-s",
        str(args.cmd_vel_timeout_s),
        "--laser-min-range-m",
        str(args.laser_min_range_m),
        "--laser-max-range-m",
        str(args.laser_max_range_m),
        "--scan-band-height-px",
        str(args.scan_band_height_px),
        f"--{'laser-fill-no-return' if args.laser_fill_no_return else 'no-laser-fill-no-return'}",
        "--realtime-factor",
        str(args.realtime_factor),
        f"--{'publish-head-camera' if args.publish_head_camera else 'no-publish-head-camera'}",
        f"--{'auto-reset' if args.auto_reset else 'no-auto-reset'}",
    ]
    if args.build_config_idx is not None:
        backend_args.extend(["--build-config-idx", str(args.build_config_idx)])
    if args.spawn_x is not None:
        backend_args.extend(["--spawn-x", str(args.spawn_x)])
    if args.spawn_y is not None:
        backend_args.extend(["--spawn-y", str(args.spawn_y)])
    if args.spawn_x is not None or args.spawn_y is not None or args.spawn_yaw != 0.0:
        backend_args.extend(["--spawn-yaw", str(args.spawn_yaw)])
    if args.ros_base_yaw_offset_rad is not None:
        backend_args.extend(["--ros-base-yaw-offset-rad", str(args.ros_base_yaw_offset_rad)])
    if args.linear_cmd_gain is not None:
        backend_args.extend(["--linear-cmd-gain", str(args.linear_cmd_gain)])
    if args.angular_cmd_gain is not None:
        backend_args.extend(["--angular-cmd-gain", str(args.angular_cmd_gain)])
    if args.force_reload:
        backend_args.append("--force-reload")
    if args.max_steps is not None:
        backend_args.extend(["--max-steps", str(args.max_steps)])
    if args.max_episode_steps is not None:
        backend_args.extend(["--max-episode-steps", str(args.max_episode_steps)])

    return exec_python_module(
        "xlerobot_playground.maniskill_ros_bridge",
        python_bin=python_bin,
        argv=backend_args,
        cwd=REPO_ROOT,
    )


if __name__ == "__main__":
    raise SystemExit(main())
