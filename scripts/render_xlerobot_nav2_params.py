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
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xlerobot_playground.nav2_params import (
    dump_yaml,
    load_yaml,
    patch_nav2_params,
    rectangular_footprint,
    render_slam_toolbox_params,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Nav2 and slam_toolbox parameter files for the XLeRobot "
            "ManiSkill ROS bridge by patching the distro's default Nav2 params."
        )
    )
    parser.add_argument("--ros-distro", default="humble")
    parser.add_argument(
        "--nav2-default-params",
        default=None,
        help="Optional explicit path to the upstream nav2_params.yaml to patch.",
    )
    parser.add_argument(
        "--output-dir",
        default="./artifacts/nav2",
        help="Directory where the rendered YAML files will be written.",
    )
    parser.add_argument("--scan-topic", default="/scan")
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--odom-frame", default="odom")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--robot-radius", type=float, default=0.24)
    parser.add_argument("--footprint-length-m", type=float, default=0.3913)
    parser.add_argument("--footprint-width-m", type=float, default=0.459)
    parser.add_argument("--footprint-padding", type=float, default=0.0)
    parser.add_argument("--obstacle-max-range", type=float, default=9.5)
    parser.add_argument("--raytrace-max-range", type=float, default=10.0)
    parser.add_argument("--inflation-radius", type=float, default=0.0)
    parser.add_argument("--max-linear-velocity", type=float, default=0.65)
    parser.add_argument("--max-angular-velocity", type=float, default=0.45)
    parser.add_argument("--trans-stopped-velocity", type=float, default=0.05)
    parser.add_argument("--path-align-scale", type=float, default=16.0)
    parser.add_argument("--goal-align-scale", type=float, default=12.0)
    parser.add_argument("--rotate-to-goal-scale", type=float, default=8.0)
    parser.add_argument("--rotate-to-goal-slowing-factor", type=float, default=3.0)
    parser.add_argument("--slam-resolution", type=float, default=0.05)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    default_params = Path(
        args.nav2_default_params
        or f"/opt/ros/{args.ros_distro}/share/nav2_bringup/params/nav2_params.yaml"
    ).expanduser()
    if not default_params.exists():
        raise SystemExit(
            f"Could not find upstream Nav2 params at {default_params}. "
            "Install `nav2_bringup` first or pass `--nav2-default-params`."
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    footprint = rectangular_footprint(
        length_m=args.footprint_length_m,
        width_m=args.footprint_width_m,
    )

    nav2_params = patch_nav2_params(
        load_yaml(default_params),
        use_sim_time=True,
        map_frame=args.map_frame,
        odom_frame=args.odom_frame,
        base_frame=args.base_frame,
        scan_topic=args.scan_topic,
        robot_radius=args.robot_radius,
        footprint=footprint,
        footprint_padding=args.footprint_padding,
        obstacle_max_range=args.obstacle_max_range,
        raytrace_max_range=args.raytrace_max_range,
        inflation_radius=args.inflation_radius,
        max_linear_velocity=args.max_linear_velocity,
        max_angular_velocity=args.max_angular_velocity,
        trans_stopped_velocity=args.trans_stopped_velocity,
        path_align_scale=args.path_align_scale,
        goal_align_scale=args.goal_align_scale,
        rotate_to_goal_scale=args.rotate_to_goal_scale,
        rotate_to_goal_slowing_factor=args.rotate_to_goal_slowing_factor,
    )
    slam_params = render_slam_toolbox_params(
        use_sim_time=True,
        scan_topic=args.scan_topic,
        map_frame=args.map_frame,
        odom_frame=args.odom_frame,
        base_frame=args.base_frame,
        resolution=args.slam_resolution,
        max_laser_range=args.raytrace_max_range,
    )

    nav2_out = output_dir / "xlerobot_nav2_params.yaml"
    slam_out = output_dir / "xlerobot_slam_toolbox.yaml"
    dump_yaml(nav2_out, nav2_params)
    dump_yaml(slam_out, slam_params)

    print(f"Wrote {nav2_out}")
    print(f"Wrote {slam_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
