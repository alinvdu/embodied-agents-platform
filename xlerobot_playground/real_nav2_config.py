from __future__ import annotations

import argparse
import os
from pathlib import Path

def default_nav2_params_path() -> Path:
    ros_distro = os.environ.get("ROS_DISTRO", "humble")
    return Path(f"/opt/ros/{ros_distro}/share/nav2_bringup/params/nav2_params.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate real-XLeRobot SLAM Toolbox and Nav2 params.")
    parser.add_argument("--base-nav2-params", default=str(default_nav2_params_path()))
    parser.add_argument("--output-dir", default="artifacts/nav2")
    parser.add_argument("--slam-output", default="xlerobot_slam_toolbox.yaml")
    parser.add_argument("--nav2-output", default="xlerobot_nav2_params.yaml")
    parser.add_argument("--scan-topic", default="/scan")
    parser.add_argument("--global-map-topic", default="/projected_map")
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--odom-frame", default="odom")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--max-laser-range", type=float, default=6.0)
    parser.add_argument("--robot-length-m", type=float, default=0.3913)
    parser.add_argument("--robot-width-m", type=float, default=0.459)
    parser.add_argument("--max-linear-velocity", type=float, default=0.03)
    parser.add_argument("--max-angular-velocity", type=float, default=0.18)
    parser.add_argument(
        "--min-linear-velocity-threshold",
        type=float,
        default=0.01,
        help="Controller velocity threshold below which tiny x commands are treated as zero.",
    )
    parser.add_argument(
        "--min-angular-velocity-threshold",
        type=float,
        default=0.02,
        help="Controller velocity threshold below which tiny theta commands are treated as zero.",
    )
    parser.add_argument(
        "--min-speed-theta",
        type=float,
        default=0.02,
        help="Smallest nonzero angular speed sampled by DWB; keep above the real base deadband.",
    )
    parser.add_argument(
        "--trans-stopped-velocity",
        type=float,
        default=0.01,
        help="DWB translational stopped threshold. Keep below the real max linear velocity.",
    )
    parser.add_argument(
        "--follow-path-xy-goal-tolerance-m",
        type=float,
        default=0.10,
        help=(
            "DWB internal XY goal tolerance. Keep below PositionGoalChecker tolerance so DWB "
            "does not enter near-goal behavior before the action can succeed."
        ),
    )
    parser.add_argument(
        "--rotate-to-goal-slowing-factor",
        type=float,
        default=1.0,
        help="DWB RotateToGoal slowdown factor. Lower values reduce near-goal crawling on the real base.",
    )
    parser.add_argument(
        "--path-align-scale",
        type=float,
        default=4.0,
        help="DWB PathAlign critic scale. Low default keeps path following as a soft preference.",
    )
    parser.add_argument(
        "--goal-align-scale",
        type=float,
        default=0.0,
        help="DWB GoalAlign critic scale. Default 0 removes heading alignment for XY-only exploration goals.",
    )
    parser.add_argument(
        "--rotate-to-goal-scale",
        type=float,
        default=0.0,
        help="DWB RotateToGoal critic scale. Default disables final-yaw rotation for XY-only exploration goals.",
    )
    parser.add_argument("--oscillation-reset-dist-m", type=float, default=0.01)
    parser.add_argument("--oscillation-reset-angle-rad", type=float, default=0.05)
    parser.add_argument("--oscillation-reset-time-s", type=float, default=5.0)
    parser.add_argument(
        "--goal-dist-scale",
        type=float,
        default=8.0,
        help="DWB GoalDist critic scale. Lower values make goal distance less greedy near the waypoint.",
    )
    parser.add_argument("--local-costmap-width", type=int, default=2)
    parser.add_argument("--local-costmap-height", type=int, default=2)
    parser.add_argument("--transform-tolerance-s", type=float, default=0.5)
    parser.add_argument("--progress-required-movement-radius", type=float, default=0.01)
    parser.add_argument("--progress-movement-time-allowance-s", type=float, default=60.0)
    parser.add_argument("--xy-goal-tolerance-m", type=float, default=0.18)
    parser.add_argument("--yaw-goal-tolerance-rad", type=float, default=3.14)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from xlerobot_playground.nav2_params import (
        dump_yaml,
        load_yaml,
        patch_nav2_params,
        rectangular_footprint,
        render_slam_toolbox_params,
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    slam_params = render_slam_toolbox_params(
        use_sim_time=False,
        scan_topic=args.scan_topic,
        map_frame=args.map_frame,
        odom_frame=args.odom_frame,
        base_frame=args.base_frame,
        resolution=args.resolution,
        max_laser_range=args.max_laser_range,
    )
    slam_path = output_dir / args.slam_output
    dump_yaml(slam_path, slam_params)

    base_nav2 = load_yaml(args.base_nav2_params)
    footprint = rectangular_footprint(length_m=args.robot_length_m, width_m=args.robot_width_m)
    nav2_params = patch_nav2_params(
        base_nav2,
        use_sim_time=False,
        scan_topic=args.scan_topic,
        global_map_topic=args.global_map_topic,
        map_frame=args.map_frame,
        odom_frame=args.odom_frame,
        base_frame=args.base_frame,
        footprint=footprint,
        obstacle_max_range=args.max_laser_range * 0.95,
        raytrace_max_range=args.max_laser_range,
        max_linear_velocity=args.max_linear_velocity,
        max_angular_velocity=args.max_angular_velocity,
        min_linear_velocity_threshold=args.min_linear_velocity_threshold,
        min_angular_velocity_threshold=args.min_angular_velocity_threshold,
        min_speed_theta=args.min_speed_theta,
        trans_stopped_velocity=args.trans_stopped_velocity,
        follow_path_xy_goal_tolerance_m=args.follow_path_xy_goal_tolerance_m,
        path_align_scale=args.path_align_scale,
        goal_align_scale=args.goal_align_scale,
        oscillation_reset_dist_m=args.oscillation_reset_dist_m,
        oscillation_reset_angle_rad=args.oscillation_reset_angle_rad,
        oscillation_reset_time_s=args.oscillation_reset_time_s,
        goal_dist_scale=args.goal_dist_scale,
        rotate_to_goal_scale=args.rotate_to_goal_scale,
        rotate_to_goal_slowing_factor=args.rotate_to_goal_slowing_factor,
        local_costmap_width=args.local_costmap_width,
        local_costmap_height=args.local_costmap_height,
        transform_tolerance_s=args.transform_tolerance_s,
        progress_required_movement_radius=args.progress_required_movement_radius,
        progress_movement_time_allowance_s=args.progress_movement_time_allowance_s,
        xy_goal_tolerance_m=args.xy_goal_tolerance_m,
        yaw_goal_tolerance_rad=args.yaw_goal_tolerance_rad,
        inflation_radius=0.0,
    )
    nav2_path = output_dir / args.nav2_output
    dump_yaml(nav2_path, nav2_params)

    print(f"Wrote SLAM params: {slam_path}")
    print(f"Wrote Nav2 params: {nav2_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
