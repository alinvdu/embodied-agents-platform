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
from xlerobot_playground.nav2_defaults import default_nav2_behavior_tree


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the XLeRobot exploration playground in either teleport-debug mode or "
            "the live Nav2-shaped runtime with the review UI."
        )
    )
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--sim-python-bin", default=None)
    parser.add_argument("--movement-mode", choices=("teleport", "simulated", "ros"), default=None)
    parser.add_argument("--ui-flavor", choices=("user", "developer"), default="user")
    parser.add_argument("--persist-path", default="./artifacts/xlerobot_exploration_map.json")
    parser.add_argument("--area", default="workspace")
    parser.add_argument("--session", default="house_v1")
    parser.add_argument("--source", default="operator")
    parser.add_argument("--env-id", default="SceneManipulation-v1")
    parser.add_argument("--robot-uid", default="xlerobot")
    parser.add_argument("--control-mode", default="pd_joint_delta_pos_dual_arm")
    parser.add_argument("--render-mode", default="human")
    parser.add_argument("--shader", default="default")
    parser.add_argument("--sim-backend", default="auto")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--force-reload", action="store_true")
    parser.add_argument("--occupancy-resolution", type=float, default=0.25)
    parser.add_argument("--max-control-steps", type=int, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--show-cameras", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-rerun", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--camera-log-stride", type=int, default=2)
    parser.add_argument("--realtime-sleep-s", type=float, default=0.01)
    parser.add_argument("--explorer-policy", choices=("heuristic", "llm"), default="llm")
    parser.add_argument("--llm-provider", default="mock")
    parser.add_argument("--llm-model", default="mock")
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--llm-temperature", type=float, default=0.1)
    parser.add_argument("--llm-max-tokens", type=int, default=1200)
    parser.add_argument("--llm-reasoning-effort", default=None)
    parser.add_argument("--semantic-waypoints-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--automatic-semantic-waypoints", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--semantic-llm-provider", default=None)
    parser.add_argument("--semantic-llm-model", default=None)
    parser.add_argument("--semantic-llm-base-url", default=None)
    parser.add_argument("--semantic-llm-api-key", default=None)
    parser.add_argument("--semantic-vlm-async", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trace-policy-stdout", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trace-llm-stdout", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--serve-review-ui", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--review-host", default="127.0.0.1")
    parser.add_argument("--review-port", type=int, default=8770)
    parser.add_argument("--open-browser", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--display-yaw-offset-deg", type=float, default=None)
    parser.add_argument("--build-config-idx", type=int, default=None)
    parser.add_argument("--spawn-x", type=float, default=None)
    parser.add_argument("--spawn-y", type=float, default=None)
    parser.add_argument("--spawn-yaw", type=float, default=0.0)
    parser.add_argument("--spawn-facing", choices=("front", "left", "right", "back"), default="front")
    parser.add_argument("--scan-mode", choices=("turnaround", "front_only"), default="turnaround")
    parser.add_argument("--scan-yaw-samples", type=int, default=12)
    parser.add_argument("--depth-beam-stride", type=int, default=2)
    parser.add_argument("--teleport-z", type=float, default=None)
    parser.add_argument("--teleport-settle-steps", type=int, default=1)
    parser.add_argument("--max-frontiers", type=int, default=12)
    parser.add_argument("--sensor-range-m", type=float, default=10.0)
    parser.add_argument("--finish-coverage-threshold", type=float, default=0.96)
    parser.add_argument("--max-decisions", type=int, default=32)
    parser.add_argument("--nav2-mode", choices=("simulated", "ros"), default="simulated")
    parser.add_argument("--nav2-planner-id", default="GridBased")
    parser.add_argument("--nav2-controller-id", default="FollowPath")
    parser.add_argument("--nav2-behavior-tree", default=default_nav2_behavior_tree())
    parser.add_argument("--nav2-recovery-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ros-navigation-map-source", choices=("fused_scan", "external"), default="fused_scan")
    parser.add_argument("--ros-map-topic", default="/map")
    parser.add_argument("--ros-scan-topic", default="/scan")
    parser.add_argument("--ros-rgb-topic", default="/camera/head/image_raw")
    parser.add_argument("--ros-cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--ros-map-frame", default="map")
    parser.add_argument("--ros-adapter-url", default=None)
    parser.add_argument("--ros-adapter-timeout-s", type=float, default=30.0)
    parser.add_argument(
        "--ros-odom-frame",
        default="odom",
        help=(
            "Accepted for ROS launch command compatibility. The playground reads the robot pose from TF "
            "between --ros-map-frame and --ros-base-frame; odom frame configuration lives in the bridge/Nav2 launch."
        ),
    )
    parser.add_argument("--ros-base-frame", default="base_link")
    parser.add_argument("--ros-server-timeout-s", type=float, default=10.0)
    parser.add_argument("--ros-ready-timeout-s", type=float, default=20.0)
    parser.add_argument("--ros-turn-scan-timeout-s", type=float, default=45.0)
    parser.add_argument("--ros-turn-scan-settle-s", type=float, default=1.0)
    parser.add_argument("--ros-manual-spin-angular-speed-rad-s", type=float, default=0.25)
    parser.add_argument("--ros-manual-spin-publish-hz", type=float, default=20.0)
    parser.add_argument("--sim-motion-speed", choices=("normal", "faster", "fastest"), default="normal")
    parser.add_argument("--ros-allow-multiple-action-servers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--use-keyboard-controls",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable arrow key controls for manual robot teleoperation.",
    )
    parser.add_argument(
        "--keyboard-speed",
        choices=("slow", "normal", "fast"),
        default="normal",
        help="Speed profile for keyboard movement.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = resolve_repo_root(args.repo_root)
    python_bin = args.sim_python_bin or default_sim_python_bin(REPO_ROOT)
    movement_mode = args.movement_mode or args.nav2_mode
    if movement_mode in {"teleport", "ros"}:
        interactive_args = [
            "--backend",
            "ros" if movement_mode == "ros" else "maniskill",
            "--repo-root",
            str(repo_root),
            "--session",
            args.session,
            "--area",
            args.area,
            "--source",
            args.source,
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
            "--spawn-yaw",
            str(args.spawn_yaw),
            "--spawn-facing",
            args.spawn_facing,
            "--scan-mode",
            args.scan_mode,
            "--scan-yaw-samples",
            str(args.scan_yaw_samples),
            "--depth-beam-stride",
            str(args.depth_beam_stride),
            "--teleport-settle-steps",
            str(args.teleport_settle_steps),
            "--max-frontiers",
            str(args.max_frontiers),
            "--occupancy-resolution",
            str(args.occupancy_resolution),
            "--sensor-range-m",
            str(args.sensor_range_m),
            "--finish-coverage-threshold",
            str(args.finish_coverage_threshold),
            "--max-decisions",
            str(args.max_decisions),
            "--explorer-policy",
            args.explorer_policy,
            "--llm-provider",
            args.llm_provider,
            "--llm-model",
            args.llm_model,
            "--llm-temperature",
            str(args.llm_temperature),
            "--llm-max-tokens",
            str(args.llm_max_tokens),
            "--host",
            args.review_host,
            "--port",
            str(args.review_port),
            f"--{'open-browser' if args.open_browser else 'no-open-browser'}",
            "--ui-flavor",
            args.ui_flavor,
            f"--{'semantic-waypoints-enabled' if args.semantic_waypoints_enabled else 'no-semantic-waypoints-enabled'}",
            f"--{'automatic-semantic-waypoints' if args.automatic_semantic_waypoints else 'no-automatic-semantic-waypoints'}",
            f"--{'semantic-vlm-async' if args.semantic_vlm_async else 'no-semantic-vlm-async'}",
            "--nav2-planner-id",
            args.nav2_planner_id,
            "--nav2-controller-id",
            args.nav2_controller_id,
            "--nav2-behavior-tree",
            args.nav2_behavior_tree,
            f"--{'nav2-recovery-enabled' if args.nav2_recovery_enabled else 'no-nav2-recovery-enabled'}",
            "--ros-navigation-map-source",
            args.ros_navigation_map_source,
            "--ros-map-topic",
            args.ros_map_topic,
            "--ros-scan-topic",
            args.ros_scan_topic,
            "--ros-rgb-topic",
            args.ros_rgb_topic,
            "--ros-cmd-vel-topic",
            args.ros_cmd_vel_topic,
            "--ros-map-frame",
            args.ros_map_frame,
            "--ros-adapter-timeout-s",
            str(args.ros_adapter_timeout_s),
            "--ros-odom-frame",
            args.ros_odom_frame,
            "--ros-base-frame",
            args.ros_base_frame,
            "--ros-server-timeout-s",
            str(args.ros_server_timeout_s),
            "--ros-ready-timeout-s",
            str(args.ros_ready_timeout_s),
            "--ros-turn-scan-timeout-s",
            str(args.ros_turn_scan_timeout_s),
            "--ros-turn-scan-settle-s",
            str(args.ros_turn_scan_settle_s),
            "--ros-manual-spin-angular-speed-rad-s",
            str(args.ros_manual_spin_angular_speed_rad_s),
            "--ros-manual-spin-publish-hz",
            str(args.ros_manual_spin_publish_hz),
            "--sim-motion-speed",
            args.sim_motion_speed,
            f"--{'ros-allow-multiple-action-servers' if args.ros_allow_multiple_action_servers else 'no-ros-allow-multiple-action-servers'}",
            f"--{'use-keyboard-controls' if args.use_keyboard_controls else 'no-use-keyboard-controls'}",
            "--keyboard-speed",
            args.keyboard_speed,
        ]
        if args.display_yaw_offset_deg is not None:
            interactive_args.extend(["--display-yaw-offset-deg", str(args.display_yaw_offset_deg)])
        if args.build_config_idx is not None:
            interactive_args.extend(["--build-config-idx", str(args.build_config_idx)])
        if args.spawn_x is not None:
            interactive_args.extend(["--spawn-x", str(args.spawn_x)])
        if args.spawn_y is not None:
            interactive_args.extend(["--spawn-y", str(args.spawn_y)])
        if args.teleport_z is not None:
            interactive_args.extend(["--teleport-z", str(args.teleport_z)])
        if args.ros_adapter_url:
            interactive_args.extend(["--ros-adapter-url", args.ros_adapter_url])
        if args.llm_base_url:
            interactive_args.extend(["--llm-base-url", args.llm_base_url])
        if args.llm_api_key:
            interactive_args.extend(["--llm-api-key", args.llm_api_key])
        if args.llm_reasoning_effort:
            interactive_args.extend(["--llm-reasoning-effort", args.llm_reasoning_effort])
        if args.semantic_llm_provider:
            interactive_args.extend(["--semantic-llm-provider", args.semantic_llm_provider])
        if args.semantic_llm_model:
            interactive_args.extend(["--semantic-llm-model", args.semantic_llm_model])
        if args.semantic_llm_base_url:
            interactive_args.extend(["--semantic-llm-base-url", args.semantic_llm_base_url])
        if args.semantic_llm_api_key:
            interactive_args.extend(["--semantic-llm-api-key", args.semantic_llm_api_key])
        if args.force_reload:
            interactive_args.append("--force-reload")
        return exec_python_module(
            "xlerobot_playground.interactive_exploration_playground",
            python_bin=python_bin,
            argv=interactive_args,
            cwd=REPO_ROOT,
        )
    backend_args = [
        "--repo-root",
        str(repo_root),
        "--persist-path",
        args.persist_path,
        "--area",
        args.area,
        "--session",
        args.session,
        "--source",
        args.source,
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
        "--occupancy-resolution",
        str(args.occupancy_resolution),
        f"--{'semantic-waypoints-enabled' if args.semantic_waypoints_enabled else 'no-semantic-waypoints-enabled'}",
        f"--{'automatic-semantic-waypoints' if args.automatic_semantic_waypoints else 'no-automatic-semantic-waypoints'}",
        f"--{'semantic-vlm-async' if args.semantic_vlm_async else 'no-semantic-vlm-async'}",
    ]
    if args.max_control_steps is not None:
        backend_args.extend([
            "--max-control-steps",
            str(args.max_control_steps),
        ])
    if args.max_episode_steps is not None:
        backend_args.extend([
            "--max-episode-steps",
            str(args.max_episode_steps),
        ])
    backend_args.extend([
        f"--{'show-cameras' if args.show_cameras else 'no-show-cameras'}",
        f"--{'use-rerun' if args.use_rerun else 'no-use-rerun'}",
        "--camera-log-stride",
        str(args.camera_log_stride),
    ])
    backend_args.extend([
        "--realtime-sleep-s",
        str(args.realtime_sleep_s),
        "--explorer-policy",
        args.explorer_policy,
        "--llm-provider",
        args.llm_provider,
        "--llm-model",
        args.llm_model,
        "--llm-temperature",
        str(args.llm_temperature),
        "--llm-max-tokens",
        str(args.llm_max_tokens),
        "--sensor-range-m",
        str(args.sensor_range_m),
        "--finish-coverage-threshold",
        str(args.finish_coverage_threshold),
        "--max-decisions",
        str(args.max_decisions),
        "--nav2-mode",
        movement_mode,
        "--nav2-planner-id",
        args.nav2_planner_id,
        "--nav2-controller-id",
        args.nav2_controller_id,
        "--nav2-behavior-tree",
        args.nav2_behavior_tree,
        "--review-host",
        args.review_host,
        "--review-port",
        str(args.review_port),
        "--review-ui-flavor",
        args.ui_flavor,
        f"--{'open-browser' if args.open_browser else 'no-open-browser'}",
        "--ros-map-topic",
        args.ros_map_topic,
        "--ros-scan-topic",
        args.ros_scan_topic,
        "--ros-rgb-topic",
        args.ros_rgb_topic,
        "--ros-cmd-vel-topic",
        args.ros_cmd_vel_topic,
        "--ros-map-frame",
        args.ros_map_frame,
        "--ros-adapter-timeout-s",
        str(args.ros_adapter_timeout_s),
        "--ros-odom-frame",
        args.ros_odom_frame,
        "--ros-base-frame",
        args.ros_base_frame,
        "--ros-server-timeout-s",
        str(args.ros_server_timeout_s),
        "--ros-ready-timeout-s",
        str(args.ros_ready_timeout_s),
        "--ros-turn-scan-timeout-s",
        str(args.ros_turn_scan_timeout_s),
        "--ros-turn-scan-settle-s",
        str(args.ros_turn_scan_settle_s),
        "--ros-manual-spin-angular-speed-rad-s",
        str(args.ros_manual_spin_angular_speed_rad_s),
        "--ros-manual-spin-publish-hz",
        str(args.ros_manual_spin_publish_hz),
        "--sim-motion-speed",
        args.sim_motion_speed,
    ])
    backend_args.append(f"--{'nav2-recovery-enabled' if args.nav2_recovery_enabled else 'no-nav2-recovery-enabled'}")
    backend_args.append(
        f"--{'ros-allow-multiple-action-servers' if args.ros_allow_multiple_action_servers else 'no-ros-allow-multiple-action-servers'}"
    )
    if args.ros_adapter_url:
        backend_args.extend(["--ros-adapter-url", args.ros_adapter_url])
    if args.llm_base_url:
        backend_args.extend(["--llm-base-url", args.llm_base_url])
    if args.llm_api_key:
        backend_args.extend(["--llm-api-key", args.llm_api_key])
    if args.llm_reasoning_effort:
        backend_args.extend(["--llm-reasoning-effort", args.llm_reasoning_effort])
    if args.semantic_llm_provider:
        backend_args.extend(["--semantic-llm-provider", args.semantic_llm_provider])
    if args.semantic_llm_model:
        backend_args.extend(["--semantic-llm-model", args.semantic_llm_model])
    if args.semantic_llm_base_url:
        backend_args.extend(["--semantic-llm-base-url", args.semantic_llm_base_url])
    if args.semantic_llm_api_key:
        backend_args.extend(["--semantic-llm-api-key", args.semantic_llm_api_key])
    backend_args.extend([
        f"--{'trace-policy-stdout' if args.trace_policy_stdout else 'no-trace-policy-stdout'}",
        f"--{'trace-llm-stdout' if args.trace_llm_stdout else 'no-trace-llm-stdout'}",
    ])
    if args.force_reload:
        backend_args.append("--force-reload")
    if args.serve_review_ui:
        backend_args.append("--serve-review-ui")
    return exec_python_module(
        "xlerobot_playground.sim_exploration_backend",
        python_bin=python_bin,
        argv=backend_args,
        cwd=REPO_ROOT,
    )


if __name__ == "__main__":
    raise SystemExit(main())
