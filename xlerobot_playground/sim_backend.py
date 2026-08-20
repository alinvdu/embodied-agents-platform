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

from multido_xlerobot.bootstrap import resolve_xlerobot_repo_root
from multido_xlerobot.maniskill import run_keyboard_play_demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backend launcher for XLeRobot ManiSkill manipulation and recording playgrounds."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    manipulate = subparsers.add_parser("manipulate", help="Launch sim teleoperation.")
    _add_shared_sim_args(manipulate)
    manipulate.add_argument("--controller", choices=("keyboard", "vr"), default="keyboard")

    record = subparsers.add_parser("record", help="Launch sim teleop dataset recording.")
    _add_shared_sim_args(record)
    record.add_argument("--controller", choices=("keyboard", "vr"), default="keyboard")
    record.add_argument("--dataset-name", default="xlerobot_sim_playground")
    record.add_argument("--output-dir", default="./datasets")
    record.add_argument("--num-episodes", type=int, default=10)
    record.add_argument("--episode-length", type=int, default=1000)
    record.add_argument("--fps", type=int, default=30)
    record.add_argument("--task-description", default="XLeRobot sim teleoperation")
    return parser


def _add_shared_sim_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", default=str(resolve_xlerobot_repo_root()))
    parser.add_argument("--env-id", default="SceneManipulation-v1")
    parser.add_argument("--robot-uid", default="xlerobot")
    parser.add_argument("--control-mode", default=None)
    parser.add_argument("--speed-profile", choices=("normal", "fast"), default="normal")
    parser.add_argument("--obs-mode", default=None)
    parser.add_argument("--render-mode", default="human")
    parser.add_argument("--shader", default="default")
    parser.add_argument("--sim-backend", default="auto")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--force-reload", action="store_true")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.mode == "manipulate":
        demo = "camera_rerun" if args.controller == "keyboard" else "vr"
        run_keyboard_play_demo(
            repo_root=args.repo_root,
            demo=demo,
            env_id=args.env_id,
            robot_uid=args.robot_uid,
            control_mode=args.control_mode,
            obs_mode=args.obs_mode,
            render_mode=args.render_mode,
            shader=args.shader,
            sim_backend=args.sim_backend,
            num_envs=args.num_envs,
            show_cameras=True,
            use_rerun=True,
            speed_profile=args.speed_profile,
            force_reload=args.force_reload,
        )
        return 0

    if args.controller == "vr":
        raise SystemExit(
            "sim+vr dataset recording is not wired yet in the upstream XLeRobot demo set. "
            "Use `manipulate --controller vr` for VR teleop, or `record --controller keyboard` for dataset capture."
        )

    run_keyboard_play_demo(
        repo_root=args.repo_root,
        demo="record_dataset",
        env_id=args.env_id,
        robot_uid=args.robot_uid,
        control_mode=args.control_mode,
        obs_mode=args.obs_mode or "sensor_data",
        render_mode=args.render_mode,
        shader=args.shader,
        sim_backend=args.sim_backend,
        num_envs=args.num_envs,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        num_episodes=args.num_episodes,
        episode_length=args.episode_length,
        fps=args.fps,
        task_description=args.task_description,
        speed_profile=args.speed_profile,
        force_reload=args.force_reload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
