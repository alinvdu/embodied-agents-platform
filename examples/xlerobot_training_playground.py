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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xlerobot_playground.launcher import default_sim_python_bin, exec_python_module, resolve_repo_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Unified recording playground for XLeRobot. "
            "Choose sim or real, and keyboard or VR control when available."
        )
    )
    parser.add_argument("--backend", choices=("sim", "real"), default="sim")
    parser.add_argument("--controller", choices=("keyboard", "vr"), default="keyboard")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--sim-python-bin", default=None)
    parser.add_argument("--runtime-python-bin", default=sys.executable)

    parser.add_argument("--dataset-id", default="local/xlerobot_playground")
    parser.add_argument("--dataset-name", default="xlerobot_playground")
    parser.add_argument("--dataset-root", default="./datasets")
    parser.add_argument("--task", default="XLeRobot teleoperation")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--episode-length", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--use-videos", action="store_true")
    parser.add_argument("--start-key", default="[")
    parser.add_argument("--stop-key", default="]")
    parser.add_argument("--quit-key", default="\\")

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

    parser.add_argument("--port1", default="/dev/ttyACM0")
    parser.add_argument("--port2", default="/dev/ttyACM1")
    parser.add_argument("--camera", action="append", default=[])
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--use-degrees", action="store_true")
    parser.add_argument("--xlevr-path", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = resolve_repo_root(args.repo_root)

    if args.backend == "sim":
        python_bin = args.sim_python_bin or default_sim_python_bin(REPO_ROOT)
        backend_args = [
            "record",
            "--controller",
            args.controller,
            "--repo-root",
            str(repo_root),
            "--env-id",
            args.env_id,
            "--robot-uid",
            args.robot_uid,
            "--render-mode",
            args.render_mode,
            "--shader",
            args.shader,
            "--sim-backend",
            args.sim_backend,
            "--num-envs",
            str(args.num_envs),
            "--speed-profile",
            args.speed_profile,
            "--dataset-name",
            args.dataset_name,
            "--output-dir",
            args.dataset_root,
            "--num-episodes",
            str(args.num_episodes),
            "--episode-length",
            str(args.episode_length),
            "--fps",
            str(args.fps),
            "--task-description",
            args.task,
        ]
        if args.control_mode is not None:
            backend_args.extend(["--control-mode", args.control_mode])
        if args.obs_mode is not None:
            backend_args.extend(["--obs-mode", args.obs_mode])
        if args.force_reload:
            backend_args.append("--force-reload")
        return exec_python_module(
            "xlerobot_playground.sim_backend",
            python_bin=python_bin,
            argv=backend_args,
            cwd=REPO_ROOT,
        )

    backend_args = [
        "record",
        "--controller",
        args.controller,
        "--repo-root",
        str(repo_root),
        "--port1",
        args.port1,
        "--port2",
        args.port2,
        "--fps",
        str(args.fps),
        "--dataset-id",
        args.dataset_id,
        "--dataset-root",
        args.dataset_root,
        "--task",
        args.task,
        "--start-key",
        args.start_key,
        "--stop-key",
        args.stop_key,
        "--quit-key",
        args.quit_key,
        "--camera-width",
        str(args.camera_width),
        "--camera-height",
        str(args.camera_height),
        "--camera-fps",
        str(args.camera_fps),
    ]
    for camera in args.camera:
        backend_args.extend(["--camera", camera])
    if args.use_videos:
        backend_args.append("--use-videos")
    if args.use_degrees:
        backend_args.append("--use-degrees")
    if args.xlevr_path is not None:
        backend_args.extend(["--xlevr-path", args.xlevr_path])

    return exec_python_module(
        "xlerobot_playground.real_backend",
        python_bin=args.runtime_python_bin,
        argv=backend_args,
        cwd=REPO_ROOT,
    )


if __name__ == "__main__":
    raise SystemExit(main())
