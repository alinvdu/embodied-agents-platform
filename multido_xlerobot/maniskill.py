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
import importlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from .bootstrap import resolve_xlerobot_repo_root


class XLeRobotManiSkillError(RuntimeError):
    """Raised when the local ManiSkill + XLeRobot play stack is not ready."""


@dataclass(frozen=True)
class XLeRobotManiSkillBootstrapResult:
    repo_root: Path
    sim_root: Path
    assets_root: Path
    agent_module: ModuleType
    env_module: ModuleType


def bootstrap_xlerobot_maniskill(
    repo_root: str | Path | None = None,
    *,
    force_reload: bool = False,
) -> XLeRobotManiSkillBootstrapResult:
    root = resolve_xlerobot_repo_root(repo_root)
    sim_root = root / "simulation" / "Maniskill"
    assets_root = sim_root / "assets" / "xlerobot"
    agent_file = sim_root / "agents" / "xlerobot" / "xlerobot.py"
    env_file = sim_root / "envs" / "scenes" / "base_env.py"

    required = {
        "repo_root": root,
        "sim_root": sim_root,
        "assets_root": assets_root,
        "agent_file": agent_file,
        "env_file": env_file,
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        formatted = ", ".join(f"{name}={required[name]}" for name in missing)
        raise XLeRobotManiSkillError(
            f"XLeRobot ManiSkill files are missing or moved: {formatted}"
        )

    _ensure_mani_skill_runtime_installed()
    _patch_replicacad_scene_builder()

    robots_module = importlib.import_module("mani_skill.agents.robots")
    agent_module = _load_module(
        "multido_xlerobot._sim.xlerobot_agent",
        agent_file,
        force_reload=force_reload,
    )

    xlerobot_cls = getattr(agent_module, "Xlerobot", None)
    if xlerobot_cls is None:
        raise XLeRobotManiSkillError(f"`Xlerobot` class not found in {agent_file}")

    # Point the registered agent at the URDF in the fork so no site-packages patching
    # is required.
    xlerobot_cls.urdf_path = str(assets_root / "xlerobot.urdf")
    _override_registered_agent(xlerobot_cls)
    setattr(robots_module, "Xlerobot", xlerobot_cls)
    if hasattr(robots_module, "__all__") and "Xlerobot" not in robots_module.__all__:
        robots_module.__all__ = list(robots_module.__all__) + ["Xlerobot"]

    env_module = _load_module(
        "multido_xlerobot._sim.scene_manipulation_env",
        env_file,
        force_reload=force_reload,
    )
    scene_manipulation_cls = getattr(env_module, "SceneManipulationEnv", None)
    if scene_manipulation_cls is None:
        raise XLeRobotManiSkillError(
            f"`SceneManipulationEnv` class not found in {env_file}"
        )
    _override_registered_env(scene_manipulation_cls)

    return XLeRobotManiSkillBootstrapResult(
        repo_root=root,
        sim_root=sim_root,
        assets_root=assets_root,
        agent_module=agent_module,
        env_module=env_module,
    )


def run_keyboard_play_demo(
    *,
    repo_root: str | Path | None = None,
    demo: str = "ee_keyboard",
    env_id: str = "SceneManipulation-v1",
    robot_uid: str = "xlerobot",
    control_mode: str | None = None,
    obs_mode: str | None = None,
    render_mode: str = "human",
    shader: str = "default",
    sim_backend: str = "auto",
    num_envs: int = 1,
    record_dir: str | None = None,
    dataset_name: str | None = None,
    output_dir: str | None = None,
    num_episodes: int | None = None,
    episode_length: int | None = None,
    fps: int | None = None,
    task_description: str | None = None,
    use_rerun: bool | None = None,
    show_cameras: bool | None = None,
    speed_profile: str | None = None,
    force_reload: bool = False,
) -> Any:
    root = resolve_xlerobot_repo_root(repo_root)
    bootstrap_xlerobot_maniskill(root, force_reload=force_reload)
    if demo == "vr":
        _prepare_vr_monitor_module(root, force_reload=force_reload)
    demo_module = _load_demo_module(root, demo, force_reload=force_reload)

    args = demo_module.Args()
    args.env_id = env_id
    args.robot_uids = robot_uid
    args.render_mode = render_mode
    args.shader = shader
    args.sim_backend = sim_backend
    args.num_envs = num_envs

    effective_control_mode = control_mode or _default_control_mode_for_demo(demo, robot_uid)
    if effective_control_mode is not None:
        args.control_mode = effective_control_mode
    if obs_mode is not None:
        args.obs_mode = obs_mode
    if record_dir is not None and hasattr(args, "record_dir"):
        args.record_dir = record_dir
    if dataset_name is not None and hasattr(args, "dataset_name"):
        args.dataset_name = dataset_name
    if output_dir is not None and hasattr(args, "output_dir"):
        args.output_dir = output_dir
    if num_episodes is not None and hasattr(args, "num_episodes"):
        args.num_episodes = num_episodes
    if episode_length is not None and hasattr(args, "episode_length"):
        args.episode_length = episode_length
    if fps is not None and hasattr(args, "fps"):
        args.fps = fps
    if task_description is not None and hasattr(args, "task_description"):
        args.task_description = task_description
    if use_rerun is not None and hasattr(args, "use_rerun"):
        args.use_rerun = use_rerun
    if show_cameras is not None and hasattr(args, "show_cameras"):
        args.show_cameras = show_cameras
    if speed_profile is not None and hasattr(args, "speed_profile"):
        args.speed_profile = speed_profile

    return demo_module.main(args)


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Launch the XLeRobot play or teleop dataset demos in ManiSkill without "
            "manually copying files into site-packages."
        )
    )
    parser.add_argument(
        "--repo-root",
        default=str(resolve_xlerobot_repo_root()),
        help="Path to the local xlerobot_forked repository.",
    )
    parser.add_argument(
        "--demo",
        choices=("ee_keyboard", "joint_control", "camera_rerun", "vr", "record_dataset"),
        default="ee_keyboard",
        help="Which local XLeRobot ManiSkill demo to run.",
    )
    parser.add_argument(
        "--env-id",
        default="SceneManipulation-v1",
        help=(
            "Gym environment id. `SceneManipulation-v1` is the safest default "
            "because it is registered locally at runtime."
        ),
    )
    parser.add_argument(
        "--robot-uid",
        default="xlerobot",
        help="Robot uid to pass to ManiSkill.",
    )
    parser.add_argument(
        "--control-mode",
        default=None,
        help="Override the demo's default control mode.",
    )
    parser.add_argument(
        "--obs-mode",
        default=None,
        help="Override the demo's default observation mode.",
    )
    parser.add_argument(
        "--render-mode",
        default="human",
        help="ManiSkill render mode. Use `human` for interactive play.",
    )
    parser.add_argument(
        "--shader",
        default="default",
        help="Shader pack to use. `default` is a good starting point.",
    )
    parser.add_argument(
        "--sim-backend",
        default="auto",
        help="Simulation backend to pass through to ManiSkill.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of environments to run.",
    )
    parser.add_argument(
        "--record-dir",
        default=None,
        help="Optional ManiSkill recording directory.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Dataset name for demos that can write LeRobot episodes.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for demos that can write LeRobot episodes.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Episode count for demos that support dataset recording.",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=None,
        help="Maximum episode length for demos that support dataset recording.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Recording FPS for demos that support dataset recording.",
    )
    parser.add_argument(
        "--task-description",
        default=None,
        help="Task label stored in dataset-recording demos.",
    )
    parser.add_argument(
        "--use-rerun",
        choices=("true", "false"),
        default=None,
        help="Override rerun usage for demos that expose it.",
    )
    parser.add_argument(
        "--show-cameras",
        choices=("true", "false"),
        default=None,
        help="Override camera display for demos that expose it.",
    )
    parser.add_argument(
        "--speed-profile",
        choices=("normal", "fast"),
        default=None,
        help="Keyboard teleop speed profile for demos that expose it.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the local setup and exit without launching the demo.",
    )
    parser.add_argument(
        "--force-reload",
        action="store_true",
        help="Force reloading local XLeRobot sim modules before launching.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_cli_parser()
    args = parser.parse_args(argv)

    try:
        result = bootstrap_xlerobot_maniskill(
            args.repo_root,
            force_reload=args.force_reload,
        )
    except XLeRobotManiSkillError as exc:
        parser.exit(
            2,
            f"Setup error: {exc}\n"
            "Tip: run scripts/setup_xlerobot_maniskill_env.sh first, and make sure "
            "you launch this command from that environment.\n",
        )

    if args.check:
        print("XLeRobot ManiSkill bootstrap succeeded.")
        print(f"repo_root: {result.repo_root}")
        print(f"sim_root: {result.sim_root}")
        print(f"assets_root: {result.assets_root}")
        print("registered_agent: xlerobot")
        print("registered_env: SceneManipulation-v1")
        return 0

    run_keyboard_play_demo(
        repo_root=args.repo_root,
        demo=args.demo,
        env_id=args.env_id,
        robot_uid=args.robot_uid,
        control_mode=args.control_mode,
        obs_mode=args.obs_mode,
        render_mode=args.render_mode,
        shader=args.shader,
        sim_backend=args.sim_backend,
        num_envs=args.num_envs,
        record_dir=args.record_dir,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        num_episodes=args.num_episodes,
        episode_length=args.episode_length,
        fps=args.fps,
        task_description=args.task_description,
        use_rerun=_parse_optional_bool(args.use_rerun),
        show_cameras=_parse_optional_bool(args.show_cameras),
        speed_profile=args.speed_profile,
        force_reload=args.force_reload,
    )
    return 0


def _ensure_mani_skill_runtime_installed() -> None:
    missing: list[str] = []
    for module_name in ("mani_skill", "gymnasium", "sapien", "pygame", "tyro"):
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name:
                missing.append(module_name)
            else:
                missing.append(f"{module_name} (import failed because {exc.name} is missing)")
        except Exception as exc:  # pragma: no cover - defensive import diagnostics
            missing.append(f"{module_name} (import failed: {exc})")

    if missing:
        raise XLeRobotManiSkillError(
            "Missing Python runtime dependencies for the play setup: "
            + ", ".join(missing)
        )


def _override_registered_agent(agent_cls: type[Any]) -> None:
    registration_module = importlib.import_module("mani_skill.agents.registration")
    register_agent = getattr(registration_module, "register_agent")
    register_agent(asset_download_ids=[], override=True)(agent_cls)


def _override_registered_env(env_cls: type[Any]) -> None:
    registration_module = importlib.import_module("mani_skill.utils.registration")
    register_env = getattr(registration_module, "register_env")
    register_env(
        "SceneManipulation-v1",
        max_episode_steps=200,
        override=True,
        asset_download_ids=[],
    )(env_cls)


def _patch_replicacad_scene_builder() -> None:
    scene_builder_module = importlib.import_module(
        "mani_skill.utils.scene_builder.replicacad.scene_builder"
    )
    builder_cls = getattr(scene_builder_module, "ReplicaCADSceneBuilder", None)
    if builder_cls is None:
        raise XLeRobotManiSkillError(
            "ReplicaCADSceneBuilder could not be imported from mani_skill"
        )
    if getattr(builder_cls, "_xlerobot_patched", False):
        return

    articulation_cls = getattr(scene_builder_module, "Articulation")
    sapien_module = getattr(scene_builder_module, "sapien")

    def initialize(self, env_idx):
        # Mirror ManiSkill's default ReplicaCAD reset flow, but allow xlerobot
        # to respawn the same way fetch does.
        self.env.agent.robot.set_pose(sapien_module.Pose([-10, 0, -100]))

        for obj, pose in self._default_object_poses:
            obj.set_pose(pose)
            if isinstance(obj, articulation_cls):
                obj.set_qpos(obj.qpos[0] * 0)
                obj.set_qvel(obj.qvel[0] * 0)

        if self.scene.gpu_sim_enabled and len(env_idx) == self.env.num_envs:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene.px.step()
            self.scene._gpu_fetch_all()

        if self.env.robot_uids in {"fetch", "xlerobot"}:
            self.env.agent.reset(self.env.agent.keyframes["rest"].qpos)
            self.env.agent.robot.set_pose(self.robot_initial_pose)
            return

        raise NotImplementedError(self.env.robot_uids)

    builder_cls.initialize = initialize
    builder_cls._xlerobot_patched = True


def _prepare_vr_monitor_module(repo_root: Path, *, force_reload: bool) -> ModuleType:
    examples_pkg_name = "mani_skill.examples"
    examples_pkg = sys.modules.get(examples_pkg_name)
    if examples_pkg is None:
        examples_pkg = ModuleType(examples_pkg_name)
        examples_pkg.__path__ = []
        sys.modules[examples_pkg_name] = examples_pkg
        mani_skill_pkg = sys.modules.get("mani_skill")
        if mani_skill_pkg is not None:
            setattr(mani_skill_pkg, "examples", examples_pkg)

    module = _load_module(
        f"{examples_pkg_name}.vr_monitor",
        repo_root / "simulation" / "Maniskill" / "examples" / "vr_monitor.py",
        force_reload=force_reload,
    )
    if hasattr(module, "XLEVR_PATH"):
        module.XLEVR_PATH = str((repo_root / "XLeVR").resolve())
    setattr(examples_pkg, "vr_monitor", module)
    return module


def _load_demo_module(repo_root: Path, demo: str, *, force_reload: bool) -> ModuleType:
    mapping = {
        "ee_keyboard": repo_root
        / "simulation"
        / "Maniskill"
        / "examples"
        / "demo_ctrl_action_ee_keyboard.py",
        "joint_control": repo_root
        / "simulation"
        / "Maniskill"
        / "examples"
        / "demo_ctrl_action.py",
        "camera_rerun": repo_root
        / "simulation"
        / "Maniskill"
        / "examples"
        / "demo_ctrl_action_ee_cam_rerun.py",
        "vr": repo_root
        / "simulation"
        / "Maniskill"
        / "examples"
        / "demo_ctrl_action_ee_VR.py",
        "record_dataset": repo_root
        / "simulation"
        / "Maniskill"
        / "examples"
        / "demo_ctrl_ee_keyboard_record_dataset.py",
    }
    file_path = mapping[demo]
    return _load_module(
        f"multido_xlerobot._sim.demo_{demo}",
        file_path,
        force_reload=force_reload,
    )


def _load_module(module_name: str, file_path: Path, *, force_reload: bool) -> ModuleType:
    if not file_path.exists():
        raise XLeRobotManiSkillError(f"Missing module file: {file_path}")

    if force_reload and module_name in sys.modules:
        del sys.modules[module_name]
    elif module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise XLeRobotManiSkillError(f"Unable to build import spec for {module_name}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _parse_optional_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    return value == "true"


def _default_control_mode_for_demo(demo: str, robot_uid: str) -> str | None:
    if robot_uid != "xlerobot":
        return None
    if demo in {"camera_rerun", "vr", "record_dataset"}:
        return "pd_joint_delta_pos_dual_arm"
    return None


if __name__ == "__main__":
    raise SystemExit(main())
