from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
import shlex
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xlerobot_agent.home_agent import (
    HomeAgentConfig,
    HomeAgentController,
    HomeAgentModelConfig,
    HomeAgentServer,
    config_from_env,
    resolve_home_memory_path,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Robot42 HomeTaskAgent backend for the React UI.")
    parser.add_argument("--home-memory-path", default=None)
    parser.add_argument("--memory-root", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--provider", choices=("mock", "openai", "openai-compatible", "litellm"), default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--exploration-backend-url", default=None)
    parser.add_argument("--navigation-waypoint-horizon-m", type=float, default=None)
    parser.add_argument("--navigation-auto-rotate-threshold-deg", type=float, default=None)
    parser.add_argument("--backend-request-timeout-s", type=float, default=None)
    parser.add_argument("--agent-artifacts-root", default=None)
    parser.add_argument("--object-detector-provider", choices=("none", "mock", "replicate_grounding_dino"), default=None)
    parser.add_argument("--object-detector-api-key", default=None)
    parser.add_argument("--object-detector-model", default=None)
    parser.add_argument("--object-detector-model-version", default=None)
    parser.add_argument("--object-detector-box-threshold", type=float, default=None)
    parser.add_argument("--object-detector-text-threshold", type=float, default=None)
    parser.add_argument("--object-detector-min-confidence", type=float, default=None)
    parser.add_argument("--object-detector-timeout-s", type=float, default=None)
    parser.add_argument("--object-focus-horizontal-fov-deg", type=float, default=None)
    parser.add_argument("--object-focus-center-tolerance-norm", type=float, default=None)
    parser.add_argument("--object-focus-max-attempts", type=int, default=None)
    parser.add_argument("--object-approach-target-min-m", type=float, default=None)
    parser.add_argument("--object-approach-target-max-m", type=float, default=None)
    parser.add_argument("--object-approach-step-m", type=float, default=None)
    parser.add_argument("--object-approach-max-attempts", type=int, default=None)
    parser.add_argument("--object-approach-robot-width-m", type=float, default=None)
    parser.add_argument("--object-approach-clearance-m", type=float, default=None)
    parser.add_argument("--specialist-provider", choices=("openai", "openai-compatible", "litellm"), default=None)
    parser.add_argument("--specialist-model", default=None)
    parser.add_argument("--specialist-base-url", default=None)
    parser.add_argument("--specialist-api-key", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    _load_dotenv(REPO_ROOT / ".env")
    args = build_parser().parse_args(argv)
    config = _merge_args(config_from_env(), args)
    controller = HomeAgentController.from_config(config)
    server = HomeAgentServer(controller, host=config.host, port=config.port)
    print(f"Robot42 HomeTaskAgent backend: http://{config.host}:{config.port}")
    resolved_memory = resolve_home_memory_path(config)
    if resolved_memory is not None:
        print(f"Home memory: {resolved_memory}")
    elif config.home_memory_path:
        print(f"Home memory: {config.home_memory_path} (not found yet)")
    else:
        print("Home memory: auto-discovery enabled, none found yet")
    print(f"Main model: {config.model.provider}/{config.model.model}")
    if config.specialist_model:
        print(f"Specialist model: {config.specialist_model.provider}/{config.specialist_model.model}")
    print(
        "Exposed tools: resolve_navigation_to_region, plan_region_exploration, execute_region_exploration_plan, "
        "navigate_to_waypoint, relocalize_here, rotate_by, rotate_towards_point, micro_adjust_to_pose, "
        "focus_detected_object, approach_detected_object, grab_object"
    )
    print(f"Exploration/Nav2 backend: {config.exploration_backend_url or 'not configured'}")
    print(f"Navigation auto-rotate threshold: {config.navigation_auto_rotate_threshold_deg:g} deg")
    print(f"Agent artifacts: {config.agent_artifacts_root}")
    print(f"Object detector: {config.object_detector_provider}")
    print(
        "Object approach: "
        f"target={config.object_approach_target_min_m:g}-{config.object_approach_target_max_m:g}m, "
        f"step={config.object_approach_step_m:g}m"
    )
    if config.model.provider == "mock":
        print("Mode: mock provider. The backend will resolve previews only; it will not call Nav2 or OpenAI traces.")
    else:
        print("Mode: live agent provider. Navigation tools can call Nav2 when the exploration nav session is active.")
    server.serve_forever()
    return 0


def _merge_args(config: HomeAgentConfig, args: argparse.Namespace) -> HomeAgentConfig:
    model = config.model
    if args.provider or args.model or args.base_url or args.api_key:
        model = replace(
            model,
            provider=args.provider or model.provider,
            model=args.model or model.model,
            base_url=args.base_url if args.base_url is not None else model.base_url,
            api_key=args.api_key if args.api_key is not None else model.api_key,
        )
    specialist = config.specialist_model
    if args.specialist_provider and args.specialist_model:
        specialist = HomeAgentModelConfig(
            provider=args.specialist_provider,
            model=args.specialist_model,
            base_url=args.specialist_base_url,
            api_key=args.specialist_api_key,
        )
    return replace(
        config,
        home_memory_path=args.home_memory_path or config.home_memory_path,
        home_memory_search_roots=(args.memory_root,) if args.memory_root else config.home_memory_search_roots,
        host=args.host or config.host,
        port=args.port or config.port,
        max_turns=args.max_turns if args.max_turns is not None else config.max_turns,
        model=model,
        specialist_model=specialist,
        exploration_backend_url=(
            args.exploration_backend_url
            if args.exploration_backend_url is not None
            else config.exploration_backend_url
        ),
        navigation_waypoint_horizon_m=(
            args.navigation_waypoint_horizon_m
            if args.navigation_waypoint_horizon_m is not None
            else config.navigation_waypoint_horizon_m
        ),
        navigation_auto_rotate_threshold_deg=(
            args.navigation_auto_rotate_threshold_deg
            if args.navigation_auto_rotate_threshold_deg is not None
            else config.navigation_auto_rotate_threshold_deg
        ),
        backend_request_timeout_s=(
            args.backend_request_timeout_s
            if args.backend_request_timeout_s is not None
            else config.backend_request_timeout_s
        ),
        agent_artifacts_root=args.agent_artifacts_root or config.agent_artifacts_root,
        object_detector_provider=args.object_detector_provider or config.object_detector_provider,
        object_detector_api_key=(
            args.object_detector_api_key
            if args.object_detector_api_key is not None
            else config.object_detector_api_key
        ),
        object_detector_model=args.object_detector_model or config.object_detector_model,
        object_detector_model_version=(
            args.object_detector_model_version
            if args.object_detector_model_version is not None
            else config.object_detector_model_version
        ),
        object_detector_box_threshold=(
            args.object_detector_box_threshold
            if args.object_detector_box_threshold is not None
            else config.object_detector_box_threshold
        ),
        object_detector_text_threshold=(
            args.object_detector_text_threshold
            if args.object_detector_text_threshold is not None
            else config.object_detector_text_threshold
        ),
        object_detector_min_confidence=(
            args.object_detector_min_confidence
            if args.object_detector_min_confidence is not None
            else config.object_detector_min_confidence
        ),
        object_detector_timeout_s=(
            args.object_detector_timeout_s
            if args.object_detector_timeout_s is not None
            else config.object_detector_timeout_s
        ),
        object_focus_horizontal_fov_deg=(
            args.object_focus_horizontal_fov_deg
            if args.object_focus_horizontal_fov_deg is not None
            else config.object_focus_horizontal_fov_deg
        ),
        object_focus_center_tolerance_norm=(
            args.object_focus_center_tolerance_norm
            if args.object_focus_center_tolerance_norm is not None
            else config.object_focus_center_tolerance_norm
        ),
        object_focus_max_attempts=(
            args.object_focus_max_attempts
            if args.object_focus_max_attempts is not None
            else config.object_focus_max_attempts
        ),
        object_approach_target_min_m=(
            args.object_approach_target_min_m
            if args.object_approach_target_min_m is not None
            else config.object_approach_target_min_m
        ),
        object_approach_target_max_m=(
            args.object_approach_target_max_m
            if args.object_approach_target_max_m is not None
            else config.object_approach_target_max_m
        ),
        object_approach_step_m=(
            args.object_approach_step_m
            if args.object_approach_step_m is not None
            else config.object_approach_step_m
        ),
        object_approach_max_attempts=(
            args.object_approach_max_attempts
            if args.object_approach_max_attempts is not None
            else config.object_approach_max_attempts
        ),
        object_approach_robot_width_m=(
            args.object_approach_robot_width_m
            if args.object_approach_robot_width_m is not None
            else config.object_approach_robot_width_m
        ),
        object_approach_clearance_m=(
            args.object_approach_clearance_m
            if args.object_approach_clearance_m is not None
            else config.object_approach_clearance_m
        ),
    )


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, raw_value = stripped.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        if not key or key in os.environ:
            continue
        value = raw_value.strip()
        try:
            parsed = shlex.split(value, comments=False, posix=True)
            if len(parsed) == 1:
                value = parsed[0]
        except ValueError:
            value = value.strip("\"'")
        os.environ[key] = value


if __name__ == "__main__":
    raise SystemExit(main())
