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
    parser.add_argument("--robot-brain-url", default=None)
    parser.add_argument(
        "--vla-handoff",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow grab_object to call the robot brain's on-demand VLA endpoint.",
    )
    parser.add_argument("--vla-handoff-duration-s", type=float, default=None)
    parser.add_argument(
        "--basket-verifier-provider",
        choices=("openai", "openai-compatible", "litellm", "ollama"),
        default=None,
    )
    parser.add_argument("--basket-verifier-model", default=None)
    parser.add_argument("--basket-verifier-base-url", default=None)
    parser.add_argument("--basket-verifier-api-key", default=None)
    parser.add_argument("--basket-verification-manifest", default=None)
    parser.add_argument("--basket-verification-minimum-confidence", type=float, default=None)
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--navigation-waypoint-horizon-m", type=float, default=None)
    parser.add_argument(
        "--navigation-waypoint-breakdown",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Split semantic navigation into short agent-level waypoints. "
            "Use --no-navigation-waypoint-breakdown to send the final resolved goal directly to Nav2."
        ),
    )
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
    parser.add_argument("--object-detector-max-image-edge-px", type=int, default=None)
    parser.add_argument("--object-detector-jpeg-quality", type=int, default=None)
    parser.add_argument("--no-object-confirmation", action="store_false", dest="object_confirmation_required", default=None)
    parser.add_argument("--object-confirmation-timeout-s", type=float, default=None)
    parser.add_argument("--object-focus-horizontal-fov-deg", type=float, default=None)
    parser.add_argument("--object-focus-center-tolerance-norm", type=float, default=None)
    parser.add_argument("--object-focus-max-attempts", type=int, default=None)
    parser.add_argument("--object-approach-target-min-m", type=float, default=None)
    parser.add_argument("--object-approach-target-max-m", type=float, default=None)
    parser.add_argument("--object-approach-target-tolerance-m", type=float, default=None)
    parser.add_argument("--object-approach-step-m", type=float, default=None)
    parser.add_argument("--object-approach-step-fraction", type=float, default=None)
    parser.add_argument("--object-approach-max-attempts", type=int, default=None)
    parser.add_argument("--object-approach-robot-width-m", type=float, default=None)
    parser.add_argument("--object-approach-clearance-m", type=float, default=None)
    parser.add_argument("--agent-tool-output-mode", choices=("compact", "full"), default=None)
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
        "focus_detected_object, approach_detected_object, grab_object, return_to_start, stop_robot"
    )
    print(f"Exploration/Nav2 backend: {config.exploration_backend_url or 'not configured'}")
    print(f"Robot brain: {config.robot_brain_url or 'not configured'}")
    print(
        "VLA handoff: "
        f"{'enabled' if config.vla_handoff_enabled and not config.dry_run else 'disabled'}"
    )
    basket_model = config.basket_verifier_model or config.specialist_model or config.model
    print(
        "Basket verifier: "
        f"{basket_model.provider}/{basket_model.model}, "
        f"minimum confidence={config.basket_verification_minimum_confidence:g}"
    )
    print(f"Navigation auto-rotate threshold: {config.navigation_auto_rotate_threshold_deg:g} deg")
    print(
        "Navigation waypoint breakdown: "
        f"{'enabled' if config.navigation_waypoint_breakdown_enabled else 'disabled'}"
    )
    print(f"Agent artifacts: {config.agent_artifacts_root}")
    print(f"Object detector: {config.object_detector_provider}")
    print(
        "Object confirmation: "
        f"{'required' if config.object_detection_confirmation_required else 'disabled'}"
        + (
            f", timeout={config.object_detection_confirmation_timeout_s:g}s"
            if config.object_detection_confirmation_timeout_s > 0
            else ", no timeout"
        )
    )
    if config.object_detector_provider == "replicate_grounding_dino":
        print(
            "Object detector thresholds: "
            f"box={config.object_detector_box_threshold:g}, "
            f"text={config.object_detector_text_threshold:g}, "
            f"min_confidence={config.object_detector_min_confidence:g}"
        )
        print(
            "Object detector image: "
            f"max_edge={config.object_detector_max_image_edge_px}px, jpeg_quality={config.object_detector_jpeg_quality}"
        )
    print(
        "Object approach: "
        f"target={config.object_approach_target_min_m:g}-{config.object_approach_target_max_m:g}m, "
        f"tolerance={config.object_approach_target_tolerance_m:g}m, "
        f"max_step={config.object_approach_step_m:g}m, "
        f"step_fraction={config.object_approach_step_fraction:g}, "
        f"attempts={config.object_approach_max_attempts}"
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
    basket_verifier = config.basket_verifier_model
    if args.basket_verifier_provider or args.basket_verifier_model:
        fallback = basket_verifier or specialist or model
        basket_verifier = HomeAgentModelConfig(
            provider=args.basket_verifier_provider or fallback.provider,
            model=args.basket_verifier_model or fallback.model,
            base_url=(
                args.basket_verifier_base_url
                if args.basket_verifier_base_url is not None
                else fallback.base_url
            ),
            api_key=(
                args.basket_verifier_api_key
                if args.basket_verifier_api_key is not None
                else fallback.api_key
            ),
            temperature=0.0,
            max_tokens=300,
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
        robot_brain_url=(
            args.robot_brain_url
            if args.robot_brain_url is not None
            else config.robot_brain_url
        ),
        vla_handoff_enabled=(
            args.vla_handoff
            if args.vla_handoff is not None
            else config.vla_handoff_enabled
        ),
        vla_handoff_duration_s=(
            args.vla_handoff_duration_s
            if args.vla_handoff_duration_s is not None
            else config.vla_handoff_duration_s
        ),
        basket_verifier_model=basket_verifier,
        basket_verification_manifest_path=(
            args.basket_verification_manifest
            if args.basket_verification_manifest is not None
            else config.basket_verification_manifest_path
        ),
        basket_verification_minimum_confidence=(
            args.basket_verification_minimum_confidence
            if args.basket_verification_minimum_confidence is not None
            else config.basket_verification_minimum_confidence
        ),
        dry_run=args.dry_run if args.dry_run is not None else config.dry_run,
        navigation_waypoint_horizon_m=(
            args.navigation_waypoint_horizon_m
            if args.navigation_waypoint_horizon_m is not None
            else config.navigation_waypoint_horizon_m
        ),
        navigation_waypoint_breakdown_enabled=(
            args.navigation_waypoint_breakdown
            if args.navigation_waypoint_breakdown is not None
            else config.navigation_waypoint_breakdown_enabled
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
        object_detector_max_image_edge_px=(
            args.object_detector_max_image_edge_px
            if args.object_detector_max_image_edge_px is not None
            else config.object_detector_max_image_edge_px
        ),
        object_detector_jpeg_quality=(
            args.object_detector_jpeg_quality
            if args.object_detector_jpeg_quality is not None
            else config.object_detector_jpeg_quality
        ),
        object_detection_confirmation_required=(
            args.object_confirmation_required
            if args.object_confirmation_required is not None
            else config.object_detection_confirmation_required
        ),
        object_detection_confirmation_timeout_s=(
            args.object_confirmation_timeout_s
            if args.object_confirmation_timeout_s is not None
            else config.object_detection_confirmation_timeout_s
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
        object_approach_target_tolerance_m=(
            args.object_approach_target_tolerance_m
            if args.object_approach_target_tolerance_m is not None
            else config.object_approach_target_tolerance_m
        ),
        object_approach_step_m=(
            args.object_approach_step_m
            if args.object_approach_step_m is not None
            else config.object_approach_step_m
        ),
        object_approach_step_fraction=(
            args.object_approach_step_fraction
            if args.object_approach_step_fraction is not None
            else config.object_approach_step_fraction
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
        agent_tool_output_mode=args.agent_tool_output_mode or config.agent_tool_output_mode,
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
