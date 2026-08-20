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
import json
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xlerobot_agent import (
    DelegatedNavigationBackend,
    MockVoiceTranslator,
    MockWakeWordDetector,
    NavigationSkillExecutionMode,
    VoiceCommandPipeline,
    WakeWordConfig,
    WorldState,
)
from xlerobot_agent.environment import build_environment_adapter
from xlerobot_agent.llm import AgentLLMRouter, AgentModelSuite, ModelConfig
from xlerobot_agent.offload import OffloadClient
from xlerobot_agent.playground import PlaygroundAgentController, PlaygroundAgentRuntime
from xlerobot_agent.reporting import LiveAgentReport
from xlerobot_agent.tools import build_default_tool_registry
from xlerobot_agent.ui import PlaygroundUIServer
from xlerobot_playground.launcher import default_sim_python_bin


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Agent playground for XLeRobot. Launch a sim or real robot planning session, "
            "route planning/review/code generation through an LLM of your choice, and inspect "
            "the live execution report in the local web UI."
        )
    )
    parser.add_argument("--backend", choices=("sim", "real"), default="sim")
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--voice-transcript", default=None)
    parser.add_argument("--wake-word", default="hey xlerobot")
    parser.add_argument("--serve-ui", action="store_true")
    parser.add_argument("--ui-host", default="127.0.0.1")
    parser.add_argument("--ui-port", type=int, default=8765)

    parser.add_argument("--provider", choices=("mock", "openai-compatible", "litellm"), default="mock")
    parser.add_argument("--model", default="mock-planner")
    parser.add_argument("--planner-model", default=None)
    parser.add_argument("--critic-model", default=None)
    parser.add_argument("--coder-model", default=None)
    parser.add_argument("--visual-model", default=None)
    parser.add_argument("--visual-summary-model", default=None)
    parser.add_argument("--visual-diff-model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument(
        "--model-extra-json",
        default=None,
        help="Optional JSON object merged into each model request body.",
    )

    parser.add_argument(
        "--navigation-mode",
        choices=[mode.value for mode in NavigationSkillExecutionMode],
        default=NavigationSkillExecutionMode.DELEGATED_NAVIGATION_MODULE.value,
    )
    parser.add_argument(
        "--delegated-backend",
        choices=[backend.value for backend in DelegatedNavigationBackend],
        default=DelegatedNavigationBackend.GLOBAL_MAP.value,
    )
    parser.add_argument("--skill-catalog", default=None)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--sim-python-bin", default=None)
    parser.add_argument("--offload-server-url", default=None)
    parser.add_argument("--brain-name", default="xlerobot-macos-brain")
    parser.add_argument("--brain-id", default=None)
    parser.add_argument("--brain-meta-json", default=None)

    parser.add_argument("--current-task", default="XLeRobot agent playground task")
    parser.add_argument("--current-pose", default="hallway")
    parser.add_argument("--localization-confidence", type=float, default=0.82)
    parser.add_argument("--visible-object", action="append", default=[])
    parser.add_argument("--visible-landmark", action="append", default=[])
    parser.add_argument("--image-description", action="append", default=[])
    parser.add_argument("--observation", action="append", default=[])
    parser.add_argument("--precondition", action="append", default=[])
    parser.add_argument("--resource-lock", action="append", default=[])
    parser.add_argument("--history", action="append", default=[])
    parser.add_argument("--semantic-memory", default="")
    parser.add_argument("--spatial-memory", default="")
    parser.add_argument(
        "--affordance",
        action="append",
        default=[],
        metavar="SKILL=SCORE",
        help="Prompt-side affordance prediction, e.g. `open_fridge=0.7`.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.serve_ui and not args.instruction and not args.voice_transcript:
        raise SystemExit("Provide `--serve-ui`, `--instruction`, or `--voice-transcript`.")

    world_state = WorldState(
        current_task=args.current_task,
        current_pose=args.current_pose,
        localization_confidence=args.localization_confidence,
        visible_objects=frozenset(args.visible_object),
        visible_landmarks=frozenset(args.visible_landmark),
        image_descriptions=tuple(args.image_description),
        semantic_memory_summary=args.semantic_memory,
        spatial_memory_summary=args.spatial_memory,
        active_resource_locks=frozenset(args.resource_lock),
        recent_execution_history=tuple(args.history),
        available_observations=frozenset(args.observation),
        satisfied_preconditions=frozenset(args.precondition),
        affordance_predictions=_parse_affordances(args.affordance),
    )

    navigation_mode = NavigationSkillExecutionMode(args.navigation_mode)
    delegated_backend = None
    if navigation_mode == NavigationSkillExecutionMode.DELEGATED_NAVIGATION_MODULE:
        delegated_backend = DelegatedNavigationBackend(args.delegated_backend)

    adapter = build_environment_adapter(
        backend=args.backend,
        initial_world_state=world_state,
        navigation_mode=navigation_mode,
        delegated_backend=delegated_backend,
        repo_root=args.repo_root,
        sim_python_bin=args.sim_python_bin or str(default_sim_python_bin(REPO_ROOT)),
        skill_catalog_path=args.skill_catalog,
    )

    model_suite = _build_model_suite(args)
    brain_metadata = json.loads(args.brain_meta_json) if args.brain_meta_json else {}
    offload_client = None
    if args.offload_server_url:
        offload_client = OffloadClient(
            args.offload_server_url,
            brain_name=args.brain_name,
            brain_id=args.brain_id,
            metadata=brain_metadata,
        )
    report = LiveAgentReport(
        backend=args.backend,
        models={role: (config.__dict__ if config is not None else None) for role, config in model_suite.__dict__.items()},
        environment=adapter.describe_environment(),
    )
    runtime = PlaygroundAgentRuntime(
        environment=adapter,
        llm_router=AgentLLMRouter(model_suite),
        tools=build_default_tool_registry(offload_client=offload_client, exploration_mode=args.backend),
        report=report,
        voice_pipeline=VoiceCommandPipeline(
            wake_word_detector=MockWakeWordDetector(WakeWordConfig(wake_word=args.wake_word)),
            translator=MockVoiceTranslator(),
        ),
        offload_client=offload_client,
    )
    controller = PlaygroundAgentController(runtime, report)

    if args.instruction:
        controller.start_instruction(args.instruction)
    elif args.voice_transcript:
        controller.start_voice_transcript(args.voice_transcript)

    if args.serve_ui:
        server = PlaygroundUIServer(controller, host=args.ui_host, port=args.ui_port)
        print(f"XLeRobot agent UI: http://{args.ui_host}:{args.ui_port}")
        server.serve_forever()
        return 0

    while True:
        snapshot = controller.snapshot()
        if snapshot["status"] in {"completed", "failed", "stopped", "ignored"}:
            break
        time.sleep(0.1)

    snapshot = controller.snapshot()
    print(json.dumps(snapshot, indent=2))
    return 0


def _build_model_suite(args) -> AgentModelSuite:
    extra_body = json.loads(args.model_extra_json) if args.model_extra_json else {}

    def make(model_name: str | None) -> ModelConfig:
        return ModelConfig(
            provider=args.provider,
            model=model_name or args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            thinking=args.thinking,
            reasoning_effort=args.reasoning_effort,
            extra_body=extra_body,
        )

    return AgentModelSuite(
        planner=make(args.planner_model),
        critic=make(args.critic_model),
        coder=make(args.coder_model),
        visual_summary=make(args.visual_summary_model or args.visual_model) if (args.visual_summary_model or args.visual_model) else None,
        visual_diff=make(args.visual_diff_model or args.visual_model) if (args.visual_diff_model or args.visual_model) else None,
    )


def _parse_affordances(items: list[str]) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid affordance `{item}`. Use `skill_id=0.7`.")
        skill_id, score = item.split("=", 1)
        parsed[skill_id] = float(score)
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
