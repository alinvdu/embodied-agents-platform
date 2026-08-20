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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xlerobot_agent import (
    DelegatedNavigationBackend,
    MockPromptClient,
    MockWakeWordDetector,
    MockVoiceTranslator,
    NavigationSkillExecutionMode,
    PromptPlanner,
    SkillContract,
    SkillRegistry,
    SkillType,
    VoiceCommandPipeline,
    WorldState,
    XLeRobotAgentRuntime,
    build_default_navigation_skills,
    create_executor_config,
)
from xlerobot_agent.integration import XLeRobotAgentBindings, default_executor_registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inference playground for the current XLeRobot agent runtime."
    )
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--voice-transcript", default=None)
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
    parser.add_argument("--current-task", default="")
    parser.add_argument("--current-pose", default="unknown")
    parser.add_argument("--localization-confidence", type=float, default=0.8)
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
        help="Set a prompt-side affordance prediction, e.g. `open_fridge=0.7`.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the run record as JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.instruction and not args.voice_transcript:
        raise SystemExit("Provide either `--instruction` or `--voice-transcript`.")

    navigation_mode = NavigationSkillExecutionMode(args.navigation_mode)
    delegated_backend = None
    if navigation_mode == NavigationSkillExecutionMode.DELEGATED_NAVIGATION_MODULE:
        delegated_backend = DelegatedNavigationBackend(args.delegated_backend)
    executor_config = create_executor_config(navigation_mode, delegated_backend)
    bindings = XLeRobotAgentBindings()

    registry = SkillRegistry()
    registry.register_many(build_default_navigation_skills(executor_config, bindings))
    registry.register_many(
        [
            SkillContract(
                skill_id="open_fridge",
                skill_type=SkillType.MANIPULATION,
                language_description="Open the fridge door when the handle is visible and reachable.",
                executor_binding=bindings.generic_skill_binding,
                required_resources=frozenset({"left_arm", "right_arm"}),
                required_observations=frozenset({"fridge_handle_visible"}),
                expected_postcondition="fridge is open",
                value_function_id="open_fridge_success",
                min_localization_confidence=0.55,
            ),
            SkillContract(
                skill_id="search_for_target",
                skill_type=SkillType.SEARCH,
                language_description="Search the scene with the head and wrist cameras to find the target.",
                executor_binding=bindings.generic_skill_binding,
                required_resources=frozenset({"head"}),
                expected_postcondition="target searched for",
                value_function_id="search_success",
            ),
            SkillContract(
                skill_id="grab_bread_from_table",
                skill_type=SkillType.MANIPULATION,
                language_description="Pick a loaf of bread from a table or countertop.",
                executor_binding=bindings.generic_skill_binding,
                required_resources=frozenset({"right_arm"}),
                required_observations=frozenset({"bread_visible"}),
                expected_postcondition="bread grasped",
                value_function_id="grasp_bread_success",
                min_localization_confidence=0.45,
            ),
            SkillContract(
                skill_id="clean_pens_on_desk",
                skill_type=SkillType.MANIPULATION,
                language_description="Gather pens from a desk and place them into a tidy arrangement.",
                executor_binding=bindings.generic_skill_binding,
                required_resources=frozenset({"left_arm", "right_arm"}),
                required_observations=frozenset({"desk_visible"}),
                expected_postcondition="pens organized",
                value_function_id="clean_desk_success",
            ),
        ]
    )

    planner = PromptPlanner(prompt_client=MockPromptClient())
    runtime = XLeRobotAgentRuntime(
        registry,
        planner,
        default_executor_registry(executor_config, bindings),
        voice_pipeline=VoiceCommandPipeline(
            wake_word_detector=MockWakeWordDetector(),
            translator=MockVoiceTranslator(),
        ),
    )

    world_state = WorldState(
        current_task=args.current_task or (args.instruction or args.voice_transcript or ""),
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
        executor_configuration=executor_config,
    )

    if args.voice_transcript:
        record = runtime.run_voice_transcript(args.voice_transcript, world_state)
        if record is None:
            raise SystemExit("Wake word was not detected in the provided transcript.")
    else:
        record = runtime.run_instruction(args.instruction, world_state)

    if args.json:
        print(json.dumps(_serialize_record(record), indent=2))
        return 0

    print(f"Normalized instruction: {record.normalized_instruction}")
    print(f"Discovered places: {[place.name for place in record.discovered_places]}")
    print(f"Subgoals: {[subgoal.text for subgoal in record.subgoals]}")
    for index, step in enumerate(record.steps, start=1):
        print(
            f"Step {index}: {step.subgoal.text} -> {step.selected_skill.skill_id} "
            f"({step.execution_result.status.value})"
        )
        print(f"  Reasoning: {step.selected_score.reasoning}")
        print(f"  Evidence: {step.execution_result.postcondition_evidence}")
    return 0


def _parse_affordances(items: list[str]) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid affordance `{item}`. Use `skill_id=0.7`.")
        skill_id, score = item.split("=", 1)
        parsed[skill_id] = float(score)
    return parsed


def _serialize_record(record) -> dict:
    return {
        "normalized_instruction": record.normalized_instruction,
        "discovered_places": [
            {
                "name": place.name,
                "confidence": place.confidence,
                "evidence": place.evidence,
            }
            for place in record.discovered_places
        ],
        "subgoals": [{"text": subgoal.text, "kind": subgoal.kind, "target": subgoal.target} for subgoal in record.subgoals],
        "steps": [
            {
                "subgoal": step.subgoal.text,
                "skill_id": step.selected_skill.skill_id,
                "status": step.execution_result.status.value,
                "reasoning": step.selected_score.reasoning,
                "evidence": step.execution_result.postcondition_evidence,
            }
            for step in record.steps
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main())
