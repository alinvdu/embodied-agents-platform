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

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xlerobot_agent import (
    MockPromptClient,
    MockVoiceCommandApp,
    MockVoiceTranslator,
    MockWakeWordDetector,
    NavigationSkillExecutionMode,
    PromptPlanner,
    SkillContract,
    SkillRegistry,
    SkillType,
    VoiceCommandPipeline,
    WakeWordConfig,
    WorldState,
    XLeRobotAgentRuntime,
    build_default_navigation_skills,
    create_executor_config,
)
from xlerobot_agent.integration import XLeRobotAgentBindings, default_executor_registry


def build_runtime() -> tuple[XLeRobotAgentRuntime, WorldState]:
    bindings = XLeRobotAgentBindings()
    executor_config = create_executor_config(NavigationSkillExecutionMode.VLA_NAVIGATION_SKILLS)

    registry = SkillRegistry()
    registry.register_many(build_default_navigation_skills(executor_config, bindings))
    registry.register_many(
        [
            SkillContract(
                skill_id="open_fridge",
                skill_type=SkillType.MANIPULATION,
                language_description="Open the fridge door.",
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
                language_description="Search for a target object using available cameras.",
                executor_binding=bindings.generic_skill_binding,
                required_resources=frozenset({"head"}),
                expected_postcondition="target was searched for",
                value_function_id="search_success",
            ),
        ]
    )

    runtime = XLeRobotAgentRuntime(
        registry=registry,
        planner=PromptPlanner(prompt_client=MockPromptClient()),
        executors=default_executor_registry(executor_config, bindings),
        voice_pipeline=VoiceCommandPipeline(
            wake_word_detector=MockWakeWordDetector(WakeWordConfig(wake_word="hey xlerobot")),
            translator=MockVoiceTranslator(),
        ),
    )
    world_state = WorldState(
        current_task="Voice-driven household assistance.",
        current_pose="living_room",
        localization_confidence=0.85,
        available_observations=frozenset({"hallway_visible"}),
        affordance_predictions={
            "navigate_to_region": 0.9,
            "search_for_target": 0.7,
            "open_fridge": 0.3,
            "align_for_skill": 0.5,
        },
        executor_configuration=executor_config,
    )
    return runtime, world_state


def main() -> None:
    runtime, world_state = build_runtime()
    assert runtime.voice_pipeline is not None
    app = MockVoiceCommandApp(runtime.voice_pipeline, runtime, world_state)
    app.run_cli()


if __name__ == "__main__":
    main()
