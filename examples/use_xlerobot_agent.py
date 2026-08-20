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

from multido_xlerobot import XLeRobotInterface
from multido_xlerobot.bootstrap import resolve_xlerobot_repo_root
from xlerobot_agent import (
    DelegatedNavigationBackend,
    MockPromptClient,
    NavigationSkillExecutionMode,
    PromptPlanner,
    SkillContract,
    SkillRegistry,
    SkillType,
    WorldState,
    XLeRobotAgentRuntime,
    build_default_navigation_skills,
    create_executor_config,
)
from xlerobot_agent.integration import XLeRobotAgentBindings, default_executor_registry


def main() -> None:
    # This import surface is safe without bootstrapping `lerobot`; only summary/help is used here.
    interface = XLeRobotInterface(resolve_xlerobot_repo_root())
    print("Using XLeRobot fork:", interface.repo_root)

    executor_config = create_executor_config(
        NavigationSkillExecutionMode.DELEGATED_NAVIGATION_MODULE,
        DelegatedNavigationBackend.GLOBAL_MAP,
    )
    bindings = XLeRobotAgentBindings()

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

    planner = PromptPlanner(prompt_client=MockPromptClient())
    executors = default_executor_registry(executor_config, bindings)
    runtime = XLeRobotAgentRuntime(registry, planner, executors)

    world_state = WorldState(
        current_task="Open the fridge in the kitchen.",
        current_pose="living_room",
        localization_confidence=0.82,
        visible_objects=frozenset({"table"}),
        visible_landmarks=frozenset({"hallway"}),
        available_observations=frozenset({"hallway_visible"}),
        satisfied_preconditions=frozenset(),
        affordance_predictions={
            "navigate_to_region": 0.92,
            "search_for_target": 0.65,
            "open_fridge": 0.2,
            "align_for_skill": 0.4,
        },
        executor_configuration=executor_config,
    )
    record = runtime.run_instruction("Go to the kitchen and open the fridge.", world_state)
    print("Normalized instruction:", record.normalized_instruction)
    print("Discovered places:", [place.name for place in record.discovered_places])
    print("Subgoals:", [subgoal.text for subgoal in record.subgoals])
    if record.steps:
        first = record.steps[0]
        print("Selected skill:", first.selected_skill.skill_id)
        print("Combined score:", first.selected_score.combined_score)
        print("Execution status:", first.execution_result.status.value)
        print("Evidence:", first.execution_result.postcondition_evidence)
        print("Reasoning:", first.selected_score.reasoning)


if __name__ == "__main__":
    main()
