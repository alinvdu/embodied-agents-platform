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

from dataclasses import dataclass
from typing import Any

from .executors import StaticSkillExecutor, make_blocked_result, make_success_result
from .models import (
    DelegatedNavigationBackend,
    ExecutorConfig,
    GoalContext,
    NavigationSkillExecutionMode,
    ReadinessState,
    SkillContract,
    SkillType,
    WorldState,
)


@dataclass(frozen=True)
class XLeRobotAgentBindings:
    generic_skill_binding: str = "xlerobot.skill.generic"
    vla_navigation_binding: str = "xlerobot.navigation.vla"
    delegated_progressive_map_binding: str = "xlerobot.navigation.delegated.progressive_map"
    delegated_global_map_binding: str = "xlerobot.navigation.delegated.global_map"

    def navigation_binding(self, config: ExecutorConfig) -> str:
        if config.navigation_skill_execution_mode == NavigationSkillExecutionMode.VLA_NAVIGATION_SKILLS:
            return self.vla_navigation_binding
        if config.delegated_navigation_backend == DelegatedNavigationBackend.PROGRESSIVE_MAP:
            return self.delegated_progressive_map_binding
        if config.delegated_navigation_backend == DelegatedNavigationBackend.GLOBAL_MAP:
            return self.delegated_global_map_binding
        raise ValueError("delegated navigation mode requires a delegated backend")


def create_executor_config(
    navigation_mode: NavigationSkillExecutionMode,
    delegated_backend: DelegatedNavigationBackend | None = None,
) -> ExecutorConfig:
    if navigation_mode == NavigationSkillExecutionMode.DELEGATED_NAVIGATION_MODULE and delegated_backend is None:
        raise ValueError("delegated navigation mode requires delegated_backend")
    return ExecutorConfig(
        navigation_skill_execution_mode=navigation_mode,
        delegated_navigation_backend=delegated_backend,
    )


def build_default_navigation_skills(
    config: ExecutorConfig,
    bindings: XLeRobotAgentBindings | None = None,
) -> list[SkillContract]:
    bindings = bindings or XLeRobotAgentBindings()
    executor_binding = bindings.navigation_binding(config)

    return [
        SkillContract(
            skill_id="navigate_to_region",
            skill_type=SkillType.NAVIGATION,
            language_description="Navigate the robot base to the requested room or region.",
            executor_binding=executor_binding,
            required_resources=frozenset({"base"}),
            expected_postcondition="robot reached requested region",
            value_function_id="nav_success",
            terminal_error_codes=("nav_failed", "nav_blocked"),
            min_localization_confidence=0.45,
            expected_execution_cost=2.0,
            expected_latency_s=3.0,
        ),
        SkillContract(
            skill_id="approach_target",
            skill_type=SkillType.NAVIGATION,
            language_description="Move the robot toward a visible or remembered target.",
            executor_binding=executor_binding,
            required_resources=frozenset({"base"}),
            expected_postcondition="robot reached target approach pose",
            value_function_id="approach_success",
            terminal_error_codes=("approach_failed", "approach_blocked"),
            min_localization_confidence=0.45,
            expected_execution_cost=1.5,
            expected_latency_s=2.0,
        ),
        SkillContract(
            skill_id="move_to_viewpoint",
            skill_type=SkillType.NAVIGATION,
            language_description="Move the robot to obtain a good viewpoint on the target.",
            executor_binding=executor_binding,
            required_resources=frozenset({"base", "head"}),
            expected_postcondition="robot reached perception-ready pose",
            value_function_id="viewpoint_success",
            terminal_error_codes=("view_failed",),
            min_localization_confidence=0.35,
            expected_execution_cost=1.0,
            expected_latency_s=1.5,
        ),
        SkillContract(
            skill_id="align_for_skill",
            skill_type=SkillType.ALIGNMENT,
            language_description="Align the base and sensors for the next manipulation skill.",
            executor_binding=executor_binding,
            required_resources=frozenset({"base", "head"}),
            expected_postcondition="robot reached skill-ready pose",
            value_function_id="align_success",
            terminal_error_codes=("align_failed",),
            min_localization_confidence=0.55,
            expected_execution_cost=1.0,
            expected_latency_s=1.0,
        ),
        SkillContract(
            skill_id="retreat_from_target",
            skill_type=SkillType.RECOVERY,
            language_description="Retreat the robot from the current target to a safer pose.",
            executor_binding=executor_binding,
            required_resources=frozenset({"base"}),
            expected_postcondition="robot retreated from target",
            value_function_id="retreat_success",
            terminal_error_codes=("retreat_failed",),
            min_localization_confidence=0.25,
            expected_execution_cost=0.5,
            expected_latency_s=0.5,
        ),
    ]


def default_executor_registry(
    config: ExecutorConfig,
    bindings: XLeRobotAgentBindings | None = None,
    offload_client: Any | None = None,
):
    from .executors import ExecutorRegistry

    bindings = bindings or XLeRobotAgentBindings()
    registry = ExecutorRegistry()

    def remote_handler(skill: SkillContract, goal: GoalContext, world_state: WorldState):
        if offload_client is None:
            return None
        try:
            return offload_client.execute_skill(skill, goal, world_state)
        except Exception as exc:
            return make_blocked_result(
                skill,
                world_state,
                "remote_offload_failed",
                evidence=f"Remote offload failed for `{skill.skill_id}`: {exc}",
                replan_hint="remote_offload_failed",
                readiness_state=world_state.readiness_state,
            )

    def generic_handler(skill: SkillContract, goal: GoalContext, world_state: WorldState):
        remote_result = remote_handler(skill, goal, world_state)
        if remote_result is not None:
            return remote_result
        return make_success_result(skill, world_state, evidence=f"Executed generic skill `{skill.skill_id}`.")

    def vla_nav_handler(skill: SkillContract, goal: GoalContext, world_state: WorldState):
        remote_result = remote_handler(skill, goal, world_state)
        if remote_result is not None:
            return remote_result
        readiness = (
            ReadinessState.NAVIGATION_READY_POSE
            if skill.skill_type == SkillType.NAVIGATION
            else ReadinessState.SKILL_READY_POSE
        )
        return make_success_result(
            skill,
            world_state,
            evidence=f"Executed navigation skill `{skill.skill_id}` via learned VLA navigation.",
            readiness_state=readiness,
        )

    def delegated_handler(skill: SkillContract, goal: GoalContext, world_state: WorldState):
        remote_result = remote_handler(skill, goal, world_state)
        if remote_result is not None:
            return remote_result
        if config.delegated_navigation_backend is None:
            return make_blocked_result(
                skill,
                world_state,
                "missing_delegated_backend",
                evidence="Delegated navigation mode was selected without a backend.",
                replan_hint="configure_delegated_backend",
            )
        readiness = (
            ReadinessState.NAVIGATION_READY_POSE
            if skill.skill_type == SkillType.NAVIGATION
            else ReadinessState.SKILL_READY_POSE
        )
        return make_success_result(
            skill,
            world_state,
            evidence=(
                f"Executed navigation skill `{skill.skill_id}` via delegated navigation backend "
                f"`{config.delegated_navigation_backend.value}`."
            ),
            readiness_state=readiness,
        )

    registry.register(bindings.generic_skill_binding, StaticSkillExecutor(generic_handler))
    registry.register(bindings.vla_navigation_binding, StaticSkillExecutor(vla_nav_handler))
    registry.register(bindings.delegated_progressive_map_binding, StaticSkillExecutor(delegated_handler))
    registry.register(bindings.delegated_global_map_binding, StaticSkillExecutor(delegated_handler))
    return registry
