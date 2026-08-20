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

from dataclasses import dataclass, field
from typing import Callable, Protocol

from .models import ExecutionResult, ExecutionStatus, GoalContext, ReadinessState, SkillContract, WorldState


class SkillExecutor(Protocol):
    def execute(self, skill: SkillContract, goal: GoalContext, world_state: WorldState) -> ExecutionResult:
        ...


@dataclass
class StaticSkillExecutor:
    """Small callable-backed executor that keeps the architecture runnable."""

    handler: Callable[[SkillContract, GoalContext, WorldState], ExecutionResult]

    def execute(self, skill: SkillContract, goal: GoalContext, world_state: WorldState) -> ExecutionResult:
        return self.handler(skill, goal, world_state)


@dataclass
class ExecutorRegistry:
    _executors: dict[str, SkillExecutor] = field(default_factory=dict)

    def register(self, binding: str, executor: SkillExecutor) -> None:
        self._executors[binding] = executor

    def execute(self, skill: SkillContract, goal: GoalContext, world_state: WorldState) -> ExecutionResult:
        executor = self._executors.get(skill.executor_binding)
        if executor is None:
            return ExecutionResult(
                skill_id=skill.skill_id,
                status=ExecutionStatus.BLOCKED,
                error_code="missing_executor",
                postcondition_evidence=f"No executor registered for binding `{skill.executor_binding}`.",
                retry_budget_impact=1,
                updated_localization_confidence=world_state.localization_confidence,
                replan_hint="register_executor_binding",
                readiness_state=world_state.readiness_state,
            )
        return executor.execute(skill, goal, world_state)


def make_success_result(
    skill: SkillContract,
    world_state: WorldState,
    *,
    evidence: str | None = None,
    readiness_state: ReadinessState = ReadinessState.SKILL_READY_POSE,
) -> ExecutionResult:
    return ExecutionResult(
        skill_id=skill.skill_id,
        status=ExecutionStatus.SUCCEEDED,
        postcondition_evidence=evidence or skill.expected_postcondition,
        updated_localization_confidence=world_state.localization_confidence,
        readiness_state=readiness_state,
    )


def make_blocked_result(
    skill: SkillContract,
    world_state: WorldState,
    error_code: str,
    *,
    evidence: str | None = None,
    replan_hint: str | None = None,
    readiness_state: ReadinessState = ReadinessState.NOT_READY,
) -> ExecutionResult:
    return ExecutionResult(
        skill_id=skill.skill_id,
        status=ExecutionStatus.BLOCKED,
        error_code=error_code,
        postcondition_evidence=evidence,
        retry_budget_impact=1,
        updated_localization_confidence=world_state.localization_confidence,
        replan_hint=replan_hint,
        readiness_state=readiness_state,
    )
