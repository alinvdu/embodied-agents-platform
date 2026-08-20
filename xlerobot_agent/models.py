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
from enum import Enum
from typing import Any, Callable


class SkillType(str, Enum):
    MANIPULATION = "manipulation"
    NAVIGATION = "navigation"
    SEARCH = "search"
    ALIGNMENT = "alignment"
    RECOVERY = "recovery"


class NavigationSkillExecutionMode(str, Enum):
    VLA_NAVIGATION_SKILLS = "vla_navigation_skills"
    DELEGATED_NAVIGATION_MODULE = "delegated_navigation_module"


class DelegatedNavigationBackend(str, Enum):
    PROGRESSIVE_MAP = "progressive_map"
    GLOBAL_MAP = "global_map"


class ReadinessState(str, Enum):
    NOT_READY = "not_ready"
    NAVIGATION_READY_POSE = "navigation_ready_pose"
    PERCEPTION_READY_POSE = "perception_ready_pose"
    SKILL_READY_POSE = "skill_ready_pose"


class ExecutionStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"
    ABORTED = "aborted"
    IN_PROGRESS = "in_progress"


VerifierFn = Callable[["WorldState"], bool]


@dataclass(frozen=True)
class ExecutorConfig:
    navigation_skill_execution_mode: NavigationSkillExecutionMode
    delegated_navigation_backend: DelegatedNavigationBackend | None = None


@dataclass(frozen=True)
class GoalContext:
    user_instruction: str
    structured_goal: str | None = None


@dataclass(frozen=True)
class Subgoal:
    text: str
    kind: str = "general"
    target: str | None = None


@dataclass(frozen=True)
class PlaceMemory:
    name: str
    confidence: float
    evidence: str


@dataclass
class SkillContract:
    skill_id: str
    skill_type: SkillType
    language_description: str
    executor_binding: str
    preconditions: frozenset[str] = frozenset()
    required_pose_class: str | None = None
    required_observations: frozenset[str] = frozenset()
    required_resources: frozenset[str] = frozenset()
    expected_postcondition: str | None = None
    success_verifier: VerifierFn | None = None
    value_function_id: str | None = None
    terminal_error_codes: tuple[str, ...] = tuple()
    retry_cap: int = 2
    min_localization_confidence: float = 0.0
    enabled: bool = True
    expected_execution_cost: float = 1.0
    expected_latency_s: float = 1.0
    safety_risk: float = 0.0
    tags: frozenset[str] = frozenset()


@dataclass
class WorldState:
    current_task: str
    current_pose: str = "unknown"
    localization_confidence: float = 1.0
    visible_objects: frozenset[str] = frozenset()
    visible_landmarks: frozenset[str] = frozenset()
    image_descriptions: tuple[str, ...] = tuple()
    semantic_memory_summary: str = ""
    spatial_memory_summary: str = ""
    place_memories: tuple[PlaceMemory, ...] = tuple()
    active_resource_locks: frozenset[str] = frozenset()
    recent_execution_history: tuple[str, ...] = tuple()
    available_observations: frozenset[str] = frozenset()
    satisfied_preconditions: frozenset[str] = frozenset()
    affordance_predictions: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    executor_configuration: ExecutorConfig | None = None
    readiness_state: ReadinessState = ReadinessState.NOT_READY


@dataclass(frozen=True)
class CandidateSkillScore:
    skill_id: str
    p_useful: float
    p_success: float
    combined_score: float
    feasible: bool
    rejection_reasons: tuple[str, ...] = tuple()
    reasoning: str | None = None


@dataclass(frozen=True)
class ExecutionResult:
    skill_id: str
    status: ExecutionStatus
    error_code: str | None = None
    postcondition_evidence: str | None = None
    retry_budget_impact: int = 0
    updated_localization_confidence: float | None = None
    replan_hint: str | None = None
    readiness_state: ReadinessState = ReadinessState.NOT_READY


@dataclass(frozen=True)
class StepRecord:
    goal: GoalContext
    subgoal: Subgoal
    selected_skill: SkillContract
    selected_score: CandidateSkillScore
    ranked_scores: tuple[CandidateSkillScore, ...]
    execution_result: ExecutionResult


@dataclass(frozen=True)
class AgentRunRecord:
    goal: GoalContext
    normalized_instruction: str
    discovered_places: tuple[PlaceMemory, ...]
    subgoals: tuple[Subgoal, ...]
    steps: tuple[StepRecord, ...]
