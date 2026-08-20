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

import re
from dataclasses import dataclass, field
from typing import Protocol

from .llm import AgentLLMRouter, AgentModelSuite, ModelConfig
from .models import CandidateSkillScore, GoalContext, PlaceMemory, SkillContract, Subgoal, WorldState
from .prompts import (
    build_instruction_normalization_system_prompt,
    build_instruction_normalization_user_prompt,
    build_place_discovery_system_prompt,
    build_place_discovery_user_prompt,
    build_skill_selection_system_prompt,
    build_skill_selection_user_prompt,
    build_subgoal_planning_system_prompt,
    build_subgoal_planning_user_prompt,
)


def _clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(frozen=True)
class PromptSkillAssessment:
    skill_id: str
    goal_usefulness: float
    success_likelihood: float
    combined_score: float
    reasoning: str


class PromptClient(Protocol):
    def normalize_instruction(self, text: str) -> str:
        ...

    def discover_places(self, world_state: WorldState) -> list[PlaceMemory]:
        ...

    def plan_subgoals(
        self,
        *,
        goal: GoalContext,
        world_state: WorldState,
        skills: list[SkillContract],
    ) -> list[Subgoal]:
        ...

    def score_skills(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        subgoal: Subgoal,
        skills: list[SkillContract],
        goal: GoalContext,
        world_state: WorldState,
    ) -> list[PromptSkillAssessment]:
        ...


class MockPromptClient:
    """Small prompt-shaped mock that keeps the design simple and runnable."""

    _word_re = re.compile(r"[a-z0-9_]+")

    def normalize_instruction(self, text: str) -> str:
        return " ".join(text.strip().split())

    def discover_places(self, world_state: WorldState) -> list[PlaceMemory]:
        observed = " ".join(
            list(world_state.visible_objects)
            + list(world_state.visible_landmarks)
            + list(world_state.image_descriptions)
            + [world_state.semantic_memory_summary, world_state.spatial_memory_summary]
        ).lower()

        places: list[PlaceMemory] = list(world_state.place_memories)
        if world_state.current_pose != "unknown":
            places.append(
                PlaceMemory(
                    name=world_state.current_pose,
                    confidence=0.7,
                    evidence="current_pose",
                )
            )
        if any(token in observed for token in ("fridge", "oven", "sink", "kitchen")):
            places.append(PlaceMemory(name="kitchen", confidence=0.9, evidence="fridge_or_kitchen_features"))
        if any(token in observed for token in ("sofa", "tv", "living")):
            places.append(PlaceMemory(name="living_room", confidence=0.85, evidence="living_room_features"))
        if any(token in observed for token in ("bed", "pillow", "bedroom")):
            places.append(PlaceMemory(name="bedroom", confidence=0.85, evidence="bedroom_features"))

        deduped: dict[str, PlaceMemory] = {}
        for place in places:
            current = deduped.get(place.name)
            if current is None or place.confidence > current.confidence:
                deduped[place.name] = place
        return list(deduped.values())

    def plan_subgoals(
        self,
        *,
        goal: GoalContext,
        world_state: WorldState,
        skills: list[SkillContract],
    ) -> list[Subgoal]:
        text = f"{goal.user_instruction} {goal.structured_goal or ''}".lower()
        available_skill_ids = {skill.skill_id for skill in skills}
        subgoals: list[Subgoal] = []

        if "kitchen" in text and world_state.current_pose != "kitchen":
            subgoals.append(Subgoal(text="go to the kitchen", kind="navigate", target="kitchen"))
        if "fridge" in text and "search_for_target" in available_skill_ids and "fridge" not in world_state.visible_objects:
            subgoals.append(Subgoal(text="find the fridge", kind="search", target="fridge"))
        if "fridge" in text and "align_for_skill" in available_skill_ids:
            subgoals.append(Subgoal(text="align with the fridge", kind="align", target="fridge"))
        if "open" in text and "fridge" in text and "open_fridge" in available_skill_ids:
            subgoals.append(Subgoal(text="open the fridge", kind="manipulate", target="fridge"))

        if not subgoals:
            subgoals.append(Subgoal(text=self.normalize_instruction(goal.user_instruction), kind="general"))
        return subgoals

    def score_skills(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        subgoal: Subgoal,
        skills: list[SkillContract],
        goal: GoalContext,
        world_state: WorldState,
    ) -> list[PromptSkillAssessment]:
        del system_prompt, user_prompt, goal

        goal_words = set(self._word_re.findall(subgoal.text.lower()))
        assessments: list[PromptSkillAssessment] = []
        for skill in skills:
            skill_words = set(self._word_re.findall(f"{skill.skill_id} {skill.language_description}".lower()))
            overlap = len(goal_words & skill_words) / max(len(skill_words), 1)
            goal_usefulness = _clamp_probability(max(0.1, overlap))

            success_likelihood = world_state.affordance_predictions.get(skill.skill_id, 0.6)
            if skill.required_observations and not skill.required_observations.issubset(world_state.available_observations):
                success_likelihood *= 0.5
            if world_state.localization_confidence < skill.min_localization_confidence:
                success_likelihood *= 0.2
            if skill.required_resources & world_state.active_resource_locks:
                success_likelihood *= 0.1
            success_likelihood = _clamp_probability(success_likelihood)

            combined = goal_usefulness * success_likelihood
            assessments.append(
                PromptSkillAssessment(
                    skill_id=skill.skill_id,
                    goal_usefulness=goal_usefulness,
                    success_likelihood=success_likelihood,
                    combined_score=combined,
                    reasoning=(
                        f"Subgoal `{subgoal.text}` maps to skill `{skill.skill_id}` "
                        f"with usefulness={goal_usefulness:.2f} and executability={success_likelihood:.2f}."
                    ),
                )
            )
        return assessments


@dataclass
class LLMPromptClient:
    model_config: ModelConfig
    fallback: MockPromptClient = field(default_factory=MockPromptClient)

    def __post_init__(self) -> None:
        self._router = AgentLLMRouter(
            AgentModelSuite(
                planner=self.model_config,
                critic=self.model_config,
                coder=self.model_config,
            )
        )

    def normalize_instruction(self, text: str) -> str:
        if self.model_config.provider == "mock":
            return self.fallback.normalize_instruction(text)
        parsed = self._complete(
            build_instruction_normalization_system_prompt(),
            build_instruction_normalization_user_prompt(text),
        )
        normalized = str((parsed or {}).get("normalized_instruction", "")).strip()
        return normalized or self.fallback.normalize_instruction(text)

    def discover_places(self, world_state: WorldState) -> list[PlaceMemory]:
        fallback_places = self.fallback.discover_places(world_state)
        if self.model_config.provider == "mock":
            return fallback_places
        parsed = self._complete(
            build_place_discovery_system_prompt(),
            build_place_discovery_user_prompt(world_state),
        )
        payload = (parsed or {}).get("places", [])
        if not isinstance(payload, list):
            return fallback_places
        places: list[PlaceMemory] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            confidence = _clamp_probability(float(item.get("confidence", 0.5)))
            evidence = str(item.get("evidence", "llm_place_discovery")).strip() or "llm_place_discovery"
            places.append(PlaceMemory(name=name, confidence=confidence, evidence=evidence))
        return _dedupe_place_memories(places or fallback_places)

    def plan_subgoals(
        self,
        *,
        goal: GoalContext,
        world_state: WorldState,
        skills: list[SkillContract],
    ) -> list[Subgoal]:
        fallback_subgoals = self.fallback.plan_subgoals(goal=goal, world_state=world_state, skills=skills)
        if self.model_config.provider == "mock":
            return fallback_subgoals
        parsed = self._complete(
            build_subgoal_planning_system_prompt(),
            build_subgoal_planning_user_prompt(goal, world_state, skills),
        )
        payload = (parsed or {}).get("subgoals", [])
        if not isinstance(payload, list):
            return fallback_subgoals
        subgoals: list[Subgoal] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            subgoals.append(
                Subgoal(
                    text=text,
                    kind=str(item.get("kind", "general")).strip() or "general",
                    target=(str(item["target"]).strip() if item.get("target") is not None else None) or None,
                )
            )
        return subgoals or fallback_subgoals

    def score_skills(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        subgoal: Subgoal,
        skills: list[SkillContract],
        goal: GoalContext,
        world_state: WorldState,
    ) -> list[PromptSkillAssessment]:
        fallback_scores = self.fallback.score_skills(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            subgoal=subgoal,
            skills=skills,
            goal=goal,
            world_state=world_state,
        )
        if self.model_config.provider == "mock":
            return fallback_scores
        parsed = self._complete(system_prompt, user_prompt)
        payload = (parsed or {}).get("scores", [])
        if not isinstance(payload, list):
            return fallback_scores
        by_skill_id = {skill.skill_id: skill for skill in skills}
        assessments: list[PromptSkillAssessment] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            skill_id = str(item.get("skill_id", "")).strip()
            if skill_id not in by_skill_id:
                continue
            usefulness = _clamp_probability(float(item.get("goal_usefulness", 0.0)))
            success = _clamp_probability(float(item.get("success_likelihood", 0.0)))
            combined = item.get("combined_score")
            if combined is None:
                combined_value = usefulness * success
            else:
                combined_value = _clamp_probability(float(combined))
            assessments.append(
                PromptSkillAssessment(
                    skill_id=skill_id,
                    goal_usefulness=usefulness,
                    success_likelihood=success,
                    combined_score=combined_value,
                    reasoning=str(item.get("reasoning", "")).strip() or "LLM skill assessment",
                )
            )
        return assessments or fallback_scores

    def _complete(self, system_prompt: str, user_prompt: str) -> dict | None:
        parsed, _trace = self._router.complete_json_prompt(
            config=self.model_config,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        return parsed


@dataclass
class PromptPlanner:
    prompt_client: PromptClient

    def normalize_instruction(self, text: str) -> str:
        return self.prompt_client.normalize_instruction(text)

    def discover_places(self, world_state: WorldState) -> list[PlaceMemory]:
        return self.prompt_client.discover_places(world_state)

    def plan_subgoals(
        self,
        goal: GoalContext,
        world_state: WorldState,
        skills: list[SkillContract],
    ) -> list[Subgoal]:
        return self.prompt_client.plan_subgoals(goal=goal, world_state=world_state, skills=skills)

    def score_skills_for_subgoal(
        self,
        goal: GoalContext,
        subgoal: Subgoal,
        world_state: WorldState,
        skills: list[SkillContract],
    ) -> list[CandidateSkillScore]:
        feasible_skills: list[SkillContract] = []
        rejected_scores: list[CandidateSkillScore] = []

        for skill in skills:
            rejection_reasons = tuple(self._gate(skill, world_state))
            if rejection_reasons:
                rejected_scores.append(
                    CandidateSkillScore(
                        skill_id=skill.skill_id,
                        p_useful=0.0,
                        p_success=0.0,
                        combined_score=0.0,
                        feasible=False,
                        rejection_reasons=rejection_reasons,
                        reasoning="Rejected by deterministic feasibility gate.",
                    )
                )
            else:
                feasible_skills.append(skill)

        assessments_by_id: dict[str, PromptSkillAssessment] = {}
        if feasible_skills:
            system_prompt = build_skill_selection_system_prompt()
            user_prompt = build_skill_selection_user_prompt(goal, subgoal, world_state, feasible_skills)
            assessments_by_id = {
                item.skill_id: item
                for item in self.prompt_client.score_skills(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    subgoal=subgoal,
                    skills=feasible_skills,
                    goal=goal,
                    world_state=world_state,
                )
            }

        feasible_scores: list[CandidateSkillScore] = []
        for skill in feasible_skills:
            assessment = assessments_by_id.get(skill.skill_id)
            if assessment is None:
                feasible_scores.append(
                    CandidateSkillScore(
                        skill_id=skill.skill_id,
                        p_useful=0.0,
                        p_success=0.0,
                        combined_score=0.0,
                        feasible=False,
                        rejection_reasons=("prompt_missing_skill_assessment",),
                        reasoning="Prompt client did not return an assessment for this skill.",
                    )
                )
                continue
            feasible_scores.append(
                CandidateSkillScore(
                    skill_id=skill.skill_id,
                    p_useful=assessment.goal_usefulness,
                    p_success=assessment.success_likelihood,
                    combined_score=assessment.combined_score,
                    feasible=True,
                    rejection_reasons=tuple(),
                    reasoning=assessment.reasoning,
                )
            )

        all_scores = feasible_scores + rejected_scores
        skill_by_id = {skill.skill_id: skill for skill in skills}
        all_scores.sort(
            key=lambda item: (
                item.feasible,
                item.combined_score,
                -skill_by_id[item.skill_id].safety_risk,
                -skill_by_id[item.skill_id].expected_execution_cost,
                -skill_by_id[item.skill_id].expected_latency_s,
            ),
            reverse=True,
        )
        return all_scores

    def select_skill_for_subgoal(
        self,
        goal: GoalContext,
        subgoal: Subgoal,
        world_state: WorldState,
        skills: list[SkillContract],
    ) -> tuple[SkillContract, CandidateSkillScore, list[CandidateSkillScore]]:
        all_scores = self.score_skills_for_subgoal(goal, subgoal, world_state, skills)
        if not all_scores or not all_scores[0].feasible:
            reasons = ", ".join(all_scores[0].rejection_reasons) if all_scores else "no candidates available"
            raise RuntimeError(f"No feasible skills available for subgoal `{subgoal.text}`: {reasons}")
        best = all_scores[0]
        skill_by_id = {skill.skill_id: skill for skill in skills}
        return skill_by_id[best.skill_id], best, all_scores

    def _gate(self, skill: SkillContract, world_state: WorldState) -> list[str]:
        reasons: list[str] = []
        if not skill.enabled:
            reasons.append("skill_disabled")
        if not skill.executor_binding:
            reasons.append("missing_executor_binding")
        if skill.required_resources & world_state.active_resource_locks:
            reasons.append("required_resource_locked")
        if world_state.localization_confidence < skill.min_localization_confidence:
            reasons.append("localization_below_threshold")
        if skill.required_observations and not skill.required_observations.issubset(world_state.available_observations):
            reasons.append("required_observations_missing")
        if skill.preconditions and not skill.preconditions.issubset(world_state.satisfied_preconditions):
            reasons.append("preconditions_unsatisfied")
        return reasons


def build_prompt_planner(model_config: ModelConfig | None = None) -> PromptPlanner:
    if model_config is None or model_config.provider == "mock":
        return PromptPlanner(prompt_client=MockPromptClient())
    return PromptPlanner(prompt_client=LLMPromptClient(model_config))


def _dedupe_place_memories(places: list[PlaceMemory]) -> list[PlaceMemory]:
    deduped: dict[str, PlaceMemory] = {}
    for place in places:
        current = deduped.get(place.name)
        if current is None or place.confidence > current.confidence:
            deduped[place.name] = place
    return list(deduped.values())
