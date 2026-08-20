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

from dataclasses import dataclass, field, replace
from typing import Any
import threading

from .environment import PlaygroundEnvironmentAdapter
from .llm import AgentLLMRouter
from .models import (
    ExecutionStatus,
    GoalContext,
    SkillContract,
    Subgoal,
    WorldState,
)
from .registry import SkillRegistry
from .reporting import AgentStopRequested, LiveAgentReport
from .scoring import PromptPlanner, build_prompt_planner
from .tools import ToolCallContext, ToolRegistry
from .visual_differencing import VisualDifferencingModule, VisualObservation
from .voice import VoiceCommandPipeline


@dataclass(frozen=True)
class ActionCandidate:
    action_type: str
    action_id: str
    label: str
    description: str
    score: float
    reasoning: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "action_id": self.action_id,
            "label": self.label,
            "description": self.description,
            "score": self.score,
            "reasoning": self.reasoning,
            "payload": self.payload,
        }


@dataclass(frozen=True)
class PlaygroundRunRecord:
    normalized_instruction: str
    discovered_places: tuple[str, ...]
    subgoals: tuple[str, ...]
    status: str
    final_summary: str


class PlaygroundAgentRuntime:
    def __init__(
        self,
        *,
        environment: PlaygroundEnvironmentAdapter,
        llm_router: AgentLLMRouter,
        tools: ToolRegistry,
        report: LiveAgentReport,
        voice_pipeline: VoiceCommandPipeline | None = None,
        planner: PromptPlanner | None = None,
        visual_differencer: VisualDifferencingModule | None = None,
        max_reflections: int = 2,
        max_steps: int = 16,
        offload_client: Any | None = None,
    ) -> None:
        self.environment = environment
        self.llm_router = llm_router
        self.tools = tools
        self.report = report
        self.voice_pipeline = voice_pipeline
        self.planner = planner or build_prompt_planner(self.llm_router.model_suite.planner)
        self.visual_differencer = visual_differencer or VisualDifferencingModule(self.llm_router)
        self.max_reflections = max_reflections
        self.max_steps = max_steps
        self.offload_client = offload_client

        self.registry: SkillRegistry = self.environment.build_skill_registry()
        self.executors = self.environment.build_executor_registry_with_offload(self.offload_client)

    def run_instruction(self, instruction: str) -> PlaygroundRunRecord:
        self.report.begin_run(instruction)
        try:
            world_state = self.environment.build_world_state()
            registration = None
            if self.offload_client is not None:
                registration = self.offload_client.register()
                world_state = self._hydrate_world_state_from_offload(world_state)
            normalized = self.planner.normalize_instruction(instruction)
            goal = GoalContext(user_instruction=normalized, structured_goal=normalized)
            world_state = self._record_initial_visual_context(goal.user_instruction, world_state)
            if self.offload_client is not None:
                self.report.add_event(
                    "brain_registered",
                    "Brain Registered",
                    f"Registered brain `{registration.brain_id}` with remote offload server `{registration.server_url}`.",
                    details=registration.__dict__,
                )
                self.offload_client.publish_state(world_state, reason="run_started")

            discovered_places = tuple(place.name for place in self.planner.discover_places(world_state))
            world_state.place_memories = tuple(self.planner.discover_places(world_state))
            subgoals = tuple(self.planner.plan_subgoals(goal, world_state, self.registry.list_enabled()))
            self.report.set_plan(
                normalized_instruction=normalized,
                discovered_places=list(discovered_places),
                subgoals=[subgoal.text for subgoal in subgoals],
            )
            self.report.add_event(
                "plan_created",
                "Plan Created",
                f"Created a {len(subgoals)}-step plan for the current command.",
                details={
                    "normalized_instruction": normalized,
                    "discovered_places": list(discovered_places),
                    "subgoals": [self._serialize_subgoal(subgoal) for subgoal in subgoals],
                    "world_state": self._serialize_world_state(world_state),
                },
            )

            steps_taken = 0
            for subgoal in subgoals:
                self.report.ensure_not_stopped()
                completed, world_state, used_steps = self._run_subgoal(goal, subgoal, world_state)
                steps_taken += used_steps
                if steps_taken >= self.max_steps:
                    raise RuntimeError("maximum step budget exceeded")
                if not completed:
                    raise RuntimeError(f"unable to complete subgoal `{subgoal.text}`")

            final_summary = "All subgoals completed."
            self.report.finish("completed", final_summary)
            return PlaygroundRunRecord(
                normalized_instruction=normalized,
                discovered_places=discovered_places,
                subgoals=tuple(subgoal.text for subgoal in subgoals),
                status="completed",
                final_summary=final_summary,
            )
        except AgentStopRequested:
            final_summary = "Execution stopped by the operator."
            self.report.finish("stopped", final_summary)
            return PlaygroundRunRecord(
                normalized_instruction=self.report.normalized_instruction,
                discovered_places=tuple(self.report.discovered_places),
                subgoals=tuple(self.report.subgoals),
                status="stopped",
                final_summary=final_summary,
            )
        except Exception as exc:
            final_summary = f"Execution failed: {exc}"
            self.report.add_event(
                "error",
                "Run Failed",
                final_summary,
                details={"error": str(exc)},
            )
            self.report.finish("failed", final_summary)
            return PlaygroundRunRecord(
                normalized_instruction=self.report.normalized_instruction,
                discovered_places=tuple(self.report.discovered_places),
                subgoals=tuple(self.report.subgoals),
                status="failed",
                final_summary=final_summary,
            )

    def run_voice_transcript(self, transcript: str) -> PlaygroundRunRecord | None:
        if self.voice_pipeline is None:
            raise RuntimeError("voice pipeline is not configured")
        command = self.voice_pipeline.process_transcript(transcript)
        if command is None:
            self.report.begin_run(transcript)
            self.report.add_event(
                "voice_ignored",
                "Voice Ignored",
                "Wake word was not detected in the provided transcript.",
                details={"transcript": transcript},
            )
            self.report.finish("ignored", "Wake word was not detected.")
            return None
        self.report.add_event(
            "voice_transcript",
            "Voice Transcript",
            f"Accepted a wake-word-triggered voice command: {command.normalized_command}",
            details={"raw_transcript": transcript},
        )
        return self.run_instruction(command.normalized_command)

    def _run_subgoal(
        self,
        goal: GoalContext,
        subgoal: Subgoal,
        world_state: WorldState,
    ) -> tuple[bool, WorldState, int]:
        self.report.set_active_subgoal(subgoal.text)
        preferred_action_id: str | None = None
        used_steps = 0
        for attempt in range(self.max_reflections + 1):
            self.report.wait_if_paused()
            self.report.ensure_not_stopped()
            candidates = self._build_action_candidates(goal, subgoal, world_state, attempt, preferred_action_id)
            if not candidates:
                return False, world_state, used_steps
            self.report.add_event(
                "action_candidates",
                "Candidates Scored",
                f"Scored {len(candidates)} action candidates for subgoal `{subgoal.text}`.",
                details={"subgoal": self._serialize_subgoal(subgoal), "candidates": [item.to_dict() for item in candidates]},
            )

            decision, trace = self.llm_router.select_action(
                instruction=goal.user_instruction,
                subgoal=subgoal,
                world_state=world_state,
                candidates=[item.to_dict() for item in candidates],
                preferred_action_id=preferred_action_id,
            )
            used_steps += 1
            self.report.add_event(
                "llm_decision",
                "Planner Decision",
                decision.summary,
                details={
                    "decision": decision.__dict__,
                    "subgoal": self._serialize_subgoal(subgoal),
                    "llm_trace": trace.__dict__,
                },
            )

            if decision.action_type == "finish":
                return True, world_state, used_steps

            if decision.action_type == "tool":
                previous_world_state = world_state
                tool_result, world_state, preferred_action_id = self._execute_tool(
                    goal,
                    subgoal,
                    world_state,
                    attempt,
                    decision.action_id,
                    candidates,
                )
                used_steps += 1
                world_state, visual_observation = self._record_visual_difference(
                    goal,
                    subgoal,
                    previous_world_state,
                    world_state,
                    action={"action_type": "tool", "action_id": tool_result.tool_id},
                )
                if self.offload_client is not None:
                    self.offload_client.publish_state(
                        world_state,
                        reason=f"tool:{tool_result.tool_id}",
                    )
                review, review_trace = self.llm_router.review_action(
                    instruction=goal.user_instruction,
                    subgoal=subgoal,
                    world_state=world_state,
                    action={"action_type": "tool", "action_id": tool_result.tool_id},
                    action_status=tool_result.status,
                    action_summary=self._merge_action_summary(tool_result.summary, visual_observation),
                )
                self.report.add_event(
                    "review",
                    "Critic Review",
                    review.summary,
                    details={"review": review.__dict__, "llm_trace": review_trace.__dict__},
                )
                if tool_result.resolved_subgoal:
                    return True, world_state, used_steps
                if tool_result.recommended_action_id:
                    self.report.add_event(
                        "replan",
                        "Replan Suggested",
                        f"Helper reasoning suggested `{tool_result.recommended_action_id}` as the next action.",
                        details={"recommended_action_id": tool_result.recommended_action_id},
                    )
                    continue
                continue

            skill = self.registry.get(self._unwrap_vla_skill_action_id(decision.action_id))
            previous_world_state = world_state
            execution_result = self.executors.execute(skill, goal, world_state)
            used_steps += 1
            world_state = self.environment.apply_execution_effects(world_state, subgoal, skill, execution_result)
            world_state, visual_observation = self._record_visual_difference(
                goal,
                subgoal,
                previous_world_state,
                world_state,
                action={"action_type": "skill", "action_id": skill.skill_id},
            )
            if self.offload_client is not None:
                self.offload_client.publish_state(
                    world_state,
                    reason=f"skill:{skill.skill_id}",
                )
            self.report.add_event(
                "skill_executed",
                "Skill Executed",
                f"Executed `{skill.skill_id}` with status `{execution_result.status.value}`.",
                details={
                    "skill": self._serialize_skill(skill),
                    "subgoal": self._serialize_subgoal(subgoal),
                    "execution_result": {
                        "status": execution_result.status.value,
                        "error_code": execution_result.error_code,
                        "postcondition_evidence": execution_result.postcondition_evidence,
                        "replan_hint": execution_result.replan_hint,
                        "readiness_state": execution_result.readiness_state.value,
                    },
                    "world_state": self._serialize_world_state(world_state),
                },
            )
            review, review_trace = self.llm_router.review_action(
                instruction=goal.user_instruction,
                subgoal=subgoal,
                world_state=world_state,
                action={"action_type": "skill", "action_id": skill.skill_id},
                action_status=execution_result.status,
                action_summary=self._merge_action_summary(
                    execution_result.postcondition_evidence or "",
                    visual_observation,
                ),
            )
            self.report.add_event(
                "review",
                "Critic Review",
                review.summary,
                details={
                    "review": review.__dict__,
                    "llm_trace": review_trace.__dict__,
                    "world_state": self._serialize_world_state(world_state),
                },
            )
            if review.success:
                return True, world_state, used_steps
            preferred_action_id = execution_result.replan_hint
            self.report.add_event(
                "replan",
                "Reflect And Replan",
                f"Subgoal `{subgoal.text}` needs another attempt.",
                details={"attempt": attempt + 1, "preferred_action_id": preferred_action_id},
            )

        return False, world_state, used_steps

    def _execute_tool(
        self,
        goal: GoalContext,
        subgoal: Subgoal,
        world_state: WorldState,
        attempt: int,
        tool_id: str,
        candidates: list[ActionCandidate],
    ) -> tuple[Any, WorldState, str | None]:
        payload: dict[str, Any] = {}
        if tool_id in {"perceive_scene", "ground_object_3d", "set_waypoint_from_object"}:
            payload["target"] = subgoal.target or subgoal.text
        if tool_id == "go_to_pose":
            payload["pose"] = subgoal.target or subgoal.text
            payload["constraints"] = {"position_tolerance": 0.2, "yaw_tolerance": 0.3}
        if tool_id == "get_map":
            payload["summary_only"] = True
            if subgoal.target:
                payload["target"] = subgoal.target
        if tool_id == "explore":
            payload["area"] = subgoal.target or world_state.current_pose
            payload["strategy"] = "frontier"
            payload["build_map"] = True
        if tool_id == "create_map":
            payload["session"] = f"agent_map_{attempt + 1}"
            payload["area"] = subgoal.target or world_state.current_pose
        if tool_id == "code_execution":
            code_result, code_trace = self.llm_router.generate_helper_code(
                instruction=goal.user_instruction,
                subgoal=subgoal,
                world_state=world_state,
                candidates=[item.to_dict() for item in candidates],
                question=f"How should the agent make progress on `{subgoal.text}`?",
            )
            payload["code"] = code_result.code
            self.report.add_event(
                "code_generated",
                "Helper Code Generated",
                code_result.summary,
                details={
                    "generated_code": code_result.code,
                    "reasoning_summary": code_result.reasoning_summary,
                    "llm_trace": code_trace.__dict__,
                },
            )
        context = ToolCallContext(
            goal=goal,
            subgoal=subgoal,
            world_state=world_state,
            attempts=attempt,
            payload=payload,
            candidates=[item.to_dict() for item in candidates],
            question=f"How should the agent make progress on `{subgoal.text}`?",
            recent_events=self.report.snapshot()["events"][-8:],
        )
        tool_result = self.tools.execute(tool_id, context)
        if tool_result.updated_world_state is not None:
            world_state = tool_result.updated_world_state
        self.report.add_event(
            "tool_executed",
            "Tool Executed",
            tool_result.summary,
            details={
                "tool_id": tool_result.tool_id,
                "status": tool_result.status.value,
                "details": tool_result.details,
                "recommended_action_id": tool_result.recommended_action_id,
                "recommended_action_type": tool_result.recommended_action_type,
            },
        )
        return tool_result, world_state, tool_result.recommended_action_id

    def _build_action_candidates(
        self,
        goal: GoalContext,
        subgoal: Subgoal,
        world_state: WorldState,
        attempt: int,
        preferred_action_id: str | None,
    ) -> list[ActionCandidate]:
        tool_ids = {tool.tool_id for tool in self.tools.list_tools()}
        map_available = self._map_is_available(world_state)
        scores = self.planner.score_skills_for_subgoal(goal, subgoal, world_state, self.registry.list_enabled())
        candidates: list[ActionCandidate] = []
        for score in scores:
            if not score.feasible:
                continue
            skill = self.registry.get(score.skill_id)
            wrapped_action_id = self._wrap_vla_skill_action_id(skill.skill_id)
            bonus = 0.15 if wrapped_action_id == preferred_action_id or skill.skill_id == preferred_action_id else 0.0
            candidates.append(
                ActionCandidate(
                    action_type="skill",
                    action_id=wrapped_action_id,
                    label="run_vla_skill",
                    description=f"Execute the registered VLA skill `{skill.skill_id}`. {skill.language_description}",
                    score=score.combined_score + bonus,
                    reasoning=score.reasoning or "",
                    payload={"skill_id": skill.skill_id, "skill_type": skill.skill_type.value},
                )
            )
        if subgoal.kind == "navigate" and "go_to_pose" in tool_ids:
            candidates.append(
                ActionCandidate(
                    action_type="tool",
                    action_id="go_to_pose",
                    label="go_to_pose",
                    description="Hand the move-base portion of the task to the simplified navigation service.",
                    score=0.24 + (0.14 if preferred_action_id == "go_to_pose" else 0.0),
                    reasoning="Useful when the agent wants a direct navigation primitive instead of a skill policy.",
                    payload={"pose": subgoal.target or subgoal.text},
                )
            )
        if subgoal.kind in {"navigate", "search", "general"} and "get_map" in tool_ids:
            candidates.append(
                ActionCandidate(
                    action_type="tool",
                    action_id="get_map",
                    label="get_map",
                    description="Inspect whether a reusable navigation map is already available.",
                    score=0.16 + (0.12 if preferred_action_id == "get_map" else 0.0),
                    reasoning="Useful before exploration or navigation when the robot may already have a map.",
                )
            )
        if not map_available and subgoal.kind in {"navigate", "search", "general"} and "create_map" in tool_ids:
            candidates.append(
                ActionCandidate(
                    action_type="tool",
                    action_id="create_map",
                    label="create_map",
                    description="Create a new map session when no usable map is available.",
                    score=0.15 + (0.1 if preferred_action_id == "create_map" else 0.0),
                    reasoning="Useful when the robot needs a map before it can navigate reliably.",
                )
            )
        if subgoal.kind in {"search", "general"} and "explore" in tool_ids:
            candidates.append(
                ActionCandidate(
                    action_type="tool",
                    action_id="explore",
                    label="explore",
                    description="Run a bounded exploration pass and optionally build map coverage.",
                    score=0.14 + (0.1 if preferred_action_id == "explore" else 0.0),
                    reasoning="Useful when the robot lacks coverage of the current area or needs to discover new space.",
                )
            )
        if subgoal.kind in {"search", "align", "manipulate"} and "perceive_scene" in tool_ids:
            candidates.append(
                ActionCandidate(
                    action_type="tool",
                    action_id="perceive_scene",
                    label="perceive_scene",
                    description="Refresh RGB-D scene understanding with segmentation-style 3D annotations.",
                    score=0.18 + (0.1 if preferred_action_id == "perceive_scene" else 0.0),
                    reasoning="Useful when the agent needs an updated scene summary before grounding or manipulation.",
                    payload={"target": subgoal.target or subgoal.text},
                )
            )
        if subgoal.target and subgoal.kind in {"search", "align", "manipulate"} and "ground_object_3d" in tool_ids:
            candidates.append(
                ActionCandidate(
                    action_type="tool",
                    action_id="ground_object_3d",
                    label="ground_object_3d",
                    description="Ground a text target into a segmented 3D object candidate.",
                    score=0.17 + (0.11 if preferred_action_id == "ground_object_3d" else 0.0),
                    reasoning="Useful when the agent needs a concrete object anchor rather than a generic scene summary.",
                    payload={"target": subgoal.target},
                )
            )
        if subgoal.target and subgoal.kind in {"align", "manipulate"} and "set_waypoint_from_object" in tool_ids:
            candidates.append(
                ActionCandidate(
                    action_type="tool",
                    action_id="set_waypoint_from_object",
                    label="set_waypoint_from_object",
                    description="Turn a grounded object anchor into an approach waypoint.",
                    score=0.15 + (0.12 if preferred_action_id == "set_waypoint_from_object" else 0.0),
                    reasoning="Useful when the agent wants to move to a perception-derived pose before running a skill.",
                    payload={"target": subgoal.target},
                )
            )
        candidates.append(
            ActionCandidate(
                action_type="tool",
                action_id="describe_world_state",
                label="describe_world_state",
                description="Summarize the current observations and memory before acting.",
                score=0.08,
                reasoning="Low-cost analysis tool for clarifying the current world state.",
            )
        )
        if preferred_action_id == "get_task_status" and "get_task_status" in tool_ids:
            candidates.append(
                ActionCandidate(
                    action_type="tool",
                    action_id="get_task_status",
                    label="get_task_status",
                    description="Inspect the latest delegated task after a navigation or mapping request.",
                    score=0.3,
                    reasoning="Useful immediately after a task-oriented navigation tool is launched.",
                )
            )
        if preferred_action_id == "cancel_task" and "cancel_task" in tool_ids:
            candidates.append(
                ActionCandidate(
                    action_type="tool",
                    action_id="cancel_task",
                    label="cancel_task",
                    description="Cancel the most recent delegated navigation or mapping task.",
                    score=0.22,
                    reasoning="Useful when the current delegated task should be aborted before replanning.",
                )
            )
        skill_candidates = [item for item in candidates if item.action_type == "skill"]
        if attempt > 0 or not skill_candidates:
            candidates.append(
                ActionCandidate(
                    action_type="tool",
                    action_id="code_execution",
                    label="code_execution",
                    description="Use bounded helper code to reason about the remaining options.",
                    score=0.22 if skill_candidates else 0.6,
                    reasoning="Fallback analysis path when normal skill selection is not making progress.",
                )
            )
        candidates.sort(key=lambda item: (item.score, item.action_type == "skill"), reverse=True)
        return candidates[:8]

    def _wrap_vla_skill_action_id(self, skill_id: str) -> str:
        return f"run_vla_skill::{skill_id}"

    def _unwrap_vla_skill_action_id(self, action_id: str) -> str:
        if action_id.startswith("run_vla_skill::"):
            return action_id.split("::", 1)[1]
        return action_id

    def _serialize_skill(self, skill: SkillContract) -> dict[str, Any]:
        return {
            "skill_id": skill.skill_id,
            "skill_type": skill.skill_type.value,
            "description": skill.language_description,
            "executor_binding": skill.executor_binding,
            "required_observations": sorted(skill.required_observations),
            "required_resources": sorted(skill.required_resources),
            "expected_postcondition": skill.expected_postcondition,
        }

    def _serialize_subgoal(self, subgoal: Subgoal) -> dict[str, Any]:
        return {"text": subgoal.text, "kind": subgoal.kind, "target": subgoal.target}

    def _serialize_world_state(self, world_state: WorldState) -> dict[str, Any]:
        return {
            "current_task": world_state.current_task,
            "current_pose": world_state.current_pose,
            "localization_confidence": world_state.localization_confidence,
            "visible_objects": sorted(world_state.visible_objects),
            "visible_landmarks": sorted(world_state.visible_landmarks),
            "image_descriptions": list(world_state.image_descriptions),
            "semantic_memory_summary": world_state.semantic_memory_summary,
            "spatial_memory_summary": world_state.spatial_memory_summary,
            "place_memories": [
                {"name": place.name, "confidence": place.confidence, "evidence": place.evidence}
                for place in world_state.place_memories
            ],
            "recent_execution_history": list(world_state.recent_execution_history),
            "available_observations": sorted(world_state.available_observations),
            "satisfied_preconditions": sorted(world_state.satisfied_preconditions),
            "metadata": world_state.metadata,
            "readiness_state": world_state.readiness_state.value,
        }

    def _map_is_available(self, world_state: WorldState) -> bool:
        metadata = world_state.metadata
        if isinstance(metadata.get("map"), dict) and metadata["map"]:
            return True
        if isinstance(metadata.get("navigation_map"), dict) and metadata["navigation_map"]:
            return True
        if isinstance(metadata.get("nav_map"), dict) and metadata["nav_map"]:
            return True
        maps = metadata.get("maps")
        return isinstance(maps, list) and any(isinstance(item, dict) and item for item in maps)

    def _hydrate_world_state_from_offload(self, world_state: WorldState) -> WorldState:
        if self.offload_client is None:
            return world_state
        try:
            snapshot = self.offload_client.brain_snapshot()
        except Exception:
            return world_state
        latest_state = snapshot.get("latest_state", {})
        latest_sensors = snapshot.get("latest_sensors", {})
        if not isinstance(latest_state, dict) and not isinstance(latest_sensors, dict):
            return world_state

        updated = replace(world_state)
        updated.metadata = dict(world_state.metadata)

        if isinstance(latest_state, dict):
            if world_state.current_pose == "unknown" and latest_state.get("current_pose"):
                updated.current_pose = str(latest_state["current_pose"])
            if not world_state.visible_objects and isinstance(latest_state.get("visible_objects"), list):
                updated.visible_objects = frozenset(str(item) for item in latest_state["visible_objects"])
            if not world_state.visible_landmarks and isinstance(latest_state.get("visible_landmarks"), list):
                updated.visible_landmarks = frozenset(str(item) for item in latest_state["visible_landmarks"])
            if not world_state.image_descriptions and isinstance(latest_state.get("image_descriptions"), list):
                updated.image_descriptions = tuple(str(item) for item in latest_state["image_descriptions"])
            latest_metadata = latest_state.get("metadata")
            if isinstance(latest_metadata, dict):
                merged_metadata = dict(latest_metadata)
                merged_metadata.update(updated.metadata)
                updated.metadata = merged_metadata

        if isinstance(latest_sensors, dict):
            sensors = (
                dict(updated.metadata.get("sensors", {}))
                if isinstance(updated.metadata.get("sensors"), dict)
                else {}
            )
            for key, value in latest_sensors.items():
                sensors.setdefault(str(key), value)
            updated.metadata["sensors"] = sensors
        return updated

    def _record_initial_visual_context(self, instruction: str, world_state: WorldState) -> WorldState:
        observation, trace = self.visual_differencer.describe_initial_scene(
            instruction=instruction,
            world_state=world_state,
        )
        updated_world_state = self._apply_visual_observation(world_state, observation, stage="initial")
        self.report.add_event(
            "visual_context",
            "Visual Context",
            observation.summary,
            details={
                "visual_observation": self._serialize_visual_observation(observation),
                "llm_trace": trace.__dict__ if trace is not None else None,
                "world_state": self._serialize_world_state(updated_world_state),
            },
        )
        return updated_world_state

    def _record_visual_difference(
        self,
        goal: GoalContext,
        subgoal: Subgoal,
        previous_world_state: WorldState,
        current_world_state: WorldState,
        *,
        action: dict[str, Any],
    ) -> tuple[WorldState, VisualObservation]:
        observation, trace = self.visual_differencer.describe_scene_difference(
            instruction=goal.user_instruction,
            subgoal=subgoal,
            previous_world_state=previous_world_state,
            current_world_state=current_world_state,
            action=action,
        )
        updated_world_state = self._apply_visual_observation(current_world_state, observation, stage="delta")
        self.report.add_event(
            "visual_diff",
            "Visual Difference",
            observation.summary,
            details={
                "action": action,
                "subgoal": self._serialize_subgoal(subgoal),
                "visual_observation": self._serialize_visual_observation(observation),
                "llm_trace": trace.__dict__ if trace is not None else None,
                "world_state": self._serialize_world_state(updated_world_state),
            },
        )
        return updated_world_state, observation

    def _apply_visual_observation(
        self,
        world_state: WorldState,
        observation: VisualObservation,
        *,
        stage: str,
    ) -> WorldState:
        updated = replace(world_state)
        updated.metadata = dict(world_state.metadata)
        vdm_metadata = (
            dict(updated.metadata.get("vdm", {}))
            if isinstance(updated.metadata.get("vdm"), dict)
            else {}
        )
        history = list(vdm_metadata.get("history", []))
        history.append(
            {
                "stage": stage,
                "summary": observation.summary,
                "reasoning_summary": observation.reasoning_summary,
                "task_completed": observation.task_completed,
                "change_detected": observation.change_detected,
                "task_relevant_attributes": list(observation.task_relevant_attributes),
            }
        )
        vdm_metadata["history"] = history[-8:]
        vdm_metadata["last_summary"] = observation.summary
        vdm_metadata["last_reasoning_summary"] = observation.reasoning_summary
        vdm_metadata["task_completed"] = observation.task_completed
        vdm_metadata["change_detected"] = observation.change_detected
        vdm_metadata["task_relevant_attributes"] = list(observation.task_relevant_attributes)
        updated.metadata["vdm"] = vdm_metadata

        descriptions = list(updated.image_descriptions)
        if not descriptions or descriptions[-1] != observation.summary:
            descriptions.append(observation.summary)
        updated.image_descriptions = tuple(descriptions[-16:])
        return updated

    def _merge_action_summary(self, action_summary: str, observation: VisualObservation) -> str:
        pieces = [action_summary.strip(), f"Visual feedback: {observation.summary}"]
        if observation.reasoning_summary:
            pieces.append(f"Visual reasoning: {observation.reasoning_summary}")
        return "\n".join(piece for piece in pieces if piece)

    def _serialize_visual_observation(self, observation: VisualObservation) -> dict[str, Any]:
        return {
            "summary": observation.summary,
            "reasoning_summary": observation.reasoning_summary,
            "task_completed": observation.task_completed,
            "change_detected": observation.change_detected,
            "task_relevant_attributes": list(observation.task_relevant_attributes),
            "delta": list(observation.delta),
            "scene": observation.scene,
        }


class PlaygroundAgentController:
    def __init__(self, runtime: PlaygroundAgentRuntime, report: LiveAgentReport) -> None:
        self.runtime = runtime
        self.report = report
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()

    def start_instruction(self, instruction: str) -> bool:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._thread = threading.Thread(
                target=self.runtime.run_instruction,
                args=(instruction,),
                daemon=True,
            )
            self._thread.start()
            return True

    def start_voice_transcript(self, transcript: str) -> bool:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._thread = threading.Thread(
                target=self.runtime.run_voice_transcript,
                args=(transcript,),
                daemon=True,
            )
            self._thread.start()
            return True

    def pause(self) -> None:
        self.report.request_pause()

    def resume(self) -> None:
        self.report.resume()

    def stop(self) -> None:
        self.report.request_stop()

    def snapshot(self) -> dict[str, Any]:
        return self.report.snapshot()
