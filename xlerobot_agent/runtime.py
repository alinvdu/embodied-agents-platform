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

from .executors import ExecutorRegistry
from .models import AgentRunRecord, GoalContext, StepRecord, WorldState
from .registry import SkillRegistry
from .scoring import PromptPlanner
from .voice import VoiceCommandPipeline


@dataclass
class XLeRobotAgentRuntime:
    registry: SkillRegistry
    planner: PromptPlanner
    executors: ExecutorRegistry
    voice_pipeline: VoiceCommandPipeline | None = None
    history: list[str] = field(default_factory=list)

    def run_instruction(self, instruction: str, world_state: WorldState) -> AgentRunRecord:
        normalized = self.planner.normalize_instruction(instruction)
        goal = GoalContext(user_instruction=normalized, structured_goal=normalized)
        return self._run_goal(goal, normalized, world_state)

    def run_voice_transcript(self, transcript: str, world_state: WorldState) -> AgentRunRecord | None:
        if self.voice_pipeline is None:
            raise RuntimeError("voice pipeline is not configured")
        command = self.voice_pipeline.process_transcript(transcript)
        if command is None:
            return None
        goal = GoalContext(
            user_instruction=command.normalized_command,
            structured_goal=command.normalized_command,
        )
        return self._run_goal(goal, command.normalized_command, world_state)

    def _run_goal(self, goal: GoalContext, normalized_instruction: str, world_state: WorldState) -> AgentRunRecord:
        skills = self.registry.list_enabled()
        discovered_places = tuple(self.planner.discover_places(world_state))
        world_state.place_memories = discovered_places

        subgoals = tuple(self.planner.plan_subgoals(goal, world_state, skills))
        steps: list[StepRecord] = []
        for subgoal in subgoals:
            selected_skill, selected_score, ranked_scores = self.planner.select_skill_for_subgoal(
                goal,
                subgoal,
                world_state,
                skills,
            )
            execution_result = self.executors.execute(selected_skill, goal, world_state)
            self.history.append(selected_skill.skill_id)
            world_state.recent_execution_history = tuple(self.history)
            steps.append(
                StepRecord(
                    goal=goal,
                    subgoal=subgoal,
                    selected_skill=selected_skill,
                    selected_score=selected_score,
                    ranked_scores=tuple(ranked_scores),
                    execution_result=execution_result,
                )
            )

        return AgentRunRecord(
            goal=goal,
            normalized_instruction=normalized_instruction,
            discovered_places=discovered_places,
            subgoals=subgoals,
            steps=tuple(steps),
        )
