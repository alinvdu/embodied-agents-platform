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

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

from multido_xlerobot import XLeRobotInterface

from .executors import ExecutorRegistry
from .integration import (
    XLeRobotAgentBindings,
    build_default_navigation_skills,
    create_executor_config,
    default_executor_registry,
)
from .models import (
    DelegatedNavigationBackend,
    ExecutionResult,
    ExecutorConfig,
    NavigationSkillExecutionMode,
    PlaceMemory,
    ReadinessState,
    SkillContract,
    SkillType,
    Subgoal,
    WorldState,
)
from .registry import SkillRegistry


def load_skill_catalog(path: str | Path, bindings: XLeRobotAgentBindings | None = None) -> list[SkillContract]:
    bindings = bindings or XLeRobotAgentBindings()
    raw = json.loads(Path(path).read_text())
    items = raw.get("skills", raw)
    skills: list[SkillContract] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("skill catalog entries must be JSON objects")
        payload = dict(item)
        payload["skill_type"] = SkillType(payload["skill_type"])
        payload.setdefault("executor_binding", bindings.generic_skill_binding)
        payload["preconditions"] = frozenset(payload.get("preconditions", []))
        payload["required_observations"] = frozenset(payload.get("required_observations", []))
        payload["required_resources"] = frozenset(payload.get("required_resources", []))
        payload["terminal_error_codes"] = tuple(payload.get("terminal_error_codes", ()))
        payload["tags"] = frozenset(payload.get("tags", []))
        skills.append(SkillContract(**payload))
    return skills


def build_playground_skill_registry(
    executor_config: ExecutorConfig,
    *,
    bindings: XLeRobotAgentBindings | None = None,
    extra_skills: list[SkillContract] | None = None,
    skill_catalog_path: str | Path | None = None,
) -> SkillRegistry:
    bindings = bindings or XLeRobotAgentBindings()
    registry = SkillRegistry()
    registry.register_many(build_default_navigation_skills(executor_config, bindings))
    registry.register_many(_default_manipulation_and_search_skills(bindings))
    if skill_catalog_path is not None:
        registry.register_many(load_skill_catalog(skill_catalog_path, bindings))
    if extra_skills:
        registry.register_many(extra_skills)
    return registry


@dataclass
class PlaygroundEnvironmentAdapter:
    backend: str
    initial_world_state: WorldState
    executor_config: ExecutorConfig
    repo_root: str | None = None
    bindings: XLeRobotAgentBindings = XLeRobotAgentBindings()
    extra_skills: list[SkillContract] | None = None
    skill_catalog_path: str | None = None

    def build_world_state(self) -> WorldState:
        world_state = replace(self.initial_world_state)
        metadata = dict(world_state.metadata)
        metadata.update(
            {
                "backend": self.backend,
                "environment": self.describe_environment(),
                "executor_config": {
                    "navigation_skill_execution_mode": self.executor_config.navigation_skill_execution_mode.value,
                    "delegated_navigation_backend": (
                        self.executor_config.delegated_navigation_backend.value
                        if self.executor_config.delegated_navigation_backend is not None
                        else None
                    ),
                },
            }
        )
        world_state.metadata = metadata
        world_state.executor_configuration = self.executor_config
        return world_state

    def build_skill_registry(self) -> SkillRegistry:
        return build_playground_skill_registry(
            self.executor_config,
            bindings=self.bindings,
            extra_skills=self.extra_skills,
            skill_catalog_path=self.skill_catalog_path,
        )

    def build_executor_registry(self) -> ExecutorRegistry:
        return default_executor_registry(self.executor_config, self.bindings)

    def build_executor_registry_with_offload(self, offload_client: Any | None = None) -> ExecutorRegistry:
        return default_executor_registry(self.executor_config, self.bindings, offload_client=offload_client)

    def describe_environment(self) -> dict[str, Any]:
        return {"backend": self.backend}

    def apply_execution_effects(
        self,
        world_state: WorldState,
        subgoal: Subgoal,
        skill: SkillContract,
        result: ExecutionResult,
    ) -> WorldState:
        updated = replace(world_state)
        updated.metadata = dict(world_state.metadata)
        updated.affordance_predictions = dict(world_state.affordance_predictions)
        updated.readiness_state = result.readiness_state
        if result.updated_localization_confidence is not None:
            updated.localization_confidence = result.updated_localization_confidence

        history = list(updated.recent_execution_history)
        history.append(skill.skill_id)
        updated.recent_execution_history = tuple(history[-25:])

        if result.status.value != "succeeded":
            return updated

        if skill.skill_type == SkillType.NAVIGATION and subgoal.target:
            updated.current_pose = subgoal.target
            place_memories = list(updated.place_memories)
            place_memories.append(
                PlaceMemory(
                    name=subgoal.target,
                    confidence=0.8,
                    evidence=f"successful_navigation:{skill.skill_id}",
                )
            )
            updated.place_memories = tuple(_dedupe_places(place_memories))

        if skill.skill_id == "search_for_target" and subgoal.target:
            updated.visible_objects = frozenset(set(updated.visible_objects) | {subgoal.target})
            updated.available_observations = frozenset(
                set(updated.available_observations) | {f"{subgoal.target}_visible"}
            )

        if skill.skill_id == "inspect_scene":
            updated.available_observations = frozenset(
                set(updated.available_observations) | {"scene_inspected"}
            )
            updated.image_descriptions = tuple(updated.image_descriptions) + (
                f"Inspection from {updated.current_pose} completed.",
            )

        if skill.skill_id == "open_fridge":
            updated.satisfied_preconditions = frozenset(set(updated.satisfied_preconditions) | {"fridge_open"})
            updated.available_observations = frozenset(set(updated.available_observations) | {"fridge_open"})
            updated.readiness_state = ReadinessState.SKILL_READY_POSE

        if skill.skill_id == "grab_bread_from_table":
            updated.satisfied_preconditions = frozenset(set(updated.satisfied_preconditions) | {"bread_grasped"})

        return updated


@dataclass
class SimPlaygroundEnvironmentAdapter(PlaygroundEnvironmentAdapter):
    sim_python_bin: str | None = None

    def describe_environment(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "sim_python_bin": self.sim_python_bin,
            "render_stack": "mani_skill + sapien + rerun",
        }


@dataclass
class RealPlaygroundEnvironmentAdapter(PlaygroundEnvironmentAdapter):
    def describe_environment(self) -> dict[str, Any]:
        summary: dict[str, Any] = {"backend": self.backend}
        if self.repo_root is None:
            summary["xlerobot"] = "repo_root_not_provided"
            return summary
        try:
            interface = XLeRobotInterface(self.repo_root)
            summary["xlerobot"] = interface.summary()
        except Exception as exc:
            summary["xlerobot_error"] = str(exc)
        return summary


def build_environment_adapter(
    *,
    backend: str,
    initial_world_state: WorldState,
    navigation_mode: NavigationSkillExecutionMode,
    delegated_backend: DelegatedNavigationBackend | None = None,
    repo_root: str | None = None,
    sim_python_bin: str | None = None,
    skill_catalog_path: str | None = None,
) -> PlaygroundEnvironmentAdapter:
    if (
        navigation_mode == NavigationSkillExecutionMode.DELEGATED_NAVIGATION_MODULE
        and delegated_backend is None
    ):
        delegated_backend = DelegatedNavigationBackend.GLOBAL_MAP
    executor_config = create_executor_config(navigation_mode, delegated_backend)
    if backend == "real":
        return RealPlaygroundEnvironmentAdapter(
            backend=backend,
            initial_world_state=initial_world_state,
            executor_config=executor_config,
            repo_root=repo_root,
            skill_catalog_path=skill_catalog_path,
        )
    return SimPlaygroundEnvironmentAdapter(
        backend=backend,
        initial_world_state=initial_world_state,
        executor_config=executor_config,
        repo_root=repo_root,
        sim_python_bin=sim_python_bin,
        skill_catalog_path=skill_catalog_path,
    )


def _default_manipulation_and_search_skills(bindings: XLeRobotAgentBindings) -> list[SkillContract]:
    return [
        SkillContract(
            skill_id="search_for_target",
            skill_type=SkillType.SEARCH,
            language_description="Search the scene with the available cameras to find a target object.",
            executor_binding=bindings.generic_skill_binding,
            required_resources=frozenset({"head"}),
            expected_postcondition="target searched for",
            value_function_id="search_success",
            expected_execution_cost=0.8,
            expected_latency_s=1.0,
        ),
        SkillContract(
            skill_id="inspect_scene",
            skill_type=SkillType.SEARCH,
            language_description="Inspect the current scene to improve world understanding before acting.",
            executor_binding=bindings.generic_skill_binding,
            required_resources=frozenset({"head"}),
            expected_postcondition="scene inspected",
            value_function_id="inspect_scene_success",
            expected_execution_cost=0.4,
            expected_latency_s=0.5,
        ),
        SkillContract(
            skill_id="open_fridge",
            skill_type=SkillType.MANIPULATION,
            language_description="Open the fridge door when the handle is visible and reachable.",
            executor_binding=bindings.generic_skill_binding,
            required_resources=frozenset({"left_arm", "right_arm"}),
            required_observations=frozenset({"fridge_visible"}),
            expected_postcondition="fridge is open",
            value_function_id="open_fridge_success",
            min_localization_confidence=0.55,
            expected_execution_cost=2.2,
            expected_latency_s=2.5,
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
            language_description="Gather pens from a desk and arrange them neatly.",
            executor_binding=bindings.generic_skill_binding,
            required_resources=frozenset({"left_arm", "right_arm"}),
            required_observations=frozenset({"desk_visible"}),
            expected_postcondition="pens organized",
            value_function_id="clean_desk_success",
        ),
    ]


def _dedupe_places(places: list[PlaceMemory]) -> list[PlaceMemory]:
    deduped: dict[str, PlaceMemory] = {}
    for place in places:
        current = deduped.get(place.name)
        if current is None or place.confidence > current.confidence:
            deduped[place.name] = place
    return list(deduped.values())
