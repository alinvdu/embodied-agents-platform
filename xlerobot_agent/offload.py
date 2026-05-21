from __future__ import annotations

import base64
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Mapping

from .exploration import ExplorationBackend, ExplorationBackendConfig
from .models import (
    ExecutionResult,
    ExecutionStatus,
    GoalContext,
    ReadinessState,
    SkillContract,
    SkillType,
    Subgoal,
    WorldState,
)
from .perception_service import execute_perception_tool, extract_scene_from_tool_result


def serialize_goal_context(goal: GoalContext) -> dict[str, Any]:
    return {
        "user_instruction": goal.user_instruction,
        "structured_goal": goal.structured_goal,
    }


def serialize_subgoal(subgoal: Subgoal) -> dict[str, Any]:
    return {
        "text": subgoal.text,
        "kind": subgoal.kind,
        "target": subgoal.target,
    }


def serialize_skill_contract(skill: SkillContract) -> dict[str, Any]:
    return {
        "skill_id": skill.skill_id,
        "skill_type": skill.skill_type.value,
        "language_description": skill.language_description,
        "executor_binding": skill.executor_binding,
        "preconditions": sorted(skill.preconditions),
        "required_pose_class": skill.required_pose_class,
        "required_observations": sorted(skill.required_observations),
        "required_resources": sorted(skill.required_resources),
        "expected_postcondition": skill.expected_postcondition,
        "value_function_id": skill.value_function_id,
        "terminal_error_codes": list(skill.terminal_error_codes),
        "retry_cap": skill.retry_cap,
        "min_localization_confidence": skill.min_localization_confidence,
        "enabled": skill.enabled,
        "expected_execution_cost": skill.expected_execution_cost,
        "expected_latency_s": skill.expected_latency_s,
        "safety_risk": skill.safety_risk,
        "tags": sorted(skill.tags),
    }


def serialize_world_state(world_state: WorldState) -> dict[str, Any]:
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
        "active_resource_locks": sorted(world_state.active_resource_locks),
        "recent_execution_history": list(world_state.recent_execution_history),
        "available_observations": sorted(world_state.available_observations),
        "satisfied_preconditions": sorted(world_state.satisfied_preconditions),
        "affordance_predictions": dict(world_state.affordance_predictions),
        "metadata": _to_json_safe(world_state.metadata),
        "readiness_state": world_state.readiness_state.value,
    }


def serialize_execution_result(result: ExecutionResult) -> dict[str, Any]:
    return {
        "skill_id": result.skill_id,
        "status": result.status.value,
        "error_code": result.error_code,
        "postcondition_evidence": result.postcondition_evidence,
        "retry_budget_impact": result.retry_budget_impact,
        "updated_localization_confidence": result.updated_localization_confidence,
        "replan_hint": result.replan_hint,
        "readiness_state": result.readiness_state.value,
    }


def execution_result_from_payload(payload: dict[str, Any], fallback_skill_id: str) -> ExecutionResult:
    return ExecutionResult(
        skill_id=str(payload.get("skill_id", fallback_skill_id)),
        status=ExecutionStatus(str(payload.get("status", ExecutionStatus.BLOCKED.value))),
        error_code=payload.get("error_code"),
        postcondition_evidence=payload.get("postcondition_evidence"),
        retry_budget_impact=int(payload.get("retry_budget_impact", 0)),
        updated_localization_confidence=payload.get("updated_localization_confidence"),
        replan_hint=payload.get("replan_hint"),
        readiness_state=ReadinessState(str(payload.get("readiness_state", ReadinessState.NOT_READY.value))),
    )


def _to_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, bytes):
        return {
            "data_base64": base64.b64encode(value).decode("utf-8"),
            "shape": [len(value)],
            "dtype": "uint8",
        }
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return item_method()
        except Exception:
            pass
    if all(hasattr(value, attr) for attr in ("tobytes", "shape", "dtype")):
        try:
            return {
                "data_base64": base64.b64encode(value.tobytes()).decode("utf-8"),
                "shape": [int(item) for item in value.shape],
                "dtype": str(value.dtype),
            }
        except Exception:
            pass
    return str(value)


@dataclass(frozen=True)
class BrainRegistration:
    brain_id: str
    brain_name: str
    server_url: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OffloadServerConfig:
    host: str = "0.0.0.0"
    port: int = 8890
    nav2_service_url: str | None = None
    perception_service_url: str | None = None
    vla_service_url: str | None = None
    state_log_path: str | None = None


@dataclass
class BrainRecord:
    brain_id: str
    brain_name: str
    metadata: dict[str, Any] = field(default_factory=dict)
    registered_at: float = field(default_factory=time.time)
    last_heartbeat_at: float = field(default_factory=time.time)
    last_publish_at: float | None = None
    last_publish_reason: str | None = None
    latest_state: dict[str, Any] = field(default_factory=dict)
    latest_sensors: dict[str, Any] = field(default_factory=dict)
    latest_scene: dict[str, Any] = field(default_factory=dict)
    tasks: dict[str, dict[str, Any]] = field(default_factory=dict)
    synthetic_map: dict[str, Any] | None = None
    exploration_backend: ExplorationBackend | None = field(default=None, repr=False, compare=False)

    def summary(self) -> dict[str, Any]:
        return {
            "brain_id": self.brain_id,
            "brain_name": self.brain_name,
            "metadata": self.metadata,
            "registered_at": self.registered_at,
            "last_heartbeat_at": self.last_heartbeat_at,
            "last_publish_at": self.last_publish_at,
            "last_publish_reason": self.last_publish_reason,
            "latest_state": self.latest_state,
            "latest_sensors": self.latest_sensors,
            "latest_scene": self.latest_scene,
            "task_count": len(self.tasks),
            "synthetic_map_available": self.synthetic_map is not None,
            "map_available": (self.exploration_backend.get_map() is not None) if self.exploration_backend is not None else self.synthetic_map is not None,
        }


class _BrainRegistry:
    def __init__(self, config: OffloadServerConfig) -> None:
        self.config = config
        self._lock = threading.RLock()
        self._brains: dict[str, BrainRecord] = {}

    def register(self, brain_name: str, metadata: dict[str, Any], brain_id: str | None = None) -> BrainRecord:
        with self._lock:
            resolved_id = brain_id or f"brain-{uuid.uuid4().hex[:12]}"
            record = self._brains.get(resolved_id)
            if record is None:
                record = BrainRecord(brain_id=resolved_id, brain_name=brain_name, metadata=dict(metadata))
                self._brains[resolved_id] = record
            else:
                record.brain_name = brain_name
                record.metadata = dict(metadata)
                record.last_heartbeat_at = time.time()
            self._persist_state()
            return record

    def heartbeat(self, brain_id: str) -> BrainRecord:
        with self._lock:
            record = self._brains[brain_id]
            record.last_heartbeat_at = time.time()
            self._persist_state()
            return record

    def update_state(
        self,
        brain_id: str,
        *,
        world_state: dict[str, Any],
        sensors: dict[str, Any],
        reason: str,
    ) -> BrainRecord:
        with self._lock:
            record = self._brains[brain_id]
            record.latest_state = dict(world_state)
            record.latest_sensors = dict(sensors)
            record.last_publish_reason = reason
            record.last_publish_at = time.time()
            record.last_heartbeat_at = time.time()
            self._persist_state()
            return record

    def get(self, brain_id: str) -> BrainRecord:
        with self._lock:
            return self._brains[brain_id]

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [record.summary() for record in self._brains.values()]

    def _persist_state(self) -> None:
        if self.config.state_log_path is None:
            return
        path = Path(self.config.state_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "brains": [record.summary() for record in self._brains.values()],
            "updated_at": time.time(),
        }
        path.write_text(json.dumps(payload, indent=2))

    def persist(self) -> None:
        with self._lock:
            self._persist_state()


class _OffloadRouter:
    def __init__(self, config: OffloadServerConfig, registry: _BrainRegistry) -> None:
        self.config = config
        self.registry = registry

    def _exploration_backend(self, record: BrainRecord) -> ExplorationBackend:
        if record.exploration_backend is None:
            persist_path = None
            if self.config.state_log_path is not None:
                base = Path(self.config.state_log_path)
                persist_path = str(base.with_name(f"{base.stem}_{record.brain_id}_map.json"))
            record.exploration_backend = ExplorationBackend(
                ExplorationBackendConfig(mode="offload", persist_path=persist_path)
            )
        return record.exploration_backend

    def execute_tool(self, brain_id: str, tool_id: str, context: dict[str, Any]) -> dict[str, Any]:
        record = self.registry.get(brain_id)
        target_url = self._service_url_for_tool(tool_id)
        payload = {
            "brain": record.summary(),
            "tool_id": tool_id,
            "context": context,
        }
        if target_url is not None:
            try:
                result = self._proxy_json(f"{target_url.rstrip('/')}/api/tools/{tool_id}", payload)
                self._update_scene_cache(record, result)
                return result
            except Exception as exc:
                return {
                    "tool_id": tool_id,
                    "status": ExecutionStatus.FAILED.value,
                    "summary": f"Remote offload tool call failed: {exc}",
                    "details": {"error": str(exc)},
                    "resolved_subgoal": False,
                }
        return self._stub_tool(record, tool_id, context)

    def execute_skill(
        self,
        brain_id: str,
        skill: dict[str, Any],
        goal: dict[str, Any],
        world_state: dict[str, Any],
    ) -> dict[str, Any]:
        record = self.registry.get(brain_id)
        self.registry.update_state(
            brain_id,
            world_state=world_state,
            sensors=record.latest_sensors,
            reason=f"skill:{skill['skill_id']}",
        )
        target_url = self._service_url_for_skill(skill)
        payload = {
            "brain": self.registry.get(brain_id).summary(),
            "skill": skill,
            "goal": goal,
            "world_state": world_state,
        }
        if target_url is not None:
            try:
                return self._proxy_json(f"{target_url.rstrip('/')}/api/skills/execute", payload)
            except Exception as exc:
                return {
                    "skill_id": skill["skill_id"],
                    "status": ExecutionStatus.FAILED.value,
                    "error_code": "remote_offload_failed",
                    "postcondition_evidence": f"Remote skill execution failed: {exc}",
                    "retry_budget_impact": 1,
                    "updated_localization_confidence": world_state.get("localization_confidence"),
                    "replan_hint": "remote_offload_failed",
                    "readiness_state": world_state.get("readiness_state", ReadinessState.NOT_READY.value),
                }
        return self._stub_skill(skill, world_state)

    def mapping_snapshot(self, brain_id: str) -> dict[str, Any]:
        record = self.registry.get(brain_id)
        return self._exploration_backend(record).snapshot()

    def start_mapping_task(
        self,
        brain_id: str,
        *,
        tool_id: str,
        area: str,
        session: str | None = None,
        source: str = "operator",
    ) -> dict[str, Any]:
        record = self.registry.get(brain_id)
        backend = self._exploration_backend(record)
        world_state = dict(record.latest_state)
        if tool_id == "explore":
            return backend.start_explore(
                area=area,
                session=session,
                source=source,
                build_map=True,
                world_state=world_state,
            )
        if tool_id == "create_map":
            return backend.start_create_map(
                session=session or f"map_{uuid.uuid4().hex[:8]}",
                area=area,
                source=source,
                world_state=world_state,
            )
        raise ValueError(f"unsupported mapping tool `{tool_id}`")

    def pause_mapping_task(self, brain_id: str, task_id: str) -> dict[str, Any] | None:
        record = self.registry.get(brain_id)
        return self._exploration_backend(record).pause_task(task_id)

    def resume_mapping_task(self, brain_id: str, task_id: str) -> dict[str, Any] | None:
        record = self.registry.get(brain_id)
        return self._exploration_backend(record).resume_task(task_id)

    def cancel_mapping_task(self, brain_id: str, task_id: str | None = None) -> dict[str, Any] | None:
        record = self.registry.get(brain_id)
        return self._exploration_backend(record).cancel_task(task_id)

    def update_region(
        self,
        brain_id: str,
        region_id: str,
        *,
        label: str | None = None,
        purpose: str | None = None,
        polygon_2d: list[list[float]] | None = None,
        default_waypoints: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        record = self.registry.get(brain_id)
        return self._exploration_backend(record).update_region(
            region_id,
            label=label,
            purpose=purpose,
            polygon_2d=polygon_2d,
            default_waypoints=default_waypoints,
        )

    def merge_regions(self, brain_id: str, region_ids: list[str], *, new_label: str | None = None) -> dict[str, Any] | None:
        record = self.registry.get(brain_id)
        return self._exploration_backend(record).merge_regions(region_ids, new_label=new_label)

    def split_region(self, brain_id: str, region_id: str, polygons: list[list[list[float]]] | None = None) -> list[dict[str, Any]]:
        record = self.registry.get(brain_id)
        return self._exploration_backend(record).split_region(region_id, polygons)

    def set_named_place(self, brain_id: str, name: str, pose: dict[str, Any], *, region_id: str | None = None) -> dict[str, Any] | None:
        record = self.registry.get(brain_id)
        return self._exploration_backend(record).set_named_place(name, pose, region_id=region_id)

    def approve_map(self, brain_id: str) -> dict[str, Any] | None:
        record = self.registry.get(brain_id)
        return self._exploration_backend(record).approve_current_map()

    def _service_url_for_tool(self, tool_id: str) -> str | None:
        if tool_id in {"go_to_pose", "get_map", "explore", "create_map", "get_task_status", "cancel_task"}:
            return self.config.nav2_service_url
        if tool_id in {"perceive_scene", "ground_object_3d", "set_waypoint_from_object"} or tool_id.startswith("orbbec_"):
            return self.config.perception_service_url
        if tool_id.startswith("run_vla") or tool_id.startswith("vla_"):
            return self.config.vla_service_url
        return None

    def _service_url_for_skill(self, skill: dict[str, Any]) -> str | None:
        skill_type = str(skill.get("skill_type", ""))
        if skill_type in {SkillType.NAVIGATION.value, SkillType.ALIGNMENT.value, SkillType.RECOVERY.value}:
            return self.config.nav2_service_url
        if skill_type in {SkillType.MANIPULATION.value, SkillType.SEARCH.value}:
            return self.config.vla_service_url
        return None

    def _stub_tool(self, record: BrainRecord, tool_id: str, context: dict[str, Any]) -> dict[str, Any]:
        subgoal = context.get("subgoal", {})
        payload = context.get("payload", {})
        world_state = dict(record.latest_state or context.get("world_state", {}))
        target = payload.get("target") or payload.get("pose") or subgoal.get("target") or subgoal.get("text")
        exploration = self._exploration_backend(record)
        if tool_id in {"perceive_scene", "ground_object_3d", "set_waypoint_from_object"}:
            result = execute_perception_tool(tool_id, context=context, brain=record.summary())
            self._update_scene_cache(record, result)
            return result
        if tool_id == "go_to_pose":
            resolved_pose, pose_label = self._resolve_pose(target, self._map_info(record, world_state))
            task = self._create_task(
                record,
                tool_id="go_to_pose",
                message=f"Reached `{pose_label}` through the simplified offload stub.",
                result={
                    "goal_pose": resolved_pose,
                    "goal_label": pose_label,
                    "constraints": dict(payload.get("constraints", {})),
                },
            )
            return {
                "tool_id": tool_id,
                "status": ExecutionStatus.SUCCEEDED.value,
                "summary": f"Accepted a high-level navigation request toward `{pose_label}`.",
                "details": {
                    "task": task,
                    "pose": resolved_pose,
                    "pose_label": pose_label,
                    "constraints": dict(payload.get("constraints", {})),
                },
                "resolved_subgoal": subgoal.get("kind") == "navigate",
            }
        if tool_id == "get_map":
            map_info = self._map_info(record, world_state)
            if map_info is None:
                return {
                    "tool_id": tool_id,
                    "status": ExecutionStatus.SUCCEEDED.value,
                    "summary": "No navigation map is currently available on the Ubuntu offload node.",
                    "details": {"available": False, "map": None},
                    "resolved_subgoal": False,
            }
            recommended_action_id = None
            named_places = {
                item.get("name", item)
                for item in map_info.get("named_places", [])
                if isinstance(item, dict) or isinstance(item, str)
            }
            region_labels = {
                item.get("label")
                for item in map_info.get("regions", [])
                if isinstance(item, dict) and item.get("label")
            }
            target_name = subgoal.get("target") or payload.get("target")
            if target_name and (target_name in named_places or target_name in region_labels):
                recommended_action_id = "go_to_pose"
            return {
                "tool_id": tool_id,
                "status": ExecutionStatus.SUCCEEDED.value,
                "summary": f"Navigation map `{map_info.get('map_id', 'unknown_map')}` is available.",
                "details": {"available": True, "map": map_info},
                "resolved_subgoal": False,
                "recommended_action_id": recommended_action_id,
            }
        if tool_id == "explore":
            area = str(payload.get("area") or subgoal.get("target") or world_state.get("current_pose") or "workspace")
            strategy = str(payload.get("strategy", "frontier"))
            task = exploration.start_explore(
                area=area,
                session=f"explore_{uuid.uuid4().hex[:8]}",
                source="planner",
                build_map=bool(payload.get("build_map", True)),
                world_state=world_state,
            )
            return {
                "tool_id": tool_id,
                "status": ExecutionStatus.IN_PROGRESS.value,
                "summary": f"Started exploration for `{area}` using strategy `{strategy}`.",
                "details": {"task": task, "area": area, "strategy": strategy},
                "resolved_subgoal": False,
                "recommended_action_id": "get_task_status",
            }
        if tool_id == "create_map":
            session = str(payload.get("session") or f"map_{uuid.uuid4().hex[:8]}")
            area = str(payload.get("area") or subgoal.get("target") or world_state.get("current_pose") or "workspace")
            task = exploration.start_create_map(
                session=session,
                area=area,
                source="planner",
                world_state=world_state,
            )
            return {
                "tool_id": tool_id,
                "status": ExecutionStatus.IN_PROGRESS.value,
                "summary": f"Started a navigation map build for `{area}`.",
                "details": {
                    "task": task,
                },
                "resolved_subgoal": False,
                "recommended_action_id": "get_task_status",
            }
        if tool_id == "get_task_status":
            task_id = payload.get("task_id")
            task = self._get_task(record, task_id)
            if task is None:
                return {
                    "tool_id": tool_id,
                    "status": ExecutionStatus.BLOCKED.value,
                    "summary": "No delegated task is available to inspect.",
                    "details": {"task_id": task_id},
                    "resolved_subgoal": False,
                }
            resolved_subgoal = (
                task.get("tool_id") == "go_to_pose"
                and task.get("state") == ExecutionStatus.SUCCEEDED.value
                and subgoal.get("kind") == "navigate"
            )
            recommended_action_id = None
            if task.get("tool_id") in {"explore", "create_map"}:
                if task.get("state") == ExecutionStatus.SUCCEEDED.value:
                    recommended_action_id = "get_map"
                elif task.get("state") == ExecutionStatus.IN_PROGRESS.value:
                    recommended_action_id = "get_task_status"
            return {
                "tool_id": tool_id,
                "status": ExecutionStatus.SUCCEEDED.value,
                "summary": f"Task `{task['task_id']}` is `{task['state']}`.",
                "details": {"task": task},
                "resolved_subgoal": resolved_subgoal,
                "recommended_action_id": recommended_action_id,
            }
        if tool_id == "cancel_task":
            task_id = payload.get("task_id")
            task = self._cancel_task(record, task_id)
            if task is None:
                return {
                    "tool_id": tool_id,
                    "status": ExecutionStatus.BLOCKED.value,
                    "summary": "No delegated task is available to cancel.",
                    "details": {"task_id": task_id},
                    "resolved_subgoal": False,
                }
            return {
                "tool_id": tool_id,
                "status": ExecutionStatus.SUCCEEDED.value,
                "summary": f"Processed cancel request for task `{task['task_id']}`.",
                "details": {
                    "task": task,
                    "canceled": task["state"] == ExecutionStatus.ABORTED.value,
                },
                "resolved_subgoal": False,
            }
        return {
            "tool_id": tool_id,
            "status": ExecutionStatus.BLOCKED.value,
            "summary": f"Tool `{tool_id}` is not implemented by the offload server.",
            "details": {},
            "resolved_subgoal": False,
        }

    def _stub_skill(self, skill: dict[str, Any], world_state: dict[str, Any]) -> dict[str, Any]:
        skill_type = str(skill.get("skill_type", ""))
        if skill_type == SkillType.NAVIGATION.value:
            readiness = ReadinessState.NAVIGATION_READY_POSE.value
        elif skill_type == SkillType.ALIGNMENT.value:
            readiness = ReadinessState.SKILL_READY_POSE.value
        else:
            readiness = ReadinessState.SKILL_READY_POSE.value
        return {
            "skill_id": skill["skill_id"],
            "status": ExecutionStatus.SUCCEEDED.value,
            "error_code": None,
            "postcondition_evidence": (
                f"Executed `{skill['skill_id']}` on the Ubuntu offload node "
                f"using the simplified remote execution path."
            ),
            "retry_budget_impact": 0,
            "updated_localization_confidence": world_state.get("localization_confidence"),
            "replan_hint": None,
            "readiness_state": readiness,
        }

    def _proxy_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc

    def _create_task(
        self,
        record: BrainRecord,
        *,
        tool_id: str,
        message: str,
        result: dict[str, Any] | None = None,
        state: str = ExecutionStatus.SUCCEEDED.value,
        progress: float = 1.0,
    ) -> dict[str, Any]:
        task_id = f"{tool_id}_{uuid.uuid4().hex[:8]}"
        task = {
            "task_id": task_id,
            "tool_id": tool_id,
            "state": state,
            "progress": progress,
            "message": message,
            "result": dict(result or {}),
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        record.tasks[task_id] = task
        self.registry.persist()
        return dict(task)

    def _get_task(self, record: BrainRecord, task_id: str | None) -> dict[str, Any] | None:
        exploration = self._exploration_backend(record)
        mapping_task = exploration.get_task(task_id)
        if mapping_task is not None:
            return mapping_task
        if task_id:
            task = record.tasks.get(task_id)
            return None if task is None else dict(task)
        if not record.tasks:
            return None
        latest_id = next(reversed(record.tasks))
        return dict(record.tasks[latest_id])

    def _cancel_task(self, record: BrainRecord, task_id: str | None) -> dict[str, Any] | None:
        exploration = self._exploration_backend(record)
        mapping_task = exploration.cancel_task(task_id)
        if mapping_task is not None:
            return mapping_task
        resolved_id = task_id or (next(reversed(record.tasks)) if record.tasks else None)
        if resolved_id is None:
            return None
        task = record.tasks.get(resolved_id)
        if task is None:
            return None
        if task["state"] not in {
            ExecutionStatus.SUCCEEDED.value,
            ExecutionStatus.FAILED.value,
            ExecutionStatus.ABORTED.value,
        }:
            task["state"] = ExecutionStatus.ABORTED.value
            task["message"] = f"Canceled task `{resolved_id}`."
            task["updated_at"] = time.time()
            self.registry.persist()
        return dict(task)

    def _map_info(self, record: BrainRecord, world_state: dict[str, Any]) -> dict[str, Any] | None:
        exploration = self._exploration_backend(record)
        active_map = exploration.get_map()
        if active_map is not None:
            return dict(active_map)
        if record.synthetic_map is not None:
            return dict(record.synthetic_map)
        metadata = dict(world_state.get("metadata", {}))
        candidates = (
            metadata.get("map"),
            metadata.get("navigation_map"),
            metadata.get("nav_map"),
        )
        for candidate in candidates:
            if isinstance(candidate, dict) and candidate:
                return dict(candidate)
        maps = metadata.get("maps")
        if isinstance(maps, list):
            for candidate in maps:
                if isinstance(candidate, dict) and candidate:
                    return dict(candidate)
        return None

    def _resolve_pose(self, pose: Any, map_info: dict[str, Any] | None) -> tuple[Any, str]:
        if not isinstance(map_info, dict):
            return pose, self._pose_label(pose)
        if isinstance(pose, dict):
            return pose, self._pose_label(pose)
        target_name = str(pose)
        for place in map_info.get("named_places", []):
            if isinstance(place, dict) and place.get("name") == target_name:
                return dict(place.get("pose", {})), target_name
        for region in map_info.get("regions", []):
            if isinstance(region, dict) and region.get("label") == target_name:
                waypoints = region.get("default_waypoints", [])
                if waypoints:
                    return dict(waypoints[0]), target_name
        return pose, target_name

    def _set_synthetic_map(
        self,
        record: BrainRecord,
        *,
        session: str,
        area: str,
        world_state: dict[str, Any],
        source: str,
    ) -> dict[str, Any]:
        place_memories = world_state.get("place_memories", [])
        named_places = [item.get("name") for item in place_memories if isinstance(item, dict) and item.get("name")]
        current_pose = world_state.get("current_pose")
        if current_pose:
            named_places.append(str(current_pose))
        if area:
            named_places.append(area)
        record.synthetic_map = {
            "map_id": session,
            "frame": "map",
            "resolution": 0.05,
            "width": 1024,
            "height": 1024,
            "named_places": sorted(set(named_places)),
            "summary": f"Synthetic {source} map covering `{area}`.",
        }
        self.registry.persist()
        return dict(record.synthetic_map)

    def _pose_label(self, pose: Any) -> str:
        if isinstance(pose, dict):
            if "name" in pose:
                return str(pose["name"])
            if {"x", "y"} <= set(pose):
                return f"({pose['x']}, {pose['y']})"
        return str(pose)

    def _update_scene_cache(self, record: BrainRecord, payload: Mapping[str, Any]) -> None:
        scene = extract_scene_from_tool_result(payload)
        if scene is None:
            return
        record.latest_scene = scene
        self.registry.persist()


class OffloadClient:
    def __init__(
        self,
        server_url: str,
        *,
        brain_name: str = "xlerobot-brain",
        brain_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        timeout_s: float = 15.0,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.brain_name = brain_name
        self.brain_id = brain_id
        self.metadata = dict(metadata or {})
        self.timeout_s = timeout_s
        self._registration_lock = threading.RLock()

    def register(self) -> BrainRegistration:
        with self._registration_lock:
            payload = {
                "brain_id": self.brain_id,
                "brain_name": self.brain_name,
                "metadata": self.metadata,
            }
            response = self._request_json("/api/brains/register", payload)
            self.brain_id = str(response["brain_id"])
            return BrainRegistration(
                brain_id=self.brain_id,
                brain_name=str(response["brain_name"]),
                server_url=self.server_url,
                metadata=dict(response.get("metadata", {})),
            )

    def ensure_registered(self) -> BrainRegistration:
        if self.brain_id is None:
            return self.register()
        return BrainRegistration(
            brain_id=self.brain_id,
            brain_name=self.brain_name,
            server_url=self.server_url,
            metadata=dict(self.metadata),
        )

    def heartbeat(self) -> dict[str, Any]:
        registration = self.ensure_registered()
        return self._request_json(f"/api/brains/{registration.brain_id}/heartbeat", {})

    def publish_state(
        self,
        world_state: WorldState,
        *,
        reason: str = "state_update",
        sensors: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.publish_state_payload(
            serialize_world_state(world_state),
            reason=reason,
            sensors=sensors,
        )

    def publish_state_payload(
        self,
        world_state: dict[str, Any],
        *,
        reason: str = "state_update",
        sensors: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        registration = self.ensure_registered()
        payload = {
            "world_state": world_state,
            "sensors": dict(sensors or world_state.get("metadata", {}).get("sensors", {})),
            "reason": reason,
        }
        return self._request_json(f"/api/brains/{registration.brain_id}/state", payload)

    def execute_tool(self, tool_id: str, context: Any) -> dict[str, Any]:
        registration = self.ensure_registered()
        world_state = getattr(context, "world_state", None)
        if isinstance(world_state, WorldState):
            self.publish_state(world_state, reason=f"tool:{tool_id}")
        payload = {
            "brain_id": registration.brain_id,
            "tool_id": tool_id,
            "context": {
                "goal": serialize_goal_context(context.goal),
                "subgoal": serialize_subgoal(context.subgoal),
                "world_state": serialize_world_state(context.world_state),
                "attempts": context.attempts,
                "payload": dict(context.payload),
                "candidates": list(context.candidates),
                "question": context.question,
                "recent_events": list(context.recent_events),
            },
        }
        return self._request_json("/api/offload/execute", payload)

    def execute_skill(self, skill: SkillContract, goal: GoalContext, world_state: WorldState) -> ExecutionResult:
        registration = self.ensure_registered()
        payload = {
            "brain_id": registration.brain_id,
            "skill": serialize_skill_contract(skill),
            "goal": serialize_goal_context(goal),
            "world_state": serialize_world_state(world_state),
        }
        response = self._request_json("/api/skills/execute", payload)
        return execution_result_from_payload(response, skill.skill_id)

    def brain_snapshot(self) -> dict[str, Any]:
        registration = self.ensure_registered()
        return self._request_json(f"/api/brains/{registration.brain_id}", payload=None, method="GET")

    def mapping_snapshot(self) -> dict[str, Any]:
        registration = self.ensure_registered()
        return self._request_json(f"/api/brains/{registration.brain_id}/mapping", payload=None, method="GET")

    def start_explore(self, *, area: str, session: str | None = None, source: str = "operator") -> dict[str, Any]:
        registration = self.ensure_registered()
        return self._request_json(
            f"/api/brains/{registration.brain_id}/mapping/explore",
            {"area": area, "session": session, "source": source},
        )

    def start_create_map(self, *, area: str, session: str, source: str = "operator") -> dict[str, Any]:
        registration = self.ensure_registered()
        return self._request_json(
            f"/api/brains/{registration.brain_id}/mapping/create_map",
            {"area": area, "session": session, "source": source},
        )

    def pause_mapping_task(self, task_id: str) -> dict[str, Any]:
        registration = self.ensure_registered()
        return self._request_json(f"/api/brains/{registration.brain_id}/mapping/tasks/{task_id}/pause", {})

    def resume_mapping_task(self, task_id: str) -> dict[str, Any]:
        registration = self.ensure_registered()
        return self._request_json(f"/api/brains/{registration.brain_id}/mapping/tasks/{task_id}/resume", {})

    def cancel_mapping_task(self, task_id: str | None = None) -> dict[str, Any]:
        registration = self.ensure_registered()
        path = f"/api/brains/{registration.brain_id}/mapping/tasks/{task_id}/cancel" if task_id else f"/api/brains/{registration.brain_id}/mapping/cancel"
        return self._request_json(path, {})

    def update_mapping_region(
        self,
        region_id: str,
        *,
        label: str | None = None,
        purpose: str | None = None,
        polygon_2d: list[list[float]] | None = None,
        default_waypoints: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        registration = self.ensure_registered()
        payload = {
            "label": label,
            "purpose": purpose,
            "polygon_2d": polygon_2d,
            "default_waypoints": default_waypoints,
        }
        return self._request_json(f"/api/brains/{registration.brain_id}/mapping/regions/{region_id}/update", payload)

    def merge_mapping_regions(self, region_ids: list[str], *, new_label: str | None = None) -> dict[str, Any]:
        registration = self.ensure_registered()
        return self._request_json(
            f"/api/brains/{registration.brain_id}/mapping/regions/merge",
            {"region_ids": region_ids, "new_label": new_label},
        )

    def split_mapping_region(
        self,
        region_id: str,
        *,
        polygons: list[list[list[float]]] | None = None,
    ) -> dict[str, Any]:
        registration = self.ensure_registered()
        return self._request_json(
            f"/api/brains/{registration.brain_id}/mapping/regions/{region_id}/split",
            {"polygons": polygons},
        )

    def set_named_place(self, *, name: str, pose: dict[str, Any], region_id: str | None = None) -> dict[str, Any]:
        registration = self.ensure_registered()
        return self._request_json(
            f"/api/brains/{registration.brain_id}/mapping/named_places",
            {"name": name, "pose": pose, "region_id": region_id},
        )

    def approve_mapping_map(self) -> dict[str, Any]:
        registration = self.ensure_registered()
        return self._request_json(f"/api/brains/{registration.brain_id}/mapping/approve", {})

    def _request_json(self, path: str, payload: dict[str, Any] | None = None, method: str = "POST") -> dict[str, Any]:
        url = f"{self.server_url}{path}"
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        return json.loads(body or "{}")


class OffloadServer:
    def __init__(self, config: OffloadServerConfig | None = None) -> None:
        self.config = config or OffloadServerConfig()
        self.registry = _BrainRegistry(self.config)
        self.router = _OffloadRouter(self.config, self.registry)
        self._server = ThreadingHTTPServer((self.config.host, self.config.port), self._build_handler())
        self.host, self.port = self._server.server_address

    def _build_handler(self):
        registry = self.registry
        router = self.router
        config = self.config

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                path = self.path.rstrip("/")
                if path == "/api/health":
                    self._send_json(
                        {
                            "status": "ok",
                            "nav2_service_url": config.nav2_service_url,
                            "perception_service_url": config.perception_service_url,
                            "vla_service_url": config.vla_service_url,
                            "brain_count": len(registry.list()),
                        }
                    )
                    return
                if path == "/api/brains":
                    self._send_json({"brains": registry.list()})
                    return
                if path.startswith("/api/brains/"):
                    parts = path.split("/")
                    brain_id = parts[3]
                    if len(parts) >= 5 and parts[4] == "mapping":
                        self._send_json(router.mapping_snapshot(brain_id))
                        return
                    self._send_json(registry.get(brain_id).summary())
                    return
                self.send_error(HTTPStatus.NOT_FOUND, "not found")

            def do_POST(self) -> None:
                path = self.path.rstrip("/")
                payload = self._read_json_body()
                if path == "/api/brains/register":
                    record = registry.register(
                        brain_name=str(payload.get("brain_name", "xlerobot-brain")),
                        metadata=dict(payload.get("metadata", {})),
                        brain_id=payload.get("brain_id"),
                    )
                    self._send_json(record.summary())
                    return
                if path.startswith("/api/brains/") and path.endswith("/heartbeat"):
                    brain_id = path.split("/")[3]
                    self._send_json(registry.heartbeat(brain_id).summary())
                    return
                if path.startswith("/api/brains/") and path.endswith("/state"):
                    brain_id = path.split("/")[3]
                    record = registry.update_state(
                        brain_id,
                        world_state=dict(payload.get("world_state", {})),
                        sensors=dict(payload.get("sensors", {})),
                        reason=str(payload.get("reason", "state_update")),
                    )
                    self._send_json(record.summary())
                    return
                if path.startswith("/api/brains/") and "/mapping/" in path:
                    parts = path.split("/")
                    brain_id = parts[3]
                    if len(parts) >= 6 and parts[4] == "mapping" and parts[5] in {"explore", "create_map"}:
                        task = router.start_mapping_task(
                            brain_id,
                            tool_id=parts[5],
                            area=str(payload.get("area", "workspace")),
                            session=payload.get("session"),
                            source=str(payload.get("source", "operator")),
                        )
                        self._send_json(task)
                        return
                    if len(parts) >= 8 and parts[4] == "mapping" and parts[5] == "tasks" and parts[7] in {"pause", "resume", "cancel"}:
                        task_id = parts[6]
                        action = parts[7]
                        if action == "pause":
                            response = router.pause_mapping_task(brain_id, task_id)
                        elif action == "resume":
                            response = router.resume_mapping_task(brain_id, task_id)
                        else:
                            response = router.cancel_mapping_task(brain_id, task_id)
                        if response is None:
                            self.send_error(HTTPStatus.NOT_FOUND, "task not found")
                            return
                        self._send_json(response)
                        return
                    if len(parts) >= 6 and parts[4] == "mapping" and parts[5] == "cancel":
                        response = router.cancel_mapping_task(brain_id, None)
                        if response is None:
                            self.send_error(HTTPStatus.NOT_FOUND, "task not found")
                            return
                        self._send_json(response)
                        return
                    if len(parts) >= 8 and parts[4] == "mapping" and parts[5] == "regions" and parts[7] == "update":
                        response = router.update_region(
                            brain_id,
                            parts[6],
                            label=payload.get("label"),
                            purpose=payload.get("purpose"),
                            polygon_2d=payload.get("polygon_2d"),
                            default_waypoints=payload.get("default_waypoints"),
                        )
                        if response is None:
                            self.send_error(HTTPStatus.NOT_FOUND, "region not found")
                            return
                        self._send_json(response)
                        return
                    if len(parts) >= 7 and parts[4] == "mapping" and parts[5] == "regions" and parts[6] == "merge":
                        response = router.merge_regions(
                            brain_id,
                            list(payload.get("region_ids", [])),
                            new_label=payload.get("new_label"),
                        )
                        if response is None:
                            self.send_error(HTTPStatus.BAD_REQUEST, "unable to merge regions")
                            return
                        self._send_json(response)
                        return
                    if len(parts) >= 8 and parts[4] == "mapping" and parts[5] == "regions" and parts[7] == "split":
                        response = router.split_region(
                            brain_id,
                            parts[6],
                            polygons=payload.get("polygons"),
                        )
                        self._send_json({"regions": response})
                        return
                    if len(parts) >= 6 and parts[4] == "mapping" and parts[5] == "named_places":
                        response = router.set_named_place(
                            brain_id,
                            str(payload.get("name")),
                            dict(payload.get("pose", {})),
                            region_id=payload.get("region_id"),
                        )
                        if response is None:
                            self.send_error(HTTPStatus.BAD_REQUEST, "unable to set named place")
                            return
                        self._send_json(response)
                        return
                    if len(parts) >= 6 and parts[4] == "mapping" and parts[5] == "approve":
                        response = router.approve_map(brain_id)
                        if response is None:
                            self.send_error(HTTPStatus.NOT_FOUND, "map not found")
                            return
                        self._send_json(response)
                        return
                if path == "/api/offload/execute":
                    result = router.execute_tool(
                        str(payload["brain_id"]),
                        str(payload["tool_id"]),
                        dict(payload.get("context", {})),
                    )
                    self._send_json(result)
                    return
                if path == "/api/skills/execute":
                    result = router.execute_skill(
                        str(payload["brain_id"]),
                        dict(payload["skill"]),
                        dict(payload["goal"]),
                        dict(payload["world_state"]),
                    )
                    self._send_json(result)
                    return
                self.send_error(HTTPStatus.NOT_FOUND, "not found")

            def log_message(self, format: str, *args: Any) -> None:
                return

            def _read_json_body(self) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length", "0"))
                if length == 0:
                    return {}
                body = self.rfile.read(length).decode("utf-8")
                return json.loads(body)

            def _send_json(self, payload: dict[str, Any]) -> None:
                encoded = json.dumps(payload).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        return Handler

    def serve_forever(self) -> None:
        self._server.serve_forever()

    def start_in_thread(self) -> threading.Thread:
        thread = threading.Thread(target=self.serve_forever, daemon=True)
        thread.start()
        return thread

    def shutdown(self) -> None:
        self._server.shutdown()
        self._server.server_close()
