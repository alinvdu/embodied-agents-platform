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

import ast
import contextlib
from dataclasses import dataclass, field, replace
import io
import json
import math
import time
from typing import Any, Protocol
import uuid

from .exploration import ExplorationBackend, ExplorationBackendConfig
from .models import ExecutionStatus, GoalContext, PlaceMemory, Subgoal, WorldState
from .offload import serialize_world_state
from .perception_service import execute_perception_tool, extract_scene_from_tool_result


@dataclass(frozen=True)
class ToolCallContext:
    goal: GoalContext
    subgoal: Subgoal
    world_state: WorldState
    attempts: int = 0
    payload: dict[str, Any] = field(default_factory=dict)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    question: str = ""
    recent_events: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ToolSpec:
    tool_id: str
    description: str
    category: str = "analysis"


@dataclass(frozen=True)
class ToolResult:
    tool_id: str
    status: ExecutionStatus
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    resolved_subgoal: bool = False
    recommended_action_id: str | None = None
    recommended_action_type: str | None = None
    updated_world_state: WorldState | None = None


class ToolExecutor(Protocol):
    def execute(self, context: ToolCallContext) -> ToolResult:
        ...


@dataclass
class StaticToolExecutor:
    handler: Any

    def execute(self, context: ToolCallContext) -> ToolResult:
        return self.handler(context)


@dataclass
class _LocalToolRuntime:
    tasks: dict[str, dict[str, Any]] = field(default_factory=dict)
    latest_scene: dict[str, Any] | None = None
    exploration_backend: ExplorationBackend = field(
        default_factory=lambda: ExplorationBackend(ExplorationBackendConfig(mode="sim"))
    )

    def create_task(
        self,
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
        self.tasks[task_id] = task
        return dict(task)

    def latest_task(self) -> dict[str, Any] | None:
        if not self.tasks:
            return None
        latest_id = next(reversed(self.tasks))
        return dict(self.tasks[latest_id])

    def get_task(self, task_id: str | None) -> dict[str, Any] | None:
        mapping_task = self.exploration_backend.get_task(task_id)
        if mapping_task is not None:
            return mapping_task
        if task_id:
            task = self.tasks.get(task_id)
            return None if task is None else dict(task)
        return self.latest_task()

    def cancel_task(self, task_id: str | None) -> dict[str, Any] | None:
        mapping_task = self.exploration_backend.cancel_task(task_id)
        if mapping_task is not None:
            return mapping_task
        resolved_id = task_id or (next(reversed(self.tasks)) if self.tasks else None)
        if resolved_id is None:
            return None
        task = self.tasks.get(resolved_id)
        if task is None:
            return None
        if task["state"] not in {
            ExecutionStatus.SUCCEEDED.value,
            ExecutionStatus.FAILED.value,
            ExecutionStatus.ABORTED.value,
        }:
            task["state"] = ExecutionStatus.ABORTED.value
            task["progress"] = min(float(task.get("progress", 0.0)), 1.0)
            task["message"] = f"Canceled task `{resolved_id}`."
            task["updated_at"] = time.time()
        return dict(task)

    def get_map(self, world_state: WorldState) -> dict[str, Any] | None:
        return self.exploration_backend.get_map() or _extract_map_info(world_state)

    def resolve_pose(self, pose: Any, world_state: WorldState) -> tuple[Any, str]:
        map_info = self.get_map(world_state)
        if not isinstance(map_info, dict):
            return pose, _pose_label(pose)
        if isinstance(pose, dict):
            return pose, _pose_label(pose)
        target_name = str(pose)
        for place in map_info.get("named_places", []):
            if isinstance(place, dict) and place.get("name") == target_name:
                return dict(place.get("pose", {})), target_name
        for region in map_info.get("regions", []):
            if not isinstance(region, dict):
                continue
            if region.get("label") == target_name:
                waypoints = region.get("default_waypoints", [])
                if waypoints:
                    return dict(waypoints[0]), target_name
        return pose, target_name


@dataclass
class ToolRegistry:
    _tools: dict[str, ToolSpec] = field(default_factory=dict)
    _executors: dict[str, ToolExecutor] = field(default_factory=dict)

    def register(self, spec: ToolSpec, executor: ToolExecutor) -> None:
        self._tools[spec.tool_id] = spec
        self._executors[spec.tool_id] = executor

    def list_tools(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def get(self, tool_id: str) -> ToolSpec:
        return self._tools[tool_id]

    def execute(self, tool_id: str, context: ToolCallContext) -> ToolResult:
        executor = self._executors.get(tool_id)
        if executor is None:
            return ToolResult(
                tool_id=tool_id,
                status=ExecutionStatus.BLOCKED,
                summary=f"No executor is registered for tool `{tool_id}`.",
                details={},
            )
        return executor.execute(context)


class BoundedCodeExecutor:
    _banned_names = {
        "__import__",
        "breakpoint",
        "compile",
        "eval",
        "exec",
        "getattr",
        "globals",
        "help",
        "input",
        "locals",
        "open",
        "setattr",
        "vars",
    }
    _banned_nodes = (
        ast.AsyncFor,
        ast.AsyncFunctionDef,
        ast.AsyncWith,
        ast.Await,
        ast.ClassDef,
        ast.Delete,
        ast.Global,
        ast.Import,
        ast.ImportFrom,
        ast.Lambda,
        ast.Nonlocal,
        ast.Raise,
        ast.Try,
        ast.With,
    )

    def execute(
        self,
        code: str,
        *,
        world_state: WorldState,
        candidates: list[dict[str, Any]],
        question: str,
    ) -> dict[str, Any]:
        try:
            tree = ast.parse(code, mode="exec")
            self._validate_tree(tree)
        except Exception as exc:
            return {
                "ok": False,
                "stdout": "",
                "stderr": str(exc),
                "result": None,
            }

        safe_builtins = {
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "float": float,
            "int": int,
            "len": len,
            "list": list,
            "max": max,
            "min": min,
            "print": print,
            "range": range,
            "round": round,
            "set": set,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "zip": zip,
        }
        env: dict[str, Any] = {
            "__builtins__": safe_builtins,
            "WORLD_STATE": _serialize_world_state(world_state),
            "CANDIDATES": candidates,
            "QUESTION": question,
            "RESULT": None,
            "json": json,
            "math": math,
        }
        stdout = io.StringIO()
        stderr = io.StringIO()
        try:
            with (
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                exec(compile(tree, "<bounded-code>", "exec"), env, env)
        except Exception as exc:
            stderr.write(str(exc))
            return {
                "ok": False,
                "stdout": stdout.getvalue(),
                "stderr": stderr.getvalue(),
                "result": env.get("RESULT"),
            }
        return {
            "ok": True,
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
            "result": env.get("RESULT"),
        }

    def _validate_tree(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if isinstance(node, self._banned_nodes):
                raise ValueError(f"unsupported syntax in bounded code executor: {type(node).__name__}")
            if isinstance(node, ast.Name) and node.id in self._banned_names:
                raise ValueError(f"access to `{node.id}` is not allowed in bounded code execution")
            if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                raise ValueError("dunder attribute access is not allowed in bounded code execution")


def build_default_tool_registry(
    offload_client: Any | None = None,
    *,
    exploration_mode: str = "sim",
    exploration_backend: ExplorationBackend | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    code_executor = BoundedCodeExecutor()
    local_runtime = _LocalToolRuntime(
        exploration_backend=exploration_backend or ExplorationBackend(ExplorationBackendConfig(mode=exploration_mode))
    )

    registry.register(
        ToolSpec(
            tool_id="describe_world_state",
            description="Summarize the currently known world state, observations, and resources.",
            category="analysis",
        ),
        StaticToolExecutor(_describe_world_state),
    )
    registry.register(
        ToolSpec(
            tool_id="perceive_scene",
            description="Refresh scene understanding from RGB-D, segmentation, and 3D annotation cues.",
            category="perception",
        ),
        StaticToolExecutor(
            _make_remote_tool_handler(
                "perceive_scene",
                offload_client,
                lambda context: _perceive_scene(context, local_runtime),
            )
        ),
    )
    registry.register(
        ToolSpec(
            tool_id="ground_object_3d",
            description="Find and segment a queried object and estimate its 3D anchor.",
            category="perception",
        ),
        StaticToolExecutor(
            _make_remote_tool_handler(
                "ground_object_3d",
                offload_client,
                lambda context: _ground_object_3d(context, local_runtime),
            )
        ),
    )
    registry.register(
        ToolSpec(
            tool_id="set_waypoint_from_object",
            description="Convert a grounded object anchor into a navigation waypoint or approach pose.",
            category="perception",
        ),
        StaticToolExecutor(
            _make_remote_tool_handler(
                "set_waypoint_from_object",
                offload_client,
                lambda context: _set_waypoint_from_object(context, local_runtime),
            )
        ),
    )
    registry.register(
        ToolSpec(
            tool_id="go_to_pose",
            description="Send a high-level navigation request toward a named or explicit pose.",
            category="navigation",
        ),
        StaticToolExecutor(
            _make_remote_tool_handler(
                "go_to_pose",
                offload_client,
                lambda context: _go_to_pose(context, local_runtime),
            )
        ),
    )
    registry.register(
        ToolSpec(
            tool_id="get_map",
            description="Inspect whether a usable navigation map is available and summarize it.",
            category="mapping",
        ),
        StaticToolExecutor(
            _make_remote_tool_handler(
                "get_map",
                offload_client,
                lambda context: _get_map(context, local_runtime),
            )
        ),
    )
    registry.register(
        ToolSpec(
            tool_id="explore",
            description="Run a high-level exploration pass, optionally building map coverage as it goes.",
            category="mapping",
        ),
        StaticToolExecutor(
            _make_remote_tool_handler(
                "explore",
                offload_client,
                lambda context: _explore(context, local_runtime),
            )
        ),
    )
    registry.register(
        ToolSpec(
            tool_id="create_map",
            description="Start a dedicated mapping session and persist a map handle for later navigation.",
            category="mapping",
        ),
        StaticToolExecutor(
            _make_remote_tool_handler(
                "create_map",
                offload_client,
                lambda context: _create_map(context, local_runtime),
            )
        ),
    )
    registry.register(
        ToolSpec(
            tool_id="get_task_status",
            description="Inspect the state of the latest or specified delegated navigation/mapping task.",
            category="control",
        ),
        StaticToolExecutor(
            _make_remote_tool_handler(
                "get_task_status",
                offload_client,
                lambda context: _get_task_status(context, local_runtime),
            )
        ),
    )
    registry.register(
        ToolSpec(
            tool_id="cancel_task",
            description="Cancel the latest or specified delegated navigation/mapping task.",
            category="control",
        ),
        StaticToolExecutor(
            _make_remote_tool_handler(
                "cancel_task",
                offload_client,
                lambda context: _cancel_task(context, local_runtime),
            )
        ),
    )
    registry.register(
        ToolSpec(
            tool_id="code_execution",
            description="Run bounded helper Python code to reason about the current state and suggest the next move.",
            category="reasoning",
        ),
        StaticToolExecutor(lambda context: _code_execution(context, code_executor)),
    )
    return registry


def _describe_world_state(context: ToolCallContext) -> ToolResult:
    summary = (
        f"Pose `{context.world_state.current_pose}` with visible objects "
        f"{sorted(context.world_state.visible_objects)} and observations "
        f"{sorted(context.world_state.available_observations)}."
    )
    return ToolResult(
        tool_id="describe_world_state",
        status=ExecutionStatus.SUCCEEDED,
        summary=summary,
        details={"world_state": _serialize_world_state(context.world_state)},
    )


def _perceive_scene(context: ToolCallContext, runtime: _LocalToolRuntime) -> ToolResult:
    return _execute_local_perception_tool("perceive_scene", context, runtime)


def _ground_object_3d(context: ToolCallContext, runtime: _LocalToolRuntime) -> ToolResult:
    return _execute_local_perception_tool("ground_object_3d", context, runtime)


def _set_waypoint_from_object(context: ToolCallContext, runtime: _LocalToolRuntime) -> ToolResult:
    return _execute_local_perception_tool("set_waypoint_from_object", context, runtime)


def _go_to_pose(context: ToolCallContext, runtime: _LocalToolRuntime) -> ToolResult:
    requested_pose = context.payload.get("pose") or context.payload.get("target") or context.subgoal.target or context.subgoal.text
    pose, pose_label = runtime.resolve_pose(requested_pose, context.world_state)
    constraints = dict(context.payload.get("constraints", {}))
    task = runtime.create_task(
        tool_id="go_to_pose",
        message=f"Reached `{pose_label}` in the simplified local navigation runtime.",
        result={
            "goal_pose": pose,
            "goal_label": pose_label,
            "constraints": constraints,
        },
    )
    return ToolResult(
        tool_id="go_to_pose",
        status=ExecutionStatus.SUCCEEDED,
        summary=f"Executed a high-level navigation request toward `{pose_label}`.",
        details={"task": task, "pose": pose, "pose_label": pose_label, "constraints": constraints},
        resolved_subgoal=context.subgoal.kind == "navigate",
    )


def _get_map(context: ToolCallContext, runtime: _LocalToolRuntime) -> ToolResult:
    map_info = runtime.get_map(context.world_state)
    if map_info is None:
        return ToolResult(
            tool_id="get_map",
            status=ExecutionStatus.SUCCEEDED,
            summary="No navigation map is currently available in the local runtime.",
            details={"available": False, "map": None},
        )
    recommended_action_id = None
    target = context.subgoal.target or context.payload.get("target")
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
    if target and (target in named_places or target in region_labels):
        recommended_action_id = "go_to_pose"
    return ToolResult(
        tool_id="get_map",
        status=ExecutionStatus.SUCCEEDED,
        summary=f"Navigation map `{map_info.get('map_id', 'unknown_map')}` is available.",
        details={"available": True, "map": map_info},
        recommended_action_id=recommended_action_id,
    )


def _explore(context: ToolCallContext, runtime: _LocalToolRuntime) -> ToolResult:
    area = str(context.payload.get("area") or context.subgoal.target or context.world_state.current_pose)
    strategy = str(context.payload.get("strategy", "frontier"))
    task = runtime.exploration_backend.start_explore(
        area=area,
        session=f"explore_{uuid.uuid4().hex[:8]}",
        source="planner",
        build_map=bool(context.payload.get("build_map", True)),
        world_state=serialize_world_state(context.world_state),
    )
    return ToolResult(
        tool_id="explore",
        status=ExecutionStatus.IN_PROGRESS,
        summary=f"Started exploration for `{area}` using strategy `{strategy}`.",
        details={"task": task, "area": area, "strategy": strategy},
        recommended_action_id="get_task_status",
    )


def _create_map(context: ToolCallContext, runtime: _LocalToolRuntime) -> ToolResult:
    session = str(context.payload.get("session") or f"map_{uuid.uuid4().hex[:8]}")
    area = context.payload.get("area") or context.subgoal.target or context.world_state.current_pose
    task = runtime.exploration_backend.start_create_map(
        session=session,
        area=str(area) if area is not None else "workspace",
        source="planner",
        world_state=serialize_world_state(context.world_state),
    )
    return ToolResult(
        tool_id="create_map",
        status=ExecutionStatus.IN_PROGRESS,
        summary=f"Started a navigation map build for `{area}`.",
        details={"task": task},
        recommended_action_id="get_task_status",
    )


def _get_task_status(context: ToolCallContext, runtime: _LocalToolRuntime) -> ToolResult:
    task_id = context.payload.get("task_id")
    task = runtime.get_task(task_id)
    if task is None:
        return ToolResult(
            tool_id="get_task_status",
            status=ExecutionStatus.BLOCKED,
            summary="No delegated task is available to inspect.",
            details={"task_id": task_id},
        )
    resolved_subgoal = False
    recommended_action_id = None
    if task.get("tool_id") == "go_to_pose":
        resolved_subgoal = (
            task.get("state") == ExecutionStatus.SUCCEEDED.value
            and context.subgoal.kind == "navigate"
        )
    elif task.get("tool_id") in {"explore", "create_map"}:
        if task.get("state") == ExecutionStatus.SUCCEEDED.value:
            recommended_action_id = "get_map"
        elif task.get("state") == ExecutionStatus.IN_PROGRESS.value:
            recommended_action_id = "get_task_status"
    return ToolResult(
        tool_id="get_task_status",
        status=ExecutionStatus.SUCCEEDED,
        summary=f"Task `{task['task_id']}` is `{task['state']}`.",
        details={"task": task},
        resolved_subgoal=resolved_subgoal,
        recommended_action_id=recommended_action_id,
    )


def _cancel_task(context: ToolCallContext, runtime: _LocalToolRuntime) -> ToolResult:
    task_id = context.payload.get("task_id")
    task = runtime.cancel_task(task_id)
    if task is None:
        return ToolResult(
            tool_id="cancel_task",
            status=ExecutionStatus.BLOCKED,
            summary="No delegated task is available to cancel.",
            details={"task_id": task_id},
        )
    return ToolResult(
        tool_id="cancel_task",
        status=ExecutionStatus.SUCCEEDED,
        summary=f"Processed cancel request for task `{task['task_id']}`.",
        details={"task": task, "canceled": task["state"] == ExecutionStatus.ABORTED.value},
    )


def _code_execution(context: ToolCallContext, executor: BoundedCodeExecutor) -> ToolResult:
    code = str(context.payload.get("code", "")).strip()
    if not code:
        return ToolResult(
            tool_id="code_execution",
            status=ExecutionStatus.BLOCKED,
            summary="No helper code was provided for execution.",
            details={},
        )
    execution = executor.execute(
        code,
        world_state=context.world_state,
        candidates=context.candidates,
        question=context.question,
    )
    status = ExecutionStatus.SUCCEEDED if execution["ok"] else ExecutionStatus.FAILED
    result = execution.get("result") if isinstance(execution.get("result"), dict) else {}
    return ToolResult(
        tool_id="code_execution",
        status=status,
        summary="Executed bounded helper code for agent-side reasoning.",
        details={
            "code": code,
            "stdout": execution.get("stdout", ""),
            "stderr": execution.get("stderr", ""),
            "result": execution.get("result"),
        },
        recommended_action_id=result.get("recommended_action_id") if isinstance(result, dict) else None,
        recommended_action_type=result.get("recommended_action_type") if isinstance(result, dict) else None,
    )


def _make_remote_tool_handler(
    tool_id: str,
    offload_client: Any | None,
    fallback_handler: Any,
):
    def handler(context: ToolCallContext) -> ToolResult:
        if offload_client is None:
            return _enrich_tool_result(fallback_handler(context), context)
        try:
            payload = offload_client.execute_tool(tool_id, context)
            return _enrich_tool_result(_tool_result_from_payload(tool_id, payload), context)
        except Exception as exc:
            fallback = fallback_handler(context)
            return _enrich_tool_result(
                ToolResult(
                tool_id=tool_id,
                status=fallback.status,
                summary=f"{fallback.summary} Remote offload failed: {exc}",
                details={"remote_error": str(exc), "fallback": fallback.details},
                resolved_subgoal=fallback.resolved_subgoal,
                recommended_action_id=fallback.recommended_action_id,
                recommended_action_type=fallback.recommended_action_type,
                ),
                context,
            )

    return handler


def _tool_result_from_payload(tool_id: str, payload: dict[str, Any]) -> ToolResult:
    return ToolResult(
        tool_id=str(payload.get("tool_id", tool_id)),
        status=ExecutionStatus(str(payload.get("status", ExecutionStatus.BLOCKED.value))),
        summary=str(payload.get("summary", "")),
        details=dict(payload.get("details", {})),
        resolved_subgoal=bool(payload.get("resolved_subgoal", False)),
        recommended_action_id=payload.get("recommended_action_id"),
        recommended_action_type=payload.get("recommended_action_type"),
    )


def _execute_local_perception_tool(
    tool_id: str,
    context: ToolCallContext,
    runtime: _LocalToolRuntime,
) -> ToolResult:
    payload = execute_perception_tool(
        tool_id,
        context={
            "goal": {
                "user_instruction": context.goal.user_instruction,
                "structured_goal": context.goal.structured_goal,
            },
            "subgoal": {
                "text": context.subgoal.text,
                "kind": context.subgoal.kind,
                "target": context.subgoal.target,
            },
            "world_state": serialize_world_state(context.world_state),
            "payload": dict(context.payload),
        },
        brain={"latest_scene": runtime.latest_scene} if runtime.latest_scene else None,
    )
    scene = extract_scene_from_tool_result(payload)
    if scene is not None:
        runtime.latest_scene = scene
    return _enrich_tool_result(_tool_result_from_payload(tool_id, payload), context)


def _serialize_world_state(world_state: WorldState) -> dict[str, Any]:
    return serialize_world_state(world_state)


def _extract_map_info(world_state: WorldState) -> dict[str, Any] | None:
    metadata = world_state.metadata
    candidates = (
        metadata.get("map"),
        metadata.get("navigation_map"),
        metadata.get("nav_map"),
    )
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate:
            map_info = dict(candidate)
            map_info.setdefault("named_places", sorted({place.name for place in world_state.place_memories}))
            return map_info
    maps = metadata.get("maps")
    if isinstance(maps, list):
        for candidate in maps:
            if isinstance(candidate, dict) and candidate:
                map_info = dict(candidate)
                map_info.setdefault("named_places", sorted({place.name for place in world_state.place_memories}))
                return map_info
    return None


def _pose_label(pose: Any) -> str:
    if isinstance(pose, dict):
        if "name" in pose:
            return str(pose["name"])
        if {"x", "y"} <= set(pose):
            return f"({pose['x']}, {pose['y']})"
    return str(pose)


def _enrich_tool_result(tool_result: ToolResult, context: ToolCallContext) -> ToolResult:
    if tool_result.updated_world_state is not None:
        return tool_result
    updated_world_state = _infer_updated_world_state(tool_result, context)
    if updated_world_state is None:
        return tool_result
    return ToolResult(
        tool_id=tool_result.tool_id,
        status=tool_result.status,
        summary=tool_result.summary,
        details=tool_result.details,
        resolved_subgoal=tool_result.resolved_subgoal,
        recommended_action_id=tool_result.recommended_action_id,
        recommended_action_type=tool_result.recommended_action_type,
        updated_world_state=updated_world_state,
    )


def _infer_updated_world_state(tool_result: ToolResult, context: ToolCallContext) -> WorldState | None:
    world_state = context.world_state
    updated = replace(world_state)
    updated.metadata = dict(world_state.metadata)
    changed = False

    scene = _scene_from_details(tool_result.details)
    if scene is not None:
        updated = _inject_scene_into_world_state(updated, scene)
        changed = True

    if tool_result.tool_id == "go_to_pose" and tool_result.status == ExecutionStatus.SUCCEEDED:
        pose = tool_result.details.get("pose_label") or tool_result.details.get("pose")
        if pose is None:
            task = tool_result.details.get("task")
            if isinstance(task, dict):
                result = task.get("result")
                if isinstance(result, dict):
                    pose = result.get("goal_label") or result.get("goal_pose")
        pose_label = _normalize_pose_label(pose)
        if pose_label and pose_label != updated.current_pose:
            updated.current_pose = pose_label
            updated.place_memories = tuple(_merge_places(updated.place_memories, pose_label, "tool:go_to_pose"))
            changed = True

    if tool_result.tool_id in {"get_map", "create_map", "explore", "get_task_status"}:
        map_info = tool_result.details.get("map")
        if map_info is None:
            task = tool_result.details.get("task")
            if isinstance(task, dict):
                result = task.get("result")
                if isinstance(result, dict):
                    map_info = result.get("map")
        if isinstance(map_info, dict) and map_info:
            updated.metadata["map"] = dict(map_info)
            for item in map_info.get("regions", []):
                if isinstance(item, dict) and item.get("label"):
                    updated.place_memories = tuple(
                        _merge_places(updated.place_memories, str(item["label"]), "tool:map_region")
                    )
            for item in map_info.get("named_places", []):
                name = item.get("name") if isinstance(item, dict) else item
                if name:
                    updated.place_memories = tuple(
                        _merge_places(updated.place_memories, str(name), "tool:named_place")
                    )
            changed = True

    if tool_result.tool_id == "get_task_status" and tool_result.status == ExecutionStatus.SUCCEEDED:
        task = tool_result.details.get("task")
        if isinstance(task, dict) and task.get("tool_id") == "go_to_pose" and task.get("state") == ExecutionStatus.SUCCEEDED.value:
            result = task.get("result")
            if isinstance(result, dict):
                pose_label = _normalize_pose_label(result.get("goal_label") or result.get("goal_pose"))
                if pose_label and pose_label != updated.current_pose:
                    updated.current_pose = pose_label
                    updated.place_memories = tuple(_merge_places(updated.place_memories, pose_label, "tool:get_task_status"))
                    changed = True

    if tool_result.tool_id == "set_waypoint_from_object":
        waypoint = tool_result.details.get("waypoint")
        if isinstance(waypoint, dict):
            updated.metadata["waypoint_hint"] = dict(waypoint)
            changed = True

    return updated if changed else None


def _scene_from_details(details: dict[str, Any]) -> dict[str, Any] | None:
    if details.get("annotations"):
        return dict(details)
    scene = details.get("scene")
    if isinstance(scene, dict):
        return dict(scene)
    return None


def _inject_scene_into_world_state(world_state: WorldState, scene: dict[str, Any]) -> WorldState:
    updated = replace(world_state)
    updated.metadata = dict(world_state.metadata)
    perception = dict(updated.metadata.get("perception", {})) if isinstance(updated.metadata.get("perception"), dict) else {}
    perception["annotations"] = [
        dict(item)
        for item in scene.get("annotations", [])
        if isinstance(item, dict)
    ]
    perception["scene_summary"] = scene.get("scene_summary")
    perception["generated_at"] = scene.get("generated_at")
    updated.metadata["perception"] = perception

    labels = {
        str(item.get("label"))
        for item in scene.get("annotations", [])
        if isinstance(item, dict) and item.get("label")
    }
    if labels:
        updated.visible_objects = frozenset(set(updated.visible_objects) | labels)
        observation_tokens = {f"{label.lower().replace(' ', '_')}_visible" for label in labels}
        updated.available_observations = frozenset(set(updated.available_observations) | observation_tokens)

    summary = scene.get("scene_summary")
    if isinstance(summary, str) and summary.strip():
        descriptions = list(updated.image_descriptions)
        if not descriptions or descriptions[-1] != summary:
            descriptions.append(summary)
        updated.image_descriptions = tuple(descriptions[-12:])
    return updated


def _normalize_pose_label(pose: Any) -> str | None:
    if pose is None:
        return None
    if isinstance(pose, dict):
        if pose.get("name"):
            return str(pose["name"])
        if {"x", "y"} <= set(pose):
            return "explicit_pose"
    label = str(pose).strip()
    return label or None


def _merge_places(existing: tuple[PlaceMemory, ...], name: str, evidence: str) -> list[PlaceMemory]:
    places = list(existing)
    places.append(PlaceMemory(name=name, confidence=0.8, evidence=evidence))
    deduped: dict[str, PlaceMemory] = {}
    for place in places:
        current = deduped.get(place.name)
        if current is None or place.confidence > current.confidence:
            deduped[place.name] = place
    return list(deduped.values())
