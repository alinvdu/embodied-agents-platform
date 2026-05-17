from __future__ import annotations

from dataclasses import dataclass, field, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable
import urllib.error
import urllib.request
import uuid

from .home_memory import (
    DEFAULT_NAVIGATION_CLEARANCE_M,
    DEFAULT_NAVIGATION_WAYPOINT_HORIZON_M,
    HomeMemoryStore,
    home_memory_preview_map,
    home_memory_agent_context,
    resolve_region_navigation_goal,
    summarize_home_memory,
)
from .memory_discovery import EnvironmentMemoryDiscovery
from .llm import AgentLLMRouter, AgentModelSuite, ModelConfig
from .perception_service import execute_perception_tool


EventSink = Callable[[str, str, str, dict[str, Any] | None], None]


@dataclass(frozen=True)
class HomeAgentModelConfig:
    provider: str = "mock"
    model: str = "mock"
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.2
    max_tokens: int = 1200
    reasoning_effort: str | None = None
    verbosity: str | None = None


@dataclass(frozen=True)
class HomeAgentConfig:
    home_memory_path: str | None = None
    home_memory_search_roots: tuple[str, ...] = field(default_factory=tuple)
    model: HomeAgentModelConfig = field(default_factory=HomeAgentModelConfig)
    specialist_model: HomeAgentModelConfig | None = None
    dry_run: bool = True
    auto_execute_navigation: bool = False
    require_skill_approval: bool = True
    host: str = "127.0.0.1"
    port: int = 8765
    max_turns: int = 18
    exploration_backend_url: str | None = "http://127.0.0.1:8770"
    navigation_waypoint_horizon_m: float = DEFAULT_NAVIGATION_WAYPOINT_HORIZON_M
    backend_request_timeout_s: float = 120.0


@dataclass
class HomeAgentRunRecord:
    run_id: str
    command: str
    status: str = "running"
    summary: str = ""
    actions: list[dict[str, Any]] = field(default_factory=list)
    memory_summary: str = ""
    started_at: float = field(default_factory=time.time)
    completed_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "command": self.command,
            "status": self.status,
            "summary": self.summary,
            "actions": list(self.actions),
            "memory_summary": self.memory_summary,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


class HomeAgentToolRuntime:
    def __init__(
        self,
        *,
        memory: dict[str, Any] | None,
        config: HomeAgentConfig,
        emit: EventSink,
    ) -> None:
        self.memory = memory or {}
        self.config = config
        self.emit = emit
        self.current_pose = self._initial_pose()
        self.stopped = False

    def preview_path_to_pose(
        self,
        *,
        target_label: str,
        pose: dict[str, Any],
        constraints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = {
            "tool": "preview_path_to_pose",
            "status": "succeeded",
            "target_label": target_label,
            "goal_pose": _json_pose(pose),
            "path": self._straight_line_path(pose),
            "constraints": constraints or {},
            "planner": "nav2_preview_placeholder",
            "dry_run": True,
        }
        self.emit(
            "tool_executed",
            "Path Preview",
            f"Prepared a navigation preview toward `{target_label}`.",
            result,
        )
        return result

    def resolve_navigation_to_region(
        self,
        *,
        target_label: str,
        constraints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        constraints = constraints or {}
        result = resolve_region_navigation_goal(
            self.memory,
            target_label,
            current_pose=self.current_pose,
            min_clearance_m=float(constraints.get("min_clearance_m", DEFAULT_NAVIGATION_CLEARANCE_M) or DEFAULT_NAVIGATION_CLEARANCE_M),
            waypoint_horizon_m=float(
                constraints.get("waypoint_horizon_m", self.config.navigation_waypoint_horizon_m)
                or self.config.navigation_waypoint_horizon_m
            ),
        )
        self.emit(
            "tool_executed" if result.get("status") in {"succeeded", "low_clearance"} else "tool_blocked",
            "Region Navigation Resolver",
            (
                f"Resolved `{target_label}` to known free space."
                if result.get("goal_pose")
                else f"Could not resolve `{target_label}` to a safe navigation pose."
            ),
            result,
        )
        return result

    def navigate_to_waypoint(
        self,
        *,
        waypoint_id: str,
        x: float,
        y: float,
        yaw: float = 0.0,
        constraints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        pose = _json_pose({"x": x, "y": y, "yaw": yaw})
        if not self.config.exploration_backend_url:
            result = {
                "tool": "navigate_to_waypoint",
                "status": "unavailable",
                "waypoint_id": waypoint_id,
                "requested_pose": pose,
                "reason": "No exploration backend URL is configured for Nav2 waypoint execution.",
            }
            self.emit("tool_blocked", "Waypoint Navigation", result["reason"], result)
            return result

        response = _post_exploration_backend(
            self.config,
            "/api/nav/waypoint",
            {"pose": pose, "waypoint_id": waypoint_id, "constraints": constraints or {}},
        )
        result = _navigation_tool_result(
            response,
            waypoint_id=waypoint_id,
            requested_pose=pose,
            backend_url=self.config.exploration_backend_url,
        )
        current_pose = result.get("current_pose")
        if isinstance(current_pose, dict):
            self.current_pose = _json_pose(current_pose)
        elif result.get("status") == "succeeded":
            self.current_pose = pose
            result["current_pose"] = dict(pose)
        self.emit(
            "tool_executed" if result.get("status") == "succeeded" else "tool_blocked",
            "Waypoint Navigation",
            (
                f"Nav2 reached waypoint `{waypoint_id}`."
                if result.get("status") == "succeeded"
                else f"Nav2 waypoint `{waypoint_id}` returned `{result.get('status')}`."
            ),
            result,
        )
        return result

    def relocalize_here(self) -> dict[str, Any]:
        if not self.config.exploration_backend_url:
            result = {
                "tool": "relocalize_here",
                "status": "unavailable",
                "reason": "No exploration backend URL is configured for relocalization.",
            }
            self.emit("tool_blocked", "Relocalization", result["reason"], result)
            return result

        response = _post_exploration_backend(self.config, "/api/nav/relocalize", {})
        result = _relocalization_tool_result(response, backend_url=self.config.exploration_backend_url)
        current_pose = result.get("current_pose")
        if isinstance(current_pose, dict):
            self.current_pose = _json_pose(current_pose)
        self.emit(
            "tool_executed" if result.get("status") in {"corrected", "skipped"} else "tool_blocked",
            "Relocalization",
            str(result.get("message") or result.get("reason") or f"Relocalization {result.get('status')}."),
            result,
        )
        return result

    def preview_path_to_region(
        self,
        *,
        target_label: str,
        constraints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved = self.resolve_navigation_to_region(target_label=target_label, constraints=constraints)
        pose = resolved.get("goal_pose")
        if not isinstance(pose, dict):
            return resolved
        preview = self.preview_path_to_pose(target_label=target_label, pose=pose, constraints=constraints)
        preview["resolved_goal"] = resolved
        return preview

    def navigate_to_pose(
        self,
        *,
        target_label: str,
        pose: dict[str, Any],
        constraints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        dry_run = bool(self.config.dry_run)
        result = {
            "tool": "navigate_to_pose",
            "status": "dry_run" if dry_run else "accepted",
            "target_label": target_label,
            "goal_pose": _json_pose(pose),
            "constraints": constraints or {},
            "nav_backend": "nav2_placeholder",
            "dry_run": dry_run,
        }
        if not dry_run:
            self.current_pose = dict(result["goal_pose"])
        self.emit(
            "tool_executed",
            "Navigation Goal",
            (
                f"Prepared a dry-run Nav2 goal for `{target_label}`."
                if dry_run
                else f"Accepted a Nav2 goal for `{target_label}`."
            ),
            result,
        )
        return result

    def navigate_to_region(
        self,
        *,
        target_label: str,
        constraints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved = self.resolve_navigation_to_region(target_label=target_label, constraints=constraints)
        pose = resolved.get("goal_pose")
        if not isinstance(pose, dict):
            return resolved
        if resolved.get("status") != "succeeded":
            return {
                **resolved,
                "status": "blocked",
                "reason": resolved.get("reason") or "Resolved region target did not meet navigation clearance requirements.",
            }
        nav = self.navigate_to_pose(target_label=target_label, pose=pose, constraints=constraints)
        nav["resolved_goal"] = resolved
        return nav

    def perceive_scene(self, *, target_label: str = "") -> dict[str, Any]:
        world_state = {
            "current_task": "home_agent_perception",
            "current_pose": self.current_pose,
            "metadata": {"home_memory": home_memory_agent_context(self.memory) if self.memory else {}},
        }
        payload = execute_perception_tool(
            "perceive_scene",
            context={
                "world_state": world_state,
                "payload": {"target": target_label},
                "subgoal": {"text": f"perceive {target_label}".strip(), "kind": "search", "target": target_label},
            },
            brain=None,
        )
        self.emit(
            "tool_executed",
            "Perception",
            f"Refreshed scene perception{f' for `{target_label}`' if target_label else ''}.",
            payload,
        )
        return payload

    def analyze_embodied_scene(self, *, target_label: str = "", question: str = "") -> dict[str, Any]:
        model = self.config.specialist_model
        if model is None:
            result = {
                "tool": "analyze_embodied_scene",
                "status": "unavailable",
                "target_label": target_label,
                "question": question,
                "reason": "no specialist embodied-reasoning model configured",
            }
            self.emit(
                "specialist_unavailable",
                "Specialist Unavailable",
                "No embodied-reasoning specialist model is configured.",
                result,
            )
            return result
        if model.provider == "mock":
            result = {
                "tool": "analyze_embodied_scene",
                "status": "succeeded",
                "target_label": target_label,
                "question": question,
                "analysis": "Mock specialist: use long-term approach pose, refresh RGB-D locally, then verify reachability before skill execution.",
                "confidence": 0.5,
            }
            self.emit("specialist_result", "Embodied Reasoning", result["analysis"], result)
            return result

        router_config = _llm_model_config(model)
        router = AgentLLMRouter(
            AgentModelSuite(planner=router_config, critic=router_config, coder=router_config)
        )
        prompt = json.dumps(
            {
                "target_label": target_label,
                "question": question,
                "current_pose": self.current_pose,
                "home_memory_context": home_memory_agent_context(self.memory) if self.memory else {},
            },
            indent=2,
            sort_keys=True,
        )
        parsed, trace = router.complete_json_prompt(
            config=router_config,
            system_prompt=(
                "You are a robotics embodied-reasoning specialist. "
                "Analyze physical/spatial feasibility for a household robot. "
                "Return JSON only with keys: analysis, confidence, suggested_next_tool, risks."
            ),
            user_prompt=prompt,
        )
        if parsed is None:
            result = {
                "tool": "analyze_embodied_scene",
                "status": "failed",
                "target_label": target_label,
                "question": question,
                "error": trace.error,
            }
            self.emit("specialist_failed", "Specialist Failed", trace.error or "Specialist call failed.", result)
            return result
        result = {
            "tool": "analyze_embodied_scene",
            "status": "succeeded",
            "target_label": target_label,
            "question": question,
            "analysis": str(parsed.get("analysis", "")),
            "confidence": parsed.get("confidence"),
            "suggested_next_tool": parsed.get("suggested_next_tool"),
            "risks": parsed.get("risks", []),
        }
        self.emit("specialist_result", "Embodied Reasoning", result["analysis"], result)
        return result

    def run_skill(
        self,
        *,
        skill_id: str,
        target_label: str = "",
        constraints: dict[str, Any] | None = None,
        approved: bool = False,
    ) -> dict[str, Any]:
        skill = self._skill(skill_id)
        if skill is None:
            result = {
                "tool": "run_skill",
                "status": "blocked",
                "skill_id": skill_id,
                "target_label": target_label,
                "reason": "skill_not_registered_in_home_memory",
            }
            self.emit("tool_blocked", "Skill Blocked", f"`{skill_id}` is not registered yet.", result)
            return result
        requires_approval = bool((skill.get("safety") or {}).get("requires_human_approval", False))
        if self.config.require_skill_approval and requires_approval and not approved:
            result = {
                "tool": "run_skill",
                "status": "approval_required",
                "skill_id": skill_id,
                "target_label": target_label,
                "constraints": constraints or {},
                "reason": "first manipulation/specialized skill attempts require operator approval",
            }
            self.emit(
                "approval_required",
                "Approval Required",
                f"`{skill_id}` is ready to be invoked, but needs operator approval first.",
                result,
            )
            return result
        result = {
            "tool": "run_skill",
            "status": "dry_run" if self.config.dry_run else "accepted",
            "skill_id": skill_id,
            "target_label": target_label,
            "constraints": constraints or {},
            "executor": skill.get("executor_binding", "vla_skill_runner"),
            "dry_run": bool(self.config.dry_run),
        }
        self.emit("tool_executed", "Skill", f"Prepared `{skill_id}` for `{target_label}`.", result)
        return result

    def ask_human_approval(self, *, reason: str, action: dict[str, Any] | None = None) -> dict[str, Any]:
        result = {
            "tool": "ask_human_approval",
            "status": "pending",
            "reason": reason,
            "action": action or {},
        }
        self.emit("approval_required", "Approval Requested", reason, result)
        return result

    def stop_robot(self, *, reason: str = "") -> dict[str, Any]:
        self.stopped = True
        result = {
            "tool": "stop_robot",
            "status": "accepted",
            "reason": reason or "operator_or_agent_requested_stop",
        }
        self.emit("tool_executed", "Stop", "Stop request accepted by the home agent runtime.", result)
        return result

    def _initial_pose(self) -> dict[str, Any]:
        start = self.memory.get("start_pose") if isinstance(self.memory, dict) else None
        if isinstance(start, dict) and isinstance(start.get("pose"), dict):
            return _json_pose(start["pose"])
        return {"x": 0.0, "y": 0.0, "yaw": 0.0}

    def _straight_line_path(self, pose: dict[str, Any]) -> list[dict[str, float]]:
        return [dict(self.current_pose), _json_pose(pose)]

    def _skill(self, skill_id: str) -> dict[str, Any] | None:
        for skill in self.memory.get("skills", []):
            if isinstance(skill, dict) and skill.get("skill_id") == skill_id:
                return skill
        return None


class HomeTaskAgent:
    def __init__(self, config: HomeAgentConfig, emit: EventSink | None = None) -> None:
        self.config = config
        self.emit = emit or (lambda kind, title, summary, details=None: None)

    def run(self, command: str) -> HomeAgentRunRecord:
        memory = self._load_memory()
        record = HomeAgentRunRecord(
            run_id=f"home_agent_{uuid.uuid4().hex[:10]}",
            command=command,
            memory_summary=summarize_home_memory(memory) if memory else "No home memory loaded.",
        )
        self.emit("session_started", "Run Started", f"Started HomeTaskAgent for `{command}`.", record.to_dict())
        runtime = HomeAgentToolRuntime(memory=memory, config=self.config, emit=self._recording_emit(record))
        try:
            if self.config.model.provider == "mock":
                self._run_deterministic(command, memory, runtime, record)
            else:
                self._run_agents_sdk(command, memory, runtime, record)
            if record.status == "running":
                record.status = "completed"
        except Exception as exc:
            record.status = "failed"
            record.summary = f"HomeTaskAgent failed: {exc}"
            self.emit("session_failed", "Run Failed", record.summary, {"error": str(exc)})
        record.completed_at = time.time()
        self.emit("session_finished", "Run Finished", record.summary or record.status, record.to_dict())
        return record

    def _load_memory(self) -> dict[str, Any] | None:
        path = resolve_home_memory_path(self.config)
        if path is None:
            return None
        return HomeMemoryStore(path).load()

    def _recording_emit(self, record: HomeAgentRunRecord) -> EventSink:
        def emit(kind: str, title: str, summary: str, details: dict[str, Any] | None = None) -> None:
            if details and details.get("tool"):
                record.actions.append(dict(details))
            self.emit(kind, title, summary, details)

        return emit

    def _run_deterministic(
        self,
        command: str,
        memory: dict[str, Any] | None,
        runtime: HomeAgentToolRuntime,
        record: HomeAgentRunRecord,
    ) -> None:
        if not memory:
            record.status = "blocked"
            record.summary = "No home memory path is configured, so the agent cannot resolve places yet."
            self.emit("agent_blocked", "Missing Memory", record.summary, {})
            return
        if "stop" in command.lower():
            record.status = "blocked"
            record.summary = "Only region navigation resolution is exposed in this phase."
            self.emit("agent_blocked", "Tool Not Exposed", record.summary, {"requested": "stop_robot"})
            return

        target = self._target_from_command(command, memory)
        if target is None:
            labels = ", ".join(_known_region_labels(memory)) or "none"
            record.status = "blocked"
            record.summary = f"I could not match the command to a known region. Known regions: {labels}."
            self.emit("agent_blocked", "Target Not Found", record.summary, {"known_regions": _known_region_labels(memory)})
            return

        self.emit("memory_resolved", "Memory Target", f"Resolved `{target['label']}` from home memory.", target)
        result = self._navigation_call(runtime, target)
        if result.get("goal_pose"):
            record.summary = f"Resolved `{target['label']}` to a concrete navigation preview point."
        else:
            record.status = "blocked"
            record.summary = f"Could not resolve `{target['label']}` to a safe navigation preview point."

    def _run_agents_sdk(
        self,
        command: str,
        memory: dict[str, Any] | None,
        runtime: HomeAgentToolRuntime,
        record: HomeAgentRunRecord,
    ) -> None:
        try:
            from agents import Agent, ModelSettings, Runner, function_tool
        except ImportError as exc:
            raise RuntimeError("OpenAI Agents SDK is not installed") from exc

        @function_tool
        def resolve_navigation_to_region(target_label: str, constraints_json: str = "{}") -> str:
            """Resolve a semantic region label into a concrete safe path, final pose, and short-horizon waypoint."""
            return json.dumps(
                runtime.resolve_navigation_to_region(
                    target_label=target_label,
                    constraints=_loads_object(constraints_json),
                )
            )

        @function_tool
        def navigate_to_waypoint(waypoint_id: str, x: float, y: float, yaw: float = 0.0, constraints_json: str = "{}") -> str:
            """Send a resolved waypoint to the live exploration/Nav2 backend and wait for the result."""
            return json.dumps(
                runtime.navigate_to_waypoint(
                    waypoint_id=waypoint_id,
                    x=x,
                    y=y,
                    yaw=yaw,
                    constraints=_loads_object(constraints_json),
                )
            )

        @function_tool
        def relocalize_here() -> str:
            """Run the existing exploration backend relocalization scan and odometry correction."""
            return json.dumps(runtime.relocalize_here())

        agent = Agent(
            name="NavigationAgent",
            instructions=self._agent_instructions(memory),
            model=self._sdk_model(),
            model_settings=self._sdk_model_settings(ModelSettings),
            tools=[
                resolve_navigation_to_region,
                navigate_to_waypoint,
                relocalize_here,
            ],
        )
        result = Runner.run_sync(agent, command, max_turns=self.config.max_turns)
        record.summary = str(getattr(result, "final_output", "") or "Agent run completed.").strip()

    def _agent_instructions(self, memory: dict[str, Any] | None) -> str:
        context = home_memory_agent_context(memory) if memory else {}
        return "\n".join(
            [
                "You are Robot42's HomeTaskAgent.",
                "For navigation commands, act as the NavigationAgent delegated by HomeTaskAgent.",
                "You receive the full long-term home memory in context. Do not call memory lookup tools.",
                "Available navigation tools are resolve_navigation_to_region, navigate_to_waypoint, and relocalize_here.",
                "For semantic region navigation such as `go to kitchen`, first call resolve_navigation_to_region.",
                f"The default short-horizon waypoint length is {self.config.navigation_waypoint_horizon_m:.1f} meters.",
                "Use the resolver's next_waypoint exactly; do not invent arbitrary waypoint coordinates.",
                "Call navigate_to_waypoint with next_waypoint.waypoint_id, x, y, and yaw.",
                "After each successful waypoint, call relocalize_here before resolving the next waypoint.",
                "If navigation succeeds and next_waypoint.is_final_waypoint is false, call resolve_navigation_to_region again from the updated pose and repeat.",
                "If navigation fails but distance_remaining_m is small enough for the task, say that clearly; otherwise report the failure and stop.",
                "Do not call or describe perception, skill execution, approval, or stop tools; they are intentionally not exposed yet.",
                "Do not infer a navigation target yourself from a region shape.",
                "Return a concise final summary of the navigation status, final/current pose, and any relocalization correction.",
                "",
                "Example navigation loop for `go to kitchen`:",
                "1. Call resolve_navigation_to_region(target_label='kitchen', constraints_json='{}').",
                "2. If the result has status succeeded or low_clearance and includes next_waypoint, call navigate_to_waypoint with that exact waypoint and constraints_json='{}'.",
                "3. If the waypoint succeeded, call relocalize_here.",
                "4. If the waypoint succeeded and was not final, repeat from step 1.",
                "5. If the waypoint succeeded and was final, summarize that the region was reached.",
                "6. If the waypoint failed, summarize status, reason, distance_remaining_m, and current_pose.",
                "",
                "Example custom horizon: resolve_navigation_to_region(target_label='kitchen', constraints_json='{\"waypoint_horizon_m\": 1.5}').",
                "",
                "Long-term home memory context:",
                json.dumps(context, indent=2, sort_keys=True),
            ]
        )

    def _sdk_model(self) -> Any:
        provider = self.config.model.provider
        if provider == "litellm":
            from agents.extensions.models.litellm_model import LitellmModel

            return LitellmModel(
                model=self.config.model.model,
                base_url=self.config.model.base_url,
                api_key=self.config.model.api_key,
            )
        if provider == "openai-compatible":
            from agents import OpenAIChatCompletionsModel
            from openai import AsyncOpenAI

            if not self.config.model.base_url:
                raise ValueError("openai-compatible model provider requires base_url")
            client = AsyncOpenAI(
                base_url=self.config.model.base_url,
                api_key=self.config.model.api_key or "not-needed",
            )
            return OpenAIChatCompletionsModel(model=self.config.model.model, openai_client=client)
        if provider == "openai" and self.config.model.api_key and not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = self.config.model.api_key
        return self.config.model.model

    def _sdk_model_settings(self, model_settings_cls: Any) -> Any:
        kwargs: dict[str, Any] = {
            "max_tokens": self.config.model.max_tokens,
        }
        if self._uses_gpt5_model_settings():
            kwargs["reasoning"] = _reasoning_setting(self.config.model.reasoning_effort or "medium")
            kwargs["verbosity"] = self.config.model.verbosity or "low"
        else:
            kwargs["temperature"] = self.config.model.temperature
        return model_settings_cls(**kwargs)

    def _uses_gpt5_model_settings(self) -> bool:
        return self.config.model.provider == "openai" and _normalized_model_name(self.config.model.model).startswith("gpt-5")

    def _target_from_command(self, command: str, memory: dict[str, Any]) -> dict[str, Any] | None:
        labels = sorted(_known_region_labels(memory), key=len, reverse=True)
        command_key = command.lower().replace("_", " ")
        for label in labels:
            if label.lower().replace("_", " ") in command_key:
                region = self._semantic_region_target(memory, label)
                if region is not None:
                    return region
        return self._semantic_region_target(memory, command)

    def _semantic_region_target(self, memory: dict[str, Any], name_or_label: str) -> dict[str, Any] | None:
        query = name_or_label.lower().replace("_", " ")
        for region in memory.get("regions", []):
            if not isinstance(region, dict):
                continue
            label = str(region.get("label") or region.get("region_id") or "")
            normalized = label.lower().replace("_", " ")
            if normalized and (query == normalized or normalized in query or query in normalized):
                return {
                    "target_type": "region",
                    "label": label,
                    "region_id": region.get("region_id"),
                    "source": "home_memory.regions.semantic",
                }
        return None

    def _skill_from_command(self, command: str, memory: dict[str, Any]) -> str | None:
        normalized = command.lower().replace("_", " ")
        skills = [skill for skill in memory.get("skills", []) if isinstance(skill, dict)]
        for skill in skills:
            skill_id = str(skill.get("skill_id") or "")
            if skill_id and skill_id.replace("_", " ") in normalized:
                return skill_id
        if "open" in normalized and "fridge" in normalized:
            return "open_fridge"
        if ("inspect" in normalized or "what" in normalized) and "fridge" in normalized:
            return "inspect_fridge_contents"
        if "pick" in normalized and ("can" in normalized or "coke" in normalized):
            return "pick_can"
        if "place" in normalized:
            return "place_item"
        return None

    def _navigation_call(self, runtime: HomeAgentToolRuntime, target: dict[str, Any]) -> dict[str, Any]:
        label = str(target.get("label") or "target")
        return runtime.resolve_navigation_to_region(target_label=label)


class HomeAgentController:
    def __init__(self, agent: HomeTaskAgent, config: HomeAgentConfig) -> None:
        self.agent = agent
        self.config = config
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._status = "idle"
        self._events: list[dict[str, Any]] = []
        self._record: HomeAgentRunRecord | None = None
        self._paused = False

    @classmethod
    def from_config(cls, config: HomeAgentConfig) -> "HomeAgentController":
        controller: HomeAgentController | None = None

        def emit(kind: str, title: str, summary: str, details: dict[str, Any] | None = None) -> None:
            if controller is not None:
                controller.emit(kind, title, summary, details)

        agent = HomeTaskAgent(config, emit=emit)
        controller = cls(agent, config)
        return controller

    def start(self, command: str) -> bool:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._events = []
            self._record = None
            self._status = "running"
            self._thread = threading.Thread(target=self._run, args=(command,), daemon=True)
            self._thread.start()
            return True

    def pause(self) -> None:
        with self._lock:
            self._paused = True
            self.emit("paused", "Paused", "Pause requested. Long-running robot calls should honor this.", {})

    def resume(self) -> None:
        with self._lock:
            self._paused = False
            self.emit("resumed", "Resumed", "Resume requested.", {})

    def stop(self) -> None:
        with self._lock:
            self._status = "stopped"
            self.emit("stop_requested", "Stop Requested", "Operator requested stop.", {})

    def emit(self, kind: str, title: str, summary: str, details: dict[str, Any] | None = None) -> None:
        event = {
            "kind": kind,
            "title": title,
            "summary": summary,
            "details": details or {},
            "timestamp": _timestamp(),
        }
        with self._lock:
            self._events.append(event)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            memory = self._safe_memory()
            memory_path = resolve_home_memory_path(self.config)
            record = self._record.to_dict() if self._record is not None else None
            return {
                "status": self._status,
                "backend": "home_agent",
                "paused": self._paused,
                "models": {
                    "main": self.config.model.__dict__,
                    "specialist": self.config.specialist_model.__dict__ if self.config.specialist_model else None,
                },
                "home_memory": {
                    "path": str(memory_path) if memory_path is not None else self.config.home_memory_path,
                    "summary": summarize_home_memory(memory) if memory else "No home memory loaded.",
                    "context": home_memory_agent_context(memory) if memory else {},
                    "preview_map": home_memory_preview_map(memory) if memory else {},
                },
                "environment_memories": self.list_environment_memories(),
                "record": record,
                "plan": record or {},
                "report": {"events": list(self._events)},
                "events": list(self._events),
            }

    def list_environment_memories(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        for discovery in _memory_discoveries(self.config):
            for record in discovery.list():
                key = str(record.home_memory_path or record.directory)
                if key in seen:
                    continue
                seen.add(key)
                records.append(record.to_dict())
        return sorted(records, key=lambda item: (float(item.get("updated_at") or 0.0), item.get("memory_id") or ""), reverse=True)

    def select_environment_memory(self, memory_id: str) -> dict[str, Any] | None:
        for discovery in _memory_discoveries(self.config):
            record = discovery.get(memory_id)
            if record is None or record.home_memory_path is None:
                continue
            self.config = replace(self.config, home_memory_path=str(record.home_memory_path))
            self.agent.config = self.config
            self.emit("memory_selected", "Memory Selected", f"Selected `{record.memory_id}`.", record.to_dict())
            return record.to_dict()
        return None

    def create_environment_memory(self, memory_id: str, *, label: str | None = None) -> dict[str, Any]:
        discovery = _memory_discoveries(self.config)[0]
        record = discovery.create(memory_id, label=label)
        self.emit("memory_created", "Memory Created", f"Created draft memory `{record.memory_id}`.", record.to_dict())
        return record.to_dict()

    def _run(self, command: str) -> None:
        record = self.agent.run(command)
        with self._lock:
            self._record = record
            if self._status != "stopped":
                self._status = record.status

    def _safe_memory(self) -> dict[str, Any] | None:
        path = resolve_home_memory_path(self.config)
        if path is None:
            return None
        try:
            return HomeMemoryStore(path).load()
        except Exception:
            return None


class HomeAgentServer:
    def __init__(self, controller: HomeAgentController, *, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.controller = controller
        self.host = host
        self.port = port
        self._server: ThreadingHTTPServer | None = None

    def serve_forever(self) -> None:
        controller = self.controller

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path == "/api/state":
                    self._send_json(controller.snapshot())
                    return
                if self.path == "/api/memories":
                    self._send_json({"memories": controller.list_environment_memories()})
                    return
                if self.path == "/" or self.path == "/index.html":
                    self._send_json({"service": "Robot42 HomeAgent", "state": "/api/state"})
                    return
                self.send_error(HTTPStatus.NOT_FOUND, "not found")

            def do_POST(self) -> None:
                payload = self._read_json_body()
                if self.path == "/api/start":
                    command = str(payload.get("command") or "").strip()
                    if not command:
                        self.send_error(HTTPStatus.BAD_REQUEST, "command is required")
                        return
                    accepted = controller.start(command)
                    if not accepted:
                        self.send_error(HTTPStatus.CONFLICT, "an agent run is already active")
                        return
                    self._send_json({"status": "started"})
                    return
                if self.path == "/api/pause":
                    controller.pause()
                    self._send_json({"status": "paused"})
                    return
                if self.path == "/api/resume":
                    controller.resume()
                    self._send_json({"status": "running"})
                    return
                if self.path == "/api/stop":
                    controller.stop()
                    self._send_json({"status": "stopping"})
                    return
                if self.path == "/api/memory/select":
                    response = controller.select_environment_memory(str(payload.get("memory_id") or ""))
                    self._send_json(response or {"status": "missing"})
                    return
                if self.path == "/api/memory/create":
                    memory_id = str(payload.get("memory_id") or payload.get("label") or f"home_{int(time.time())}")
                    self._send_json(
                        controller.create_environment_memory(
                            memory_id,
                            label=payload.get("label"),
                        )
                    )
                    return
                self.send_error(HTTPStatus.NOT_FOUND, "not found")

            def log_message(self, format: str, *args: Any) -> None:
                return

            def _read_json_body(self) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length", "0"))
                if length == 0:
                    return {}
                return json.loads(self.rfile.read(length).decode("utf-8"))

            def _send_json(self, payload: dict[str, Any]) -> None:
                encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._server.serve_forever()

    def shutdown(self) -> None:
        if self._server is not None:
            self._server.shutdown()


def config_from_env() -> HomeAgentConfig:
    provider = os.getenv("ROBOT42_AGENT_PROVIDER", "mock")
    model = os.getenv("ROBOT42_AGENT_MODEL", "mock" if provider == "mock" else "gpt-5.5")
    specialist_provider = os.getenv("ROBOT42_SPECIALIST_PROVIDER")
    specialist_model = os.getenv("ROBOT42_SPECIALIST_MODEL")
    specialist = None
    if specialist_provider and specialist_model:
        specialist = HomeAgentModelConfig(
            provider=specialist_provider,
            model=specialist_model,
            base_url=os.getenv("ROBOT42_SPECIALIST_BASE_URL"),
            api_key=os.getenv("ROBOT42_SPECIALIST_API_KEY"),
        )
    return HomeAgentConfig(
        home_memory_path=os.getenv("ROBOT42_HOME_MEMORY_PATH"),
        home_memory_search_roots=_search_roots_from_env(),
        model=HomeAgentModelConfig(
            provider=provider,
            model=model,
            base_url=os.getenv("ROBOT42_AGENT_BASE_URL"),
            api_key=os.getenv("ROBOT42_AGENT_API_KEY") or os.getenv("OPENAI_API_KEY"),
            temperature=float(os.getenv("ROBOT42_AGENT_TEMPERATURE", "0.2")),
            max_tokens=int(os.getenv("ROBOT42_AGENT_MAX_TOKENS", "1200")),
            reasoning_effort=os.getenv("ROBOT42_AGENT_REASONING_EFFORT"),
            verbosity=os.getenv("ROBOT42_AGENT_VERBOSITY"),
        ),
        specialist_model=specialist,
        dry_run=_env_bool("ROBOT42_AGENT_DRY_RUN", True),
        auto_execute_navigation=_env_bool("ROBOT42_AGENT_AUTO_NAV", False),
        require_skill_approval=_env_bool("ROBOT42_AGENT_REQUIRE_SKILL_APPROVAL", True),
        host=os.getenv("ROBOT42_AGENT_HOST", "127.0.0.1"),
        port=int(os.getenv("ROBOT42_AGENT_PORT", "8765")),
        max_turns=int(os.getenv("ROBOT42_AGENT_MAX_TURNS", "18")),
        exploration_backend_url=os.getenv("ROBOT42_EXPLORATION_BACKEND_URL", "http://127.0.0.1:8770"),
        navigation_waypoint_horizon_m=float(
            os.getenv("ROBOT42_NAVIGATION_WAYPOINT_HORIZON_M", str(DEFAULT_NAVIGATION_WAYPOINT_HORIZON_M))
        ),
        backend_request_timeout_s=float(os.getenv("ROBOT42_AGENT_BACKEND_REQUEST_TIMEOUT_S", "120")),
    )


def resolve_home_memory_path(config: HomeAgentConfig) -> Path | None:
    if config.home_memory_path:
        path = Path(config.home_memory_path)
        if path.exists():
            return path
    return discover_latest_home_memory_path(config.home_memory_search_roots)


def discover_latest_home_memory_path(search_roots: tuple[str, ...] = tuple()) -> Path | None:
    candidates: list[Path] = []
    for discovery in _memory_discoveries(HomeAgentConfig(home_memory_search_roots=search_roots)):
        for record in discovery.list():
            if record.home_memory_path is not None:
                candidates.append(record.home_memory_path)
    for root in _raw_search_roots(search_roots):
        if root.is_file() and root.name.endswith(".home_memory.json"):
            candidates.append(root)
        elif root.exists():
            candidates.extend(path for path in root.rglob("*.home_memory.json") if "home_memory" in path.parts)
    existing = [path for path in set(candidates) if path.exists()]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)


def _memory_discoveries(config: HomeAgentConfig) -> list[EnvironmentMemoryDiscovery]:
    roots = _raw_search_roots(config.home_memory_search_roots)
    memory_roots: list[Path] = []
    for root in roots:
        if root.name == "memories":
            memory_roots.append(root)
        else:
            memory_roots.append(root / "memories")
            memory_roots.append(root / "artifacts" / "memories")
    if not memory_roots:
        memory_roots = [Path.cwd() / "artifacts" / "memories"]
    deduped: list[Path] = []
    for root in memory_roots:
        expanded = root.expanduser()
        if expanded not in deduped:
            deduped.append(expanded)
    return [EnvironmentMemoryDiscovery(root) for root in deduped]


def _raw_search_roots(search_roots: tuple[str, ...]) -> list[Path]:
    roots = [Path(item).expanduser() for item in search_roots if item]
    return roots or [Path.cwd()]


def _search_roots_from_env() -> tuple[str, ...]:
    value = os.getenv("ROBOT42_HOME_MEMORY_SEARCH_ROOTS", "")
    if not value.strip():
        return tuple()
    return tuple(item.strip() for item in value.split(":") if item.strip())


def _json_pose(pose: dict[str, Any]) -> dict[str, float]:
    return {
        "x": round(float(pose.get("x", 0.0) or 0.0), 3),
        "y": round(float(pose.get("y", 0.0) or 0.0), 3),
        "yaw": round(float(pose.get("yaw", 0.0) or 0.0), 3),
    }


def _loads_object(payload: str) -> dict[str, Any]:
    try:
        parsed = json.loads(payload or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _post_exploration_backend(config: HomeAgentConfig, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    base_url = str(config.exploration_backend_url or "").rstrip("/")
    url = f"{base_url}{path}"
    body = json.dumps(payload or {}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=max(float(config.backend_request_timeout_s), 1.0)) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        try:
            raw_error = exc.read().decode("utf-8")
        except Exception:
            raw_error = str(exc)
        return {
            "status": "failed",
            "reason": f"Exploration backend returned HTTP {exc.code}: {raw_error[:400]}",
            "_transport_error": True,
            "_backend_url": url,
        }
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {
            "status": "unavailable",
            "reason": f"Exploration backend is unavailable at {url}: {exc}",
            "_transport_error": True,
            "_backend_url": url,
        }
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {
            "status": "failed",
            "reason": f"Exploration backend returned non-JSON response from {url}.",
            "_transport_error": True,
            "_backend_url": url,
        }
    return parsed if isinstance(parsed, dict) else {"status": "failed", "reason": "Exploration backend response was not an object."}


def _navigation_tool_result(
    response: dict[str, Any],
    *,
    waypoint_id: str,
    requested_pose: dict[str, float],
    backend_url: str | None,
) -> dict[str, Any]:
    nav2 = response.get("nav2_result") if isinstance(response.get("nav2_result"), dict) else {}
    plan = nav2.get("plan") if isinstance(nav2.get("plan"), dict) else {}
    status = str(response.get("status") or nav2.get("status") or "failed")
    current_pose = _pose_from_backend_response(response)
    distance_remaining_m = _remaining_distance_m(nav2)
    if distance_remaining_m is None and status == "succeeded":
        distance_remaining_m = 0.0
    if distance_remaining_m is None and current_pose is not None:
        distance_remaining_m = round(_pose_distance_m(current_pose, requested_pose), 3)
    return {
        "tool": "navigate_to_waypoint",
        "status": status,
        "waypoint_id": waypoint_id,
        "requested_pose": requested_pose,
        "current_pose": current_pose,
        "normalized_pose": _json_pose(response["normalized_pose"]) if isinstance(response.get("normalized_pose"), dict) else None,
        "distance_remaining_m": distance_remaining_m,
        "reason": response.get("reason") or nav2.get("reason") or response.get("message") or "",
        "backend_url": backend_url,
        "nav2": {
            "status": nav2.get("status"),
            "reason": nav2.get("reason"),
            "travelled_distance_m": nav2.get("travelled_distance_m"),
            "reached_pose": _json_pose(nav2["reached_pose"]) if isinstance(nav2.get("reached_pose"), dict) else None,
            "plan_status": plan.get("status"),
            "plan_reason": plan.get("reason"),
            "path_length_m": plan.get("path_length_m"),
        },
    }


def _relocalization_tool_result(response: dict[str, Any], *, backend_url: str | None) -> dict[str, Any]:
    match = response.get("match") if isinstance(response.get("match"), dict) else {}
    correction = response.get("correction") if isinstance(response.get("correction"), dict) else {}
    current_pose = _pose_from_backend_response(response)
    corrected_pose = _json_pose(match["corrected_pose"]) if isinstance(match.get("corrected_pose"), dict) else None
    if current_pose is None and corrected_pose is not None:
        current_pose = corrected_pose
    return {
        "tool": "relocalize_here",
        "status": str(response.get("status") or "failed"),
        "message": response.get("message"),
        "reason": response.get("reason"),
        "backend_url": backend_url,
        "current_pose": current_pose,
        "match": {
            "status": match.get("status"),
            "confidence": match.get("confidence"),
            "delta": match.get("delta"),
            "corrected_pose": corrected_pose,
            "reason": match.get("reason"),
        },
        "correction": {
            "status": correction.get("status"),
            "reason": correction.get("reason"),
        },
    }


def _pose_from_backend_response(response: dict[str, Any]) -> dict[str, float] | None:
    nav2 = response.get("nav2_result") if isinstance(response.get("nav2_result"), dict) else {}
    for candidate in (
        nav2.get("reached_pose"),
        (response.get("map") or {}).get("robot_pose") if isinstance(response.get("map"), dict) else None,
    ):
        if isinstance(candidate, dict):
            return _json_pose(candidate)
    return None


def _remaining_distance_m(nav2: dict[str, Any]) -> float | None:
    samples = nav2.get("feedback_samples")
    if not isinstance(samples, list) or not samples:
        return None
    last = samples[-1]
    if not isinstance(last, dict):
        return None
    value = last.get("remaining_distance_m") or last.get("distance_remaining_m")
    if value is None:
        return None
    try:
        return round(float(value), 3)
    except Exception:
        return None


def _pose_distance_m(a: dict[str, Any], b: dict[str, Any]) -> float:
    return round(
        ((float(a.get("x", 0.0) or 0.0) - float(b.get("x", 0.0) or 0.0)) ** 2
         + (float(a.get("y", 0.0) or 0.0) - float(b.get("y", 0.0) or 0.0)) ** 2)
        ** 0.5,
        3,
    )


def _known_region_labels(memory: dict[str, Any]) -> list[str]:
    labels = [
        str(region.get("label") or region.get("region_id"))
        for region in memory.get("regions", [])
        if isinstance(region, dict) and (region.get("label") or region.get("region_id"))
    ]
    return sorted(set(labels), key=lambda item: item.lower())


def _llm_model_config(config: HomeAgentModelConfig) -> ModelConfig:
    provider = config.provider
    base_url = config.base_url
    if provider == "openai":
        provider = "openai-compatible"
        base_url = base_url or "https://api.openai.com/v1/chat/completions"
    return ModelConfig(
        provider=provider,
        model=config.model,
        base_url=base_url,
        api_key=config.api_key,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )


def _normalized_model_name(model: str) -> str:
    return model.strip().lower()


def _reasoning_setting(effort: str) -> Any:
    try:
        from openai.types.shared import Reasoning
    except Exception:
        return {"effort": effort}
    return Reasoning(effort=effort)


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
