from __future__ import annotations

import base64
from dataclasses import dataclass, field, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import mimetypes
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable
import urllib.error
import urllib.request
import uuid

from .home_memory import (
    DEFAULT_DIRECT_NAVIGATION_FALLBACK_MAX_DISTANCE_M,
    DEFAULT_NAVIGATION_CLEARANCE_M,
    DEFAULT_NAVIGATION_WAYPOINT_HORIZON_M,
    HomeMemoryStore,
    home_memory_preview_map,
    home_memory_agent_context,
    plan_region_exploration as plan_home_region_exploration,
    resolve_direct_navigation_fallback,
    resolve_local_clearance_recovery,
    resolve_object_surface_approach_pose,
    resolve_region_navigation_goal,
    summarize_home_memory,
)
from .memory_discovery import EnvironmentMemoryDiscovery
from .llm import AgentLLMRouter, AgentModelSuite, ModelConfig
from .object_detection import ObjectDetectorConfig, detect_object_in_image
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
    navigation_auto_rotate_threshold_deg: float = 45.0
    backend_request_timeout_s: float = 120.0
    agent_artifacts_root: str = "artifacts/agent_runs"
    object_detector_provider: str = "none"
    object_detector_api_key: str | None = None
    object_detector_model: str = "adirik/grounding-dino"
    object_detector_model_version: str | None = None
    object_detector_box_threshold: float = 0.25
    object_detector_text_threshold: float = 0.25
    object_detector_min_confidence: float = 0.65
    object_detector_timeout_s: float = 90.0
    object_detector_max_image_edge_px: int = 1280
    object_detector_jpeg_quality: int = 85
    object_focus_horizontal_fov_deg: float = 65.0
    object_focus_center_tolerance_norm: float = 0.08
    object_focus_max_attempts: int = 3
    object_approach_target_min_m: float = 0.35
    object_approach_target_max_m: float = 0.45
    object_approach_target_tolerance_m: float = 0.025
    object_approach_step_m: float = 0.25
    object_approach_step_fraction: float = 0.8
    object_approach_max_attempts: int = 20
    object_approach_robot_width_m: float = 0.459
    object_approach_clearance_m: float = 0.06


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
        run_id: str | None = None,
    ) -> None:
        self.memory = memory or {}
        self.config = config
        self.emit = emit
        self.run_id = run_id or f"manual_{int(time.time())}"
        self.current_pose = self._initial_pose()
        self.stopped = False
        self.detection_tracking: dict[str, dict[str, Any]] = {}
        self.selected_detection_id: str | None = None
        self.object_approach_state: dict[str, dict[str, Any]] = {}

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
        waypoint = result.get("next_waypoint") if isinstance(result.get("next_waypoint"), dict) else None
        if waypoint is not None and isinstance(self.current_pose, dict):
            result["current_pose"] = _json_pose(self.current_pose)
            result["next_waypoint_bearing_error_deg"] = _bearing_error_deg(self.current_pose, waypoint)
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

    def plan_region_exploration(
        self,
        *,
        region_label: str,
        constraints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        constraints = constraints or {}
        result = plan_home_region_exploration(
            self.memory,
            region_label,
            fov_deg=constraints.get("fov_deg"),
            max_stops=constraints.get("max_stops"),
            shots_per_stop=constraints.get("shots_per_stop"),
            min_clearance_m=float(
                constraints.get("min_clearance_m", DEFAULT_NAVIGATION_CLEARANCE_M)
                or DEFAULT_NAVIGATION_CLEARANCE_M
            ),
            boundary_margin_m=float(
                constraints.get("boundary_margin_m", 0.65)
                or 0.65
            ),
            min_stop_separation_m=constraints.get("min_stop_separation_m"),
        )
        self.emit(
            "tool_executed" if result.get("status") == "succeeded" else "tool_blocked",
            "Region Exploration Plan",
            (
                f"Planned visual exploration for `{region_label}`."
                if result.get("status") == "succeeded"
                else f"Could not plan visual exploration for `{region_label}`."
            ),
            result,
        )
        return result

    def execute_region_exploration_plan(
        self,
        *,
        region_label: str,
        object_label: str = "",
        constraints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        constraints = constraints or {}
        plan = self.plan_region_exploration(region_label=region_label, constraints=constraints)
        if plan.get("status") != "succeeded":
            result = {
                "tool": "execute_region_exploration_plan",
                "status": "blocked",
                "region_label": region_label,
                "object_label": object_label,
                "reason": plan.get("reason") or "Region exploration plan failed.",
                "plan": plan,
            }
            self.emit("tool_blocked", "Region Exploration", result["reason"], result)
            return result
        if not self.config.exploration_backend_url:
            result = {
                "tool": "execute_region_exploration_plan",
                "status": "unavailable",
                "region_label": region_label,
                "object_label": object_label,
                "reason": "No exploration backend URL is configured for region exploration execution.",
                "plan": plan,
            }
            self.emit("tool_blocked", "Region Exploration", result["reason"], result)
            return result

        navigation_constraints = _region_exploration_navigation_constraints(constraints)
        shot_yaw_tolerance_deg = _bounded_float(
            constraints.get("shot_yaw_tolerance_deg", 5.0),
            5.0,
            minimum=0.0,
            maximum=45.0,
        )
        executed_stops: list[dict[str, Any]] = []
        captured_shot_count = 0
        saved_rgb_count = 0
        selected_detection: dict[str, Any] | None = None
        for stop_index, stop in enumerate(plan.get("stops", []), start=1):
            if not isinstance(stop, dict) or not isinstance(stop.get("pose"), dict):
                continue
            stop_id = str(stop.get("stop_id") or f"{region_label}_stop_{stop_index}")
            pose = _json_pose(stop["pose"])
            navigation = self.navigate_to_waypoint(
                waypoint_id=stop_id,
                x=pose["x"],
                y=pose["y"],
                yaw=pose["yaw"],
                constraints=navigation_constraints,
            )
            stop_execution: dict[str, Any] = {
                "stop_id": stop_id,
                "pose": pose,
                "navigation": navigation,
                "shots": [],
            }
            executed_stops.append(stop_execution)
            if navigation.get("status") != "succeeded":
                result = {
                    "tool": "execute_region_exploration_plan",
                    "status": "blocked",
                    "region_label": region_label,
                    "object_label": object_label,
                    "reason": f"Navigation to exploration stop `{stop_id}` returned `{navigation.get('status')}`.",
                    "plan": plan,
                    "stops": executed_stops,
                    "visited_stop_count": len(executed_stops) - 1,
                    "captured_shot_count": captured_shot_count,
                    "saved_rgb_count": saved_rgb_count,
                    "detection_status": "not_configured",
                }
                self.emit("tool_blocked", "Region Exploration", result["reason"], result)
                return result

            for shot_index, shot in enumerate(stop.get("shots", []), start=1):
                if not isinstance(shot, dict):
                    continue
                shot_id = str(shot.get("shot_id") or f"{stop_id}_shot_{shot_index}")
                alignment = self._align_to_region_exploration_shot(
                    stop_id=stop_id,
                    shot=shot,
                    tolerance_deg=shot_yaw_tolerance_deg,
                )
                shot_execution = {
                    "shot_id": shot_id,
                    "yaw": shot.get("yaw"),
                    "yaw_deg": shot.get("yaw_deg"),
                    "alignment": alignment,
                    "capture": {
                        "status": "skipped",
                        "reason": "Shot capture waits until yaw alignment succeeds.",
                    },
                    "detection": {
                        "status": "not_configured",
                        "object_label": object_label,
                    },
                }
                stop_execution["shots"].append(shot_execution)
                if alignment.get("status") not in {"succeeded", "partial", "skipped"}:
                    result = {
                        "tool": "execute_region_exploration_plan",
                        "status": "blocked",
                        "region_label": region_label,
                        "object_label": object_label,
                        "reason": (
                            f"Could not align to shot `{shot_execution['shot_id']}`: "
                            f"{alignment.get('reason') or alignment.get('status')}"
                        ),
                        "plan": plan,
                        "stops": executed_stops,
                        "visited_stop_count": len(executed_stops),
                        "captured_shot_count": captured_shot_count,
                        "saved_rgb_count": saved_rgb_count,
                        "detection_status": "not_configured",
                    }
                    self.emit("tool_blocked", "Region Exploration", result["reason"], result)
                    return result
                capture = self._capture_rgb_region_exploration_shot(
                    region_label=region_label,
                    object_label=object_label,
                    stop_id=stop_id,
                    stop_pose=pose,
                    shot=shot,
                )
                shot_execution["capture"] = capture
                if capture.get("status") == "succeeded":
                    saved_rgb_count += 1
                    detection = self._detect_object_in_capture(
                        object_label=object_label,
                        shot_id=shot_id,
                        capture=capture,
                    )
                    shot_execution["detection"] = detection
                    if detection.get("status") == "matched":
                        selected_detection = detection.get("selected_detection") if isinstance(detection.get("selected_detection"), dict) else None
                        result = {
                            "tool": "execute_region_exploration_plan",
                            "status": "object_found",
                            "region_label": region_label,
                            "object_label": object_label,
                            "reason": f"Detected `{object_label}` during shot `{shot_id}`; remaining region exploration was aborted.",
                            "plan": plan,
                            "stops": executed_stops,
                            "visited_stop_count": len(executed_stops),
                            "captured_shot_count": captured_shot_count + 1,
                            "saved_rgb_count": saved_rgb_count,
                            "detection_status": "matched",
                            "selected_detection": selected_detection,
                            "selected_detection_id": detection.get("selected_detection_id"),
                            "detection": detection,
                        }
                        self.emit(
                            "tool_executed",
                            "Region Exploration",
                            f"Detected `{object_label}` during region exploration; stopped the remaining shots.",
                            result,
                        )
                        return result
                    if detection.get("status") in {"failed", "unavailable"}:
                        result = {
                            "tool": "execute_region_exploration_plan",
                            "status": "blocked",
                            "region_label": region_label,
                            "object_label": object_label,
                            "reason": detection.get("reason") or f"Object detection returned `{detection.get('status')}`.",
                            "plan": plan,
                            "stops": executed_stops,
                            "visited_stop_count": len(executed_stops),
                            "captured_shot_count": captured_shot_count + 1,
                            "saved_rgb_count": saved_rgb_count,
                            "detection_status": str(detection.get("status") or "failed"),
                            "detection": detection,
                        }
                        self.emit("tool_blocked", "Region Exploration", result["reason"], result)
                        return result
                captured_shot_count += 1

        detection_status = _aggregate_detection_status(executed_stops, object_label)
        result = {
            "tool": "execute_region_exploration_plan",
            "status": "succeeded",
            "region_label": region_label,
            "object_label": object_label,
            "plan": plan,
            "stops": executed_stops,
            "visited_stop_count": len(executed_stops),
            "captured_shot_count": captured_shot_count,
            "saved_rgb_count": saved_rgb_count,
            "detection_status": detection_status,
            "selected_detection": selected_detection,
            "reason": _region_exploration_result_reason(detection_status),
        }
        self.emit(
            "tool_executed",
            "Region Exploration",
            (
                f"Executed region exploration for `{region_label}` with "
                f"{len(executed_stops)} stops and {captured_shot_count} shot alignments."
            ),
            result,
        )
        return result

    def _detect_object_in_capture(
        self,
        *,
        object_label: str,
        shot_id: str,
        capture: dict[str, Any],
    ) -> dict[str, Any]:
        if not object_label.strip():
            return {
                "status": "skipped",
                "object_label": object_label,
                "shot_id": shot_id,
                "reason": "No object label was requested for this shot.",
            }
        image_path = capture.get("image_path")
        if not isinstance(image_path, str) or not image_path:
            return {
                "status": "failed",
                "object_label": object_label,
                "shot_id": shot_id,
                "reason": "Capture succeeded but did not include an image path for detection.",
            }
        try:
            data_url = _image_file_to_data_url(
                Path(image_path),
                str(capture.get("mime_type") or "image/png"),
            )
        except Exception as exc:
            return {
                "status": "failed",
                "object_label": object_label,
                "shot_id": shot_id,
                "reason": f"Could not prepare captured image for object detection: {exc}",
            }
        detection = detect_object_in_image(
            config=_object_detector_config(self.config),
            image_data_url=data_url,
            object_label=object_label,
            shot_id=shot_id,
            image_path=image_path,
        )
        if detection.get("status") == "matched":
            self._track_detection_result(
                object_label=object_label,
                detection=detection,
                capture=capture,
            )
        _record_capture_detection(config=self.config, run_id=self.run_id, capture=capture, detection=detection)
        return detection

    def _align_to_region_exploration_shot(
        self,
        *,
        stop_id: str,
        shot: dict[str, Any],
        tolerance_deg: float,
    ) -> dict[str, Any]:
        target_yaw = _bounded_float(shot.get("yaw"), float(self.current_pose.get("yaw", 0.0) or 0.0), minimum=-math.pi, maximum=math.pi)
        delta_yaw_deg = _yaw_delta_deg(float(self.current_pose.get("yaw", 0.0) or 0.0), target_yaw)
        if abs(delta_yaw_deg) <= tolerance_deg:
            return {
                "tool": "rotate_by",
                "status": "skipped",
                "reason": "Current yaw is already inside shot tolerance.",
                "stop_id": stop_id,
                "target_yaw": round(target_yaw, 3),
                "delta_yaw_deg": delta_yaw_deg,
                "tolerance_deg": tolerance_deg,
                "current_pose": _json_pose(self.current_pose),
            }
        result = self.rotate_by(
            delta_yaw_deg=delta_yaw_deg,
            reason=f"Align to region exploration shot `{shot.get('shot_id') or stop_id}`.",
        )
        result["target_yaw"] = round(target_yaw, 3)
        result["delta_yaw_deg_requested"] = delta_yaw_deg
        result["tolerance_deg"] = tolerance_deg
        return result

    def _capture_rgb_region_exploration_shot(
        self,
        *,
        region_label: str,
        object_label: str,
        stop_id: str,
        stop_pose: dict[str, Any],
        shot: dict[str, Any],
    ) -> dict[str, Any]:
        shot_id = str(shot.get("shot_id") or f"{stop_id}_shot")
        response = _post_exploration_backend(
            self.config,
            "/api/nav/capture_rgb",
            {
                "reason": "region_exploration_shot",
                "region_label": region_label,
                "object_label": object_label,
                "stop_id": stop_id,
                "shot_id": shot_id,
                "stop_pose": stop_pose,
                "shot": {
                    "yaw": shot.get("yaw"),
                    "yaw_deg": shot.get("yaw_deg"),
                    "fov_deg": shot.get("fov_deg"),
                },
            },
        )
        data_url = response.get("image_data_url")
        robot_pose = _json_pose(response["robot_pose"]) if isinstance(response.get("robot_pose"), dict) else None
        if robot_pose is not None:
            self.current_pose = robot_pose
        if not isinstance(data_url, str) or not data_url.startswith("data:image/"):
            return {
                "status": str(response.get("status") or "unavailable"),
                "reason": response.get("reason") or "Exploration backend did not return an RGB image.",
                "backend_url": self.config.exploration_backend_url,
                "captured_at": response.get("captured_at"),
                "robot_pose": robot_pose,
            }
        metadata = {
            "run_id": self.run_id,
            "region_label": region_label,
            "object_label": object_label,
            "stop_id": stop_id,
            "stop_pose": stop_pose,
            "shot_id": shot_id,
            "shot": {
                "yaw": shot.get("yaw"),
                "yaw_deg": shot.get("yaw_deg"),
                "fov_deg": shot.get("fov_deg"),
            },
            "current_pose": _json_pose(self.current_pose),
            "robot_pose": robot_pose,
            "captured_at": response.get("captured_at") or time.time(),
            "backend_url": self.config.exploration_backend_url,
        }
        try:
            saved = _save_rgb_capture_artifact(
                config=self.config,
                run_id=self.run_id,
                file_stem=_safe_artifact_name(f"{region_label}_{stop_id}_{shot_id}"),
                data_url=data_url,
                metadata=metadata,
            )
        except Exception as exc:
            return {
                "status": "failed",
                "reason": f"RGB image was received but could not be saved: {exc}",
                "backend_url": self.config.exploration_backend_url,
                "captured_at": metadata.get("captured_at"),
                "robot_pose": metadata.get("robot_pose"),
            }
        return {
            "status": "succeeded",
            "reason": "RGB shot saved.",
            "backend_url": self.config.exploration_backend_url,
            **saved,
        }

    def focus_detected_object(
        self,
        *,
        detection_id: str = "",
        object_label: str = "",
        constraints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        constraints = constraints or {}
        state = self._resolve_detection_state(detection_id=detection_id, object_label=object_label)
        label = object_label or str(state.get("object_label") or "")
        if not label.strip():
            result = {
                "tool": "focus_detected_object",
                "status": "blocked",
                "reason": "No object label or tracked detection is available to focus.",
            }
            self.emit("tool_blocked", "Focus Object", result["reason"], result)
            return result
        attempts: list[dict[str, Any]] = []
        tracked = self._tracked_detection_and_capture(state)
        max_attempts = int(_bounded_float(
            constraints.get("max_attempts", self.config.object_focus_max_attempts),
            self.config.object_focus_max_attempts,
            minimum=1,
            maximum=12,
        ))
        tolerance = _bounded_float(
            constraints.get("center_tolerance_norm", self.config.object_focus_center_tolerance_norm),
            self.config.object_focus_center_tolerance_norm,
            minimum=0.01,
            maximum=0.5,
        )
        if tracked is not None:
            selected, capture = tracked
            center = _detection_center_error(selected, capture)
            attempt: dict[str, Any] = {
                "attempt": 0,
                "source": "tracked_detection",
                "capture": capture,
                "selected_detection": selected,
                "center": center,
            }
            attempts.append(attempt)
            if center.get("status") != "succeeded":
                result = {
                    "tool": "focus_detected_object",
                    "status": "blocked",
                    "object_label": label,
                    "detection_id": state.get("detection_id") or detection_id,
                    "reason": center.get("reason") or "Tracked detection bbox could not be centered.",
                    "attempts": attempts,
                }
                self.emit("tool_blocked", "Focus Object", result["reason"], result)
                return result
            if abs(float(center["error_norm"])) <= tolerance:
                result = {
                    "tool": "focus_detected_object",
                    "status": "succeeded",
                    "object_label": label,
                    "detection_id": state.get("detection_id") or selected.get("detection_id") or detection_id,
                    "selected_detection": selected,
                    "center_error_norm": center["error_norm"],
                    "attempt_count": 0,
                    "attempts": attempts,
                    "detector_refreshed": False,
                    "reason": "Tracked object bbox is already centered; detector was not re-run.",
                }
                self.emit("tool_executed", "Focus Object", result["reason"], result)
                return result
            rotation = self.rotate_by(
                delta_yaw_deg=_visual_servo_yaw_step_deg(center, constraints, self.config),
                reason=f"Center `{label}` from tracked bbox before approach.",
            )
            attempt["rotation"] = rotation
            if rotation.get("status") not in {"succeeded", "partial"}:
                result = {
                    "tool": "focus_detected_object",
                    "status": "blocked",
                    "object_label": label,
                    "detection_id": state.get("detection_id") or selected.get("detection_id") or detection_id,
                    "reason": rotation.get("reason") or "Focus rotation failed.",
                    "attempts": attempts,
                }
                self.emit("tool_blocked", "Focus Object", result["reason"], result)
                return result
            predicted = _horizontally_centered_detection(selected, capture)
            if predicted is not None:
                selected = predicted
                self._update_tracked_detection(str(state.get("detection_id") or detection_id or ""), selected)
                attempt["predicted_selected_detection"] = selected
            result = {
                "tool": "focus_detected_object",
                "status": "succeeded",
                "object_label": label,
                "detection_id": state.get("detection_id") or selected.get("detection_id") or detection_id,
                "selected_detection": selected,
                "center_error_norm_before": center["error_norm"],
                "attempt_count": 0,
                "attempts": attempts,
                "detector_refreshed": False,
                "reason": "Applied one tracked-bbox focus rotation; detector was not re-run.",
            }
            self.emit("tool_executed", "Focus Object", result["reason"], result)
            return result

        for attempt_index in range(1, max_attempts + 1):
            capture = self._capture_rgb_for_object_tool(
                tool_name="focus_detected_object",
                object_label=label,
                attempt_index=attempt_index,
            )
            if capture.get("status") != "succeeded":
                result = {
                    "tool": "focus_detected_object",
                    "status": "blocked",
                    "object_label": label,
                    "reason": capture.get("reason") or "RGB capture failed while focusing the object.",
                    "attempts": attempts,
                    "capture": capture,
                }
                self.emit("tool_blocked", "Focus Object", result["reason"], result)
                return result
            detection = self._detect_object_in_capture(
                object_label=label,
                shot_id=str(capture.get("shot_id") or f"focus_{attempt_index}"),
                capture=capture,
            )
            attempt: dict[str, Any] = {
                "attempt": attempt_index,
                "capture": capture,
                "detection": detection,
            }
            attempts.append(attempt)
            if detection.get("status") != "matched":
                result = {
                    "tool": "focus_detected_object",
                    "status": "object_lost",
                    "object_label": label,
                    "reason": detection.get("reason") or "Object was not detected during focus.",
                    "attempts": attempts,
                }
                self.emit("tool_blocked", "Focus Object", result["reason"], result)
                return result
            selected = detection.get("selected_detection") if isinstance(detection.get("selected_detection"), dict) else {}
            center = _detection_center_error(selected, capture)
            attempt["center"] = center
            if center.get("status") != "succeeded":
                result = {
                    "tool": "focus_detected_object",
                    "status": "blocked",
                    "object_label": label,
                    "reason": center.get("reason") or "Detection bbox could not be centered.",
                    "attempts": attempts,
                }
                self.emit("tool_blocked", "Focus Object", result["reason"], result)
                return result
            if abs(float(center["error_norm"])) <= tolerance:
                state = self._resolve_detection_state(
                    detection_id=str(detection.get("selected_detection_id") or ""),
                    object_label=label,
                )
                result = {
                    "tool": "focus_detected_object",
                    "status": "succeeded",
                    "object_label": label,
                    "detection_id": state.get("detection_id") or detection.get("selected_detection_id"),
                    "selected_detection": selected,
                    "center_error_norm": center["error_norm"],
                    "attempt_count": attempt_index,
                    "attempts": attempts,
                    "reason": "Object is centered in the camera view.",
                }
                self.emit("tool_executed", "Focus Object", result["reason"], result)
                return result
            rotation = self.rotate_by(
                delta_yaw_deg=_visual_servo_yaw_step_deg(center, constraints, self.config),
                reason=f"Center `{label}` in the camera before approach.",
            )
            attempt["rotation"] = rotation
            if rotation.get("status") not in {"succeeded", "partial"}:
                result = {
                    "tool": "focus_detected_object",
                    "status": "blocked",
                    "object_label": label,
                    "reason": rotation.get("reason") or "Focus rotation failed.",
                    "attempts": attempts,
                }
                self.emit("tool_blocked", "Focus Object", result["reason"], result)
                return result
        result = {
            "tool": "focus_detected_object",
            "status": "partial",
            "object_label": label,
            "reason": "Focus attempts ended before the detection reached the center tolerance.",
            "attempts": attempts,
        }
        self.emit("tool_blocked", "Focus Object", result["reason"], result)
        return result

    def approach_detected_object(
        self,
        *,
        detection_id: str = "",
        object_label: str = "",
        constraints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        constraints = constraints or {}
        state = self._resolve_detection_state(detection_id=detection_id, object_label=object_label)
        label = object_label or str(state.get("object_label") or "")
        if not label.strip():
            result = {
                "tool": "approach_detected_object",
                "status": "blocked",
                "reason": "No object label or tracked detection is available to approach.",
            }
            self.emit("tool_blocked", "Approach Object", result["reason"], result)
            return result
        target_min_m = _bounded_float(
            constraints.get("target_min_m", self.config.object_approach_target_min_m),
            self.config.object_approach_target_min_m,
            minimum=0.1,
            maximum=2.0,
        )
        target_max_m = _bounded_float(
            constraints.get("target_max_m", self.config.object_approach_target_max_m),
            self.config.object_approach_target_max_m,
            minimum=max(target_min_m, 0.1),
            maximum=2.5,
        )
        target_tolerance_m = _bounded_float(
            constraints.get("target_tolerance_m", self.config.object_approach_target_tolerance_m),
            self.config.object_approach_target_tolerance_m,
            minimum=0.0,
            maximum=0.1,
        )
        max_step_m = _bounded_float(
            constraints.get("step_m", self.config.object_approach_step_m),
            self.config.object_approach_step_m,
            minimum=0.02,
            maximum=0.35,
        )
        step_fraction = _bounded_float(
            constraints.get("step_fraction", self.config.object_approach_step_fraction),
            self.config.object_approach_step_fraction,
            minimum=0.1,
            maximum=1.0,
        )
        max_attempts = int(_bounded_float(
            constraints.get("max_attempts", self.config.object_approach_max_attempts),
            self.config.object_approach_max_attempts,
            minimum=1,
            maximum=30,
        ))
        redetect_after_motion_steps = int(_bounded_float(
            constraints.get("redetect_after_motion_steps", 1),
            1,
            minimum=1,
            maximum=6,
        ))
        approach_key = _object_approach_key(label)
        approach_state = self.object_approach_state.get(approach_key, {})
        surface_alignment_already_attempted = bool(approach_state.get("surface_alignment_attempted"))
        allow_surface_alignment = _constraint_bool(
            constraints,
            "allow_surface_alignment",
            not surface_alignment_already_attempted,
        )
        retry_after_surface_alignment = surface_alignment_already_attempted and not allow_surface_alignment
        force_detector_refresh_first = (
            surface_alignment_already_attempted
            and _constraint_bool(constraints, "refresh_detector_on_retry", True)
        )
        attempts: list[dict[str, Any]] = []
        motion_steps_since_detection = redetect_after_motion_steps if force_detector_refresh_first else 0
        refresh_after_geometry_failure = False
        last_state = state
        surface_alignment_attempted = False
        retry_policy = None
        if surface_alignment_already_attempted:
            retry_policy = {
                "surface_alignment_disabled": not allow_surface_alignment,
                "detector_refresh_forced": force_detector_refresh_first,
                "reason": "Surface alignment was already attempted for this object in this run.",
            }
        for attempt_index in range(1, max_attempts + 1):
            tracked = self._tracked_detection_and_capture(last_state)
            use_tracked_detection = (
                tracked is not None
                and motion_steps_since_detection < redetect_after_motion_steps
                and not refresh_after_geometry_failure
            )
            if use_tracked_detection and tracked is not None:
                selected, capture = tracked
                detection = dict(last_state.get("detection")) if isinstance(last_state.get("detection"), dict) else {}
                detection.update(
                    {
                        "status": "matched",
                        "object_label": label,
                        "selected_detection": selected,
                        "selected_detection_id": last_state.get("detection_id")
                        or selected.get("tracking_id")
                        or selected.get("detection_id"),
                    }
                )
                source = "tracked_detection"
            else:
                capture = self._capture_rgb_for_object_tool(
                    tool_name="approach_detected_object",
                    object_label=label,
                    attempt_index=attempt_index,
                )
                if capture.get("status") != "succeeded":
                    result = {
                        "tool": "approach_detected_object",
                        "status": "blocked",
                        "object_label": label,
                        "reason": capture.get("reason") or "RGB capture failed during object approach.",
                        "attempts": attempts,
                        "capture": capture,
                    }
                    self.emit("tool_blocked", "Approach Object", result["reason"], result)
                    return result
                detection = self._detect_object_in_capture(
                    object_label=label,
                    shot_id=str(capture.get("shot_id") or f"approach_{attempt_index}"),
                    capture=capture,
                )
                source = "detector_refresh"
                refresh_after_geometry_failure = False
                motion_steps_since_detection = 0
                if detection.get("status") == "matched":
                    last_state = self._resolve_detection_state(
                        detection_id=str(detection.get("selected_detection_id") or ""),
                        object_label=label,
                    )
            attempt: dict[str, Any] = {
                "attempt": attempt_index,
                "source": source,
                "motion_steps_since_detection": motion_steps_since_detection,
                "capture": capture,
                "detection": detection,
            }
            if retry_policy and attempt_index == 1:
                attempt["retry_policy"] = retry_policy
            attempts.append(attempt)
            if detection.get("status") != "matched":
                result = {
                    "tool": "approach_detected_object",
                    "status": "object_lost",
                    "object_label": label,
                    "reason": detection.get("reason") or "Object was lost during approach.",
                    "attempts": attempts,
                }
                self.emit("tool_blocked", "Approach Object", result["reason"], result)
                return result
            selected = detection.get("selected_detection") if isinstance(detection.get("selected_detection"), dict) else {}
            geometry = self._estimate_detection_geometry(
                detection=selected,
                capture=capture,
                constraints={
                    **constraints,
                    "target_max_m": target_max_m,
                    "max_step_m": max_step_m,
                },
            )
            attempt["geometry"] = geometry
            if geometry.get("status") != "succeeded":
                if source == "tracked_detection" and not refresh_after_geometry_failure:
                    refresh_after_geometry_failure = True
                    attempt["next_action"] = "refresh_detector_due_to_geometry_failure"
                    continue
                result = {
                    "tool": "approach_detected_object",
                    "status": "blocked",
                    "object_label": label,
                    "reason": geometry.get("reason") or "Could not solve detection depth.",
                    "attempts": attempts,
                }
                self.emit("tool_blocked", "Approach Object", result["reason"], result)
                return result
            forward_m = float(geometry.get("forward_m", geometry.get("distance_m", 999.0)) or 999.0)
            bearing_error_deg = float(geometry.get("bearing_error_deg", 0.0) or 0.0)
            if allow_surface_alignment and not surface_alignment_attempted:
                surface_alignment_attempted = True
                alignment = self._maybe_align_to_object_surface(
                    object_label=label,
                    geometry=geometry,
                    constraints=constraints,
                    target_max_m=target_max_m,
                )
                attempt["surface_alignment"] = alignment
                if alignment.get("status") in {"succeeded", "partial", "blocked"}:
                    self._record_object_surface_alignment(
                        object_label=label,
                        detection_id=str(detection.get("selected_detection_id") or ""),
                        alignment=alignment,
                    )
                if alignment.get("status") in {"succeeded", "partial"} and alignment.get("motion"):
                    if _constraint_bool(constraints, "relocalize_after_surface_alignment", True):
                        attempt["surface_alignment_relocalization"] = self.relocalize_here()
                    motion_steps_since_detection = redetect_after_motion_steps
                    refresh_after_geometry_failure = False
                    continue
                if alignment.get("status") == "blocked":
                    result = {
                        "tool": "approach_detected_object",
                        "status": "blocked",
                        "object_label": label,
                        "geometry": geometry,
                        "attempt_count": attempt_index,
                        "attempts": attempts,
                        "reason": alignment.get("reason") or "Surface approach alignment failed.",
                    }
                    self.emit("tool_blocked", "Approach Object", result["reason"], result)
                    return result
            center = _detection_center_error(selected, capture)
            attempt["image_center"] = center
            center_tolerance = _bounded_float(
                constraints.get(
                    "approach_center_tolerance_norm",
                    constraints.get("center_tolerance_norm", self.config.object_focus_center_tolerance_norm),
                ),
                self.config.object_focus_center_tolerance_norm,
                minimum=0.01,
                maximum=0.5,
            )
            should_image_center = (
                center.get("status") == "succeeded"
                and abs(float(center["error_norm"])) > center_tolerance
            )
            if target_min_m <= forward_m <= target_max_m + target_tolerance_m:
                if forward_m <= target_max_m:
                    reason = f"Object is within grasp staging range ({forward_m:.2f} m)."
                else:
                    reason = (
                        f"Object is within grasp staging tolerance "
                        f"({forward_m:.3f} m, target max {target_max_m:.3f} m)."
                    )
                result = {
                    "tool": "approach_detected_object",
                    "status": "succeeded",
                    "object_label": label,
                    "detection_id": detection.get("selected_detection_id"),
                    "selected_detection": selected,
                    "geometry": geometry,
                    "target_tolerance_m": target_tolerance_m,
                    "attempt_count": attempt_index,
                    "attempts": attempts,
                    "reason": reason,
                }
                self._track_detection_result(object_label=label, detection=detection, capture=capture, geometry=geometry)
                self.emit("tool_executed", "Approach Object", result["reason"], result)
                return result
            if forward_m <= 0.0:
                attempt["geometry_consistency"] = {
                    "status": "invalid_forward",
                    "forward_m": round(forward_m, 3),
                    "reason": (
                        "RGB-D geometry placed an image-visible detection behind the robot base frame; "
                        "recenter or refresh before using this range estimate."
                    ),
                }
            if should_image_center:
                rotation = self.rotate_by(
                    delta_yaw_deg=_visual_servo_yaw_step_deg(center, constraints, self.config),
                    reason=f"Center `{label}` from image bbox before forward approach.",
                )
                attempt["rotation"] = rotation
                attempt["image_centering"] = {
                    "center_error_norm": center["error_norm"],
                    "tolerance_norm": round(center_tolerance, 3),
                    "rotation_source": "image_bbox",
                }
                if rotation.get("status") not in {"succeeded", "partial"}:
                    result = {
                        "tool": "approach_detected_object",
                        "status": "blocked",
                        "object_label": label,
                        "reason": rotation.get("reason") or "Approach image-centering rotation failed.",
                        "attempts": attempts,
                    }
                    self.emit("tool_blocked", "Approach Object", result["reason"], result)
                    return result
                motion_steps_since_detection = redetect_after_motion_steps
                attempt["next_action"] = "refresh_detector_after_image_centering_rotation"
                continue
            if forward_m <= 0.0:
                result = {
                    "tool": "approach_detected_object",
                    "status": "blocked",
                    "object_label": label,
                    "geometry": geometry,
                    "attempt_count": attempt_index,
                    "attempts": attempts,
                    "reason": (
                        "RGB-D geometry placed the detected object behind the robot base frame; "
                        "refresh detection or verify the head-camera transform before approaching."
                    ),
                }
                self.emit("tool_blocked", "Approach Object", result["reason"], result)
                return result
            if forward_m < target_min_m:
                result = {
                    "tool": "approach_detected_object",
                    "status": "too_close",
                    "object_label": label,
                    "detection_id": detection.get("selected_detection_id"),
                    "selected_detection": selected,
                    "geometry": geometry,
                    "attempt_count": attempt_index,
                    "attempts": attempts,
                    "reason": f"Object is closer than the minimum grasp staging distance ({forward_m:.2f} m).",
                }
                self.emit("tool_blocked", "Approach Object", result["reason"], result)
                return result
            bearing_tolerance_deg = _bounded_float(
                constraints.get("bearing_tolerance_deg", 6.0),
                6.0,
                minimum=1.0,
                maximum=30.0,
            )
            use_geometry_bearing_correction = _constraint_bool(
                constraints,
                "use_geometry_bearing_correction",
                center.get("status") != "succeeded",
            )
            if use_geometry_bearing_correction and abs(bearing_error_deg) > bearing_tolerance_deg:
                rotation = self.rotate_by(
                    delta_yaw_deg=max(min(bearing_error_deg, 12.0), -12.0),
                    reason=f"Center `{label}` from RGB-D bearing before forward approach.",
                )
                attempt["rotation"] = rotation
                attempt["geometry_bearing_correction"] = {
                    "bearing_error_deg": round(bearing_error_deg, 3),
                    "tolerance_deg": round(bearing_tolerance_deg, 3),
                    "rotation_source": "rgbd_geometry",
                }
                if rotation.get("status") not in {"succeeded", "partial"}:
                    result = {
                        "tool": "approach_detected_object",
                        "status": "blocked",
                        "object_label": label,
                        "reason": rotation.get("reason") or "Approach bearing rotation failed.",
                        "attempts": attempts,
                    }
                    self.emit("tool_blocked", "Approach Object", result["reason"], result)
                    return result
                motion_steps_since_detection = redetect_after_motion_steps
                attempt["next_action"] = "refresh_detector_after_geometry_bearing_rotation"
                continue
            safety = geometry.get("safety") if isinstance(geometry.get("safety"), dict) else {}
            if not bool(safety.get("safe", False)):
                result = {
                    "tool": "approach_detected_object",
                    "status": "blocked",
                    "object_label": label,
                    "geometry": geometry,
                    "attempt_count": attempt_index,
                    "attempts": attempts,
                    "reason": safety.get("reason") or "RGB-D corridor safety blocked the forward approach step.",
                }
                self.emit("tool_blocked", "Approach Object", result["reason"], result)
                return result
            remaining_to_staging_m = max(forward_m - target_max_m, 0.0)
            desired_forward_step_m = remaining_to_staging_m * step_fraction
            safe_forward_step_m = float(safety.get("safe_forward_step_m", max_step_m) or 0.0)
            forward_step = min(safe_forward_step_m, max_step_m, desired_forward_step_m)
            attempt["approach_step"] = {
                "remaining_to_staging_m": round(remaining_to_staging_m, 3),
                "step_fraction": round(step_fraction, 3),
                "desired_forward_step_m": round(desired_forward_step_m, 3),
                "safe_forward_step_m": round(safe_forward_step_m, 3),
                "max_step_m": round(max_step_m, 3),
                "chosen_forward_step_m": round(forward_step, 3),
            }
            if forward_step <= 0.01:
                result = {
                    "tool": "approach_detected_object",
                    "status": "partial",
                    "object_label": label,
                    "geometry": geometry,
                    "attempt_count": attempt_index,
                    "attempts": attempts,
                    "reason": "Object is just outside range, but the safe forward step is too small.",
                }
                self.emit("tool_blocked", "Approach Object", result["reason"], result)
                return result
            target_pose = _pose_forward_step(self.current_pose, forward_step)
            motion = self.micro_adjust_to_pose(
                x=target_pose["x"],
                y=target_pose["y"],
                yaw=target_pose["yaw"],
                max_distance_m=max(forward_step + 0.04, 0.08),
                reason=f"Closed-loop visual approach toward `{label}` by {forward_step:.2f} m.",
            )
            attempt["motion"] = motion
            if motion.get("status") not in {"succeeded", "partial"}:
                result = {
                    "tool": "approach_detected_object",
                    "status": "blocked",
                    "object_label": label,
                    "geometry": geometry,
                    "attempt_count": attempt_index,
                    "attempts": attempts,
                    "reason": motion.get("reason") or "Forward visual-servo approach step failed.",
                }
                self.emit("tool_blocked", "Approach Object", result["reason"], result)
                return result
            motion_steps_since_detection += 1
            if _constraint_bool(constraints, "refresh_detector_after_forward_motion", True):
                motion_steps_since_detection = redetect_after_motion_steps
                attempt["next_action"] = "refresh_detector_after_forward_motion"
        result = {
            "tool": "approach_detected_object",
            "status": "partial",
            "object_label": label,
            "reason": "Approach loop reached max attempts before grasp staging range.",
            "attempts": attempts,
        }
        self.emit("tool_blocked", "Approach Object", result["reason"], result)
        return result

    def _record_object_surface_alignment(
        self,
        *,
        object_label: str,
        detection_id: str,
        alignment: dict[str, Any],
    ) -> None:
        key = _object_approach_key(object_label)
        if not key:
            return
        self.object_approach_state[key] = {
            **self.object_approach_state.get(key, {}),
            "object_label": object_label,
            "surface_alignment_attempted": True,
            "surface_alignment_had_motion": bool(alignment.get("motion")),
            "surface_alignment_status": alignment.get("status"),
            "surface_alignment_reason": alignment.get("reason"),
            "detection_id": detection_id,
            "updated_at": time.time(),
        }

    def _maybe_align_to_object_surface(
        self,
        *,
        object_label: str,
        geometry: dict[str, Any],
        constraints: dict[str, Any],
        target_max_m: float,
    ) -> dict[str, Any]:
        object_pose = _object_map_pose_from_geometry(geometry, self.current_pose)
        if object_pose is None:
            return {
                "tool": "resolve_object_surface_approach_pose",
                "status": "skipped",
                "reason": "Object map pose is unavailable for surface approach alignment.",
            }
        min_clearance_m = _bounded_float(
            constraints.get("surface_alignment_min_clearance_m", DEFAULT_NAVIGATION_CLEARANCE_M),
            DEFAULT_NAVIGATION_CLEARANCE_M,
            minimum=0.0,
            maximum=1.5,
        )
        standoff_m = _bounded_float(
            constraints.get("surface_alignment_standoff_m", max(target_max_m + 0.20, 0.62)),
            max(target_max_m + 0.20, 0.62),
            minimum=target_max_m,
            maximum=1.5,
        )
        max_distance_m = _bounded_float(
            constraints.get("surface_alignment_max_distance_m", 2.0),
            2.0,
            minimum=0.05,
            maximum=3.0,
        )
        alignment = resolve_object_surface_approach_pose(
            self.memory,
            self.current_pose,
            object_pose,
            min_clearance_m=min_clearance_m,
            standoff_m=standoff_m,
            search_beyond_m=_bounded_float(
                constraints.get("surface_alignment_search_beyond_m", 0.9),
                0.9,
                minimum=0.0,
                maximum=2.5,
            ),
            support_radius_m=_bounded_float(
                constraints.get("surface_alignment_support_radius_m", 0.75),
                0.75,
                minimum=0.2,
                maximum=2.0,
            ),
            max_alignment_distance_m=max_distance_m,
            yaw_tolerance_deg=_bounded_float(
                constraints.get("surface_alignment_yaw_tolerance_deg", 18.0),
                18.0,
                minimum=1.0,
                maximum=90.0,
            ),
            distance_tolerance_m=_bounded_float(
                constraints.get("surface_alignment_distance_tolerance_m", 0.12),
                0.12,
                minimum=0.0,
                maximum=0.5,
            ),
        )
        if alignment.get("status") != "succeeded" or not bool(alignment.get("needs_alignment", False)):
            return alignment
        approach_pose = alignment.get("approach_pose") if isinstance(alignment.get("approach_pose"), dict) else None
        if not approach_pose:
            return {**alignment, "status": "blocked", "reason": "Surface alignment did not return an approach pose."}
        motion = self.micro_adjust_to_pose(
            x=float(approach_pose["x"]),
            y=float(approach_pose["y"]),
            yaw=float(approach_pose.get("yaw", 0.0) or 0.0),
            max_distance_m=max(float(alignment.get("distance_m", max_distance_m) or max_distance_m) + 0.08, 0.12),
            reason=f"Align robot body perpendicular to `{object_label}` support surface before close approach.",
        )
        return {
            **alignment,
            "motion": motion,
            "status": "succeeded" if motion.get("status") in {"succeeded", "partial"} else "blocked",
            "reason": (
                "Aligned robot body to the occupied support surface; detection should be refreshed."
                if motion.get("status") in {"succeeded", "partial"}
                else motion.get("reason") or "Surface alignment motion failed."
            ),
        }

    def grab_object(
        self,
        *,
        object_label: str,
        detection_id: str = "",
        object_description: str = "",
        constraints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = self._resolve_detection_state(detection_id=detection_id, object_label=object_label)
        result = {
            "tool": "grab_object",
            "status": "mock_succeeded",
            "object_label": object_label,
            "detection_id": state.get("detection_id") or detection_id or self.selected_detection_id,
            "object_description": object_description,
            "constraints": constraints or {},
            "detection_state": state or None,
            "reason": "Mock grab_object completed. This is the future VLA skill entrypoint.",
        }
        self.emit("tool_executed", "Grab Object Mock", result["reason"], result)
        return result

    def _capture_rgb_for_object_tool(self, *, tool_name: str, object_label: str, attempt_index: int) -> dict[str, Any]:
        shot_id = _safe_artifact_name(f"{tool_name}_{object_label}_{attempt_index}_{int(time.time() * 1000)}")
        response = _post_exploration_backend(
            self.config,
            "/api/nav/capture_rgb",
            {
                "reason": tool_name,
                "object_label": object_label,
                "shot_id": shot_id,
                "settle_s": 0.08,
            },
        )
        data_url = response.get("image_data_url")
        robot_pose = _json_pose(response["robot_pose"]) if isinstance(response.get("robot_pose"), dict) else None
        if robot_pose is not None:
            self.current_pose = robot_pose
        if not isinstance(data_url, str) or not data_url.startswith("data:image/"):
            return {
                "status": str(response.get("status") or "unavailable"),
                "reason": response.get("reason") or "Exploration backend did not return an RGB image.",
                "backend_url": self.config.exploration_backend_url,
                "captured_at": response.get("captured_at"),
                "robot_pose": robot_pose,
                "shot_id": shot_id,
            }
        metadata = {
            "run_id": self.run_id,
            "object_label": object_label,
            "shot_id": shot_id,
            "reason": tool_name,
            "current_pose": _json_pose(self.current_pose),
            "robot_pose": robot_pose,
            "captured_at": response.get("captured_at") or time.time(),
            "backend_url": self.config.exploration_backend_url,
        }
        try:
            saved = _save_rgb_capture_artifact(
                config=self.config,
                run_id=self.run_id,
                file_stem=shot_id,
                data_url=data_url,
                metadata=metadata,
            )
        except Exception as exc:
            return {
                "status": "failed",
                "reason": f"RGB image was received but could not be saved: {exc}",
                "backend_url": self.config.exploration_backend_url,
                "captured_at": metadata.get("captured_at"),
                "robot_pose": metadata.get("robot_pose"),
                "shot_id": shot_id,
            }
        return {
            "status": "succeeded",
            "reason": "RGB object-tool shot saved.",
            "shot_id": shot_id,
            "backend_url": self.config.exploration_backend_url,
            **saved,
        }

    def _estimate_detection_geometry(
        self,
        *,
        detection: dict[str, Any],
        capture: dict[str, Any],
        constraints: dict[str, Any],
    ) -> dict[str, Any]:
        bbox = detection.get("bbox_xyxy")
        if not isinstance(bbox, list):
            return {"status": "rejected", "reason": "Detection did not include bbox_xyxy."}
        image_width = capture.get("image_width")
        image_height = capture.get("image_height")
        bbox_payload = bbox
        try:
            if image_width and image_height:
                bbox_payload = list(
                    _bbox_to_pixel_xyxy(
                        [float(item) for item in bbox[:4]],
                        image_width=float(image_width),
                        image_height=float(image_height),
                    )
                )
        except Exception:
            bbox_payload = bbox
        response = _post_exploration_backend(
            self.config,
            "/api/nav/estimate_detection_geometry",
            {
                "bbox_xyxy": bbox_payload,
                "image_width": image_width,
                "image_height": image_height,
                "target_max_m": constraints.get("target_max_m", self.config.object_approach_target_max_m),
                "max_step_m": constraints.get(
                    "max_step_m",
                    constraints.get("step_m", self.config.object_approach_step_m),
                ),
                "robot_width_m": constraints.get("robot_width_m", self.config.object_approach_robot_width_m),
                "clearance_m": constraints.get("clearance_m", self.config.object_approach_clearance_m),
                "rgbd_update_timeout_s": constraints.get("rgbd_update_timeout_s", 2.0),
                "max_depth_age_s": constraints.get("max_depth_age_s", 1.5),
                "require_depth_image": constraints.get("require_depth_image", True),
                "disable_point_cloud_fallback": constraints.get("disable_point_cloud_fallback", True),
                "bbox_sample_inner_ratio": constraints.get("bbox_sample_inner_ratio", 0.65),
                "min_valid_points": constraints.get("min_valid_points", 12),
                "object_label": detection.get("label") or capture.get("object_label"),
                "detection_id": detection.get("detection_id"),
            },
        )
        current_pose = response.get("current_pose") if isinstance(response, dict) else None
        if isinstance(current_pose, dict):
            self.current_pose = _json_pose(current_pose)
        return response if isinstance(response, dict) else {"status": "failed", "reason": "Geometry endpoint response was invalid."}

    def _tracked_detection_and_capture(self, state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]] | None:
        selected = state.get("selected_detection")
        capture = state.get("capture")
        if not isinstance(selected, dict) or not isinstance(capture, dict):
            return None
        if not isinstance(selected.get("bbox_xyxy"), list):
            return None
        if not (capture.get("image_width") and capture.get("image_height")):
            image_path = capture.get("image_path")
            if isinstance(image_path, str):
                size = _image_size_from_file(Path(image_path))
                if size is not None:
                    capture = dict(capture)
                    capture["image_width"], capture["image_height"] = size
        if not (capture.get("image_width") and capture.get("image_height")):
            return None
        return dict(selected), dict(capture)

    def _update_tracked_detection(self, detection_id: str, selected: dict[str, Any]) -> None:
        if not detection_id or detection_id not in self.detection_tracking:
            return
        state = self.detection_tracking[detection_id]
        state["selected_detection"] = dict(selected)
        state["updated_at"] = time.time()
        detection = state.get("detection")
        if isinstance(detection, dict):
            detection["selected_detection"] = dict(selected)
            detection["selected_detection_id"] = detection_id

    def _track_detection_result(
        self,
        *,
        object_label: str,
        detection: dict[str, Any],
        capture: dict[str, Any],
        geometry: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        selected = detection.get("selected_detection") if isinstance(detection.get("selected_detection"), dict) else {}
        detection_id = str(
            selected.get("tracking_id")
            or selected.get("detection_id")
            or detection.get("selected_detection_id")
            or f"det_{uuid.uuid4().hex[:10]}"
        )
        selected["tracking_id"] = detection_id
        selected["detection_id"] = str(selected.get("detection_id") or detection_id)
        detection["selected_detection"] = selected
        detection["selected_detection_id"] = detection_id
        state = {
            "detection_id": detection_id,
            "object_label": object_label,
            "updated_at": time.time(),
            "selected_detection": selected,
            "detection": detection,
            "capture": {
                key: capture.get(key)
                for key in (
                    "image_path",
                    "metadata_path",
                    "artifact_url",
                    "metadata_url",
                    "shot_id",
                    "captured_at",
                    "robot_pose",
                    "current_pose",
                    "image_width",
                    "image_height",
                )
                if capture.get(key) is not None
            },
            "geometry": geometry,
        }
        self.detection_tracking[detection_id] = state
        self.selected_detection_id = detection_id
        return state

    def _resolve_detection_state(self, *, detection_id: str = "", object_label: str = "") -> dict[str, Any]:
        if detection_id and detection_id in self.detection_tracking:
            return dict(self.detection_tracking[detection_id])
        if self.selected_detection_id and self.selected_detection_id in self.detection_tracking:
            state = self.detection_tracking[self.selected_detection_id]
            if not object_label or str(state.get("object_label") or "").lower() == object_label.lower():
                return dict(state)
        if object_label:
            for state in reversed(list(self.detection_tracking.values())):
                if str(state.get("object_label") or "").lower() == object_label.lower():
                    return dict(state)
        return {}

    def navigate_to_waypoint(
        self,
        *,
        waypoint_id: str,
        x: float,
        y: float,
        yaw: float = 0.0,
        constraints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        constraints = constraints or {}
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

        self._refresh_current_pose_from_exploration_backend()
        pre_nav_auto_rotation = self._maybe_auto_rotate_before_waypoint(
            waypoint_id=waypoint_id,
            pose=pose,
            constraints=constraints,
        )
        response = _post_exploration_backend(
            self.config,
            "/api/nav/waypoint",
            {"pose": pose, "waypoint_id": waypoint_id, "constraints": constraints},
        )
        result = _navigation_tool_result(
            response,
            waypoint_id=waypoint_id,
            requested_pose=pose,
            backend_url=self.config.exploration_backend_url,
        )
        if pre_nav_auto_rotation is not None:
            result["pre_nav_auto_rotation"] = pre_nav_auto_rotation
        self._update_current_pose_after_navigation_result(result, pose)
        if result.get("status") != "succeeded" and _constraint_bool(constraints, "allow_direct_fallback", True):
            result = self._maybe_apply_direct_waypoint_fallback(
                result,
                waypoint_id=waypoint_id,
                pose=pose,
                constraints=constraints,
            )
        self.emit(
            "tool_executed" if result.get("status") == "succeeded" else "tool_blocked",
            "Waypoint Navigation",
            (
                (
                    f"Nav2 fallback reached waypoint `{waypoint_id}`."
                    if result.get("fallback_navigation")
                    else f"Nav2 reached waypoint `{waypoint_id}`."
                )
                if result.get("status") == "succeeded"
                else f"Nav2 waypoint `{waypoint_id}` returned `{result.get('status')}`."
            ),
            result,
        )
        return result

    def _maybe_auto_rotate_before_waypoint(
        self,
        *,
        waypoint_id: str,
        pose: dict[str, Any],
        constraints: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not _constraint_bool(constraints, "allow_auto_rotate", True):
            return None
        threshold_deg = _bounded_float(
            constraints.get("auto_rotate_threshold_deg", self.config.navigation_auto_rotate_threshold_deg),
            self.config.navigation_auto_rotate_threshold_deg,
            minimum=0.0,
            maximum=181.0,
        )
        if threshold_deg > 180.0:
            return {
                "status": "skipped",
                "reason": "Auto-rotate threshold is above 180 degrees.",
                "threshold_deg": threshold_deg,
            }
        bearing_error_deg = _bearing_error_deg(self.current_pose, pose)
        if abs(bearing_error_deg) <= threshold_deg:
            return {
                "status": "skipped",
                "reason": "Bearing error is below the auto-rotate threshold.",
                "threshold_deg": threshold_deg,
                "bearing_error_deg": bearing_error_deg,
            }
        rotation = self._local_motion(
            title="Pre-Nav Auto Rotate",
            payload={
                "primitive": "rotate_towards_point",
                "x": pose["x"],
                "y": pose["y"],
                "reason": (
                    f"Auto-rotate before Nav2 waypoint `{waypoint_id}` because bearing error "
                    f"{bearing_error_deg:.1f} deg exceeds threshold {threshold_deg:.1f} deg."
                ),
            },
        )
        rotation["threshold_deg"] = threshold_deg
        rotation["bearing_error_deg"] = bearing_error_deg
        return rotation

    def _refresh_current_pose_from_exploration_backend(self) -> dict[str, float] | None:
        response = _get_exploration_backend(self.config, "/api/state")
        current_map = response.get("current_map") if isinstance(response.get("current_map"), dict) else {}
        robot_pose = current_map.get("robot_pose") if isinstance(current_map.get("robot_pose"), dict) else None
        if isinstance(robot_pose, dict):
            self.current_pose = _json_pose(robot_pose)
            return self.current_pose
        return None

    def _update_current_pose_after_navigation_result(self, result: dict[str, Any], requested_pose: dict[str, Any]) -> None:
        current_pose = result.get("current_pose")
        if isinstance(current_pose, dict):
            self.current_pose = _json_pose(current_pose)
        elif result.get("status") == "succeeded":
            self.current_pose = requested_pose
            result["current_pose"] = dict(requested_pose)

    def _maybe_apply_direct_waypoint_fallback(
        self,
        result: dict[str, Any],
        *,
        waypoint_id: str,
        pose: dict[str, Any],
        constraints: dict[str, Any],
    ) -> dict[str, Any]:
        original_status = str(result.get("status") or "failed")
        original_reason = str(result.get("reason") or "")
        start_pose = result.get("current_pose") if isinstance(result.get("current_pose"), dict) else self.current_pose
        requested_max_distance_m = float(
            constraints.get("direct_fallback_max_distance_m", DEFAULT_DIRECT_NAVIGATION_FALLBACK_MAX_DISTANCE_M)
            or DEFAULT_DIRECT_NAVIGATION_FALLBACK_MAX_DISTANCE_M
        )
        max_distance_m = min(max(requested_max_distance_m, 0.0), DEFAULT_DIRECT_NAVIGATION_FALLBACK_MAX_DISTANCE_M)
        min_clearance_m = float(
            constraints.get("direct_fallback_min_clearance_m", DEFAULT_NAVIGATION_CLEARANCE_M)
            or DEFAULT_NAVIGATION_CLEARANCE_M
        )
        fallback_plan = resolve_direct_navigation_fallback(
            self.memory,
            start_pose,
            pose,
            min_clearance_m=min_clearance_m,
            max_distance_m=max_distance_m,
        )
        result["direct_fallback_plan"] = fallback_plan
        if fallback_plan.get("status") != "succeeded":
            recovery = resolve_local_clearance_recovery(
                self.memory,
                start_pose,
                pose,
                min_clearance_m=min_clearance_m,
                max_distance_m=float(constraints.get("local_recovery_max_distance_m", 0.45) or 0.45),
            )
            result["local_clearance_recovery"] = recovery
            if recovery.get("status") == "succeeded":
                result["failure_hint"] = (
                    "Nav2 failed and direct fallback was not available. Use local_clearance_recovery: "
                    "micro_adjust_to_pose to the suggested recovery_pose, relocalize_here, then retry navigation."
                )
            result["fallback_navigation"] = {
                "status": "skipped",
                "reason": fallback_plan.get("reason"),
            }
            return result

        fallback_result = self._local_motion(
            title="Direct Waypoint Fallback",
            payload={
                "primitive": "micro_adjust_to_pose",
                "pose": pose,
                "max_distance_m": max_distance_m,
                "reason": f"Nav2 `{original_status}` for `{waypoint_id}`; direct line is footprint-clear.",
            },
        )
        result["fallback_navigation"] = fallback_result
        result["nav2_status_before_fallback"] = original_status
        result["nav2_reason_before_fallback"] = original_reason
        if fallback_result.get("status") == "succeeded":
            result["status"] = "succeeded"
            result["reason"] = "Nav2 failed, but direct footprint-clear local fallback succeeded."
            result["current_pose"] = fallback_result.get("current_pose") or pose
            if not isinstance(fallback_result.get("current_pose"), dict):
                self.current_pose = pose
            result["distance_remaining_m"] = fallback_result.get("distance_remaining_m")
            result["failure_hint"] = None
        else:
            result["reason"] = (
                f"{original_reason} Direct fallback was attempted but returned "
                f"`{fallback_result.get('status')}`: {fallback_result.get('reason') or ''}"
            ).strip()
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

    def rotate_by(self, *, delta_yaw_deg: float, reason: str = "") -> dict[str, Any]:
        return self._local_motion(
            title="Local Rotate",
            payload={
                "primitive": "rotate_by",
                "delta_yaw_deg": float(delta_yaw_deg),
                "reason": reason,
            },
        )

    def rotate_towards_point(self, *, x: float, y: float, reason: str = "") -> dict[str, Any]:
        return self._local_motion(
            title="Local Rotate Toward Point",
            payload={
                "primitive": "rotate_towards_point",
                "x": float(x),
                "y": float(y),
                "reason": reason,
            },
        )

    def micro_adjust_to_pose(
        self,
        *,
        x: float,
        y: float,
        yaw: float = 0.0,
        max_distance_m: float = 0.5,
        reason: str = "",
    ) -> dict[str, Any]:
        return self._local_motion(
            title="Local Micro Adjust",
            payload={
                "primitive": "micro_adjust_to_pose",
                "pose": _json_pose({"x": x, "y": y, "yaw": yaw}),
                "max_distance_m": float(max_distance_m),
                "reason": reason,
            },
        )

    def _local_motion(self, *, title: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.config.exploration_backend_url:
            result = {
                "tool": str(payload.get("primitive") or "local_motion"),
                "status": "unavailable",
                "reason": "No exploration backend URL is configured for local motion.",
            }
            self.emit("tool_blocked", title, result["reason"], result)
            return result
        response = _post_exploration_backend(self.config, "/api/nav/local_motion", payload)
        result = _local_motion_tool_result(
            response,
            primitive=str(payload.get("primitive") or "local_motion"),
            backend_url=self.config.exploration_backend_url,
        )
        current_pose = result.get("current_pose")
        if isinstance(current_pose, dict):
            self.current_pose = _json_pose(current_pose)
        self.emit(
            "tool_executed" if result.get("status") in {"succeeded", "partial"} else "tool_blocked",
            title,
            f"{result.get('tool')} returned `{result.get('status')}`: {result.get('reason') or ''}".strip(),
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
        runtime = HomeAgentToolRuntime(
            memory=memory,
            config=self.config,
            emit=self._recording_emit(record),
            run_id=record.run_id,
        )
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
        lower_command = command.lower()
        if any(token in lower_command for token in ("find", "search", "scan", "explore")):
            result = runtime.plan_region_exploration(region_label=target["label"], constraints={})
            if result.get("status") == "succeeded":
                record.summary = (
                    f"Planned region exploration for `{target['label']}` with "
                    f"{len(result.get('stops', []))} stops."
                )
            else:
                record.status = "blocked"
                record.summary = f"Could not plan region exploration for `{target['label']}`."
            return

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
        def plan_region_exploration(region_label: str, constraints_json: str = "{}") -> str:
            """Plan inspection stops and 65-degree visual-search shots for a semantic region."""
            return json.dumps(
                runtime.plan_region_exploration(
                    region_label=region_label,
                    constraints=_loads_object(constraints_json),
                )
            )

        @function_tool
        def execute_region_exploration_plan(
            region_label: str,
            object_label: str = "",
            constraints_json: str = "{}",
        ) -> str:
            """Navigate visual-search stops and align the robot to each planned shot cone."""
            return json.dumps(
                runtime.execute_region_exploration_plan(
                    region_label=region_label,
                    object_label=object_label,
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

        @function_tool
        def rotate_by(delta_yaw_deg: float, reason: str = "") -> str:
            """Run a bounded backend-controlled in-place rotation, in degrees."""
            return json.dumps(runtime.rotate_by(delta_yaw_deg=delta_yaw_deg, reason=reason))

        @function_tool
        def rotate_towards_point(x: float, y: float, reason: str = "") -> str:
            """Rotate the robot toward a target point before using Nav2 or as recovery."""
            return json.dumps(runtime.rotate_towards_point(x=x, y=y, reason=reason))

        @function_tool
        def micro_adjust_to_pose(x: float, y: float, yaw: float = 0.0, max_distance_m: float = 0.5, reason: str = "") -> str:
            """Use bounded backend-controlled local motion for close final pose adjustment."""
            return json.dumps(
                runtime.micro_adjust_to_pose(
                    x=x,
                    y=y,
                    yaw=yaw,
                    max_distance_m=max_distance_m,
                    reason=reason,
                )
            )

        @function_tool
        def focus_detected_object(detection_id: str = "", object_label: str = "", constraints_json: str = "{}") -> str:
            """Closed-loop visual servo: reuse the tracked box when possible and rotate until the object is centered."""
            return json.dumps(
                runtime.focus_detected_object(
                    detection_id=detection_id,
                    object_label=object_label,
                    constraints=_loads_object(constraints_json),
                )
            )

        @function_tool
        def approach_detected_object(detection_id: str = "", object_label: str = "", constraints_json: str = "{}") -> str:
            """Closed-loop RGB-D approach: align to the support surface, solve depth, then move tiny safe steps."""
            return json.dumps(
                runtime.approach_detected_object(
                    detection_id=detection_id,
                    object_label=object_label,
                    constraints=_loads_object(constraints_json),
                )
            )

        @function_tool
        def grab_object(object_label: str, detection_id: str = "", object_description: str = "", constraints_json: str = "{}") -> str:
            """Mock VLA grasp entrypoint for a focused object at grasp staging range."""
            return json.dumps(
                runtime.grab_object(
                    object_label=object_label,
                    detection_id=detection_id,
                    object_description=object_description,
                    constraints=_loads_object(constraints_json),
                )
            )

        agent = Agent(
            name="NavigationAgent",
            instructions=self._agent_instructions(memory),
            model=self._sdk_model(),
            model_settings=self._sdk_model_settings(ModelSettings),
            tools=[
                resolve_navigation_to_region,
                plan_region_exploration,
                execute_region_exploration_plan,
                navigate_to_waypoint,
                relocalize_here,
                rotate_by,
                rotate_towards_point,
                micro_adjust_to_pose,
                focus_detected_object,
                approach_detected_object,
                grab_object,
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
                "Available tools are resolve_navigation_to_region, plan_region_exploration, execute_region_exploration_plan, navigate_to_waypoint, relocalize_here, rotate_by, rotate_towards_point, micro_adjust_to_pose, focus_detected_object, approach_detected_object, and grab_object.",
                "Navigation model:",
                "- resolve_navigation_to_region computes a safe centered path and a short waypoint.",
                f"- navigate_to_waypoint auto-rotates toward the waypoint before Nav2 when bearing error is above {self.config.navigation_auto_rotate_threshold_deg:.1f} degrees, then uses Nav2 first.",
                "- If Nav2 fails, navigate_to_waypoint can automatically use a short direct primitive fallback only when the saved occupancy map proves the straight corridor is footprint-clear.",
                "- If Nav2 fails with little/no progress and direct fallback is too far or blocked, inspect local_clearance_recovery. If it contains a recovery_pose, call micro_adjust_to_pose to that pose, then relocalize_here, then retry navigation.",
                "- Nav2 can sometimes fail to find paths toward objects or places, even when a route may exist.",
                "- Nav2 can be noisy for pure rotations, 180-degree turns, very close targets, and recovery after failed to make progress.",
                "- Local motion tools are bounded backend-controlled motions. Use them for orientation, tiny final corrections, or recovery, not for long navigation.",
                "- plan_region_exploration generates inspection stops and 65-degree shot cones for visual search inside a known region.",
                "- execute_region_exploration_plan runs that region search motion: it navigates to each inspection stop, rotates to each shot yaw, saves RGB debug shots, and runs object detection when a detector provider is configured.",
                "- execute_region_exploration_plan is for local visual search after the robot is already near the target region. Do not use exploration stops as a shortcut for long-distance navigation.",
                "- For far regions such as kitchen from another room, first use the short-horizon navigation loop: resolve_navigation_to_region, navigate_to_waypoint, relocalize_here, then resolve again until the region/final waypoint is reached. Only then run execute_region_exploration_plan.",
                "- For nearby regions already within roughly one short-horizon waypoint, execute_region_exploration_plan may be used directly for search.",
                "- Local clearance recovery is only an unstick maneuver after Nav2 fails: one small micro_adjust_to_pose to the suggested recovery_pose, relocalize_here, then retry Nav2. Never chain local recovery or micro_adjust_to_pose calls to create a route to a far region.",
                "- If execute_region_exploration_plan returns detection_status='matched', remaining shots were aborted and selected_detection contains the best box.",
                "- focus_detected_object reuses the tracked bbox when possible, then uses bounded rotation to center the object.",
                "- approach_detected_object solves RGB-D bbox depth from the depth image using real camera_info or fallback-FOV intrinsics, uses the RGB bbox center for approach recentering, infers nearby occupied support-surface angle from long-term memory, aligns the robot body perpendicular when useful, relocalizes after that alignment, then moves toward grasp staging using 80% of the remaining gap capped by max step and RGB-D corridor safety until the object is 0.35-0.45m away.",
                "- After each physical rotation or forward approach step, approach_detected_object refreshes detection by default so the next center/distance estimate uses a fresh RGB bbox with the latest depth image.",
                "- Object search/grab uses many local rotations and micro-motions, so odometry drift matters. If approach returns partial after useful motion and the object was still detected, call relocalize_here once, then retry approach_detected_object with constraints_json='{\"allow_surface_alignment\": false}' so the retry re-detects/checks reachability instead of repeating support-surface realignment. If that retry is still partial or blocked while the object is visible, report that it was found but not safely reachable.",
                "- grab_object is mocked for now; it is the future VLA skill entrypoint and should only be called after focus and approach succeed. If it returns mock_succeeded, report that the mock VLA handoff succeeded, not that a real physical pickup happened.",
                "For semantic region navigation such as `go to kitchen`, first call resolve_navigation_to_region.",
                "For object-search-and-grab commands such as `bring me the Coke from the kitchen`, first decide whether the region is nearby. If it is far, navigate to the region using the short-horizon loop before any execute_region_exploration_plan call. Once near the region, call execute_region_exploration_plan for that region and object label. If it matches, call focus_detected_object, then approach_detected_object, then grab_object. If detection_status is not_configured, say detection is not configured. If it is not_found, summarize where the robot looked.",
                f"The default short-horizon waypoint length is {self.config.navigation_waypoint_horizon_m:.1f} meters.",
                "Use the resolver's next_waypoint exactly; do not invent arbitrary waypoint coordinates.",
                "Call navigate_to_waypoint with next_waypoint.waypoint_id, x, y, and yaw.",
                "After each successful waypoint, call relocalize_here before resolving the next waypoint.",
                "If navigation succeeds and next_waypoint.is_final_waypoint is false, call resolve_navigation_to_region again from the updated pose and repeat.",
                "If navigation fails, inspect reason, pre_nav_auto_rotation, direct_fallback_plan, local_clearance_recovery, fallback_navigation, nav2.nav2_logs, distance_remaining_m, actual_pose_delta_m, estimated_feedback_path_m, and current_pose.",
                "If the target is within 0.5m and only a small final correction remains, use micro_adjust_to_pose.",
                "Do not call or describe unrelated perception, skill execution, approval, or stop tools; they are intentionally not exposed yet.",
                "Do not infer a navigation target yourself from a region shape.",
                "Return a concise final summary of the navigation status, final/current pose, and any relocalization correction.",
                "",
                "Example navigation loop for `go to kitchen`:",
                "1. Call resolve_navigation_to_region(target_label='kitchen', constraints_json='{}').",
                "2. Call navigate_to_waypoint with that exact waypoint and constraints_json='{}'. The tool handles pre-Nav2 auto-rotation and direct fallback internally.",
                "3. If the waypoint succeeded, call relocalize_here.",
                "4. If the waypoint succeeded and was not final, repeat from step 1.",
                "5. If the waypoint succeeded and was final, summarize that the region was reached.",
                "6. If the waypoint failed, summarize status, reason, distance_remaining_m, actual_pose_delta_m, estimated_feedback_path_m, and current_pose.",
                "",
                "Example custom horizon: resolve_navigation_to_region(target_label='kitchen', constraints_json='{\"waypoint_horizon_m\": 1.5}').",
                "Example far object-search request: for `navigate to kitchen, recognize the small yellow bottle, and move into grabbing position`, do not start with execute_region_exploration_plan if the kitchen is far. First run the navigation loop to kitchen. After the final kitchen waypoint succeeds and relocalize_here has run, call execute_region_exploration_plan(region_label='kitchen', object_label='small yellow bottle').",
                "Example nearby object-search request: if the robot is already beside the TV area and the target region is local, execute_region_exploration_plan(region_label='TV Area', object_label='small yellow bottle', constraints_json='{\"max_stops\": 2, \"shots_per_stop\": 2}') is appropriate.",
                "Example object-grab flow after reaching the region: execute_region_exploration_plan(region_label='kitchen', object_label='coke can'), then focus_detected_object(detection_id=selected_detection_id, object_label='coke can'), then approach_detected_object(detection_id=selected_detection_id, object_label='coke can'), then grab_object(object_label='coke can', detection_id=selected_detection_id, object_description='detected coke can').",
                "Bad example: do not respond to a far kitchen search by repeatedly using execute_region_exploration_plan, micro_adjust_to_pose, or local_clearance_recovery as the primary route. Those are local search/recovery tools; long travel must go through navigate_to_waypoint/Nav2 and relocalize_here.",
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
                if self.path.startswith("/api/artifacts/"):
                    self._send_artifact(self.path[len("/api/artifacts/") :])
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

            def _send_artifact(self, relpath: str) -> None:
                root = _agent_artifacts_root(controller.config).resolve()
                target = (root / relpath).resolve()
                if root != target and root not in target.parents:
                    self.send_error(HTTPStatus.FORBIDDEN, "artifact path escapes artifact root")
                    return
                if not target.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND, "artifact not found")
                    return
                content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
                data = target.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

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
        navigation_auto_rotate_threshold_deg=float(os.getenv("ROBOT42_NAVIGATION_AUTO_ROTATE_THRESHOLD_DEG", "45")),
        backend_request_timeout_s=float(os.getenv("ROBOT42_AGENT_BACKEND_REQUEST_TIMEOUT_S", "120")),
        agent_artifacts_root=os.getenv("ROBOT42_AGENT_ARTIFACTS_ROOT", "artifacts/agent_runs"),
        object_detector_provider=os.getenv("ROBOT42_OBJECT_DETECTOR_PROVIDER", "none"),
        object_detector_api_key=(
            os.getenv("ROBOT42_OBJECT_DETECTOR_API_KEY")
            or os.getenv("REPLICATE_API_TOKEN")
        ),
        object_detector_model=os.getenv("ROBOT42_OBJECT_DETECTOR_MODEL", "adirik/grounding-dino"),
        object_detector_model_version=os.getenv("ROBOT42_OBJECT_DETECTOR_MODEL_VERSION"),
        object_detector_box_threshold=float(os.getenv("ROBOT42_OBJECT_DETECTOR_BOX_THRESHOLD", "0.25")),
        object_detector_text_threshold=float(os.getenv("ROBOT42_OBJECT_DETECTOR_TEXT_THRESHOLD", "0.25")),
        object_detector_min_confidence=float(os.getenv("ROBOT42_OBJECT_DETECTOR_MIN_CONFIDENCE", "0.65")),
        object_detector_timeout_s=float(os.getenv("ROBOT42_OBJECT_DETECTOR_TIMEOUT_S", "90")),
        object_detector_max_image_edge_px=int(os.getenv("ROBOT42_OBJECT_DETECTOR_MAX_IMAGE_EDGE_PX", "1280")),
        object_detector_jpeg_quality=int(os.getenv("ROBOT42_OBJECT_DETECTOR_JPEG_QUALITY", "85")),
        object_focus_horizontal_fov_deg=float(os.getenv("ROBOT42_OBJECT_FOCUS_HORIZONTAL_FOV_DEG", "65")),
        object_focus_center_tolerance_norm=float(os.getenv("ROBOT42_OBJECT_FOCUS_CENTER_TOLERANCE_NORM", "0.08")),
        object_focus_max_attempts=int(os.getenv("ROBOT42_OBJECT_FOCUS_MAX_ATTEMPTS", "3")),
        object_approach_target_min_m=float(os.getenv("ROBOT42_OBJECT_APPROACH_TARGET_MIN_M", "0.35")),
        object_approach_target_max_m=float(os.getenv("ROBOT42_OBJECT_APPROACH_TARGET_MAX_M", "0.45")),
        object_approach_target_tolerance_m=float(os.getenv("ROBOT42_OBJECT_APPROACH_TARGET_TOLERANCE_M", "0.025")),
        object_approach_step_m=float(os.getenv("ROBOT42_OBJECT_APPROACH_STEP_M", "0.25")),
        object_approach_step_fraction=float(os.getenv("ROBOT42_OBJECT_APPROACH_STEP_FRACTION", "0.8")),
        object_approach_max_attempts=int(os.getenv("ROBOT42_OBJECT_APPROACH_MAX_ATTEMPTS", "20")),
        object_approach_robot_width_m=float(os.getenv("ROBOT42_OBJECT_APPROACH_ROBOT_WIDTH_M", "0.459")),
        object_approach_clearance_m=float(os.getenv("ROBOT42_OBJECT_APPROACH_CLEARANCE_M", "0.06")),
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


def _get_exploration_backend(config: HomeAgentConfig, path: str) -> dict[str, Any]:
    base_url = str(config.exploration_backend_url or "").rstrip("/")
    url = f"{base_url}{path}"
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=max(float(config.backend_request_timeout_s), 1.0)) as response:
            raw = response.read().decode("utf-8")
    except Exception as exc:
        return {
            "status": "unavailable",
            "reason": f"Exploration backend state is unavailable at {url}: {exc}",
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
    diagnostics = nav2.get("diagnostics") if isinstance(nav2.get("diagnostics"), dict) else {}
    status = str(response.get("status") or nav2.get("status") or "failed")
    current_pose = _pose_from_backend_response(response)
    distance_remaining_m = _remaining_distance_m(nav2)
    if distance_remaining_m is None and status == "succeeded":
        distance_remaining_m = 0.0
    if distance_remaining_m is None and current_pose is not None:
        distance_remaining_m = round(_pose_distance_m(current_pose, requested_pose), 3)
    feedback_summary = (
        diagnostics.get("feedback_summary")
        if isinstance(diagnostics.get("feedback_summary"), dict)
        else _feedback_summary(nav2)
    )
    failure_hint = _navigation_failure_hint(
        status=status,
        reason=response.get("reason") or nav2.get("reason") or response.get("message") or "",
        distance_remaining_m=distance_remaining_m,
        actual_pose_delta_m=nav2.get("actual_pose_delta_m"),
        feedback_summary=feedback_summary,
        nav2_logs=diagnostics.get("nav2_logs", []),
    )
    return {
        "tool": "navigate_to_waypoint",
        "status": status,
        "waypoint_id": waypoint_id,
        "requested_pose": requested_pose,
        "current_pose": current_pose,
        "normalized_pose": _json_pose(response["normalized_pose"]) if isinstance(response.get("normalized_pose"), dict) else None,
        "distance_remaining_m": distance_remaining_m,
        "actual_pose_delta_m": nav2.get("actual_pose_delta_m"),
        "actual_yaw_delta_deg": nav2.get("actual_yaw_delta_deg"),
        "estimated_feedback_path_m": feedback_summary.get("feedback_path_distance_m"),
        "reason": response.get("reason") or nav2.get("reason") or response.get("message") or "",
        "failure_hint": failure_hint,
        "backend_url": backend_url,
        "nav2": {
            "status": nav2.get("status"),
            "reason": nav2.get("reason"),
            "travelled_distance_m": nav2.get("travelled_distance_m"),
            "start_pose": _json_pose(nav2["start_pose"]) if isinstance(nav2.get("start_pose"), dict) else None,
            "end_pose": _json_pose(nav2["end_pose"]) if isinstance(nav2.get("end_pose"), dict) else None,
            "reached_pose": _json_pose(nav2["reached_pose"]) if isinstance(nav2.get("reached_pose"), dict) else None,
            "actual_pose_delta_m": nav2.get("actual_pose_delta_m"),
            "actual_yaw_delta_deg": nav2.get("actual_yaw_delta_deg"),
            "feedback_summary": feedback_summary,
            "nav2_logs": diagnostics.get("nav2_logs", []),
            "plan_status": plan.get("status"),
            "plan_reason": plan.get("reason"),
            "path_length_m": plan.get("path_length_m"),
        },
    }


def _local_motion_tool_result(response: dict[str, Any], *, primitive: str, backend_url: str | None) -> dict[str, Any]:
    motion = response.get("local_motion") if isinstance(response.get("local_motion"), dict) else {}
    current_pose = _pose_from_backend_response(response)
    if current_pose is None and isinstance(motion.get("end_pose"), dict):
        current_pose = _json_pose(motion["end_pose"])
    return {
        "tool": primitive,
        "status": str(response.get("status") or motion.get("status") or "failed"),
        "reason": response.get("reason") or motion.get("reason") or "",
        "backend_url": backend_url,
        "current_pose": current_pose,
        "start_pose": _json_pose(motion["start_pose"]) if isinstance(motion.get("start_pose"), dict) else None,
        "end_pose": _json_pose(motion["end_pose"]) if isinstance(motion.get("end_pose"), dict) else None,
        "actual_pose_delta_m": motion.get("actual_pose_delta_m"),
        "actual_yaw_delta_deg": motion.get("actual_yaw_delta_deg"),
        "distance_remaining_m": motion.get("distance_remaining_m"),
        "local_motion": motion,
    }


def _feedback_summary(nav2: dict[str, Any]) -> dict[str, Any]:
    samples = nav2.get("feedback_samples")
    if not isinstance(samples, list) or not samples:
        return {"sample_count": 0}
    last = samples[-1] if isinstance(samples[-1], dict) else {}
    remaining = last.get("remaining_distance_m", last.get("distance_remaining_m"))
    return {
        "sample_count": len(samples),
        "last_distance_remaining_m": _round_optional(remaining),
        "last_navigation_time_s": _round_optional(last.get("navigation_time_s")),
        "last_estimated_time_remaining_s": _round_optional(last.get("estimated_time_remaining_s")),
        "feedback_path_distance_m": _feedback_path_distance_m(samples),
        "number_of_recoveries": last.get("number_of_recoveries"),
        "last_pose": _json_pose(last["current_pose"]) if isinstance(last.get("current_pose"), dict) else None,
    }


def _feedback_path_distance_m(samples: list[Any]) -> float | None:
    poses: list[tuple[float, float]] = []
    for sample in samples:
        pose = sample.get("current_pose") if isinstance(sample, dict) else None
        if not isinstance(pose, dict):
            continue
        try:
            poses.append((float(pose.get("x", 0.0) or 0.0), float(pose.get("y", 0.0) or 0.0)))
        except Exception:
            continue
    if len(poses) < 2:
        return None
    distance = 0.0
    for previous, current in zip(poses, poses[1:]):
        distance += ((current[0] - previous[0]) ** 2 + (current[1] - previous[1]) ** 2) ** 0.5
    return round(distance, 3)


def _navigation_failure_hint(
    *,
    status: str,
    reason: str,
    distance_remaining_m: float | None,
    actual_pose_delta_m: Any,
    feedback_summary: dict[str, Any],
    nav2_logs: Any = None,
) -> str | None:
    if status == "succeeded":
        return None
    log_text = " ".join(
        str(event.get("message", ""))
        for event in (nav2_logs if isinstance(nav2_logs, list) else [])
        if isinstance(event, dict)
    )
    reason_lower = f"{reason or ''} {log_text}".lower()
    moved = _round_optional(actual_pose_delta_m)
    if "failed to make progress" in reason_lower or "aborted" in reason_lower:
        if moved is not None and moved < 0.05:
            return "Nav2 did not make useful physical progress; rotate toward the waypoint or use local recovery before retrying."
        return "Nav2 aborted after some execution; inspect current pose and retry only after changing orientation, waypoint, or localization."
    if "planner" in reason_lower or "path" in reason_lower:
        return "Nav2 could not compute a path; resolve a shorter waypoint or pick a different target."
    if distance_remaining_m is not None and distance_remaining_m < 0.25:
        return "The robot is close to the target; micro_adjust_to_pose may be more appropriate than another Nav2 goal."
    if int(feedback_summary.get("sample_count") or 0) == 0:
        return "Nav2 produced no feedback samples; check whether the goal was accepted and whether the navigation session is active."
    return None


def _round_optional(value: Any, digits: int = 3) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except Exception:
        return None


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


def _bearing_error_deg(current_pose: dict[str, Any], target_pose: dict[str, Any]) -> float:
    bearing = math.atan2(
        float(target_pose.get("y", 0.0) or 0.0) - float(current_pose.get("y", 0.0) or 0.0),
        float(target_pose.get("x", 0.0) or 0.0) - float(current_pose.get("x", 0.0) or 0.0),
    )
    yaw = float(current_pose.get("yaw", 0.0) or 0.0)
    delta = math.atan2(math.sin(bearing - yaw), math.cos(bearing - yaw))
    return round(math.degrees(delta), 2)


def _yaw_delta_deg(current_yaw: float, target_yaw: float) -> float:
    delta = math.atan2(math.sin(target_yaw - current_yaw), math.cos(target_yaw - current_yaw))
    return round(math.degrees(delta), 2)


def _constraint_bool(constraints: dict[str, Any], key: str, default: bool) -> bool:
    value = constraints.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _bounded_float(value: Any, fallback: float, *, minimum: float, maximum: float) -> float:
    try:
        result = float(value)
    except Exception:
        result = float(fallback)
    return min(max(result, minimum), maximum)


def _agent_artifacts_root(config: HomeAgentConfig) -> Path:
    return Path(config.agent_artifacts_root).expanduser()


def _object_detector_config(config: HomeAgentConfig) -> ObjectDetectorConfig:
    return ObjectDetectorConfig(
        provider=config.object_detector_provider,
        api_key=config.object_detector_api_key,
        model=config.object_detector_model,
        model_version=config.object_detector_model_version,
        box_threshold=config.object_detector_box_threshold,
        text_threshold=config.object_detector_text_threshold,
        min_confidence=config.object_detector_min_confidence,
        timeout_s=config.object_detector_timeout_s,
        max_image_edge_px=config.object_detector_max_image_edge_px,
        jpeg_quality=config.object_detector_jpeg_quality,
    )


def _image_file_to_data_url(path: Path, mime_type: str) -> str:
    data = path.read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type or 'image/png'};base64,{encoded}"


def _detection_center_error(detection: dict[str, Any], capture: dict[str, Any]) -> dict[str, Any]:
    bbox = detection.get("bbox_xyxy")
    if not isinstance(bbox, list) or len(bbox) < 4:
        return {"status": "failed", "reason": "Detection does not include bbox_xyxy."}
    width = capture.get("image_width")
    height = capture.get("image_height")
    if not width or not height:
        image_path = capture.get("image_path")
        if isinstance(image_path, str):
            size = _image_size_from_file(Path(image_path))
            if size is not None:
                width, height = size
    try:
        image_width = float(width)
        image_height = float(height)
        x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    except Exception:
        return {"status": "failed", "reason": "Detection bbox or image size is invalid."}
    if image_width <= 0.0 or image_height <= 0.0:
        return {"status": "failed", "reason": "Image dimensions are unavailable."}
    x0, y0, x1, y1 = _bbox_to_pixel_xyxy([x0, y0, x1, y1], image_width=image_width, image_height=image_height)
    center_x = (x0 + x1) * 0.5
    center_y = (y0 + y1) * 0.5
    return {
        "status": "succeeded",
        "image_width": int(image_width),
        "image_height": int(image_height),
        "center_x": round(center_x, 3),
        "center_y": round(center_y, 3),
        "error_norm": round((center_x - image_width * 0.5) / (image_width * 0.5), 4),
        "vertical_error_norm": round((center_y - image_height * 0.5) / (image_height * 0.5), 4),
    }


def _horizontally_centered_detection(detection: dict[str, Any], capture: dict[str, Any]) -> dict[str, Any] | None:
    bbox = detection.get("bbox_xyxy")
    if not isinstance(bbox, list) or len(bbox) < 4:
        return None
    width = capture.get("image_width")
    height = capture.get("image_height")
    if not width or not height:
        image_path = capture.get("image_path")
        if isinstance(image_path, str):
            size = _image_size_from_file(Path(image_path))
            if size is not None:
                width, height = size
    try:
        image_width = float(width)
        image_height = float(height)
    except Exception:
        return None
    if image_width <= 0.0 or image_height <= 0.0:
        return None
    left, top, right, bottom = _bbox_to_pixel_xyxy(
        [float(item) for item in bbox[:4]],
        image_width=image_width,
        image_height=image_height,
    )
    box_width = max(right - left, 1.0)
    new_left = max(min(image_width * 0.5 - box_width * 0.5, image_width - box_width), 0.0)
    predicted = dict(detection)
    predicted["bbox_xyxy"] = [
        round(new_left, 3),
        round(top, 3),
        round(new_left + box_width, 3),
        round(bottom, 3),
    ]
    predicted["tracking_prediction"] = "horizontally_centered_after_focus_rotation"
    return predicted


def _object_map_pose_from_geometry(geometry: dict[str, Any], current_pose: dict[str, Any]) -> dict[str, float] | None:
    estimated_map = geometry.get("estimated_pose_map")
    if isinstance(estimated_map, dict) and estimated_map.get("x") is not None and estimated_map.get("y") is not None:
        try:
            return {"x": float(estimated_map["x"]), "y": float(estimated_map["y"]), "yaw": 0.0}
        except Exception:
            pass
    pose = geometry.get("current_pose") if isinstance(geometry.get("current_pose"), dict) else current_pose
    if not isinstance(pose, dict):
        return None
    try:
        x = float(pose.get("x", 0.0) or 0.0)
        y = float(pose.get("y", 0.0) or 0.0)
        yaw = float(pose.get("yaw", 0.0) or 0.0)
        forward_m = float(geometry.get("forward_m"))
        lateral_m = float(geometry.get("lateral_m", 0.0) or 0.0)
    except Exception:
        return None
    return {
        "x": round(x + math.cos(yaw) * forward_m - math.sin(yaw) * lateral_m, 3),
        "y": round(y + math.sin(yaw) * forward_m + math.cos(yaw) * lateral_m, 3),
        "yaw": 0.0,
    }


def _bbox_to_pixel_xyxy(bbox: list[float], *, image_width: float, image_height: float) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.5:
        x0 *= image_width
        x1 *= image_width
        y0 *= image_height
        y1 *= image_height
    left, right = sorted((x0, x1))
    top, bottom = sorted((y0, y1))
    return left, top, right, bottom


def _visual_servo_yaw_step_deg(
    center: dict[str, Any],
    constraints: dict[str, Any],
    config: HomeAgentConfig,
) -> float:
    error_norm = float(center.get("error_norm", 0.0) or 0.0)
    horizontal_fov_deg = _bounded_float(
        constraints.get("horizontal_fov_deg", config.object_focus_horizontal_fov_deg),
        config.object_focus_horizontal_fov_deg,
        minimum=20.0,
        maximum=120.0,
    )
    max_step_deg = _bounded_float(
        constraints.get("max_yaw_step_deg", 12.0),
        12.0,
        minimum=1.0,
        maximum=45.0,
    )
    yaw_sign = _bounded_float(constraints.get("image_yaw_sign", -1.0), -1.0, minimum=-1.0, maximum=1.0)
    if abs(yaw_sign) < 1e-6:
        yaw_sign = -1.0
    delta = yaw_sign * error_norm * horizontal_fov_deg * 0.5
    return round(max(min(delta, max_step_deg), -max_step_deg), 3)


def _pose_forward_step(pose: dict[str, Any], distance_m: float) -> dict[str, float]:
    yaw = float(pose.get("yaw", 0.0) or 0.0)
    distance = float(distance_m)
    return _json_pose(
        {
            "x": float(pose.get("x", 0.0) or 0.0) + distance * math.cos(yaw),
            "y": float(pose.get("y", 0.0) or 0.0) + distance * math.sin(yaw),
            "yaw": yaw,
        }
    )


def _aggregate_detection_status(stops: list[dict[str, Any]], object_label: str) -> str:
    if not object_label.strip():
        return "skipped"
    statuses = [
        str(shot.get("detection", {}).get("status") or "")
        for stop in stops
        for shot in stop.get("shots", [])
        if isinstance(shot, dict)
    ]
    if any(status == "matched" for status in statuses):
        return "matched"
    if any(status == "not_found" for status in statuses):
        return "not_found"
    if any(status in {"failed", "unavailable"} for status in statuses):
        return "failed"
    if any(status == "skipped" for status in statuses):
        return "skipped"
    return "not_configured"


def _region_exploration_result_reason(detection_status: str) -> str:
    if detection_status == "not_found":
        return "Region exploration motion completed; no matching object was detected."
    if detection_status == "failed":
        return "Region exploration motion completed, but object detection failed on at least one shot."
    if detection_status == "skipped":
        return "Region exploration motion completed; object detection was skipped."
    return "Region exploration motion completed; object detection is not configured."


def _record_capture_detection(
    *,
    config: HomeAgentConfig,
    run_id: str,
    capture: dict[str, Any],
    detection: dict[str, Any],
) -> None:
    annotation = _write_detection_annotation_artifact(
        config=config,
        run_id=run_id,
        capture=capture,
        detection=detection,
    )
    if annotation:
        capture.update(annotation)
    metadata_path = capture.get("metadata_path")
    if isinstance(metadata_path, str):
        path = Path(metadata_path)
        metadata = _load_json_file(path)
        if isinstance(metadata, dict):
            metadata["detection"] = detection
            metadata.update(annotation)
            try:
                path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            except Exception:
                pass

    manifest_path = _agent_artifacts_root(config) / run_id / "vision_report" / "manifest.json"
    manifest = _load_json_file(manifest_path)
    if not isinstance(manifest, dict):
        return
    image_path = capture.get("image_path")
    captures = manifest.get("captures")
    if not isinstance(captures, list):
        return
    for item in captures:
        if isinstance(item, dict) and item.get("image_path") == image_path:
            item["detection"] = detection
            item.update(annotation)
            break
    manifest["updated_at"] = time.time()
    try:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception:
        pass


def _write_detection_annotation_artifact(
    *,
    config: HomeAgentConfig,
    run_id: str,
    capture: dict[str, Any],
    detection: dict[str, Any],
) -> dict[str, Any]:
    boxes = _detection_boxes_for_annotation(detection)
    image_path_value = capture.get("image_path")
    if not boxes or not isinstance(image_path_value, str):
        return {}
    image_path = Path(image_path_value)
    if not image_path.is_file():
        return {}
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return {}
    try:
        with Image.open(image_path) as opened:
            image = opened.convert("RGBA")
        width, height = image.size
        if width <= 0 or height <= 0:
            return {}
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        font = ImageFont.load_default()
        stroke = max(2, int(round(max(width, height) / 320)))
        for index, box in enumerate(boxes):
            bbox = box.get("bbox_xyxy")
            if not isinstance(bbox, list) or len(bbox) < 4:
                continue
            left, top, right, bottom = _bbox_to_pixel_xyxy(
                [float(item) for item in bbox[:4]],
                image_width=float(width),
                image_height=float(height),
            )
            left = max(0.0, min(float(width), left))
            right = max(0.0, min(float(width), right))
            top = max(0.0, min(float(height), top))
            bottom = max(0.0, min(float(height), bottom))
            if right <= left or bottom <= top:
                continue
            selected = bool(box.get("selected") or (index == 0 and not any(item.get("selected") for item in boxes)))
            outline = (20, 160, 90, 255) if selected else (224, 130, 20, 255)
            fill = (20, 160, 90, 42) if selected else (224, 130, 20, 38)
            draw.rectangle([left, top, right, bottom], fill=fill, outline=outline, width=stroke)
            label = str(box.get("label") or detection.get("object_label") or "object")
            confidence = box.get("confidence")
            try:
                confidence_value = float(confidence)
                if confidence_value > 0.0:
                    label = f"{label} {confidence_value * 100:.0f}%"
            except Exception:
                pass
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_width = max(float(text_bbox[2] - text_bbox[0]), 1.0)
            text_height = max(float(text_bbox[3] - text_bbox[1]), 1.0)
            label_x = min(max(left, 0.0), max(float(width) - text_width - 8.0, 0.0))
            label_y = max(top - text_height - 8.0, 0.0)
            draw.rectangle(
                [label_x, label_y, label_x + text_width + 8.0, label_y + text_height + 6.0],
                fill=(17, 24, 39, 218),
            )
            draw.text((label_x + 4.0, label_y + 3.0), label, fill=(255, 255, 255, 255), font=font)
        annotated = Image.alpha_composite(image, overlay).convert("RGB")
        annotated_path = image_path.with_name(f"{image_path.stem}_detections.png")
        annotated.save(annotated_path, format="PNG")
    except Exception:
        return {}
    root = _agent_artifacts_root(config)
    try:
        relpath = str(annotated_path.relative_to(root))
    except Exception:
        relpath = f"{run_id}/vision_report/shots/{annotated_path.name}"
    return {
        "annotated_image_path": str(annotated_path),
        "annotated_artifact_relpath": relpath,
        "annotated_artifact_url": f"/api/artifacts/{relpath}",
    }


def _detection_boxes_for_annotation(detection: dict[str, Any]) -> list[dict[str, Any]]:
    selected = detection.get("selected_detection") if isinstance(detection.get("selected_detection"), dict) else None
    selected_id = None
    if selected is not None:
        selected_id = selected.get("tracking_id") or selected.get("detection_id")
    selected_id = selected_id or detection.get("selected_detection_id")
    raw_boxes = detection.get("detections") if isinstance(detection.get("detections"), list) else []
    if not raw_boxes and selected is not None:
        raw_boxes = [selected]
    boxes: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_boxes):
        if not isinstance(raw, dict):
            continue
        bbox = raw.get("bbox_xyxy")
        if not isinstance(bbox, list) or len(bbox) < 4:
            continue
        box_id = raw.get("tracking_id") or raw.get("detection_id") or f"box_{index}"
        boxes.append(
            {
                "bbox_xyxy": bbox,
                "label": raw.get("label") or detection.get("object_label") or "object",
                "confidence": raw.get("confidence"),
                "selected": bool(selected_id and box_id == selected_id),
            }
        )
    return boxes


def _save_rgb_capture_artifact(
    *,
    config: HomeAgentConfig,
    run_id: str,
    file_stem: str,
    data_url: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    mime_type, image_bytes = _decode_image_data_url(data_url)
    extension = _image_extension_for_mime(mime_type)
    image_size = _image_size_from_bytes(image_bytes)
    report_dir = _agent_artifacts_root(config) / run_id / "vision_report"
    shots_dir = report_dir / "shots"
    shots_dir.mkdir(parents=True, exist_ok=True)
    image_path = shots_dir / f"{file_stem}{extension}"
    metadata_path = shots_dir / f"{file_stem}.json"
    image_path.write_bytes(image_bytes)
    metadata = {
        **metadata,
        "mime_type": mime_type,
        "image_filename": image_path.name,
        "metadata_filename": metadata_path.name,
    }
    if image_size is not None:
        metadata["image_width"] = image_size[0]
        metadata["image_height"] = image_size[1]
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path = report_dir / "manifest.json"
    manifest = _load_json_file(manifest_path)
    if not isinstance(manifest, dict):
        manifest = {}
    captures = manifest.get("captures")
    if not isinstance(captures, list):
        captures = []
    artifact_relpath = f"{run_id}/vision_report/shots/{image_path.name}"
    metadata_relpath = f"{run_id}/vision_report/shots/{metadata_path.name}"
    manifest_capture = {
        **metadata,
        "image_path": str(image_path),
        "metadata_path": str(metadata_path),
        "artifact_relpath": artifact_relpath,
        "metadata_relpath": metadata_relpath,
        "artifact_url": f"/api/artifacts/{artifact_relpath}",
    }
    captures.append(manifest_capture)
    manifest = {
        "run_id": run_id,
        "updated_at": time.time(),
        "capture_count": len(captures),
        "captures": captures,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "mime_type": mime_type,
        "image_path": str(image_path),
        "metadata_path": str(metadata_path),
        "manifest_path": str(manifest_path),
        "artifact_relpath": artifact_relpath,
        "metadata_relpath": metadata_relpath,
        "artifact_url": f"/api/artifacts/{artifact_relpath}",
        "metadata_url": f"/api/artifacts/{metadata_relpath}",
        "vision_report_url": f"/api/artifacts/{run_id}/vision_report/manifest.json",
        "captured_at": metadata.get("captured_at"),
        "robot_pose": metadata.get("robot_pose"),
        "current_pose": metadata.get("current_pose"),
        "image_width": metadata.get("image_width"),
        "image_height": metadata.get("image_height"),
    }


def _decode_image_data_url(data_url: str) -> tuple[str, bytes]:
    header, encoded = data_url.split(",", 1)
    mime_type = header.removeprefix("data:").split(";", 1)[0] or "image/png"
    return mime_type, base64.b64decode(encoded)


def _image_size_from_file(path: Path) -> tuple[int, int] | None:
    try:
        return _image_size_from_bytes(path.read_bytes())
    except Exception:
        return None


def _image_size_from_bytes(data: bytes) -> tuple[int, int] | None:
    if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n":
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    if len(data) >= 10 and data[:2] == b"\xff\xd8":
        index = 2
        while index + 9 < len(data):
            if data[index] != 0xFF:
                index += 1
                continue
            marker = data[index + 1]
            index += 2
            if marker in {0xD8, 0xD9}:
                continue
            if index + 2 > len(data):
                return None
            segment_length = int.from_bytes(data[index : index + 2], "big")
            if segment_length < 2 or index + segment_length > len(data):
                return None
            if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
                if segment_length >= 7:
                    height = int.from_bytes(data[index + 3 : index + 5], "big")
                    width = int.from_bytes(data[index + 5 : index + 7], "big")
                    return width, height
                return None
            index += segment_length
    return None


def _image_extension_for_mime(mime_type: str) -> str:
    if mime_type == "image/jpeg":
        return ".jpg"
    if mime_type == "image/webp":
        return ".webp"
    if mime_type == "image/gif":
        return ".gif"
    return ".png"


def _safe_artifact_name(value: str) -> str:
    normalized = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip().lower())
    normalized = "_".join(part for part in normalized.split("_") if part)
    return normalized[:120] or f"capture_{int(time.time())}"


def _object_approach_key(value: str) -> str:
    return " ".join(str(value or "").replace("_", " ").lower().split())


def _load_json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _region_exploration_navigation_constraints(constraints: dict[str, Any]) -> dict[str, Any]:
    nested = constraints.get("navigation") or constraints.get("navigation_constraints")
    if isinstance(nested, dict):
        return dict(nested)
    allowed = {
        "allow_auto_rotate",
        "auto_rotate_threshold_deg",
        "allow_direct_fallback",
        "direct_fallback_max_distance_m",
        "direct_fallback_min_clearance_m",
        "local_recovery_max_distance_m",
    }
    return {key: constraints[key] for key in allowed if key in constraints}


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
