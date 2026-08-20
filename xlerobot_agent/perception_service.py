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

from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from typing import Any, Mapping

from .models import ExecutionStatus
from .perception import build_scene_snapshot, ground_object_matches, waypoint_from_matches


PERCEPTION_TOOL_IDS = frozenset({"perceive_scene", "ground_object_3d", "set_waypoint_from_object"})


def execute_perception_tool(
    tool_id: str,
    *,
    context: Mapping[str, Any],
    brain: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if tool_id not in PERCEPTION_TOOL_IDS:
        return {
            "tool_id": tool_id,
            "status": ExecutionStatus.BLOCKED.value,
            "summary": f"Perception service does not implement `{tool_id}`.",
            "details": {},
            "resolved_subgoal": False,
        }

    payload = dict(_mapping_value(context, "payload"))
    subgoal = dict(_mapping_value(context, "subgoal"))
    world_state = dict(_mapping_value(context, "world_state"))
    target = payload.get("target") or payload.get("pose") or subgoal.get("target") or subgoal.get("text")
    scene = _scene_from_context(world_state, target=target, brain=brain)

    if tool_id == "perceive_scene":
        return {
            "tool_id": tool_id,
            "status": ExecutionStatus.SUCCEEDED.value,
            "summary": scene["scene_summary"],
            "details": scene,
            "resolved_subgoal": False,
            "recommended_action_id": "ground_object_3d" if target else None,
        }

    if tool_id == "ground_object_3d":
        grounding = ground_object_matches(scene, str(target) if target is not None else None)
        matches = grounding["matches"]
        if not matches:
            return {
                "tool_id": tool_id,
                "status": ExecutionStatus.BLOCKED.value,
                "summary": f"No grounded 3D match was found for `{target}`.",
                "details": {"scene": scene, **grounding},
                "resolved_subgoal": False,
                "recommended_action_id": "perceive_scene",
            }
        best = matches[0]
        return {
            "tool_id": tool_id,
            "status": ExecutionStatus.SUCCEEDED.value,
            "summary": f"Grounded `{target}` with confidence {float(best.get('confidence', 0.0)):.2f}.",
            "details": {"scene": scene, **grounding, "best_match": best},
            "resolved_subgoal": False,
            "recommended_action_id": "set_waypoint_from_object",
        }

    waypoint = waypoint_from_matches(scene, query=str(target) if target is not None else None)
    if waypoint is None:
        return {
            "tool_id": tool_id,
            "status": ExecutionStatus.BLOCKED.value,
            "summary": f"No waypoint could be derived for `{target}`.",
            "details": {"scene": scene, "target": target},
            "resolved_subgoal": False,
            "recommended_action_id": "ground_object_3d",
        }
    return {
        "tool_id": tool_id,
        "status": ExecutionStatus.SUCCEEDED.value,
        "summary": f"Derived an approach waypoint for `{target}` from 3D perception.",
        "details": {"scene": scene, "target": target, "waypoint": waypoint},
        "resolved_subgoal": False,
        "recommended_action_id": "go_to_pose",
    }


def extract_scene_from_tool_result(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    details = _mapping_value(payload, "details")
    if details.get("annotations"):
        return dict(details)
    scene = details.get("scene")
    if isinstance(scene, Mapping):
        return dict(scene)
    return None


def _scene_from_context(
    world_state: Mapping[str, Any],
    *,
    target: Any,
    brain: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cached_scene = None
    if isinstance(brain, Mapping):
        candidate = brain.get("latest_scene")
        if isinstance(candidate, Mapping):
            cached_scene = candidate
    return build_scene_snapshot(
        current_pose=str(world_state.get("current_pose", "unknown")),
        visible_objects=list(world_state.get("visible_objects", [])),
        image_descriptions=list(world_state.get("image_descriptions", [])),
        metadata=dict(_mapping_value(world_state, "metadata")),
        target=(str(target) if target is not None else None),
        cached_scene=cached_scene,
    )


def _mapping_value(source: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = source.get(key, {})
    if isinstance(value, Mapping):
        return value
    return {}


@dataclass(frozen=True)
class PerceptionServiceConfig:
    host: str = "0.0.0.0"
    port: int = 8892


class PerceptionService:
    def __init__(self, config: PerceptionServiceConfig | None = None) -> None:
        self.config = config or PerceptionServiceConfig()
        self._server = ThreadingHTTPServer((self.config.host, self.config.port), self._build_handler())
        self.host, self.port = self._server.server_address

    def _build_handler(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                path = self.path.rstrip("/")
                if path == "/api/health":
                    self._send_json({"status": "ok", "tools": sorted(PERCEPTION_TOOL_IDS)})
                    return
                self.send_error(HTTPStatus.NOT_FOUND, "not found")

            def do_POST(self) -> None:
                path = self.path.rstrip("/")
                payload = self._read_json_body()
                if path.startswith("/api/tools/"):
                    tool_id = path.split("/")[-1]
                    result = execute_perception_tool(
                        tool_id,
                        context=dict(payload.get("context", payload)),
                        brain=_mapping_value(payload, "brain") or None,
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
