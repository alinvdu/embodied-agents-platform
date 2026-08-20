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
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from typing import Any

from .offload import OffloadClient, serialize_world_state
from .models import WorldState


@dataclass
class BrainBridge:
    offload_client: OffloadClient
    latest_state: dict[str, Any] = field(default_factory=dict)
    latest_sensors: dict[str, Any] = field(default_factory=dict)
    last_publish_reason: str | None = None
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)

    def register(self) -> dict[str, Any]:
        registration = self.offload_client.register()
        return registration.__dict__

    def publish_state(
        self,
        world_state: WorldState,
        *,
        reason: str = "brain_bridge_publish",
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
        reason: str = "brain_bridge_publish",
        sensors: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            response = self.offload_client.publish_state_payload(
                world_state,
                reason=reason,
                sensors=sensors,
            )
            self.latest_state = dict(world_state)
            self.latest_sensors = dict(sensors or world_state.get("metadata", {}).get("sensors", {}))
            self.last_publish_reason = reason
            return response

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            registration = self.offload_client.ensure_registered()
            return {
                "brain_id": registration.brain_id,
                "brain_name": registration.brain_name,
                "server_url": registration.server_url,
                "metadata": registration.metadata,
                "latest_state": self.latest_state,
                "latest_sensors": self.latest_sensors,
                "last_publish_reason": self.last_publish_reason,
            }


class BrainServiceServer:
    def __init__(
        self,
        bridge: BrainBridge,
        *,
        host: str = "127.0.0.1",
        port: int = 8891,
    ) -> None:
        self.bridge = bridge
        self.host = host
        self.port = port
        self._server = ThreadingHTTPServer((self.host, self.port), self._build_handler())
        self.host, self.port = self._server.server_address

    def _build_handler(self):
        bridge = self.bridge

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                path = self.path.rstrip("/")
                if path == "/api/health":
                    self._send_json({"status": "ok", "bridge": bridge.snapshot()})
                    return
                if path == "/api/state":
                    self._send_json(bridge.snapshot())
                    return
                if path == "/api/registration":
                    self._send_json(bridge.register())
                    return
                self.send_error(HTTPStatus.NOT_FOUND, "not found")

            def do_POST(self) -> None:
                path = self.path.rstrip("/")
                payload = self._read_json_body()
                if path == "/api/register":
                    self._send_json(bridge.register())
                    return
                if path == "/api/heartbeat":
                    self._send_json(bridge.offload_client.heartbeat())
                    return
                if path == "/api/state":
                    response = bridge.publish_state_payload(
                        dict(payload.get("world_state", {})),
                        reason=str(payload.get("reason", "brain_bridge_publish")),
                        sensors=dict(payload.get("sensors", {})),
                    )
                    self._send_json(response)
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
