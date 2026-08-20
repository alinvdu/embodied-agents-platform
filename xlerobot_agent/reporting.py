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
from datetime import datetime, timezone
import threading
from typing import Any


class AgentStopRequested(RuntimeError):
    pass


@dataclass(frozen=True)
class AgentEvent:
    event_id: int
    timestamp: str
    kind: str
    title: str
    summary: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class LiveAgentReport:
    backend: str
    models: dict[str, Any]
    environment: dict[str, Any]
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)
    _pause_condition: threading.Condition = field(init=False)
    _events: list[AgentEvent] = field(default_factory=list, init=False)
    _next_event_id: int = field(default=1, init=False)
    status: str = field(default="idle", init=False)
    paused: bool = field(default=False, init=False)
    stop_requested: bool = field(default=False, init=False)
    active_command: str = field(default="", init=False)
    normalized_instruction: str = field(default="", init=False)
    subgoals: list[str] = field(default_factory=list, init=False)
    discovered_places: list[str] = field(default_factory=list, init=False)
    active_subgoal: str | None = field(default=None, init=False)
    run_id: int = field(default=0, init=False)
    final_summary: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._pause_condition = threading.Condition(self._lock)

    def begin_run(self, command: str) -> int:
        with self._lock:
            self.run_id += 1
            self.status = "running"
            self.paused = False
            self.stop_requested = False
            self.active_command = command
            self.normalized_instruction = ""
            self.subgoals = []
            self.discovered_places = []
            self.active_subgoal = None
            self.final_summary = None
            self._events = []
            self._next_event_id = 1
            self.add_event(
                "session_started",
                "Session Started",
                f"Started a new agent run for command: {command}",
            )
            return self.run_id

    def add_event(
        self,
        kind: str,
        title: str,
        summary: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> AgentEvent:
        with self._lock:
            event = AgentEvent(
                event_id=self._next_event_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                kind=kind,
                title=title,
                summary=summary,
                details=details or {},
            )
            self._next_event_id += 1
            self._events.append(event)
            return event

    def set_plan(
        self,
        *,
        normalized_instruction: str,
        discovered_places: list[str],
        subgoals: list[str],
    ) -> None:
        with self._lock:
            self.normalized_instruction = normalized_instruction
            self.discovered_places = list(discovered_places)
            self.subgoals = list(subgoals)

    def set_active_subgoal(self, subgoal: str | None) -> None:
        with self._lock:
            self.active_subgoal = subgoal

    def request_pause(self) -> None:
        with self._lock:
            self.paused = True
            self.add_event("paused", "Paused", "Pause was requested by the operator.")

    def resume(self) -> None:
        with self._pause_condition:
            self.paused = False
            self._pause_condition.notify_all()
            self.add_event("resumed", "Resumed", "Execution resumed after an operator pause.")

    def request_stop(self) -> None:
        with self._pause_condition:
            self.stop_requested = True
            self.paused = False
            self._pause_condition.notify_all()
            self.add_event("stop_requested", "Stop Requested", "Stop was requested by the operator.")

    def wait_if_paused(self) -> None:
        with self._pause_condition:
            while self.paused and not self.stop_requested:
                self.status = "paused"
                self._pause_condition.wait(timeout=0.5)
            if self.stop_requested:
                raise AgentStopRequested("stop requested by operator")
            if self.status != "running":
                self.status = "running"

    def ensure_not_stopped(self) -> None:
        with self._lock:
            if self.stop_requested:
                raise AgentStopRequested("stop requested by operator")

    def finish(self, status: str, summary: str) -> None:
        with self._lock:
            self.status = status
            self.final_summary = summary
            self.active_subgoal = None
            self.add_event("session_finished", "Session Finished", summary)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "backend": self.backend,
                "models": self.models,
                "environment": self.environment,
                "status": self.status,
                "paused": self.paused,
                "stop_requested": self.stop_requested,
                "run_id": self.run_id,
                "active_command": self.active_command,
                "normalized_instruction": self.normalized_instruction,
                "discovered_places": list(self.discovered_places),
                "subgoals": list(self.subgoals),
                "active_subgoal": self.active_subgoal,
                "final_summary": self.final_summary,
                "events": [
                    {
                        "event_id": event.event_id,
                        "timestamp": event.timestamp,
                        "kind": event.kind,
                        "title": event.title,
                        "summary": event.summary,
                        "details": event.details,
                    }
                    for event in self._events
                ],
            }
