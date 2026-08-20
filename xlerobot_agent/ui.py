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

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from typing import Any

from .playground import PlaygroundAgentController


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>XLeRobot Agent Playground</title>
  <style>
    :root {
      --bg: #f5f1e8;
      --panel: rgba(255,255,255,0.75);
      --line: rgba(26, 35, 47, 0.14);
      --text: #1a232f;
      --muted: #5c6670;
      --accent: #0f766e;
      --accent-2: #b45309;
      --danger: #b91c1c;
      --ok: #166534;
      --shadow: 0 20px 40px rgba(26, 35, 47, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(15,118,110,0.18), transparent 35%),
        radial-gradient(circle at top right, rgba(180,83,9,0.16), transparent 30%),
        linear-gradient(180deg, #f8f5ed 0%, #efe7d7 100%);
      min-height: 100vh;
    }
    .shell {
      max-width: 1400px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }
    .hero {
      display: grid;
      gap: 12px;
      margin-bottom: 24px;
    }
    .eyebrow {
      text-transform: uppercase;
      letter-spacing: 0.14em;
      font-size: 12px;
      color: var(--muted);
    }
    h1 {
      margin: 0;
      font-family: "IBM Plex Serif", Georgia, serif;
      font-size: clamp(32px, 5vw, 54px);
      line-height: 0.98;
      max-width: 12ch;
    }
    .grid {
      display: grid;
      grid-template-columns: 380px 1fr;
      gap: 20px;
      align-items: start;
    }
    .panel {
      background: var(--panel);
      backdrop-filter: blur(16px);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: var(--shadow);
      padding: 18px;
    }
    .stack { display: grid; gap: 16px; }
    .controls form {
      display: grid;
      gap: 12px;
    }
    .row {
      display: grid;
      gap: 8px;
    }
    label {
      font-size: 12px;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      color: var(--muted);
    }
    input, textarea, button {
      font: inherit;
    }
    textarea, input {
      width: 100%;
      border-radius: 14px;
      border: 1px solid var(--line);
      padding: 12px 14px;
      background: rgba(255,255,255,0.9);
      color: var(--text);
    }
    textarea {
      min-height: 100px;
      resize: vertical;
    }
    .button-row {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }
    button {
      border: 0;
      border-radius: 999px;
      padding: 12px 16px;
      font-weight: 600;
      cursor: pointer;
      transition: transform 120ms ease, opacity 120ms ease;
    }
    button:hover { transform: translateY(-1px); }
    .primary { background: var(--accent); color: white; }
    .secondary { background: #fff7ed; color: var(--accent-2); }
    .danger { background: #fee2e2; color: var(--danger); }
    .meta-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 12px;
    }
    .meta-card {
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 12px;
      background: rgba(255,255,255,0.72);
    }
    .meta-card .key {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 6px;
    }
    .meta-card .value {
      font-size: 16px;
      font-weight: 600;
    }
    .subgoals {
      display: grid;
      gap: 10px;
    }
    .subgoal {
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: rgba(255,255,255,0.7);
    }
    .subgoal.active {
      border-color: rgba(15,118,110,0.35);
      background: rgba(15,118,110,0.08);
    }
    .event-stream {
      display: grid;
      gap: 12px;
      max-height: 72vh;
      overflow: auto;
      padding-right: 4px;
    }
    details.event {
      border: 1px solid var(--line);
      border-radius: 16px;
      background: rgba(255,255,255,0.74);
      padding: 12px 14px;
    }
    details.event summary {
      cursor: pointer;
      list-style: none;
    }
    details.event summary::-webkit-details-marker {
      display: none;
    }
    .event-top {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: baseline;
      margin-bottom: 8px;
    }
    .event-kind {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }
    .event-title {
      font-size: 18px;
      font-weight: 600;
      margin: 0;
    }
    .event-summary {
      margin: 0;
      color: var(--text);
    }
    pre {
      white-space: pre-wrap;
      word-break: break-word;
      border-radius: 14px;
      background: #f8fafc;
      padding: 12px;
      border: 1px solid var(--line);
      color: #0f172a;
      overflow: auto;
    }
    .status-completed { color: var(--ok); }
    .status-failed, .status-stopped { color: var(--danger); }
    .muted { color: var(--muted); }
    @media (max-width: 980px) {
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="eyebrow">Agent Playground</div>
      <h1>Live robot planning, action, review, and reflection.</h1>
      <div class="muted">This interface shows the command, active plan, model outputs, tool use, skill execution, and replanning flow in real time.</div>
    </section>

    <div class="grid">
      <div class="stack">
        <section class="panel controls">
          <form id="command-form">
            <div class="row">
              <label for="command">Command</label>
              <textarea id="command" placeholder="Go to the kitchen and open the fridge."></textarea>
            </div>
            <div class="row">
              <label for="voice-transcript">Voice Transcript</label>
              <input id="voice-transcript" placeholder="hey xlerobot go to the kitchen and open the fridge" />
            </div>
            <div class="button-row">
              <button class="primary" type="submit">Start Run</button>
              <button class="secondary" type="button" id="pause">Pause</button>
              <button class="secondary" type="button" id="resume">Resume</button>
              <button class="danger" type="button" id="stop">Stop</button>
            </div>
          </form>
        </section>

        <section class="panel">
          <div class="eyebrow">Session</div>
          <div id="meta-grid" class="meta-grid"></div>
        </section>

        <section class="panel">
          <div class="eyebrow">Plan</div>
          <h3>Current plan</h3>
          <div id="plan-summary" class="muted">No active plan.</div>
          <div id="subgoals" class="subgoals"></div>
        </section>
      </div>

      <section class="panel">
        <div class="eyebrow">Trace</div>
        <h3>Event stream</h3>
        <div id="events" class="event-stream"></div>
      </section>
    </div>
  </div>

  <script>
    async function postJson(url, payload) {
      const res = await fetch(url, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload || {})
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || ('Request failed: ' + res.status));
      }
      return res.json();
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;');
    }

    function renderMeta(state) {
      const models = Object.entries(state.models || {}).map(([role, config]) => (
        role + ': ' + config.provider + '/' + config.model
      )).join(' | ');
      const items = [
        ['Status', state.status + (state.paused ? ' (paused)' : '')],
        ['Backend', state.backend],
        ['Run ID', state.run_id],
        ['Active Subgoal', state.active_subgoal || 'None'],
        ['Models', models || 'n/a'],
        ['Environment', JSON.stringify(state.environment || {})]
      ];
      document.getElementById('meta-grid').innerHTML = items.map(([key, value]) => `
        <div class="meta-card">
          <div class="key">${escapeHtml(key)}</div>
          <div class="value">${escapeHtml(value)}</div>
        </div>
      `).join('');
    }

    function renderPlan(state) {
      const summary = state.normalized_instruction
        ? `Command: ${state.normalized_instruction}`
        : 'No active plan.';
      document.getElementById('plan-summary').textContent = summary;
      const subgoals = state.subgoals || [];
      document.getElementById('subgoals').innerHTML = subgoals.map((subgoal) => `
        <div class="subgoal ${subgoal === state.active_subgoal ? 'active' : ''}">
          ${escapeHtml(subgoal)}
        </div>
      `).join('');
    }

    function renderEvents(state) {
      const events = [...(state.events || [])].reverse();
      document.getElementById('events').innerHTML = events.map((event) => `
        <details class="event" ${event.kind === 'session_finished' ? 'open' : ''}>
          <summary>
            <div class="event-top">
              <div>
                <div class="event-kind">${escapeHtml(event.kind)}</div>
                <div class="event-title">${escapeHtml(event.title)}</div>
              </div>
              <div class="muted">${escapeHtml(event.timestamp)}</div>
            </div>
            <p class="event-summary">${escapeHtml(event.summary)}</p>
          </summary>
          <pre>${escapeHtml(JSON.stringify(event.details || {}, null, 2))}</pre>
        </details>
      `).join('');
    }

    async function refresh() {
      const res = await fetch('/api/state');
      const state = await res.json();
      renderMeta(state);
      renderPlan(state);
      renderEvents(state);
    }

    document.getElementById('command-form').addEventListener('submit', async (event) => {
      event.preventDefault();
      const command = document.getElementById('command').value.trim();
      const voiceTranscript = document.getElementById('voice-transcript').value.trim();
      if (!command && !voiceTranscript) {
        return;
      }
      await postJson('/api/start', {command, voice_transcript: voiceTranscript});
      await refresh();
    });

    document.getElementById('pause').addEventListener('click', async () => {
      await postJson('/api/pause');
      await refresh();
    });
    document.getElementById('resume').addEventListener('click', async () => {
      await postJson('/api/resume');
      await refresh();
    });
    document.getElementById('stop').addEventListener('click', async () => {
      await postJson('/api/stop');
      await refresh();
    });

    refresh();
    setInterval(refresh, 1000);
  </script>
</body>
</html>
"""


class PlaygroundUIServer:
    def __init__(
        self,
        controller: PlaygroundAgentController,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
    ) -> None:
        self.controller = controller
        self.host = host
        self.port = port
        self._server: ThreadingHTTPServer | None = None

    def serve_forever(self) -> None:
        controller = self.controller

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path == "/" or self.path == "/index.html":
                    self._send_html(HTML_PAGE)
                    return
                if self.path == "/api/state":
                    self._send_json(controller.snapshot())
                    return
                self.send_error(HTTPStatus.NOT_FOUND, "not found")

            def do_POST(self) -> None:
                payload = self._read_json_body()
                if self.path == "/api/start":
                    command = str(payload.get("command", "")).strip()
                    voice_transcript = str(payload.get("voice_transcript", "")).strip()
                    if voice_transcript:
                        accepted = controller.start_voice_transcript(voice_transcript)
                    else:
                        accepted = controller.start_instruction(command)
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
                self.send_error(HTTPStatus.NOT_FOUND, "not found")

            def log_message(self, format: str, *args: Any) -> None:
                return

            def _read_json_body(self) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length", "0"))
                if length == 0:
                    return {}
                body = self.rfile.read(length).decode("utf-8")
                return json.loads(body)

            def _send_html(self, content: str) -> None:
                encoded = content.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def _send_json(self, payload: dict[str, Any]) -> None:
                encoded = json.dumps(payload).encode("utf-8")
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
