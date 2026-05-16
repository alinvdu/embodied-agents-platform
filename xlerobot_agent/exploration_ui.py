from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import time
from typing import Any, Callable, Protocol

from .exploration import ExplorationBackend
from .offload import OffloadClient


class ExplorationUIController(Protocol):
    def snapshot(self) -> dict[str, Any]:
        ...

    def start_explore(self, *, area: str, session: str | None = None, source: str = "operator") -> dict[str, Any]:
        ...

    def start_create_map(self, *, area: str, session: str, source: str = "operator") -> dict[str, Any]:
        ...

    def list_environment_memories(self) -> list[dict[str, Any]]:
        ...

    def create_environment_memory(self, memory_id: str, *, label: str | None = None) -> dict[str, Any]:
        ...

    def load_environment_memory(self, memory_id: str) -> dict[str, Any] | None:
        ...

    def start_navigation_session(self) -> dict[str, Any]:
        ...

    def stop_navigation_session(self) -> dict[str, Any]:
        ...

    def pause_task(self, task_id: str) -> dict[str, Any] | None:
        ...

    def resume_task(self, task_id: str) -> dict[str, Any] | None:
        ...

    def cancel_task(self, task_id: str | None = None) -> dict[str, Any] | None:
        ...

    def update_region(
        self,
        region_id: str,
        *,
        label: str | None = None,
        polygon_2d: list[list[float]] | None = None,
        default_waypoints: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        ...

    def create_region(
        self,
        *,
        label: str,
        polygon_2d: list[list[float]],
        default_waypoints: list[dict[str, Any]] | None = None,
        region_id: str | None = None,
        purpose: str | None = None,
    ) -> dict[str, Any] | None:
        ...

    def merge_regions(self, region_ids: list[str], *, new_label: str | None = None) -> dict[str, Any] | None:
        ...

    def split_region(self, region_id: str, *, polygons: list[list[list[float]]] | None = None) -> list[dict[str, Any]]:
        ...

    def set_named_place(self, *, name: str, pose: dict[str, Any], region_id: str | None = None) -> dict[str, Any] | None:
        ...

    def set_dock_pose(self, *, pose: dict[str, Any] | None = None) -> dict[str, Any] | None:
        ...

    def approve_map(self) -> dict[str, Any] | None:
        ...

    def update_occupancy_edits(
        self,
        *,
        task_id: str | None = None,
        mode: str,
        cells: list[dict[str, Any]],
    ) -> dict[str, Any]:
        ...

    def perform_scan(self) -> dict[str, Any]:
        ...

    def navigate_to_waypoint(self, *, pose: dict[str, Any]) -> dict[str, Any]:
        ...

    def preview_waypoint(self, *, pose: dict[str, Any]) -> dict[str, Any]:
        ...

    def relocalize_here(self) -> dict[str, Any]:
        ...


class LocalExplorationUIController:
    def __init__(
        self,
        backend: ExplorationBackend,
        *,
        waypoint_navigator: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        waypoint_previewer: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        scan_performer: Callable[[], dict[str, Any]] | None = None,
        relocalizer: Callable[[], dict[str, Any]] | None = None,
        navigation_session_starter: Callable[[], dict[str, Any]] | None = None,
        navigation_session_stopper: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self.backend = backend
        self.waypoint_navigator = waypoint_navigator
        self.waypoint_previewer = waypoint_previewer
        self.scan_performer = scan_performer
        self.relocalizer = relocalizer
        self.navigation_session_starter = navigation_session_starter
        self.navigation_session_stopper = navigation_session_stopper

    def snapshot(self) -> dict[str, Any]:
        return self.backend.snapshot()

    def start_explore(self, *, area: str, session: str | None = None, source: str = "operator") -> dict[str, Any]:
        return self.backend.start_explore(area=area, session=session, source=source)

    def start_create_map(self, *, area: str, session: str, source: str = "operator") -> dict[str, Any]:
        return self.backend.start_create_map(area=area, session=session, source=source)

    def list_environment_memories(self) -> list[dict[str, Any]]:
        return self.backend.list_environment_memories()

    def create_environment_memory(self, memory_id: str, *, label: str | None = None) -> dict[str, Any]:
        return self.backend.create_environment_memory(memory_id, label=label)

    def load_environment_memory(self, memory_id: str) -> dict[str, Any] | None:
        return self.backend.load_environment_memory(memory_id)

    def start_navigation_session(self) -> dict[str, Any]:
        if self.navigation_session_starter is None:
            return {"status": "unavailable", "reason": "Navigation-only sessions are not available for this backend."}
        return self.navigation_session_starter()

    def stop_navigation_session(self) -> dict[str, Any]:
        if self.navigation_session_stopper is None:
            return {"status": "unavailable", "reason": "Navigation-only sessions are not available for this backend."}
        return self.navigation_session_stopper()

    def pause_task(self, task_id: str) -> dict[str, Any] | None:
        return self.backend.pause_task(task_id)

    def resume_task(self, task_id: str) -> dict[str, Any] | None:
        return self.backend.resume_task(task_id)

    def cancel_task(self, task_id: str | None = None) -> dict[str, Any] | None:
        return self.backend.cancel_task(task_id)

    def update_region(
        self,
        region_id: str,
        *,
        label: str | None = None,
        polygon_2d: list[list[float]] | None = None,
        default_waypoints: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        return self.backend.update_region(
            region_id,
            label=label,
            polygon_2d=polygon_2d,
            default_waypoints=default_waypoints,
        )

    def create_region(
        self,
        *,
        label: str,
        polygon_2d: list[list[float]],
        default_waypoints: list[dict[str, Any]] | None = None,
        region_id: str | None = None,
        purpose: str | None = None,
    ) -> dict[str, Any] | None:
        return self.backend.create_region(
            label=label,
            polygon_2d=polygon_2d,
            default_waypoints=default_waypoints,
            region_id=region_id,
            purpose=purpose,
        )

    def merge_regions(self, region_ids: list[str], *, new_label: str | None = None) -> dict[str, Any] | None:
        return self.backend.merge_regions(region_ids, new_label=new_label)

    def split_region(self, region_id: str, *, polygons: list[list[list[float]]] | None = None) -> list[dict[str, Any]]:
        return self.backend.split_region(region_id, polygons)

    def set_named_place(self, *, name: str, pose: dict[str, Any], region_id: str | None = None) -> dict[str, Any] | None:
        return self.backend.set_named_place(name, pose, region_id=region_id)

    def set_dock_pose(self, *, pose: dict[str, Any] | None = None) -> dict[str, Any] | None:
        return self.backend.set_dock_pose(pose)

    def approve_map(self) -> dict[str, Any] | None:
        return self.backend.approve_current_map()

    def update_occupancy_edits(
        self,
        *,
        task_id: str | None = None,
        mode: str,
        cells: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return self.backend.update_occupancy_edits(task_id=task_id, mode=mode, cells=cells)

    def perform_scan(self) -> dict[str, Any]:
        if self.scan_performer is None:
            return {"status": "unavailable", "reason": "No live exploration session is attached to the review UI."}
        return self.scan_performer()

    def navigate_to_waypoint(self, *, pose: dict[str, Any]) -> dict[str, Any]:
        if self.waypoint_navigator is None:
            return {"status": "unavailable", "reason": "No live navigation session is attached to the review UI."}
        return self.waypoint_navigator(pose)

    def preview_waypoint(self, *, pose: dict[str, Any]) -> dict[str, Any]:
        if self.waypoint_previewer is None:
            return {"status": "unavailable", "reason": "No live navigation session is attached to the review UI."}
        return self.waypoint_previewer(pose)

    def relocalize_here(self) -> dict[str, Any]:
        if self.relocalizer is None:
            return {"status": "unavailable", "reason": "No live navigation session is attached to the review UI."}
        return self.relocalizer()


class RemoteExplorationUIController:
    def __init__(self, client: OffloadClient) -> None:
        self.client = client
        self.client.ensure_registered()

    def snapshot(self) -> dict[str, Any]:
        return self.client.mapping_snapshot()

    def start_explore(self, *, area: str, session: str | None = None, source: str = "operator") -> dict[str, Any]:
        return self.client.start_explore(area=area, session=session, source=source)

    def start_create_map(self, *, area: str, session: str, source: str = "operator") -> dict[str, Any]:
        return self.client.start_create_map(area=area, session=session, source=source)

    def list_environment_memories(self) -> list[dict[str, Any]]:
        return []

    def create_environment_memory(self, memory_id: str, *, label: str | None = None) -> dict[str, Any]:
        return {"status": "unavailable", "reason": "Remote memory creation is not implemented yet."}

    def load_environment_memory(self, memory_id: str) -> dict[str, Any] | None:
        return {"status": "unavailable", "reason": "Remote memory loading is not implemented yet."}

    def start_navigation_session(self) -> dict[str, Any]:
        return {"status": "unavailable", "reason": "Remote navigation-only sessions are not implemented yet."}

    def stop_navigation_session(self) -> dict[str, Any]:
        return {"status": "unavailable", "reason": "Remote navigation-only sessions are not implemented yet."}

    def pause_task(self, task_id: str) -> dict[str, Any] | None:
        return self.client.pause_mapping_task(task_id)

    def resume_task(self, task_id: str) -> dict[str, Any] | None:
        return self.client.resume_mapping_task(task_id)

    def cancel_task(self, task_id: str | None = None) -> dict[str, Any] | None:
        return self.client.cancel_mapping_task(task_id)

    def update_region(
        self,
        region_id: str,
        *,
        label: str | None = None,
        polygon_2d: list[list[float]] | None = None,
        default_waypoints: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        return self.client.update_mapping_region(
            region_id,
            label=label,
            polygon_2d=polygon_2d,
            default_waypoints=default_waypoints,
        )

    def create_region(
        self,
        *,
        label: str,
        polygon_2d: list[list[float]],
        default_waypoints: list[dict[str, Any]] | None = None,
        region_id: str | None = None,
        purpose: str | None = None,
    ) -> dict[str, Any] | None:
        return {"status": "unavailable", "reason": "Remote region creation is not implemented yet."}

    def merge_regions(self, region_ids: list[str], *, new_label: str | None = None) -> dict[str, Any] | None:
        return self.client.merge_mapping_regions(region_ids, new_label=new_label)

    def split_region(self, region_id: str, *, polygons: list[list[list[float]]] | None = None) -> list[dict[str, Any]]:
        return list(self.client.split_mapping_region(region_id, polygons=polygons).get("regions", []))

    def set_named_place(self, *, name: str, pose: dict[str, Any], region_id: str | None = None) -> dict[str, Any] | None:
        return self.client.set_named_place(name=name, pose=pose, region_id=region_id)

    def set_dock_pose(self, *, pose: dict[str, Any] | None = None) -> dict[str, Any] | None:
        return {"status": "unavailable", "reason": "Remote dock pose editing is not implemented yet."}

    def approve_map(self) -> dict[str, Any] | None:
        return self.client.approve_mapping_map()

    def update_occupancy_edits(
        self,
        *,
        task_id: str | None = None,
        mode: str,
        cells: list[dict[str, Any]],
    ) -> dict[str, Any]:
        raise NotImplementedError("Remote occupancy editing is not implemented yet.")

    def perform_scan(self) -> dict[str, Any]:
        return {"status": "unavailable", "reason": "Remote manual scanning is not implemented yet."}

    def navigate_to_waypoint(self, *, pose: dict[str, Any]) -> dict[str, Any]:
        return {"status": "unavailable", "reason": "Remote waypoint navigation is not implemented yet."}

    def preview_waypoint(self, *, pose: dict[str, Any]) -> dict[str, Any]:
        return {"status": "unavailable", "reason": "Remote waypoint previews are not implemented yet."}

    def relocalize_here(self) -> dict[str, Any]:
        return {"status": "unavailable", "reason": "Remote relocalization is not implemented yet."}


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>XLeRobot Exploration Review</title>
  <style>
    :root {
      --bg: #f7f4ec;
      --panel: rgba(255,255,255,0.8);
      --line: rgba(15, 23, 42, 0.12);
      --text: #172033;
      --muted: #52606d;
      --accent: #0f766e;
      --accent-2: #a16207;
      --danger: #b91c1c;
      --shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(15,118,110,0.16), transparent 30%),
        radial-gradient(circle at right, rgba(161,98,7,0.16), transparent 24%),
        linear-gradient(180deg, #faf7f0 0%, #efe7d8 100%);
      min-height: 100vh;
    }
    .shell {
      max-width: 1500px;
      margin: 0 auto;
      padding: 28px 18px 40px;
    }
    .hero {
      display: grid;
      gap: 8px;
      margin-bottom: 20px;
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
      font-size: clamp(30px, 5vw, 52px);
      line-height: 0.98;
      max-width: 13ch;
    }
    .grid {
      display: grid;
      grid-template-columns: 360px 1fr 340px;
      gap: 18px;
      align-items: start;
    }
    .grid > * {
      min-width: 0;
      position: relative;
    }
    .left-column { z-index: 3; }
    .center-column { z-index: 2; }
    .right-column { z-index: 3; }
    .panel {
      background: var(--panel);
      backdrop-filter: blur(16px);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 16px;
      box-shadow: var(--shadow);
    }
    .stack { display: grid; gap: 16px; }
    label {
      display: block;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 6px;
    }
    input, textarea, button {
      font: inherit;
    }
    input, textarea, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 10px 12px;
      background: rgba(255,255,255,0.92);
      color: var(--text);
    }
    textarea { min-height: 90px; resize: vertical; }
    button {
      border: 0;
      border-radius: 999px;
      padding: 11px 15px;
      cursor: pointer;
      font-weight: 600;
    }
    .primary { background: var(--accent); color: white; }
    .secondary { background: #fff7ed; color: var(--accent-2); }
    .danger { background: #fee2e2; color: var(--danger); }
    .button-row { display: flex; flex-wrap: wrap; gap: 8px; }
    .meta-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .meta-card {
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 12px;
      background: rgba(255,255,255,0.68);
    }
    .meta-key {
      font-size: 11px;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 5px;
    }
    .meta-value {
      font-size: 15px;
      font-weight: 600;
    }
    .map-shell {
      display: grid;
      gap: 12px;
    }
    #map-canvas {
      width: 100%;
      height: 720px;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: rgba(255,255,255,0.84);
    }
    .legend {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      font-size: 13px;
      color: var(--muted);
    }
    .legend span::before {
      content: "";
      display: inline-block;
      width: 11px;
      height: 11px;
      border-radius: 999px;
      margin-right: 6px;
      vertical-align: middle;
    }
    .free::before { background: rgba(148, 163, 184, 0.45); }
    .traj::before { background: #0f766e; }
    .region::before { background: rgba(180, 83, 9, 0.35); }
    .list {
      display: grid;
      gap: 10px;
      max-height: 260px;
      overflow: auto;
    }
    .list-card {
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 10px 12px;
      background: rgba(255,255,255,0.74);
      cursor: pointer;
    }
    .list-card.active {
      border-color: rgba(15,118,110,0.35);
      background: rgba(15,118,110,0.08);
    }
    .thumbs {
      display: grid;
      gap: 10px;
    }
    .thumbs img {
      width: 100%;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: white;
    }
    .muted { color: var(--muted); }
    @media (max-width: 1180px) {
      .grid { grid-template-columns: 1fr; }
      #map-canvas { height: 520px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="eyebrow">Exploration Review</div>
      <h1>Review, correct, and approve semantic maps.</h1>
      <div class="muted">__HERO_SUBTITLE__</div>
    </section>

    <div class="grid">
      <div class="stack left-column">
        <section class="panel" __CONTROL_PANEL_ATTR__>
          <div class="eyebrow">Control</div>
          <div __TASK_LAUNCH_ATTR__>
            <label for="area">Area</label>
            <input id="area" value="downstairs" />
            <label for="session">Session</label>
            <input id="session" value="house_v1" />
          </div>
          <div class="button-row" style="margin-top:10px;" __TASK_LAUNCH_ATTR__>
            <button class="primary" id="start-explore">Start Explore</button>
            <button class="secondary" id="start-map">Create Map</button>
          </div>
          <div class="button-row" style="margin-top:10px;" __TASK_STATE_ATTR__>
            <button class="secondary" id="pause-task">Pause</button>
            <button class="secondary" id="resume-task">Resume</button>
            <button class="danger" id="cancel-task">Cancel</button>
          </div>
          <div class="button-row" style="margin-top:10px;" __APPROVE_ATTR__>
            <button class="primary" id="approve-map">Approve Map</button>
          </div>
        </section>

        <section class="panel">
          <div class="eyebrow">Status</div>
          <div id="meta-grid" class="meta-grid"></div>
        </section>

        <section class="panel">
          <div class="eyebrow">Map Editing</div>
          <div class="button-row" style="margin-top:10px;">
            <button class="secondary" id="edit-block">Draw Wall</button>
            <button class="secondary" id="edit-clear">Erase Wall</button>
            <button class="secondary" id="edit-reset">Reset Cell</button>
            <button class="secondary" id="nav-preview">Preview Path</button>
            <button class="primary" id="nav-waypoint">Waypoint</button>
          </div>
          <div id="edit-mode-summary" class="muted" style="margin-top:10px;">Click or drag on the map to add or remove occupancy overrides.</div>
        </section>

        <section class="panel">
          <div class="eyebrow">Regions</div>
          <div id="region-list" class="list"></div>
        </section>
      </div>

      <section class="panel map-shell center-column">
        <div class="eyebrow">Map</div>
        <svg id="map-canvas" viewBox="0 0 1000 700"></svg>
        <div class="legend">
          <span class="free">occupancy</span>
          <span class="traj">trajectory</span>
          <span class="region">region overlay</span>
        </div>
      </section>

      <div class="stack right-column">
        <section class="panel" __DEVELOPER_PANEL_ATTR__>
          <div class="eyebrow">Selected Region</div>
          <div id="selected-summary" class="muted">Select a region from the map or list.</div>
          <label for="region-label">Label</label>
          <input id="region-label" />
          <label for="region-polygon">Polygon JSON</label>
          <textarea id="region-polygon"></textarea>
          <label for="region-waypoints">Waypoints JSON</label>
          <textarea id="region-waypoints"></textarea>
          <div class="button-row" style="margin-top:10px;">
            <button class="primary" id="save-region">Save Region</button>
            <button class="secondary" id="split-region">Split Region</button>
          </div>
        </section>

        <section class="panel" __DEVELOPER_PANEL_ATTR__>
          <div class="eyebrow">Merge Regions</div>
          <label for="merge-ids">Region IDs JSON</label>
          <textarea id="merge-ids">[]</textarea>
          <label for="merge-label">Merged Label</label>
          <input id="merge-label" />
          <div class="button-row" style="margin-top:10px;">
            <button class="secondary" id="merge-regions">Merge</button>
          </div>
        </section>

        <section class="panel" __DEVELOPER_PANEL_ATTR__>
          <div class="eyebrow">Named Place</div>
          <label for="place-name">Name</label>
          <input id="place-name" placeholder="kitchen_entry" />
          <label for="place-pose">Pose JSON</label>
          <textarea id="place-pose">{"x":0,"y":0,"yaw":0}</textarea>
          <div class="button-row" style="margin-top:10px;">
            <button class="secondary" id="save-place">Save Place</button>
          </div>
        </section>

        <section class="panel">
          <div class="eyebrow">Keyframes</div>
          <div id="keyframes" class="thumbs"></div>
        </section>

        <section class="panel" __DEVELOPER_PANEL_ATTR__>
          <div class="eyebrow">Guardrails</div>
          <div id="guardrail-list" class="list"></div>
        </section>
      </div>
    </div>
  </div>

  <script>
    let selectedRegionId = null;
    let currentState = null;
    let mapEditMode = 'block';
    let currentMapBounds = null;
    let isPaintingMap = false;
    let pendingPaintCells = new Map();
    let paintFlushTimer = null;
    let lastPaintedCellKey = null;
    let currentOccupancyCellStates = new Map();
    let lastManualWaypoint = null;
    let exploreStartRequested = false;
    let manualScanInFlight = false;

    async function postJson(url, payload) {
      const response = await fetch(url, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload || {})
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || ('Request failed: ' + response.status));
      }
      return response.json();
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;');
    }

    function mapBounds(map) {
      if (map && map.occupancy && map.occupancy.bounds) {
        return map.occupancy.bounds;
      }
      const points = [];
      (map?.regions || []).forEach((region) => (region.polygon_2d || []).forEach((point) => points.push(point)));
      if (!points.length) {
        return {min_x: 0, max_x: 10, min_y: 0, max_y: 8};
      }
      return {
        min_x: Math.min(...points.map((point) => point[0])),
        max_x: Math.max(...points.map((point) => point[0])),
        min_y: Math.min(...points.map((point) => point[1])),
        max_y: Math.max(...points.map((point) => point[1]))
      };
    }

    function mapViewport(bounds) {
      const width = Math.max(bounds.max_x - bounds.min_x, 1);
      const height = Math.max(bounds.max_y - bounds.min_y, 1);
      const pad = 36;
      const innerWidth = Math.max(1000 - pad * 2, 1);
      const innerHeight = Math.max(700 - pad * 2, 1);
      const scale = Math.min(innerWidth / width, innerHeight / height);
      const drawWidth = width * scale;
      const drawHeight = height * scale;
      return {
        scale,
        drawWidth,
        drawHeight,
        left: pad + (innerWidth - drawWidth) / 2,
        top: pad + (innerHeight - drawHeight) / 2,
      };
    }

    function makeProjector(bounds) {
      const viewport = mapViewport(bounds);
      return function(point) {
        const x = viewport.left + (point.x - bounds.min_x) * viewport.scale;
        const y = viewport.top + (bounds.max_y - point.y) * viewport.scale;
        return {x, y};
      };
    }

    function svgPointFromClient(svg, clientX, clientY) {
      const point = svg.createSVGPoint();
      point.x = clientX;
      point.y = clientY;
      const matrix = svg.getScreenCTM();
      if (!matrix) return {x: 0, y: 0};
      return point.matrixTransform(matrix.inverse());
    }

    function worldFromSvgViewPoint(bounds, svgX, svgY) {
      const viewport = mapViewport(bounds);
      const right = viewport.left + viewport.drawWidth;
      const bottom = viewport.top + viewport.drawHeight;
      if (svgX < viewport.left || svgX > right || svgY < viewport.top || svgY > bottom) {
        return null;
      }
      return {
        x: bounds.min_x + (svgX - viewport.left) / viewport.scale,
        y: bounds.max_y - (svgY - viewport.top) / viewport.scale,
      };
    }

    function cellFromMapEvent(map, event) {
      const world = worldFromMapEvent(map, event);
      if (!world) return null;
      const resolution = map.occupancy?.resolution || 0.5;
      const originX = map.occupancy?.bounds?.min_x || 0;
      const originY = map.occupancy?.bounds?.min_y || 0;
      const cell = {
        cell_x: Math.floor((world.x - originX) / resolution),
        cell_y: Math.floor((world.y - originY) / resolution),
      };
      cell.key = `${cell.cell_x}:${cell.cell_y}`;
      return cell;
    }

    function worldFromMapEvent(map, event) {
      const svg = document.getElementById('map-canvas');
      const bounds = currentMapBounds || mapBounds(map);
      const point = svgPointFromClient(svg, event.clientX, event.clientY);
      return worldFromSvgViewPoint(bounds, point.x, point.y);
    }

    function pathLengthMeters(points) {
      let length = 0;
      let previous = null;
      for (const point of points || []) {
        const x = Number(point?.x);
        const y = Number(point?.y);
        if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
        if (previous) {
          length += Math.hypot(x - previous.x, y - previous.y);
        }
        previous = {x, y};
      }
      return length;
    }

    function previewPathLengthMeters(robotPose, previewPlan) {
      const pathPoses = (previewPlan?.path_poses || []).filter((pose) => {
        return Number.isFinite(Number(pose?.x)) && Number.isFinite(Number(pose?.y));
      });
      const measuredPoses = [];
      if (robotPose && Number.isFinite(Number(robotPose.x)) && Number.isFinite(Number(robotPose.y))) {
        measuredPoses.push(robotPose);
      }
      measuredPoses.push(...pathPoses);
      const measuredLength = pathLengthMeters(measuredPoses);
      if (measuredLength > 0) return measuredLength;
      const plannedLength = Number(previewPlan?.path_length_m);
      return Number.isFinite(plannedLength) && plannedLength > 0 ? plannedLength : null;
    }

    function formatCentimeters(meters) {
      if (!Number.isFinite(Number(meters))) return '';
      return `${Math.round(Number(meters) * 100)} cm`;
    }

    function shouldPaintCell(cell) {
      if (!cell) return false;
      if (mapEditMode !== 'clear') return true;
      const state = currentOccupancyCellStates.get(cell.key);
      return !!state && (state.state === 'occupied' || state.manual_override === 'blocked');
    }

    function enqueuePaintCell(cell) {
      if (!shouldPaintCell(cell)) return;
      pendingPaintCells.set(cell.key, {cell_x: cell.cell_x, cell_y: cell.cell_y});
      if (!paintFlushTimer) {
        paintFlushTimer = setTimeout(flushPaintCells, 80);
      }
    }

    async function flushPaintCells() {
      if (paintFlushTimer) {
        clearTimeout(paintFlushTimer);
        paintFlushTimer = null;
      }
      const cells = Array.from(pendingPaintCells.values());
      pendingPaintCells.clear();
      if (!cells.length) return;
      await postJson('/api/map/edit', {
        task_id: currentState?.active_task?.task_id || null,
        mode: mapEditMode,
        cells,
      });
      await refresh();
    }

    function renderMeta(state) {
      const task = state.active_task;
      const map = state.current_map;
      const items = [
        ['Mode', state.mode || 'unknown'],
        ['Map', map ? map.map_id : 'none'],
        ['Approved', map && map.approved ? 'yes' : 'no'],
        ['Regions', map ? (map.regions || []).length : 0],
        ['Coverage', map ? String(map.coverage || 0) : '0'],
        ['Task', task ? (task.tool_id + ' ' + task.state) : 'idle'],
        ['Paused', task && task.paused ? 'yes' : 'no']
      ];
      if (map && map.automatic_semantic_waypoints) {
        items.splice(4, 0, ['Automatic Semantic Places', ((map.semantic_memory || {}).named_places || []).length]);
      }
      document.getElementById('meta-grid').innerHTML = items.map(([key, value]) => `
        <div class="meta-card">
          <div class="meta-key">${escapeHtml(key)}</div>
          <div class="meta-value">${escapeHtml(value)}</div>
        </div>
      `).join('');
      renderLaunchControls(state);
    }

    function renderLaunchControls(state) {
      const startExplore = document.getElementById('start-explore');
      const startMap = document.getElementById('start-map');
      if (!startExplore) return;
      const scanAvailable = exploreStartRequested || !!state?.active_task;
      const scanReady = !!state?.active_task && !!state?.current_map && !manualScanInFlight;
      startExplore.textContent = manualScanInFlight ? 'Scanning...' : scanAvailable ? 'Perform Scan' : 'Start Explore';
      startExplore.dataset.mode = scanAvailable ? 'scan' : 'start';
      startExplore.disabled = scanAvailable && !scanReady;
      if (startMap) {
        startMap.disabled = scanAvailable;
        startMap.style.display = scanAvailable ? 'none' : '';
      }
    }

    function renderRegions(state) {
      const regions = state.current_map?.regions || [];
      document.getElementById('region-list').innerHTML = regions.map((region) => `
        <div class="list-card ${region.region_id === selectedRegionId ? 'active' : ''}" data-region-id="${escapeHtml(region.region_id)}">
          <strong>${escapeHtml(region.label)}</strong><br/>
          <span class="muted">${escapeHtml(region.region_id)} · ${escapeHtml(String(region.confidence))}</span>
        </div>
      `).join('');
      for (const element of document.querySelectorAll('[data-region-id]')) {
        element.addEventListener('click', () => {
          selectedRegionId = element.getAttribute('data-region-id');
          refreshSelectedRegion();
          renderRegions(currentState);
          renderMap(currentState);
        });
      }
    }

    function renderExploration(state) {
      const map = state.current_map || {};
      const edits = ((map.artifacts || {}).manual_occupancy_edits || {});
      const blocked = (edits.blocked_cells || []).length;
      const cleared = (edits.cleared_cells || []).length;
      const verb = mapEditMode === 'block'
        ? 'draw occupied wall cells'
        : mapEditMode === 'clear'
          ? 'erase wall cells into free space'
          : mapEditMode === 'reset'
            ? 'remove manual overrides'
            : mapEditMode === 'preview'
              ? 'click once to preview Nav2 path without moving'
              : 'click once to send a Nav2 waypoint';
      document.getElementById('edit-mode-summary').textContent = `Map mode: ${mapEditMode} (${verb}). Manual walls ${blocked}, manual clears ${cleared}.`;
      const guardrails = ((map.artifacts || {}).guardrail_events || []).slice(-12).reverse();
      const guardrailElement = document.getElementById('guardrail-list');
      if (guardrailElement) {
        guardrailElement.innerHTML = guardrails.map((event) => `
          <div class="list-card">
            <strong>${escapeHtml(event.type || 'event')}</strong><br/>
            <span class="muted">${escapeHtml(JSON.stringify(event))}</span>
          </div>
        `).join('') || '<div class="muted">No guardrail events.</div>';
      }
    }

    function renderMap(state) {
      const svg = document.getElementById('map-canvas');
      const map = state.current_map;
      if (!map) {
        svg.innerHTML = '<text x="40" y="60" fill="#52606d" font-size="22">No map yet.</text>';
        return;
      }
      const bounds = mapBounds(map);
      currentMapBounds = bounds;
      const project = makeProjector(bounds);
      currentOccupancyCellStates = new Map();
      const occupancy = (map.occupancy?.cells || []).map((cell) => {
        const resolution = map.occupancy.resolution || 0.5;
        const originX = map.occupancy?.bounds?.min_x || 0;
        const originY = map.occupancy?.bounds?.min_y || 0;
        const cellX = Math.floor((cell.x - originX) / resolution);
        const cellY = Math.floor((cell.y - originY) / resolution);
        currentOccupancyCellStates.set(`${cellX}:${cellY}`, {
          state: cell.state,
          manual_override: cell.manual_override || null,
        });
        const p = project({x: cell.x, y: cell.y});
        const p2 = project({x: cell.x + resolution, y: cell.y + resolution});
        const fill = cell.manual_override === 'blocked'
          ? 'rgba(15,23,42,0.85)'
          : cell.manual_override === 'cleared'
            ? 'rgba(148,163,184,0.22)'
            : cell.state === 'occupied'
              ? 'rgba(15,23,42,0.55)'
              : cell.state === 'free'
                ? 'rgba(148,163,184,0.22)'
                : 'rgba(148,163,184,0.10)';
        return `<rect x="${p.x}" y="${p2.y}" width="${Math.max(2, p2.x - p.x)}" height="${Math.max(2, p.y - p2.y)}" fill="${fill}" />`;
      }).join('');
      const trajectory = (map.trajectory || []).map((point) => {
        const p = project(point);
        return `${p.x},${p.y}`;
      }).join(' ');
      const regions = (map.regions || []).map((region, index) => {
        const polygon = (region.polygon_2d || []).map((point) => {
          const p = project({x: point[0], y: point[1]});
          return `${p.x},${p.y}`;
        }).join(' ');
        const centroid = project(region.centroid || {x: 0, y: 0});
        const active = region.region_id === selectedRegionId;
        const hue = 24 + index * 47;
        const fill = active ? `hsla(${hue}, 72%, 45%, 0.35)` : `hsla(${hue}, 62%, 52%, 0.18)`;
        const stroke = active ? '#0f766e' : `hsla(${hue}, 62%, 42%, 0.62)`;
        return `
          <polygon data-region-id="${escapeHtml(region.region_id)}" points="${polygon}" fill="${fill}" stroke="${stroke}" stroke-width="${active ? 4 : 2}" />
          <text x="${centroid.x}" y="${centroid.y}" text-anchor="middle" font-size="16" fill="#172033" font-weight="600">${escapeHtml(region.label)}</text>
        `;
      }).join('');
      const namedPlaces = (map.named_places || []).filter((place) => place.source !== 'semantic_memory').map((place) => {
        const p = project(place.pose || {x: 0, y: 0});
        return `
          <circle cx="${p.x}" cy="${p.y}" r="5" fill="#0f766e" />
          <text x="${p.x + 8}" y="${p.y - 8}" font-size="12" fill="#0f766e">${escapeHtml(place.name)}</text>
        `;
      }).join('');
      const semanticMemory = map.automatic_semantic_waypoints ? (map.semantic_memory || {}) : {};
      const semanticEvidence = (semanticMemory.evidence || []).map((ev) => {
        const p = project(ev.evidence_pose || {x: 0, y: 0});
        return `<circle cx="${p.x}" cy="${p.y}" r="5" fill="#7c3aed" opacity="0.72"><title>${escapeHtml(ev.label_hint || '')} evidence</title></circle>`;
      }).join('');
      const semanticPlaces = (semanticMemory.named_places || []).map((place) => {
        const anchor = project(place.anchor_pose || {x: 0, y: 0});
        const evidence = place.evidence_pose ? project(place.evidence_pose) : null;
        return `
          ${evidence ? `<line x1="${evidence.x}" y1="${evidence.y}" x2="${anchor.x}" y2="${anchor.y}" stroke="#7c3aed" stroke-width="1.5" stroke-dasharray="3 5" opacity="0.58" />` : ''}
          <rect x="${anchor.x - 7}" y="${anchor.y - 7}" width="14" height="14" rx="3" fill="#7c3aed" opacity="0.9" />
          <text x="${anchor.x + 10}" y="${anchor.y + 4}" font-size="12" fill="#4c1d95" font-weight="700">${escapeHtml(place.label || '')}</text>
        `;
      }).join('');
      const manualWaypoint = lastManualWaypoint ? (() => {
        const p = project(lastManualWaypoint);
        return `
          <circle cx="${p.x}" cy="${p.y}" r="9" fill="#2563eb" opacity="0.88" />
          <circle cx="${p.x}" cy="${p.y}" r="15" fill="none" stroke="#2563eb" stroke-width="2" opacity="0.6" />
          <text x="${p.x + 12}" y="${p.y + 4}" font-size="12" fill="#1d4ed8" font-weight="700">waypoint</text>
        `;
      })() : '';
      const preview = ((map.artifacts || {}).nav2_preview || {});
      const previewPlan = preview.plan || null;
      const previewPathPoses = previewPlan?.path_poses || [];
      const previewPath = previewPathPoses.map((pose) => {
        const p = project(pose);
        return `${p.x},${p.y}`;
      }).join(' ');
      const previewGoal = preview.requested_pose || null;
      const robotPose = map.robot_pose || (map.trajectory || []).slice(-1)[0] || null;
      const previewDistanceM = previewPathLengthMeters(robotPose, previewPlan);
      const previewGoalMarkup = previewGoal ? (() => {
        const p = project(previewGoal);
        const distanceLabel = previewDistanceM == null ? '' : `path ${formatCentimeters(previewDistanceM)}`;
        return `
          <circle cx="${p.x}" cy="${p.y}" r="8" fill="#f59e0b" opacity="0.92" />
          <text x="${p.x + 12}" y="${p.y - 8}" font-size="12" fill="#92400e" font-weight="700">preview</text>
          ${distanceLabel ? `<text x="${p.x + 12}" y="${p.y + 8}" font-size="12" fill="#92400e" font-weight="700">${escapeHtml(distanceLabel)}</text>` : ''}
        `;
      })() : '';
      const robot = robotPose ? project(robotPose) : null;
      const headingLength = (map.occupancy?.resolution || 0.5) * 2.5;
      const robotHeading = robotPose ? project({
        x: robotPose.x + Math.cos(Number(robotPose.yaw || 0)) * headingLength,
        y: robotPose.y + Math.sin(Number(robotPose.yaw || 0)) * headingLength,
      }) : null;
      svg.innerHTML = `
        <rect x="0" y="0" width="1000" height="700" fill="rgba(255,255,255,0.92)" />
        ${occupancy}
        <polyline points="${trajectory}" fill="none" stroke="#0f766e" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" />
        ${regions}
        ${namedPlaces}
        ${semanticEvidence}
        ${semanticPlaces}
        ${previewPath ? `<polyline points="${previewPath}" fill="none" stroke="#f59e0b" stroke-width="5" stroke-linecap="round" stroke-linejoin="round" stroke-dasharray="10 7" opacity="0.95" />` : ''}
        ${previewGoalMarkup}
        ${manualWaypoint}
        ${robot ? `<circle cx="${robot.x}" cy="${robot.y}" r="11" fill="#a52820" />` : ''}
        ${robot && robotHeading ? `<line x1="${robot.x}" y1="${robot.y}" x2="${robotHeading.x}" y2="${robotHeading.y}" stroke="#6d0f0a" stroke-width="4.5" stroke-linecap="round" />` : ''}
        ${robot && robotHeading ? `<circle cx="${robotHeading.x}" cy="${robotHeading.y}" r="4" fill="#6d0f0a" />` : ''}
        ${robot ? `<text x="${robot.x + 12}" y="${robot.y - 12}" font-size="13" fill="#a52820" font-weight="700">robot</text>` : ''}
      `;
      for (const element of svg.querySelectorAll('[data-region-id]')) {
        element.addEventListener('click', () => {
          selectedRegionId = element.getAttribute('data-region-id');
          refreshSelectedRegion();
          renderRegions(currentState);
          renderMap(currentState);
        });
      }
      svg.onpointerdown = (event) => {
        if (!currentMapBounds) return;
        event.preventDefault();
        if (mapEditMode === 'waypoint' || mapEditMode === 'preview') {
          const world = worldFromMapEvent(map, event);
          if (!world) return;
          const robotPose = map.robot_pose || (map.trajectory || []).slice(-1)[0] || {};
          lastManualWaypoint = {x: world.x, y: world.y, yaw: Number(robotPose.yaw || 0)};
          renderMap(currentState);
          const endpoint = mapEditMode === 'preview' ? '/api/nav/preview' : '/api/nav/waypoint';
          postJson(endpoint, {pose: lastManualWaypoint})
            .then((response) => {
              lastManualWaypoint = response.normalized_pose || response.requested_pose || lastManualWaypoint;
              if (response.map) {
                currentState.current_map = response.map;
              }
              renderMap(currentState);
              if (response.status && response.status !== 'succeeded') {
                alert((response.status || 'failed') + ': ' + (response.reason || 'Waypoint request failed.'));
              }
              return refresh();
            })
            .catch((error) => {
              alert(error.message || String(error));
            });
          return;
        }
        const cell = cellFromMapEvent(map, event);
        if (!cell) return;
        isPaintingMap = true;
        lastPaintedCellKey = null;
        lastPaintedCellKey = cell.key;
        enqueuePaintCell(cell);
      };
      svg.onpointermove = (event) => {
        if (!isPaintingMap) return;
        event.preventDefault();
        const cell = cellFromMapEvent(map, event);
        if (!cell) return;
        if (cell.key === lastPaintedCellKey) return;
        lastPaintedCellKey = cell.key;
        enqueuePaintCell(cell);
      };
      svg.onpointerup = async (event) => {
        if (!isPaintingMap) return;
        event.preventDefault();
        isPaintingMap = false;
        lastPaintedCellKey = null;
        await flushPaintCells();
      };
      svg.onpointerleave = async () => {
        if (!isPaintingMap) return;
        isPaintingMap = false;
        lastPaintedCellKey = null;
        await flushPaintCells();
      };
    }

    function refreshSelectedRegion() {
      const map = currentState?.current_map;
      const region = (map?.regions || []).find((item) => item.region_id === selectedRegionId);
      const summary = document.getElementById('selected-summary');
      const keyframes = document.getElementById('keyframes');
      if (!region) {
        summary.textContent = 'Select a region from the map or list.';
        document.getElementById('region-label').value = '';
        document.getElementById('region-polygon').value = '';
        document.getElementById('region-waypoints').value = '';
        keyframes.innerHTML = '';
        return;
      }
      summary.textContent = `${region.label} (${region.region_id})`;
      document.getElementById('region-label').value = region.label || '';
      document.getElementById('region-polygon').value = JSON.stringify(region.polygon_2d || [], null, 2);
      document.getElementById('region-waypoints').value = JSON.stringify(region.default_waypoints || [], null, 2);
      const framesById = Object.fromEntries((map.keyframes || []).map((item) => [item.frame_id, item]));
      keyframes.innerHTML = (region.representative_keyframes || []).map((frameId) => {
        const frame = framesById[frameId];
        if (!frame) return '';
        return `
          <div>
            <img src="${frame.thumbnail_data_url}" alt="${escapeHtml(frame.frame_id)}" />
            <div class="muted">${escapeHtml(frame.description)}</div>
          </div>
        `;
      }).join('');
      if (region.centroid) {
        document.getElementById('place-pose').value = JSON.stringify({x: region.centroid.x, y: region.centroid.y, yaw: 0}, null, 2);
      }
    }

    async function refresh() {
      const response = await fetch('/api/state');
      currentState = await response.json();
      renderMeta(currentState);
      renderExploration(currentState);
      renderRegions(currentState);
      renderMap(currentState);
      refreshSelectedRegion();
    }

    document.getElementById('start-explore').addEventListener('click', async () => {
      const button = document.getElementById('start-explore');
      if (button.dataset.mode === 'scan') {
        manualScanInFlight = true;
        renderLaunchControls(currentState || {});
        try {
          const response = await postJson('/api/scan/perform', {});
          if (response.status && !['succeeded', 'started'].includes(response.status)) {
            alert((response.status || 'failed') + ': ' + (response.reason || response.message || 'Manual scan request failed.'));
          }
        } finally {
          manualScanInFlight = false;
        }
      } else {
        exploreStartRequested = true;
        renderLaunchControls(currentState || {});
        try {
          await postJson('/api/explore/start', {
            area: document.getElementById('area').value.trim(),
            session: document.getElementById('session').value.trim() || null
          });
        } catch (error) {
          exploreStartRequested = false;
          renderLaunchControls(currentState || {});
          throw error;
        }
      }
      await refresh();
    });
    document.getElementById('start-map').addEventListener('click', async () => {
      await postJson('/api/create_map/start', {
        area: document.getElementById('area').value.trim(),
        session: document.getElementById('session').value.trim()
      });
      await refresh();
    });
    document.getElementById('pause-task').addEventListener('click', async () => {
      const taskId = currentState?.active_task?.task_id;
      if (!taskId) return;
      await postJson('/api/task/pause', {task_id: taskId});
      await refresh();
    });
    document.getElementById('resume-task').addEventListener('click', async () => {
      const taskId = currentState?.active_task?.task_id;
      if (!taskId) return;
      await postJson('/api/task/resume', {task_id: taskId});
      await refresh();
    });
    document.getElementById('cancel-task').addEventListener('click', async () => {
      const taskId = currentState?.active_task?.task_id || null;
      await postJson('/api/task/cancel', {task_id: taskId});
      await refresh();
    });
    document.getElementById('approve-map').addEventListener('click', async () => {
      await postJson('/api/approve');
      await refresh();
    });
    document.getElementById('edit-block').addEventListener('click', () => {
      mapEditMode = 'block';
      renderExploration(currentState || {});
    });
    document.getElementById('edit-clear').addEventListener('click', () => {
      mapEditMode = 'clear';
      renderExploration(currentState || {});
    });
    document.getElementById('edit-reset').addEventListener('click', () => {
      mapEditMode = 'reset';
      renderExploration(currentState || {});
    });
    document.getElementById('nav-preview').addEventListener('click', () => {
      mapEditMode = 'preview';
      renderExploration(currentState || {});
    });
    document.getElementById('nav-waypoint').addEventListener('click', () => {
      mapEditMode = 'waypoint';
      renderExploration(currentState || {});
    });
    document.getElementById('save-region').addEventListener('click', async () => {
      if (!selectedRegionId) return;
      await postJson('/api/region/update', {
        region_id: selectedRegionId,
        label: document.getElementById('region-label').value.trim(),
        polygon_2d: JSON.parse(document.getElementById('region-polygon').value || '[]'),
        default_waypoints: JSON.parse(document.getElementById('region-waypoints').value || '[]')
      });
      await refresh();
    });
    document.getElementById('split-region').addEventListener('click', async () => {
      if (!selectedRegionId) return;
      await postJson('/api/region/split', {region_id: selectedRegionId});
      await refresh();
    });
    document.getElementById('merge-regions').addEventListener('click', async () => {
      await postJson('/api/regions/merge', {
        region_ids: JSON.parse(document.getElementById('merge-ids').value || '[]'),
        new_label: document.getElementById('merge-label').value.trim() || null
      });
      selectedRegionId = null;
      await refresh();
    });
    document.getElementById('save-place').addEventListener('click', async () => {
      await postJson('/api/named_place', {
        name: document.getElementById('place-name').value.trim(),
        pose: JSON.parse(document.getElementById('place-pose').value || '{}'),
        region_id: selectedRegionId
      });
      await refresh();
    });

    refresh();
    setInterval(refresh, 1000);
  </script>
</body>
</html>
"""


class ExplorationReviewServer:
    def __init__(
        self,
        controller: ExplorationUIController,
        *,
        host: str = "127.0.0.1",
        port: int = 8770,
        allow_task_controls: bool = True,
        allow_task_launch_controls: bool | None = None,
        allow_task_state_controls: bool | None = None,
        allow_map_approval: bool = True,
        ui_flavor: str = "user",
    ) -> None:
        self.controller = controller
        self.host = host
        self.port = port
        self.allow_task_controls = allow_task_controls
        self.allow_task_launch_controls = (
            allow_task_controls if allow_task_launch_controls is None else allow_task_launch_controls
        )
        self.allow_task_state_controls = (
            allow_task_controls if allow_task_state_controls is None else allow_task_state_controls
        )
        self.allow_map_approval = allow_map_approval
        self.ui_flavor = ui_flavor
        self._server: ThreadingHTTPServer | None = None

    def _html_page(self) -> str:
        subtitle = (
            "Live map progress, pause/resume controls, manual wall editing, semantic room editing, and final approval all happen here."
            if (self.allow_task_controls or self.allow_task_state_controls or self.allow_task_launch_controls)
            else "Post-run map review, region correction, waypoint edits, and approval happen here."
        )
        show_control_panel = (
            self.allow_task_controls
            or self.allow_task_launch_controls
            or self.allow_task_state_controls
            or self.allow_map_approval
        )
        control_attr = "" if show_control_panel else 'style="display:none;"'
        launch_attr = "" if self.allow_task_launch_controls else 'style="display:none;"'
        task_state_attr = "" if self.allow_task_state_controls else 'style="display:none;"'
        approve_attr = "" if self.allow_map_approval else 'style="display:none;"'
        developer_attr = "" if self.ui_flavor == "developer" else 'style="display:none;"'
        return (
            HTML_PAGE.replace("__HERO_SUBTITLE__", subtitle)
            .replace("__CONTROL_PANEL_ATTR__", control_attr)
            .replace("__TASK_LAUNCH_ATTR__", launch_attr)
            .replace("__TASK_STATE_ATTR__", task_state_attr)
            .replace("__APPROVE_ATTR__", approve_attr)
            .replace("__DEVELOPER_PANEL_ATTR__", developer_attr)
        )

    def serve_in_background(self) -> Any:
        import threading

        thread = threading.Thread(
            target=self.serve_forever,
            name="exploration_review_http",
            daemon=True,
        )
        thread.start()
        return thread

    def serve_forever(self) -> None:
        controller = self.controller

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path == "/" or self.path == "/index.html":
                    self._send_html(self.server.codex_html_page())
                    return
                if self.path == "/api/state":
                    self._send_json(controller.snapshot())
                    return
                if self.path == "/api/memories":
                    self._send_json({"memories": controller.list_environment_memories()})
                    return
                self.send_error(HTTPStatus.NOT_FOUND, "not found")

            def do_POST(self) -> None:
                payload = self._read_json_body()
                if self.path == "/api/explore/start":
                    self._send_json(
                        controller.start_explore(
                            area=str(payload.get("area", "workspace")),
                            session=payload.get("session"),
                            source="operator",
                        )
                    )
                    return
                if self.path == "/api/create_map/start":
                    session = str(payload.get("session") or f"map_{int(time.time())}")
                    self._send_json(
                        controller.start_create_map(
                            area=str(payload.get("area", "workspace")),
                            session=session,
                            source="operator",
                        )
                    )
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
                if self.path == "/api/memory/load":
                    response = controller.load_environment_memory(str(payload.get("memory_id") or ""))
                    self._send_json(response or {"status": "missing"})
                    return
                if self.path == "/api/nav/session/start":
                    self._send_json(controller.start_navigation_session())
                    return
                if self.path == "/api/nav/session/stop":
                    self._send_json(controller.stop_navigation_session())
                    return
                if self.path == "/api/task/pause":
                    response = controller.pause_task(str(payload.get("task_id")))
                    self._send_json(response or {"status": "missing"})
                    return
                if self.path == "/api/task/resume":
                    response = controller.resume_task(str(payload.get("task_id")))
                    self._send_json(response or {"status": "missing"})
                    return
                if self.path == "/api/task/cancel":
                    response = controller.cancel_task(payload.get("task_id"))
                    self._send_json(response or {"status": "missing"})
                    return
                if self.path == "/api/region/update":
                    response = controller.update_region(
                        str(payload.get("region_id")),
                        label=payload.get("label"),
                        polygon_2d=payload.get("polygon_2d"),
                        default_waypoints=payload.get("default_waypoints"),
                    )
                    self._send_json(response or {"status": "missing"})
                    return
                if self.path == "/api/region/create":
                    response = controller.create_region(
                        label=str(payload.get("label", "region")),
                        polygon_2d=list(payload.get("polygon_2d", [])),
                        default_waypoints=payload.get("default_waypoints"),
                        region_id=payload.get("region_id"),
                        purpose=payload.get("purpose"),
                    )
                    self._send_json(response or {"status": "missing"})
                    return
                if self.path == "/api/region/split":
                    response = controller.split_region(
                        str(payload.get("region_id")),
                        polygons=payload.get("polygons"),
                    )
                    self._send_json({"regions": response})
                    return
                if self.path == "/api/regions/merge":
                    response = controller.merge_regions(
                        list(payload.get("region_ids", [])),
                        new_label=payload.get("new_label"),
                    )
                    self._send_json(response or {"status": "missing"})
                    return
                if self.path == "/api/named_place":
                    response = controller.set_named_place(
                        name=str(payload.get("name")),
                        pose=dict(payload.get("pose", {})),
                        region_id=payload.get("region_id"),
                    )
                    self._send_json(response or {"status": "missing"})
                    return
                if self.path == "/api/dock_pose":
                    pose = payload.get("pose")
                    response = controller.set_dock_pose(
                        pose=dict(pose) if isinstance(pose, dict) else None,
                    )
                    self._send_json(response or {"status": "missing"})
                    return
                if self.path == "/api/map/edit":
                    response = controller.update_occupancy_edits(
                        task_id=payload.get("task_id"),
                        mode=str(payload.get("mode", "block")),
                        cells=list(payload.get("cells", [])),
                    )
                    self._send_json(response)
                    return
                if self.path == "/api/scan/perform":
                    self._send_json(controller.perform_scan())
                    return
                if self.path == "/api/nav/waypoint":
                    response = controller.navigate_to_waypoint(
                        pose=dict(payload.get("pose", {})),
                    )
                    self._send_json(response)
                    return
                if self.path == "/api/nav/preview":
                    response = controller.preview_waypoint(
                        pose=dict(payload.get("pose", {})),
                    )
                    self._send_json(response)
                    return
                if self.path == "/api/nav/relocalize":
                    self._send_json(controller.relocalize_here())
                    return
                if self.path == "/api/approve":
                    self._send_json(controller.approve_map() or {"status": "missing"})
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
        setattr(self._server, "codex_html_page", self._html_page)
        self._server.serve_forever()

    def shutdown(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
