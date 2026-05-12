import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Bot,
  Check,
  Eraser,
  Eye,
  Home,
  Map,
  Navigation,
  Pause,
  PenLine,
  Play,
  RotateCcw,
  Save,
  Send,
  Square,
} from "lucide-react";
import "./styles.css";

const AGENT_BASE = "/agent-api";
const EXPLORATION_BASE = "/exploration-api";
const VIEW_W = 1000;
const VIEW_H = 700;
const PAD = 34;

function App() {
  const [view, setView] = useState("agent");

  return (
    <div className="app-shell">
      <aside className="rail">
        <div className="brand">
          <div className="brand-mark">42</div>
          <div>
            <h1>Robot42</h1>
            <span>local robot console</span>
          </div>
        </div>
        <button className={view === "agent" ? "nav active" : "nav"} onClick={() => setView("agent")}>
          <Bot size={18} /> Agent
        </button>
        <button className={view === "exploration" ? "nav active" : "nav"} onClick={() => setView("exploration")}>
          <Map size={18} /> Exploration
        </button>
      </aside>
      <main className="workspace">{view === "agent" ? <AgentChat /> : <ExplorationConsole />}</main>
    </div>
  );
}

function AgentChat() {
  const [state, setState] = useState(null);
  const [command, setCommand] = useState("");
  const [messages, setMessages] = useState([
    { role: "assistant", text: "Ready." },
  ]);
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    try {
      const next = await apiGet(AGENT_BASE, "/api/state");
      setState(next);
      setError("");
    } catch (err) {
      setError(err.message || String(err));
    }
  }, []);

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, 1200);
    return () => clearInterval(timer);
  }, [refresh]);

  const submit = async (event) => {
    event.preventDefault();
    const text = command.trim();
    if (!text) return;
    setMessages((items) => [...items, { role: "user", text }]);
    setCommand("");
    try {
      await apiPost(AGENT_BASE, "/api/start", { command: text });
      await refresh();
    } catch (err) {
      setError(err.message || String(err));
      setMessages((items) => [...items, { role: "assistant", text: `Blocked: ${err.message || err}` }]);
    }
  };

  const events = state?.report?.events || state?.events || [];
  const lastEvent = events.slice(-1)[0];

  return (
    <section className="agent-grid">
      <div className="chat-panel">
        <div className="section-head">
          <span>Agent</span>
          <StatusPill value={state?.status || "offline"} />
        </div>
        <div className="chat-log">
          {messages.map((message, index) => (
            <div className={`message ${message.role}`} key={`${message.role}-${index}`}>{message.text}</div>
          ))}
          {lastEvent ? <div className="message assistant subtle">{eventText(lastEvent)}</div> : null}
        </div>
        <form className="composer" onSubmit={submit}>
          <input value={command} onChange={(event) => setCommand(event.target.value)} placeholder="Ask Robot42" />
          <button className="icon primary" type="submit" title="Send"><Send size={18} /></button>
        </form>
        {error ? <div className="error-line">{error}</div> : null}
      </div>
      <div className="side-stack">
        <Panel title="Run">
          <div className="toolbar">
            <button onClick={() => apiPost(AGENT_BASE, "/api/pause", {}).then(refresh)}><Pause size={16} /> Pause</button>
            <button onClick={() => apiPost(AGENT_BASE, "/api/resume", {}).then(refresh)}><Play size={16} /> Resume</button>
            <button className="danger" onClick={() => apiPost(AGENT_BASE, "/api/stop", {}).then(refresh)}><Square size={16} /> Stop</button>
          </div>
        </Panel>
        <Panel title="Plan">
          <JsonBlock value={state?.record || state?.current_run || state?.plan || {}} />
        </Panel>
        <Panel title="Events">
          <div className="event-list">
            {events.slice(-8).reverse().map((event, index) => (
              <div className="event" key={index}>
                <strong>{event.kind || event.type || "event"}</strong>
                <span>{event.summary || event.message || ""}</span>
              </div>
            ))}
          </div>
        </Panel>
      </div>
    </section>
  );
}

function ExplorationConsole() {
  const [state, setState] = useState(null);
  const [error, setError] = useState("");
  const [mode, setMode] = useState("block");
  const [selectedRegionId, setSelectedRegionId] = useState("");
  const [regionLabel, setRegionLabel] = useState("");
  const [regionPolygon, setRegionPolygon] = useState("[]");
  const [regionWaypoints, setRegionWaypoints] = useState("[]");
  const [placeName, setPlaceName] = useState("kitchen_entry");
  const [placePose, setPlacePose] = useState('{"x":0,"y":0,"yaw":0}');
  const [lastWaypoint, setLastWaypoint] = useState(null);

  const refresh = useCallback(async () => {
    try {
      const next = await apiGet(EXPLORATION_BASE, "/api/state");
      setState(next);
      setError("");
    } catch (err) {
      setError(err.message || String(err));
    }
  }, []);

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, 1200);
    return () => clearInterval(timer);
  }, [refresh]);

  const map = state?.current_map;
  const activeTask = state?.active_task;
  const regions = map?.regions || [];
  const selectedRegion = regions.find((item) => item.region_id === selectedRegionId);

  useEffect(() => {
    if (!selectedRegion) return;
    setRegionLabel(selectedRegion.label || "");
    setRegionPolygon(JSON.stringify(selectedRegion.polygon_2d || [], null, 2));
    setRegionWaypoints(JSON.stringify(selectedRegion.default_waypoints || [], null, 2));
    if (selectedRegion.centroid) {
      setPlacePose(JSON.stringify({ x: selectedRegion.centroid.x, y: selectedRegion.centroid.y, yaw: 0 }, null, 2));
    }
  }, [selectedRegion]);

  const post = async (path, payload = {}) => {
    const response = await apiPost(EXPLORATION_BASE, path, payload);
    await refresh();
    return response;
  };

  const saveRegion = async () => {
    if (!selectedRegionId) return;
    await post("/api/region/update", {
      region_id: selectedRegionId,
      label: regionLabel,
      polygon_2d: JSON.parse(regionPolygon || "[]"),
      default_waypoints: JSON.parse(regionWaypoints || "[]"),
    });
  };

  return (
    <section className="exploration-grid">
      <div className="exploration-left">
        <Panel title="Control">
          <div className="toolbar">
            <button className="primary" onClick={() => post("/api/explore/start", { area: "downstairs", session: "house_v1" })}><Play size={16} /> Start</button>
            <button onClick={() => post("/api/create_map/start", { area: "downstairs", session: "house_v1" })}><Map size={16} /> Create Map</button>
            <button onClick={() => post("/api/scan/perform", {})}><Eye size={16} /> Scan</button>
            <button onClick={() => post("/api/task/pause", { task_id: activeTask?.task_id })}><Pause size={16} /> Pause</button>
            <button onClick={() => post("/api/task/resume", { task_id: activeTask?.task_id })}><Play size={16} /> Resume</button>
            <button className="danger" onClick={() => post("/api/task/cancel", { task_id: activeTask?.task_id })}><Square size={16} /> Cancel</button>
            <button className="primary" onClick={() => post("/api/approve", {})}><Check size={16} /> Approve</button>
          </div>
        </Panel>
        <Panel title="Map Tools">
          <div className="segmented">
            <ModeButton mode={mode} value="block" setMode={setMode} icon={<PenLine size={16} />} label="Wall" />
            <ModeButton mode={mode} value="clear" setMode={setMode} icon={<Eraser size={16} />} label="Erase" />
            <ModeButton mode={mode} value="reset" setMode={setMode} icon={<RotateCcw size={16} />} label="Reset" />
            <ModeButton mode={mode} value="preview" setMode={setMode} icon={<Eye size={16} />} label="Preview" />
            <ModeButton mode={mode} value="waypoint" setMode={setMode} icon={<Navigation size={16} />} label="Go" />
          </div>
          <MapStats state={state} mode={mode} />
        </Panel>
        <Panel title="Regions">
          <div className="list">
            {regions.map((region) => (
              <button className={region.region_id === selectedRegionId ? "list-row active" : "list-row"} key={region.region_id} onClick={() => setSelectedRegionId(region.region_id)}>
                <span>{region.label}</span>
                <small>{region.region_id}</small>
              </button>
            ))}
          </div>
        </Panel>
      </div>
      <div className="map-panel">
        <ExplorationMap
          map={map}
          mode={mode}
          selectedRegionId={selectedRegionId}
          setSelectedRegionId={setSelectedRegionId}
          post={post}
          lastWaypoint={lastWaypoint}
          setLastWaypoint={setLastWaypoint}
        />
      </div>
      <div className="exploration-right">
        <Panel title="Selected Region">
          <label>Label</label>
          <input value={regionLabel} onChange={(event) => setRegionLabel(event.target.value)} />
          <label>Polygon JSON</label>
          <textarea value={regionPolygon} onChange={(event) => setRegionPolygon(event.target.value)} />
          <label>Waypoints JSON</label>
          <textarea value={regionWaypoints} onChange={(event) => setRegionWaypoints(event.target.value)} />
          <div className="toolbar">
            <button className="primary" onClick={saveRegion}><Save size={16} /> Save</button>
            <button onClick={() => selectedRegionId && post("/api/region/split", { region_id: selectedRegionId })}>Split</button>
          </div>
        </Panel>
        <Panel title="Named Place">
          <label>Name</label>
          <input value={placeName} onChange={(event) => setPlaceName(event.target.value)} />
          <label>Pose JSON</label>
          <textarea value={placePose} onChange={(event) => setPlacePose(event.target.value)} />
          <button onClick={() => post("/api/named_place", { name: placeName, pose: JSON.parse(placePose || "{}"), region_id: selectedRegionId || null })}>
            <Home size={16} /> Save Place
          </button>
        </Panel>
        <Panel title="Keyframes">
          <div className="thumbs">
            {(map?.keyframes || []).slice(-4).map((frame) => (
              <div key={frame.frame_id}>
                <img src={frame.thumbnail_data_url} alt={frame.frame_id} />
                <span>{frame.description}</span>
              </div>
            ))}
          </div>
        </Panel>
        {error ? <div className="error-line">{error}</div> : null}
      </div>
    </section>
  );
}

function ExplorationMap({ map, mode, selectedRegionId, setSelectedRegionId, post, lastWaypoint, setLastWaypoint }) {
  const svgRef = useRef(null);
  const painting = useRef(false);
  const lastCell = useRef("");
  const pendingCells = useRef(new Map());

  const bounds = useMemo(() => mapBounds(map), [map]);
  const project = useMemo(() => makeProjector(bounds), [bounds]);

  const flush = useCallback(async () => {
    const cells = Array.from(pendingCells.current.values());
    pendingCells.current.clear();
    if (cells.length) await post("/api/map/edit", { mode, cells });
  }, [mode, post]);

  const cellFromEvent = (event) => {
    if (!map || !svgRef.current) return null;
    const point = svgRef.current.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    const matrix = svgRef.current.getScreenCTM();
    if (!matrix) return null;
    const local = point.matrixTransform(matrix.inverse());
    const world = worldFromSvg(bounds, local.x, local.y);
    const resolution = map.occupancy?.resolution || 0.25;
    const cell = {
      cell_x: Math.floor((world.x - bounds.min_x) / resolution),
      cell_y: Math.floor((world.y - bounds.min_y) / resolution),
      x: world.x,
      y: world.y,
    };
    cell.key = `${cell.cell_x}:${cell.cell_y}`;
    return cell;
  };

  const pointerDown = async (event) => {
    event.preventDefault();
    const cell = cellFromEvent(event);
    if (!cell) return;
    if (mode === "preview" || mode === "waypoint") {
      const robotPose = map.robot_pose || (map.trajectory || []).slice(-1)[0] || {};
      const pose = { x: cell.x, y: cell.y, yaw: Number(robotPose.yaw || 0) };
      setLastWaypoint(pose);
      await post(mode === "preview" ? "/api/nav/preview" : "/api/nav/waypoint", { pose });
      return;
    }
    painting.current = true;
    lastCell.current = cell.key;
    pendingCells.current.set(cell.key, { cell_x: cell.cell_x, cell_y: cell.cell_y });
  };

  const pointerMove = (event) => {
    if (!painting.current) return;
    const cell = cellFromEvent(event);
    if (!cell || cell.key === lastCell.current) return;
    lastCell.current = cell.key;
    pendingCells.current.set(cell.key, { cell_x: cell.cell_x, cell_y: cell.cell_y });
  };

  const pointerUp = async () => {
    if (!painting.current) return;
    painting.current = false;
    lastCell.current = "";
    await flush();
  };

  if (!map) {
    return <div className="empty-map">No map</div>;
  }

  const cells = (map.occupancy?.cells || []).map((cell, index) => {
    const resolution = map.occupancy?.resolution || 0.25;
    const p1 = project({ x: cell.x, y: cell.y });
    const p2 = project({ x: cell.x + resolution, y: cell.y + resolution });
    const fill = cell.manual_override === "blocked"
      ? "#1f2937"
      : cell.manual_override === "cleared"
        ? "rgba(134, 148, 166, 0.22)"
        : cell.state === "occupied"
          ? "rgba(31, 41, 55, 0.52)"
          : "rgba(134, 148, 166, 0.2)";
    return <rect key={index} x={p1.x} y={p2.y} width={Math.max(1.5, p2.x - p1.x)} height={Math.max(1.5, p1.y - p2.y)} fill={fill} />;
  });

  const regions = (map.regions || []).map((region, index) => {
    const points = (region.polygon_2d || []).map((point) => {
      const p = project({ x: point[0], y: point[1] });
      return `${p.x},${p.y}`;
    }).join(" ");
    const centroid = project(region.centroid || { x: 0, y: 0 });
    const active = region.region_id === selectedRegionId;
    return (
      <g key={region.region_id} onClick={() => setSelectedRegionId(region.region_id)}>
        <polygon points={points} fill={active ? "rgba(29, 78, 216, 0.22)" : `hsla(${index * 53 + 170}, 45%, 45%, 0.13)`} stroke={active ? "#1d4ed8" : "rgba(17, 24, 39, 0.36)"} strokeWidth={active ? 3 : 1.5} />
        <text x={centroid.x} y={centroid.y} textAnchor="middle" className="map-label">{region.label}</text>
      </g>
    );
  });

  const trajectory = (map.trajectory || []).map((point) => {
    const p = project(point);
    return `${p.x},${p.y}`;
  }).join(" ");
  const robotPose = map.robot_pose || (map.trajectory || []).slice(-1)[0];
  const robot = robotPose ? project(robotPose) : null;
  const places = (map.named_places || []).map((place) => {
    const p = project(place.pose || { x: 0, y: 0 });
    return (
      <g key={place.name}>
        <circle cx={p.x} cy={p.y} r="5" fill="#0f766e" />
        <text x={p.x + 9} y={p.y - 8} className="place-label">{place.name}</text>
      </g>
    );
  });
  const manual = lastWaypoint ? project(lastWaypoint) : null;

  return (
    <svg
      ref={svgRef}
      viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
      className="map-canvas"
      onPointerDown={pointerDown}
      onPointerMove={pointerMove}
      onPointerUp={pointerUp}
      onPointerLeave={pointerUp}
    >
      <rect width={VIEW_W} height={VIEW_H} fill="#fbfbf8" />
      {cells}
      {trajectory ? <polyline points={trajectory} fill="none" stroke="#0f766e" strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round" /> : null}
      {regions}
      {places}
      {manual ? <circle cx={manual.x} cy={manual.y} r="9" fill="#2563eb" /> : null}
      {robot ? <circle cx={robot.x} cy={robot.y} r="11" fill="#b42318" /> : null}
      {robot ? <text x={robot.x + 12} y={robot.y - 12} className="robot-label">robot</text> : null}
    </svg>
  );
}

function Panel({ title, children }) {
  return (
    <section className="panel">
      <div className="panel-title">{title}</div>
      {children}
    </section>
  );
}

function ModeButton({ mode, value, setMode, icon, label }) {
  return <button className={mode === value ? "active" : ""} onClick={() => setMode(value)}>{icon}{label}</button>;
}

function StatusPill({ value }) {
  return <span className={`status-pill ${value}`}>{value}</span>;
}

function JsonBlock({ value }) {
  return <pre className="json-block">{JSON.stringify(value || {}, null, 2)}</pre>;
}

function MapStats({ state, mode }) {
  const map = state?.current_map || {};
  const edits = map.artifacts?.manual_occupancy_edits || {};
  const items = [
    ["Mode", mode],
    ["Map", map.map_id || "none"],
    ["Approved", map.approved ? "yes" : "no"],
    ["Regions", String((map.regions || []).length)],
    ["Walls", String((edits.blocked_cells || []).length)],
    ["Clears", String((edits.cleared_cells || []).length)],
  ];
  return (
    <div className="stat-grid">
      {items.map(([key, value]) => (
        <div className="stat" key={key}><span>{key}</span><strong>{value}</strong></div>
      ))}
    </div>
  );
}

async function apiGet(base, path) {
  const response = await fetch(`${base}${path}`);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

async function apiPost(base, path, payload) {
  const response = await fetch(`${base}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {}),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

function eventText(event) {
  return event.summary || event.message || event.kind || event.type || "Updated.";
}

function mapBounds(map) {
  if (map?.occupancy?.bounds) return map.occupancy.bounds;
  const points = [];
  for (const region of map?.regions || []) {
    for (const point of region.polygon_2d || []) points.push(point);
  }
  if (!points.length) return { min_x: 0, max_x: 10, min_y: 0, max_y: 8 };
  return {
    min_x: Math.min(...points.map((point) => point[0])),
    max_x: Math.max(...points.map((point) => point[0])),
    min_y: Math.min(...points.map((point) => point[1])),
    max_y: Math.max(...points.map((point) => point[1])),
  };
}

function makeProjector(bounds) {
  const viewport = mapViewport(bounds);
  return (point) => ({
    x: viewport.left + (Number(point?.x || 0) - bounds.min_x) * viewport.scale,
    y: viewport.top + (bounds.max_y - Number(point?.y || 0)) * viewport.scale,
  });
}

function mapViewport(bounds) {
  const worldW = Math.max(bounds.max_x - bounds.min_x, 1);
  const worldH = Math.max(bounds.max_y - bounds.min_y, 1);
  const scale = Math.min((VIEW_W - PAD * 2) / worldW, (VIEW_H - PAD * 2) / worldH);
  const drawW = worldW * scale;
  const drawH = worldH * scale;
  const left = PAD + (VIEW_W - PAD * 2 - drawW) / 2;
  const top = PAD + (VIEW_H - PAD * 2 - drawH) / 2;
  return { left, top, scale, drawW, drawH };
}

function worldFromSvg(bounds, svgX, svgY) {
  const viewport = mapViewport(bounds);
  const x = bounds.min_x + (svgX - viewport.left) / viewport.scale;
  const y = bounds.max_y - (svgY - viewport.top) / viewport.scale;
  return {
    x: Math.min(Math.max(x, bounds.min_x), bounds.max_x),
    y: Math.min(Math.max(y, bounds.min_y), bounds.max_y),
  };
}

createRoot(document.getElementById("root")).render(<App />);
