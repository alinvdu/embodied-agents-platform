import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  ArrowLeft,
  Bot,
  Bug,
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
  Settings,
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
    <div className="app-shell single">
      <main className="workspace">
        {view === "agent" ? (
          <AgentChat onConfigure={() => setView("environment")} />
        ) : (
          <ExplorationConsole onExit={() => setView("agent")} />
        )}
      </main>
    </div>
  );
}

function AgentChat({ onConfigure }) {
  const [state, setState] = useState(null);
  const [explorationState, setExplorationState] = useState(null);
  const [command, setCommand] = useState("");
  const [agentConnected, setAgentConnected] = useState(false);
  const [debugOpen, setDebugOpen] = useState(false);
  const [messages, setMessages] = useState([
    { role: "assistant", text: "Ready when you are." },
  ]);
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    try {
      const next = await apiGet(AGENT_BASE, "/api/state");
      setState(next);
      setAgentConnected(true);
      setError("");
    } catch (err) {
      setAgentConnected(false);
      setState(null);
      setError("");
    }
    try {
      setExplorationState(await apiGet(EXPLORATION_BASE, "/api/state"));
    } catch {
      setExplorationState(null);
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
      setAgentConnected(false);
      setMessages((items) => [
        ...items,
        { role: "assistant", text: "Agent backend is not connected. Start the agent backend, then send the command again." },
      ]);
    }
  };

  const events = state?.report?.events || state?.events || [];
  const lastEvent = events.slice(-1)[0];
  const memoryContext = state?.home_memory?.context || {};
  const regions = memoryContext.regions || [];
  const objects = memoryContext.objects || [];
  const environment = environmentStatus(state, explorationState);

  return (
    <section className="agent-home">
      <header className="agent-header">
        <div className="brand compact">
          <div className="brand-mark">42</div>
          <div>
            <h1>Robot42</h1>
            <span>{agentConnected ? "agent backend connected" : "agent backend offline"}</span>
          </div>
        </div>
        <button onClick={onConfigure}><Settings size={16} /> Configure Environment</button>
      </header>

      <div className="agent-main">
        <EnvironmentCard environment={environment} onConfigure={onConfigure} />
        <div className="chat-panel minimal">
          <div className="section-head">
            <span><Bot size={17} /> Agent</span>
            <StatusPill value={agentConnected ? state?.status || "idle" : "offline"} />
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
      </div>
      <div className="agent-actions">
        <button disabled={!agentConnected} onClick={() => apiPost(AGENT_BASE, "/api/pause", {}).then(refresh)}><Pause size={16} /> Pause</button>
        <button disabled={!agentConnected} onClick={() => apiPost(AGENT_BASE, "/api/resume", {}).then(refresh)}><Play size={16} /> Resume</button>
        <button disabled={!agentConnected} className="danger" onClick={() => apiPost(AGENT_BASE, "/api/stop", {}).then(refresh)}><Square size={16} /> Stop</button>
        <button onClick={() => setDebugOpen((value) => !value)}><Bug size={16} /> Debug</button>
      </div>

      {debugOpen ? (
        <div className="debug-grid">
          <Panel title="Memory">
            <div className="memory-summary">{state?.home_memory?.summary || environment.description}</div>
            <div className="chip-row">
              {regions.slice(0, 8).map((region) => <span className="chip" key={region.region_id || region.label}>{region.label}</span>)}
              {objects.slice(0, 8).map((object) => <span className="chip object" key={object.object_id || object.label}>{object.label}</span>)}
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
      ) : null}
    </section>
  );
}

function EnvironmentCard({ environment, onConfigure }) {
  return (
    <div className={`environment-card ${environment.configured ? "configured" : ""}`}>
      <div className="environment-icon"><Home size={22} /></div>
      <div>
        <div className="environment-title">{environment.configured ? "Environment configured" : "Environment not configured"}</div>
        <div className="environment-detail">{environment.description}</div>
      </div>
      <div className="environment-metrics">
        <span><Map size={15} /> {environment.regionCount} regions</span>
        <span><Home size={15} /> {environment.objectCount} objects</span>
      </div>
      <button className="compact-button" onClick={onConfigure}><Settings size={15} /> Configure</button>
    </div>
  );
}

function ExplorationConsole({ onExit }) {
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
  const [draftRegion, setDraftRegion] = useState([]);
  const [newRegionLabel, setNewRegionLabel] = useState("kitchen");
  const [newRegionPurpose, setNewRegionPurpose] = useState("");
  const [saveMessage, setSaveMessage] = useState("");

  const refresh = useCallback(async () => {
    try {
      const next = await apiGet(EXPLORATION_BASE, "/api/state");
      setState(next);
      setError("");
    } catch (err) {
      setState(null);
      setError(err.message || String(err));
    }
  }, []);

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, 1200);
    return () => clearInterval(timer);
  }, [refresh]);

  const map = state?.current_map;
  const backendOnline = Boolean(state);
  const activeTask = state?.active_task;
  const regions = map?.regions || [];
  const selectedRegion = regions.find((item) => item.region_id === selectedRegionId);
  const selectedRegionForMap = useMemo(() => {
    if (!selectedRegion) return null;
    try {
      return { ...selectedRegion, polygon_2d: JSON.parse(regionPolygon || "[]") };
    } catch {
      return selectedRegion;
    }
  }, [selectedRegion, regionPolygon]);

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

  const createDraftRegion = async () => {
    if (draftRegion.length < 3) return;
    const created = await post("/api/region/create", {
      label: newRegionLabel || "region",
      purpose: newRegionPurpose || undefined,
      polygon_2d: draftRegion,
    });
    if (created?.region_id) setSelectedRegionId(created.region_id);
    setDraftRegion([]);
    setMode("block");
  };

  const updateSelectedPolygon = (polygon) => {
    setRegionPolygon(JSON.stringify(polygon, null, 2));
  };

  const approveAndSaveMemory = async () => {
    const approved = await post("/api/approve", {});
    const memory = approved?.artifacts?.home_memory;
    setSaveMessage(
      memory?.path
        ? `Environment saved to long-term memory: ${memory.path}`
        : "Environment approved. Long-term memory export is not available for this backend."
    );
  };

  const setDockPose = async () => {
    const robotPose = map?.robot_pose || (map?.trajectory || []).slice(-1)[0] || null;
    const saved = await post("/api/dock_pose", robotPose ? { pose: robotPose } : {});
    if (saved?.pose) {
      setSaveMessage(`Dock pose saved at (${saved.pose.x}, ${saved.pose.y}).`);
    }
  };

  return (
    <section className="environment-shell">
      <header className="environment-topbar">
        <button onClick={onExit}><ArrowLeft size={16} /> Exit Exploration</button>
        <div>
          <strong>Configure Environment</strong>
          <span>{backendOnline ? (map?.approved ? "approved map" : "editing map") : "exploration backend offline"}</span>
        </div>
      </header>
      <div className="exploration-grid">
      <div className="exploration-left">
        <Panel title="Control">
          <div className="toolbar">
            <button className="primary" disabled={!backendOnline} onClick={() => post("/api/explore/start", { area: "downstairs", session: "house_v1" })}><Play size={16} /> Start</button>
            <button disabled={!backendOnline} onClick={() => post("/api/create_map/start", { area: "downstairs", session: "house_v1" })}><Map size={16} /> Create Map</button>
            <button disabled={!backendOnline} onClick={() => post("/api/scan/perform", {})}><Eye size={16} /> Scan</button>
            <button disabled={!backendOnline} onClick={() => post("/api/task/pause", { task_id: activeTask?.task_id })}><Pause size={16} /> Pause</button>
            <button disabled={!backendOnline} onClick={() => post("/api/task/resume", { task_id: activeTask?.task_id })}><Play size={16} /> Resume</button>
            <button disabled={!backendOnline} className="danger" onClick={() => post("/api/task/cancel", { task_id: activeTask?.task_id })}><Square size={16} /> Cancel</button>
            <button disabled={!backendOnline || !map} onClick={setDockPose}><Home size={16} /> Set Dock Pose</button>
            <button disabled={!backendOnline || !map} className="primary" onClick={approveAndSaveMemory}><Check size={16} /> Approve + Save Memory</button>
            {!backendOnline ? <button onClick={refresh}><RotateCcw size={16} /> Retry Connection</button> : null}
          </div>
          {!backendOnline ? <p className="offline-note">Start the exploration backend to load, edit, and save the environment.</p> : null}
          {saveMessage ? <p className="success-line">{saveMessage}</p> : null}
        </Panel>
        <Panel title="Map Tools">
          <div className="segmented">
            <ModeButton disabled={!backendOnline} mode={mode} value="block" setMode={setMode} icon={<PenLine size={16} />} label="Wall" />
            <ModeButton disabled={!backendOnline} mode={mode} value="clear" setMode={setMode} icon={<Eraser size={16} />} label="Erase" />
            <ModeButton disabled={!backendOnline} mode={mode} value="reset" setMode={setMode} icon={<RotateCcw size={16} />} label="Reset" />
            <ModeButton disabled={!backendOnline} mode={mode} value="preview" setMode={setMode} icon={<Eye size={16} />} label="Preview" />
            <ModeButton disabled={!backendOnline} mode={mode} value="waypoint" setMode={setMode} icon={<Navigation size={16} />} label="Go" />
            <ModeButton disabled={!backendOnline} mode={mode} value="region" setMode={setMode} icon={<Map size={16} />} label="Region" />
          </div>
          {mode === "region" ? (
            <div className="region-draft">
              <label>New Region Label</label>
              <input value={newRegionLabel} onChange={(event) => setNewRegionLabel(event.target.value)} />
              <label>Purpose</label>
              <input value={newRegionPurpose} onChange={(event) => setNewRegionPurpose(event.target.value)} placeholder="optional" />
              <div className="toolbar">
                <button className="primary" disabled={!backendOnline || draftRegion.length < 3} onClick={createDraftRegion}><Save size={16} /> Save Region</button>
                <button onClick={() => setDraftRegion((points) => points.slice(0, -1))}>Undo Point</button>
                <button onClick={() => setDraftRegion([])}>Clear</button>
              </div>
              <p className="hint">{draftRegion.length ? `${draftRegion.length} point polygon draft` : "Click the map to place polygon points."}</p>
            </div>
          ) : null}
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
        {backendOnline ? (
          <ExplorationMap
            map={map}
            mode={mode}
            selectedRegionId={selectedRegionId}
            setSelectedRegionId={setSelectedRegionId}
            post={post}
            lastWaypoint={lastWaypoint}
            setLastWaypoint={setLastWaypoint}
            draftRegion={draftRegion}
            setDraftRegion={setDraftRegion}
            selectedRegion={selectedRegionForMap}
            updateSelectedPolygon={updateSelectedPolygon}
          />
        ) : (
          <OfflineExplorationPlaceholder onRetry={refresh} />
        )}
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
            <button className="primary" disabled={!backendOnline || !selectedRegionId} onClick={saveRegion}><Save size={16} /> Save</button>
            <button disabled={!backendOnline || !selectedRegionId} onClick={() => selectedRegionId && post("/api/region/split", { region_id: selectedRegionId })}>Split</button>
          </div>
        </Panel>
        <Panel title="Named Place">
          <label>Name</label>
          <input value={placeName} onChange={(event) => setPlaceName(event.target.value)} />
          <label>Pose JSON</label>
          <textarea value={placePose} onChange={(event) => setPlacePose(event.target.value)} />
          <button disabled={!backendOnline} onClick={() => post("/api/named_place", { name: placeName, pose: JSON.parse(placePose || "{}"), region_id: selectedRegionId || null })}>
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
        {error ? <div className="error-line">{backendOnline ? error : "Exploration backend is not connected."}</div> : null}
      </div>
      </div>
    </section>
  );
}

function ExplorationMap({
  map,
  mode,
  selectedRegionId,
  setSelectedRegionId,
  post,
  lastWaypoint,
  setLastWaypoint,
  draftRegion,
  setDraftRegion,
  selectedRegion,
  updateSelectedPolygon,
}) {
  const svgRef = useRef(null);
  const painting = useRef(false);
  const lastCell = useRef("");
  const pendingCells = useRef(new Map());
  const draggingVertex = useRef(null);

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
    if (event.target.closest?.("[data-region-id]") && mode !== "region") return;
    const cell = cellFromEvent(event);
    if (!cell) return;
    if (mode === "region") {
      setDraftRegion((points) => [...points, [round3(cell.x), round3(cell.y)]]);
      return;
    }
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
    if (draggingVertex.current) {
      const cell = cellFromEvent(event);
      if (!cell) return;
      const { regionId, index } = draggingVertex.current;
      if (regionId !== selectedRegionId || !selectedRegion) return;
      const nextPolygon = (selectedRegion.polygon_2d || []).map((point, pointIndex) =>
        pointIndex === index ? [round3(cell.x), round3(cell.y)] : point
      );
      updateSelectedPolygon(nextPolygon);
      return;
    }
    if (!painting.current) return;
    const cell = cellFromEvent(event);
    if (!cell || cell.key === lastCell.current) return;
    lastCell.current = cell.key;
    pendingCells.current.set(cell.key, { cell_x: cell.cell_x, cell_y: cell.cell_y });
  };

  const pointerUp = async () => {
    if (draggingVertex.current) {
      draggingVertex.current = null;
      return;
    }
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
    const polygonPoints = region.region_id === selectedRegionId && selectedRegion
      ? selectedRegion.polygon_2d || region.polygon_2d || []
      : region.polygon_2d || [];
    const points = polygonPoints.map((point) => {
      const p = project({ x: point[0], y: point[1] });
      return `${p.x},${p.y}`;
    }).join(" ");
    const centroid = project(region.centroid || { x: 0, y: 0 });
    const active = region.region_id === selectedRegionId;
    return (
      <g key={region.region_id} data-region-id={region.region_id} onClick={() => setSelectedRegionId(region.region_id)}>
        <polygon points={points} fill={active ? "rgba(29, 78, 216, 0.22)" : `hsla(${index * 53 + 170}, 45%, 45%, 0.13)`} stroke={active ? "#1d4ed8" : "rgba(17, 24, 39, 0.36)"} strokeWidth={active ? 3 : 1.5} />
        <text x={centroid.x} y={centroid.y} textAnchor="middle" className="map-label">{region.label}</text>
        {active ? polygonPoints.map((point, pointIndex) => {
          const p = project({ x: point[0], y: point[1] });
          return (
            <circle
              key={`${region.region_id}-vertex-${pointIndex}`}
              cx={p.x}
              cy={p.y}
              r="7"
              className="vertex-handle"
              onPointerDown={(event) => {
                event.stopPropagation();
                event.preventDefault();
                draggingVertex.current = { regionId: region.region_id, index: pointIndex };
              }}
            />
          );
        }) : null}
      </g>
    );
  });
  const draftPoints = (draftRegion || []).map((point) => {
    const p = project({ x: point[0], y: point[1] });
    return `${p.x},${p.y}`;
  }).join(" ");
  const draftHandles = (draftRegion || []).map((point, index) => {
    const p = project({ x: point[0], y: point[1] });
    return <circle key={`draft-${index}`} cx={p.x} cy={p.y} r="6" className="draft-handle" />;
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
      {draftPoints ? <polyline points={draftPoints} fill={draftRegion.length > 2 ? "rgba(15, 118, 110, 0.14)" : "none"} stroke="#0f766e" strokeWidth="3" strokeDasharray="8 6" /> : null}
      {draftHandles}
      {places}
      {manual ? <circle cx={manual.x} cy={manual.y} r="9" fill="#2563eb" /> : null}
      {robot ? <circle cx={robot.x} cy={robot.y} r="11" fill="#b42318" /> : null}
      {robot ? <text x={robot.x + 12} y={robot.y - 12} className="robot-label">robot</text> : null}
    </svg>
  );
}

function OfflineExplorationPlaceholder({ onRetry }) {
  return (
    <div className="offline-map">
      <div className="offline-icon"><Map size={26} /></div>
      <strong>Exploration backend offline</strong>
      <span>Start the exploration backend to load the map editor, then retry the connection.</span>
      <button onClick={onRetry}><RotateCcw size={16} /> Retry Connection</button>
    </div>
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

function ModeButton({ mode, value, setMode, icon, label, disabled = false }) {
  return <button disabled={disabled} className={mode === value ? "active" : ""} onClick={() => setMode(value)}>{icon}{label}</button>;
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

function environmentStatus(agentState, explorationState) {
  const memory = agentState?.home_memory?.context || {};
  const map = explorationState?.current_map || {};
  const memoryId = memory.memory_id || map.artifacts?.home_memory?.memory_id || map.map_id;
  const regions = memory.regions || map.regions || [];
  const objects = memory.objects || map.objects || [];
  const configured = Boolean(memory.memory_id || map.approved || map.artifacts?.home_memory);
  const description = configured
    ? `${memoryId || "home"} is ready for agent context.`
    : "Approve a map to create long-term home memory.";
  return {
    configured,
    description,
    regionCount: regions.length || 0,
    objectCount: objects.length || 0,
  };
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
