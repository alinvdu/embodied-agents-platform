import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  ArrowLeft,
  Bot,
  Bug,
  Check,
  Crosshair,
  Eraser,
  Eye,
  Home,
  Map as MapIcon,
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
const AGENT_VIEW_H = 390;
const AGENT_PAD = 14;

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
  const [selectedMemoryId, setSelectedMemoryId] = useState("");
  const [newMemoryId, setNewMemoryId] = useState("house_v1");
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
  const homeMemory = state?.home_memory?.preview_map || {};
  const regions = memoryContext.regions || [];
  const objects = memoryContext.objects || [];
  const agentMemories = state?.environment_memories || [];
  const environment = environmentStatus(state, explorationState);
  const navigationPreview = latestNavigationPreview(events);
  const visionShots = latestVisionShots(events);

  const selectAgentMemory = async (memoryId) => {
    if (!memoryId) return;
    setSelectedMemoryId(memoryId);
    if (!agentConnected) return;
    await apiPost(AGENT_BASE, "/api/memory/select", { memory_id: memoryId });
    await refresh();
  };

  const createAgentMemory = async () => {
    const memoryId = newMemoryId.trim();
    if (!memoryId || !agentConnected) {
      onConfigure();
      return;
    }
    await apiPost(AGENT_BASE, "/api/memory/create", { memory_id: memoryId, label: memoryId });
    setSelectedMemoryId(memoryId);
    await refresh();
    onConfigure();
  };

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
        <EnvironmentCard
          environment={environment}
          memories={agentMemories}
          selectedMemoryId={selectedMemoryId || memoryContext.memory_id || ""}
          onSelectMemory={selectAgentMemory}
          newMemoryId={newMemoryId}
          setNewMemoryId={setNewMemoryId}
          onCreateMemory={createAgentMemory}
          onConfigure={onConfigure}
        />
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
      <AgentMapPreview memory={homeMemory} liveMap={explorationState?.current_map || null} preview={navigationPreview} />
      <AgentVisionReport shots={visionShots} />
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

function AgentVisionReport({ shots }) {
  if (!shots.length) return null;
  return (
    <section className="vision-report-card">
      <div className="section-head">
        <span><Eye size={17} /> What Robot Saw</span>
        <small>{shots.length} RGB shot{shots.length === 1 ? "" : "s"}</small>
      </div>
      <div className="vision-strip">
        {shots.map((shot) => {
          const capture = shot.capture || {};
          const src = artifactSrc(capture.artifact_url || capture.image_url);
          return (
            <div className="vision-shot" key={`${shot.stop_id}-${shot.shot_id}`}>
              {src ? <img src={src} alt={`${shot.stop_id} ${shot.shot_id}`} /> : <div className="vision-missing">No RGB</div>}
              <div>
                <strong>{shot.region_label || "region"}</strong>
                <span>{shot.stop_id} / {shot.shot_id}</span>
                <span>{capture.status || "unknown"}</span>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function EnvironmentCard({
  environment,
  memories = [],
  selectedMemoryId = "",
  onSelectMemory,
  newMemoryId,
  setNewMemoryId,
  onCreateMemory,
  onConfigure,
}) {
  return (
    <div className={`environment-card ${environment.configured ? "configured" : ""}`}>
      <div className="environment-icon"><Home size={22} /></div>
      <div>
        <div className="environment-title">{environment.configured ? "Environment configured" : "Environment not configured"}</div>
        <div className="environment-detail">{environment.description}</div>
      </div>
      <div className="environment-metrics">
        <span><MapIcon size={15} /> {environment.regionCount} regions</span>
        <span><Home size={15} /> {environment.objectCount} objects</span>
      </div>
      <div className="memory-picker">
        <select value={selectedMemoryId} onChange={(event) => onSelectMemory?.(event.target.value)}>
          <option value="">Select memory</option>
          {memories.map((memory) => (
            <option value={memory.memory_id} key={`${memory.memory_id}-${memory.home_memory_path || memory.directory}`}>
              {memory.label || memory.memory_id}
            </option>
          ))}
        </select>
        <div className="new-memory-row">
          <input value={newMemoryId || ""} onChange={(event) => setNewMemoryId?.(event.target.value)} placeholder="new environment" />
          <button className="compact-button" onClick={onCreateMemory}><MapIcon size={15} /> New</button>
        </div>
      </div>
      <button className="compact-button" onClick={onConfigure}><Settings size={15} /> Configure</button>
    </div>
  );
}

function AgentMapPreview({ memory, liveMap, preview }) {
  const displayMap = liveMap?.occupancy?.cells?.length || liveMap?.regions?.length ? liveMap : memory;
  const hasMap = Boolean(displayMap?.occupancy?.cells?.length || displayMap?.regions?.length);
  const bounds = useMemo(() => agentPreviewBounds(displayMap, preview), [displayMap, preview]);
  const project = useMemo(() => makeProjector(bounds, { height: AGENT_VIEW_H, pad: AGENT_PAD }), [bounds]);
  if (!hasMap && !preview) return null;
  const cells = (displayMap?.occupancy?.cells || []).map((cell, index) => {
    const resolution = displayMap?.occupancy?.resolution || 0.25;
    const p1 = project({ x: cell.x, y: cell.y });
    const p2 = project({ x: Number(cell.x || 0) + resolution, y: Number(cell.y || 0) + resolution });
    const fill = cell.manual_override === "blocked"
      ? "#1f2937"
      : cell.manual_override === "cleared"
        ? "rgba(134, 148, 166, 0.22)"
        : cell.state === "occupied"
          ? "rgba(31, 41, 55, 0.52)"
          : cell.state === "free"
            ? "rgba(134, 148, 166, 0.20)"
            : "rgba(134, 148, 166, 0.10)";
    return <rect key={index} x={p1.x} y={p2.y} width={Math.max(1.5, p2.x - p1.x)} height={Math.max(1.5, p1.y - p2.y)} fill={fill} />;
  });
  const regions = (displayMap?.regions || []).map((region, index) => {
    const points = (region.polygon_2d || []).map((point) => {
      const p = project({ x: point[0], y: point[1] });
      return `${p.x},${p.y}`;
    }).join(" ");
    if (!points) return null;
    const labelPoint = polygonLabelPoint(region.polygon_2d || []);
    const label = project(labelPoint);
    return (
      <g key={region.region_id || region.label || index}>
        <polygon points={points} fill={`hsla(${index * 53 + 170}, 45%, 45%, 0.10)`} stroke="rgba(17, 24, 39, 0.24)" strokeWidth="1.4" />
        <text x={label.x} y={label.y} textAnchor="middle" className="agent-map-region-label">{region.label}</text>
      </g>
    );
  });
  const pathPoints = (preview?.path || []).map((point) => {
    const p = project(point);
    return `${p.x},${p.y}`;
  }).join(" ");
  const explorationStops = preview?.tool === "plan_region_exploration" ? (preview?.stops || []) : [];
  const explorationCones = explorationStops.flatMap((stop, stopIndex) =>
    (stop.shots || []).map((shot, shotIndex) => {
      const origin = project(shot.cone?.origin || stop.pose || {});
      const left = project(shot.cone?.left || {});
      const center = project(shot.cone?.center || {});
      const right = project(shot.cone?.right || {});
      return { stop, stopIndex, shot, shotIndex, origin, left, center, right };
    })
  );
  const explorationStopPoints = explorationStops.map((stop) => ({ stop, point: project(stop.pose || {}) }));
  const goal = preview?.goal_pose ? project(preview.goal_pose) : null;
  const nextWaypoint = preview?.next_waypoint ? project(preview.next_waypoint) : null;
  const robotPose = displayMap?.robot_pose || (displayMap?.trajectory || []).slice(-1)[0] || null;
  const robot = robotPose ? project(robotPose) : null;
  const headingLength = Math.max(displayMap?.occupancy?.resolution || 0.25, 0.25) * 3.5;
  const robotHeading = robotPose ? project({
    x: Number(robotPose.x || 0) + Math.cos(Number(robotPose.yaw || 0)) * headingLength,
    y: Number(robotPose.y || 0) + Math.sin(Number(robotPose.yaw || 0)) * headingLength,
  }) : null;
  const startPose = displayMap?.start_pose?.pose || displayMap?.start_pose;
  const start = startPose ? project(startPose) : null;
  const subtitle = preview?.tool === "plan_region_exploration"
    ? `${preview.target_label || "region"} -> ${explorationStops.length} stops, ${explorationCones.length} shots`
    : preview?.goal_pose
    ? `${preview.target_label || "target"} -> next (${round3(preview.next_waypoint?.x ?? preview.goal_pose.x)}, ${round3(preview.next_waypoint?.y ?? preview.goal_pose.y)})`
    : "Ask for a region target to see the selected point.";

  return (
    <section className="agent-map-card">
      <div className="section-head">
        <span><MapIcon size={17} /> Navigation Preview</span>
        <small>{subtitle}</small>
      </div>
      <svg viewBox={`0 0 ${VIEW_W} ${AGENT_VIEW_H}`} className="agent-map-canvas">
        <rect width={VIEW_W} height={AGENT_VIEW_H} fill="#fbfbf8" />
        {cells}
        {regions}
        {explorationCones.map((item) => (
          <g key={`${item.stop.stop_id || item.stopIndex}-${item.shot.shot_id || item.shotIndex}`}>
            <polygon
              points={`${item.origin.x},${item.origin.y} ${item.left.x},${item.left.y} ${item.right.x},${item.right.y}`}
              fill="rgba(37, 99, 235, 0.08)"
              stroke="rgba(37, 99, 235, 0.48)"
              strokeWidth="2"
            />
            <line x1={item.origin.x} y1={item.origin.y} x2={item.center.x} y2={item.center.y} stroke="rgba(37, 99, 235, 0.82)" strokeWidth="3" strokeLinecap="round" />
          </g>
        ))}
        {pathPoints ? <polyline points={pathPoints} fill="none" stroke="#0f766e" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round" /> : null}
        {start ? <circle cx={start.x} cy={start.y} r="8" fill="#111827" /> : null}
        {explorationStopPoints.map((item, index) => (
          <g key={item.stop.stop_id || index}>
            <circle cx={item.point.x} cy={item.point.y} r="11" fill="#ef3b24" />
            <text x={item.point.x + 12} y={item.point.y - 12} className="agent-map-goal-label">{item.stop.name || `stop ${index + 1}`}</text>
          </g>
        ))}
        {nextWaypoint ? <circle cx={nextWaypoint.x} cy={nextWaypoint.y} r="10" fill="#d97706" /> : null}
        {nextWaypoint ? <circle cx={nextWaypoint.x} cy={nextWaypoint.y} r="18" fill="none" stroke="#d97706" strokeWidth="3" opacity="0.28" /> : null}
        {nextWaypoint ? <text x={nextWaypoint.x + 13} y={nextWaypoint.y + 20} className="agent-map-waypoint-label">waypoint</text> : null}
        {goal ? <circle cx={goal.x} cy={goal.y} r="12" fill="#b42318" /> : null}
        {goal ? <circle cx={goal.x} cy={goal.y} r="21" fill="none" stroke="#b42318" strokeWidth="3" opacity="0.22" /> : null}
        {goal ? <text x={goal.x + 14} y={goal.y - 13} className="agent-map-goal-label">goal</text> : null}
        {robot ? <circle cx={robot.x} cy={robot.y} r="11" fill="#b42318" /> : null}
        {robot && robotHeading ? <line x1={robot.x} y1={robot.y} x2={robotHeading.x} y2={robotHeading.y} className="robot-heading" /> : null}
        {robot && robotHeading ? <circle cx={robotHeading.x} cy={robotHeading.y} r="4" fill="#6d0f0a" /> : null}
        {robot ? <text x={robot.x + 12} y={robot.y - 12} className="robot-label">robot</text> : null}
      </svg>
    </section>
  );
}

function ExplorationConsole({ onExit }) {
  const [state, setState] = useState(null);
  const [memories, setMemories] = useState([]);
  const [selectedMemoryId, setSelectedMemoryId] = useState("");
  const [newEnvironmentId, setNewEnvironmentId] = useState("house_v1");
  const [error, setError] = useState("");
  const [mode, setMode] = useState("block");
  const [selectedRegionId, setSelectedRegionId] = useState("");
  const [regionLabel, setRegionLabel] = useState("");
  const [regionPurpose, setRegionPurpose] = useState("");
  const [regionPolygon, setRegionPolygon] = useState("[]");
  const [regionWaypoints, setRegionWaypoints] = useState("[]");
  const [placeName, setPlaceName] = useState("kitchen_entry");
  const [placePose, setPlacePose] = useState('{"x":0,"y":0,"yaw":0}');
  const [lastWaypoint, setLastWaypoint] = useState(null);
  const [draftRegion, setDraftRegion] = useState([]);
  const [newRegionLabel, setNewRegionLabel] = useState("kitchen");
  const [newRegionPurpose, setNewRegionPurpose] = useState("");
  const [saveMessage, setSaveMessage] = useState("");
  const [navMessage, setNavMessage] = useState("");
  const [regionDirty, setRegionDirty] = useState(false);
  const loadedRegionKey = useRef("");
  const regionDraftDirty = useRef(false);

  const refresh = useCallback(async () => {
    try {
      const next = await apiGet(EXPLORATION_BASE, "/api/state");
      setState(next);
      const memoryList = await apiGet(EXPLORATION_BASE, "/api/memories");
      setMemories(memoryList.memories || []);
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
  const selectedRegionKey = selectedRegion
    ? [
        selectedMemoryId || "",
        map?.artifacts?.home_memory?.memory_id || map?.map_id || "",
        selectedRegion.region_id,
      ].join(":")
    : "";
  const selectedRegionForMap = useMemo(() => {
    if (!selectedRegion) return null;
    try {
      return { ...selectedRegion, polygon_2d: JSON.parse(regionPolygon || "[]") };
    } catch {
      return selectedRegion;
    }
  }, [selectedRegion, regionPolygon]);

  useEffect(() => {
    if (!selectedRegion) {
      loadedRegionKey.current = "";
      return;
    }
    if (loadedRegionKey.current === selectedRegionKey && regionDraftDirty.current) return;
    loadedRegionKey.current = selectedRegionKey;
    regionDraftDirty.current = false;
    setRegionDirty(false);
    setRegionLabel(selectedRegion.label || "");
    setRegionPurpose(selectedRegion.purpose || "");
    setRegionPolygon(JSON.stringify(selectedRegion.polygon_2d || [], null, 2));
    setRegionWaypoints(JSON.stringify(selectedRegion.default_waypoints || [], null, 2));
    if (selectedRegion.centroid) {
      setPlacePose(JSON.stringify({ x: selectedRegion.centroid.x, y: selectedRegion.centroid.y, yaw: 0 }, null, 2));
    }
  }, [selectedRegion, selectedRegionKey]);

  const post = async (path, payload = {}) => {
    const response = await apiPost(EXPLORATION_BASE, path, payload);
    await refresh();
    return response;
  };

  const activeEnvironmentId = () => selectedMemoryId || newEnvironmentId.trim() || "house_v1";

  const createEnvironment = async () => {
    const memoryId = activeEnvironmentId();
    const created = await post("/api/memory/create", { memory_id: memoryId, label: memoryId });
    setSelectedMemoryId(created?.memory_id || memoryId);
    setSaveMessage(`Environment workspace ready: ${created?.memory_id || memoryId}.`);
  };

  const loadEnvironment = async () => {
    if (!selectedMemoryId) return;
    const loaded = await post("/api/memory/load", { memory_id: selectedMemoryId });
    setSaveMessage(
      loaded?.current_map
        ? `Loaded environment: ${selectedMemoryId}.`
        : `Could not load environment: ${selectedMemoryId}.`
    );
  };

  const selectedRegionUpdatePayload = () => ({
    region_id: selectedRegionId,
    label: regionLabel,
    purpose: regionPurpose,
    polygon_2d: JSON.parse(regionPolygon || "[]"),
    default_waypoints: JSON.parse(regionWaypoints || "[]"),
  });

  const markRegionDirty = () => {
    regionDraftDirty.current = true;
    setRegionDirty(true);
  };

  const commitSelectedRegionDraft = async ({ refreshAfter = true } = {}) => {
    if (!selectedRegionId) return null;
    const response = await apiPost(EXPLORATION_BASE, "/api/region/update", selectedRegionUpdatePayload());
    regionDraftDirty.current = false;
    setRegionDirty(false);
    if (refreshAfter) await refresh();
    return response;
  };

  const saveRegion = async () => {
    if (!selectedRegionId) return;
    await commitSelectedRegionDraft();
    setSaveMessage(`Region saved: ${regionLabel || selectedRegionId}.`);
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
    markRegionDirty();
    setRegionPolygon(JSON.stringify(polygon, null, 2));
  };

  const editRegionLabel = (value) => {
    markRegionDirty();
    setRegionLabel(value);
  };

  const editRegionPurpose = (value) => {
    markRegionDirty();
    setRegionPurpose(value);
  };

  const editRegionPolygon = (value) => {
    markRegionDirty();
    setRegionPolygon(value);
  };

  const editRegionWaypoints = (value) => {
    markRegionDirty();
    setRegionWaypoints(value);
  };

  const approveAndSaveMemory = async () => {
    if (selectedRegionId && regionDraftDirty.current) {
      await commitSelectedRegionDraft({ refreshAfter: false });
    }
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

  const startNavigationSession = async () => {
    const result = await post("/api/nav/session/start", {});
    setSaveMessage(
      result?.status === "active"
        ? "Navigation session active. Preview and Go now use the loaded environment."
        : `Navigation session did not start: ${result?.reason || result?.status || "unknown"}`
    );
  };

  const stopNavigationSession = async () => {
    const result = await post("/api/nav/session/stop", {});
    setSaveMessage(
      result?.status === "stopped"
        ? "Navigation session stopped."
        : `Navigation session status: ${result?.reason || result?.status || "unknown"}`
    );
  };

  const relocalizeHere = async () => {
    setSaveMessage("Running relocalization scan...");
    const result = await post("/api/nav/relocalize", {});
    const match = result?.match || {};
    const delta = match?.delta || {};
    if (result?.status === "corrected") {
      setSaveMessage(
        `Relocalized: dx ${delta.dx_m ?? 0}m, dy ${delta.dy_m ?? 0}m, yaw ${delta.dyaw_deg ?? 0}deg, confidence ${match.confidence ?? "n/a"}.`
      );
    } else {
      setSaveMessage(`Relocalization ${result?.status || "skipped"}: ${result?.reason || "confidence too low"}.`);
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
            <button className="primary" disabled={!backendOnline} onClick={() => post("/api/explore/start", { area: "downstairs", session: activeEnvironmentId() })}><Play size={16} /> Start</button>
            <button disabled={!backendOnline} onClick={() => post("/api/create_map/start", { area: "downstairs", session: activeEnvironmentId() })}><MapIcon size={16} /> Create Map</button>
            <button disabled={!backendOnline} onClick={() => post("/api/scan/perform", {})}><Eye size={16} /> Scan</button>
            <button disabled={!backendOnline} onClick={() => post("/api/task/pause", { task_id: activeTask?.task_id })}><Pause size={16} /> Pause</button>
            <button disabled={!backendOnline} onClick={() => post("/api/task/resume", { task_id: activeTask?.task_id })}><Play size={16} /> Resume</button>
            <button disabled={!backendOnline} className="danger" onClick={() => post("/api/task/cancel", { task_id: activeTask?.task_id })}><Square size={16} /> Cancel</button>
            <button disabled={!backendOnline || !map} onClick={startNavigationSession}><Navigation size={16} /> Start Nav Session</button>
            <button disabled={!backendOnline} onClick={stopNavigationSession}><Square size={16} /> Stop Nav Session</button>
            <button disabled={!backendOnline || !map} onClick={relocalizeHere}><Crosshair size={16} /> Relocalize</button>
            <button disabled={!backendOnline || !map} onClick={setDockPose}><Home size={16} /> Set Dock Pose</button>
            <button disabled={!backendOnline || !map} className="primary" onClick={approveAndSaveMemory}><Check size={16} /> Approve + Save Memory</button>
            {!backendOnline ? <button onClick={refresh}><RotateCcw size={16} /> Retry Connection</button> : null}
          </div>
          {!backendOnline ? <p className="offline-note">Start the exploration backend to load, edit, and save the environment.</p> : null}
          {saveMessage ? <p className="success-line">{saveMessage}</p> : null}
        </Panel>
        <Panel title="Environments">
          <div className="memory-list">
            <select value={selectedMemoryId} onChange={(event) => setSelectedMemoryId(event.target.value)}>
              <option value="">Select saved environment</option>
              {memories.map((memory) => (
                <option value={memory.memory_id} key={`${memory.memory_id}-${memory.directory}`}>
                  {memory.label || memory.memory_id}
                </option>
              ))}
            </select>
            <div className="toolbar">
              <button disabled={!backendOnline || !selectedMemoryId} onClick={loadEnvironment}><Eye size={16} /> Load</button>
              <button disabled={!backendOnline} onClick={createEnvironment}><MapIcon size={16} /> New</button>
            </div>
            <input value={newEnvironmentId} onChange={(event) => setNewEnvironmentId(event.target.value)} placeholder="environment id" />
            <div className="memory-mini-list">
              {memories.slice(0, 4).map((memory) => (
                <button
                  key={`${memory.memory_id}-${memory.updated_at}`}
                  className={memory.memory_id === selectedMemoryId ? "memory-row active" : "memory-row"}
                  onClick={() => setSelectedMemoryId(memory.memory_id)}
                >
                  <span>{memory.label || memory.memory_id}</span>
                  <small>{memory.region_count || 0} regions</small>
                </button>
              ))}
            </div>
          </div>
        </Panel>
        <Panel title="Map Tools">
          <div className="segmented">
            <ModeButton disabled={!backendOnline} mode={mode} value="block" setMode={setMode} icon={<PenLine size={16} />} label="Wall" />
            <ModeButton disabled={!backendOnline} mode={mode} value="clear" setMode={setMode} icon={<Eraser size={16} />} label="Erase" />
            <ModeButton disabled={!backendOnline} mode={mode} value="reset" setMode={setMode} icon={<RotateCcw size={16} />} label="Reset" />
            <ModeButton disabled={!backendOnline} mode={mode} value="preview" setMode={setMode} icon={<Eye size={16} />} label="Preview" />
            <ModeButton disabled={!backendOnline} mode={mode} value="waypoint" setMode={setMode} icon={<Navigation size={16} />} label="Go" />
            <ModeButton disabled={!backendOnline} mode={mode} value="region" setMode={setMode} icon={<MapIcon size={16} />} label="Region" />
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
          <>
            <ExplorationMap
              map={map}
              mode={mode}
              selectedRegionId={selectedRegionId}
              setSelectedRegionId={setSelectedRegionId}
              post={post}
              setNavMessage={setNavMessage}
              activeTask={activeTask}
              lastWaypoint={lastWaypoint}
              setLastWaypoint={setLastWaypoint}
              draftRegion={draftRegion}
              setDraftRegion={setDraftRegion}
              selectedRegion={selectedRegionForMap}
              updateSelectedPolygon={updateSelectedPolygon}
            />
            {navMessage ? <div className="map-status">{navMessage}</div> : null}
          </>
        ) : (
          <OfflineExplorationPlaceholder onRetry={refresh} />
        )}
      </div>
      <div className="exploration-right">
        <Panel title="Selected Region">
          <label>Label</label>
          <input value={regionLabel} onChange={(event) => editRegionLabel(event.target.value)} />
          <label>Purpose</label>
          <textarea value={regionPurpose} onChange={(event) => editRegionPurpose(event.target.value)} />
          <label>Polygon JSON</label>
          <textarea value={regionPolygon} onChange={(event) => editRegionPolygon(event.target.value)} />
          <label>Waypoints JSON</label>
          <textarea value={regionWaypoints} onChange={(event) => editRegionWaypoints(event.target.value)} />
          <div className="toolbar">
            <button className="primary" disabled={!backendOnline || !selectedRegionId} onClick={saveRegion}><Save size={16} /> Save</button>
            <button disabled={!backendOnline || !selectedRegionId} onClick={() => selectedRegionId && post("/api/region/split", { region_id: selectedRegionId })}>Split</button>
          </div>
          {regionDirty ? <p className="hint">Unsaved region edit. Save or Approve + Save Memory will commit it.</p> : null}
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
  setNavMessage,
  activeTask,
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
    if (!world) return null;
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
    if (event.target.closest?.("[data-region-id]") && !["region", "preview", "waypoint"].includes(mode)) return;
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
      setNavMessage?.(`${mode === "preview" ? "Previewing" : "Sending"} waypoint (${round3(pose.x)}, ${round3(pose.y)})...`);
      try {
        const response = await post(mode === "preview" ? "/api/nav/preview" : "/api/nav/waypoint", { pose });
        const resolvedPose = response.normalized_pose || response.requested_pose || pose;
        setLastWaypoint(resolvedPose);
        setNavMessage?.(`${mode === "preview" ? "Preview" : "Go"} ${response.status || "sent"}: ${response.reason || "Nav2 request returned."}`);
      } catch (error) {
        setNavMessage?.(`${mode === "preview" ? "Preview" : "Go"} failed: ${error.message || String(error)}`);
      }
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

  const showTrajectory = Boolean(activeTask) && !map.artifacts?.hide_exploration_trajectory;
  const trajectory = showTrajectory ? (map.trajectory || []).map((point) => {
    const p = project(point);
    return `${p.x},${p.y}`;
  }).join(" ") : "";
  const robotPose = map.robot_pose || (map.trajectory || []).slice(-1)[0];
  const robot = robotPose ? project(robotPose) : null;
  const headingLength = Math.max(map.occupancy?.resolution || 0.25, 0.25) * 3.5;
  const robotHeading = robotPose ? project({
    x: Number(robotPose.x || 0) + Math.cos(Number(robotPose.yaw || 0)) * headingLength,
    y: Number(robotPose.y || 0) + Math.sin(Number(robotPose.yaw || 0)) * headingLength,
  }) : null;
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
      {robot && robotHeading ? <line x1={robot.x} y1={robot.y} x2={robotHeading.x} y2={robotHeading.y} className="robot-heading" /> : null}
      {robot && robotHeading ? <circle cx={robotHeading.x} cy={robotHeading.y} r="4" fill="#6d0f0a" /> : null}
      {robot ? <text x={robot.x + 12} y={robot.y - 12} className="robot-label">robot</text> : null}
    </svg>
  );
}

function OfflineExplorationPlaceholder({ onRetry }) {
  return (
    <div className="offline-map">
      <div className="offline-icon"><MapIcon size={26} /></div>
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

function latestNavigationPreview(events) {
  for (let index = (events || []).length - 1; index >= 0; index -= 1) {
    const details = events[index]?.details || {};
    if (details.tool === "execute_region_exploration_plan" && details.plan?.stops) {
      return details.plan;
    }
    if (details.tool === "plan_region_exploration" && details.stops) {
      return details;
    }
    if (details.tool === "resolve_region_navigation_goal" && details.goal_pose) {
      return details;
    }
  }
  return null;
}

function latestVisionShots(events) {
  for (let index = (events || []).length - 1; index >= 0; index -= 1) {
    const details = events[index]?.details || {};
    if (details.tool !== "execute_region_exploration_plan" || !Array.isArray(details.stops)) continue;
    return details.stops.flatMap((stop) =>
      (stop.shots || []).map((shot) => ({
        ...shot,
        stop_id: stop.stop_id,
        region_label: details.region_label,
      }))
    ).filter((shot) => shot.capture);
  }
  return [];
}

function artifactSrc(url) {
  if (!url) return "";
  if (url.startsWith("data:") || url.startsWith("http://") || url.startsWith("https://")) return url;
  if (url.startsWith("/api/")) return `${AGENT_BASE}${url}`;
  return url;
}

function round3(value) {
  return Math.round(Number(value || 0) * 1000) / 1000;
}

function polygonLabelPoint(polygon) {
  if (!polygon?.length) return { x: 0, y: 0 };
  return {
    x: polygon.reduce((sum, point) => sum + Number(point[0] || 0), 0) / polygon.length,
    y: polygon.reduce((sum, point) => sum + Number(point[1] || 0), 0) / polygon.length,
  };
}

function agentPreviewBounds(memory, preview) {
  const points = [];
  const resolution = memory?.occupancy?.resolution || 0.25;
  for (const cell of memory?.occupancy?.cells || []) {
    if (cell?.state !== "free" && cell?.state !== "occupied") continue;
    const x = Number(cell.x || 0);
    const y = Number(cell.y || 0);
    points.push([x, y], [x + resolution, y + resolution]);
  }
  for (const region of memory?.regions || []) {
    for (const point of region.polygon_2d || []) points.push(point);
  }
  for (const point of preview?.path || []) points.push([point.x, point.y]);
  for (const stop of preview?.stops || []) {
    if (stop?.pose) points.push([stop.pose.x, stop.pose.y]);
    for (const shot of stop?.shots || []) {
      for (const point of [shot?.cone?.origin, shot?.cone?.left, shot?.cone?.center, shot?.cone?.right]) {
        if (point) points.push([point.x, point.y]);
      }
    }
  }
  if (preview?.next_waypoint) points.push([preview.next_waypoint.x, preview.next_waypoint.y]);
  if (preview?.goal_pose) points.push([preview.goal_pose.x, preview.goal_pose.y]);
  const startPose = memory?.start_pose?.pose || memory?.start_pose;
  if (startPose) points.push([startPose.x, startPose.y]);
  const robotPose = memory?.robot_pose || (memory?.trajectory || []).slice(-1)[0];
  if (robotPose) points.push([robotPose.x, robotPose.y]);
  if (!points.length) return mapBounds(memory);
  const minX = Math.min(...points.map((point) => Number(point[0] || 0)));
  const maxX = Math.max(...points.map((point) => Number(point[0] || 0)));
  const minY = Math.min(...points.map((point) => Number(point[1] || 0)));
  const maxY = Math.max(...points.map((point) => Number(point[1] || 0)));
  const pad = Math.max(resolution * 2, 0.16);
  return {
    min_x: minX - pad,
    max_x: maxX + pad,
    min_y: minY - pad,
    max_y: maxY + pad,
  };
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

function makeProjector(bounds, options = {}) {
  const viewport = mapViewport(bounds, options);
  return (point) => ({
    x: viewport.left + (Number(point?.x || 0) - bounds.min_x) * viewport.scale,
    y: viewport.top + (bounds.max_y - Number(point?.y || 0)) * viewport.scale,
  });
}

function mapViewport(bounds, options = {}) {
  const height = options.height || VIEW_H;
  const pad = options.pad ?? PAD;
  const worldW = Math.max(bounds.max_x - bounds.min_x, 1);
  const worldH = Math.max(bounds.max_y - bounds.min_y, 1);
  const scale = Math.min((VIEW_W - pad * 2) / worldW, (height - pad * 2) / worldH);
  const drawW = worldW * scale;
  const drawH = worldH * scale;
  const left = pad + (VIEW_W - pad * 2 - drawW) / 2;
  const top = pad + (height - pad * 2 - drawH) / 2;
  return { left, top, scale, drawW, drawH };
}

function worldFromSvg(bounds, svgX, svgY) {
  const viewport = mapViewport(bounds);
  const right = viewport.left + viewport.drawW;
  const bottom = viewport.top + viewport.drawH;
  if (svgX < viewport.left || svgX > right || svgY < viewport.top || svgY > bottom) {
    return null;
  }
  const x = bounds.min_x + (svgX - viewport.left) / viewport.scale;
  const y = bounds.max_y - (svgY - viewport.top) / viewport.scale;
  return { x, y };
}

createRoot(document.getElementById("root")).render(<App />);
