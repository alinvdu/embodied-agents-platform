"""React UI served by the interactive exploration playground."""

INTERACTIVE_REACT_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Robot Exploration Mode</title>
  <style>
    :root {
      --bg: #eef2e6;
      --panel: rgba(255,255,255,0.86);
      --ink: #18230f;
      --muted: #5d6b52;
      --line: rgba(24,35,15,0.14);
      --green: #31572c;
      --leaf: #4f772d;
      --gold: #b0891f;
      --red: #a52820;
      --blue: #1d4ed8;
      --cyan: #0e7490;
      --shadow: 0 24px 54px rgba(24,35,15,0.13);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Aptos", "Segoe UI", sans-serif;
      color: var(--ink);
      min-height: 100vh;
      background:
        radial-gradient(circle at 12% 6%, rgba(79,119,45,0.24), transparent 28%),
        radial-gradient(circle at 85% 16%, rgba(176,137,31,0.22), transparent 24%),
        linear-gradient(145deg, #f6f1db 0%, #e4ecd2 50%, #d9e7da 100%);
    }
    .shell { max-width: 1760px; margin: 0 auto; padding: 24px; }
    header { display: flex; justify-content: space-between; gap: 18px; align-items: end; margin-bottom: 18px; }
    h1 { margin: 0; font-family: Georgia, serif; font-size: clamp(32px, 5vw, 62px); line-height: .9; max-width: 11ch; }
    .subtitle { color: var(--muted); max-width: 760px; }
    .layout { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 18px; align-items: start; }
    .layout > * { min-width: 0; }
    .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 16px; box-shadow: var(--shadow); backdrop-filter: blur(18px); overflow: hidden; }
    .stack { display: grid; gap: 16px; }
    .eyebrow { font-size: 12px; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin-bottom: 10px; }
    .stats { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
    .stat { border: 1px solid var(--line); border-radius: 8px; background: rgba(255,255,255,.66); padding: 11px; }
    .key { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .06em; }
    .value { font-weight: 750; margin-top: 4px; }
    button { border: 0; border-radius: 8px; padding: 11px 14px; font-weight: 750; cursor: pointer; }
    button.primary { color: white; background: var(--green); }
    button.secondary { color: var(--green); background: #f5f0d6; border: 1px solid rgba(79,119,45,.22); }
    button.danger { color: var(--red); background: #fee8e1; }
    .buttons { display: flex; flex-wrap: wrap; gap: 9px; }
    #map { width: 100%; height: 600px; border-radius: 8px; border: 1px solid var(--line); background: rgba(255,255,255,.9); touch-action: none; }
    pre, textarea { width: 100%; border: 1px solid var(--line); border-radius: 8px; background: rgba(255,255,255,.72); color: #12210f; padding: 12px; overflow: auto; }
    pre { white-space: pre-wrap; max-height: 360px; margin: 0; }
    textarea { min-height: 320px; resize: vertical; font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 12px; }
    .frontier-list { display: grid; gap: 8px; max-height: 420px; overflow: auto; }
    .frontier { border: 1px solid var(--line); border-radius: 8px; padding: 10px; background: rgba(255,255,255,.66); }
    .frontier.pending { border-color: var(--gold); background: rgba(176,137,31,.12); }
    .frontier.suppressed { opacity: .55; }
    .thumbs { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; }
    .thumbs img { width: 100%; border-radius: 8px; border: 1px solid var(--line); background: white; }
    .muted { color: var(--muted); }
    .error { color: var(--red); font-weight: 750; min-height: 1.4em; }
    .map-legend { display: flex; flex-wrap: wrap; gap: 16px; align-items: center; margin-top: 10px; color: #3f4a56; font-weight: 650; }
    .legend-item { display: inline-flex; align-items: center; gap: 7px; }
    .legend-swatch { width: 12px; height: 12px; border-radius: 3px; border: 1px solid rgba(15,23,42,.16); }
    @media (max-width: 980px) { .layout { grid-template-columns: 1fr; } #map { height: 560px; } }
  </style>
</head>
<body>
  <div id="root"></div>
  <script>
    window.INTERACTIVE_UI_FLAVOR = "__UI_FLAVOR__";
  </script>
  <script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
  <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
  <script>
    const e = React.createElement;
    const {useCallback, useEffect, useMemo, useRef, useState} = React;
    const VIEW_W = 1000;
    const VIEW_H = 760;
    const PAD = 32;

    async function requestJson(path, payload) {
      const options = payload === undefined
        ? {}
        : {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload || {})};
      const res = await fetch(path, options);
      if (!res.ok) throw new Error(await res.text());
      return await res.json();
    }

    function mapBounds(map) {
      return map?.occupancy?.bounds || {min_x: 0, max_x: 10, min_y: 0, max_y: 8};
    }

    function makeProjector(bounds) {
      const worldW = Math.max(bounds.max_x - bounds.min_x, 1);
      const worldH = Math.max(bounds.max_y - bounds.min_y, 1);
      return (point) => ({
        x: PAD + ((Number(point?.x || 0) - bounds.min_x) / worldW) * (VIEW_W - PAD * 2),
        y: VIEW_H - PAD - ((Number(point?.y || 0) - bounds.min_y) / worldH) * (VIEW_H - PAD * 2),
      });
    }

    function worldFromSvgPoint(bounds, svgX, svgY) {
      const worldW = Math.max(bounds.max_x - bounds.min_x, 1);
      const worldH = Math.max(bounds.max_y - bounds.min_y, 1);
      const nx = Math.min(Math.max((svgX - PAD) / Math.max(VIEW_W - PAD * 2, 1), 0), 1);
      const ny = Math.min(Math.max((svgY - PAD) / Math.max(VIEW_H - PAD * 2, 1), 0), 1);
      return {x: bounds.min_x + nx * worldW, y: bounds.max_y - ny * worldH};
    }

    function cellCenterPose(map, cell) {
      const resolution = map?.occupancy?.resolution || 0.25;
      return {x: (cell.cell_x + 0.5) * resolution, y: (cell.cell_y + 0.5) * resolution, yaw: 0};
    }

    function StatGrid({state, mapEditMode}) {
      const pose = state?.robot_pose || {};
      const items = [
        ['Status', state?.status || 'loading'],
        ['Coverage', state?.coverage ?? 'n/a'],
        ['Pose', `${Number(pose.x || 0).toFixed(2)}, ${Number(pose.y || 0).toFixed(2)}`],
        ['Frontiers', (state?.candidate_frontiers || []).length],
        ['Stored Memory', (state?.map?.remembered_frontiers || []).length],
        ['Manual Regions', (state?.map?.regions || []).length],
        ['Pending', state?.pending_target?.frontier_id || 'none'],
        ['Provider', state?.map?.artifacts?.llm_policy?.provider || 'unknown'],
        ['Edit Mode', mapEditMode],
      ];
      return e('div', {className: 'stats'}, items.map(([key, value]) =>
        e('div', {className: 'stat', key}, e('div', {className: 'key'}, key), e('div', {className: 'value'}, String(value)))
      ));
    }

    function FrontierList({state, onPost}) {
      const pending = state?.pending_target?.frontier_id;
      const frontiers = state?.candidate_frontiers || [];
      if (!frontiers.length) return e('div', {className: 'muted'}, 'No active frontier information.');
      return e(React.Fragment, null, frontiers.map((frontier) =>
        e('div', {
          className: `frontier ${frontier.frontier_id === pending ? 'pending' : ''} ${frontier.status === 'suppressed' ? 'suppressed' : ''}`,
          key: frontier.frontier_id,
        },
          e('strong', null, frontier.frontier_id), ` · ${frontier.status || 'unknown'}`,
          e('br'),
          `gain ${frontier.unknown_gain ?? 'n/a'} · path ${frontier.path_cost_m ?? 'n/a'}m · priority ${frontier.llm_memory_priority ?? 'n/a'}`,
          e('br'),
          e('span', {className: 'muted'}, (frontier.evidence || []).slice(0, 2).join(' | ')),
          frontier.frontier_id === pending
            ? e('div', {className: 'buttons', style: {marginTop: 8}},
                e('button', {className: 'primary', onClick: () => onPost('/api/frontier/solved')}, 'Mark Solved')
              )
            : null
        )
      ));
    }

    function NavigationMemory({regions}) {
      if (!regions.length) {
        return e('div', {className: 'muted'}, 'No manual regions yet. Click Add Region, select free cells, then Done Region.');
      }
      return e(React.Fragment, null, regions.map((region) =>
        e('div', {className: 'frontier', key: region.region_id},
          e('strong', null, region.label), ` · ${region.region_id}`,
          e('br'),
          e('span', {className: 'muted'}, region.description || ''),
          e('br'),
          ...(region.default_waypoints || []).flatMap((wp, index) => [
            index ? e('br', {key: `${wp.name}-br`}) : null,
            e('span', {key: wp.name}, `${wp.name} (${Number(wp.x || 0).toFixed(2)}, ${Number(wp.y || 0).toFixed(2)})`),
          ])
        )
      ));
    }

    function SemanticPanel({map}) {
      if (!map?.automatic_semantic_waypoints) return null;
      const memory = map.semantic_memory || {};
      const anchors = memory.anchors || [];
      const anchorById = new Map(anchors.map((anchor) => [anchor.anchor_id, anchor]));
      const places = memory.named_places || [];
      return e('section', {className: 'panel'},
        e('div', {className: 'eyebrow'}, 'Automatic Semantic Places'),
        e('div', {className: 'frontier-list'}, places.length ? places.map((place) => {
          const anchor = (place.source_anchor_ids || []).map((id) => anchorById.get(id)).find(Boolean);
          const pose = place.anchor_pose || {};
          return e('div', {className: 'frontier', key: place.place_id || place.label},
            e('strong', null, place.label), ` · ${place.status || 'unknown'} · ${Number(place.confidence || 0).toFixed(2)}`,
            e('br'),
            `anchor ${Number(pose.x || 0).toFixed(2)}, ${Number(pose.y || 0).toFixed(2)} · ${anchor?.reachability_status || 'unknown'}`,
            e('br'),
            e('span', {className: 'muted'}, (place.evidence || []).slice(0, 2).join(' | '))
          );
        }) : e('div', {className: 'muted'}, 'No semantic places yet.'))
      );
    }

    function RecentViews({frames}) {
      return e(React.Fragment, null, frames.slice(-4).map((frame) =>
        e('div', {key: frame.frame_id},
          e('img', {src: frame.thumbnail_data_url, alt: frame.frame_id || 'keyframe'}),
          e('div', {className: 'muted'}, `${frame.frame_id || 'frame'} · ${frame.description || ''}`)
        )
      ));
    }

    function MapView({
      state,
      mapEditMode,
      setUiMessage,
      selectedRegionCells,
      setSelectedRegionCells,
      regionMode,
      pendingSubwaypoint,
      setPendingSubwaypoint,
      onPost,
    }) {
      const svgRef = useRef(null);
      const paintingRef = useRef(false);
      const draggingWaypointRef = useRef(null);
      const lastPaintedCellRef = useRef(null);
      const pendingPaintCellsRef = useRef(new Map());
      const occupancyRef = useRef(new Map());
      const map = state?.map;

      const bounds = useMemo(() => mapBounds(map), [map]);
      const project = useMemo(() => makeProjector(bounds), [bounds]);
      const resolution = map?.occupancy?.resolution || 0.25;

      const cellFromEvent = useCallback((event) => {
        const svg = svgRef.current;
        if (!svg || !map) return null;
        const point = svg.createSVGPoint();
        point.x = event.clientX;
        point.y = event.clientY;
        const matrix = svg.getScreenCTM();
        if (!matrix) return null;
        const local = point.matrixTransform(matrix.inverse());
        const world = worldFromSvgPoint(bounds, local.x, local.y);
        const cell = {cell_x: Math.floor(world.x / resolution), cell_y: Math.floor(world.y / resolution)};
        cell.key = `${cell.cell_x}:${cell.cell_y}`;
        return cell;
      }, [bounds, map, resolution]);

      const flushPaint = useCallback(async () => {
        const cells = Array.from(pendingPaintCellsRef.current.values());
        pendingPaintCellsRef.current.clear();
        if (!cells.length || mapEditMode === 'none') return;
        await onPost('/api/map/edit', {mode: mapEditMode, cells});
      }, [mapEditMode, onPost]);

      const selectRegionCell = useCallback((cell) => {
        if (!cell) return;
        const state = occupancyRef.current.get(cell.key);
        if (!state || state.state !== 'free') return;
        setSelectedRegionCells((previous) => {
          const next = new Map(previous);
          next.set(cell.key, {cell_x: cell.cell_x, cell_y: cell.cell_y});
          return next;
        });
      }, [setSelectedRegionCells]);

      const enqueuePaintCell = useCallback((cell) => {
        if (!cell) return;
        if (mapEditMode === 'none') return;
        if (mapEditMode === 'clear') {
          const state = occupancyRef.current.get(cell.key);
          if (!state || (state.state !== 'occupied' && state.manual_override !== 'blocked')) return;
        }
        pendingPaintCellsRef.current.set(cell.key, {cell_x: cell.cell_x, cell_y: cell.cell_y});
      }, [mapEditMode]);

      const onPointerDown = useCallback((event) => {
        if (!map) return;
        event.preventDefault();
        const cell = cellFromEvent(event);
        if (!cell) return;
        if (regionMode === 'select') {
          paintingRef.current = true;
          selectRegionCell(cell);
          return;
        }
        if (pendingSubwaypoint) {
          setPendingSubwaypoint(null);
          const pose = cellCenterPose(map, cell);
          onPost('/api/region/subwaypoint', {
            region_id: pendingSubwaypoint.region_id,
            name: pendingSubwaypoint.name,
            pose,
          });
          return;
        }
        if (mapEditMode === 'none') return;
        paintingRef.current = true;
        lastPaintedCellRef.current = cell.key;
        enqueuePaintCell(cell);
      }, [cellFromEvent, enqueuePaintCell, map, onPost, pendingSubwaypoint, regionMode, selectRegionCell, setPendingSubwaypoint]);

      const onPointerMove = useCallback((event) => {
        if (!paintingRef.current) return;
        event.preventDefault();
        const cell = cellFromEvent(event);
        if (!cell) return;
        if (regionMode === 'select') {
          selectRegionCell(cell);
          return;
        }
        if (cell.key === lastPaintedCellRef.current) return;
        lastPaintedCellRef.current = cell.key;
        enqueuePaintCell(cell);
      }, [cellFromEvent, enqueuePaintCell, regionMode, selectRegionCell]);

      const finishPointerAction = useCallback(async (event) => {
        if (!map) return;
        if (draggingWaypointRef.current) {
          const cell = cellFromEvent(event);
          if (!cell) {
            draggingWaypointRef.current = null;
            return;
          }
          const payload = {...draggingWaypointRef.current, pose: cellCenterPose(map, cell)};
          draggingWaypointRef.current = null;
          await onPost('/api/region/waypoint', payload);
          return;
        }
        if (!paintingRef.current) return;
        event?.preventDefault?.();
        paintingRef.current = false;
        lastPaintedCellRef.current = null;
        if (regionMode === 'select') return;
        await flushPaint();
      }, [cellFromEvent, flushPaint, map, onPost, regionMode]);

      if (!map) {
        return e('svg', {id: 'map', viewBox: `0 0 ${VIEW_W} ${VIEW_H}`},
          e('text', {x: 40, y: 60, fill: '#52606d', fontSize: 22}, 'No map yet.')
        );
      }

      occupancyRef.current = new Map();
      const occupancyCells = (map.occupancy?.cells || []).map((cell) => {
        const cellX = Math.floor(Number(cell.x) / resolution);
        const cellY = Math.floor(Number(cell.y) / resolution);
        occupancyRef.current.set(`${cellX}:${cellY}`, {state: cell.state, manual_override: cell.manual_override || null});
        const p = project({x: cell.x, y: cell.y});
        const p2 = project({x: Number(cell.x) + resolution, y: Number(cell.y) + resolution});
        const fill = cell.manual_override === 'blocked'
          ? 'rgba(15,23,42,.92)'
          : cell.manual_override === 'cleared'
            ? 'rgba(20,184,166,.30)'
            : cell.state === 'occupied'
              ? 'rgba(15,23,42,.82)'
              : 'rgba(20,184,166,.30)';
        return e('rect', {
          key: `${cellX}:${cellY}`,
          x: p.x,
          y: p2.y,
          width: Math.max(2, p2.x - p.x),
          height: Math.max(2, p.y - p2.y),
          fill,
        });
      });

      const trajectory = (map.trajectory || []).map((point) => {
        const p = project(point);
        return `${p.x},${p.y}`;
      }).join(' ');

      const plannedNavPath = (map.artifacts?.planned_nav_path || []).map((point) => {
        const p = project(point);
        return `${p.x},${p.y}`;
      }).join(' ');

      const regionColors = ['#2563eb', '#0891b2', '#9333ea', '#db2777', '#0f766e'];
      const regionCells = (map.regions || []).flatMap((region, index) =>
        (region.selected_cells || []).map((cell) => {
          const x = Number(cell.cell_x) * resolution;
          const y = Number(cell.cell_y) * resolution;
          const p = project({x, y});
          const p2 = project({x: x + resolution, y: y + resolution});
          return e('rect', {
            key: `${region.region_id}:${cell.cell_x}:${cell.cell_y}`,
            x: p.x,
            y: p2.y,
            width: Math.max(2, p2.x - p.x),
            height: Math.max(2, p.y - p2.y),
            fill: regionColors[index % regionColors.length],
            opacity: 0.28,
          }, e('title', null, region.label));
        })
      );

      const selectedCells = Array.from(selectedRegionCells.values()).map((cell) => {
        const x = Number(cell.cell_x) * resolution;
        const y = Number(cell.cell_y) * resolution;
        const p = project({x, y});
        const p2 = project({x: x + resolution, y: y + resolution});
        return e('rect', {
          key: `selected:${cell.cell_x}:${cell.cell_y}`,
          x: p.x,
          y: p2.y,
          width: Math.max(2, p2.x - p.x),
          height: Math.max(2, p.y - p2.y),
          fill: '#f59e0b',
          opacity: 0.46,
        });
      });

      const remembered = (map.remembered_frontiers || []).map((frontier) => {
        const p = project(frontier.nav_pose);
        const b = frontier.frontier_boundary_pose ? project(frontier.frontier_boundary_pose) : p;
        return e('g', {key: `remembered:${frontier.frontier_id}`},
          e('circle', {cx: b.x, cy: b.y, r: 3.5, fill: 'none', stroke: '#647067', strokeWidth: 1.5, opacity: 0.38}),
          e('line', {x1: b.x, y1: b.y, x2: p.x, y2: p.y, stroke: '#647067', strokeWidth: 1, strokeDasharray: '2 5', opacity: 0.28}),
          e('circle', {cx: p.x, cy: p.y, r: 5, fill: '#647067', opacity: 0.32}),
          e('text', {x: p.x + 8, y: p.y + 14, fontSize: 11, fill: '#647067', opacity: 0.7}, `${frontier.frontier_id} memory`)
        );
      });

      const frontiers = (map.frontiers || []).map((frontier) => {
        const p = project(frontier.nav_pose);
        const b = frontier.frontier_boundary_pose ? project(frontier.frontier_boundary_pose) : p;
        const selected = frontier.frontier_id === state?.pending_target?.frontier_id;
        const color = frontier.status === 'suppressed' ? '#71717a' : selected ? '#b0891f' : '#31572c';
        return e('g', {key: `frontier:${frontier.frontier_id}`},
          e('circle', {cx: b.x, cy: b.y, r: 4, fill: 'none', stroke: color, strokeWidth: 2, opacity: 0.72}),
          e('line', {x1: b.x, y1: b.y, x2: p.x, y2: p.y, stroke: color, strokeWidth: 1.5, strokeDasharray: '4 4', opacity: 0.5}),
          e('circle', {cx: p.x, cy: p.y, r: selected ? 11 : 7, fill: color, opacity: selected ? 1 : 0.76}),
          e('text', {x: p.x + 10, y: p.y - 8, fontSize: 12, fill: color}, frontier.frontier_id)
        );
      });

      const manualWaypoints = (map.regions || []).flatMap((region) =>
        (region.default_waypoints || []).map((waypoint) => {
          const p = project(waypoint);
          const primary = (waypoint.kind || 'primary') === 'primary';
          return e('g', {
            key: `${region.region_id}:${waypoint.name}`,
            onPointerDown: (event) => {
              event.stopPropagation();
              setUiMessage('Drag to a known-free cell and release.');
              draggingWaypointRef.current = {region_id: region.region_id, waypoint_name: waypoint.name};
            },
            style: {cursor: 'grab'},
          },
            e('circle', {cx: p.x, cy: p.y, r: primary ? 8 : 6, fill: primary ? '#1d4ed8' : '#0e7490'}),
            e('text', {x: p.x + 10, y: p.y + 4, fontSize: 12, fill: '#1e3a8a', fontWeight: 800}, waypoint.name)
          );
        })
      );

      const semanticMemory = map.automatic_semantic_waypoints ? (map.semantic_memory || {}) : {};
      const semanticEvidence = (semanticMemory.evidence || []).map((ev) => {
        const p = project(ev.evidence_pose || {x: 0, y: 0});
        return e('circle', {key: ev.evidence_id, cx: p.x, cy: p.y, r: 5, fill: '#7c3aed', opacity: 0.72},
          e('title', null, `${ev.label_hint || ''} evidence`)
        );
      });
      const semanticPlaces = (semanticMemory.named_places || []).map((place) => {
        const anchor = project(place.anchor_pose || {x: 0, y: 0});
        const evidence = place.evidence_pose ? project(place.evidence_pose) : null;
        return e('g', {key: place.place_id || place.label},
          evidence ? e('line', {x1: evidence.x, y1: evidence.y, x2: anchor.x, y2: anchor.y, stroke: '#7c3aed', strokeWidth: 1.6, strokeDasharray: '3 5', opacity: 0.58}) : null,
          e('rect', {x: anchor.x - 7, y: anchor.y - 7, width: 14, height: 14, rx: 3, fill: '#7c3aed', opacity: 0.9}),
          e('text', {x: anchor.x + 10, y: anchor.y + 4, fontSize: 12, fill: '#4c1d95', fontWeight: 800}, place.label || '')
        );
      });

      const robot = project(state?.robot_pose || {x: 0, y: 0});
      const yaw = Number(state?.robot_pose?.yaw || 0);
      const heading = project({
        x: Number(state?.robot_pose?.x || 0) + Math.cos(yaw) * Math.max(resolution * 2.5, 0.45),
        y: Number(state?.robot_pose?.y || 0) + Math.sin(yaw) * Math.max(resolution * 2.5, 0.45),
      });

      return e('svg', {
        id: 'map',
        ref: svgRef,
        viewBox: `0 0 ${VIEW_W} ${VIEW_H}`,
        onPointerDown,
        onPointerMove,
        onPointerUp: finishPointerAction,
        onPointerLeave: finishPointerAction,
      },
        e('rect', {width: VIEW_W, height: VIEW_H, fill: 'rgba(255,255,255,.92)'}),
        occupancyCells,
        trajectory ? e('polyline', {points: trajectory, fill: 'none', stroke: '#4f772d', strokeWidth: 4, strokeLinecap: 'round'}) : null,
        plannedNavPath ? e('polyline', {points: plannedNavPath, fill: 'none', stroke: '#f59e0b', strokeWidth: 3, strokeLinecap: 'round', strokeDasharray: '8,4'}) : null,
        regionCells,
        selectedCells,
        remembered,
        manualWaypoints,
        semanticEvidence,
        semanticPlaces,
        frontiers,
        e('circle', {cx: robot.x, cy: robot.y, r: 13, fill: '#a52820'}),
        e('line', {x1: robot.x, y1: robot.y, x2: heading.x, y2: heading.y, stroke: '#6d0f0a', strokeWidth: 5, strokeLinecap: 'round'}),
        e('circle', {cx: heading.x, cy: heading.y, r: 4.5, fill: '#6d0f0a'}),
        e('text', {x: robot.x + 14, y: robot.y - 12, fill: '#a52820', fontWeight: 800}, 'robot')
      );
    }

    function App() {
      const [state, setState] = useState(null);
      const [mapEditMode, setMapEditMode] = useState('none');
      const [uiMessage, setUiMessage] = useState('');
      const [regionMode, setRegionMode] = useState(null);
      const [selectedRegionCells, setSelectedRegionCells] = useState(new Map());
      const [pendingSubwaypoint, setPendingSubwaypoint] = useState(null);
      const [showMapEditing, setShowMapEditing] = useState(false);
      const [showRegionEditing, setShowRegionEditing] = useState(false);
      const autoStepInFlight = useRef(false);

      const refresh = useCallback(async () => {
        const next = await requestJson('/api/state');
        setState(next);
      }, []);

      const post = useCallback(async (path, payload) => {
        const next = await requestJson(path, payload || {});
        setState(next);
      }, []);

      useEffect(() => {
        refresh().catch((err) => setUiMessage(err.message));
        const timer = setInterval(() => refresh().catch((err) => setUiMessage(err.message)), 1000);
        return () => clearInterval(timer);
      }, [refresh]);

      useEffect(() => {
        const status = state?.status;
        if (!status || autoStepInFlight.current) return;
        if (!['initial_scan_complete', 'waiting_for_llm', 'llm_response_ready'].includes(status)) return;
        autoStepInFlight.current = true;
        requestJson('/api/auto_explore', {})
          .then((next) => setState(next))
          .catch((err) => setUiMessage(err.message))
          .finally(() => { autoStepInFlight.current = false; });
      }, [state?.status, state?.session, state?.pending_target?.frontier_id]);

      const startRegion = () => {
        setRegionMode('select');
        setSelectedRegionCells(new Map());
        setUiMessage('Select free cells for the region, then click Done Region.');
      };

      const finishRegion = async () => {
        if (!selectedRegionCells.size) {
          setUiMessage('Select at least one free cell before finishing the region.');
          return;
        }
        const label = prompt('Region name');
        if (!label) return;
        const description = prompt('Region description') || '';
        const cells = Array.from(selectedRegionCells.values());
        setRegionMode(null);
        setSelectedRegionCells(new Map());
        await post('/api/region/create', {label, description, cells});
      };

      const startSubwaypoint = () => {
        const regions = state?.map?.regions || [];
        if (!regions.length) {
          setUiMessage('Create a region before adding subwaypoints.');
          return;
        }
        const region_id = prompt(`Region id for subwaypoint:\\n${regions.map((r) => `${r.region_id}: ${r.label}`).join('\\n')}`);
        if (!region_id) return;
        const name = prompt('Subwaypoint name') || 'subwaypoint';
        setPendingSubwaypoint({region_id, name});
        setUiMessage('Click a free map cell to place the subwaypoint.');
      };

      const lastError = state?.last_error || uiMessage || '';
      const map = state?.map || {};
      const hasStarted = state?.status && state.status !== 'not_started';
      const canWebManualControl = !!state?.capabilities?.web_manual_control;
      const manualDrive = (direction) => post('/api/manual_drive', {direction});

      return e('div', {className: 'shell'},
        e('header', null,
          e('div', null,
            e('h1', null, 'Robot Exploration Mode')
          )
        ),
        e('div', {className: 'layout'},
          e('div', {className: 'stack left-column'},
            e('section', {className: 'panel'},
              e('div', {className: 'eyebrow'}, 'Controls'),
              e('div', {className: 'buttons'},
                e('button', {className: hasStarted ? 'secondary' : 'primary', onClick: () => post('/api/reset')}, hasStarted ? 'Reset + Scan' : 'Start Explore'),
                e('button', {className: 'secondary', onClick: () => post('/api/pause')}, 'Pause'),
                e('button', {className: 'secondary', onClick: () => post('/api/resume')}, 'Resume'),
                e('button', {className: 'primary', onClick: () => post('/api/control_robot')}, 'Control Robot')
              ),
              canWebManualControl ? e('div', {className: 'buttons', style: {marginTop: 10}},
                e('button', {className: 'secondary', onClick: () => manualDrive('forward')}, 'Forward'),
                e('button', {className: 'secondary', onClick: () => manualDrive('left')}, 'Left'),
                e('button', {className: 'secondary', onClick: () => manualDrive('right')}, 'Right'),
                e('button', {className: 'secondary', onClick: () => manualDrive('back')}, 'Back'),
                e('button', {className: 'danger', onClick: () => post('/api/manual_stop')}, 'Stop'),
                e('button', {className: 'primary', onClick: () => post('/api/manual_scan')}, 'Scan 360')
              ) : null,
              e('div', {className: 'buttons', style: {marginTop: 10}},
                e('button', {className: 'secondary', onClick: () => setShowMapEditing((value) => {
                  if (value) setMapEditMode('none');
                  return !value;
                })}, 'Map Editing'),
                e('button', {className: 'secondary', onClick: () => setShowRegionEditing((value) => {
                  if (value) {
                    setRegionMode(null);
                    setSelectedRegionCells(new Map());
                  }
                  return !value;
                })}, 'Region Edit')
              ),
              showMapEditing ? e('div', {className: 'buttons', style: {marginTop: 10}},
                e('button', {className: 'secondary', onClick: () => setMapEditMode('block')}, 'Draw Wall'),
                e('button', {className: 'secondary', onClick: () => setMapEditMode('clear')}, 'Erase Wall'),
                e('button', {className: 'secondary', onClick: () => setMapEditMode('reset')}, 'Reset Cell')
              ) : null,
              showRegionEditing ? e('div', {className: 'buttons', style: {marginTop: 10}},
                e('button', {className: 'secondary', onClick: startRegion}, 'Add Region'),
                e('button', {className: 'primary', onClick: finishRegion}, 'Done Region'),
                e('button', {className: 'secondary', onClick: startSubwaypoint}, 'Add Subwaypoint')
              ) : null,
              e('div', {className: 'error'}, lastError)
            ),
            e('section', {className: 'panel'},
              e('div', {className: 'eyebrow'}, 'State'),
              e(StatGrid, {state, mapEditMode})
            ),
            e('section', {className: 'panel'},
              e('div', {className: 'eyebrow'}, 'Frontier Information'),
              e('div', {className: 'frontier-list'}, e(FrontierList, {state, onPost: post}))
            ),
            e('section', {className: 'panel'},
              e('div', {className: 'eyebrow'}, 'Navigation Memory'),
              e('div', {className: 'frontier-list'}, e(NavigationMemory, {regions: map.regions || []}))
            ),
            e(SemanticPanel, {map})
          ),
          e('div', {className: 'stack right-column'},
            e('section', {className: 'panel'},
              e('div', {className: 'eyebrow'}, 'Scanned 2D Map'),
              e(MapView, {
                state,
                mapEditMode,
                setUiMessage,
                selectedRegionCells,
                setSelectedRegionCells,
                regionMode,
                pendingSubwaypoint,
                setPendingSubwaypoint,
                onPost: post,
              }),
              e('div', {className: 'map-legend'},
                e('span', {className: 'legend-item'}, e('span', {className: 'legend-swatch', style: {background: 'rgba(20,184,166,.30)'}}), 'known free'),
                e('span', {className: 'legend-item'}, e('span', {className: 'legend-swatch', style: {background: 'rgba(15,23,42,.82)'}}), 'occupied'),
                e('span', {className: 'legend-item'}, e('span', {className: 'legend-swatch', style: {background: '#4f772d'}}), 'trajectory'),
                e('span', {className: 'legend-item'}, e('span', {className: 'legend-swatch', style: {background: '#f59e0b'}}), 'planned path')
              )
            )
          )
        )
      );
    }

    ReactDOM.createRoot(document.getElementById('root')).render(e(App));
  </script>
</body>
</html>
"""
