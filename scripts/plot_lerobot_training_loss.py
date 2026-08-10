#!/usr/bin/env python3
"""Plot LeRobot loss logs as SVG or serve an auto-refreshing local dashboard."""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


LOG_PATTERN = re.compile(
    r"INFO (?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*"
    r"step:(?P<step>\d+K?|\d+M).*loss:(?P<loss>\d+(?:\.\d+)?)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, help="Write a static SVG file.")
    parser.add_argument("--serve-port", type=int, help="Serve a live dashboard on localhost.")
    parser.add_argument("--refresh-seconds", type=float, default=3.0)
    parser.add_argument("--log-freq", type=int, default=1_000)
    parser.add_argument("--smooth-steps", type=int, default=5_000)
    args = parser.parse_args()
    if args.output is None and args.serve_port is None:
        parser.error("at least one of --output or --serve-port is required")
    if args.log_freq <= 0 or args.smooth_steps <= 0:
        parser.error("--log-freq and --smooth-steps must be positive")
    return args


def read_losses(paths: list[Path]) -> list[float]:
    by_timestamp: dict[datetime, float] = {}
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(errors="replace").splitlines():
            match = LOG_PATTERN.search(line)
            if match:
                timestamp = datetime.strptime(match.group("timestamp"), "%Y-%m-%d %H:%M:%S")
                by_timestamp[timestamp] = float(match.group("loss"))
    return [loss for _, loss in sorted(by_timestamp.items())]


def block_average(values: list[float], block_size: int) -> list[float]:
    """Average complete, non-overlapping blocks of logged values."""
    return [
        sum(values[index : index + block_size]) / block_size
        for index in range(0, len(values) - block_size + 1, block_size)
    ]


def polyline(points: list[tuple[float, float]], color: str, width: float, opacity: float = 1) -> str:
    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return (
        f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{width}" '
        f'stroke-opacity="{opacity}" stroke-linejoin="round"/>'
    )


def circles(points: list[tuple[float, float, int, float]], color: str, label: str) -> str:
    return "\n".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}" stroke="white" stroke-width="1.5" '
        f'data-tooltip="{label} at step {step:,}: {value:.6f}">'
        f'<title>{label} at step {step:,}: {value:.6f}</title></circle>'
        for x, y, step, value in points
    )


def panel(
    steps: list[int],
    losses: list[float],
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    min_step: int,
    max_step: int,
    max_loss: float,
    title: str,
    color: str,
    point_label: str,
    empty_message: str | None = None,
) -> str:
    left, right, top, bottom = 70, 25, 35, 45
    plot_x, plot_y = x + left, y + top
    plot_w, plot_h = width - left - right, height - top - bottom
    max_step = max(max_step, min_step + 1)
    max_loss = max(max_loss, 1e-6)

    def sx(step: int) -> float:
        return plot_x + (step - min_step) / (max_step - min_step) * plot_w

    def sy(loss: float) -> float:
        return plot_y + plot_h - loss / max_loss * plot_h

    parts = [
        f'<text x="{x + width / 2}" y="{y + 21}" text-anchor="middle" class="panel-title">{title}</text>',
        f'<rect x="{plot_x}" y="{plot_y}" width="{plot_w}" height="{plot_h}" class="plot-bg"/>',
    ]
    for tick in range(6):
        loss = max_loss * tick / 5
        tick_y = sy(loss)
        parts.append(f'<line x1="{plot_x}" y1="{tick_y:.1f}" x2="{plot_x + plot_w}" y2="{tick_y:.1f}" class="grid"/>')
        parts.append(f'<text x="{plot_x - 9}" y="{tick_y + 4:.1f}" text-anchor="end" class="tick">{loss:.3f}</text>')
    for tick in range(6):
        step = round(min_step + (max_step - min_step) * tick / 5)
        tick_x = sx(step)
        parts.append(f'<line x1="{tick_x:.1f}" y1="{plot_y}" x2="{tick_x:.1f}" y2="{plot_y + plot_h}" class="grid"/>')
        parts.append(f'<text x="{tick_x:.1f}" y="{plot_y + plot_h + 23}" text-anchor="middle" class="tick">{step / 1000:g}K</text>')
    for checkpoint in range(10_000, max_step + 1, 10_000):
        if checkpoint >= min_step:
            checkpoint_x = sx(checkpoint)
            parts.append(f'<line x1="{checkpoint_x:.1f}" y1="{plot_y}" x2="{checkpoint_x:.1f}" y2="{plot_y + plot_h}" class="checkpoint"/>')

    points = [(sx(step), sy(loss), step, loss) for step, loss in zip(steps, losses) if step >= min_step]
    if points:
        parts.append(polyline([(x, y) for x, y, _, _ in points], color, 2.5, 0.8))
        parts.append(circles(points, color, point_label))
    elif empty_message:
        parts.append(
            f'<text x="{plot_x + plot_w / 2}" y="{plot_y + plot_h / 2}" '
            f'text-anchor="middle" class="empty">{empty_message}</text>'
        )
    parts.append(f'<text x="{x + 17}" y="{plot_y + plot_h / 2}" text-anchor="middle" class="axis-label" transform="rotate(-90 {x + 17} {plot_y + plot_h / 2})">Loss</text>')
    parts.append(f'<text x="{plot_x + plot_w / 2}" y="{y + height - 5}" text-anchor="middle" class="axis-label">Training step</text>')
    return "\n".join(parts)


def waiting_svg() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="700" viewBox="0 0 1200 700">
<rect width="1200" height="700" fill="#f7f9fb"/>
<text x="600" y="330" text-anchor="middle" font-family="Arial, sans-serif" font-size="28" fill="#1f2933">Waiting for the first loss entry...</text>
<text x="600" y="370" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" fill="#52606d">The graph will update automatically when training reaches the first log interval.</text>
</svg>'''


def render_svg(losses: list[float], log_freq: int, smooth_steps_count: int) -> str:
    if not losses:
        return waiting_svg()

    average_block_size = max(1, round(smooth_steps_count / log_freq))
    steps = [(index + 1) * log_freq for index in range(len(losses))]
    averages = block_average(losses, average_block_size)
    average_steps = [(index + 1) * average_block_size * log_freq for index in range(len(averages))]
    max_step = steps[-1]
    latest_average = f"{averages[-1]:.4f}" if averages else "pending"

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="900" viewBox="0 0 1200 900">
<style>
  text {{ font-family: Inter, Arial, sans-serif; fill: #1f2933; }}
  .title {{ font-size: 24px; font-weight: 700; }}
  .subtitle {{ font-size: 14px; fill: #52606d; }}
  .metric {{ font-size: 15px; font-weight: 600; }}
  .panel-title {{ font-size: 17px; font-weight: 600; }}
  .tick {{ font-size: 12px; fill: #52606d; }}
  .axis-label {{ font-size: 13px; font-weight: 600; }}
  .empty {{ font-size: 16px; fill: #52606d; }}
  .plot-bg {{ fill: #fbfcfd; stroke: #cbd2d9; }}
  .grid {{ stroke: #d9e2ec; stroke-width: 1; }}
  .checkpoint {{ stroke: #616e7c; stroke-width: 1; stroke-dasharray: 5 5; opacity: 0.5; }}
</style>
<rect width="1200" height="900" fill="white"/>
<text x="600" y="32" text-anchor="middle" class="title">SmolVLA live training loss</text>
<text x="600" y="55" text-anchor="middle" class="subtitle">Exact {log_freq:,}-step log points (blue), non-overlapping {smooth_steps_count:,}-step averages (red)</text>
<text x="360" y="78" text-anchor="middle" class="metric">Current step: {max_step:,}</text>
<text x="600" y="78" text-anchor="middle" class="metric">Latest loss: {losses[-1]:.4f}</text>
<text x="840" y="78" text-anchor="middle" class="metric">Latest {smooth_steps_count // 1000}K average: {latest_average}</text>
{panel(steps, losses, x=20, y=90, width=1160, height=375, min_step=0, max_step=max_step, max_loss=max(losses) * 1.05, title=f"Loss logged every {log_freq // 1000}K steps", color="#2389b7", point_label=f"{log_freq // 1000}K loss")}
{panel(average_steps, averages, x=20, y=485, width=1160, height=375, min_step=0, max_step=max_step, max_loss=max(averages, default=max(losses)) * 1.05, title=f"Average loss for each {smooth_steps_count // 1000}K-step block", color="#d13f4c", point_label=f"{smooth_steps_count // 1000}K block average", empty_message=f"First average appears at step {smooth_steps_count:,}")}
</svg>'''


def dashboard_html(refresh_seconds: float) -> str:
    refresh_ms = max(500, round(refresh_seconds * 1000))
    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SmolVLA Live Loss</title>
  <style>
    body {{ margin: 0; background: #eef2f6; font-family: Arial, sans-serif; }}
    main {{ max-width: 1220px; margin: 20px auto; padding: 0 16px; }}
    #chart {{ width: 100%; aspect-ratio: 4 / 3; background: white; border: 1px solid #cbd2d9; border-radius: 6px; overflow: hidden; }}
    #chart svg {{ display: block; width: 100%; height: auto; }}
    #tooltip {{ position: fixed; display: none; z-index: 10; pointer-events: none; padding: 7px 10px; color: white; background: #1f2933; border-radius: 4px; font-size: 13px; box-shadow: 0 2px 8px rgb(0 0 0 / 25%); }}
    p {{ color: #52606d; font-size: 13px; text-align: right; }}
  </style>
</head>
<body>
  <main>
    <div id="chart" aria-label="Live SmolVLA loss graph"></div>
    <div id="tooltip" role="tooltip"></div>
    <p>Auto-refreshes every {refresh_seconds:g} seconds</p>
  </main>
  <script>
    const chart = document.getElementById("chart");
    const tooltip = document.getElementById("tooltip");

    async function refreshChart() {{
      const response = await fetch("/graph.svg?t=" + Date.now(), {{ cache: "no-store" }});
      chart.innerHTML = await response.text();
    }}

    chart.addEventListener("mousemove", event => {{
      const point = event.target.closest("circle[data-tooltip]");
      if (!point) {{
        tooltip.style.display = "none";
        return;
      }}
      tooltip.textContent = point.dataset.tooltip;
      tooltip.style.display = "block";
      tooltip.style.left = Math.max(8, event.clientX - tooltip.offsetWidth - 14) + "px";
      tooltip.style.top = (event.clientY + 14) + "px";
    }});
    chart.addEventListener("mouseleave", () => {{ tooltip.style.display = "none"; }});

    refreshChart();
    setInterval(refreshChart, {refresh_ms});
  </script>
</body>
</html>'''


def serve_dashboard(args: argparse.Namespace) -> None:
    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/" or self.path.startswith("/index.html"):
                body = dashboard_html(args.refresh_seconds).encode()
                content_type = "text/html; charset=utf-8"
            elif self.path.startswith("/graph.svg"):
                body = render_svg(read_losses(args.logs), args.log_freq, args.smooth_steps).encode()
                content_type = "image/svg+xml; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *values: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", args.serve_port), DashboardHandler)
    print(f"Live loss dashboard: http://127.0.0.1:{args.serve_port}")
    print("Press Ctrl+C to stop the dashboard; training is unaffected.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    args = parse_args()
    if args.output is not None:
        losses = read_losses(args.logs)
        if not losses:
            raise ValueError("No training loss entries found")
        svg = render_svg(losses, args.log_freq, args.smooth_steps)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(svg)
        print(f"Saved {args.output} ({len(losses)} points, through step {len(losses) * args.log_freq})")
    if args.serve_port is not None:
        serve_dashboard(args)


if __name__ == "__main__":
    main()
