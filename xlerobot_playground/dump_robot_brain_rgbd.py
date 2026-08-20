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

import argparse
import json
from pathlib import Path
from urllib.request import urlopen

from xlerobot_playground.rgbd_transport import unpack_rgbd_frame


def _depth_values_mm(depth_be: bytes) -> list[int]:
    if len(depth_be) % 2 != 0:
        raise ValueError("Depth payload length is not divisible by 2.")
    return [int.from_bytes(depth_be[index : index + 2], "big") for index in range(0, len(depth_be), 2)]


def _percentile(values: list[int], fraction: float) -> int:
    finite = sorted(value for value in values if value > 0)
    if not finite:
        return 1
    index = int(max(0, min(len(finite) - 1, round((len(finite) - 1) * fraction))))
    return finite[index]


def _depth_grayscale_rgb(depth_be: bytes) -> bytes:
    values = _depth_values_mm(depth_be)
    near = _percentile(values, 0.02)
    far = max(_percentile(values, 0.98), near + 1)
    pixels = bytearray()
    for value in values:
        if value <= 0:
            shade = 0
        else:
            normalized = max(0.0, min(1.0, (value - near) / (far - near)))
            shade = int(round(255.0 * (1.0 - normalized)))
        pixels.extend((shade, shade, shade))
    return bytes(pixels)


def _side_by_side_ppm(*, rgb: bytes, depth_rgb: bytes, width: int, height: int) -> bytes:
    row_bytes = width * 3
    rows = [f"P6\n{width * 2} {height}\n255\n".encode("ascii")]
    for row in range(height):
        start = row * row_bytes
        end = start + row_bytes
        rows.append(rgb[start:end] + depth_rgb[start:end])
    return b"".join(rows)


def dump_frame(*, url: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=5.0) as response:
        payload = response.read()
    frame = unpack_rgbd_frame(payload)
    if frame.depth_width != frame.rgb_width or frame.depth_height != frame.rgb_height:
        raise ValueError(
            "RGB and depth dimensions differ: "
            f"rgb={frame.rgb_width}x{frame.rgb_height} "
            f"depth={frame.depth_width}x{frame.depth_height}"
        )

    rgb_path = output_dir / "rgb.ppm"
    depth_path = output_dir / "depth_mm.pgm"
    depth_preview_path = output_dir / "depth_preview.ppm"
    side_by_side_path = output_dir / "rgb_depth_side_by_side.ppm"
    metadata_path = output_dir / "metadata.json"

    rgb_path.write_bytes(f"P6\n{frame.rgb_width} {frame.rgb_height}\n255\n".encode("ascii") + frame.rgb)
    depth_path.write_bytes(
        f"P5\n{frame.depth_width} {frame.depth_height}\n65535\n".encode("ascii") + frame.depth_be
    )
    depth_rgb = _depth_grayscale_rgb(frame.depth_be)
    depth_preview_path.write_bytes(
        f"P6\n{frame.depth_width} {frame.depth_height}\n255\n".encode("ascii") + depth_rgb
    )
    side_by_side_path.write_bytes(
        _side_by_side_ppm(rgb=frame.rgb, depth_rgb=depth_rgb, width=frame.rgb_width, height=frame.rgb_height)
    )
    metadata_path.write_text(
        json.dumps(
            {
                "frame_index": frame.frame_index,
                "timestamp_s": frame.timestamp_s,
                "rgb_width": frame.rgb_width,
                "rgb_height": frame.rgb_height,
                "depth_width": frame.depth_width,
                "depth_height": frame.depth_height,
                "metadata": frame.metadata,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote RGB: {rgb_path}")
    print(f"Wrote raw depth: {depth_path}")
    print(f"Wrote depth preview: {depth_preview_path}")
    print(f"Wrote side-by-side preview: {side_by_side_path}")
    print(f"Wrote metadata: {metadata_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dump one paired RGB-D frame from robot_brain_agent.")
    parser.add_argument("--url", default="http://127.0.0.1:8765/rgbd")
    parser.add_argument("--output-dir", default="artifacts/rgbd_snapshot")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dump_frame(url=args.url, output_dir=Path(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
