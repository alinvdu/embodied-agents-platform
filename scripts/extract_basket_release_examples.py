#!/usr/bin/env python3
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

"""Extract right-wrist frames shortly after the second gripper opening."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--settle-frames", type=int, default=30)
    return parser.parse_args()


def _second_open_frame(gripper: np.ndarray) -> int:
    state = "closed" if gripper[0] < 20.0 else "open"
    open_transitions: list[int] = []
    for index, value in enumerate(gripper):
        next_state = "open" if value > 30.0 else ("closed" if value < 10.0 else state)
        if next_state != state:
            if next_state == "open":
                open_transitions.append(index)
            state = next_state
    if len(open_transitions) < 2:
        raise RuntimeError(f"Expected two gripper openings, found {open_transitions}.")
    return open_transitions[1]


def _to_image(tensor: object) -> Image.Image:
    array = tensor.detach().cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(np.clip(array * 255.0, 0, 255).astype(np.uint8), "RGB")


def _make_contact_sheets(rows: list[dict[str, object]], output_dir: Path) -> None:
    font = ImageFont.load_default()
    tile_width, tile_height = 320, 264
    columns, page_size = 4, 20
    for page_start in range(0, len(rows), page_size):
        page_rows = rows[page_start : page_start + page_size]
        page = Image.new("RGB", (columns * tile_width, 5 * tile_height), "white")
        draw = ImageDraw.Draw(page)
        for slot, row in enumerate(page_rows):
            image = Image.open(str(row["path"])).convert("RGB").resize((320, 240))
            x = (slot % columns) * tile_width
            y = (slot // columns) * tile_height
            page.paste(image, (x, y))
            draw.text(
                (x + 6, y + 244),
                f"episode {row['episode_index']:02d} | +{row['settle_s']:.2f}s",
                fill="black",
                font=font,
            )
        page_number = page_start // page_size + 1
        page.save(output_dir / f"contact-sheet-{page_number:02d}.jpg", quality=92)


def main() -> int:
    args = _parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    args.output_dir.mkdir(parents=True, exist_ok=False)
    candidates_dir = args.output_dir / "candidates"
    candidates_dir.mkdir()

    data_files = sorted((args.dataset_root / "data").glob("chunk-*/*.parquet"))
    table = pa.concat_tables(
        [
            pq.read_table(path, columns=["episode_index", "frame_index", "index", "observation.state"])
            for path in data_files
        ]
    )
    data = table.to_pydict()
    episode_indices = np.asarray(data["episode_index"], dtype=np.int64)
    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root.resolve())
    fps = float(dataset.fps)
    rows: list[dict[str, object]] = []

    for episode_index in range(dataset.num_episodes):
        positions = np.flatnonzero(episode_indices == episode_index)
        gripper = np.asarray([data["observation.state"][index][-1] for index in positions])
        release_frame = _second_open_frame(gripper)
        sample_frame = min(release_frame + args.settle_frames, len(positions) - 1)
        source_row = int(positions[sample_frame])
        dataset_index = int(data["index"][source_row])
        item = dataset[dataset_index]
        image = _to_image(item["observation.images.right_wrist"])
        filename = f"episode-{episode_index:02d}_frame-{sample_frame:04d}.jpg"
        path = candidates_dir / filename
        image.save(path, quality=95)
        rows.append(
            {
                "episode_index": episode_index,
                "dataset_index": dataset_index,
                "release_frame": release_frame,
                "sample_frame": sample_frame,
                "release_s": release_frame / fps,
                "sample_s": sample_frame / fps,
                "settle_s": (sample_frame - release_frame) / fps,
                "path": str(path.resolve()),
            }
        )

    with (args.output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(rows, stream, indent=2)
    _make_contact_sheets(rows, args.output_dir)
    print(f"Extracted {len(rows)} right-wrist release examples to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
