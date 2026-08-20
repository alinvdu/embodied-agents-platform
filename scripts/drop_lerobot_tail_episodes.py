#!/usr/bin/env python
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
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a repaired LeRobot v3 dataset without one or more trailing episodes. "
            "The source is never modified. Shared video shards are trimmed at the last "
            "retained frame so the result can be resumed safely."
        )
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--drop-count", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = Path(args.source).expanduser().resolve()
    destination = Path(args.destination).expanduser().resolve()
    _validate_paths(source, destination, drop_count=args.drop_count)

    source_info = _read_json(source / "meta/info.json")
    source_data = _read_parquet_tree(source / "data")
    source_episodes = _read_parquet_tree(source / "meta/episodes")
    total_episodes = int(source_info["total_episodes"])
    retained_count = total_episodes - args.drop_count
    retained_indices = list(range(retained_count))
    _validate_contiguous_episodes(source_data, source_episodes, total_episodes)

    build_root = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.building-", dir=destination.parent)
    )
    try:
        _hardlink_tree(source, build_root)
        retained_data = _filter_table(source_data, retained_count)
        retained_episodes = _filter_table(source_episodes, retained_count)
        _rewrite_filtered_shards(
            source / "data",
            build_root / "data",
            retained_count=retained_count,
        )
        _rewrite_filtered_shards(
            source / "meta/episodes",
            build_root / "meta/episodes",
            retained_count=retained_count,
        )
        _rewrite_info_and_stats(
            build_root,
            source_info=source_info,
            retained_data=retained_data,
            retained_episodes=retained_episodes,
        )
        trimmed_videos = _trim_video_tails(
            source,
            build_root,
            source_info=source_info,
            retained_episodes=retained_episodes,
            source_episodes=source_episodes,
        )
        summary = validate_repaired_dataset(
            build_root,
            expected_episodes=retained_count,
            expected_indices=retained_indices,
        )
        _write_json(
            build_root / "meta/tail_episode_repair.json",
            {
                "source": str(source),
                "dropped_episode_indices": list(range(retained_count, total_episodes)),
                "retained_episodes": retained_count,
                "retained_frames": retained_data.num_rows,
                "trimmed_video_files": trimmed_videos,
            },
        )
        os.replace(build_root, destination)
    except Exception:
        shutil.rmtree(build_root, ignore_errors=True)
        raise

    print(f"Created repaired dataset: {destination}")
    print(
        f"episodes={summary['episodes']} frames={summary['frames']} "
        f"trimmed_videos={trimmed_videos}"
    )
    return 0


def _validate_paths(source: Path, destination: Path, *, drop_count: int) -> None:
    if source == destination:
        raise ValueError("Source and destination must be different directories.")
    if not (source / "meta/info.json").is_file():
        raise FileNotFoundError(f"Not a LeRobot v3 dataset root: {source}")
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    if drop_count <= 0:
        raise ValueError("--drop-count must be positive.")
    destination.parent.mkdir(parents=True, exist_ok=True)


def _validate_contiguous_episodes(
    data: pa.Table,
    episodes: pa.Table,
    total_episodes: int,
) -> None:
    episode_indices = sorted(int(value) for value in episodes["episode_index"].to_pylist())
    if episode_indices != list(range(total_episodes)):
        raise ValueError(f"Episode metadata is not contiguous: {episode_indices}")
    data_indices = sorted(set(int(value) for value in data["episode_index"].to_pylist()))
    if data_indices != episode_indices:
        raise ValueError("Data and episode metadata contain different episode indices.")


def _hardlink_tree(source: Path, destination: Path) -> None:
    for source_path in source.rglob("*"):
        destination_path = destination / source_path.relative_to(source)
        if source_path.is_dir():
            destination_path.mkdir(parents=True, exist_ok=True)
            continue
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source_path, destination_path)
        except OSError:
            shutil.copy2(source_path, destination_path)


def _filter_table(table: pa.Table, retained_count: int) -> pa.Table:
    return table.filter(pc.less(table["episode_index"], pa.scalar(retained_count)))


def _rewrite_filtered_shards(
    source_root: Path,
    destination_root: Path,
    *,
    retained_count: int,
) -> None:
    for source_path in sorted(source_root.glob("**/*.parquet")):
        destination_path = destination_root / source_path.relative_to(source_root)
        filtered = _filter_table(pq.read_table(source_path), retained_count)
        if filtered.num_rows == 0:
            destination_path.unlink(missing_ok=True)
            continue
        _replace_parquet(destination_path, filtered)


def _rewrite_info_and_stats(
    destination: Path,
    *,
    source_info: dict[str, Any],
    retained_data: pa.Table,
    retained_episodes: pa.Table,
) -> None:
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.io_utils import write_stats

    info = dict(source_info)
    info["total_episodes"] = retained_episodes.num_rows
    info["total_frames"] = retained_data.num_rows
    info["splits"] = {"train": f"0:{retained_episodes.num_rows}"}
    _replace_json(destination / "meta/info.json", info)

    episode_stats = []
    for row in retained_episodes.to_pylist():
        stats: dict[str, dict[str, np.ndarray]] = {}
        for key, value in row.items():
            if not key.startswith("stats/") or value is None:
                continue
            _, feature_name, stat_name = key.split("/", 2)
            stats.setdefault(feature_name, {})[stat_name] = np.asarray(value)
        episode_stats.append(stats)
    write_stats(aggregate_stats(episode_stats), destination)


def _trim_video_tails(
    source: Path,
    destination: Path,
    *,
    source_info: dict[str, Any],
    retained_episodes: pa.Table,
    source_episodes: pa.Table,
) -> int:
    fps = float(source_info["fps"])
    video_keys = [
        key for key, feature in source_info["features"].items() if feature.get("dtype") == "video"
    ]
    retained_rows = retained_episodes.to_pylist()
    source_rows = source_episodes.to_pylist()
    trimmed = 0
    for video_key in video_keys:
        prefix = f"videos/{video_key}"
        retained_refs = _video_references(retained_rows, prefix)
        source_refs = _video_references(source_rows, prefix)
        for ref, source_end_s in source_refs.items():
            chunk_index, file_index = ref
            relative_path = (
                Path("videos")
                / video_key
                / f"chunk-{chunk_index:03d}"
                / f"file-{file_index:03d}.mp4"
            )
            destination_path = destination / relative_path
            if ref not in retained_refs:
                destination_path.unlink(missing_ok=True)
                continue
            retained_end_s = retained_refs[ref]
            if math.isclose(retained_end_s, source_end_s, abs_tol=1e-6):
                continue
            retained_frames = int(round(retained_end_s * fps))
            _trim_video_packets(
                source / relative_path,
                destination_path,
                retained_frames=retained_frames,
            )
            trimmed += 1
    return trimmed


def _video_references(rows: list[dict[str, Any]], prefix: str) -> dict[tuple[int, int], float]:
    references: dict[tuple[int, int], float] = {}
    for row in rows:
        ref = (int(row[f"{prefix}/chunk_index"]), int(row[f"{prefix}/file_index"]))
        references[ref] = max(references.get(ref, 0.0), float(row[f"{prefix}/to_timestamp"]))
    return references


def _trim_video_packets(source_path: Path, destination_path: Path, *, retained_frames: int) -> None:
    ffmpeg = shutil.which("ffmpeg") or str(Path(sys.executable).with_name("ffmpeg"))
    if not Path(ffmpeg).is_file():
        raise FileNotFoundError("ffmpeg is required to trim shared video shards.")
    output_path = destination_path.with_suffix(".repairing.mp4")
    result = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source_path),
            "-map",
            "0:v:0",
            "-frames:v",
            str(retained_frames),
            "-c:v",
            "copy",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed for {source_path}: {result.stderr.strip()}")
    os.replace(output_path, destination_path)


def validate_repaired_dataset(
    root: Path,
    *,
    expected_episodes: int,
    expected_indices: list[int],
) -> dict[str, int]:
    info = _read_json(root / "meta/info.json")
    data = _read_parquet_tree(root / "data")
    episodes = _read_parquet_tree(root / "meta/episodes")
    episode_indices = [int(value) for value in episodes["episode_index"].to_pylist()]
    if episode_indices != expected_indices:
        raise ValueError(f"Unexpected retained episode indices: {episode_indices}")
    if info["total_episodes"] != expected_episodes or episodes.num_rows != expected_episodes:
        raise ValueError("Repaired episode totals do not match.")
    if info["total_frames"] != data.num_rows:
        raise ValueError("Repaired frame totals do not match.")
    if sum(int(value) for value in episodes["length"].to_pylist()) != data.num_rows:
        raise ValueError("Episode lengths do not sum to the retained data rows.")
    indexes = np.asarray(data["index"].to_pylist(), dtype=np.int64)
    if not np.array_equal(indexes, np.arange(data.num_rows)):
        raise ValueError("Retained global frame indices are not contiguous.")

    fps = float(info["fps"])
    for video_key, feature in info["features"].items():
        if feature.get("dtype") != "video":
            continue
        prefix = f"videos/{video_key}"
        for ref, end_s in _video_references(episodes.to_pylist(), prefix).items():
            chunk_index, file_index = ref
            video_path = (
                root
                / "videos"
                / video_key
                / f"chunk-{chunk_index:03d}"
                / f"file-{file_index:03d}.mp4"
            )
            frame_count = _probe_video_frame_count(video_path)
            expected_frames = int(round(end_s * fps))
            if frame_count != expected_frames:
                raise ValueError(
                    f"Video {video_path} has {frame_count} frames; expected {expected_frames}."
                )
    return {"episodes": episodes.num_rows, "frames": data.num_rows}


def _probe_video_frame_count(path: Path) -> int:
    ffprobe = shutil.which("ffprobe") or str(Path(sys.executable).with_name("ffprobe"))
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    return int(result.stdout.strip())


def _read_parquet_tree(root: Path) -> pa.Table:
    paths = sorted(root.glob("**/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No Parquet files found under {root}")
    return pa.concat_tables([pq.read_table(path) for path in paths])


def _replace_parquet(path: Path, table: pa.Table) -> None:
    temporary_path = path.with_suffix(".repairing.parquet")
    pq.write_table(table, temporary_path, compression="snappy")
    os.replace(temporary_path, path)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _replace_json(path: Path, value: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(".repairing.json")
    _write_json(temporary_path, value)
    os.replace(temporary_path, path)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=4)
        stream.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
