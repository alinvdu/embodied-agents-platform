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
import copy
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


RIGHT_ARM_NAMES = [
    "right_arm_shoulder_pan.pos",
    "right_arm_shoulder_lift.pos",
    "right_arm_elbow_flex.pos",
    "right_arm_wrist_flex.pos",
    "right_arm_wrist_roll.pos",
    "right_arm_gripper.pos",
]
KEPT_CAMERAS = ("head", "right_wrist")
DEFAULT_TASK = "Grab the Tabasco sauce bottle and put it in the robot basket."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Derive a LeRobot v3 right-arm-only dataset while retaining the head and "
            "right-wrist videos. The source dataset is never modified."
        )
    )
    parser.add_argument("--source", required=True, help="Existing LeRobot v3 dataset root.")
    parser.add_argument("--destination", required=True, help="New dataset root; it must not exist.")
    parser.add_argument("--task", default=DEFAULT_TASK, help="Single task label for every episode.")
    parser.add_argument(
        "--copy-videos",
        action="store_true",
        help="Copy retained videos instead of hard-linking them.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = Path(args.source).expanduser().resolve()
    destination = Path(args.destination).expanduser().resolve()
    if source == destination:
        raise ValueError("Source and destination must be different directories.")
    if not (source / "meta/info.json").is_file():
        raise FileNotFoundError(f"Not a LeRobot v3 dataset root: {source}")
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    build_root = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.building-", dir=destination.parent)
    )
    try:
        source_info = _read_json(source / "meta/info.json")
        action_indices = _feature_indices(source_info, "action", RIGHT_ARM_NAMES)
        state_indices = _feature_indices(source_info, "observation.state", RIGHT_ARM_NAMES)

        _write_json(
            build_root / "meta/info.json",
            _derive_info(source_info),
        )
        _write_json(
            build_root / "meta/stats.json",
            _derive_global_stats(
                _read_json(source / "meta/stats.json"),
                action_indices=action_indices,
                state_indices=state_indices,
            ),
        )
        _rewrite_tasks(source / "meta/tasks.parquet", build_root / "meta/tasks.parquet", args.task)
        _rewrite_data_shards(
            source,
            build_root,
            action_indices=action_indices,
            state_indices=state_indices,
        )
        _rewrite_episode_shards(
            source,
            build_root,
            task=args.task,
            action_indices=action_indices,
            state_indices=state_indices,
        )
        linked_videos, copied_videos = _materialize_retained_videos(
            source,
            build_root,
            copy_videos=args.copy_videos,
        )
        summary = validate_derived_dataset(build_root, expected_task=args.task)
        os.replace(build_root, destination)
    except Exception:
        shutil.rmtree(build_root, ignore_errors=True)
        raise

    print(f"Created right-arm dataset: {destination}")
    print(
        f"episodes={summary['episodes']} frames={summary['frames']} "
        f"state_dim={summary['state_dim']} action_dim={summary['action_dim']} "
        f"cameras={','.join(summary['cameras'])} tasks={summary['tasks']}"
    )
    print(f"videos_hard_linked={linked_videos} videos_copied={copied_videos}")
    return 0


def _feature_indices(info: dict[str, Any], feature_key: str, names: list[str]) -> list[int]:
    source_names = info["features"][feature_key].get("names")
    if not source_names:
        raise ValueError(f"Feature {feature_key!r} does not define named dimensions.")
    missing = [name for name in names if name not in source_names]
    if missing:
        raise ValueError(f"Feature {feature_key!r} is missing dimensions: {missing}")
    return [source_names.index(name) for name in names]


def _derive_info(source_info: dict[str, Any]) -> dict[str, Any]:
    info = copy.deepcopy(source_info)
    info["total_tasks"] = 1
    info["features"] = {
        key: value
        for key, value in info["features"].items()
        if key != "observation.images.left_wrist"
    }
    for feature_key in ("action", "observation.state"):
        info["features"][feature_key]["shape"] = [len(RIGHT_ARM_NAMES)]
        info["features"][feature_key]["names"] = list(RIGHT_ARM_NAMES)
    return info


def _derive_global_stats(
    source_stats: dict[str, Any],
    *,
    action_indices: list[int],
    state_indices: list[int],
) -> dict[str, Any]:
    stats = copy.deepcopy(source_stats)
    stats.pop("observation.images.left_wrist", None)
    stats["action"] = _slice_stats(stats["action"], action_indices)
    stats["observation.state"] = _slice_stats(stats["observation.state"], state_indices)
    stats["task_index"] = _zero_stats(stats["task_index"])
    return stats


def _slice_stats(stats: dict[str, Any], indices: list[int]) -> dict[str, Any]:
    return {
        name: copy.deepcopy(values)
        if name == "count"
        else [copy.deepcopy(values[index]) for index in indices]
        for name, values in stats.items()
    }


def _zero_stats(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        name: copy.deepcopy(values) if name == "count" else [0]
        for name, values in stats.items()
    }


def _rewrite_tasks(source_path: Path, destination_path: Path, task: str) -> None:
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing source task table: {source_path}")
    tasks = pd.DataFrame(
        {"task_index": [0]},
        index=pd.Index([task], name="task"),
    )
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    tasks.to_parquet(destination_path, compression="snappy")


def _rewrite_data_shards(
    source: Path,
    destination: Path,
    *,
    action_indices: list[int],
    state_indices: list[int],
) -> None:
    source_paths = sorted((source / "data").glob("**/*.parquet"))
    if not source_paths:
        raise FileNotFoundError(f"No data Parquet shards found under {source / 'data'}")
    for source_path in source_paths:
        table = pq.read_table(source_path)
        table = _replace_fixed_vector(table, "action", action_indices)
        table = _replace_fixed_vector(table, "observation.state", state_indices)
        task_index = table.schema.get_field_index("task_index")
        table = table.set_column(
            task_index,
            "task_index",
            pa.array(np.zeros(table.num_rows, dtype=np.int64), type=pa.int64()),
        )
        table = _rewrite_huggingface_metadata(table)
        destination_path = destination / source_path.relative_to(source)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, destination_path, compression="snappy")


def _replace_fixed_vector(
    table: pa.Table,
    column_name: str,
    indices: list[int],
) -> pa.Table:
    values = np.asarray(table[column_name].to_pylist(), dtype=np.float32)
    selected = np.ascontiguousarray(values[:, indices], dtype=np.float32)
    array = pa.FixedSizeListArray.from_arrays(
        pa.array(selected.reshape(-1), type=pa.float32()),
        list_size=len(indices),
    )
    column_index = table.schema.get_field_index(column_name)
    return table.set_column(column_index, column_name, array)


def _rewrite_huggingface_metadata(table: pa.Table) -> pa.Table:
    metadata = dict(table.schema.metadata or {})
    encoded = metadata.get(b"huggingface")
    if encoded is None:
        return table
    huggingface = json.loads(encoded.decode("utf-8"))
    features = huggingface["info"]["features"]
    for feature_key in ("action", "observation.state"):
        features[feature_key]["length"] = len(RIGHT_ARM_NAMES)
    fingerprint_source = json.dumps(features, sort_keys=True).encode("utf-8")
    huggingface["fingerprint"] = hashlib.sha256(fingerprint_source).hexdigest()[:16]
    metadata[b"huggingface"] = json.dumps(huggingface, separators=(",", ":")).encode("utf-8")
    return table.replace_schema_metadata(metadata)


def _rewrite_episode_shards(
    source: Path,
    destination: Path,
    *,
    task: str,
    action_indices: list[int],
    state_indices: list[int],
) -> None:
    source_paths = sorted((source / "meta/episodes").glob("**/*.parquet"))
    if not source_paths:
        raise FileNotFoundError(f"No episode Parquet shards found under {source / 'meta/episodes'}")
    dropped_prefixes = (
        "videos/observation.images.left_wrist/",
        "stats/observation.images.left_wrist/",
    )
    for source_path in source_paths:
        source_table = pq.read_table(source_path)
        arrays: list[pa.Array] = []
        fields: list[pa.Field] = []
        for field in source_table.schema:
            name = field.name
            if name.startswith(dropped_prefixes):
                continue
            values = source_table[name].to_pylist()
            if name == "tasks":
                values = [[task] for _ in values]
            elif name.startswith("stats/action/") and not name.endswith("/count"):
                values = _slice_episode_stat_values(values, action_indices)
            elif name.startswith("stats/observation.state/") and not name.endswith("/count"):
                values = _slice_episode_stat_values(values, state_indices)
            elif name.startswith("stats/task_index/") and not name.endswith("/count"):
                values = [[0] for _ in values]
            arrays.append(pa.array(values, type=field.type))
            fields.append(field)
        table = pa.Table.from_arrays(arrays, schema=pa.schema(fields))
        destination_path = destination / source_path.relative_to(source)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, destination_path, compression="snappy")


def _slice_episode_stat_values(values: list[Any], indices: list[int]) -> list[Any]:
    return [
        None if value is None else [copy.deepcopy(value[index]) for index in indices]
        for value in values
    ]


def _materialize_retained_videos(
    source: Path,
    destination: Path,
    *,
    copy_videos: bool,
) -> tuple[int, int]:
    linked = 0
    copied = 0
    for camera in KEPT_CAMERAS:
        source_dir = source / "videos" / f"observation.images.{camera}"
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Missing retained camera directory: {source_dir}")
        for source_path in sorted(source_dir.glob("**/*.mp4")):
            destination_path = destination / source_path.relative_to(source)
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            if copy_videos:
                shutil.copy2(source_path, destination_path)
                copied += 1
            else:
                os.link(source_path, destination_path)
                linked += 1
    return linked, copied


def validate_derived_dataset(root: Path, *, expected_task: str) -> dict[str, Any]:
    info = _read_json(root / "meta/info.json")
    expected_camera_keys = {f"observation.images.{camera}" for camera in KEPT_CAMERAS}
    actual_camera_keys = {
        key for key, feature in info["features"].items() if feature.get("dtype") == "video"
    }
    if actual_camera_keys != expected_camera_keys:
        raise ValueError(f"Unexpected camera features: {sorted(actual_camera_keys)}")
    for feature_key in ("action", "observation.state"):
        feature = info["features"][feature_key]
        if feature["shape"] != [len(RIGHT_ARM_NAMES)] or feature["names"] != RIGHT_ARM_NAMES:
            raise ValueError(f"Invalid {feature_key} feature metadata: {feature}")

    data_paths = sorted((root / "data").glob("**/*.parquet"))
    data = pq.read_table(data_paths, columns=["action", "observation.state", "task_index"])
    if data.schema.field("action").type.list_size != len(RIGHT_ARM_NAMES):
        raise ValueError("Action Parquet dimension is not six.")
    if data.schema.field("observation.state").type.list_size != len(RIGHT_ARM_NAMES):
        raise ValueError("State Parquet dimension is not six.")
    if any(value != 0 for value in data["task_index"].to_pylist()):
        raise ValueError("Data contains a task_index other than zero.")

    episode_paths = sorted((root / "meta/episodes").glob("**/*.parquet"))
    episodes = pq.read_table(episode_paths)
    if any(tasks != [expected_task] for tasks in episodes["tasks"].to_pylist()):
        raise ValueError("Episode task labels were not consolidated.")
    if any("left_wrist" in name for name in episodes.column_names):
        raise ValueError("Left-wrist metadata remains in episode shards.")
    if sum(episodes["length"].to_pylist()) != data.num_rows:
        raise ValueError("Episode lengths do not sum to the data frame count.")
    _validate_video_references(root, episodes)

    tasks = pq.read_table(root / "meta/tasks.parquet")
    if tasks.to_pylist() != [{"task_index": 0, "task": expected_task}]:
        raise ValueError(f"Unexpected task table: {tasks.to_pylist()}")
    if info["total_frames"] != data.num_rows or info["total_episodes"] != episodes.num_rows:
        raise ValueError("info.json totals do not match the converted Parquet tables.")
    if info["total_tasks"] != 1:
        raise ValueError("info.json total_tasks is not one.")

    return {
        "episodes": episodes.num_rows,
        "frames": data.num_rows,
        "state_dim": info["features"]["observation.state"]["shape"][0],
        "action_dim": info["features"]["action"]["shape"][0],
        "cameras": list(KEPT_CAMERAS),
        "tasks": info["total_tasks"],
    }


def _validate_video_references(root: Path, episodes: pa.Table) -> None:
    for camera in KEPT_CAMERAS:
        prefix = f"videos/observation.images.{camera}"
        chunks = episodes[f"{prefix}/chunk_index"].to_pylist()
        files = episodes[f"{prefix}/file_index"].to_pylist()
        for chunk_index, file_index in set(zip(chunks, files, strict=True)):
            path = (
                root
                / "videos"
                / f"observation.images.{camera}"
                / f"chunk-{chunk_index:03d}"
                / f"file-{file_index:03d}.mp4"
            )
            if not path.is_file():
                raise FileNotFoundError(f"Missing referenced video: {path}")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=4)
        stream.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
