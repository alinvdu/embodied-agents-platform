#!/usr/bin/env python3
"""Create a right-arm-only copy of a LeRobot v3 dataset.

The source dataset is left untouched. Video files are hard-linked when possible
because the camera streams do not change during this conversion.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import datasets
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.datasets.utils import get_hf_features_from_features


VECTOR_KEYS = ("action", "observation.state")
RIGHT_ARM_NAMES = [
    "right_arm_shoulder_pan.pos",
    "right_arm_shoulder_lift.pos",
    "right_arm_elbow_flex.pos",
    "right_arm_wrist_flex.pos",
    "right_arm_wrist_roll.pos",
    "right_arm_gripper.pos",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open() as file:
        return json.load(file)


def write_json(path: Path, value: dict) -> None:
    with path.open("w") as file:
        json.dump(value, file, indent=2)
        file.write("\n")


def right_arm_indices(info: dict) -> list[int]:
    action_names = info["features"]["action"]["names"]
    state_names = info["features"]["observation.state"]["names"]
    if action_names != state_names:
        raise ValueError("action and observation.state names differ")

    missing = [name for name in RIGHT_ARM_NAMES if name not in action_names]
    if missing:
        raise ValueError(f"Missing right-arm dimensions: {missing}")

    indices = [action_names.index(name) for name in RIGHT_ARM_NAMES]
    if len(set(indices)) != len(RIGHT_ARM_NAMES):
        raise ValueError("Right-arm dimension names are not unique")
    return indices


def link_video_or_copy(source: str, destination: str) -> str:
    source_path = Path(source)
    if source_path.suffix == ".mp4":
        try:
            os.link(source, destination)
            return destination
        except OSError:
            pass
    return shutil.copy2(source, destination)


def copy_dataset(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")

    shutil.copytree(
        source,
        destination,
        copy_function=link_video_or_copy,
        ignore=shutil.ignore_patterns(".cache"),
    )


def update_info(destination: Path, indices: list[int]) -> dict:
    path = destination / "meta/info.json"
    info = load_json(path)
    for key in VECTOR_KEYS:
        feature = info["features"][key]
        feature["shape"] = [len(indices)]
        feature["names"] = [feature["names"][index] for index in indices]
    write_json(path, info)
    return info


def update_global_stats(destination: Path, indices: list[int]) -> None:
    path = destination / "meta/stats.json"
    stats = load_json(path)
    for key in VECTOR_KEYS:
        for stat_name, values in stats[key].items():
            if stat_name == "count":
                continue
            if len(values) <= max(indices):
                raise ValueError(f"Unexpected {key}/{stat_name} statistic length: {len(values)}")
            stats[key][stat_name] = [values[index] for index in indices]
    write_json(path, stats)


def parquet_temp_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def rewrite_data_files(destination: Path, info: dict, indices: list[int]) -> None:
    hf_features = get_hf_features_from_features(info["features"])
    for key, feature in info["features"].items():
        if key in hf_features and feature["shape"] == [1]:
            hf_features[key] = datasets.Value(feature["dtype"])
    target_schema = hf_features.arrow_schema

    for path in sorted((destination / "data").rglob("*.parquet")):
        source_table = pq.read_table(path)
        arrays = []
        for field in target_schema:
            if field.name in VECTOR_KEYS:
                source_values = np.asarray(source_table[field.name].to_pylist(), dtype=np.float32)
                values = source_values[:, indices]
                array = pa.FixedSizeListArray.from_arrays(
                    pa.array(values.reshape(-1), type=pa.float32()), len(indices)
                )
            else:
                array = source_table[field.name].combine_chunks()
            arrays.append(array)
        table = pa.Table.from_arrays(arrays, schema=target_schema)

        temp_path = parquet_temp_path(path)
        pq.write_table(table, temp_path, compression="snappy", use_dictionary=True)
        temp_path.replace(path)


def rewrite_episode_metadata(destination: Path, indices: list[int]) -> None:
    prefixes = tuple(f"stats/{key}/" for key in VECTOR_KEYS)
    for path in sorted((destination / "meta/episodes").rglob("*.parquet")):
        table = pq.read_table(path)
        for column_index, field in enumerate(table.schema):
            if not field.name.startswith(prefixes) or field.name.endswith("/count"):
                continue

            values = table.column(column_index).to_pylist()
            sliced_values = [[value[index] for index in indices] for value in values]
            array = pa.array(sliced_values, type=field.type)
            table = table.set_column(column_index, field, array)

        temp_path = parquet_temp_path(path)
        pq.write_table(table, temp_path, compression="snappy", use_dictionary=True)
        temp_path.replace(path)


def validate(source: Path, destination: Path, indices: list[int]) -> None:
    info = load_json(destination / "meta/info.json")
    stats = load_json(destination / "meta/stats.json")

    for key in VECTOR_KEYS:
        feature = info["features"][key]
        if feature["shape"] != [6] or feature["names"] != RIGHT_ARM_NAMES:
            raise ValueError(f"Incorrect converted feature metadata for {key}: {feature}")
        for stat_name, values in stats[key].items():
            expected_length = 1 if stat_name == "count" else 6
            if len(values) != expected_length:
                raise ValueError(f"Incorrect {key}/{stat_name} statistic length")

    source_files = sorted((source / "data").rglob("*.parquet"))
    destination_files = sorted((destination / "data").rglob("*.parquet"))
    if len(source_files) != len(destination_files):
        raise ValueError("Source and destination data file counts differ")

    total_rows = 0
    for source_path, destination_path in zip(source_files, destination_files, strict=True):
        source_table = pq.read_table(source_path, columns=list(VECTOR_KEYS))
        destination_table = pq.read_table(destination_path, columns=list(VECTOR_KEYS))
        if source_table.num_rows != destination_table.num_rows:
            raise ValueError(f"Row count changed in {destination_path}")
        total_rows += destination_table.num_rows

        for key in VECTOR_KEYS:
            source_values = np.asarray(source_table[key].to_pylist(), dtype=np.float32)
            destination_values = np.asarray(destination_table[key].to_pylist(), dtype=np.float32)
            np.testing.assert_array_equal(destination_values, source_values[:, indices])

    if total_rows != info["total_frames"]:
        raise ValueError(f"Expected {info['total_frames']} frames, found {total_rows}")

    for path in sorted((destination / "meta/episodes").rglob("*.parquet")):
        table = pq.read_table(path)
        for field in table.schema:
            if field.name.startswith(tuple(f"stats/{key}/" for key in VECTOR_KEYS)):
                expected_length = 1 if field.name.endswith("/count") else 6
                lengths = {len(value) for value in table[field.name].to_pylist()}
                if lengths != {expected_length}:
                    raise ValueError(f"Incorrect lengths for {field.name}: {lengths}")


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()
    info = load_json(source / "meta/info.json")
    indices = right_arm_indices(info)

    copy_dataset(source, destination)
    try:
        converted_info = update_info(destination, indices)
        update_global_stats(destination, indices)
        rewrite_data_files(destination, converted_info, indices)
        rewrite_episode_metadata(destination, indices)
        validate(source, destination, indices)
    except Exception:
        shutil.rmtree(destination)
        raise

    print(f"Created right-arm-only dataset: {destination}")
    print(f"Episodes: {converted_info['total_episodes']}")
    print(f"Frames: {converted_info['total_frames']}")
    print(f"Right-arm dimensions: {RIGHT_ARM_NAMES}")


if __name__ == "__main__":
    main()
