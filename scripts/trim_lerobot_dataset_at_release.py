#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


DEFAULT_GRIPPER_NAME = "right_arm_gripper.pos"


@dataclass(frozen=True)
class ReleaseCut:
    episode_index: int
    original_frames: int
    first_open_frame: int
    grasp_close_frame: int
    release_start_frame: int
    fully_open_frame: int
    reclose_start_frame: int | None
    cut_frame_exclusive: int
    retained_frames_after_fully_open: int
    removed_frames: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a new LeRobot v3 dataset whose episodes end shortly after the "
            "basket-release gripper transition. Source data and videos are never modified."
        )
    )
    parser.add_argument("--source", required=True, help="Existing LeRobot v3 dataset root.")
    parser.add_argument("--destination", required=True, help="New dataset root; it must not exist.")
    parser.add_argument("--gripper-name", default=DEFAULT_GRIPPER_NAME)
    parser.add_argument("--open-threshold", type=float, default=30.0)
    parser.add_argument("--closed-threshold", type=float, default=10.0)
    parser.add_argument("--fully-open-threshold", type=float, default=40.0)
    parser.add_argument("--hold-frames", type=int, default=5)
    parser.add_argument(
        "--post-release-s",
        type=float,
        default=1.0,
        help="Seconds retained after the gripper first becomes fully open.",
    )
    parser.add_argument(
        "--copy-videos",
        action="store_true",
        help="Copy video shards instead of hard-linking the unchanged source files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = Path(args.source).expanduser().resolve()
    destination = Path(args.destination).expanduser().resolve()
    _validate_paths(source, destination)

    source_info = _read_json(source / "meta/info.json")
    fps = float(source_info["fps"])
    action_names = source_info["features"]["action"].get("names") or []
    if args.gripper_name not in action_names:
        raise ValueError(f"Action feature does not contain gripper {args.gripper_name!r}.")
    gripper_index = action_names.index(args.gripper_name)

    source_data = _read_parquet_tree(source / "data")
    source_episodes = _read_parquet_tree(source / "meta/episodes")
    cuts = detect_release_cuts(
        source_data,
        episode_indices=source_episodes["episode_index"].to_pylist(),
        gripper_index=gripper_index,
        fps=fps,
        open_threshold=args.open_threshold,
        closed_threshold=args.closed_threshold,
        fully_open_threshold=args.fully_open_threshold,
        hold_frames=args.hold_frames,
        post_release_s=args.post_release_s,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    build_root = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.building-", dir=destination.parent)
    )
    try:
        trimmed_data = _trim_data_table(source_data, cuts)
        episode_table, numeric_episode_stats = _rewrite_episode_table(
            source_episodes,
            trimmed_data,
            cuts=cuts,
            features=source_info["features"],
            fps=fps,
        )
        _write_dataset_metadata(
            source=source,
            destination=build_root,
            source_info=source_info,
            trimmed_data=trimmed_data,
            episode_table=episode_table,
            numeric_episode_stats=numeric_episode_stats,
            cuts=cuts,
            args=args,
        )
        linked_videos, copied_videos = _materialize_videos(
            source,
            build_root,
            copy_videos=args.copy_videos,
        )
        summary = validate_trimmed_dataset(
            build_root,
            fully_open_threshold=args.fully_open_threshold,
            gripper_name=args.gripper_name,
        )
        os.replace(build_root, destination)
    except Exception:
        shutil.rmtree(build_root, ignore_errors=True)
        raise

    print(f"Created release-trimmed dataset: {destination}")
    print(
        f"episodes={summary['episodes']} frames={summary['frames']} "
        f"removed_frames={sum(cut.removed_frames for cut in cuts.values())} "
        f"median_cut_s={summary['median_cut_s']:.2f}"
    )
    print(f"videos_hard_linked={linked_videos} videos_copied={copied_videos}")
    print(f"cut_report={destination / 'meta/release_trim_report.json'}")
    return 0


def detect_release_cuts(
    data: pa.Table,
    *,
    episode_indices: list[int],
    gripper_index: int,
    fps: float,
    open_threshold: float,
    closed_threshold: float,
    fully_open_threshold: float,
    hold_frames: int,
    post_release_s: float,
) -> dict[int, ReleaseCut]:
    if hold_frames <= 0:
        raise ValueError("hold_frames must be positive.")
    if fps <= 0:
        raise ValueError("fps must be positive.")
    if post_release_s < 0:
        raise ValueError("post_release_s must not be negative.")
    if not closed_threshold < open_threshold <= fully_open_threshold:
        raise ValueError("Expected closed_threshold < open_threshold <= fully_open_threshold.")

    data = _sort_data_table(data)
    data_episode_indices = np.asarray(data["episode_index"].to_pylist(), dtype=np.int64)
    frame_indices = np.asarray(data["frame_index"].to_pylist(), dtype=np.int64)
    actions = np.asarray(data["action"].to_pylist(), dtype=np.float32)
    if not 0 <= gripper_index < actions.shape[1]:
        raise ValueError(f"Invalid gripper index {gripper_index} for action dimension {actions.shape[1]}.")

    post_release_frames = int(math.ceil(post_release_s * fps))
    cuts: dict[int, ReleaseCut] = {}
    for episode_index in episode_indices:
        episode_mask = data_episode_indices == episode_index
        episode_frames = frame_indices[episode_mask]
        if not np.array_equal(episode_frames, np.arange(len(episode_frames))):
            raise ValueError(f"Episode {episode_index} frame indexes are not contiguous from zero.")
        gripper = actions[episode_mask, gripper_index]
        first_open = _find_sustained_run_start(
            gripper >= open_threshold,
            start=0,
            hold_frames=hold_frames,
        )
        grasp_close = _find_sustained_run_start(
            gripper <= closed_threshold,
            start=first_open + hold_frames,
            hold_frames=hold_frames,
        )
        release_runs = _sustained_run_starts(
            gripper >= open_threshold,
            start=grasp_close + hold_frames,
            hold_frames=hold_frames,
        )
        if len(release_runs) != 1:
            raise ValueError(
                f"Episode {episode_index} has {len(release_runs)} sustained post-grasp openings; "
                "the release point is ambiguous."
            )
        release_start = release_runs[0]
        fully_open = _find_sustained_run_start(
            gripper >= fully_open_threshold,
            start=release_start,
            hold_frames=hold_frames,
        )
        reclose_runs = _sustained_run_starts(
            gripper < fully_open_threshold,
            start=fully_open + hold_frames,
            hold_frames=hold_frames,
        )
        reclose_start = reclose_runs[0] if reclose_runs else None
        desired_cut = min(len(gripper), fully_open + 1 + post_release_frames)
        cut_frame_exclusive = (
            desired_cut if reclose_start is None else min(desired_cut, reclose_start)
        )
        cuts[int(episode_index)] = ReleaseCut(
            episode_index=int(episode_index),
            original_frames=len(gripper),
            first_open_frame=first_open,
            grasp_close_frame=grasp_close,
            release_start_frame=release_start,
            fully_open_frame=fully_open,
            reclose_start_frame=reclose_start,
            cut_frame_exclusive=cut_frame_exclusive,
            retained_frames_after_fully_open=cut_frame_exclusive - fully_open - 1,
            removed_frames=len(gripper) - cut_frame_exclusive,
        )
    return cuts


def _find_sustained_run_start(mask: np.ndarray, *, start: int, hold_frames: int) -> int:
    starts = _sustained_run_starts(mask, start=start, hold_frames=hold_frames)
    if not starts:
        raise ValueError(
            f"No sustained transition found at or after frame {start} for {hold_frames} frames."
        )
    return starts[0]


def _sustained_run_starts(mask: np.ndarray, *, start: int, hold_frames: int) -> list[int]:
    mask = np.asarray(mask, dtype=bool)
    run_starts = np.flatnonzero(mask & np.r_[True, ~mask[:-1]])
    return [
        int(index)
        for index in run_starts
        if index >= start
        and index + hold_frames <= len(mask)
        and bool(mask[index : index + hold_frames].all())
    ]


def _trim_data_table(data: pa.Table, cuts: dict[int, ReleaseCut]) -> pa.Table:
    data = _sort_data_table(data)
    episode_indices = np.asarray(data["episode_index"].to_pylist(), dtype=np.int64)
    frame_indices = np.asarray(data["frame_index"].to_pylist(), dtype=np.int64)
    keep = np.fromiter(
        (
            frame_index < cuts[int(episode_index)].cut_frame_exclusive
            for episode_index, frame_index in zip(episode_indices, frame_indices, strict=True)
        ),
        dtype=bool,
        count=data.num_rows,
    )
    trimmed = data.filter(pa.array(keep))
    index_position = trimmed.schema.get_field_index("index")
    return trimmed.set_column(
        index_position,
        "index",
        pa.array(np.arange(trimmed.num_rows, dtype=np.int64), type=pa.int64()),
    )


def _rewrite_episode_table(
    source_episodes: pa.Table,
    trimmed_data: pa.Table,
    *,
    cuts: dict[int, ReleaseCut],
    features: dict[str, dict],
    fps: float,
) -> tuple[pa.Table, list[dict[str, dict]]]:
    source_rows = {
        int(row["episode_index"]): row for row in source_episodes.to_pylist()
    }
    data_episode_indices = np.asarray(trimmed_data["episode_index"].to_pylist(), dtype=np.int64)
    records: list[dict[str, Any]] = []
    numeric_episode_stats: list[dict[str, dict]] = []
    global_from_index = 0

    for episode_index in sorted(cuts):
        cut = cuts[episode_index]
        row = copy.deepcopy(source_rows[episode_index])
        episode_table = trimmed_data.filter(pa.array(data_episode_indices == episode_index))
        episode_stats = _compute_numeric_episode_stats(episode_table, features)
        numeric_episode_stats.append(episode_stats)

        row["length"] = cut.cut_frame_exclusive
        row["data/chunk_index"] = 0
        row["data/file_index"] = 0
        row["dataset_from_index"] = global_from_index
        row["dataset_to_index"] = global_from_index + cut.cut_frame_exclusive
        row["meta/episodes/chunk_index"] = 0
        row["meta/episodes/file_index"] = 0
        global_from_index += cut.cut_frame_exclusive

        for key in features:
            if features[key]["dtype"] != "video":
                continue
            from_key = f"videos/{key}/from_timestamp"
            to_key = f"videos/{key}/to_timestamp"
            row[to_key] = float(row[from_key]) + cut.cut_frame_exclusive / fps

        for feature_name, stats in episode_stats.items():
            for stat_name, value in stats.items():
                row[f"stats/{feature_name}/{stat_name}"] = np.asarray(value).tolist()
        records.append(row)

    return pa.Table.from_pylist(records, schema=source_episodes.schema), numeric_episode_stats


def _compute_numeric_episode_stats(
    episode_table: pa.Table,
    features: dict[str, dict],
) -> dict[str, dict]:
    from lerobot.datasets.compute_stats import compute_episode_stats

    episode_data: dict[str, np.ndarray] = {}
    numeric_features: dict[str, dict] = {}
    for feature_name, feature in features.items():
        if feature_name not in episode_table.column_names:
            continue
        if feature["dtype"] in {"image", "video", "string"}:
            continue
        column = episode_table[feature_name]
        if pa.types.is_fixed_size_list(column.type) or pa.types.is_list(column.type):
            values = np.asarray(column.to_pylist())
        else:
            values = np.asarray(column.to_pylist())
        episode_data[feature_name] = values
        numeric_features[feature_name] = feature
    return compute_episode_stats(episode_data, numeric_features)


def _write_dataset_metadata(
    *,
    source: Path,
    destination: Path,
    source_info: dict[str, Any],
    trimmed_data: pa.Table,
    episode_table: pa.Table,
    numeric_episode_stats: list[dict[str, dict]],
    cuts: dict[int, ReleaseCut],
    args: argparse.Namespace,
) -> None:
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.io_utils import load_stats, write_stats

    data_path = destination / "data/chunk-000/file-000.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(trimmed_data, data_path, compression="snappy")

    episodes_path = destination / "meta/episodes/chunk-000/file-000.parquet"
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(episode_table, episodes_path, compression="snappy")

    info = copy.deepcopy(source_info)
    info["total_frames"] = trimmed_data.num_rows
    info["total_episodes"] = episode_table.num_rows
    info["splits"] = {"train": f"0:{episode_table.num_rows}"}
    _write_json(destination / "meta/info.json", info)
    shutil.copy2(source / "meta/tasks.parquet", destination / "meta/tasks.parquet")

    source_stats = load_stats(source) or {}
    stats = {
        key: value
        for key, value in source_stats.items()
        if source_info["features"].get(key, {}).get("dtype") in {"image", "video"}
    }
    stats.update(aggregate_stats(numeric_episode_stats))
    write_stats(stats, destination)

    report = {
        "source": str(source),
        "episodes": episode_table.num_rows,
        "source_frames": sum(cut.original_frames for cut in cuts.values()),
        "retained_frames": trimmed_data.num_rows,
        "removed_frames": sum(cut.removed_frames for cut in cuts.values()),
        "fps": source_info["fps"],
        "gripper_name": args.gripper_name,
        "open_threshold": args.open_threshold,
        "closed_threshold": args.closed_threshold,
        "fully_open_threshold": args.fully_open_threshold,
        "hold_frames": args.hold_frames,
        "post_release_s": args.post_release_s,
        "episode_cuts": [asdict(cuts[index]) for index in sorted(cuts)],
    }
    _write_json(destination / "meta/release_trim_report.json", report)


def _materialize_videos(
    source: Path,
    destination: Path,
    *,
    copy_videos: bool,
) -> tuple[int, int]:
    linked = 0
    copied = 0
    video_paths = sorted((source / "videos").glob("**/*.mp4"))
    if not video_paths:
        raise FileNotFoundError(f"No video shards found under {source / 'videos'}")
    for source_path in video_paths:
        destination_path = destination / source_path.relative_to(source)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if copy_videos:
            shutil.copy2(source_path, destination_path)
            copied += 1
        else:
            os.link(source_path, destination_path)
            linked += 1
    return linked, copied


def validate_trimmed_dataset(
    root: Path,
    *,
    fully_open_threshold: float,
    gripper_name: str = DEFAULT_GRIPPER_NAME,
) -> dict[str, Any]:
    info = _read_json(root / "meta/info.json")
    data = _sort_data_table(_read_parquet_tree(root / "data"))
    episodes = _read_parquet_tree(root / "meta/episodes")
    lengths = episodes["length"].to_pylist()
    if sum(lengths) != data.num_rows:
        raise ValueError("Episode lengths do not sum to the trimmed data frame count.")
    if info["total_frames"] != data.num_rows or info["total_episodes"] != episodes.num_rows:
        raise ValueError("info.json totals do not match trimmed Parquet data.")
    indexes = np.asarray(data["index"].to_pylist(), dtype=np.int64)
    if not np.array_equal(indexes, np.arange(data.num_rows)):
        raise ValueError("Trimmed global frame indexes are not contiguous.")

    gripper_names = info["features"]["action"].get("names") or []
    gripper_index = gripper_names.index(gripper_name)
    data_episode_indices = np.asarray(data["episode_index"].to_pylist(), dtype=np.int64)
    actions = np.asarray(data["action"].to_pylist(), dtype=np.float32)
    for row in episodes.to_pylist():
        episode_index = int(row["episode_index"])
        episode_actions = actions[data_episode_indices == episode_index]
        if episode_actions[-1, gripper_index] < fully_open_threshold:
            raise ValueError(f"Episode {episode_index} does not end with a fully open gripper.")
        for feature_name, feature in info["features"].items():
            if feature.get("dtype") != "video":
                continue
            prefix = f"videos/{feature_name}"
            video_path = (
                root
                / "videos"
                / feature_name
                / f"chunk-{row[f'{prefix}/chunk_index']:03d}"
                / f"file-{row[f'{prefix}/file_index']:03d}.mp4"
            )
            if not video_path.is_file():
                raise FileNotFoundError(f"Missing referenced video: {video_path}")
            duration = row[f"{prefix}/to_timestamp"] - row[f"{prefix}/from_timestamp"]
            expected_duration = row["length"] / info["fps"]
            if not math.isclose(duration, expected_duration, abs_tol=1e-6):
                raise ValueError(f"Episode {episode_index} video duration does not match its length.")
    return {
        "episodes": episodes.num_rows,
        "frames": data.num_rows,
        "median_cut_s": float(np.median(np.asarray(lengths) / info["fps"])),
    }


def _sort_data_table(data: pa.Table) -> pa.Table:
    indices = pc.sort_indices(data, sort_keys=[("episode_index", "ascending"), ("frame_index", "ascending")])
    return pc.take(data, indices)


def _read_parquet_tree(root: Path) -> pa.Table:
    paths = sorted(root.glob("**/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No Parquet files found under {root}")
    return pa.concat_tables([pq.read_table(path) for path in paths])


def _validate_paths(source: Path, destination: Path) -> None:
    if source == destination:
        raise ValueError("Source and destination must be different directories.")
    if not (source / "meta/info.json").is_file():
        raise FileNotFoundError(f"Not a LeRobot v3 dataset root: {source}")
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")


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
