#!/usr/bin/env python
from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

from lerobot.datasets.video_utils import decode_video_frames


DEFAULT_CAMERAS = ("observation.images.head", "observation.images.right_wrist")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build browser-viewable review clips around every release cutoff. "
            "Green frames are retained by the dataset; red frames are excluded."
        )
    )
    parser.add_argument("--dataset", required=True, help="Release-trimmed LeRobot dataset root.")
    parser.add_argument("--output", required=True, help="Directory for clips and index.html.")
    parser.add_argument(
        "--source",
        help="Original dataset root. Defaults to the source recorded in release_trim_report.json.",
    )
    parser.add_argument("--included-before-s", type=float, default=3.0)
    parser.add_argument("--excluded-after-s", type=float, default=1.5)
    parser.add_argument("--review-fps", type=int, default=15)
    parser.add_argument("--camera-width", type=int, default=480)
    parser.add_argument(
        "--episodes",
        help="Optional comma-separated episode indexes and ranges, for example 0,4,10-15.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = Path(args.dataset).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    report_path = dataset / "meta/release_trim_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Missing release trim report: {report_path}")
    report = _read_json(report_path)
    source = Path(args.source or report["source"]).expanduser().resolve()
    if not (source / "meta/info.json").is_file():
        raise FileNotFoundError(f"Original dataset is unavailable: {source}")
    if args.included_before_s <= 0 or args.excluded_after_s <= 0:
        raise ValueError("Review windows must be positive.")
    if args.review_fps <= 0 or args.camera_width <= 0:
        raise ValueError("Review FPS and camera width must be positive.")

    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output}")
        shutil.rmtree(output)
    clips_dir = output / "clips"
    clips_dir.mkdir(parents=True)

    info = _read_json(source / "meta/info.json")
    source_fps = float(info["fps"])
    source_episodes = {
        int(row["episode_index"]): row
        for row in _read_parquet_tree(source / "meta/episodes").to_pylist()
    }
    cuts = {int(cut["episode_index"]): cut for cut in report["episode_cuts"]}
    selected = _parse_episode_selection(args.episodes, set(cuts))

    reviews: list[dict] = []
    for position, episode_index in enumerate(selected, start=1):
        cut = cuts[episode_index]
        episode = source_episodes[episode_index]
        start_frame = max(
            0,
            int(cut["cut_frame_exclusive"] - round(args.included_before_s * source_fps)),
        )
        end_frame = min(
            int(cut["original_frames"]),
            int(cut["cut_frame_exclusive"] + round(args.excluded_after_s * source_fps)),
        )
        clip_path = clips_dir / f"episode_{episode_index:03d}.mp4"
        _write_review_clip(
            source=source,
            episode=episode,
            camera_keys=DEFAULT_CAMERAS,
            start_frame=start_frame,
            end_frame=end_frame,
            cut_frame=int(cut["cut_frame_exclusive"]),
            source_fps=source_fps,
            review_fps=args.review_fps,
            camera_width=args.camera_width,
            output_path=clip_path,
        )
        reviews.append(
            {
                "episode_index": episode_index,
                "clip": clip_path.relative_to(output).as_posix(),
                "cut_frame": int(cut["cut_frame_exclusive"]),
                "original_frames": int(cut["original_frames"]),
                "fully_open_frame": int(cut["fully_open_frame"]),
                "removed_frames": int(cut["removed_frames"]),
                "boundary_s": (int(cut["cut_frame_exclusive"]) - start_frame) / source_fps,
                "clip_s": (end_frame - start_frame) / source_fps,
            }
        )
        print(f"[{position:03d}/{len(selected):03d}] episode {episode_index:03d}")

    _write_index(output / "index.html", reviews, source_fps=source_fps)
    print(f"Review page: {output / 'index.html'}")
    return 0


def _write_review_clip(
    *,
    source: Path,
    episode: dict,
    camera_keys: tuple[str, ...],
    start_frame: int,
    end_frame: int,
    cut_frame: int,
    source_fps: float,
    review_fps: int,
    camera_width: int,
    output_path: Path,
) -> None:
    sample_frames = np.arange(
        start_frame,
        end_frame,
        source_fps / review_fps,
        dtype=np.float64,
    )
    sample_frames = np.minimum(np.rint(sample_frames).astype(np.int64), end_frame - 1)
    sample_frames = np.unique(sample_frames)
    decoded_cameras: list[np.ndarray] = []
    for camera_key in camera_keys:
        prefix = f"videos/{camera_key}"
        video_path = (
            source
            / "videos"
            / camera_key
            / f"chunk-{episode[f'{prefix}/chunk_index']:03d}"
            / f"file-{episode[f'{prefix}/file_index']:03d}.mp4"
        )
        from_timestamp = float(episode[f"{prefix}/from_timestamp"])
        timestamps = (from_timestamp + sample_frames / source_fps).tolist()
        frames = decode_video_frames(
            video_path,
            timestamps,
            tolerance_s=0.75 / source_fps,
            backend="pyav",
        )
        decoded_cameras.append(
            (frames.permute(0, 2, 3, 1).numpy() * 255).clip(0, 255).astype(np.uint8)
        )

    first = Image.fromarray(decoded_cameras[0][0])
    camera_height = round(first.height * camera_width / first.width)
    frame_width = camera_width * len(camera_keys)
    frame_height = camera_height + 58
    container = av.open(str(output_path), mode="w")
    stream = container.add_stream("libx264", rate=review_fps)
    stream.width = frame_width
    stream.height = frame_height + (frame_height % 2)
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "25", "preset": "veryfast"}

    font = ImageFont.load_default(size=17)
    small_font = ImageFont.load_default(size=14)
    for sample_index, source_frame in enumerate(sample_frames):
        included = int(source_frame) < cut_frame
        status = "INCLUDED IN TRAINING" if included else "EXCLUDED RESET"
        color = (31, 170, 89) if included else (220, 55, 47)
        canvas = Image.new("RGB", (frame_width, stream.height), (17, 21, 24))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, frame_width - 1, stream.height - 1), outline=color, width=10)
        draw.text((16, 10), status, fill=color, font=font)
        draw.text(
            (16, 34),
            f"source frame {int(source_frame)}  |  cutoff before frame {cut_frame}",
            fill=(220, 225, 228),
            font=small_font,
        )
        for camera_index, camera_key in enumerate(camera_keys):
            image = Image.fromarray(decoded_cameras[camera_index][sample_index]).resize(
                (camera_width, camera_height), Image.Resampling.LANCZOS
            )
            x = camera_index * camera_width
            canvas.paste(image, (x, 58))
            label = camera_key.removeprefix("observation.images.").replace("_", " ").upper()
            draw.rectangle((x + 10, 68, x + 145, 94), fill=(0, 0, 0))
            draw.text((x + 17, 73), label, fill=(255, 255, 255), font=small_font)

        video_frame = av.VideoFrame.from_ndarray(np.asarray(canvas), format="rgb24")
        for packet in stream.encode(video_frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _parse_episode_selection(value: str | None, available: set[int]) -> list[int]:
    if value is None:
        return sorted(available)
    selected: set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"Invalid descending episode range: {token}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(token))
    missing = selected - available
    if missing:
        raise ValueError(f"Episodes are not present in the trim report: {sorted(missing)}")
    return sorted(selected)


def _write_index(path: Path, reviews: list[dict], *, source_fps: float) -> None:
    cards = []
    for review in reviews:
        episode = review["episode_index"]
        cards.append(
            f"""
            <article class="episode" id="episode-{episode}">
              <div class="title">
                <h2>Episode {episode:03d}</h2>
                <span>{review['removed_frames']} reset frames excluded</span>
              </div>
              <video controls preload="metadata" src="{html.escape(review['clip'])}"></video>
              <p>Green until <strong>{review['boundary_s']:.2f}s</strong>, then red.
                 Fully open at source frame {review['fully_open_frame']}; training ends before
                 frame {review['cut_frame']} of {review['original_frames']}.</p>
            </article>
            """
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Release cutoff review</title>
  <style>
    :root {{ color-scheme: dark; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }}
    body {{ margin: 0; background: #111518; color: #e8ecef; }}
    header {{ position: sticky; top: 0; z-index: 2; padding: 18px 24px; background: #171d21ee; border-bottom: 1px solid #394249; }}
    h1, h2, p {{ margin: 0; }}
    header p {{ margin-top: 7px; color: #aeb8bf; }}
    main {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 18px; padding: 18px; }}
    .episode {{ padding: 14px; border: 1px solid #394249; border-radius: 6px; background: #1a2024; }}
    .title {{ display: flex; align-items: baseline; justify-content: space-between; gap: 12px; margin-bottom: 10px; }}
    h2 {{ font-size: 18px; }}
    .title span, .episode p {{ color: #aeb8bf; font-size: 13px; }}
    video {{ display: block; width: 100%; background: #000; }}
    .episode p {{ margin-top: 10px; line-height: 1.45; }}
    strong {{ color: #fff; }}
  </style>
</head>
<body>
  <header>
    <h1>Release cutoff review</h1>
    <p>{len(reviews)} episodes at {source_fps:g} source FPS. Green frames are in training; red frames are shown only to prove the reset was excluded.</p>
  </header>
  <main>{''.join(cards)}</main>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _read_parquet_tree(root: Path) -> pa.Table:
    paths = sorted(root.glob("**/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No Parquet files found under {root}")
    return pa.concat_tables([pq.read_table(path) for path in paths])


if __name__ == "__main__":
    raise SystemExit(main())
