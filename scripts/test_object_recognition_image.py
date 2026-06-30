#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xlerobot_agent.object_detection import ObjectDetectorConfig, detect_object_in_image


DEFAULT_LABELS = [
    "Tabasco sauce bottle",
    "hot sauce bottle",
    "small glass sauce bottle",
    "red cap sauce bottle",
    "pepper sauce bottle",
    "small condiment bottle",
]


def _image_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _draw_annotation(image_path: Path, detection: dict[str, Any], output_path: Path) -> None:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return

    bbox = detection.get("bbox_xyxy")
    if not isinstance(bbox, list) or len(bbox) < 4:
        return

    with Image.open(image_path) as image:
        annotated = image.convert("RGB")
        draw = ImageDraw.Draw(annotated)
        x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
        label = str(detection.get("label") or "detection")
        confidence = detection.get("confidence")
        caption = f"{label} {float(confidence) * 100.0:.1f}%" if isinstance(confidence, (float, int)) else label
        font = _annotation_font(annotated.size)
        draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=8)
        _draw_label(draw, x1, y1, caption, (255, 0, 0), font)
        annotated.save(output_path)


def _draw_combined_annotations(
    image_path: Path,
    detections: list[tuple[str, dict[str, Any]]],
    output_path: Path,
) -> None:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return

    colors = [
        (255, 0, 0),
        (0, 128, 255),
        (0, 180, 80),
        (255, 140, 0),
        (170, 80, 255),
        (255, 0, 170),
    ]
    with Image.open(image_path) as image:
        annotated = image.convert("RGB")
        draw = ImageDraw.Draw(annotated)
        font = _annotation_font(annotated.size)
        for index, (query_label, detection) in enumerate(detections):
            bbox = detection.get("bbox_xyxy")
            if not isinstance(bbox, list) or len(bbox) < 4:
                continue

            x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
            color = colors[index % len(colors)]
            confidence = detection.get("confidence")
            if isinstance(confidence, (float, int)):
                caption = f"{query_label} {float(confidence) * 100.0:.1f}%"
            else:
                caption = query_label
            draw.rectangle((x1, y1, x2, y2), outline=color, width=8)
            _draw_label(draw, x1, y1, caption, color, font)
        annotated.save(output_path)


def _annotation_font(image_size: tuple[int, int]) -> Any:
    try:
        from PIL import ImageFont
    except Exception:
        return None

    width, height = image_size
    font_size = max(42, int(min(width, height) * 0.035))
    for font_path in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(font_path, font_size)
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_label(draw: Any, x: float, y: float, caption: str, color: tuple[int, int, int], font: Any) -> None:
    padding = 12
    label_x = max(float(x), 0.0)
    label_y = max(float(y) - 58.0, 0.0)
    text_box = draw.textbbox((label_x, label_y), caption, font=font)
    background = (
        text_box[0] - padding,
        text_box[1] - padding,
        text_box[2] + padding,
        text_box[3] + padding,
    )
    draw.rectangle(background, fill=color)
    draw.text((label_x, label_y), caption, fill=(255, 255, 255), font=font)


def _draw_all_candidates_for_label(
    image_path: Path,
    query_label: str,
    detections: list[dict[str, Any]],
    output_path: Path,
) -> None:
    labelled_detections = [
        (f"{query_label} #{index}", detection)
        for index, detection in enumerate(detections, start=1)
    ]
    _draw_combined_annotations(image_path, labelled_detections, output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Robot42 object recognition on a still image.")
    parser.add_argument("image", type=Path, help="Image path to test.")
    parser.add_argument(
        "--label",
        action="append",
        dest="labels",
        help="Object label/query to test. Repeat to test multiple labels. Defaults to hot-sauce bottle candidates.",
    )
    parser.add_argument(
        "--provider",
        choices=("none", "mock", "replicate_grounding_dino"),
        default="replicate_grounding_dino",
    )
    parser.add_argument("--api-key", default=os.environ.get("REPLICATE_API_TOKEN") or os.environ.get("ROBOT42_OBJECT_DETECTOR_API_KEY"))
    parser.add_argument("--model", default="adirik/grounding-dino")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--timeout-s", type=float, default=90.0)
    parser.add_argument("--max-image-edge-px", type=int, default=1280)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/object_recognition_tests"))
    parser.add_argument(
        "--combined-annotations",
        action="store_true",
        help="Also write cross-label combined annotation images. Per-label candidate images are always written.",
    )
    args = parser.parse_args()

    image_path = args.image.expanduser().resolve()
    if not image_path.exists():
        raise SystemExit(f"Image does not exist: {image_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_url = _image_data_url(image_path)
    labels = args.labels or DEFAULT_LABELS
    print(f"Running {len(labels)} label probe(s) as separate detector requests.")
    config = ObjectDetectorConfig(
        provider=args.provider,
        api_key=args.api_key,
        model=args.model,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        min_confidence=args.min_confidence,
        timeout_s=args.timeout_s,
        max_image_edge_px=args.max_image_edge_px,
        jpeg_quality=args.jpeg_quality,
    )

    results: list[dict[str, Any]] = []
    selected_detections: list[tuple[str, dict[str, Any]]] = []
    all_candidate_detections: list[tuple[str, dict[str, Any]]] = []
    for index, label in enumerate(labels, start=1):
        result = detect_object_in_image(
            config=config,
            image_data_url=data_url,
            object_label=label,
            shot_id=f"image_test_{index:02d}",
            image_path=str(image_path),
        )
        results.append(result)
        selected = result.get("selected_detection") if isinstance(result.get("selected_detection"), dict) else None
        detections = result.get("detections") if isinstance(result.get("detections"), list) else []
        status = result.get("status")
        if detections:
            print(f"{label}: {status} candidates={len(detections)}")
            for detection_index, detection in enumerate(detections, start=1):
                if not isinstance(detection, dict):
                    continue
                all_candidate_detections.append((f"{label} #{detection_index}", detection))
                print(
                    f"  [{detection_index}] confidence={detection.get('confidence')} "
                    f"bbox={detection.get('bbox_xyxy')} label={detection.get('label')}"
                )

            safe_label = "".join(ch if ch.isalnum() else "_" for ch in label.lower()).strip("_")
            _draw_all_candidates_for_label(
                image_path,
                label,
                [detection for detection in detections if isinstance(detection, dict)],
                args.output_dir / f"{image_path.stem}_{safe_label}_all_candidates.jpg",
            )
        else:
            print(f"{label}: {status} ({result.get('reason')})")

        if selected:
            selected_detections.append((label, selected))
            safe_label = "".join(ch if ch.isalnum() else "_" for ch in label.lower()).strip("_")
            _draw_annotation(image_path, selected, args.output_dir / f"{image_path.stem}_{safe_label}.jpg")

    if args.combined_annotations and selected_detections:
        combined_path = args.output_dir / f"{image_path.stem}_combined.jpg"
        _draw_combined_annotations(image_path, selected_detections, combined_path)
        print(f"Wrote selected annotation: {combined_path}")

    if args.combined_annotations and all_candidate_detections:
        all_candidates_path = args.output_dir / f"{image_path.stem}_all_candidates.jpg"
        _draw_combined_annotations(image_path, all_candidate_detections, all_candidates_path)
        print(f"Wrote all-candidates annotation: {all_candidates_path}")

    output_json = args.output_dir / f"{image_path.stem}_detections.json"
    output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote results: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
