#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROP_IDS = {
    "width": cv2.CAP_PROP_FRAME_WIDTH,
    "height": cv2.CAP_PROP_FRAME_HEIGHT,
    "fps": cv2.CAP_PROP_FPS,
    "brightness": cv2.CAP_PROP_BRIGHTNESS,
    "contrast": cv2.CAP_PROP_CONTRAST,
    "saturation": cv2.CAP_PROP_SATURATION,
    "hue": cv2.CAP_PROP_HUE,
    "gain": cv2.CAP_PROP_GAIN,
    "exposure": cv2.CAP_PROP_EXPOSURE,
    "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
    "auto_wb": cv2.CAP_PROP_AUTO_WB,
    "wb_temperature": cv2.CAP_PROP_WB_TEMPERATURE,
    "fourcc": cv2.CAP_PROP_FOURCC,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture a raw OpenCV camera frame for debugging.")
    parser.add_argument("--index", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--fourcc",
        default=None,
        help="Optional OpenCV FOURCC, for example MJPG or YUYV.",
    )
    parser.add_argument(
        "--warmup-s",
        type=float,
        default=2.0,
        help="Seconds to read frames before saving, so auto exposure can settle.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="PROP=VALUE",
        help="Optional camera property override, for example exposure=-5 or gain=20.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to artifacts/opencv_snapshots/opencv_INDEX_TIMESTAMP.jpg.",
    )
    parser.add_argument(
        "--preview-boost",
        action="store_true",
        help="Also write a gamma/brightness boosted preview next to the raw capture.",
    )
    parser.add_argument("--preview-gamma", type=float, default=0.65)
    parser.add_argument("--preview-alpha", type=float, default=1.8)
    parser.add_argument("--preview-beta", type=float, default=12.0)
    return parser.parse_args()


def fourcc_value(text: str) -> int:
    if len(text) != 4:
        raise ValueError("--fourcc must be exactly four characters, for example MJPG")
    return cv2.VideoWriter_fourcc(*text)


def fourcc_text(value: float) -> str:
    raw = int(value)
    chars = [chr((raw >> (8 * i)) & 0xFF) for i in range(4)]
    if all(32 <= ord(char) <= 126 for char in chars):
        return "".join(chars)
    return str(raw)


def read_props(cap: cv2.VideoCapture) -> dict[str, Any]:
    props: dict[str, Any] = {}
    for name, prop_id in PROP_IDS.items():
        value = cap.get(prop_id)
        props[name] = fourcc_text(value) if name == "fourcc" else value
    return props


def parse_property_override(raw: str) -> tuple[str, float]:
    if "=" not in raw:
        raise ValueError(f"Invalid --set value {raw!r}; expected PROP=VALUE")
    name, value = raw.split("=", 1)
    name = name.strip()
    if name not in PROP_IDS:
        valid = ", ".join(sorted(PROP_IDS))
        raise ValueError(f"Unknown property {name!r}. Valid properties: {valid}")
    return name, float(value.strip())


def apply_gamma_bgr(frame: np.ndarray, *, gamma: float, alpha: float, beta: float) -> np.ndarray:
    adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    gamma = max(gamma, 0.01)
    table = np.array([(idx / 255.0) ** gamma * 255 for idx in range(256)], dtype=np.uint8)
    return cv2.LUT(adjusted, table)


def image_stats(frame: np.ndarray) -> dict[str, Any]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return {
        "shape": list(frame.shape),
        "gray_mean": round(float(gray.mean()), 3),
        "gray_std": round(float(gray.std()), 3),
        "gray_min": int(gray.min()),
        "gray_max": int(gray.max()),
        "pct_under_16": round(float((gray < 16).mean() * 100), 3),
        "pct_over_240": round(float((gray > 240).mean() * 100), 3),
    }


def default_output(index: int) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("artifacts/opencv_snapshots") / f"opencv_{index}_{stamp}.jpg"


def main() -> int:
    args = parse_args()

    cap = cv2.VideoCapture(args.index)
    if not cap.isOpened():
        print(f"Could not open OpenCV camera index {args.index}.")
        return 1

    try:
        if args.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, fourcc_value(args.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)

        set_results = []
        for raw_override in args.set:
            name, value = parse_property_override(raw_override)
            before = cap.get(PROP_IDS[name])
            ok = cap.set(PROP_IDS[name], value)
            after = cap.get(PROP_IDS[name])
            set_results.append({"property": name, "requested": value, "ok": bool(ok), "before": before, "after": after})

        start = time.monotonic()
        frame = None
        frames_read = 0
        while time.monotonic() - start < max(args.warmup_s, 0.0) or frame is None:
            ok, candidate = cap.read()
            if ok and candidate is not None:
                frame = candidate
                frames_read += 1
            else:
                time.sleep(0.02)

        output = args.output or default_output(args.index)
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), frame):
            print(f"Failed to write {output}")
            return 1

        result: dict[str, Any] = {
            "camera_index": args.index,
            "output": str(output),
            "frames_read": frames_read,
            "properties": read_props(cap),
            "set_results": set_results,
            "image_stats": image_stats(frame),
        }

        if args.preview_boost:
            boosted = apply_gamma_bgr(
                frame,
                gamma=args.preview_gamma,
                alpha=args.preview_alpha,
                beta=args.preview_beta,
            )
            boosted_output = output.with_name(f"{output.stem}_boost{output.suffix}")
            if cv2.imwrite(str(boosted_output), boosted):
                result["preview_boost_output"] = str(boosted_output)
                result["preview_boost_stats"] = image_stats(boosted)

        print(json.dumps(result, indent=2))
        return 0
    finally:
        cap.release()


if __name__ == "__main__":
    raise SystemExit(main())
