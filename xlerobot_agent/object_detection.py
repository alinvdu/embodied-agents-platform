from __future__ import annotations

import base64
from dataclasses import dataclass
import io
import json
import time
from typing import Any
import urllib.error
import urllib.request

try:
    from PIL import Image
except Exception:  # pragma: no cover - exercised only when Pillow is unavailable.
    Image = None  # type: ignore[assignment]


_REPLICATE_VERSION_CACHE: dict[str, str] = {}


@dataclass(frozen=True)
class ObjectDetectorConfig:
    provider: str = "none"
    api_key: str | None = None
    model: str = "adirik/grounding-dino"
    model_version: str | None = None
    box_threshold: float = 0.25
    text_threshold: float = 0.25
    min_confidence: float = 0.25
    timeout_s: float = 90.0
    max_image_edge_px: int = 1280
    jpeg_quality: int = 85


def detect_object_in_image(
    *,
    config: ObjectDetectorConfig,
    image_data_url: str,
    object_label: str,
    shot_id: str,
    image_path: str | None = None,
) -> dict[str, Any]:
    provider = (config.provider or "none").strip().lower()
    target = (object_label or "").strip()
    if not target:
        return {
            "status": "skipped",
            "provider": provider,
            "shot_id": shot_id,
            "object_label": object_label,
            "reason": "No object label was requested for this shot.",
        }
    if provider in {"", "none", "disabled"}:
        return {
            "status": "not_configured",
            "provider": provider,
            "shot_id": shot_id,
            "object_label": object_label,
            "reason": "Object detection provider is not configured.",
        }
    if provider == "mock":
        return _mock_detection(target=target, shot_id=shot_id, image_path=image_path)
    if provider == "replicate_grounding_dino":
        return _replicate_grounding_dino_detection(
            config=config,
            image_data_url=image_data_url,
            object_label=target,
            shot_id=shot_id,
            image_path=image_path,
        )
    return {
        "status": "failed",
        "provider": provider,
        "shot_id": shot_id,
        "object_label": object_label,
        "reason": f"Unsupported object detection provider `{config.provider}`.",
    }


def _mock_detection(*, target: str, shot_id: str, image_path: str | None) -> dict[str, Any]:
    detection = {
        "detection_id": f"{shot_id}_mock_det_1",
        "label": target,
        "confidence": 0.99,
        "bbox_xyxy": [0.35, 0.25, 0.65, 0.75],
        "source_bbox": [0.35, 0.25, 0.65, 0.75],
        "image_path": image_path,
    }
    return {
        "status": "matched",
        "provider": "mock",
        "shot_id": shot_id,
        "object_label": target,
        "detections": [detection],
        "selected_detection_id": detection["detection_id"],
        "selected_detection": detection,
        "reason": "Mock detector returned a deterministic centered detection.",
    }


def _replicate_grounding_dino_detection(
    *,
    config: ObjectDetectorConfig,
    image_data_url: str,
    object_label: str,
    shot_id: str,
    image_path: str | None,
) -> dict[str, Any]:
    api_key = config.api_key
    if not api_key:
        return {
            "status": "unavailable",
            "provider": "replicate_grounding_dino",
            "shot_id": shot_id,
            "object_label": object_label,
            "reason": "Missing Replicate API token. Set REPLICATE_API_TOKEN or ROBOT42_OBJECT_DETECTOR_API_KEY.",
        }
    try:
        image_data_url, image_preprocess = _prepare_replicate_image_data_url(
            image_data_url,
            max_edge_px=config.max_image_edge_px,
            jpeg_quality=config.jpeg_quality,
        )
        version = config.model_version or _replicate_latest_version(
            model=config.model,
            api_key=api_key,
            timeout_s=config.timeout_s,
        )
        prediction = _replicate_create_prediction(
            version=version,
            image_data_url=image_data_url,
            object_label=object_label,
            config=config,
            api_key=api_key,
        )
    except Exception as exc:
        return {
            "status": "failed",
            "provider": "replicate_grounding_dino",
            "shot_id": shot_id,
            "object_label": object_label,
            "reason": f"Replicate Grounding DINO request failed: {exc}",
        }

    status = str(prediction.get("status") or "").lower()
    if status not in {"succeeded", "success", "completed"}:
        return {
            "status": "failed",
            "provider": "replicate_grounding_dino",
            "shot_id": shot_id,
            "object_label": object_label,
            "reason": prediction.get("error") or f"Replicate prediction ended with status `{status}`.",
            "replicate_prediction_id": prediction.get("id"),
            "replicate_status": status,
        }

    output = prediction.get("output")
    detections = _normalize_replicate_detections(
        output,
        shot_id=shot_id,
        object_label=object_label,
        image_path=image_path,
        min_confidence=config.min_confidence,
        image_preprocess=image_preprocess,
    )
    result_image = output.get("result_image") if isinstance(output, dict) else None
    result = {
        "status": "matched" if detections else "not_found",
        "provider": "replicate_grounding_dino",
        "shot_id": shot_id,
        "object_label": object_label,
        "detections": detections,
        "selected_detection_id": detections[0]["detection_id"] if detections else None,
        "selected_detection": detections[0] if detections else None,
        "replicate_prediction_id": prediction.get("id"),
        "replicate_status": status,
        "provider_result_image_url": result_image,
        "image_preprocess": image_preprocess,
        "reason": (
            f"Detected `{object_label}` with Replicate Grounding DINO."
            if detections
            else f"Replicate Grounding DINO returned no detections above confidence {config.min_confidence:.2f}."
        ),
    }
    return result


def _replicate_latest_version(*, model: str, api_key: str, timeout_s: float) -> str:
    cached = _REPLICATE_VERSION_CACHE.get(model)
    if cached:
        return cached
    owner, name = _split_replicate_model(model)
    payload = _replicate_request_json(
        url=f"https://api.replicate.com/v1/models/{owner}/{name}",
        api_key=api_key,
        method="GET",
        timeout_s=timeout_s,
    )
    latest = payload.get("latest_version") if isinstance(payload.get("latest_version"), dict) else {}
    version = latest.get("id")
    if not isinstance(version, str) or not version:
        raise RuntimeError(f"Could not resolve latest Replicate version for `{model}`.")
    _REPLICATE_VERSION_CACHE[model] = version
    return version


def _replicate_create_prediction(
    *,
    version: str,
    image_data_url: str,
    object_label: str,
    config: ObjectDetectorConfig,
    api_key: str,
) -> dict[str, Any]:
    payload = {
        "version": version,
        "input": {
            "image": image_data_url,
            "query": object_label,
            "box_threshold": float(config.box_threshold),
            "text_threshold": float(config.text_threshold),
            "show_visualisation": True,
        },
    }
    prediction = _replicate_request_json(
        url="https://api.replicate.com/v1/predictions",
        api_key=api_key,
        method="POST",
        payload=payload,
        timeout_s=config.timeout_s,
        headers={"Prefer": "wait=60", "Cancel-After": f"{max(int(config.timeout_s), 5)}s"},
    )
    status = str(prediction.get("status") or "").lower()
    deadline = time.time() + max(float(config.timeout_s), 5.0)
    get_url = (prediction.get("urls") or {}).get("get") if isinstance(prediction.get("urls"), dict) else None
    while status in {"starting", "processing", "queued"} and isinstance(get_url, str) and time.time() < deadline:
        time.sleep(1.0)
        prediction = _replicate_request_json(
            url=get_url,
            api_key=api_key,
            method="GET",
            timeout_s=min(max(deadline - time.time(), 1.0), 15.0),
        )
        status = str(prediction.get("status") or "").lower()
    return prediction


def _replicate_request_json(
    *,
    url: str,
    api_key: str,
    method: str,
    timeout_s: float,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request_headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "User-Agent": "Robot42/0.1 ReplicateAPIClient",
    }
    if payload is not None:
        request_headers["Content-Type"] = "application/json"
    if headers:
        request_headers.update(headers)
    max_attempts = 4
    raw = ""
    for attempt in range(1, max_attempts + 1):
        request = urllib.request.Request(url, data=body, headers=request_headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=max(float(timeout_s), 1.0)) as response:
                raw = response.read().decode("utf-8")
            break
        except urllib.error.HTTPError as exc:
            try:
                raw_error = exc.read().decode("utf-8")
            except Exception:
                raw_error = str(exc)
            if exc.code == 429 and attempt < max_attempts:
                time.sleep(_replicate_retry_after_s(exc, raw_error))
                continue
            raise RuntimeError(f"HTTP {exc.code}: {raw_error[:500]}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(str(exc)) from exc
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError("Replicate returned non-JSON response.") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("Replicate response was not a JSON object.")
    return parsed


def _replicate_retry_after_s(exc: urllib.error.HTTPError, raw_error: str) -> float:
    try:
        retry_after = exc.headers.get("Retry-After")
        if retry_after:
            return max(min(float(retry_after) + 0.5, 20.0), 1.0)
    except Exception:
        pass
    try:
        parsed = json.loads(raw_error or "{}")
        retry_after = parsed.get("retry_after") if isinstance(parsed, dict) else None
        if retry_after is not None:
            return max(min(float(retry_after) + 0.5, 20.0), 1.0)
    except Exception:
        pass
    return 3.0


def _normalize_replicate_detections(
    output: Any,
    *,
    shot_id: str,
    object_label: str,
    image_path: str | None,
    min_confidence: float,
    image_preprocess: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    raw_detections: list[Any]
    if isinstance(output, dict) and isinstance(output.get("detections"), list):
        raw_detections = output["detections"]
    elif isinstance(output, list):
        raw_detections = output
    else:
        raw_detections = []

    detections: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_detections, start=1):
        if not isinstance(raw, dict):
            continue
        bbox = _extract_bbox_xyxy(raw)
        confidence = _extract_confidence(raw)
        if bbox is None or confidence is None or confidence < min_confidence:
            continue
        bbox = _scale_bbox_to_source_image(bbox, image_preprocess)
        label = str(raw.get("label") or raw.get("class") or raw.get("phrase") or object_label)
        detection = {
            "detection_id": f"{shot_id}_det_{index}",
            "label": label,
            "confidence": round(float(confidence), 4),
            "bbox_xyxy": bbox,
            "source_bbox": raw.get("bbox") or raw.get("box") or raw.get("xyxy") or raw.get("coordinates"),
            "image_path": image_path,
            "raw": raw,
        }
        detections.append(detection)
    detections.sort(key=lambda item: float(item.get("confidence", 0.0) or 0.0), reverse=True)
    return detections


def _prepare_replicate_image_data_url(
    image_data_url: str,
    *,
    max_edge_px: int,
    jpeg_quality: int,
) -> tuple[str, dict[str, Any]]:
    metadata: dict[str, Any] = {
        "status": "unchanged",
        "reason": "image was not resized",
    }
    if Image is None:
        metadata["reason"] = "Pillow is unavailable; sent original data URL"
        return image_data_url, metadata
    try:
        header, encoded = image_data_url.split(",", 1)
        if ";base64" not in header:
            metadata["reason"] = "image data URL is not base64; sent original data URL"
            return image_data_url, metadata
        mime_type = header.removeprefix("data:").split(";", 1)[0] or "application/octet-stream"
        raw = base64.b64decode(encoded)
        with Image.open(io.BytesIO(raw)) as opened:
            image = opened.convert("RGB")
            source_width, source_height = image.size
            metadata.update(
                {
                    "source_mime_type": mime_type,
                    "source_width": source_width,
                    "source_height": source_height,
                    "source_bytes": len(raw),
                }
            )
            max_edge = max(int(max_edge_px or 0), 1)
            scale = min(1.0, float(max_edge) / float(max(source_width, source_height)))
            if scale < 1.0:
                resized_width = max(int(round(source_width * scale)), 1)
                resized_height = max(int(round(source_height * scale)), 1)
                resample = getattr(Image, "Resampling", Image).LANCZOS
                image = image.resize((resized_width, resized_height), resample)
            else:
                resized_width, resized_height = source_width, source_height
            buffer = io.BytesIO()
            image.save(
                buffer,
                format="JPEG",
                quality=max(1, min(int(jpeg_quality), 95)),
                optimize=True,
            )
    except Exception as exc:
        metadata["status"] = "unchanged"
        metadata["reason"] = f"could not preprocess image; sent original data URL: {exc}"
        return image_data_url, metadata

    data = buffer.getvalue()
    metadata.update(
        {
            "status": "resized" if (resized_width, resized_height) != (source_width, source_height) else "reencoded",
            "sent_mime_type": "image/jpeg",
            "sent_width": resized_width,
            "sent_height": resized_height,
            "sent_bytes": len(data),
            "max_image_edge_px": max_edge_px,
            "jpeg_quality": max(1, min(int(jpeg_quality), 95)),
        }
    )
    return f"data:image/jpeg;base64,{base64.b64encode(data).decode('ascii')}", metadata


def _scale_bbox_to_source_image(bbox: list[float], image_preprocess: dict[str, Any] | None) -> list[float]:
    if not isinstance(image_preprocess, dict):
        return bbox
    try:
        source_width = float(image_preprocess.get("source_width") or 0)
        source_height = float(image_preprocess.get("source_height") or 0)
        sent_width = float(image_preprocess.get("sent_width") or 0)
        sent_height = float(image_preprocess.get("sent_height") or 0)
    except Exception:
        return bbox
    if min(source_width, source_height, sent_width, sent_height) <= 0.0:
        return bbox
    if abs(source_width - sent_width) < 0.001 and abs(source_height - sent_height) < 0.001:
        return bbox
    if max(abs(value) for value in bbox[:4]) <= 1.5:
        return bbox
    x_scale = source_width / sent_width
    y_scale = source_height / sent_height
    return [
        round(float(bbox[0]) * x_scale, 3),
        round(float(bbox[1]) * y_scale, 3),
        round(float(bbox[2]) * x_scale, 3),
        round(float(bbox[3]) * y_scale, 3),
    ]


def _extract_bbox_xyxy(raw: dict[str, Any]) -> list[float] | None:
    for key in ("bbox", "box", "xyxy", "coordinates"):
        value = raw.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= 4:
            return [round(float(item), 3) for item in value[:4]]
        if isinstance(value, dict):
            parsed = _bbox_from_dict(value)
            if parsed is not None:
                return parsed
    parsed = _bbox_from_dict(raw)
    if parsed is not None:
        return parsed
    if all(key in raw for key in ("x", "y", "width", "height")):
        x = float(raw["x"])
        y = float(raw["y"])
        width = float(raw["width"])
        height = float(raw["height"])
        return [
            round(x - width / 2.0, 3),
            round(y - height / 2.0, 3),
            round(x + width / 2.0, 3),
            round(y + height / 2.0, 3),
        ]
    return None


def _bbox_from_dict(value: dict[str, Any]) -> list[float] | None:
    key_sets = (
        ("x1", "y1", "x2", "y2"),
        ("left", "top", "right", "bottom"),
        ("xmin", "ymin", "xmax", "ymax"),
    )
    for keys in key_sets:
        if all(key in value for key in keys):
            return [round(float(value[key]), 3) for key in keys]
    return None


def _extract_confidence(raw: dict[str, Any]) -> float | None:
    for key in ("confidence", "score", "logit", "probability"):
        value = raw.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return None


def _split_replicate_model(model: str) -> tuple[str, str]:
    if "/" not in model:
        raise RuntimeError(f"Replicate model must be in owner/name form, got `{model}`.")
    owner, name = model.split("/", 1)
    owner = owner.strip()
    name = name.strip()
    if not owner or not name:
        raise RuntimeError(f"Replicate model must be in owner/name form, got `{model}`.")
    return owner, name
