from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
import time
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is expected, but keep the fallback defensive.
    np = None  # type: ignore[assignment]


def build_scene_snapshot(
    *,
    current_pose: str,
    visible_objects: Sequence[str],
    image_descriptions: Sequence[str],
    metadata: Mapping[str, Any],
    target: str | None = None,
    cached_scene: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    sensors = dict(_mapping_value(metadata, "sensors"))
    if cached_scene and isinstance(cached_scene.get("annotations"), list):
        annotations = [dict(item) for item in cached_scene["annotations"] if isinstance(item, dict)]
    else:
        annotations = _annotations_from_metadata(metadata)
    if not annotations:
        annotations = _annotations_from_observations(
            current_pose=current_pose,
            visible_objects=list(visible_objects),
            image_descriptions=list(image_descriptions),
            metadata=metadata,
            target=target,
        )
    annotations = _enrich_annotations_with_rgbd(annotations, metadata)

    labels = [item.get("label") for item in annotations if item.get("label")]
    unique_labels = sorted({str(label) for label in labels})
    summary = (
        f"Perception at `{current_pose}` sees {unique_labels or ['no_confirmed_objects']} "
        f"with streams {sorted(sensors.keys()) or ['no_streams']}."
    )
    return {
        "scene_summary": summary,
        "annotations": annotations,
        "available_streams": sorted(sensors.keys()),
        "query_target": target,
        "generated_at": time.time(),
    }


def ground_object_matches(scene: Mapping[str, Any], query: str | None) -> dict[str, Any]:
    if not query:
        return {"query": query, "matches": []}
    query_lower = query.lower().strip()
    annotations = [dict(item) for item in scene.get("annotations", []) if isinstance(item, dict)]
    matches = [
        item for item in annotations if query_lower in str(item.get("label", "")).lower()
    ]
    matches.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
    return {
        "query": query,
        "matches": matches,
    }


def waypoint_from_matches(
    scene: Mapping[str, Any],
    *,
    query: str | None = None,
    match: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    selected = dict(match) if match else None
    if selected is None:
        matches = ground_object_matches(scene, query).get("matches", [])
        if matches:
            selected = dict(matches[0])
    if selected is None:
        return None

    waypoint = selected.get("waypoint_hint")
    if isinstance(waypoint, Mapping):
        pose = dict(waypoint)
    else:
        centroid = dict(_mapping_value(selected, "centroid_3d"))
        pose = {
            "frame": str(centroid.get("frame", "map")),
            "x": float(centroid.get("x", 1.0)) - 0.8,
            "y": float(centroid.get("y", 0.0)),
            "yaw": float(centroid.get("yaw", 0.0)),
            "approach_distance_m": 0.8,
        }
    return {
        "target_label": selected.get("label"),
        "confidence": selected.get("confidence", 0.0),
        "pose": pose,
        "source_annotation": selected,
    }


def _annotations_from_metadata(metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    perception = _mapping_value(metadata, "perception")
    annotations = perception.get("annotations")
    if isinstance(annotations, list):
        return [dict(item) for item in annotations if isinstance(item, Mapping)]
    return []


def _annotations_from_observations(
    *,
    current_pose: str,
    visible_objects: list[str],
    image_descriptions: list[str],
    metadata: Mapping[str, Any],
    target: str | None,
) -> list[dict[str, Any]]:
    labels = set(visible_objects)
    joined_descriptions = " ".join(image_descriptions).lower()
    if target and target.lower() in joined_descriptions:
        labels.add(target)

    anchors = _mapping_value(metadata, "object_anchors")
    annotations: list[dict[str, Any]] = []
    for index, label in enumerate(sorted(labels)):
        anchor = dict(_mapping_value(anchors, label))
        centroid = {
            "frame": str(anchor.get("frame", "map")),
            "x": float(anchor.get("x", 1.2 + index * 0.45)),
            "y": float(anchor.get("y", index * 0.2)),
            "z": float(anchor.get("z", 0.9)),
            "yaw": float(anchor.get("yaw", 0.0)),
        }
        depth_m = float(anchor.get("depth_m", max(0.4, centroid["x"])))
        waypoint_hint = {
            "frame": centroid["frame"],
            "x": centroid["x"] - 0.8,
            "y": centroid["y"],
            "yaw": centroid["yaw"],
            "approach_distance_m": 0.8,
        }
        annotations.append(
            {
                "label": label,
                "confidence": float(anchor.get("confidence", 0.86 if label in visible_objects else 0.55)),
                "bbox_2d": dict(
                    _mapping_value(
                        anchor,
                        "bbox_2d",
                        default={"x": 120 + index * 40, "y": 80 + index * 20, "w": 110, "h": 150},
                    )
                ),
                "mask_id": str(anchor.get("mask_id", f"mask_{label}_{index}")),
                "centroid_3d": centroid,
                "depth_m": depth_m,
                "overlay_text": f"{label} @ {depth_m:.2f}m from `{current_pose}`",
                "waypoint_hint": waypoint_hint,
                "source": "metadata_anchor" if label in anchors else "synthetic_visible_object",
            }
        )
    return annotations


def _enrich_annotations_with_rgbd(
    annotations: Sequence[dict[str, Any]],
    metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if np is None:
        return _enrich_annotations_with_rgbd_python(annotations, metadata)
    depth_image = _resolve_depth_image(metadata)
    intrinsics = _resolve_intrinsics(metadata)
    if depth_image is None or intrinsics is None:
        return [dict(item) for item in annotations]
    pose_mat = _resolve_pose_matrix(metadata)

    enriched: list[dict[str, Any]] = []
    for item in annotations:
        annotation = dict(item)
        if annotation.get("centroid_3d") and annotation.get("depth_m") is not None:
            enriched.append(annotation)
            continue
        pixels = _annotation_pixels(annotation, depth_image.shape)
        centroid = _centroid_from_depth_pixels(pixels, depth_image, intrinsics, pose_mat)
        if centroid is None:
            enriched.append(annotation)
            continue
        annotation["centroid_3d"] = centroid["centroid"]
        annotation["depth_m"] = centroid["depth_m"]
        annotation.setdefault(
            "waypoint_hint",
            {
                "frame": centroid["centroid"]["frame"],
                "x": centroid["centroid"]["x"] - 0.8,
                "y": centroid["centroid"]["y"],
                "yaw": centroid["centroid"].get("yaw", 0.0),
                "approach_distance_m": 0.8,
            },
        )
        if annotation.get("label") and not annotation.get("overlay_text"):
            annotation["overlay_text"] = (
                f"{annotation['label']} @ {centroid['depth_m']:.2f}m "
                f"from `{centroid['centroid']['frame']}`"
            )
        annotation.setdefault("source", "rgbd_projection")
        enriched.append(annotation)
    return enriched


def _enrich_annotations_with_rgbd_python(
    annotations: Sequence[dict[str, Any]],
    metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    depth_image, intrinsics, pose_mat = _resolve_depth_payload_python(metadata)
    if depth_image is None or intrinsics is None:
        return [dict(item) for item in annotations]
    height = len(depth_image)
    width = len(depth_image[0]) if height else 0
    if height <= 0 or width <= 0:
        return [dict(item) for item in annotations]

    enriched: list[dict[str, Any]] = []
    for item in annotations:
        annotation = dict(item)
        if annotation.get("centroid_3d") and annotation.get("depth_m") is not None:
            enriched.append(annotation)
            continue
        bbox = _mapping_value(annotation, "bbox_2d")
        if not bbox:
            enriched.append(annotation)
            continue
        centroid = _centroid_from_depth_bbox_python(bbox, depth_image, intrinsics, pose_mat)
        if centroid is None:
            enriched.append(annotation)
            continue
        annotation["centroid_3d"] = centroid["centroid"]
        annotation["depth_m"] = centroid["depth_m"]
        annotation.setdefault(
            "waypoint_hint",
            {
                "frame": centroid["centroid"]["frame"],
                "x": centroid["centroid"]["x"] - 0.8,
                "y": centroid["centroid"]["y"],
                "yaw": centroid["centroid"].get("yaw", 0.0),
                "approach_distance_m": 0.8,
            },
        )
        annotation.setdefault("source", "rgbd_projection")
        enriched.append(annotation)
    return enriched


def _resolve_depth_payload_python(
    metadata: Mapping[str, Any],
) -> tuple[list[list[float]] | None, list[list[float]] | None, list[list[float]] | None]:
    sensors = _mapping_value(metadata, "sensors")
    for key in ("depth", "rgbd", "orbbec_depth"):
        payload = _mapping_value(sensors, key)
        data = payload.get("data")
        intrinsics = payload.get("intrinsics")
        if not isinstance(data, Sequence) or not isinstance(intrinsics, Sequence):
            continue
        depth = _float_matrix(data)
        camera = _float_matrix(intrinsics)
        if depth and camera and len(camera) >= 3 and all(len(row) >= 3 for row in camera[:3]):
            pose = _float_matrix(payload.get("pose_mat"))
            return depth, camera, pose
    return None, None, None


def _centroid_from_depth_bbox_python(
    bbox: Mapping[str, Any],
    depth_image: list[list[float]],
    intrinsics: list[list[float]],
    pose_mat: list[list[float]] | None,
) -> dict[str, Any] | None:
    height = len(depth_image)
    width = len(depth_image[0]) if height else 0
    x0 = int(bbox.get("x", bbox.get("x1", 0)))
    y0 = int(bbox.get("y", bbox.get("y1", 0)))
    if "w" in bbox and "h" in bbox:
        x1 = x0 + int(bbox.get("w", 0))
        y1 = y0 + int(bbox.get("h", 0))
    else:
        x1 = int(bbox.get("x2", x0))
        y1 = int(bbox.get("y2", y0))
    x0 = max(0, min(width - 1, x0))
    y0 = max(0, min(height - 1, y0))
    x1 = max(x0 + 1, min(width, x1))
    y1 = max(y0 + 1, min(height, y1))

    fx = float(intrinsics[0][0])
    fy = float(intrinsics[1][1])
    cx = float(intrinsics[0][2])
    cy = float(intrinsics[1][2])
    points: list[tuple[float, float, float]] = []
    depths: list[float] = []
    for py in range(y0, y1):
        for px in range(x0, x1):
            try:
                z = float(depth_image[py][px])
            except Exception:
                continue
            if z <= 0.0:
                continue
            x = ((float(px) - cx) * z) / max(fx, 1e-6)
            y = ((float(py) - cy) * z) / max(fy, 1e-6)
            points.append((x, y, z))
            depths.append(z)
    if not points:
        return None
    x = sum(point[0] for point in points) / len(points)
    y = sum(point[1] for point in points) / len(points)
    z = sum(point[2] for point in points) / len(points)
    frame = "camera"
    if pose_mat and len(pose_mat) >= 4 and all(len(row) >= 4 for row in pose_mat[:4]):
        x, y, z = _transform_point_python((x, y, z), pose_mat)
        frame = "map"
    return {
        "centroid": {"frame": frame, "x": x, "y": y, "z": z, "yaw": 0.0},
        "depth_m": _median_python(depths),
    }


def _float_matrix(value: Any) -> list[list[float]] | None:
    if not isinstance(value, Sequence):
        return None
    matrix: list[list[float]] = []
    for row in value:
        if not isinstance(row, Sequence):
            return None
        try:
            matrix.append([float(item) for item in row])
        except Exception:
            return None
    return matrix if matrix else None


def _transform_point_python(point: tuple[float, float, float], pose_mat: list[list[float]]) -> tuple[float, float, float]:
    x, y, z = point
    return (
        pose_mat[0][0] * x + pose_mat[0][1] * y + pose_mat[0][2] * z + pose_mat[0][3],
        pose_mat[1][0] * x + pose_mat[1][1] * y + pose_mat[1][2] * z + pose_mat[1][3],
        pose_mat[2][0] * x + pose_mat[2][1] * y + pose_mat[2][2] * z + pose_mat[2][3],
    )


def _median_python(values: Sequence[float]) -> float:
    sorted_values = sorted(float(value) for value in values)
    if not sorted_values:
        return 0.0
    middle = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[middle]
    return (sorted_values[middle - 1] + sorted_values[middle]) / 2.0


def _resolve_depth_image(metadata: Mapping[str, Any]) -> Any | None:
    sensors = _mapping_value(metadata, "sensors")
    for key in ("depth", "rgbd", "orbbec_depth"):
        payload = sensors.get(key)
        array = _array_from_payload(payload)
        if array is None:
            continue
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[:, :, 0]
        if array.ndim == 2:
            return array.astype(float)
    return None


def _resolve_intrinsics(metadata: Mapping[str, Any]) -> Any | None:
    sensors = _mapping_value(metadata, "sensors")
    for sensor_key in ("depth", "rgb", "rgbd", "orbbec_depth", "orbbec_rgb"):
        payload = _mapping_value(sensors, sensor_key)
        intrinsics = _array_from_payload(payload.get("intrinsics"))
        if intrinsics is not None and getattr(intrinsics, "shape", None) == (3, 3):
            return intrinsics.astype(float)
    intrinsics = _array_from_payload(metadata.get("camera_intrinsics"))
    if intrinsics is not None and getattr(intrinsics, "shape", None) == (3, 3):
        return intrinsics.astype(float)
    return None


def _resolve_pose_matrix(metadata: Mapping[str, Any]) -> Any | None:
    sensors = _mapping_value(metadata, "sensors")
    for sensor_key in ("depth", "rgb", "rgbd", "orbbec_depth", "orbbec_rgb"):
        payload = _mapping_value(sensors, sensor_key)
        pose_mat = _array_from_payload(payload.get("pose_mat"))
        if pose_mat is not None and getattr(pose_mat, "shape", None) == (4, 4):
            return pose_mat.astype(float)
    pose_mat = _array_from_payload(metadata.get("camera_pose_mat"))
    if pose_mat is not None and getattr(pose_mat, "shape", None) == (4, 4):
        return pose_mat.astype(float)
    return None


def _annotation_pixels(annotation: Mapping[str, Any], depth_shape: tuple[int, int]) -> Any | None:
    if np is None:
        return None
    points = annotation.get("mask_pixels")
    if isinstance(points, Sequence) and points and isinstance(points[0], Sequence):
        array = np.asarray(points, dtype=int)
        if array.ndim == 2 and array.shape[1] >= 2:
            return _clip_pixels(array[:, :2], depth_shape)

    bbox = _mapping_value(annotation, "bbox_2d")
    if not bbox:
        return None
    x0 = int(bbox.get("x", bbox.get("x1", 0)))
    y0 = int(bbox.get("y", bbox.get("y1", 0)))
    if "w" in bbox and "h" in bbox:
        x1 = x0 + int(bbox.get("w", 0))
        y1 = y0 + int(bbox.get("h", 0))
    else:
        x1 = int(bbox.get("x2", x0))
        y1 = int(bbox.get("y2", y0))
    x0 = max(0, min(depth_shape[1] - 1, x0))
    y0 = max(0, min(depth_shape[0] - 1, y0))
    x1 = max(x0 + 1, min(depth_shape[1], x1))
    y1 = max(y0 + 1, min(depth_shape[0], y1))
    step_x = max(1, (x1 - x0) // 12)
    step_y = max(1, (y1 - y0) // 12)
    xs = np.arange(x0, x1, step_x, dtype=int)
    ys = np.arange(y0, y1, step_y, dtype=int)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    pixels = np.column_stack((grid_x.reshape(-1), grid_y.reshape(-1)))
    return _clip_pixels(pixels, depth_shape)


def _clip_pixels(pixels: Any, depth_shape: tuple[int, int]) -> Any:
    if np is None:
        return pixels
    clipped = np.asarray(pixels, dtype=int)
    clipped[:, 0] = np.clip(clipped[:, 0], 0, depth_shape[1] - 1)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, depth_shape[0] - 1)
    return clipped


def _centroid_from_depth_pixels(
    pixels: Any,
    depth_image: Any,
    intrinsics: Any,
    pose_mat: Any | None,
) -> dict[str, Any] | None:
    if np is None or pixels is None:
        return None
    depth = np.asarray(depth_image, dtype=float)
    points_2d = np.asarray(pixels, dtype=int)
    if points_2d.size == 0:
        return None
    zs = depth[points_2d[:, 1], points_2d[:, 0]]
    valid = np.isfinite(zs) & (zs > 0.0)
    if not np.any(valid):
        return None
    points_2d = points_2d[valid]
    zs = zs[valid]

    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    xs = ((points_2d[:, 0].astype(float) - cx) * zs) / max(fx, 1e-6)
    ys = ((points_2d[:, 1].astype(float) - cy) * zs) / max(fy, 1e-6)
    points_3d = np.column_stack((xs, ys, zs))

    frame = "camera"
    if pose_mat is not None:
        homogeneous = np.column_stack((points_3d, np.ones(points_3d.shape[0], dtype=float)))
        transformed = (pose_mat @ homogeneous.T).T
        points_3d = transformed[:, :3]
        frame = "map"
    centroid = points_3d.mean(axis=0)
    depth_m = float(np.median(zs))
    return {
        "centroid": {
            "frame": frame,
            "x": float(centroid[0]),
            "y": float(centroid[1]),
            "z": float(centroid[2]),
            "yaw": 0.0,
        },
        "depth_m": depth_m,
    }


def _array_from_payload(payload: Any) -> Any | None:
    if np is None or payload is None:
        return None
    if isinstance(payload, np.ndarray):
        return payload
    if isinstance(payload, (list, tuple)):
        array = np.asarray(payload)
        return array if array.size else None
    if not isinstance(payload, Mapping):
        return None
    if isinstance(payload.get("data"), (list, tuple)):
        array = np.asarray(payload["data"])
        return array if array.size else None
    data_base64 = payload.get("data_base64")
    shape = payload.get("shape")
    dtype = payload.get("dtype")
    if isinstance(data_base64, str) and isinstance(shape, Sequence) and dtype:
        raw = base64.b64decode(data_base64.encode("utf-8"))
        return np.frombuffer(raw, dtype=np.dtype(str(dtype))).reshape(tuple(int(item) for item in shape))
    return None


def _mapping_value(source: Mapping[str, Any], key: str, default: Any | None = None) -> Mapping[str, Any]:
    value = source.get(key, default if default is not None else {})
    if isinstance(value, Mapping):
        return value
    return {} if default is None else default
