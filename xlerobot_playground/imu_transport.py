from __future__ import annotations

import json
import time
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit


def parse_imu_json(data: bytes | str) -> dict[str, Any] | None:
    if isinstance(data, bytes):
        payload = json.loads(data.decode("utf-8"))
    else:
        payload = json.loads(data)
    return normalize_imu_payload(payload)


def normalize_imu_payload(payload: Any) -> dict[str, Any] | None:
    imu = payload.get("imu") if isinstance(payload, dict) else None
    if isinstance(imu, dict):
        payload = imu
    if not isinstance(payload, dict):
        raise ValueError("Expected IMU metadata to be a JSON object.")
    angular = _extract_xyz(payload, "angular_velocity_rad_s", aliases=("gyro", "angular_velocity"))
    linear = _extract_xyz(payload, "linear_acceleration_m_s2", aliases=("accel", "linear_acceleration"))
    if angular is None and linear is None:
        return None
    sample: dict[str, Any] = {
        "timestamp_s": _extract_timestamp_s(payload),
        "angular_velocity_rad_s": angular or {"x": 0.0, "y": 0.0, "z": 0.0},
        "linear_acceleration_m_s2": linear or {"x": 0.0, "y": 0.0, "z": 0.0},
    }
    orientation = _extract_xyzw(payload, "orientation_xyzw", aliases=("orientation",))
    if orientation is not None:
        sample["orientation_xyzw"] = orientation
    for key in (
        "has_accel",
        "has_gyro",
        "accel_frame_index",
        "gyro_frame_index",
        "accel_timestamp_us",
        "gyro_timestamp_us",
        "accel_temperature_c",
        "gyro_temperature_c",
        "device_timestamp_us",
        "system_timestamp_us",
        "timestamp_us",
    ):
        if key in payload:
            sample[key] = payload[key]
    return sample


def build_websocket_url(base_url: str, path: str) -> str:
    http_url = urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))
    parts = urlsplit(http_url)
    if parts.scheme == "http":
        scheme = "ws"
    elif parts.scheme == "https":
        scheme = "wss"
    elif parts.scheme in {"ws", "wss"}:
        scheme = parts.scheme
    else:
        raise ValueError(f"Unsupported websocket URL scheme: {parts.scheme}")
    return urlunsplit((scheme, parts.netloc, parts.path, parts.query, parts.fragment))


def _extract_timestamp_s(payload: dict[str, Any]) -> float:
    for key in ("timestamp_s", "time_s"):
        if key in payload:
            return float(payload[key])
    for key in ("system_timestamp_us", "device_timestamp_us", "timestamp_us"):
        if key in payload:
            return float(payload[key]) / 1_000_000.0
    return time.time()


def _extract_xyz(payload: dict[str, Any], key: str, *, aliases: tuple[str, ...] = ()) -> dict[str, float] | None:
    for candidate in (key, *aliases):
        value = payload.get(candidate)
        if isinstance(value, dict) and all(axis in value for axis in ("x", "y", "z")):
            return {axis: float(value[axis]) for axis in ("x", "y", "z")}
    flat = {}
    found = False
    for axis in ("x", "y", "z"):
        for prefix in (key, *aliases):
            flat_key = f"{prefix}_{axis}"
            if flat_key in payload:
                flat[axis] = float(payload[flat_key])
                found = True
                break
    if found and len(flat) == 3:
        return flat
    return None


def _extract_xyzw(payload: dict[str, Any], key: str, *, aliases: tuple[str, ...] = ()) -> dict[str, float] | None:
    for candidate in (key, *aliases):
        value = payload.get(candidate)
        if isinstance(value, dict) and all(axis in value for axis in ("x", "y", "z", "w")):
            return {axis: float(value[axis]) for axis in ("x", "y", "z", "w")}
    return None
