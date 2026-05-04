from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json
import struct
import time


RGBD_MAGIC = b"XLRGBD1\0"
RGBD_VERSION = 3
POINT_CLOUD_FORMAT_NONE = 0
POINT_CLOUD_FORMAT_XYZ_FLOAT32 = 1
RGBD_HEADER = struct.Struct("!8sIQQIIIIQQ")
RGBD_HEADER_V2 = struct.Struct("!8sIQQIIIIQQQ")
RGBD_HEADER_V3 = struct.Struct("!8sIQQIIIIQQIIIQQ")


@dataclass(frozen=True)
class PackedRgbdFrame:
    frame_index: int
    timestamp_us: int
    rgb: bytes
    rgb_width: int
    rgb_height: int
    depth_be: bytes | None
    depth_width: int | None
    depth_height: int | None
    metadata: dict[str, Any] | None = None
    point_cloud_format: int = POINT_CLOUD_FORMAT_NONE
    point_cloud_points: bytes | None = None
    point_cloud_count: int = 0
    point_cloud_stride: int = 0
    point_cloud_units: str | None = None

    @property
    def timestamp_s(self) -> float:
        return float(self.timestamp_us) / 1_000_000.0


def pack_rgbd_frame(
    *,
    frame_index: int,
    timestamp_us: int | None,
    rgb: bytes,
    rgb_width: int,
    rgb_height: int,
    depth_be: bytes | None = None,
    depth_width: int | None = None,
    depth_height: int | None = None,
    point_cloud_format: int = POINT_CLOUD_FORMAT_NONE,
    point_cloud_points: bytes | None = None,
    point_cloud_count: int = 0,
    point_cloud_stride: int = 0,
    metadata: dict[str, Any] | None = None,
) -> bytes:
    if timestamp_us is None:
        timestamp_us = int(time.time() * 1_000_000.0)
    expected_rgb = int(rgb_width) * int(rgb_height) * 3
    if len(rgb) != expected_rgb:
        raise ValueError(f"RGB payload size {len(rgb)} does not match dimensions {rgb_width}x{rgb_height}.")
    depth_payload = depth_be or b""
    if bool(depth_payload) != (depth_width is not None and depth_height is not None):
        raise ValueError("Depth payload and dimensions must be provided together.")
    if depth_payload and len(depth_payload) != int(depth_width or 0) * int(depth_height or 0) * 2:
        raise ValueError(
            f"Depth payload size {len(depth_payload)} does not match dimensions {depth_width}x{depth_height}."
        )
    point_payload = point_cloud_points or b""
    if point_payload:
        if int(point_cloud_format) != POINT_CLOUD_FORMAT_XYZ_FLOAT32:
            raise ValueError(f"Unsupported point cloud format for packing: {point_cloud_format}.")
        if int(point_cloud_stride) != 12:
            raise ValueError("XYZ float32 point clouds must use a 12 byte stride.")
        if len(point_payload) != int(point_cloud_count) * int(point_cloud_stride):
            raise ValueError(
                f"Point cloud payload size {len(point_payload)} does not match "
                f"count={point_cloud_count} stride={point_cloud_stride}."
            )
    else:
        point_cloud_format = POINT_CLOUD_FORMAT_NONE
        point_cloud_count = 0
        point_cloud_stride = 0
    metadata_payload = b""
    if metadata:
        metadata_payload = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    header = RGBD_HEADER_V3.pack(
        RGBD_MAGIC,
        RGBD_VERSION,
        int(frame_index),
        int(timestamp_us),
        int(rgb_width),
        int(rgb_height),
        int(depth_width or 0),
        int(depth_height or 0),
        len(rgb),
        len(depth_payload),
        int(point_cloud_format),
        int(point_cloud_count),
        int(point_cloud_stride),
        len(point_payload),
        len(metadata_payload),
    )
    return header + rgb + depth_payload + point_payload + metadata_payload


def unpack_rgbd_frame(data: bytes) -> PackedRgbdFrame:
    if len(data) < RGBD_HEADER.size:
        raise ValueError("RGB-D payload is shorter than the header.")
    (
        magic,
        version,
        frame_index,
        timestamp_us,
        rgb_width,
        rgb_height,
        depth_width,
        depth_height,
        rgb_size,
        depth_size,
    ) = RGBD_HEADER.unpack_from(data)
    if magic != RGBD_MAGIC:
        raise ValueError("RGB-D payload has an unsupported magic header.")
    if version not in (1, 2, RGBD_VERSION):
        raise ValueError(f"Unsupported RGB-D payload version: {version}.")
    expected_rgb_size = rgb_width * rgb_height * 3
    if rgb_size != expected_rgb_size:
        raise ValueError(f"RGB payload size {rgb_size} does not match dimensions {rgb_width}x{rgb_height}.")
    if bool(depth_size) != bool(depth_width and depth_height):
        raise ValueError("Depth payload and dimensions must be provided together.")
    if depth_size and depth_size != depth_width * depth_height * 2:
        raise ValueError(f"Depth payload size {depth_size} does not match dimensions {depth_width}x{depth_height}.")
    metadata_size = 0
    point_cloud_format = POINT_CLOUD_FORMAT_NONE
    point_cloud_count = 0
    point_cloud_stride = 0
    point_payload_size = 0
    header_size = RGBD_HEADER.size
    if version == 2:
        if len(data) < RGBD_HEADER_V2.size:
            raise ValueError("RGB-D v2 payload is shorter than the header.")
        (
            magic,
            version,
            frame_index,
            timestamp_us,
            rgb_width,
            rgb_height,
            depth_width,
            depth_height,
            rgb_size,
            depth_size,
            metadata_size,
        ) = RGBD_HEADER_V2.unpack_from(data)
        header_size = RGBD_HEADER_V2.size
    elif version == 3:
        if len(data) < RGBD_HEADER_V3.size:
            raise ValueError("RGB-D v3 payload is shorter than the header.")
        (
            magic,
            version,
            frame_index,
            timestamp_us,
            rgb_width,
            rgb_height,
            depth_width,
            depth_height,
            rgb_size,
            depth_size,
            point_cloud_format,
            point_cloud_count,
            point_cloud_stride,
            point_payload_size,
            metadata_size,
        ) = RGBD_HEADER_V3.unpack_from(data)
        header_size = RGBD_HEADER_V3.size
        if point_cloud_format not in (POINT_CLOUD_FORMAT_NONE, POINT_CLOUD_FORMAT_XYZ_FLOAT32):
            raise ValueError(f"Unsupported point cloud format: {point_cloud_format}.")
        if bool(point_payload_size) != bool(point_cloud_format and point_cloud_count and point_cloud_stride):
            raise ValueError("Point cloud payload, format, count, and stride must be provided together.")
        if point_payload_size and point_cloud_format == POINT_CLOUD_FORMAT_XYZ_FLOAT32 and point_cloud_stride != 12:
            raise ValueError("XYZ float32 point clouds must use a 12 byte stride.")
        if point_payload_size and point_payload_size != point_cloud_count * point_cloud_stride:
            raise ValueError(
                f"Point cloud payload size {point_payload_size} does not match "
                f"count={point_cloud_count} stride={point_cloud_stride}."
            )
    expected_size = header_size + rgb_size + depth_size + point_payload_size + metadata_size
    if len(data) < expected_size:
        raise ValueError(f"RGB-D payload is truncated: expected {expected_size} bytes, got {len(data)}.")
    if len(data) != expected_size:
        raise ValueError(f"RGB-D payload has trailing bytes: expected {expected_size} bytes, got {len(data)}.")
    offset = header_size
    rgb = data[offset : offset + rgb_size]
    offset += rgb_size
    depth_be = data[offset : offset + depth_size] if depth_size else None
    offset += depth_size
    point_cloud_points = data[offset : offset + point_payload_size] if point_payload_size else None
    offset += point_payload_size
    metadata = None
    if metadata_size:
        try:
            metadata = json.loads(data[offset : offset + metadata_size].decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("RGB-D metadata is not valid JSON.") from exc
    point_cloud_units = None
    if isinstance(metadata, dict):
        point_cloud_metadata = metadata.get("point_cloud")
        if isinstance(point_cloud_metadata, dict) and point_cloud_metadata.get("units") is not None:
            point_cloud_units = str(point_cloud_metadata["units"])
    return PackedRgbdFrame(
        frame_index=int(frame_index),
        timestamp_us=int(timestamp_us),
        rgb=rgb,
        rgb_width=int(rgb_width),
        rgb_height=int(rgb_height),
        depth_be=depth_be,
        depth_width=int(depth_width) if depth_size else None,
        depth_height=int(depth_height) if depth_size else None,
        metadata=metadata,
        point_cloud_format=int(point_cloud_format),
        point_cloud_points=point_cloud_points,
        point_cloud_count=int(point_cloud_count),
        point_cloud_stride=int(point_cloud_stride),
        point_cloud_units=point_cloud_units,
    )


def depth_big_endian_bytes_to_rows(data: bytes, *, width: int, height: int) -> tuple[tuple[int, ...], ...]:
    expected = width * height * 2
    if len(data) < expected:
        raise ValueError(f"Depth payload is truncated: expected {expected} bytes, got {len(data)}.")
    rows: list[tuple[int, ...]] = []
    offset = 0
    for _ in range(height):
        row: list[int] = []
        for _ in range(width):
            row.append(int.from_bytes(data[offset : offset + 2], "big"))
            offset += 2
        rows.append(tuple(row))
    return tuple(rows)


def rgb_payload_to_ppm(rgb: bytes, *, width: int, height: int) -> bytes:
    expected = width * height * 3
    if len(rgb) < expected:
        raise ValueError(f"RGB payload is truncated: expected {expected} bytes, got {len(rgb)}.")
    return b"P6\n%d %d\n255\n" % (width, height) + rgb[:expected]


def depth_payload_to_pgm(depth_be: bytes, *, width: int, height: int) -> bytes:
    expected = width * height * 2
    if len(depth_be) < expected:
        raise ValueError(f"Depth payload is truncated: expected {expected} bytes, got {len(depth_be)}.")
    return b"P5\n%d %d\n65535\n" % (width, height) + depth_be[:expected]
