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

from dataclasses import dataclass
import math
from typing import Any, Sequence

from xlerobot_agent.exploration import Pose2D
from xlerobot_playground.semantic_evidence import PixelRegion


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


def project_pixel_region_to_map(
    *,
    pixel_region: PixelRegion,
    depth_image: Sequence[Sequence[float | int | None]],
    intrinsics: CameraIntrinsics,
    camera_pose: Pose2D,
    fallback_depth_m: float | None = None,
) -> Pose2D | None:
    depth_m = pixel_region.depth_m
    if depth_m is None or not math.isfinite(depth_m) or depth_m <= 0:
        depth_m = median_valid_depth_near(depth_image, pixel_region.center_uv)
    if depth_m is None and fallback_depth_m is not None:
        depth_m = fallback_depth_m
    if depth_m is None or depth_m <= 0:
        return None
    return project_pixel_to_map(
        u=pixel_region.center_uv[0],
        v=pixel_region.center_uv[1],
        depth_m=depth_m,
        intrinsics=intrinsics,
        camera_pose=camera_pose,
    )


def project_pixel_to_map(
    *,
    u: int,
    v: int,
    depth_m: float,
    intrinsics: CameraIntrinsics,
    camera_pose: Pose2D,
) -> Pose2D | None:
    if depth_m <= 0 or not math.isfinite(depth_m):
        return None
    if intrinsics.fx <= 0 or intrinsics.fy <= 0:
        return None
    camera_x = (float(u) - intrinsics.cx) * depth_m / intrinsics.fx
    camera_z = depth_m
    world_dx = camera_z * math.cos(camera_pose.yaw) - camera_x * math.sin(camera_pose.yaw)
    world_dy = camera_z * math.sin(camera_pose.yaw) + camera_x * math.cos(camera_pose.yaw)
    return Pose2D(camera_pose.x + world_dx, camera_pose.y + world_dy, 0.0)


def median_valid_depth_near(
    depth_image: Sequence[Sequence[float | int | None]],
    center_uv: tuple[int, int],
    *,
    radius_px: int = 2,
) -> float | None:
    u, v = center_uv
    values: list[float] = []
    for row_index in range(max(0, v - radius_px), min(len(depth_image), v + radius_px + 1)):
        row = depth_image[row_index]
        for col_index in range(max(0, u - radius_px), min(len(row), u + radius_px + 1)):
            value = row[col_index]
            if value is None:
                continue
            parsed = float(value)
            if math.isfinite(parsed) and parsed > 0:
                values.append(parsed)
    if not values:
        return None
    values.sort()
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2.0


def representative_pixel_for_image_position(image_position: str, intrinsics: CameraIntrinsics) -> tuple[int, int]:
    normalized = image_position.lower().strip()
    horizontal = 0.5
    vertical = 0.5
    if "left" in normalized:
        horizontal = 0.25
    elif "right" in normalized:
        horizontal = 0.75
    if "upper" in normalized or "top" in normalized:
        vertical = 0.25
    elif "lower" in normalized or "bottom" in normalized:
        vertical = 0.75
    return (
        min(max(int(round(intrinsics.width * horizontal)), 0), max(intrinsics.width - 1, 0)),
        min(max(int(round(intrinsics.height * vertical)), 0), max(intrinsics.height - 1, 0)),
    )


def depth_hint_to_meters(depth_hint: Any) -> float | None:
    normalized = str(depth_hint or "").strip().lower()
    if normalized in {"near", "close"}:
        return 1.0
    if normalized in {"mid", "middle", "medium"}:
        return 2.0
    if normalized in {"far", "distant"}:
        return 3.5
    try:
        value = float(normalized)
    except ValueError:
        return None
    return value if value > 0 else None
