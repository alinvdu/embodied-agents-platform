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

from dataclasses import dataclass, field
import json
from typing import Any, Iterable

from xlerobot_agent.exploration import Pose2D


SEMANTIC_EVIDENCE_STATUSES = frozenset({"provisional", "accepted", "rejected", "merged"})
SEMANTIC_ANCHOR_STATUSES = frozenset({"provisional", "accepted", "rejected", "merged"})
NAMED_PLACE_STATUSES = frozenset({"provisional", "confirmed", "rejected"})
REACHABILITY_STATUSES = frozenset({"reachable", "unreachable", "unknown"})


@dataclass(frozen=True)
class PixelRegion:
    frame_id: str
    bbox_xyxy: tuple[int, int, int, int]
    center_uv: tuple[int, int]
    depth_m: float | None
    image_position: str
    object_label: str | None
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "bbox_xyxy": list(self.bbox_xyxy),
            "center_uv": list(self.center_uv),
            "depth_m": self.depth_m,
            "image_position": self.image_position,
            "object_label": self.object_label,
            "description": self.description,
        }


@dataclass(frozen=True)
class SemanticObservation:
    observation_id: str
    frame_id: str
    label_hint: str
    confidence: float
    pixel_regions: tuple[PixelRegion, ...]
    visual_cues: tuple[str, ...] = tuple()
    reasoning_summary: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "frame_id": self.frame_id,
            "label_hint": self.label_hint,
            "confidence": round(self.confidence, 3),
            "visual_cues": list(self.visual_cues),
            "pixel_regions": [item.to_dict() for item in self.pixel_regions],
            "reasoning_summary": self.reasoning_summary,
            "raw": json.loads(json.dumps(self.raw)),
        }


@dataclass(frozen=True)
class SemanticEvidence:
    evidence_id: str
    label_hint: str
    evidence_pose: Pose2D
    source_frame_ids: tuple[str, ...]
    source_pixels: tuple[PixelRegion, ...]
    confidence: float
    evidence: tuple[str, ...]
    status: str = "provisional"

    def __post_init__(self) -> None:
        if self.status not in SEMANTIC_EVIDENCE_STATUSES:
            raise ValueError(f"invalid semantic evidence status: {self.status}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "label_hint": self.label_hint,
            "evidence_pose": self.evidence_pose.to_dict(),
            "source_frame_ids": list(self.source_frame_ids),
            "source_pixels": [item.to_dict() for item in self.source_pixels],
            "confidence": round(self.confidence, 3),
            "evidence": list(self.evidence),
            "status": self.status,
        }


@dataclass(frozen=True)
class SemanticEvidenceCluster:
    cluster_id: str
    label_hint: str
    evidence_ids: tuple[str, ...]
    evidence_pose: Pose2D
    source_frame_ids: tuple[str, ...]
    confidence: float
    evidence: tuple[str, ...]
    status: str = "provisional"

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "label_hint": self.label_hint,
            "evidence_ids": list(self.evidence_ids),
            "evidence_pose": self.evidence_pose.to_dict(),
            "source_frame_ids": list(self.source_frame_ids),
            "confidence": round(self.confidence, 3),
            "evidence": list(self.evidence),
            "status": self.status,
        }


@dataclass(frozen=True)
class SemanticAnchorCandidate:
    anchor_id: str
    label_hint: str
    anchor_pose: Pose2D
    evidence_pose: Pose2D
    source_evidence_ids: tuple[str, ...]
    source_frame_ids: tuple[str, ...]
    confidence: float
    reachability_status: str
    free_space_path_distance_m: float | None
    line_of_sight_score: float
    evidence: tuple[str, ...]
    status: str = "provisional"

    def __post_init__(self) -> None:
        if self.reachability_status not in REACHABILITY_STATUSES:
            raise ValueError(f"invalid reachability status: {self.reachability_status}")
        if self.status not in SEMANTIC_ANCHOR_STATUSES:
            raise ValueError(f"invalid semantic anchor status: {self.status}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_id": self.anchor_id,
            "label_hint": self.label_hint,
            "anchor_pose": self.anchor_pose.to_dict(),
            "evidence_pose": self.evidence_pose.to_dict(),
            "source_evidence_ids": list(self.source_evidence_ids),
            "source_frame_ids": list(self.source_frame_ids),
            "confidence": round(self.confidence, 3),
            "reachability_status": self.reachability_status,
            "free_space_path_distance_m": (
                None if self.free_space_path_distance_m is None else round(self.free_space_path_distance_m, 3)
            ),
            "line_of_sight_score": round(self.line_of_sight_score, 3),
            "evidence": list(self.evidence),
            "status": self.status,
        }


@dataclass(frozen=True)
class NamedPlace:
    place_id: str
    label: str
    anchor_pose: Pose2D
    evidence_pose: Pose2D | None
    source_anchor_ids: tuple[str, ...]
    source_evidence_ids: tuple[str, ...]
    source_frame_ids: tuple[str, ...]
    confidence: float
    evidence: tuple[str, ...]
    notes: tuple[str, ...]
    status: str = "provisional"

    def __post_init__(self) -> None:
        if self.status not in NAMED_PLACE_STATUSES:
            raise ValueError(f"invalid named place status: {self.status}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "place_id": self.place_id,
            "label": self.label,
            "anchor_pose": self.anchor_pose.to_dict(),
            "evidence_pose": None if self.evidence_pose is None else self.evidence_pose.to_dict(),
            "source_anchor_ids": list(self.source_anchor_ids),
            "source_evidence_ids": list(self.source_evidence_ids),
            "source_frame_ids": list(self.source_frame_ids),
            "confidence": round(self.confidence, 3),
            "evidence": list(self.evidence),
            "notes": list(self.notes),
            "status": self.status,
        }


def deterministic_semantic_id(prefix: str, index: int) -> str:
    return f"{prefix}_{max(index, 0):06d}"


def parse_semantic_observation_payload(
    payload: dict[str, Any],
    *,
    fallback_frame_id: str | None = None,
    id_start: int = 1,
) -> tuple[list[SemanticObservation], list[str]]:
    """Validate VLM semantic observations without accepting map coordinates."""

    warnings: list[str] = []
    frame_id = str(payload.get("frame_id") or fallback_frame_id or "").strip()
    raw_observations = payload.get("semantic_observations", [])
    if not frame_id:
        warnings.append("missing frame_id")
        return [], warnings
    if not isinstance(raw_observations, list):
        warnings.append("semantic_observations must be a list")
        return [], warnings

    observations: list[SemanticObservation] = []
    for offset, item in enumerate(raw_observations):
        if not isinstance(item, dict):
            warnings.append(f"observation[{offset}] is not an object")
            continue
        forbidden_coords = {"x", "y", "yaw", "map_pose", "anchor_pose", "evidence_pose"} & set(item)
        if forbidden_coords:
            warnings.append(f"observation[{offset}] contains forbidden map coordinate fields")
            continue
        label_hint = _clean_label(item.get("label_hint"))
        if not label_hint:
            warnings.append(f"observation[{offset}] missing label_hint")
            continue
        confidence = _bounded_float(item.get("confidence"), default=0.0)
        pixel_regions = _parse_pixel_regions(
            item.get("pixel_regions", []),
            frame_id=frame_id,
            warnings=warnings,
            observation_index=offset,
        )
        if not pixel_regions:
            warnings.append(f"observation[{offset}] has no valid pixel_regions")
            continue
        visual_cues = _clean_string_list(item.get("visual_cues", []), limit=8)
        observations.append(
            SemanticObservation(
                observation_id=deterministic_semantic_id("sem_obs", id_start + len(observations)),
                frame_id=frame_id,
                label_hint=label_hint,
                confidence=confidence,
                pixel_regions=tuple(pixel_regions),
                visual_cues=tuple(visual_cues),
                reasoning_summary=str(item.get("reasoning_summary", "")).strip()[:600],
                raw=json.loads(json.dumps(item)),
            )
        )
    return observations, warnings


def _parse_pixel_regions(
    value: Any,
    *,
    frame_id: str,
    warnings: list[str],
    observation_index: int,
) -> list[PixelRegion]:
    if not isinstance(value, list):
        return []
    regions: list[PixelRegion] = []
    for region_index, item in enumerate(value):
        if not isinstance(item, dict):
            warnings.append(f"observation[{observation_index}].pixel_regions[{region_index}] is not an object")
            continue
        bbox = _parse_int_tuple(item.get("bbox_xyxy"), expected=4)
        center = _parse_int_tuple(item.get("representative_point_uv", item.get("center_uv")), expected=2)
        if bbox is None or center is None:
            warnings.append(f"observation[{observation_index}].pixel_regions[{region_index}] lacks bbox or point")
            continue
        depth = item.get("depth_m")
        depth_m = None if depth is None else _bounded_float(depth, default=0.0, minimum=0.0, maximum=30.0)
        regions.append(
            PixelRegion(
                frame_id=frame_id,
                bbox_xyxy=tuple(bbox),  # type: ignore[arg-type]
                center_uv=tuple(center),  # type: ignore[arg-type]
                depth_m=depth_m,
                image_position=str(item.get("image_position", "center")).strip()[:80] or "center",
                object_label=_optional_clean_string(item.get("object_label")),
                description=str(item.get("description", "")).strip()[:400],
            )
        )
    return regions


def _parse_int_tuple(value: Any, *, expected: int) -> tuple[int, ...] | None:
    if not isinstance(value, (list, tuple)) or len(value) != expected:
        return None
    parsed: list[int] = []
    for item in value:
        try:
            parsed.append(int(item))
        except (TypeError, ValueError):
            return None
    return tuple(parsed)


def _bounded_float(value: Any, *, default: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return min(max(parsed, minimum), maximum)


def _clean_string_list(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, dict)):
        return []
    output: list[str] = []
    seen: set[str] = set()
    for item in value:
        cleaned = str(item).strip()[:160]
        if cleaned and cleaned not in seen:
            output.append(cleaned)
            seen.add(cleaned)
        if len(output) >= limit:
            break
    return output


def _clean_label(value: Any) -> str:
    cleaned = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    return "".join(ch for ch in cleaned if ch.isalnum() or ch == "_")[:80]


def _optional_clean_string(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()[:120]
    return cleaned or None
