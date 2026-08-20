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
import math
from typing import Any

from xlerobot_agent.exploration import Pose2D
from xlerobot_playground.semantic_evidence import (
    NamedPlace,
    SemanticAnchorCandidate,
    SemanticEvidence,
    SemanticEvidenceCluster,
    deterministic_semantic_id,
)


LABEL_ALIASES = {
    "living room": "living_room",
    "livingroom": "living_room",
    "lounge": "living_room",
    "dining": "dining_area",
    "eating_area": "dining_area",
    "office_area": "desk_area",
    "office": "desk_area",
    "desk": "desk_area",
    "bathroom_entrance": "bathroom_entry",
}


@dataclass
class SemanticMemory:
    evidence: dict[str, SemanticEvidence] = field(default_factory=dict)
    clusters: dict[str, SemanticEvidenceCluster] = field(default_factory=dict)
    anchors: dict[str, SemanticAnchorCandidate] = field(default_factory=dict)
    named_places: dict[str, NamedPlace] = field(default_factory=dict)
    rejected_records: list[dict[str, Any]] = field(default_factory=list)
    merged_records: list[dict[str, Any]] = field(default_factory=list)

    def add_evidence(self, item: SemanticEvidence, *, cluster_radius_m: float = 2.0) -> SemanticEvidenceCluster:
        self.evidence[item.evidence_id] = item
        label = normalize_label(item.label_hint)
        for cluster in list(self.clusters.values()):
            if normalize_label(cluster.label_hint) != label:
                continue
            if _pose_distance(cluster.evidence_pose, item.evidence_pose) > cluster_radius_m:
                continue
            merged = _merge_cluster(cluster, item)
            self.clusters[cluster.cluster_id] = merged
            self.merged_records.append(
                {"type": "evidence_cluster_merge", "cluster_id": cluster.cluster_id, "evidence_id": item.evidence_id}
            )
            return merged
        cluster_id = deterministic_semantic_id("sem_cluster", len(self.clusters) + 1)
        cluster = SemanticEvidenceCluster(
            cluster_id=cluster_id,
            label_hint=label,
            evidence_ids=(item.evidence_id,),
            evidence_pose=item.evidence_pose,
            source_frame_ids=item.source_frame_ids,
            confidence=item.confidence,
            evidence=item.evidence,
        )
        self.clusters[cluster_id] = cluster
        return cluster

    def add_anchor(self, item: SemanticAnchorCandidate, *, merge_radius_m: float = 2.0) -> NamedPlace | None:
        self.anchors[item.anchor_id] = item
        if item.reachability_status != "reachable":
            self.rejected_records.append(
                {"type": "unreachable_anchor", "anchor_id": item.anchor_id, "label_hint": item.label_hint}
            )
            return None
        label = normalize_label(item.label_hint)
        for place in list(self.named_places.values()):
            if normalize_label(place.label) != label:
                continue
            if _pose_distance(place.anchor_pose, item.anchor_pose) > merge_radius_m:
                continue
            updated = _merge_place(place, item)
            self.named_places[place.place_id] = updated
            self.merged_records.append(
                {"type": "named_place_update", "place_id": place.place_id, "anchor_id": item.anchor_id}
            )
            return updated
        place_id = _next_place_id(label, self.named_places)
        place = NamedPlace(
            place_id=place_id,
            label=label,
            anchor_pose=item.anchor_pose,
            evidence_pose=item.evidence_pose,
            source_anchor_ids=(item.anchor_id,),
            source_evidence_ids=item.source_evidence_ids,
            source_frame_ids=item.source_frame_ids,
            confidence=item.confidence,
            evidence=item.evidence,
            notes=("created from reachable RGB-D grounded semantic anchor",),
        )
        self.named_places[place_id] = place
        return place

    def snapshot(self) -> dict[str, Any]:
        return {
            "evidence": [item.to_dict() for item in self.evidence.values()],
            "clusters": [item.to_dict() for item in self.clusters.values()],
            "anchors": [item.to_dict() for item in self.anchors.values()],
            "named_places": [item.to_dict() for item in self.named_places.values()],
            "rejected_records": list(self.rejected_records),
            "merged_records": list(self.merged_records),
        }


def normalize_label(label: str) -> str:
    normalized = str(label).strip().lower().replace("-", "_").replace(" ", "_")
    normalized = "_".join(part for part in normalized.split("_") if part)
    return LABEL_ALIASES.get(normalized, normalized)


def _merge_cluster(cluster: SemanticEvidenceCluster, item: SemanticEvidence) -> SemanticEvidenceCluster:
    count = max(len(cluster.evidence_ids), 1)
    x = (cluster.evidence_pose.x * count + item.evidence_pose.x) / (count + 1)
    y = (cluster.evidence_pose.y * count + item.evidence_pose.y) / (count + 1)
    return SemanticEvidenceCluster(
        cluster_id=cluster.cluster_id,
        label_hint=cluster.label_hint,
        evidence_ids=_unique((*cluster.evidence_ids, item.evidence_id)),
        evidence_pose=Pose2D(x, y, 0.0),
        source_frame_ids=_unique((*cluster.source_frame_ids, *item.source_frame_ids)),
        confidence=max(cluster.confidence, item.confidence),
        evidence=_unique((*cluster.evidence, *item.evidence)),
        status=cluster.status,
    )


def _merge_place(place: NamedPlace, anchor: SemanticAnchorCandidate) -> NamedPlace:
    keep_new_anchor = anchor.confidence > place.confidence
    return NamedPlace(
        place_id=place.place_id,
        label=place.label,
        anchor_pose=anchor.anchor_pose if keep_new_anchor else place.anchor_pose,
        evidence_pose=anchor.evidence_pose if keep_new_anchor else place.evidence_pose,
        source_anchor_ids=_unique((*place.source_anchor_ids, anchor.anchor_id)),
        source_evidence_ids=_unique((*place.source_evidence_ids, *anchor.source_evidence_ids)),
        source_frame_ids=_unique((*place.source_frame_ids, *anchor.source_frame_ids)),
        confidence=max(place.confidence, anchor.confidence),
        evidence=_unique((*place.evidence, *anchor.evidence)),
        notes=_unique((*place.notes, "updated with additional semantic anchor evidence")),
        status=place.status,
    )


def _next_place_id(label: str, places: dict[str, NamedPlace]) -> str:
    prefix = f"place_{label}_"
    count = sum(1 for place_id in places if place_id.startswith(prefix)) + 1
    return f"{prefix}{count:03d}"


def _pose_distance(a: Pose2D, b: Pose2D) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _unique(items: tuple[str, ...]) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()
    for item in items:
        cleaned = str(item).strip()
        if cleaned and cleaned not in seen:
            output.append(cleaned)
            seen.add(cleaned)
    return tuple(output)
