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

import json
from typing import Any


def build_semantic_evidence_extraction_system_prompt() -> str:
    return (
        "You identify visually grounded household semantic cues for a robot. "
        "Return JSON only with keys `frame_id` and `semantic_observations`. "
        "Return pixel regions, bounding boxes, representative pixels, image positions, visual cues, confidence, "
        "and concise evidence. Do not return map coordinates, waypoint coordinates, anchor poses, x/y/yaw, "
        "or navigation goals. Coordinates are produced only later by deterministic RGB-D projection. "
        "Use provisional labels when evidence is weak. Do not label empty open floor as a named place unless "
        "supporting objects are visible. Do not create observations from frontier markers."
    )


def build_semantic_evidence_extraction_user_prompt(payload: dict[str, Any]) -> str:
    redacted = _redact_images(payload)
    return "\n".join(
        [
            "Semantic Evidence Extraction Payload:",
            json.dumps(redacted, indent=2, sort_keys=True),
            "",
            "Required observation object fields:",
            json.dumps(
                {
                    "label_hint": "kitchen | dining_area | living_room | desk_area | bedroom | bathroom_entry | unknown_semantic_area",
                    "confidence": "0.0 to 1.0",
                    "visual_cues": ["counter", "cabinet"],
                    "pixel_regions": [
                        {
                            "description": "visible object or region that justifies the label",
                            "image_position": "left_center",
                            "bbox_xyxy": [20, 180, 260, 360],
                            "representative_point_uv": [140, 270],
                            "depth_hint": "near | mid | far when exact depth is unavailable",
                        }
                    ],
                    "reasoning_summary": "brief visual evidence only",
                },
                indent=2,
            ),
        ]
    )


def build_semantic_consolidation_system_prompt() -> str:
    return (
        "You consolidate RGB-D grounded semantic evidence into named robot destinations. "
        "Return JSON only with key `place_updates`. Valid actions are create, update, merge, reject, and keep. "
        "Named places are navigation destinations with one best reachable anchor pose supplied by deterministic code; "
        "do not invent map coordinates. Merge duplicate evidence aggressively when it describes the same place. "
        "Keep multiple instances only with strong spatial separation or label distinction. Prefer stable labels with "
        "source images, evidence clusters, and reachable anchors over speculative labels."
    )


def build_semantic_consolidation_user_prompt(payload: dict[str, Any]) -> str:
    redacted = _redact_images(payload)
    return "\n".join(
        [
            "Semantic Consolidation Payload:",
            json.dumps(redacted, indent=2, sort_keys=True),
            "",
            "Output contract:",
            json.dumps(
                {
                    "place_updates": [
                        {
                            "action": "create | update | merge | reject | keep",
                            "target_place_id": None,
                            "label": "kitchen",
                            "source_anchor_id": "sem_anchor_000038",
                            "confidence": 0.82,
                            "evidence": ["counter and cabinets observed"],
                            "notes": "use the reachable anchor supplied in the payload",
                        }
                    ]
                },
                indent=2,
            ),
        ]
    )


def _redact_images(value: Any) -> Any:
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(item, str) and item.startswith("data:image/"):
                prefix, _, encoded = item.partition(",")
                output[key] = {
                    "attached_as_multimodal_image": True,
                    "mime_type": prefix.split(";")[0].removeprefix("data:"),
                    "base64_bytes": len(encoded),
                }
                continue
            output[key] = _redact_images(item)
        return output
    if isinstance(value, list):
        return [_redact_images(item) for item in value]
    return value
