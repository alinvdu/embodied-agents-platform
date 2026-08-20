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

import base64
from dataclasses import dataclass, field
import io
import struct
from typing import Any
from collections.abc import Mapping
import zlib

try:
    import numpy as np
except Exception:  # pragma: no cover - defensive fallback
    np = None  # type: ignore[assignment]

try:
    from PIL import Image
except Exception:  # pragma: no cover - defensive fallback
    Image = None  # type: ignore[assignment]

from .llm import AgentLLMRouter, LLMCallTrace
from .models import Subgoal, WorldState
from .perception import build_scene_snapshot
from .prompts import (
    build_visual_difference_system_prompt,
    build_visual_difference_user_prompt,
    build_visual_scene_summary_system_prompt,
    build_visual_scene_summary_user_prompt,
)


@dataclass(frozen=True)
class VisualObservation:
    summary: str
    reasoning_summary: str
    task_completed: bool
    change_detected: bool
    task_relevant_attributes: tuple[str, ...] = tuple()
    delta: tuple[str, ...] = tuple()
    scene: dict[str, Any] = field(default_factory=dict)


class VisualDifferencingModule:
    def __init__(self, llm_router: AgentLLMRouter | None = None) -> None:
        self.llm_router = llm_router

    def describe_initial_scene(
        self,
        *,
        instruction: str,
        world_state: WorldState,
        target: str | None = None,
    ) -> tuple[VisualObservation, LLMCallTrace | None]:
        scene = _scene_from_world_state(world_state, target=target)
        fallback = _mock_initial_observation(instruction, world_state, scene, target=target)
        if self.llm_router is None:
            return fallback, None
        config = self.llm_router.model_suite.visual_summary or self.llm_router.model_suite.planner
        if config.provider == "mock":
            return fallback, None
        multimodal_messages = _build_initial_scene_messages(
            instruction=instruction,
            world_state=world_state,
            scene=scene,
            target=target,
        )
        if multimodal_messages is not None:
            parsed, trace = self.llm_router.complete_json_messages(config=config, messages=multimodal_messages)
        else:
            parsed, trace = self.llm_router.complete_json_prompt(
                config=config,
                system_prompt=build_visual_scene_summary_system_prompt(),
                user_prompt=build_visual_scene_summary_user_prompt(
                    instruction=instruction,
                    world_state=world_state,
                    scene=scene,
                    target=target,
                ),
            )
        if parsed is None:
            return fallback, trace
        return _observation_from_llm(parsed, scene, fallback), trace

    def describe_scene_difference(
        self,
        *,
        instruction: str,
        subgoal: Subgoal,
        previous_world_state: WorldState,
        current_world_state: WorldState,
        action: dict[str, Any],
    ) -> tuple[VisualObservation, LLMCallTrace | None]:
        previous_scene = _scene_from_world_state(previous_world_state, target=subgoal.target)
        current_scene = _scene_from_world_state(current_world_state, target=subgoal.target)
        fallback = _mock_difference_observation(
            instruction,
            subgoal,
            previous_world_state,
            current_world_state,
            previous_scene,
            current_scene,
            action=action,
        )
        if self.llm_router is None:
            return fallback, None
        config = self.llm_router.model_suite.visual_diff or self.llm_router.model_suite.critic
        if config.provider == "mock":
            return fallback, None
        multimodal_messages = _build_visual_difference_messages(
            instruction=instruction,
            subgoal=subgoal,
            previous_world_state=previous_world_state,
            current_world_state=current_world_state,
            action=action,
        )
        if multimodal_messages is not None:
            parsed, trace = self.llm_router.complete_json_messages(config=config, messages=multimodal_messages)
        else:
            parsed, trace = self.llm_router.complete_json_prompt(
                config=config,
                system_prompt=build_visual_difference_system_prompt(),
                user_prompt=build_visual_difference_user_prompt(
                    instruction=instruction,
                    subgoal=subgoal,
                    previous_world_state=previous_world_state,
                    current_world_state=current_world_state,
                    previous_scene=previous_scene,
                    current_scene=current_scene,
                    action=action,
                ),
            )
        if parsed is None:
            return fallback, trace
        return _observation_from_llm(parsed, current_scene, fallback), trace


def _scene_from_world_state(world_state: WorldState, *, target: str | None = None) -> dict[str, Any]:
    return build_scene_snapshot(
        current_pose=world_state.current_pose,
        visible_objects=sorted(world_state.visible_objects),
        image_descriptions=world_state.image_descriptions,
        metadata=world_state.metadata,
        target=target,
    )


def _build_initial_scene_messages(
    *,
    instruction: str,
    world_state: WorldState,
    scene: dict[str, Any],
    target: str | None,
) -> list[dict[str, Any]] | None:
    main_image = _extract_image_data_url(world_state.metadata, prefer_wrist=False)
    wrist_image = _extract_image_data_url(world_state.metadata, prefer_wrist=True)
    if main_image is None:
        return None
    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": f"Task instruction: {instruction}"},
        {
            "type": "text",
            "text": (
                "Describe the initial state of the environment with the task goal in mind. "
                "Use the images as the primary evidence source. Return JSON only with keys "
                "summary, reasoning_summary, task_completed, change_detected, "
                "task_relevant_attributes, delta."
            ),
        },
        {"type": "text", "text": f"Current pose: {world_state.current_pose}"},
        {"type": "text", "text": f"Target: {target or 'none'}"},
        {"type": "text", "text": f"Structured scene hint: {scene.get('scene_summary', '')}"},
        {"type": "text", "text": "Main camera view:"},
        {"type": "image_url", "image_url": {"url": main_image}},
    ]
    if wrist_image is not None:
        user_content.extend(
            [
                {"type": "text", "text": "Wrist camera view:"},
                {"type": "image_url", "image_url": {"url": wrist_image}},
            ]
        )
    return [
        {"role": "system", "content": build_visual_scene_summary_system_prompt()},
        {"role": "user", "content": user_content},
    ]


def _build_visual_difference_messages(
    *,
    instruction: str,
    subgoal: Subgoal,
    previous_world_state: WorldState,
    current_world_state: WorldState,
    action: dict[str, Any],
) -> list[dict[str, Any]] | None:
    previous_main = _extract_image_data_url(previous_world_state.metadata, prefer_wrist=False)
    current_main = _extract_image_data_url(current_world_state.metadata, prefer_wrist=False)
    if previous_main is None or current_main is None:
        return None
    previous_wrist = _extract_image_data_url(previous_world_state.metadata, prefer_wrist=True)
    current_wrist = _extract_image_data_url(current_world_state.metadata, prefer_wrist=True)
    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": f"Task instruction: {instruction}"},
        {
            "type": "text",
            "text": (
                "Compare the previous and current observations after one robot action. "
                "Use the images as the primary evidence source. Return JSON only with keys "
                "summary, reasoning_summary, task_completed, change_detected, "
                "task_relevant_attributes, delta."
            ),
        },
        {"type": "text", "text": f"Subgoal: {subgoal.text} ({subgoal.kind})"},
        {"type": "text", "text": f"Target: {subgoal.target or 'none'}"},
        {"type": "text", "text": f"Action: {action.get('action_id', 'unknown_action')}"},
        {"type": "text", "text": "Previous state main camera:"},
        {"type": "image_url", "image_url": {"url": previous_main}},
        {"type": "text", "text": "Current state main camera:"},
        {"type": "image_url", "image_url": {"url": current_main}},
    ]
    if previous_wrist is not None and current_wrist is not None:
        user_content.extend(
            [
                {"type": "text", "text": "Previous state wrist camera:"},
                {"type": "image_url", "image_url": {"url": previous_wrist}},
                {"type": "text", "text": "Current state wrist camera:"},
                {"type": "image_url", "image_url": {"url": current_wrist}},
            ]
        )
    return [
        {"role": "system", "content": build_visual_difference_system_prompt()},
        {"role": "user", "content": user_content},
    ]


def _extract_image_data_url(metadata: dict[str, Any], *, prefer_wrist: bool) -> str | None:
    sensors = metadata.get("sensors", {})
    if not isinstance(sensors, Mapping):
        return None
    candidates: list[tuple[int, str, Any]] = []
    for key, payload in sensors.items():
        score = _sensor_priority(str(key), prefer_wrist=prefer_wrist)
        if score is not None:
            candidates.append((score, str(key), payload))
    for _score, _key, payload in sorted(candidates, key=lambda item: item[0]):
        data_url = _image_data_url_from_payload(payload)
        if data_url is not None:
            return data_url
    return None


def _sensor_priority(key: str, *, prefer_wrist: bool) -> int | None:
    key_lower = key.lower()
    if not any(token in key_lower for token in ("rgb", "image", "camera")):
        return None
    is_wrist = any(token in key_lower for token in ("wrist", "hand", "gripper"))
    if prefer_wrist and not is_wrist:
        return None
    if not prefer_wrist and is_wrist:
        return None
    priorities = [
        "orbbec_rgb",
        "rgb",
        "front_rgb",
        "main_rgb",
        "camera_rgb",
        "robot0_robotview",
        "agentview",
    ]
    for index, name in enumerate(priorities):
        if name in key_lower:
            return index
    return len(priorities)


def _image_data_url_from_payload(payload: Any) -> str | None:
    if payload is None:
        return None
    if isinstance(payload, str):
        return payload if payload.startswith("data:image/") else None
    if isinstance(payload, Mapping):
        for key in ("image_url", "data_url", "url"):
            value = payload.get(key)
            if isinstance(value, str) and value.startswith("data:image/"):
                return value
        for key in ("rgb", "image", "frame"):
            if key in payload:
                nested = _image_data_url_from_payload(payload.get(key))
                if nested is not None:
                    return nested
        images = payload.get("images")
        if isinstance(images, Mapping) and "rgb" in images:
            nested = _image_data_url_from_payload(images.get("rgb"))
            if nested is not None:
                return nested
        encoded = _array_payload_to_png_data_url(payload)
        if encoded is not None:
            return encoded
    if np is not None:
        try:
            array = np.asarray(payload)
        except Exception:
            return None
        return _numpy_image_to_data_url(array)
    return _list_image_to_png_data_url(payload)


def _array_payload_to_png_data_url(payload: Mapping[str, Any]) -> str | None:
    if np is not None and "data_base64" in payload and "shape" in payload and "dtype" in payload:
        try:
            raw = base64.b64decode(str(payload["data_base64"]))
            array = np.frombuffer(raw, dtype=np.dtype(str(payload["dtype"])))
            shape = tuple(int(item) for item in payload["shape"])
            array = array.reshape(shape)
            return _numpy_image_to_data_url(array)
        except Exception:
            return None
    if "data" in payload:
        if np is None:
            return _list_image_to_png_data_url(payload["data"])
        try:
            array = np.asarray(payload["data"])
        except Exception:
            return _list_image_to_png_data_url(payload["data"])
        return _numpy_image_to_data_url(array)
    return None


def _numpy_image_to_data_url(array: Any) -> str | None:
    if np is None or Image is None:
        return None
    try:
        image = np.asarray(array)
    except Exception:
        return None
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        return None
    image = image.astype(np.uint8) if image.dtype != np.uint8 else image
    if image.shape[-1] == 4:
        pil_image = Image.fromarray(image, mode="RGBA")
    else:
        pil_image = Image.fromarray(image[:, :, :3], mode="RGB")
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _list_image_to_png_data_url(payload: Any) -> str | None:
    if not isinstance(payload, list) or not payload:
        return None
    rows: list[bytes] = []
    width: int | None = None
    for row in payload:
        if not isinstance(row, list) or not row:
            return None
        pixels: list[int] = []
        if row and isinstance(row[0], list):
            for pixel in row:
                if not isinstance(pixel, list) or len(pixel) < 3:
                    return None
                pixels.extend(_clamp_channel(channel) for channel in pixel[:3])
            row_width = len(row)
        else:
            for value in row:
                channel = _clamp_channel(value)
                pixels.extend((channel, channel, channel))
            row_width = len(row)
        if width is None:
            width = row_width
        elif width != row_width:
            return None
        rows.append(b"\x00" + bytes(pixels))
    if width is None:
        return None
    png = _png_bytes(width, len(rows), b"".join(rows))
    encoded = base64.b64encode(png).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _png_bytes(width: int, height: int, filtered_rows: bytes) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(filtered_rows)) + chunk(b"IEND", b"")


def _clamp_channel(value: Any) -> int:
    try:
        number = int(value)
    except Exception:
        return 0
    return max(0, min(255, number))


def _mock_initial_observation(
    instruction: str,
    world_state: WorldState,
    scene: dict[str, Any],
    *,
    target: str | None,
) -> VisualObservation:
    relevant = _task_relevant_attributes(instruction, world_state, scene, target=target)
    completion = _infer_task_completion(instruction, None, world_state, scene)
    labels = _scene_labels(scene)
    summary = (
        f"Initial visual context at `{world_state.current_pose}`: "
        f"{scene.get('scene_summary', 'no scene summary available')}"
    )
    if relevant:
        summary += f" Task-relevant cues: {'; '.join(relevant[:3])}."
    elif labels:
        summary += f" Visible labels: {', '.join(labels[:4])}."
    return VisualObservation(
        summary=summary,
        reasoning_summary="Initial scene summary prepared for planning.",
        task_completed=completion,
        change_detected=True,
        task_relevant_attributes=tuple(relevant),
        delta=tuple(),
        scene=scene,
    )


def _mock_difference_observation(
    instruction: str,
    subgoal: Subgoal,
    previous_world_state: WorldState,
    current_world_state: WorldState,
    previous_scene: dict[str, Any],
    current_scene: dict[str, Any],
    *,
    action: dict[str, Any],
) -> VisualObservation:
    deltas: list[str] = []
    previous_labels = set(_scene_labels(previous_scene))
    current_labels = set(_scene_labels(current_scene))
    added_labels = sorted(current_labels - previous_labels)
    removed_labels = sorted(previous_labels - current_labels)

    if previous_world_state.current_pose != current_world_state.current_pose:
        deltas.append(
            f"pose moved from `{previous_world_state.current_pose}` to `{current_world_state.current_pose}`"
        )
    if added_labels:
        deltas.append(f"new visible objects: {', '.join(added_labels[:4])}")
    if removed_labels:
        deltas.append(f"objects no longer visible: {', '.join(removed_labels[:4])}")

    new_observations = sorted(current_world_state.available_observations - previous_world_state.available_observations)
    if new_observations:
        deltas.append(f"new observations: {', '.join(new_observations[:4])}")

    new_preconditions = sorted(
        current_world_state.satisfied_preconditions - previous_world_state.satisfied_preconditions
    )
    if new_preconditions:
        deltas.append(f"new task evidence: {', '.join(new_preconditions[:4])}")

    relevant = _task_relevant_attributes(
        instruction,
        current_world_state,
        current_scene,
        target=subgoal.target,
    )
    completion = _infer_task_completion(instruction, subgoal, current_world_state, current_scene)
    if completion and not any("task evidence" in item for item in deltas):
        deltas.append("task now appears visually complete")

    if not deltas:
        deltas.append("no task-relevant visual change detected")

    action_id = action.get("action_id", "unknown_action")
    summary = f"Visual delta after `{action_id}`: {'; '.join(deltas)}."
    reasoning = (
        "The current scene suggests the subgoal is complete."
        if completion
        else "The current scene does not yet prove the subgoal is complete."
    )
    return VisualObservation(
        summary=summary,
        reasoning_summary=reasoning,
        task_completed=completion,
        change_detected=deltas != ["no task-relevant visual change detected"],
        task_relevant_attributes=tuple(relevant),
        delta=tuple(deltas),
        scene=current_scene,
    )


def _task_relevant_attributes(
    instruction: str,
    world_state: WorldState,
    scene: dict[str, Any],
    *,
    target: str | None,
) -> list[str]:
    query_terms = _query_terms(instruction, target=target)
    attributes: list[str] = []
    for annotation in scene.get("annotations", []):
        if not isinstance(annotation, dict):
            continue
        label = str(annotation.get("label", "")).strip()
        if not label:
            continue
        label_lower = label.lower()
        if query_terms and not any(term in label_lower for term in query_terms):
            continue
        depth = annotation.get("depth_m")
        if depth is not None:
            attributes.append(f"{label} at {float(depth):.2f}m")
        else:
            attributes.append(f"{label} visible")
    if "fridge_open" in world_state.available_observations or "fridge_open" in world_state.satisfied_preconditions:
        attributes.append("fridge appears open")
    return attributes[:6]


def _infer_task_completion(
    instruction: str,
    subgoal: Subgoal | None,
    world_state: WorldState,
    scene: dict[str, Any],
) -> bool:
    task_text = " ".join(
        part for part in (instruction, subgoal.text if subgoal is not None else "", subgoal.target if subgoal else "") if part
    ).lower()
    labels = set(_scene_labels(scene))
    if subgoal is not None and subgoal.kind == "navigate" and subgoal.target:
        return world_state.current_pose == subgoal.target
    if subgoal is not None and subgoal.kind == "search" and subgoal.target:
        return any(subgoal.target.lower() in label for label in labels)
    if "open" in task_text and "fridge" in task_text:
        return (
            "fridge_open" in world_state.available_observations
            or "fridge_open" in world_state.satisfied_preconditions
        )
    if subgoal is not None and subgoal.kind in {"align", "manipulate"} and subgoal.target:
        return any(subgoal.target.lower() in label for label in labels)
    return False


def _scene_labels(scene: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for annotation in scene.get("annotations", []):
        if isinstance(annotation, dict) and annotation.get("label"):
            labels.append(str(annotation["label"]).lower())
    return sorted(set(labels))


def _query_terms(instruction: str, *, target: str | None) -> set[str]:
    text = " ".join(part for part in (instruction, target) if part)
    tokens = {token.strip().lower() for token in text.replace("_", " ").split()}
    stop_words = {"the", "a", "an", "to", "and", "of", "near", "at", "for", "with", "go", "open", "find"}
    return {token for token in tokens if token and token not in stop_words}


def _observation_from_llm(
    parsed: dict[str, Any],
    scene: dict[str, Any],
    fallback: VisualObservation,
) -> VisualObservation:
    attributes = parsed.get("task_relevant_attributes", fallback.task_relevant_attributes)
    if not isinstance(attributes, list):
        attributes = list(fallback.task_relevant_attributes)
    delta = parsed.get("delta", fallback.delta)
    if not isinstance(delta, list):
        delta = list(fallback.delta)
    return VisualObservation(
        summary=str(parsed.get("summary", fallback.summary)),
        reasoning_summary=str(parsed.get("reasoning_summary", fallback.reasoning_summary)),
        task_completed=bool(parsed.get("task_completed", fallback.task_completed)),
        change_detected=bool(parsed.get("change_detected", fallback.change_detected)),
        task_relevant_attributes=tuple(str(item) for item in attributes if str(item).strip()),
        delta=tuple(str(item) for item in delta if str(item).strip()),
        scene=scene,
    )
