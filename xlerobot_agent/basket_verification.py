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
from dataclasses import dataclass
import io
import json
from pathlib import Path
from typing import Any, Protocol, Sequence

from PIL import Image, ImageOps

from .llm import LLMCallTrace, ModelConfig


_SYSTEM_PROMPT = """You are a conservative visual postcondition verifier for a robot manipulation task.

The target is a small square bottle of cherry juice with dark red liquid and a white cap. The destination is the white perforated plastic basket attached to the robot.

The request contains annotated task-context photographs, successful unannotated right-wrist-camera examples, and finally one or more CURRENT RUNTIME WRIST IMAGES. Blue circles exist only as human annotations in the task-context photographs. They will not appear in runtime images and are never evidence of success.

Judge only the CURRENT RUNTIME WRIST IMAGES. Approve only when the target bottle has left the gripper and is visibly resting inside the basket. The bottle may be upright, tilted, sideways, or partially hidden by the basket wall. If the evidence is ambiguous or the bottle may still be held, moving, outside, or resting on the rim, lower confidence and do not approve.

Return one JSON object only, with exactly these keys:
{
  "bottle_in_basket": true or false,
  "bottle_released": true or false,
  "confidence": number from 0.0 to 1.0,
  "reason": "brief description of visible evidence",
  "best_runtime_image": integer starting at 1
}
"""


class JSONMessageCompleter(Protocol):
    def complete_json_messages(
        self,
        *,
        config: ModelConfig,
        messages: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, LLMCallTrace]: ...


@dataclass(frozen=True)
class BasketReferenceSet:
    name: str
    task: str
    object_label: str
    destination_label: str
    task_context: tuple[tuple[Path, str], ...]
    positive_examples: tuple[Path, ...]

    @classmethod
    def load(cls, manifest_path: str | Path) -> "BasketReferenceSet":
        path = Path(manifest_path).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        root = path.parent
        context: list[tuple[Path, str]] = []
        for item in payload.get("task_context", []):
            if not isinstance(item, dict):
                raise ValueError("Every task_context entry must be an object.")
            context.append(
                (
                    _resolve_reference_path(root, item.get("path")),
                    str(item.get("description") or "Annotated task context."),
                )
            )
        positives = tuple(
            _resolve_reference_path(root, item) for item in payload.get("positive_examples", [])
        )
        if not context:
            raise ValueError("Basket reference set contains no task-context images.")
        if not positives:
            raise ValueError("Basket reference set contains no positive wrist-camera examples.")
        return cls(
            name=str(payload.get("name") or path.stem),
            task=str(payload.get("task") or "Verify the bottle is inside the robot basket."),
            object_label=str(payload.get("object_label") or "target bottle"),
            destination_label=str(payload.get("destination_label") or "robot basket"),
            task_context=tuple(context),
            positive_examples=positives,
        )


@dataclass(frozen=True)
class BasketVerificationConfig:
    manifest_path: Path
    minimum_confidence: float = 0.8
    max_positive_examples: int = 10
    max_image_edge_px: int = 1024
    jpeg_quality: int = 85


@dataclass(frozen=True)
class BasketVerificationResult:
    status: str
    bottle_in_basket: bool | None
    bottle_released: bool | None
    confidence: float
    reason: str
    best_runtime_image: int | None
    reference_set: str
    reference_image_count: int
    runtime_image_count: int
    trace: LLMCallTrace | None = None

    @property
    def approved(self) -> bool:
        return self.status == "succeeded"

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status,
            "approved": self.approved,
            "bottle_in_basket": self.bottle_in_basket,
            "bottle_released": self.bottle_released,
            "confidence": self.confidence,
            "reason": self.reason,
            "best_runtime_image": self.best_runtime_image,
            "reference_set": self.reference_set,
            "reference_image_count": self.reference_image_count,
            "runtime_image_count": self.runtime_image_count,
        }
        if self.trace is not None:
            payload["model"] = {
                "provider": self.trace.provider,
                "name": self.trace.model,
                "duration_s": self.trace.duration_s,
                "error": self.trace.error,
            }
        return payload


class BasketOutcomeVerifier:
    def __init__(
        self,
        *,
        llm_router: JSONMessageCompleter,
        model_config: ModelConfig,
        config: BasketVerificationConfig,
    ) -> None:
        self.llm_router = llm_router
        self.model_config = model_config
        self.config = config
        self.references = BasketReferenceSet.load(config.manifest_path)

    def verify(self, runtime_wrist_images: Sequence[Any]) -> BasketVerificationResult:
        runtime_images = tuple(runtime_wrist_images)
        if not runtime_images:
            return self._unavailable("No runtime wrist images were provided.", runtime_image_count=0)
        try:
            messages, reference_count = self.build_messages(runtime_images)
        except Exception as exc:
            return self._unavailable(
                f"Could not prepare basket-verification images: {exc}",
                runtime_image_count=len(runtime_images),
            )

        parsed, trace = self.llm_router.complete_json_messages(
            config=self.model_config,
            messages=messages,
        )
        if parsed is None:
            return BasketVerificationResult(
                status="unavailable",
                bottle_in_basket=None,
                bottle_released=None,
                confidence=0.0,
                reason=(trace.error or "Vision model did not return valid JSON."),
                best_runtime_image=None,
                reference_set=self.references.name,
                reference_image_count=reference_count,
                runtime_image_count=len(runtime_images),
                trace=trace,
            )
        return self._result_from_response(
            parsed,
            trace=trace,
            reference_count=reference_count,
            runtime_image_count=len(runtime_images),
        )

    def build_messages(self, runtime_wrist_images: Sequence[Any]) -> tuple[list[dict[str, Any]], int]:
        content: list[dict[str, Any]] = [
            {"type": "text", "text": f"Task: {self.references.task}"},
            {
                "type": "text",
                "text": (
                    "The next images are annotated task context only. Blue circles identify the object "
                    "and destination; do not expect or search for circles in runtime images."
                ),
            },
        ]
        reference_count = 0
        for index, (path, description) in enumerate(self.references.task_context, start=1):
            content.extend(
                [
                    {"type": "text", "text": f"TASK CONTEXT {index}: {description}"},
                    {"type": "image_url", "image_url": {"url": self._image_data_url(path)}},
                ]
            )
            reference_count += 1

        positives = self.references.positive_examples[: max(1, self.config.max_positive_examples)]
        content.append(
            {
                "type": "text",
                "text": (
                    "The next images are successful, unannotated RIGHT-WRIST CAMERA outcomes. "
                    "They show valid variation in bottle orientation, visibility, and basket framing."
                ),
            }
        )
        for index, path in enumerate(positives, start=1):
            content.extend(
                [
                    {"type": "text", "text": f"SUCCESSFUL WRIST EXAMPLE {index}:"},
                    {"type": "image_url", "image_url": {"url": self._image_data_url(path)}},
                ]
            )
            reference_count += 1

        content.append(
            {
                "type": "text",
                "text": (
                    "Judge only the CURRENT RUNTIME WRIST IMAGES below. They are consecutive views "
                    "from one release attempt. Use the clearest settled view; if they conflict, be conservative."
                ),
            }
        )
        for index, image in enumerate(runtime_wrist_images, start=1):
            content.extend(
                [
                    {"type": "text", "text": f"CURRENT RUNTIME WRIST IMAGE {index}:"},
                    {"type": "image_url", "image_url": {"url": self._image_data_url(image)}},
                ]
            )
        content.append(
            {
                "type": "text",
                "text": (
                    "Return the required JSON now. bottle_in_basket and bottle_released must both be true "
                    "only when the runtime evidence supports successful placement."
                ),
            }
        )
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ], reference_count

    def _image_data_url(self, source: Any) -> str:
        image = _image_from_source(source)
        max_edge = max(64, int(self.config.max_image_edge_px))
        width, height = image.size
        if max(width, height) > max_edge:
            scale = max_edge / max(width, height)
            image = image.resize(
                (max(1, round(width * scale)), max(1, round(height * scale))),
                Image.Resampling.LANCZOS,
            )
        buffer = io.BytesIO()
        image.convert("RGB").save(
            buffer,
            format="JPEG",
            quality=max(40, min(95, int(self.config.jpeg_quality))),
            optimize=True,
        )
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def _result_from_response(
        self,
        parsed: dict[str, Any],
        *,
        trace: LLMCallTrace,
        reference_count: int,
        runtime_image_count: int,
    ) -> BasketVerificationResult:
        bottle_in_basket = _coerce_bool(parsed.get("bottle_in_basket"))
        bottle_released = _coerce_bool(parsed.get("bottle_released"))
        confidence = _coerce_confidence(parsed.get("confidence"))
        best_runtime_image = _coerce_runtime_index(
            parsed.get("best_runtime_image"), runtime_image_count
        )
        reason = str(parsed.get("reason") or "Vision model returned no explanation.").strip()
        if bottle_in_basket is None or bottle_released is None:
            status = "uncertain"
            reason = f"Incomplete boolean verdict. {reason}"
        elif confidence < self.config.minimum_confidence:
            status = "uncertain"
        elif bottle_in_basket and bottle_released:
            status = "succeeded"
        else:
            status = "failed"
        return BasketVerificationResult(
            status=status,
            bottle_in_basket=bottle_in_basket,
            bottle_released=bottle_released,
            confidence=confidence,
            reason=reason,
            best_runtime_image=best_runtime_image,
            reference_set=self.references.name,
            reference_image_count=reference_count,
            runtime_image_count=runtime_image_count,
            trace=trace,
        )

    def _unavailable(self, reason: str, *, runtime_image_count: int) -> BasketVerificationResult:
        return BasketVerificationResult(
            status="unavailable",
            bottle_in_basket=None,
            bottle_released=None,
            confidence=0.0,
            reason=reason,
            best_runtime_image=None,
            reference_set=self.references.name,
            reference_image_count=0,
            runtime_image_count=runtime_image_count,
        )


def _resolve_reference_path(root: Path, raw_path: Any) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("Reference image path must be a non-empty string.")
    path = (root / raw_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Basket reference image does not exist: {path}")
    return path


def _image_from_source(source: Any) -> Image.Image:
    if isinstance(source, Image.Image):
        return ImageOps.exif_transpose(source).convert("RGB")
    if isinstance(source, bytes):
        with Image.open(io.BytesIO(source)) as image:
            return ImageOps.exif_transpose(image).convert("RGB")
    if isinstance(source, Path):
        return _image_from_path(source)
    if isinstance(source, str):
        if source.startswith("data:image/"):
            _, separator, encoded = source.partition(",")
            if not separator or not encoded:
                raise ValueError("Image data URL is missing base64 payload.")
            return _image_from_source(base64.b64decode(encoded))
        return _image_from_path(Path(source))

    value = source.detach().cpu().numpy() if hasattr(source, "detach") else source
    try:
        import numpy as np

        array = np.asarray(value)
    except Exception as exc:
        raise TypeError(f"Unsupported image source: {type(source).__name__}") from exc
    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim != 3 or array.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected HxWxC image array, received shape {array.shape}.")
    if np.issubdtype(array.dtype, np.floating):
        maximum = float(np.nanmax(array)) if array.size else 0.0
        if maximum <= 1.0:
            array = array * 255.0
    array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
    array = np.clip(array, 0, 255).astype(np.uint8)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=2)
    return Image.fromarray(array).convert("RGB")


def _image_from_path(path: Path) -> Image.Image:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Runtime wrist image does not exist: {resolved}")
    with Image.open(resolved) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    return None


def _coerce_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    if 1.0 < confidence <= 100.0:
        confidence /= 100.0
    return max(0.0, min(1.0, confidence))


def _coerce_runtime_index(value: Any, image_count: int) -> int | None:
    try:
        index = int(value)
    except (TypeError, ValueError):
        return None
    return index if 1 <= index <= image_count else None
