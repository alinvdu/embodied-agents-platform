from __future__ import annotations

import base64
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from .models import ExecutionStatus, Subgoal, WorldState
from .prompts import build_action_selection_system_prompt


@dataclass(frozen=True)
class ModelConfig:
    provider: str = "mock"
    model: str = "mock"
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.2
    max_tokens: int = 1024
    thinking: bool = False
    reasoning_effort: str | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)
    extra_headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentModelSuite:
    planner: ModelConfig
    critic: ModelConfig
    coder: ModelConfig
    visual_summary: ModelConfig | None = None
    visual_diff: ModelConfig | None = None


@dataclass(frozen=True)
class LLMCallTrace:
    provider: str
    model: str
    duration_s: float
    prompt: str
    response_text: str
    raw_response: Any = None
    error: str | None = None


@dataclass(frozen=True)
class ActionDecision:
    action_type: str
    action_id: str
    summary: str
    reasoning_summary: str


@dataclass(frozen=True)
class ReviewDecision:
    success: bool
    replan: bool
    summary: str
    reasoning_summary: str


@dataclass(frozen=True)
class CodeGenerationResult:
    code: str
    summary: str
    reasoning_summary: str


class AgentLLMRouter:
    def __init__(self, model_suite: AgentModelSuite) -> None:
        self.model_suite = model_suite

    def complete_json_prompt(
        self,
        *,
        config: ModelConfig,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[dict[str, Any] | None, LLMCallTrace]:
        return self._complete_json(config, system_prompt, user_prompt)

    def complete_json_messages(
        self,
        *,
        config: ModelConfig,
        messages: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, LLMCallTrace]:
        return self._complete_json_from_messages(config, messages)

    def select_action(
        self,
        *,
        instruction: str,
        subgoal: Subgoal,
        world_state: WorldState,
        candidates: list[dict[str, Any]],
        preferred_action_id: str | None = None,
    ) -> tuple[ActionDecision, LLMCallTrace]:
        fallback = self._mock_select_action(candidates, preferred_action_id=preferred_action_id)
        config = self.model_suite.planner
        if config.provider == "mock":
            return fallback, self._mock_trace(config, "planner.select_action", fallback)

        prompt = json.dumps(
            {
                "instruction": instruction,
                "subgoal": {
                    "text": subgoal.text,
                    "kind": subgoal.kind,
                    "target": subgoal.target,
                },
                "world_state": _summarize_world_state(world_state),
                "preferred_action_id": preferred_action_id,
                "candidates": candidates,
            },
            indent=2,
            sort_keys=True,
        )
        system_prompt = build_action_selection_system_prompt()
        parsed, trace = self._complete_json(config, system_prompt, prompt)
        if parsed is None:
            return fallback, trace
        decision = ActionDecision(
            action_type=str(parsed.get("action_type", fallback.action_type)),
            action_id=str(parsed.get("action_id", fallback.action_id)),
            summary=str(parsed.get("summary", fallback.summary)),
            reasoning_summary=str(parsed.get("reasoning_summary", fallback.reasoning_summary)),
        )
        if not any(
            item["action_id"] == decision.action_id and item["action_type"] == decision.action_type
            for item in candidates
        ):
            return fallback, trace
        return decision, trace

    def review_action(
        self,
        *,
        instruction: str,
        subgoal: Subgoal,
        world_state: WorldState,
        action: dict[str, Any],
        action_status: ExecutionStatus,
        action_summary: str,
    ) -> tuple[ReviewDecision, LLMCallTrace]:
        fallback = self._mock_review(action_status, action_summary)
        config = self.model_suite.critic
        if config.provider == "mock":
            return fallback, self._mock_trace(config, "critic.review_action", fallback)

        prompt = json.dumps(
            {
                "instruction": instruction,
                "subgoal": {
                    "text": subgoal.text,
                    "kind": subgoal.kind,
                    "target": subgoal.target,
                },
                "world_state": _summarize_world_state(world_state),
                "action": action,
                "action_status": action_status.value,
                "action_summary": action_summary,
            },
            indent=2,
            sort_keys=True,
        )
        system_prompt = (
            "You are a critic for a robot execution loop. "
            "Return JSON only with keys: success, replan, summary, reasoning_summary. "
            "If the action failed or was blocked, success should be false."
        )
        parsed, trace = self._complete_json(config, system_prompt, prompt)
        if parsed is None:
            return fallback, trace
        review = ReviewDecision(
            success=bool(parsed.get("success", fallback.success)),
            replan=bool(parsed.get("replan", fallback.replan)),
            summary=str(parsed.get("summary", fallback.summary)),
            reasoning_summary=str(parsed.get("reasoning_summary", fallback.reasoning_summary)),
        )
        return review, trace

    def generate_helper_code(
        self,
        *,
        instruction: str,
        subgoal: Subgoal,
        world_state: WorldState,
        candidates: list[dict[str, Any]],
        question: str,
    ) -> tuple[CodeGenerationResult, LLMCallTrace]:
        fallback = self._mock_code(candidates, question)
        config = self.model_suite.coder
        if config.provider == "mock":
            return fallback, self._mock_trace(config, "coder.generate_helper_code", fallback)

        prompt = json.dumps(
            {
                "instruction": instruction,
                "subgoal": {
                    "text": subgoal.text,
                    "kind": subgoal.kind,
                    "target": subgoal.target,
                },
                "world_state": _summarize_world_state(world_state),
                "question": question,
                "candidates": candidates,
            },
            indent=2,
            sort_keys=True,
        )
        system_prompt = (
            "You write short Python snippets for bounded reasoning only. "
            "Return JSON only with keys: code, summary, reasoning_summary. "
            "The code must not use imports. It may read WORLD_STATE, CANDIDATES, QUESTION and must assign RESULT."
        )
        parsed, trace = self._complete_json(config, system_prompt, prompt)
        if parsed is None:
            return fallback, trace
        code = str(parsed.get("code", "")).strip()
        if not code:
            return fallback, trace
        return (
            CodeGenerationResult(
                code=code,
                summary=str(parsed.get("summary", fallback.summary)),
                reasoning_summary=str(parsed.get("reasoning_summary", fallback.reasoning_summary)),
            ),
            trace,
        )

    def _complete_json(
        self,
        config: ModelConfig,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[dict[str, Any] | None, LLMCallTrace]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self._complete_json_from_messages(config, messages)

    def _complete_json_from_messages(
        self,
        config: ModelConfig,
        messages: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, LLMCallTrace]:
        started = time.time()
        prompt = _messages_to_prompt_text(messages)
        try:
            if config.provider == "litellm":
                response_text, raw_response = self._complete_via_litellm_messages(config, messages)
            elif config.provider == "ollama":
                response_text, raw_response = self._complete_via_ollama_messages(config, messages)
            else:
                response_text, raw_response = self._complete_via_openai_compatible_messages(config, messages)
            parsed = _extract_json_object(response_text)
            if parsed is None:
                raise ValueError("model response did not contain a JSON object")
            return parsed, LLMCallTrace(
                provider=config.provider,
                model=config.model,
                duration_s=time.time() - started,
                prompt=prompt,
                response_text=response_text,
                raw_response=raw_response,
            )
        except Exception as exc:
            return None, LLMCallTrace(
                provider=config.provider,
                model=config.model,
                duration_s=time.time() - started,
                prompt=prompt,
                response_text="",
                raw_response=None,
                error=str(exc),
            )

    def _complete_via_openai_compatible(
        self,
        config: ModelConfig,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self._complete_via_openai_compatible_messages(config, messages)

    def _complete_via_openai_compatible_messages(
        self,
        config: ModelConfig,
        messages: list[dict[str, Any]],
    ) -> tuple[str, Any]:
        if not config.base_url:
            raise ValueError("openai-compatible provider requires base_url")
        payload: dict[str, Any] = {
            "model": config.model,
            "messages": messages,
        }
        if _uses_reasoning_chat_completion_parameters(config.model):
            payload["max_completion_tokens"] = config.max_tokens
        else:
            payload["temperature"] = config.temperature
            payload["max_tokens"] = config.max_tokens
        if config.reasoning_effort:
            payload["reasoning_effort"] = config.reasoning_effort
        if config.thinking:
            payload["thinking"] = {"type": "enabled"}
        payload.update(config.extra_body)

        headers = {"Content-Type": "application/json", **config.extra_headers}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        request = urllib.request.Request(
            config.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        parsed = json.loads(body)
        choice = parsed["choices"][0]["message"]["content"]
        return _flatten_message_content(choice), parsed

    def _complete_via_litellm(
        self,
        config: ModelConfig,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self._complete_via_litellm_messages(config, messages)

    def _complete_via_litellm_messages(
        self,
        config: ModelConfig,
        messages: list[dict[str, Any]],
    ) -> tuple[str, Any]:
        try:
            from litellm import completion
        except ImportError as exc:
            raise RuntimeError("litellm is not installed in this environment") from exc

        kwargs: dict[str, Any] = {
            "model": config.model,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        if config.base_url:
            kwargs["api_base"] = config.base_url
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.reasoning_effort:
            kwargs["reasoning_effort"] = config.reasoning_effort
        kwargs.update(config.extra_body)
        raw_response = completion(**kwargs)
        if isinstance(raw_response, dict):
            choice = raw_response["choices"][0]["message"]["content"]
        else:
            choice = raw_response.choices[0].message.content
        return _flatten_message_content(choice), raw_response

    def _complete_via_ollama_messages(
        self,
        config: ModelConfig,
        messages: list[dict[str, Any]],
    ) -> tuple[str, Any]:
        endpoint = _ollama_generate_endpoint(config.base_url)
        options: dict[str, Any] = {
            "temperature": float(config.temperature),
            "num_predict": int(config.max_tokens),
        }
        payload: dict[str, Any] = {
            "model": config.model,
            "prompt": _messages_to_prompt_text(messages),
            "stream": False,
            "format": "json",
            "options": options,
        }
        images = _collect_ollama_images(messages)
        if images:
            payload["images"] = images
        if config.extra_body:
            extra_body = json.loads(json.dumps(config.extra_body))
            extra_options = extra_body.pop("options", None)
            if isinstance(extra_options, dict):
                payload["options"].update(extra_options)
            payload.update(extra_body)

        headers = {"Content-Type": "application/json", **config.extra_headers}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc

        parsed = json.loads(body)
        if parsed.get("error"):
            raise RuntimeError(str(parsed["error"]))
        response_text = str(parsed.get("response", "")).strip()
        if not response_text:
            raise RuntimeError("Ollama returned an empty response body")
        return response_text, parsed

    def _mock_select_action(
        self,
        candidates: list[dict[str, Any]],
        *,
        preferred_action_id: str | None = None,
    ) -> ActionDecision:
        if not candidates:
            return ActionDecision(
                action_type="finish",
                action_id="finish",
                summary="No candidates were available, so the loop should stop.",
                reasoning_summary="The planner had no feasible skill or tool candidate to choose from.",
            )
        has_skill_candidate = any(item.get("action_type") == "skill" for item in candidates)
        ranked = sorted(
            candidates,
            key=lambda item: (
                item.get("action_id") == preferred_action_id,
                item.get("action_type") == "skill" if has_skill_candidate else True,
                item.get("action_id") == "code_execution",
                item.get("score", 0.0),
            ),
            reverse=True,
        )
        best = ranked[0]
        return ActionDecision(
            action_type=str(best["action_type"]),
            action_id=str(best["action_id"]),
            summary=f"Select `{best['action_id']}` as the next action.",
            reasoning_summary=str(best.get("reasoning") or best.get("description") or ""),
        )

    def _mock_review(self, action_status: ExecutionStatus, action_summary: str) -> ReviewDecision:
        summary_lower = action_summary.lower()
        success = action_status == ExecutionStatus.SUCCEEDED
        suspicious_no_change = (
            "no task-relevant visual change detected" in summary_lower
            and any(
                token in summary_lower
                for token in (
                    "open",
                    "search",
                    "inspect",
                    "perceive",
                    "ground",
                    "waypoint",
                    "grab",
                    "pick",
                )
            )
        )
        if success and (
            suspicious_no_change
            or any(
                token in summary_lower
                for token in (
                    "no grounded 3d match",
                    "no waypoint could be derived",
                )
            )
        ):
            success = False
        if success:
            return ReviewDecision(
                success=True,
                replan=False,
                summary="The step succeeded and the agent can move to the next subgoal.",
                reasoning_summary=action_summary,
            )
        return ReviewDecision(
            success=False,
            replan=True,
            summary="The step did not succeed, so the agent should reflect and choose a different action.",
            reasoning_summary=action_summary,
        )

    def _mock_code(
        self,
        candidates: list[dict[str, Any]],
        question: str,
    ) -> CodeGenerationResult:
        return CodeGenerationResult(
            code=(
                "best = max(CANDIDATES, key=lambda item: item.get('score', 0.0), default=None)\n"
                "RESULT = {\n"
                "    'recommended_action_id': best['action_id'] if best else None,\n"
                "    'recommended_action_type': best['action_type'] if best else None,\n"
                "    'notes': 'Picked the highest-scoring remaining candidate.',\n"
                "    'question': QUESTION,\n"
                "}\n"
                "print('helper code completed')\n"
            ),
            summary="Generate helper code that inspects the candidate list and returns a recommendation.",
            reasoning_summary=question,
        )

    def _mock_trace(self, config: ModelConfig, label: str, result: Any) -> LLMCallTrace:
        return LLMCallTrace(
            provider=config.provider,
            model=config.model,
            duration_s=0.0,
            prompt=label,
            response_text=json.dumps(result.__dict__, indent=2, sort_keys=True),
            raw_response=result.__dict__,
        )


def _flatten_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            if "text" in item and item["text"] is not None:
                parts.append(str(item["text"]))
            elif "content" in item and item["content"] is not None:
                parts.append(str(item["content"]))
        return "\n".join(part for part in parts if part)
    return str(content)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None


def _uses_reasoning_chat_completion_parameters(model: str) -> bool:
    normalized = model.strip().lower()
    return normalized.startswith(("gpt-5", "o1", "o3", "o4"))


def _ollama_generate_endpoint(base_url: str | None) -> str:
    candidate = (base_url or "http://localhost:11434").rstrip("/")
    if candidate.endswith("/api/generate"):
        return candidate
    return f"{candidate}/api/generate"


def _collect_ollama_images(messages: list[dict[str, Any]]) -> list[str]:
    encoded_images: list[str] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "image_url":
                continue
            image_url = item.get("image_url", {})
            if isinstance(image_url, dict):
                url = str(image_url.get("url", "")).strip()
            else:
                url = str(image_url).strip()
            if not url:
                continue
            encoded_images.append(_image_url_to_ollama_base64(url))
    return encoded_images


def _image_url_to_ollama_base64(url: str) -> str:
    if url.startswith("data:"):
        _, _, encoded = url.partition(",")
        if not encoded:
            raise ValueError("image data URL did not contain base64 data")
        return encoded

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme in {"http", "https"}:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read()
        return base64.b64encode(data).decode("utf-8")

    if parsed.scheme == "file":
        path = Path(urllib.request.url2pathname(parsed.path))
    else:
        path = Path(url)
    data = path.read_bytes()
    return base64.b64encode(data).decode("utf-8")


def _messages_to_prompt_text(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role", "user")).upper()
        content = message.get("content", "")
        lines.append(f"{role}:")
        lines.append(_content_to_trace_text(content))
    return "\n\n".join(lines)


def _content_to_trace_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            item_type = item.get("type")
            if item_type == "text":
                parts.append(str(item.get("text", "")))
                continue
            if item_type == "image_url":
                image_url = item.get("image_url", {})
                if isinstance(image_url, dict):
                    url = str(image_url.get("url", ""))
                else:
                    url = str(image_url)
                parts.append(_summarize_image_url(url))
                continue
            parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def _summarize_image_url(url: str) -> str:
    if url.startswith("data:image/"):
        prefix, _, encoded = url.partition(",")
        mime = prefix.split(";")[0].removeprefix("data:")
        return f"[image:{mime};base64_bytes={len(encoded)}]"
    if url.startswith("data:"):
        return "[image:data_url]"
    return f"[image_url:{url}]"


def _summarize_world_state(world_state: WorldState) -> dict[str, Any]:
    return {
        "current_task": world_state.current_task,
        "current_pose": world_state.current_pose,
        "localization_confidence": world_state.localization_confidence,
        "visible_objects": sorted(world_state.visible_objects),
        "visible_landmarks": sorted(world_state.visible_landmarks),
        "image_descriptions": list(world_state.image_descriptions),
        "semantic_memory_summary": world_state.semantic_memory_summary,
        "spatial_memory_summary": world_state.spatial_memory_summary,
        "place_memories": [
            {"name": place.name, "confidence": place.confidence, "evidence": place.evidence}
            for place in world_state.place_memories
        ],
        "recent_execution_history": list(world_state.recent_execution_history),
        "available_observations": sorted(world_state.available_observations),
        "satisfied_preconditions": sorted(world_state.satisfied_preconditions),
        "metadata": world_state.metadata,
        "readiness_state": world_state.readiness_state.value,
    }
