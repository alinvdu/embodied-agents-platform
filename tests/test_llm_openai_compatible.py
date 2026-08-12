from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from xlerobot_agent.llm import AgentLLMRouter, AgentModelSuite, ModelConfig


class _FakeHttpResponse:
    def __init__(self) -> None:
        self._body = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({"status": "ok"}),
                        }
                    }
                ]
            }
        ).encode("utf-8")

    def __enter__(self) -> "_FakeHttpResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return self._body


class OpenAICompatibleRouterTests(unittest.TestCase):
    def _request_payload(self, config: ModelConfig) -> dict[str, object]:
        router = AgentLLMRouter(
            AgentModelSuite(planner=config, critic=config, coder=config)
        )
        seen: dict[str, object] = {}

        def _fake_urlopen(request, timeout=0):
            seen.update(json.loads(request.data.decode("utf-8")))
            return _FakeHttpResponse()

        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            parsed, trace = router.complete_json_messages(
                config=config,
                messages=[{"role": "user", "content": "Return JSON."}],
            )

        self.assertEqual(parsed, {"status": "ok"})
        self.assertIsNone(trace.error)
        return seen

    def test_gpt5_uses_max_completion_tokens_and_omits_temperature(self) -> None:
        payload = self._request_payload(
            ModelConfig(
                provider="openai-compatible",
                model="gpt-5.6-terra",
                base_url="https://api.openai.com/v1/chat/completions",
                temperature=0.0,
                max_tokens=300,
            )
        )

        self.assertEqual(payload["max_completion_tokens"], 300)
        self.assertNotIn("max_tokens", payload)
        self.assertNotIn("temperature", payload)

    def test_legacy_compatible_model_keeps_max_tokens_and_temperature(self) -> None:
        payload = self._request_payload(
            ModelConfig(
                provider="openai-compatible",
                model="vision-test",
                base_url="http://test/v1/chat/completions",
                temperature=0.4,
                max_tokens=512,
            )
        )

        self.assertEqual(payload["max_tokens"], 512)
        self.assertEqual(payload["temperature"], 0.4)
        self.assertNotIn("max_completion_tokens", payload)


if __name__ == "__main__":
    unittest.main()
