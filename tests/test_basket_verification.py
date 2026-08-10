from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from xlerobot_agent.basket_verification import BasketOutcomeVerifier, BasketVerificationConfig
from xlerobot_agent.llm import LLMCallTrace, ModelConfig


class _FakeRouter:
    def __init__(self, response: dict | None, *, error: str | None = None) -> None:
        self.response = response
        self.error = error
        self.messages = None

    def complete_json_messages(self, *, config, messages):
        self.messages = messages
        return self.response, LLMCallTrace(
            provider=config.provider,
            model=config.model,
            duration_s=0.01,
            prompt="basket verification",
            response_text=json.dumps(self.response) if self.response is not None else "",
            error=self.error,
        )


class BasketVerificationTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path]:
        context = root / "context.jpg"
        positive = root / "positive.jpg"
        runtime = root / "runtime.jpg"
        Image.new("RGB", (80, 60), (30, 40, 50)).save(context)
        Image.new("RGB", (80, 60), (80, 90, 100)).save(positive)
        Image.new("RGB", (80, 60), (120, 130, 140)).save(runtime)
        manifest = root / "reference_set.json"
        manifest.write_text(
            json.dumps(
                {
                    "name": "test",
                    "task": "put bottle in basket",
                    "object_label": "bottle",
                    "destination_label": "basket",
                    "task_context": [{"path": context.name, "description": "annotated context"}],
                    "positive_examples": [positive.name],
                }
            ),
            encoding="utf-8",
        )
        return manifest, runtime

    def _verifier(self, manifest: Path, router: _FakeRouter, *, confidence: float = 0.8):
        model = ModelConfig(provider="openai-compatible", model="vision-test", base_url="http://test")
        return BasketOutcomeVerifier(
            llm_router=router,
            model_config=model,
            config=BasketVerificationConfig(
                manifest_path=manifest,
                minimum_confidence=confidence,
            ),
        )

    def test_success_requires_bottle_inside_released_and_confident(self) -> None:
        with TemporaryDirectory() as tmp:
            manifest, runtime = self._fixture(Path(tmp))
            router = _FakeRouter(
                {
                    "bottle_in_basket": True,
                    "bottle_released": True,
                    "confidence": 0.93,
                    "reason": "Bottle is resting in the basket.",
                    "best_runtime_image": 1,
                }
            )
            result = self._verifier(manifest, router).verify([runtime])

        self.assertEqual(result.status, "succeeded")
        self.assertTrue(result.approved)
        self.assertEqual(result.reference_image_count, 2)
        content = router.messages[1]["content"]
        self.assertEqual(sum(item.get("type") == "image_url" for item in content), 3)
        self.assertIn("CURRENT RUNTIME WRIST IMAGE 1", " ".join(item.get("text", "") for item in content))

    def test_confident_negative_is_failed(self) -> None:
        with TemporaryDirectory() as tmp:
            manifest, runtime = self._fixture(Path(tmp))
            router = _FakeRouter(
                {
                    "bottle_in_basket": False,
                    "bottle_released": True,
                    "confidence": 0.91,
                    "reason": "Bottle is outside the basket.",
                    "best_runtime_image": 1,
                }
            )
            result = self._verifier(manifest, router).verify([runtime])

        self.assertEqual(result.status, "failed")
        self.assertFalse(result.approved)

    def test_low_confidence_and_incomplete_responses_are_uncertain(self) -> None:
        with TemporaryDirectory() as tmp:
            manifest, runtime = self._fixture(Path(tmp))
            low_confidence = _FakeRouter(
                {
                    "bottle_in_basket": True,
                    "bottle_released": True,
                    "confidence": 0.65,
                    "reason": "View is occluded.",
                    "best_runtime_image": 1,
                }
            )
            low_result = self._verifier(manifest, low_confidence).verify([runtime])
            incomplete = _FakeRouter(
                {
                    "bottle_in_basket": "maybe",
                    "bottle_released": "true",
                    "confidence": 0.99,
                    "reason": "Cannot tell.",
                    "best_runtime_image": 1,
                }
            )
            incomplete_result = self._verifier(manifest, incomplete).verify([runtime])

        self.assertEqual(low_result.status, "uncertain")
        self.assertEqual(incomplete_result.status, "uncertain")

    def test_model_error_is_unavailable_and_never_approved(self) -> None:
        with TemporaryDirectory() as tmp:
            manifest, runtime = self._fixture(Path(tmp))
            result = self._verifier(
                manifest,
                _FakeRouter(None, error="model timeout"),
            ).verify([runtime])

        self.assertEqual(result.status, "unavailable")
        self.assertFalse(result.approved)
        self.assertIn("timeout", result.reason)

    def test_repo_reference_manifest_is_complete(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest = (
            root
            / "config"
            / "basket_verification"
            / "small_cherry_juice_bottle_v0"
            / "reference_set.json"
        )
        router = _FakeRouter(None)
        verifier = self._verifier(manifest, router)
        self.assertEqual(len(verifier.references.task_context), 2)
        self.assertEqual(len(verifier.references.positive_examples), 10)


if __name__ == "__main__":
    unittest.main()
