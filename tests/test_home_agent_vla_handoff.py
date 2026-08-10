from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from xlerobot_agent.home_agent import HomeAgentConfig, HomeAgentToolRuntime


class _FakeVerificationResult:
    def __init__(self, status: str) -> None:
        self.status = status

    def to_dict(self):
        return {
            "status": self.status,
            "approved": self.status == "succeeded",
            "bottle_in_basket": self.status == "succeeded",
            "bottle_released": True,
            "confidence": 0.95,
            "reason": "verified" if self.status == "succeeded" else "bottle is not in basket",
        }


class _FakeVerifier:
    def __init__(self, status: str = "succeeded") -> None:
        self.status = status
        self.images = None

    def verify(self, images):
        self.images = list(images)
        return _FakeVerificationResult(self.status)


class HomeAgentVLAHandoffTests(unittest.TestCase):
    def test_grab_object_calls_robot_brain_only_when_explicitly_enabled(self) -> None:
        events = []
        verifier = _FakeVerifier()
        with TemporaryDirectory() as tmp:
            runtime = HomeAgentToolRuntime(
                memory={},
                config=HomeAgentConfig(
                    dry_run=False,
                    vla_handoff_enabled=True,
                    vla_handoff_duration_s=60.0,
                    agent_artifacts_root=tmp,
                ),
                emit=lambda *args: events.append(args),
                basket_verifier=verifier,
            )

            def fake_post(_config, path, payload):
                if path == "/vla/run":
                    return {
                        "status": "release_detected",
                        "reason": "release evidence ready",
                        "actions_sent": 100,
                        "release_wrist_images": ["data:image/jpeg;base64,YWJj"],
                    }
                if path == "/vla/stow":
                    return {"status": "succeeded", "reason": "stowed"}
                raise AssertionError(path)

            with patch("xlerobot_agent.home_agent._post_robot_brain", side_effect=fake_post) as post:
                result = runtime.grab_object(object_label="small cherry juice bottle")

        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["actions_sent"], 100)
        self.assertEqual(result["basket_verification"]["status"], "succeeded")
        self.assertEqual(result["stow"]["status"], "succeeded")
        self.assertEqual(len(verifier.images), 1)
        self.assertEqual(
            [(call.args[1], call.args[2]) for call in post.call_args_list],
            [("/vla/run", {"duration_s": 60.0}), ("/vla/stow", {})],
        )
        self.assertEqual(events[-1][0], "tool_executed")

    def test_failed_verification_keeps_arms_at_release_pose(self) -> None:
        with TemporaryDirectory() as tmp:
            runtime = HomeAgentToolRuntime(
                memory={},
                config=HomeAgentConfig(
                    dry_run=False,
                    vla_handoff_enabled=True,
                    agent_artifacts_root=tmp,
                ),
                emit=lambda *_args: None,
                basket_verifier=_FakeVerifier("failed"),
            )
            with patch(
                "xlerobot_agent.home_agent._post_robot_brain",
                return_value={
                    "status": "release_detected",
                    "release_wrist_images": ["data:image/jpeg;base64,YWJj"],
                },
            ) as post:
                result = runtime.grab_object(object_label="bottle")

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["stow"]["status"], "skipped")
        self.assertEqual(post.call_count, 1)

    def test_rollout_duration_alone_is_not_success(self) -> None:
        runtime = HomeAgentToolRuntime(
            memory={},
            config=HomeAgentConfig(dry_run=False, vla_handoff_enabled=True),
            emit=lambda *_args: None,
        )
        with patch(
            "xlerobot_agent.home_agent._post_robot_brain",
            return_value={"status": "succeeded", "reason": "duration elapsed"},
        ):
            result = runtime.grab_object(object_label="bottle")

        self.assertEqual(result["status"], "failed")
        self.assertIn("duration alone", result["reason"])

    def test_grab_object_remains_mocked_in_dry_run(self) -> None:
        runtime = HomeAgentToolRuntime(
            memory={},
            config=HomeAgentConfig(dry_run=True, vla_handoff_enabled=True),
            emit=lambda *_args: None,
        )
        with patch("xlerobot_agent.home_agent._post_robot_brain") as post:
            result = runtime.grab_object(object_label="bottle")

        self.assertEqual(result["status"], "mock_succeeded")
        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
