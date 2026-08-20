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

from .models import AgentRunRecord, WorldState


@dataclass(frozen=True)
class WakeWordConfig:
    wake_word: str = "hey xlerobot"
    case_sensitive: bool = False


@dataclass(frozen=True)
class VoiceCommand:
    raw_transcript: str
    normalized_command: str
    wake_word: str
    translated: bool = False


class MockWakeWordDetector:
    def __init__(self, config: WakeWordConfig | None = None) -> None:
        self.config = config or WakeWordConfig()

    def extract_command(self, transcript: str) -> str | None:
        haystack = transcript if self.config.case_sensitive else transcript.lower()
        needle = self.config.wake_word if self.config.case_sensitive else self.config.wake_word.lower()
        if not haystack.startswith(needle):
            return None
        command = transcript[len(self.config.wake_word):].strip(" ,:;.-")
        return command or None


class MockVoiceTranslator:
    """Mock translation/normalization layer.

    In v1 this is intentionally simple: it normalizes a spoken command into a
    planner-ready instruction string. A real implementation can replace this with
    ASR + translation + command normalization prompts.
    """

    def normalize(self, command: str) -> VoiceCommand:
        normalized = " ".join(command.strip().split())
        return VoiceCommand(
            raw_transcript=command,
            normalized_command=normalized,
            wake_word="",
            translated=False,
        )


class VoiceCommandPipeline:
    def __init__(
        self,
        wake_word_detector: MockWakeWordDetector | None = None,
        translator: MockVoiceTranslator | None = None,
    ) -> None:
        self.wake_word_detector = wake_word_detector or MockWakeWordDetector()
        self.translator = translator or MockVoiceTranslator()

    def process_transcript(self, transcript: str) -> VoiceCommand | None:
        command = self.wake_word_detector.extract_command(transcript)
        if command is None:
            return None
        normalized = self.translator.normalize(command)
        return VoiceCommand(
            raw_transcript=transcript,
            normalized_command=normalized.normalized_command,
            wake_word=self.wake_word_detector.config.wake_word,
            translated=normalized.translated,
        )


class MockVoiceCommandApp:
    """Small stdin-driven mock app for wake-word-triggered command dispatch."""

    def __init__(self, pipeline, runtime, world_state: WorldState) -> None:
        self.pipeline = pipeline
        self.runtime = runtime
        self.world_state = world_state

    def handle_transcript(self, transcript: str) -> AgentRunRecord | None:
        return self.runtime.run_voice_transcript(transcript, self.world_state)

    def run_cli(self) -> None:
        print("Mock voice app ready. Type transcripts, prefixing commands with the wake word.")
        print(f"Wake word: {self.pipeline.wake_word_detector.config.wake_word}")
        print("Type `quit` to exit.")
        while True:
            transcript = input("voice> ").strip()
            if transcript.lower() in {"quit", "exit"}:
                break
            record = self.handle_transcript(transcript)
            if record is None:
                print("Wake word not detected. Ignoring input.")
                continue
            print(f"Normalized instruction: {record.normalized_instruction}")
            print(f"Discovered places: {[place.name for place in record.discovered_places]}")
            print(f"Subgoals: {[subgoal.text for subgoal in record.subgoals]}")
            if record.steps:
                first = record.steps[0]
                print(f"Selected skill: {first.selected_skill.skill_id}")
                print(f"Reasoning: {first.selected_score.reasoning}")
                print(f"Execution status: {first.execution_result.status.value}")
