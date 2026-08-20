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

from .models import SkillContract, SkillType


@dataclass
class SkillRegistry:
    _skills: dict[str, SkillContract] = field(default_factory=dict)

    def register(self, skill: SkillContract) -> None:
        self._skills[skill.skill_id] = skill

    def register_many(self, skills: list[SkillContract]) -> None:
        for skill in skills:
            self.register(skill)

    def get(self, skill_id: str) -> SkillContract:
        return self._skills[skill_id]

    def list_enabled(self) -> list[SkillContract]:
        return [skill for skill in self._skills.values() if skill.enabled]

    def by_type(self, skill_type: SkillType) -> list[SkillContract]:
        return [skill for skill in self.list_enabled() if skill.skill_type == skill_type]

    def skill_ids(self) -> list[str]:
        return list(self._skills.keys())
