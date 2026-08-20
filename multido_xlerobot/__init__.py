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

"""Clean integration layer for consuming XLeRobot from this repository.

This package does not copy XLeRobot code. It bootstraps a forked XLeRobot repo
into the installed `lerobot` namespace and exposes a stable facade on top.
"""

from .bootstrap import (
    XLeRobotBootstrapError,
    XLeRobotBootstrapResult,
    bootstrap_xlerobot,
    resolve_xlerobot_repo_root,
)
from .interface import XLeRobotInterface
from .types import XLeRobotPaths

__all__ = [
    "XLeRobotBootstrapError",
    "XLeRobotBootstrapResult",
    "XLeRobotInterface",
    "XLeRobotPaths",
    "bootstrap_xlerobot",
    "resolve_xlerobot_repo_root",
    "XLeRobotManiSkillBootstrapResult",
    "XLeRobotManiSkillError",
    "bootstrap_xlerobot_maniskill",
    "run_keyboard_play_demo",
]


def __getattr__(name: str):
    if name in {
        "XLeRobotManiSkillBootstrapResult",
        "XLeRobotManiSkillError",
        "bootstrap_xlerobot_maniskill",
        "run_keyboard_play_demo",
    }:
        from .maniskill import (
            XLeRobotManiSkillBootstrapResult,
            XLeRobotManiSkillError,
            bootstrap_xlerobot_maniskill,
            run_keyboard_play_demo,
        )

        exports = {
            "XLeRobotManiSkillBootstrapResult": XLeRobotManiSkillBootstrapResult,
            "XLeRobotManiSkillError": XLeRobotManiSkillError,
            "bootstrap_xlerobot_maniskill": bootstrap_xlerobot_maniskill,
            "run_keyboard_play_demo": run_keyboard_play_demo,
        }
        return exports[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
