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
from pathlib import Path


@dataclass(frozen=True)
class XLeRobotPaths:
    repo_root: Path
    software_src: Path
    xlevr_root: Path
    record_script: Path
    model_file: Path
    robot_pkg_dir: Path
    robot_2wheels_pkg_dir: Path
    vr_pkg_dir: Path

    @classmethod
    def from_repo_root(cls, repo_root: str | Path) -> "XLeRobotPaths":
        root = Path(repo_root).expanduser().resolve()
        return cls(
            repo_root=root,
            software_src=root / "software" / "src",
            xlevr_root=root / "XLeVR",
            record_script=root / "software" / "src" / "record.py",
            model_file=root / "software" / "src" / "model" / "SO101Robot.py",
            robot_pkg_dir=root / "software" / "src" / "robots" / "xlerobot",
            robot_2wheels_pkg_dir=root / "software" / "src" / "robots" / "xlerobot_2wheels",
            vr_pkg_dir=root / "software" / "src" / "teleporators" / "xlerobot_vr",
        )
