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

import os
import subprocess
import sys
from pathlib import Path

from multido_xlerobot.bootstrap import resolve_xlerobot_repo_root


def resolve_repo_root(explicit_repo_root: str | None = None) -> Path:
    return resolve_xlerobot_repo_root(explicit_repo_root)


def default_sim_python_bin(repo_root: Path) -> Path:
    candidate = repo_root / ".venv-maniskill" / "bin" / "python"
    if candidate.exists():
        return candidate
    fallback = Path("/home/alin/Robot42/.venv-maniskill/bin/python")
    if fallback.exists():
        return fallback
    return Path(sys.executable)


def exec_python_module(
    module: str,
    *,
    python_bin: str | Path,
    argv: list[str],
    cwd: str | Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> int:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    command = [str(Path(python_bin).expanduser()), "-m", module, *argv]
    completed = subprocess.run(command, cwd=cwd, env=env)
    return completed.returncode
