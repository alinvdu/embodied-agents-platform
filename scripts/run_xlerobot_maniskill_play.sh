#!/usr/bin/env bash
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

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${XLEROBOT_MANISKILL_VENV:-$ROOT_DIR/.venv-maniskill}"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  cat <<EOF >&2
Missing ManiSkill environment at $VENV_DIR

Create it first with:
  $ROOT_DIR/scripts/setup_xlerobot_maniskill_env.sh
EOF
  exit 2
fi

cd "$ROOT_DIR"
exec "$VENV_DIR/bin/python" -m multido_xlerobot.maniskill "$@"
