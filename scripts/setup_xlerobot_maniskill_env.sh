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
PYTHON_BIN="${PYTHON_BIN:-python3}"
INSTALL_REPLICACAD="${INSTALL_REPLICACAD:-1}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<EOF
Usage: $(basename "$0")

Creates a local Python environment for the XLeRobot ManiSkill keyboard demos.

Environment variables:
  XLEROBOT_MANISKILL_VENV  Override the virtualenv path
  PYTHON_BIN               Override the base Python used to create the env
  INSTALL_REPLICACAD=0     Skip ReplicaCAD asset download
EOF
  exit 0
fi

echo "Creating or updating ManiSkill environment at: $VENV_DIR"

if [[ "$(uname -s)" == "Darwin" ]]; then
  cat <<'EOF'
Warning: ManiSkill on macOS supports CPU simulation and standard rendering, but it
requires Vulkan to be installed and configured first. If rendering fails, install
the LunarG Vulkan SDK and export the required environment variables before use.
EOF
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV_DIR/bin/pip" install \
  mani-skill \
  torch \
  pygame \
  rerun-sdk \
  opencv-python \
  pillow \
  tyro

if ! "$VENV_DIR/bin/python" - <<'PY'
import importlib.util
import sys

raise SystemExit(0 if importlib.util.find_spec("sapien") else 1)
PY
then
  cat <<'EOF' >&2

The Python environment was created, but `sapien` is not available for this
platform / interpreter combination. XLeRobot ManiSkill keyboard play requires
SAPIEN at runtime, so this setup cannot launch the simulator yet.

Recommended next step:
  use Ubuntu (the XLeRobot simulation docs assume Ubuntu) and rerun this script
  in that environment.
EOF
  exit 2
fi

if [[ "$INSTALL_REPLICACAD" == "1" ]]; then
  echo "Downloading ReplicaCAD assets through ManiSkill..."
  "$VENV_DIR/bin/python" -m mani_skill.utils.download_asset "ReplicaCAD"
fi

cat <<EOF

ManiSkill setup complete.

Quick checks:
  $VENV_DIR/bin/python -m multido_xlerobot.maniskill --check

Launch the keyboard play demo:
  $ROOT_DIR/scripts/run_xlerobot_maniskill_play.sh

Or directly with the environment's Python:
  cd $ROOT_DIR
  $VENV_DIR/bin/python -m multido_xlerobot.maniskill --demo ee_keyboard
EOF
