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
ROS_DISTRO="${ROS_DISTRO:-humble}"
INSTALL_REPLICACAD="${INSTALL_REPLICACAD:-1}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<EOF
Usage: $(basename "$0")

Install the ROS 2 packages that actual Nav2 needs, then create a ManiSkill
Python environment that can see those ROS 2 system packages.

Environment variables:
  XLEROBOT_MANISKILL_VENV  Override the virtualenv path
  PYTHON_BIN               Override the base Python used to create the env
  ROS_DISTRO               ROS 2 distro to target (default: humble)
  INSTALL_REPLICACAD=0     Skip ReplicaCAD asset download
EOF
  exit 0
fi

if [[ ! -f "/opt/ros/$ROS_DISTRO/setup.bash" ]]; then
  cat <<EOF >&2
ROS 2 $ROS_DISTRO is not installed at /opt/ros/$ROS_DISTRO.

Install ROS 2 first, then rerun this script.
EOF
  exit 2
fi

echo "Installing Nav2 and slam_toolbox packages for ROS 2 $ROS_DISTRO..."
sudo apt update
sudo apt install -y \
  "ros-$ROS_DISTRO-navigation2" \
  "ros-$ROS_DISTRO-nav2-bringup" \
  "ros-$ROS_DISTRO-slam-toolbox" \
  "ros-$ROS_DISTRO-tf2-tools" \
  "ros-$ROS_DISTRO-rviz2"

echo "Creating or updating ManiSkill + ROS environment at: $VENV_DIR"
"$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV_DIR/bin/pip" install \
  mani-skill \
  torch \
  pygame \
  rerun-sdk \
  opencv-python \
  pillow \
  tyro \
  pyyaml

if [[ "$INSTALL_REPLICACAD" == "1" ]]; then
  echo "Downloading ReplicaCAD assets through ManiSkill..."
  "$VENV_DIR/bin/python" -m mani_skill.utils.download_asset "ReplicaCAD"
fi

if ! "$VENV_DIR/bin/python" - <<'PY'
import importlib.util
modules = ["rclpy", "nav2_msgs", "mani_skill", "sensor_msgs.msg", "nav_msgs.msg", "tf2_ros"]
missing = [name for name in modules if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("missing:" + ",".join(missing))
PY
then
  cat <<EOF >&2

The environment was created, but one or more required Python packages are still
not visible from $VENV_DIR/bin/python.

Make sure ROS 2 and Nav2 are installed for $ROS_DISTRO, then check:
  source /opt/ros/$ROS_DISTRO/setup.bash
  $VENV_DIR/bin/python - <<'PY'
import rclpy, nav2_msgs, mani_skill
print("ok")
PY
EOF
  exit 3
fi

cat <<EOF

ManiSkill + Nav2 setup complete.

Recommended next steps:
  source /opt/ros/$ROS_DISTRO/setup.bash
  source $VENV_DIR/bin/activate
  $VENV_DIR/bin/python $ROOT_DIR/scripts/render_xlerobot_nav2_params.py --ros-distro $ROS_DISTRO
  python $ROOT_DIR/examples/xlerobot_nav2_bridge_playground.py --render-mode human
EOF
