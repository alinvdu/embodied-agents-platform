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

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multido_xlerobot import XLeRobotBootstrapError, XLeRobotInterface
from multido_xlerobot.bootstrap import resolve_xlerobot_repo_root


def main() -> None:
    api = XLeRobotInterface(resolve_xlerobot_repo_root())
    try:
        print(api.summary())

        robot_config = api.make_robot_config()
        vr_config = api.make_vr_config()

        print(type(robot_config).__name__)
        print(type(vr_config).__name__)
    except XLeRobotBootstrapError as exc:
        print("Bootstrap blocked:", exc)
        print(api.installation_help())


if __name__ == "__main__":
    main()
