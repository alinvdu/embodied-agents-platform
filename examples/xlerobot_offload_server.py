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

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xlerobot_agent.offload import OffloadServer, OffloadServerConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the simplified Ubuntu offload server. "
            "The macOS robot brain can register itself here, publish sensory state, and offload "
            "navigation, perception, mapping, and VLA-style tool and skill calls over Wi-Fi."
        )
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--nav2-service-url", default=None)
    parser.add_argument("--perception-service-url", default=None)
    parser.add_argument("--vla-service-url", default=None)
    parser.add_argument("--state-log-path", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    server = OffloadServer(
        OffloadServerConfig(
            host=args.host,
            port=args.port,
            nav2_service_url=args.nav2_service_url,
            perception_service_url=args.perception_service_url,
            vla_service_url=args.vla_service_url,
            state_log_path=args.state_log_path,
        )
    )
    print(f"XLeRobot offload server: http://{server.host}:{server.port}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
