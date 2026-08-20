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

from xlerobot_agent.brain_service import BrainBridge, BrainServiceServer
from xlerobot_agent.offload import OffloadClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the macOS brain bridge service. "
            "Local robot processes can POST sensor / world-state updates here, and this service "
            "registers the brain with the Ubuntu offload server and forwards state over Wi-Fi."
        )
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8891)
    parser.add_argument("--offload-server-url", required=True)
    parser.add_argument("--brain-name", default="xlerobot-macos-brain")
    parser.add_argument("--brain-id", default=None)
    parser.add_argument("--brain-meta-json", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metadata = {}
    if args.brain_meta_json:
        import json

        metadata = json.loads(args.brain_meta_json)
    bridge = BrainBridge(
        OffloadClient(
            args.offload_server_url,
            brain_name=args.brain_name,
            brain_id=args.brain_id,
            metadata=metadata,
        )
    )
    registration = bridge.register()
    server = BrainServiceServer(bridge, host=args.host, port=args.port)
    print(
        f"XLeRobot brain bridge: http://{server.host}:{server.port} "
        f"(registered as {registration['brain_id']} -> {args.offload_server_url})"
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
