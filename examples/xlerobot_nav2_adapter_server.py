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

from xlerobot_playground.ros_nav2_router import RosNav2RouterConfig, RosNav2RouterServer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the ROS/Nav2 router service. The exploration brain publishes fused maps, pose, "
            "and latest scan observations here, and the router republishes ROS topics and queries Nav2."
        )
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8891)
    parser.add_argument("--ros-map-topic", default="/map")
    parser.add_argument("--ros-scan-topic", default="/scan")
    parser.add_argument("--ros-map-frame", default="map")
    parser.add_argument("--ros-odom-frame", default="odom")
    parser.add_argument("--ros-base-frame", default="base_link")
    parser.add_argument("--ros-server-timeout-s", type=float, default=10.0)
    parser.add_argument("--ros-ready-timeout-s", type=float, default=20.0)
    parser.add_argument("--ros-allow-multiple-action-servers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--publish-internal-navigation-map", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    server = RosNav2RouterServer(
        RosNav2RouterConfig(
            map_topic=args.ros_map_topic,
            scan_topic=args.ros_scan_topic,
            map_frame=args.ros_map_frame,
            odom_frame=args.ros_odom_frame,
            base_frame=args.ros_base_frame,
            server_timeout_s=args.ros_server_timeout_s,
            ready_timeout_s=args.ros_ready_timeout_s,
            allow_multiple_action_servers=args.ros_allow_multiple_action_servers,
        ),
        host=args.host,
        port=args.port,
    )
    print(f"ROS/Nav2 router service: http://{server.host}:{server.port}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
