from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xlerobot_agent.exploration import ExplorationBackend, ExplorationBackendConfig
from xlerobot_agent.exploration_ui import (
    ExplorationReviewServer,
    LocalExplorationUIController,
    RemoteExplorationUIController,
)
from xlerobot_agent.offload import OffloadClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the exploration review UI. It can use a local sim-first exploration backend "
            "or connect to a remote offload server and review the current brain's map."
        )
    )
    parser.add_argument("--mode", choices=("local", "offload"), default="local")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--persist-path", default=None)
    parser.add_argument("--memory-root", default=None)
    parser.add_argument("--offload-server-url", default=None)
    parser.add_argument("--brain-name", default="xlerobot-review-brain")
    parser.add_argument("--brain-id", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "offload":
        if not args.offload_server_url:
            raise SystemExit("`--offload-server-url` is required in offload mode.")
        controller = RemoteExplorationUIController(
            OffloadClient(
                args.offload_server_url,
                brain_name=args.brain_name,
                brain_id=args.brain_id,
            )
        )
    else:
        backend = ExplorationBackend(
            ExplorationBackendConfig(
                mode="sim",
                persist_path=args.persist_path,
                memory_root_path=args.memory_root,
            )
        )
        controller = LocalExplorationUIController(backend)

    server = ExplorationReviewServer(controller, host=args.host, port=args.port)
    print(f"XLeRobot exploration review UI: http://{args.host}:{args.port}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
