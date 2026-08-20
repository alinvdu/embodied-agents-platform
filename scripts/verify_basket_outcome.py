#!/usr/bin/env python3
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
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xlerobot_agent.basket_verification import BasketOutcomeVerifier, BasketVerificationConfig
from xlerobot_agent.llm import AgentLLMRouter, AgentModelSuite, ModelConfig


DEFAULT_MANIFEST = (
    REPO_ROOT
    / "config"
    / "basket_verification"
    / "small_cherry_juice_bottle_v0"
    / "reference_set.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify a cherry-juice bottle release using right-wrist camera images."
    )
    parser.add_argument("--image", action="append", required=True, help="Runtime wrist image; repeat for a burst.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--minimum-confidence", type=float, default=0.8)
    parser.add_argument("--max-positive-examples", type=int, default=10)
    parser.add_argument("--max-image-edge-px", type=int, default=1024)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument(
        "--provider",
        choices=("openai", "openai-compatible", "litellm", "ollama"),
        default=os.getenv("ROBOT42_BASKET_VERIFIER_PROVIDER"),
    )
    parser.add_argument(
        "--model",
        default=os.getenv("ROBOT42_BASKET_VERIFIER_MODEL") or os.getenv("ROBOT42_AGENT_MODEL"),
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("ROBOT42_BASKET_VERIFIER_BASE_URL") or os.getenv("ROBOT42_AGENT_BASE_URL"),
    )
    parser.add_argument(
        "--api-key",
        default=(
            os.getenv("ROBOT42_BASKET_VERIFIER_API_KEY")
            or os.getenv("ROBOT42_AGENT_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and encode images without calling a model.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    provider = args.provider
    base_url = args.base_url
    if provider == "openai":
        provider = "openai-compatible"
        base_url = base_url or "https://api.openai.com/v1/chat/completions"
    if not provider:
        raise SystemExit("Set --provider or ROBOT42_BASKET_VERIFIER_PROVIDER.")
    if not args.model:
        raise SystemExit("Set --model or ROBOT42_BASKET_VERIFIER_MODEL.")

    model_config = ModelConfig(
        provider=provider,
        model=args.model,
        base_url=base_url,
        api_key=args.api_key,
        temperature=0.0,
        max_tokens=300,
    )
    mock = ModelConfig(provider="mock", model="mock")
    router = AgentLLMRouter(
        AgentModelSuite(planner=model_config, critic=mock, coder=mock)
    )
    verifier = BasketOutcomeVerifier(
        llm_router=router,
        model_config=model_config,
        config=BasketVerificationConfig(
            manifest_path=args.manifest,
            minimum_confidence=args.minimum_confidence,
            max_positive_examples=args.max_positive_examples,
            max_image_edge_px=args.max_image_edge_px,
            jpeg_quality=args.jpeg_quality,
        ),
    )
    if args.dry_run:
        messages, reference_count = verifier.build_messages(args.image)
        image_count = sum(
            1
            for message in messages
            for item in (message.get("content") if isinstance(message.get("content"), list) else [])
            if isinstance(item, dict) and item.get("type") == "image_url"
        )
        print(
            json.dumps(
                {
                    "status": "ready",
                    "provider": provider,
                    "model": args.model,
                    "reference_set": verifier.references.name,
                    "reference_image_count": reference_count,
                    "runtime_image_count": len(args.image),
                    "encoded_image_count": image_count,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    result = verifier.verify(args.image)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    if result.status == "succeeded":
        return 0
    if result.status == "unavailable":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
