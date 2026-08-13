#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_xlerobot_smolvla_policy import build_parser as build_policy_parser
from scripts.run_xlerobot_smolvla_policy import main as run_policy


DEFAULT_ACT_POLICY_PATH = REPO_ROOT / "outputs/train/pretrained_act"
DEFAULT_ACT_DURATION_S = 20.0
DEFAULT_ACT_POLICY_WARMUP_RUNS = 1
DEFAULT_ACT_ACTION_STEPS = 25
DEFAULT_ACT_MAX_JOINT_DELTA = 100.0
DEFAULT_ACT_MAX_GRIPPER_DELTA = 100.0


def build_parser():
    return build_policy_parser(
        policy_label="ACT",
        default_policy_path=DEFAULT_ACT_POLICY_PATH,
        default_duration_s=DEFAULT_ACT_DURATION_S,
        default_async_inference=False,
        default_policy_warmup_runs=DEFAULT_ACT_POLICY_WARMUP_RUNS,
        default_act_action_steps=DEFAULT_ACT_ACTION_STEPS,
        default_max_joint_delta=DEFAULT_ACT_MAX_JOINT_DELTA,
        default_max_gripper_delta=DEFAULT_ACT_MAX_GRIPPER_DELTA,
    )


def main(argv: list[str] | None = None) -> int:
    return run_policy(
        argv,
        expected_policy_type="act",
        policy_label="ACT",
        default_policy_path=DEFAULT_ACT_POLICY_PATH,
        default_duration_s=DEFAULT_ACT_DURATION_S,
        default_async_inference=False,
        default_policy_warmup_runs=DEFAULT_ACT_POLICY_WARMUP_RUNS,
        default_act_action_steps=DEFAULT_ACT_ACTION_STEPS,
        default_max_joint_delta=DEFAULT_ACT_MAX_JOINT_DELTA,
        default_max_gripper_delta=DEFAULT_ACT_MAX_GRIPPER_DELTA,
        rerun_session_name="act_inference",
    )


if __name__ == "__main__":
    raise SystemExit(main())
