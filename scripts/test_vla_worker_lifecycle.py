#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xlerobot_agent.vla_worker import VLAWorkerConfig, VLAWorkerSupervisor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start and stop the model-only VLA worker without opening robot or camera hardware."
    )
    parser.add_argument(
        "--policy-path",
        type=Path,
        default=REPO_ROOT / "outputs/train/pretrained_vla_batch_16_30k_new_dataset",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "datasets/small-juice-bottle-to-basket-right-arm",
    )
    parser.add_argument(
        "--dataset-repo-id",
        default="alindumitru/small-juice-bottle-to-basket-right-arm",
    )
    parser.add_argument(
        "--task",
        default="Pick up the small bottle of cherry juice and put it in the robot basket.",
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--startup-timeout-s", type=float, default=180.0)
    parser.add_argument("--log-path", type=Path, default=REPO_ROOT / "artifacts/vla_worker/smolvla_worker.log")
    parser.add_argument(
        "--predict-dataset-index",
        type=int,
        default=None,
        help="After loading, run one model-only prediction using this local dataset frame.",
    )
    parser.add_argument("--mock", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cache = os.getenv("HF_DATASETS_CACHE")
    config = VLAWorkerConfig(
        policy_path=args.policy_path,
        dataset_repo_id=args.dataset_repo_id,
        dataset_root=args.dataset_root,
        task=args.task,
        device=args.device,
        startup_timeout_s=args.startup_timeout_s,
        log_path=args.log_path,
        hf_datasets_cache=Path(cache) if cache else None,
        backend="mock" if args.mock else "lerobot",
    )
    with VLAWorkerSupervisor(config) as supervisor:
        supervisor.spawn()
        print(json.dumps(supervisor.status_snapshot(), indent=2, sort_keys=True))
        ready = supervisor.wait_until_ready()
        print(
            json.dumps(
                {
                    "status": "ready",
                    "worker_pid": ready.worker_pid,
                    "policy_type": ready.policy_type,
                    "action_names": list(ready.action_names),
                    "required_image_keys": list(ready.required_image_keys),
                    "chunk_size": ready.chunk_size,
                    "n_action_steps": ready.n_action_steps,
                    "load_duration_s": ready.load_duration_s,
                    "opens_robot_hardware": False,
                    "launches_orbbec": False,
                    "requires_sudo": False,
                    "log_path": str(config.log_path.expanduser().resolve()),
                },
                indent=2,
                sort_keys=True,
            )
        )
        if args.predict_dataset_index is not None:
            observation = _dataset_observation(
                repo_id=args.dataset_repo_id,
                dataset_root=args.dataset_root,
                index=args.predict_dataset_index,
                required_image_keys=ready.required_image_keys,
            )
            prediction = supervisor.predict(observation)
            print(
                json.dumps(
                    {
                        "status": "prediction_complete",
                        "request_id": prediction.request_id,
                        "action_count": len(prediction.actions),
                        "inference_duration_s": prediction.inference_duration_s,
                        "first_action": prediction.actions[0],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
    print(json.dumps(supervisor.status_snapshot(), indent=2, sort_keys=True))
    return 0


def _dataset_observation(
    *,
    repo_id: str,
    dataset_root: Path,
    index: int,
    required_image_keys: tuple[str, ...],
) -> dict[str, object]:
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id, root=dataset_root.expanduser().resolve())
    item = dataset[index]
    state = item["observation.state"]
    if hasattr(state, "detach"):
        state = state.detach().cpu().numpy()
    observation: dict[str, object] = {
        "observation.state": np.asarray(state, dtype=np.float32).copy(),
    }
    for key in required_image_keys:
        image = item[key]
        if hasattr(image, "detach"):
            image = image.detach().cpu().numpy()
        array = np.asarray(image)
        if array.ndim == 3 and array.shape[0] in (1, 3, 4):
            array = np.moveaxis(array, 0, -1)
        if np.issubdtype(array.dtype, np.floating) and float(array.max()) <= 1.0:
            array = array * 255.0
        observation[key] = np.clip(array, 0, 255).astype(np.uint8)
    return observation


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
