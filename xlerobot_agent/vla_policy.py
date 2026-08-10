from __future__ import annotations

from contextlib import nullcontext
from copy import copy
import json
from pathlib import Path
from typing import Any


CAMERA_ORDER = ("head", "left_wrist", "right_wrist")
RIGHT_ARM_ACTION_NAMES = (
    "right_arm_shoulder_pan.pos",
    "right_arm_shoulder_lift.pos",
    "right_arm_elbow_flex.pos",
    "right_arm_wrist_flex.pos",
    "right_arm_wrist_roll.pos",
    "right_arm_gripper.pos",
)


def load_policy_stack(
    *,
    policy_path: Path,
    dataset_repo_id: str,
    dataset_root: Path,
    device: str,
    expected_policy_type: str | None = None,
) -> tuple[Any, Any, Any, Any]:
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false.")

    cfg = PreTrainedConfig.from_pretrained(policy_path)
    validate_policy_type(cfg, expected_policy_type)
    ds_meta = LeRobotDatasetMetadata(dataset_repo_id, root=dataset_root)
    validate_policy_dataset_contract(cfg, ds_meta.features)
    camera_rename = policy_camera_rename_map(
        policy_path,
        ds_meta.features,
        policy_input_features=getattr(cfg, "input_features", {}),
    )
    validate_policy_camera_contract(cfg, ds_meta.features, camera_rename)
    print(f"Camera rename map: {camera_rename}")
    cfg.pretrained_path = str(policy_path)
    cfg.device = device
    if getattr(cfg, "pretrained_backbone_weights", None) is not None:
        print(
            "Skipping pretrained vision-backbone initialization; "
            "the local policy checkpoint already contains those weights."
        )
        cfg.pretrained_backbone_weights = None
    policy = make_policy(cfg, ds_meta=ds_meta, rename_map=camera_rename)
    preprocessor, postprocessor = make_pre_post_processors(
        cfg,
        pretrained_path=str(policy_path),
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": camera_rename},
        },
    )
    policy.eval()
    return policy, preprocessor, postprocessor, ds_meta


def predict_action_chunk(
    *,
    observation: dict[str, Any],
    policy: Any,
    device: Any,
    preprocessor: Any,
    postprocessor: Any,
    use_amp: bool,
    task: str,
    robot_type: str,
    ds_features: dict[str, dict],
    action_names: list[str],
) -> list[dict[str, float]]:
    import torch
    from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference

    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        prepared = prepare_observation_for_inference(copy(observation), device, task, robot_type)
        prepared = preprocessor(prepared)
        action_chunk = policy.predict_action_chunk(prepared)
        if action_chunk.ndim == 2:
            action_chunk = action_chunk.unsqueeze(0)
        if action_chunk.ndim != 3 or action_chunk.shape[0] != 1:
            raise RuntimeError(
                "Expected predict_action_chunk() to return shape [1, chunk, action_dim], "
                f"got {tuple(action_chunk.shape)}."
            )

        actions: list[dict[str, float]] = []
        for index in range(action_chunk.shape[1]):
            processed = postprocessor(action_chunk[:, index, :])
            robot_action = make_robot_action(processed, ds_features)
            actions.append(unrename_action(robot_action, action_names))
    return actions


def validate_policy_type(cfg: Any, expected_policy_type: str | None) -> None:
    if expected_policy_type is None:
        return
    actual_policy_type = str(getattr(cfg, "type", "")).lower()
    expected_policy_type = expected_policy_type.lower()
    if actual_policy_type != expected_policy_type:
        raise RuntimeError(
            f"This runner requires a {expected_policy_type!r} checkpoint, "
            f"but {actual_policy_type!r} was found."
        )


def camera_rename_map(ds_features: dict[str, dict]) -> dict[str, str]:
    image_prefix = "observation.images."
    image_keys = {
        key
        for key, feature in ds_features.items()
        if key.startswith(image_prefix) and feature.get("dtype") in {"image", "video"}
    }
    preferred = [
        f"{image_prefix}{camera}"
        for camera in CAMERA_ORDER
        if f"{image_prefix}{camera}" in image_keys
    ]
    ordered = preferred + sorted(image_keys.difference(preferred))
    return {
        source_key: f"{image_prefix}camera{index}"
        for index, source_key in enumerate(ordered, start=1)
    }


def policy_camera_rename_map(
    policy_path: Path,
    ds_features: dict[str, dict],
    policy_input_features: Any = None,
) -> dict[str, str]:
    image_prefix = "observation.images."
    dataset_image_keys = {
        key
        for key, feature in ds_features.items()
        if key.startswith(image_prefix) and feature.get("dtype") in {"image", "video"}
    }
    preprocessor_path = policy_path / "policy_preprocessor.json"
    if preprocessor_path.is_file():
        payload = json.loads(preprocessor_path.read_text())
        for step in payload.get("steps", []):
            if step.get("registry_name") != "rename_observations_processor":
                continue
            saved_map = step.get("config", {}).get("rename_map", {})
            return {
                str(source): str(target)
                for source, target in saved_map.items()
                if source in dataset_image_keys
            }

    policy_image_keys = _policy_image_keys(policy_input_features)
    if not policy_image_keys or policy_image_keys == dataset_image_keys:
        return {}
    compact_map = camera_rename_map(ds_features)
    if set(compact_map.values()) == policy_image_keys:
        return compact_map
    return {}


def validate_policy_camera_contract(
    cfg: Any,
    ds_features: dict[str, dict],
    camera_rename: dict[str, str],
) -> None:
    image_prefix = "observation.images."
    dataset_image_keys = {
        key
        for key, feature in ds_features.items()
        if key.startswith(image_prefix) and feature.get("dtype") in {"image", "video"}
    }
    expected_image_keys = _policy_image_keys(getattr(cfg, "input_features", {}))
    runtime_image_keys = {
        camera_rename.get(source_key, source_key)
        for source_key in dataset_image_keys
    }
    policy_type = str(getattr(cfg, "type", "")).lower()
    unexpected_runtime_keys = runtime_image_keys.difference(expected_image_keys)
    missing_expected_keys = expected_image_keys.difference(runtime_image_keys)
    allow_unused_policy_slots = policy_type == "smolvla"
    if expected_image_keys and (
        unexpected_runtime_keys or (missing_expected_keys and not allow_unused_policy_slots)
    ):
        raise RuntimeError(
            "Checkpoint camera inputs do not match the dataset after applying its saved rename map. "
            f"Checkpoint={sorted(expected_image_keys)}, runtime={sorted(runtime_image_keys)}, "
            f"rename_map={camera_rename}."
        )


def validate_policy_dataset_contract(cfg: Any, ds_features: dict[str, dict]) -> None:
    action_names = list(ds_features.get("action", {}).get("names") or [])
    state_names = list(ds_features.get("observation.state", {}).get("names") or [])
    if not action_names or not state_names:
        raise RuntimeError("Dataset metadata must include named action and observation.state features.")

    expected_action_dim = _policy_feature_dim(getattr(cfg, "output_features", {}), "action")
    expected_state_dim = _policy_feature_dim(
        getattr(cfg, "input_features", {}),
        "observation.state",
    )
    if expected_action_dim is not None and expected_action_dim != len(action_names):
        raise RuntimeError(
            f"Checkpoint expects {expected_action_dim} actions, but dataset metadata defines "
            f"{len(action_names)}. Use the dataset that was used to train this checkpoint."
        )
    if expected_state_dim is not None and expected_state_dim != len(state_names):
        raise RuntimeError(
            f"Checkpoint expects {expected_state_dim} state values, but dataset metadata defines "
            f"{len(state_names)}. Use the dataset that was used to train this checkpoint."
        )
    if action_names != state_names:
        raise RuntimeError(
            "This runtime requires action and observation.state names to match in the same order. "
            f"Actions={action_names}, state={state_names}."
        )
    if len(action_names) == len(RIGHT_ARM_ACTION_NAMES) and tuple(action_names) != RIGHT_ARM_ACTION_NAMES:
        raise RuntimeError(
            "The six-dimensional checkpoint must use the canonical right-arm joint order. "
            f"Expected={list(RIGHT_ARM_ACTION_NAMES)}, got={action_names}."
        )


def unrename_action(action: dict[str, float], action_names: list[str]) -> dict[str, float]:
    return {name: float(action[name]) for name in action_names if name in action}


def required_dataset_image_keys(ds_features: dict[str, dict]) -> tuple[str, ...]:
    prefix = "observation.images."
    return tuple(
        key
        for key, feature in ds_features.items()
        if key.startswith(prefix) and feature.get("dtype") in {"image", "video"}
    )


def _policy_image_keys(policy_input_features: Any) -> set[str]:
    if not policy_input_features:
        return set()
    return {
        str(key)
        for key in policy_input_features
        if str(key).startswith("observation.images.")
    }


def _policy_feature_dim(features: Any, key: str) -> int | None:
    feature = features.get(key) if features else None
    shape = getattr(feature, "shape", None)
    if shape is None and isinstance(feature, dict):
        shape = feature.get("shape")
    if not shape:
        return None
    return int(shape[0])
