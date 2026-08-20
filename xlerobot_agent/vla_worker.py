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

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
import gc
import math
import multiprocessing
from multiprocessing.connection import Connection
import os
from pathlib import Path
import threading
import time
import traceback
from typing import Any
from uuid import uuid4

from .vla_policy import (
    RIGHT_ARM_ACTION_NAMES,
    load_policy_stack,
    predict_action_chunk,
    required_dataset_image_keys,
)


@dataclass(frozen=True)
class VLAWorkerConfig:
    policy_path: Path
    dataset_repo_id: str
    dataset_root: Path
    task: str
    device: str = "mps"
    robot_type: str = "xlerobot_2wheels"
    expected_policy_type: str | None = "smolvla"
    startup_timeout_s: float = 180.0
    prediction_timeout_s: float = 60.0
    shutdown_timeout_s: float = 8.0
    log_path: Path = Path("artifacts/vla_worker/smolvla_worker.log")
    hf_datasets_cache: Path | None = None
    huggingface_offline: bool = True
    backend: str = "lerobot"
    mock_chunk_size: int = 3
    mock_load_delay_s: float = 0.0
    mock_prediction_delay_s: float = 0.0
    mock_fail_load: bool = False


@dataclass(frozen=True)
class VLAWorkerReady:
    worker_pid: int
    policy_type: str
    action_names: tuple[str, ...]
    required_image_keys: tuple[str, ...]
    chunk_size: int
    n_action_steps: int
    load_duration_s: float


@dataclass(frozen=True)
class VLAWorkerPrediction:
    request_id: str
    actions: tuple[dict[str, float], ...]
    inference_duration_s: float


class VLAWorkerError(RuntimeError):
    pass


class VLAWorkerStartError(VLAWorkerError):
    pass


class VLAWorkerPredictionError(VLAWorkerError):
    pass


class VLAWorkerSupervisor:
    """Own an on-demand model process that never opens robot or camera hardware."""

    def __init__(self, config: VLAWorkerConfig) -> None:
        self.config = config
        self._context = multiprocessing.get_context("spawn")
        self._process: multiprocessing.Process | None = None
        self._connection: Connection | None = None
        self._ready: VLAWorkerReady | None = None
        self._state = "stopped"
        self._last_error: str | None = None
        self._lock = threading.RLock()

    @property
    def state(self) -> str:
        with self._lock:
            self._refresh_dead_process()
            return self._state

    @property
    def ready_info(self) -> VLAWorkerReady | None:
        with self._lock:
            return self._ready

    @property
    def worker_pid(self) -> int | None:
        with self._lock:
            return self._process.pid if self._process is not None else None

    @property
    def start_method(self) -> str:
        return self._context.get_start_method()

    def status_snapshot(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_dead_process()
            return {
                "state": self._state,
                "worker_pid": self.worker_pid,
                "start_method": self.start_method,
                "log_path": str(self.config.log_path.expanduser().resolve()),
                "last_error": self._last_error,
                "ready": asdict(self._ready) if self._ready is not None else None,
            }

    def spawn(self) -> None:
        """Start loading in a fresh process and return without waiting for READY."""
        with self._lock:
            self._refresh_dead_process()
            if self._process is not None and self._process.is_alive():
                return
            self._cleanup_handles()
            log_path = self.config.log_path.expanduser().resolve()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            parent_connection, child_connection = self._context.Pipe(duplex=True)
            process = self._context.Process(
                target=_vla_worker_process_main,
                args=(self.config, child_connection),
                name="robot42-vla-worker",
                daemon=False,
            )
            process.start()
            child_connection.close()
            self._process = process
            self._connection = parent_connection
            self._ready = None
            self._last_error = None
            self._state = "starting"

    def wait_until_ready(self, timeout_s: float | None = None) -> VLAWorkerReady:
        with self._lock:
            if self._process is None:
                self.spawn()
            if self._ready is not None and self._process is not None and self._process.is_alive():
                return self._ready
            timeout = self.config.startup_timeout_s if timeout_s is None else timeout_s
            message = self._receive_message(timeout, phase="startup")
            if message.get("type") == "ready":
                self._ready = _ready_from_message(message)
                self._state = "ready"
                return self._ready
            self._fail_from_message(message, VLAWorkerStartError)
            raise AssertionError("unreachable")

    def ensure_ready(self, timeout_s: float | None = None) -> VLAWorkerReady:
        with self._lock:
            self.spawn()
            return self.wait_until_ready(timeout_s)

    def predict(
        self,
        observation: dict[str, Any],
        *,
        timeout_s: float | None = None,
    ) -> VLAWorkerPrediction:
        with self._lock:
            ready = self.ensure_ready()
            connection = self._require_connection()
            request_id = uuid4().hex
            self._state = "predicting"
            try:
                connection.send(
                    {
                        "type": "predict",
                        "request_id": request_id,
                        "observation": observation,
                    }
                )
            except Exception as exc:
                self._fail_and_stop(f"Could not send prediction request: {exc}")
                raise VLAWorkerPredictionError(self._last_error) from exc

            timeout = self.config.prediction_timeout_s if timeout_s is None else timeout_s
            message = self._receive_message(timeout, phase="prediction")
            if message.get("type") == "prediction" and message.get("request_id") == request_id:
                actions = _validated_actions(message.get("actions"), ready.action_names)
                prediction = VLAWorkerPrediction(
                    request_id=request_id,
                    actions=tuple(actions),
                    inference_duration_s=float(message.get("inference_duration_s") or 0.0),
                )
                self._state = "ready"
                return prediction
            self._fail_from_message(message, VLAWorkerPredictionError)
            raise AssertionError("unreachable")

    def reset_policy(self, *, timeout_s: float = 10.0) -> None:
        with self._lock:
            self.ensure_ready()
            connection = self._require_connection()
            connection.send({"type": "reset"})
            message = self._receive_message(timeout_s, phase="reset")
            if message.get("type") != "reset_complete":
                self._fail_from_message(message, VLAWorkerError)

    def stop(self) -> None:
        with self._lock:
            process = self._process
            connection = self._connection
            if process is None:
                self._state = "stopped"
                self._cleanup_handles()
                return
            if process.is_alive() and connection is not None:
                try:
                    connection.send({"type": "shutdown"})
                except Exception:
                    pass
            process.join(timeout=max(0.0, self.config.shutdown_timeout_s))
            if process.is_alive():
                process.terminate()
                process.join(timeout=3.0)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(timeout=3.0)
            self._cleanup_handles()
            self._state = "stopped"
            self._ready = None

    def __enter__(self) -> "VLAWorkerSupervisor":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.stop()

    def _receive_message(self, timeout_s: float, *, phase: str) -> dict[str, Any]:
        connection = self._require_connection()
        process = self._process
        deadline = time.monotonic() + max(0.0, timeout_s)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                message = f"VLA worker {phase} timed out after {timeout_s:.1f}s."
                self._fail_and_stop(message)
                error_cls = VLAWorkerStartError if phase == "startup" else VLAWorkerPredictionError
                raise error_cls(message)
            if connection.poll(min(remaining, 0.1)):
                try:
                    message = connection.recv()
                except EOFError as exc:
                    self._refresh_dead_process()
                    detail = self._last_error or f"VLA worker exited during {phase}."
                    raise VLAWorkerError(detail) from exc
                if not isinstance(message, dict):
                    self._fail_and_stop(f"VLA worker returned malformed {phase} response.")
                    raise VLAWorkerError(self._last_error)
                return message
            if process is not None and not process.is_alive():
                exit_code = process.exitcode
                self._last_error = (
                    f"VLA worker exited during {phase} with code {exit_code}. "
                    f"See {self.config.log_path.expanduser().resolve()}."
                )
                self._state = "failed"
                raise VLAWorkerError(self._last_error)

    def _fail_from_message(self, message: dict[str, Any], error_cls: type[VLAWorkerError]) -> None:
        detail = str(message.get("error") or f"Unexpected VLA worker response: {message.get('type')}")
        worker_traceback = str(message.get("traceback") or "").strip()
        if worker_traceback:
            detail = f"{detail}\n{worker_traceback}"
        self._fail_and_stop(detail)
        raise error_cls(detail)

    def _fail_and_stop(self, detail: str) -> None:
        self._last_error = detail
        process = self._process
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=3.0)
        self._cleanup_handles()
        self._ready = None
        self._state = "failed"

    def _refresh_dead_process(self) -> None:
        if self._process is None or self._process.is_alive():
            return
        if self._state not in {"stopped", "failed"}:
            self._last_error = (
                f"VLA worker exited unexpectedly with code {self._process.exitcode}. "
                f"See {self.config.log_path.expanduser().resolve()}."
            )
            self._state = "failed"
        self._cleanup_handles()
        self._ready = None

    def _require_connection(self) -> Connection:
        if self._connection is None:
            raise VLAWorkerError("VLA worker connection is not available.")
        return self._connection

    def _cleanup_handles(self) -> None:
        if self._connection is not None:
            try:
                self._connection.close()
            except Exception:
                pass
        if self._process is not None:
            try:
                self._process.close()
            except (ValueError, AttributeError):
                pass
        self._connection = None
        self._process = None


class _LeRobotWorkerRuntime:
    def __init__(self, config: VLAWorkerConfig) -> None:
        policy, preprocessor, postprocessor, ds_meta = load_policy_stack(
            policy_path=config.policy_path.expanduser().resolve(),
            dataset_repo_id=config.dataset_repo_id,
            dataset_root=config.dataset_root.expanduser().resolve(),
            device=config.device,
            expected_policy_type=config.expected_policy_type,
        )
        self.config = config
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.ds_features = ds_meta.features
        self.action_names = tuple(self.ds_features["action"]["names"])
        self.required_image_keys = required_dataset_image_keys(self.ds_features)
        self.device = next(policy.parameters()).device
        self.use_amp = bool(getattr(policy.config, "use_amp", False))
        self.policy_type = str(getattr(policy.config, "type", "unknown"))
        self.chunk_size = max(1, int(getattr(policy.config, "chunk_size", 1)))
        self.n_action_steps = max(1, int(getattr(policy.config, "n_action_steps", self.chunk_size)))
        self.reset()

    def predict(self, observation: dict[str, Any]) -> list[dict[str, float]]:
        missing = [
            key
            for key in ("observation.state", *self.required_image_keys)
            if key not in observation
        ]
        if missing:
            raise ValueError(f"VLA observation is missing required features: {missing}")
        return predict_action_chunk(
            observation=observation,
            policy=self.policy,
            device=self.device,
            preprocessor=self.preprocessor,
            postprocessor=self.postprocessor,
            use_amp=self.use_amp,
            task=self.config.task,
            robot_type=self.config.robot_type,
            ds_features=self.ds_features,
            action_names=list(self.action_names),
        )

    def reset(self) -> None:
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()

    def ready_payload(self, *, worker_pid: int, load_duration_s: float) -> dict[str, Any]:
        return {
            "type": "ready",
            "worker_pid": worker_pid,
            "policy_type": self.policy_type,
            "action_names": list(self.action_names),
            "required_image_keys": list(self.required_image_keys),
            "chunk_size": self.chunk_size,
            "n_action_steps": self.n_action_steps,
            "load_duration_s": load_duration_s,
        }


class _MockWorkerRuntime:
    def __init__(self, config: VLAWorkerConfig) -> None:
        if config.mock_load_delay_s > 0:
            time.sleep(config.mock_load_delay_s)
        if config.mock_fail_load:
            raise RuntimeError("Requested mock VLA load failure.")
        self.config = config
        self.action_names = RIGHT_ARM_ACTION_NAMES

    def predict(self, observation: dict[str, Any]) -> list[dict[str, float]]:
        if self.config.mock_prediction_delay_s > 0:
            time.sleep(self.config.mock_prediction_delay_s)
        state = observation.get("observation.state", [0.0] * len(self.action_names))
        if hasattr(state, "detach"):
            state = state.detach().cpu()
        if hasattr(state, "tolist"):
            state = state.tolist()
        if state and isinstance(state[0], list):
            state = state[0]
        values = [float(value) for value in state]
        values.extend([0.0] * (len(self.action_names) - len(values)))
        action = dict(zip(self.action_names, values, strict=False))
        return [dict(action) for _ in range(max(1, self.config.mock_chunk_size))]

    def reset(self) -> None:
        return

    def ready_payload(self, *, worker_pid: int, load_duration_s: float) -> dict[str, Any]:
        return {
            "type": "ready",
            "worker_pid": worker_pid,
            "policy_type": "mock_smolvla",
            "action_names": list(self.action_names),
            "required_image_keys": [],
            "chunk_size": max(1, self.config.mock_chunk_size),
            "n_action_steps": max(1, self.config.mock_chunk_size),
            "load_duration_s": load_duration_s,
        }


def _vla_worker_process_main(config: VLAWorkerConfig, connection: Connection) -> None:
    log_path = config.log_path.expanduser().resolve()
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8", buffering=1) as log_stream:
            with redirect_stdout(log_stream), redirect_stderr(log_stream):
                _run_vla_worker(config, connection)
    except BaseException as exc:
        _safe_send(
            connection,
            {
                "type": "error",
                "phase": "load",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
    finally:
        try:
            connection.close()
        except Exception:
            pass


def _run_vla_worker(config: VLAWorkerConfig, connection: Connection) -> None:
    if config.hf_datasets_cache is not None:
        os.environ["HF_DATASETS_CACHE"] = str(config.hf_datasets_cache.expanduser().resolve())
    if config.huggingface_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    print(f"Starting Robot42 VLA worker pid={os.getpid()} backend={config.backend} device={config.device}")
    load_started = time.perf_counter()
    if config.backend == "mock":
        runtime: Any = _MockWorkerRuntime(config)
    elif config.backend == "lerobot":
        runtime = _LeRobotWorkerRuntime(config)
    else:
        raise ValueError(f"Unknown VLA worker backend: {config.backend}")
    load_duration_s = time.perf_counter() - load_started
    connection.send(runtime.ready_payload(worker_pid=os.getpid(), load_duration_s=load_duration_s))
    print(f"VLA worker READY after {load_duration_s:.3f}s")

    try:
        while True:
            try:
                message = connection.recv()
            except EOFError:
                print("Parent connection closed; stopping VLA worker.")
                break
            if not isinstance(message, dict):
                _safe_send(connection, {"type": "error", "error": "Malformed worker command."})
                continue
            command = message.get("type")
            if command == "shutdown":
                _safe_send(connection, {"type": "stopped"})
                break
            if command == "reset":
                try:
                    runtime.reset()
                    connection.send({"type": "reset_complete"})
                except Exception as exc:
                    connection.send(
                        {
                            "type": "error",
                            "phase": "reset",
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                continue
            if command != "predict":
                connection.send({"type": "error", "error": f"Unknown worker command: {command}"})
                continue
            request_id = str(message.get("request_id") or "")
            try:
                started = time.perf_counter()
                actions = runtime.predict(message.get("observation") or {})
                connection.send(
                    {
                        "type": "prediction",
                        "request_id": request_id,
                        "actions": actions,
                        "inference_duration_s": time.perf_counter() - started,
                    }
                )
            except Exception as exc:
                connection.send(
                    {
                        "type": "error",
                        "phase": "prediction",
                        "request_id": request_id,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
    finally:
        del runtime
        gc.collect()
        try:
            import torch

            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass
        print("VLA worker stopped and model resources released.")


def _ready_from_message(message: dict[str, Any]) -> VLAWorkerReady:
    return VLAWorkerReady(
        worker_pid=int(message["worker_pid"]),
        policy_type=str(message["policy_type"]),
        action_names=tuple(str(item) for item in message.get("action_names", [])),
        required_image_keys=tuple(str(item) for item in message.get("required_image_keys", [])),
        chunk_size=max(1, int(message.get("chunk_size") or 1)),
        n_action_steps=max(1, int(message.get("n_action_steps") or 1)),
        load_duration_s=max(0.0, float(message.get("load_duration_s") or 0.0)),
    )


def _validated_actions(raw_actions: Any, action_names: tuple[str, ...]) -> list[dict[str, float]]:
    if not isinstance(raw_actions, list) or not raw_actions:
        raise VLAWorkerPredictionError("VLA worker returned an empty action chunk.")
    allowed = set(action_names)
    actions: list[dict[str, float]] = []
    for index, raw_action in enumerate(raw_actions):
        if not isinstance(raw_action, dict):
            raise VLAWorkerPredictionError(f"VLA action {index} is not a mapping.")
        unknown = set(raw_action).difference(allowed)
        if unknown:
            raise VLAWorkerPredictionError(f"VLA action {index} contains unknown keys: {sorted(unknown)}")
        action: dict[str, float] = {}
        for key, raw_value in raw_action.items():
            value = float(raw_value)
            if not math.isfinite(value):
                raise VLAWorkerPredictionError(f"VLA action {index} contains non-finite value for {key}.")
            action[key] = value
        actions.append(action)
    return actions


def _safe_send(connection: Connection, message: dict[str, Any]) -> None:
    try:
        connection.send(message)
    except Exception:
        pass
