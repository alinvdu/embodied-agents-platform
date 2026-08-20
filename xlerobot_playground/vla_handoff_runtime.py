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

import base64
from dataclasses import dataclass
import io
import math
from pathlib import Path
import threading
import time
from typing import Any, Callable

from xlerobot_agent.vla_policy import RIGHT_ARM_ACTION_NAMES
from xlerobot_agent.vla_worker import VLAWorkerConfig, VLAWorkerSupervisor


_NAV_STOW_ARM_POSE = {
    "shoulder_pan": -4.8316,
    "shoulder_lift": -99.1708,
    "elbow_flex": 100.0,
    "wrist_flex": 76.2061,
    "wrist_roll": 0.1709,
    "gripper": 0.9466,
}


@dataclass(frozen=True)
class VLAHandoffConfig:
    policy_path: Path
    dataset_repo_id: str
    dataset_root: Path
    task: str
    device: str = "mps"
    robot_type: str = "xlerobot_2wheels"
    duration_s: float = 60.0
    max_duration_s: float = 180.0
    fps: float = 30.0
    action_steps: int = 50
    max_joint_delta: float = 50.0
    max_gripper_delta: float = 50.0
    startup_pose: bool = True
    startup_head_pan_deg: float | None = 0.0
    startup_nav_stow_wait_s: float = 5.0
    startup_action_ready_elbow_delta: float = -65.0
    startup_action_ready_shoulder_delta: float = 55.0
    startup_action_ready_wrist_delta: float = -40.0
    startup_pose_steps: int = 40
    startup_pose_stage_delay_s: float = 0.02
    camera_ready_timeout_s: float = 10.0
    worker_startup_timeout_s: float = 180.0
    worker_prediction_timeout_s: float = 60.0
    worker_shutdown_timeout_s: float = 8.0
    worker_log_path: Path = Path("artifacts/vla_worker/smolvla_handoff.log")
    hf_datasets_cache: Path | None = None
    huggingface_offline: bool = True
    release_open_threshold: float = 30.0
    release_closed_threshold: float = 10.0
    release_transition_samples: int = 3
    release_observed_open_samples: int = 2
    release_observed_open_timeout_s: float = 2.0
    release_settle_s: float = 1.0
    release_capture_count: int = 4
    release_capture_interval_s: float = 0.25
    release_capture_jpeg_quality: int = 85


class _GripperReleaseTracker:
    """Recognize the demonstrated open -> grasp-close -> release-open sequence."""

    def __init__(self, *, open_threshold: float, closed_threshold: float, samples: int) -> None:
        if closed_threshold >= open_threshold:
            raise ValueError("release_closed_threshold must be below release_open_threshold")
        self.open_threshold = float(open_threshold)
        self.closed_threshold = float(closed_threshold)
        self.samples = max(1, int(samples))
        self.phase = "waiting_for_first_open"
        self._matching_samples = 0

    def update(self, value: float | None) -> str | None:
        if value is None or not math.isfinite(float(value)):
            self._matching_samples = 0
            return None
        gripper = float(value)
        matches = (
            gripper >= self.open_threshold
            if self.phase in {"waiting_for_first_open", "waiting_for_release_open"}
            else gripper <= self.closed_threshold
        )
        self._matching_samples = self._matching_samples + 1 if matches else 0
        if self._matching_samples < self.samples:
            return None
        self._matching_samples = 0
        if self.phase == "waiting_for_first_open":
            self.phase = "waiting_for_grasp_close"
            return "first_open"
        if self.phase == "waiting_for_grasp_close":
            self.phase = "waiting_for_release_open"
            return "grasp_closed"
        self.phase = "release_detected"
        return "release_detected"


class VLAHandoffRuntime:
    """Execute one on-demand VLA handoff while the robot brain owns hardware."""

    def __init__(
        self,
        *,
        config: VLAHandoffConfig,
        robot_runtime: Any,
        motion_lock: threading.Lock,
        head_frame_provider: Callable[[], Any | None],
        worker_factory: Callable[[VLAWorkerConfig], VLAWorkerSupervisor] = VLAWorkerSupervisor,
    ) -> None:
        self.config = config
        self.robot_runtime = robot_runtime
        self.motion_lock = motion_lock
        self.head_frame_provider = head_frame_provider
        self.worker_factory = worker_factory
        self._state_lock = threading.Lock()
        self._cancel = threading.Event()
        self._active = False
        self._awaiting_verification = False
        self._phase = "idle"
        self._worker: VLAWorkerSupervisor | None = None
        self._last_result: dict[str, Any] | None = None

    @property
    def active(self) -> bool:
        with self._state_lock:
            return self._active or self._awaiting_verification

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            worker = self._worker
            return {
                "enabled": True,
                "active": self._active or self._awaiting_verification,
                "awaiting_verification": self._awaiting_verification,
                "phase": self._phase,
                "worker": None if worker is None else worker.status_snapshot(),
                "last_result": self._last_result,
            }

    def cancel(self) -> None:
        self._cancel.set()
        with self._state_lock:
            if self._awaiting_verification:
                self._awaiting_verification = False
                self._phase = "idle"

    def close(self) -> None:
        self.cancel()
        worker = self._worker
        if worker is not None:
            worker.stop()

    def stow(self) -> dict[str, Any]:
        with self._state_lock:
            if self._active:
                return {
                    "status": "busy",
                    "reason": "Cannot stow while a VLA handoff or another stow motion is active.",
                    "phase": self._phase,
                }
            if not self._awaiting_verification:
                return {
                    "status": "blocked",
                    "reason": "NAV_STOW requires a release awaiting successful basket verification.",
                    "phase": self._phase,
                }
            self._awaiting_verification = False
            self._active = True
            self._phase = "moving_to_nav_stow"
        self._cancel.clear()
        try:
            self._stop_base()
            self._move_to_nav_stow()
            status = "cancelled" if self._cancel.is_set() else "succeeded"
            return self._finish(
                {
                    "status": status,
                    "reason": (
                        "NAV_STOW motion was cancelled."
                        if status == "cancelled"
                        else "Both arms reached the tested NAV_STOW pose."
                    ),
                }
            )
        except Exception as exc:
            return self._finish({"status": "failed", "reason": f"NAV_STOW motion failed: {exc}"})
        finally:
            self._stop_base_best_effort()
            with self._state_lock:
                self._active = False
                self._phase = "idle"

    def run(self, *, duration_s: float | None = None, task: str | None = None) -> dict[str, Any]:
        with self._state_lock:
            if self._active or self._awaiting_verification:
                return {
                    "status": "busy",
                    "reason": "A VLA handoff is already active.",
                    "phase": self._phase,
                }
            self._active = True
            self._phase = "starting_worker"
            self._last_result = None
        self._cancel.clear()

        started_at = time.monotonic()
        actions_sent = 0
        chunks_predicted = 0
        release_detection: dict[str, Any] | None = None
        worker: VLAWorkerSupervisor | None = None
        try:
            requested_duration_s = self.config.duration_s if duration_s is None else float(duration_s)
            rollout_duration_s = max(0.0, min(requested_duration_s, self.config.max_duration_s))
            worker_config = VLAWorkerConfig(
                policy_path=self.config.policy_path,
                dataset_repo_id=self.config.dataset_repo_id,
                dataset_root=self.config.dataset_root,
                task=str(task or self.config.task),
                device=self.config.device,
                robot_type=self.config.robot_type,
                expected_policy_type="smolvla",
                startup_timeout_s=self.config.worker_startup_timeout_s,
                prediction_timeout_s=self.config.worker_prediction_timeout_s,
                shutdown_timeout_s=self.config.worker_shutdown_timeout_s,
                log_path=self.config.worker_log_path,
                hf_datasets_cache=self.config.hf_datasets_cache,
                huggingface_offline=self.config.huggingface_offline,
            )
            worker = self.worker_factory(worker_config)
            with self._state_lock:
                self._worker = worker
            release_tracker = _GripperReleaseTracker(
                open_threshold=self.config.release_open_threshold,
                closed_threshold=self.config.release_closed_threshold,
                samples=self.config.release_transition_samples,
            )
            # Spawn only after this handoff request. Model loading overlaps the safe startup pose.
            worker.spawn()
            self._stop_base()
            if self.config.startup_pose:
                self._set_phase("moving_to_action_ready")
                self._move_to_action_ready()
            self._set_phase("loading_model")
            ready = worker.wait_until_ready(self.config.worker_startup_timeout_s)
            if ready.action_names != RIGHT_ARM_ACTION_NAMES:
                raise RuntimeError(
                    "The VLA handoff only accepts the six-joint right-arm policy contract; "
                    f"checkpoint returned {list(ready.action_names)}."
                )

            self._set_phase("waiting_for_cameras")
            observation = self._wait_for_observation(ready.required_image_keys, ready.action_names)
            worker.reset_policy()
            self._set_phase("running")
            deadline = time.monotonic() + rollout_duration_s
            period_s = 1.0 / max(self.config.fps, 1.0)
            while time.monotonic() < deadline and not self._cancel.is_set():
                prediction = worker.predict(observation)
                chunks_predicted += 1
                action_limit = min(
                    len(prediction.actions),
                    max(1, self.config.action_steps),
                    max(1, ready.n_action_steps),
                )
                for raw_action in prediction.actions[:action_limit]:
                    if time.monotonic() >= deadline or self._cancel.is_set():
                        break
                    loop_started = time.perf_counter()
                    current = self._read_robot_observation(use_camera=False)
                    action = _clamp_right_arm_action(
                        raw_action,
                        current,
                        max_joint_delta=self.config.max_joint_delta,
                        max_gripper_delta=self.config.max_gripper_delta,
                    )
                    with self.motion_lock:
                        self.robot_runtime.robot.send_action(action)
                    actions_sent += 1
                    transition = release_tracker.update(action.get("right_arm_gripper.pos"))
                    if transition == "release_detected":
                        release_detection = {
                            "action_index": actions_sent,
                            "commanded_gripper": action.get("right_arm_gripper.pos"),
                            "open_threshold": self.config.release_open_threshold,
                            "closed_threshold": self.config.release_closed_threshold,
                            "transition_samples": max(1, self.config.release_transition_samples),
                        }
                    elapsed_s = time.perf_counter() - loop_started
                    if elapsed_s < period_s:
                        time.sleep(period_s - elapsed_s)
                    if release_detection is not None:
                        break
                if release_detection is not None:
                    break
                if time.monotonic() < deadline and not self._cancel.is_set():
                    observation = self._wait_for_observation(ready.required_image_keys, ready.action_names)

            release_wrist_images: list[str] = []
            observed_gripper_open: float | None = None
            if self._cancel.is_set():
                status = "cancelled"
                reason = "VLA handoff was cancelled and the worker was stopped."
            elif release_detection is None:
                status = "timed_out"
                reason = "VLA rollout reached its duration limit without a complete grasp-and-release sequence."
            else:
                self._set_phase("confirming_release_open")
                observed_gripper_open = self._wait_for_observed_gripper_open()
                self._set_phase("settling_release")
                if self.config.release_settle_s > 0:
                    self._cancel.wait(self.config.release_settle_s)
                if self._cancel.is_set():
                    status = "cancelled"
                    reason = "VLA handoff was cancelled while the released object was settling."
                else:
                    self._set_phase("capturing_release_evidence")
                    release_wrist_images = self._capture_release_wrist_images()
                    status = "release_detected"
                    reason = (
                        "The second gripper opening was observed and wrist-camera evidence is ready "
                        "for basket verification."
                    )
            result = {
                "status": status,
                "reason": reason,
                "worker_pid": ready.worker_pid,
                "worker_load_duration_s": ready.load_duration_s,
                "rollout_duration_s": max(0.0, time.monotonic() - started_at),
                "chunks_predicted": chunks_predicted,
                "actions_sent": actions_sent,
                "action_names": list(ready.action_names),
                "required_image_keys": list(ready.required_image_keys),
                "release_detection": release_detection,
                "observed_gripper_open": observed_gripper_open,
                "release_wrist_image_count": len(release_wrist_images),
            }
            if release_wrist_images:
                result["release_wrist_images"] = release_wrist_images
            if status == "release_detected":
                with self._state_lock:
                    self._awaiting_verification = True
            return self._finish(result)
        except Exception as exc:
            with self._state_lock:
                self._awaiting_verification = False
            status = "cancelled" if self._cancel.is_set() else "failed"
            return self._finish(
                {
                    "status": status,
                    "reason": (
                        "VLA handoff was cancelled and motion was stopped."
                        if status == "cancelled"
                        else str(exc)
                    ),
                    "rollout_duration_s": max(0.0, time.monotonic() - started_at),
                    "chunks_predicted": chunks_predicted,
                    "actions_sent": actions_sent,
                }
            )
        finally:
            self._stop_base_best_effort()
            self._set_phase("stopping_worker")
            if worker is not None:
                worker.stop()
            with self._state_lock:
                self._worker = None
                self._active = False
                self._phase = "awaiting_verification" if self._awaiting_verification else "idle"

    def _finish(self, result: dict[str, Any]) -> dict[str, Any]:
        with self._state_lock:
            self._last_result = {
                key: value
                for key, value in result.items()
                if key != "release_wrist_images"
            }
        return result

    def _set_phase(self, phase: str) -> None:
        with self._state_lock:
            self._phase = phase

    def _move_to_action_ready(self) -> None:
        if self.config.startup_head_pan_deg is not None:
            self._move_to_joint_targets(
                {"head_motor_1.pos": float(self.config.startup_head_pan_deg)}
            )
        self._move_to_nav_stow()
        if self.config.startup_nav_stow_wait_s > 0:
            self._cancel.wait(self.config.startup_nav_stow_wait_s)
        if self._cancel.is_set():
            return

        current = self._read_robot_observation(use_camera=False)
        elbow_targets: dict[str, float] = {}
        for side in ("left", "right"):
            elbow_key = f"{side}_arm_elbow_flex.pos"
            wrist_key = f"{side}_arm_wrist_flex.pos"
            if elbow_key in current:
                elbow_targets[elbow_key] = float(current[elbow_key]) + self.config.startup_action_ready_elbow_delta
            if wrist_key in current and self.config.startup_action_ready_wrist_delta:
                elbow_targets[wrist_key] = float(current[wrist_key]) + self.config.startup_action_ready_wrist_delta
        self._move_to_joint_targets(elbow_targets)

        current = self._read_robot_observation(use_camera=False)
        shoulder_targets = {
            key: float(current[key]) + self.config.startup_action_ready_shoulder_delta
            for key in ("left_arm_shoulder_lift.pos", "right_arm_shoulder_lift.pos")
            if key in current
        }
        self._move_to_joint_targets(shoulder_targets)

    def _move_to_nav_stow(self) -> None:
        stow_targets = {
            f"{side}_arm_{joint}.pos": value
            for side in ("left", "right")
            for joint, value in _NAV_STOW_ARM_POSE.items()
        }
        self._move_to_joint_targets(stow_targets)

    def _stop_base(self) -> None:
        with self.motion_lock:
            self.robot_runtime.connect()
            self.robot_runtime.robot.send_action({"x.vel": 0.0, "theta.vel": 0.0})

    def _stop_base_best_effort(self) -> None:
        try:
            self._stop_base()
        except Exception:
            pass

    def _move_to_joint_targets(self, targets: dict[str, float]) -> None:
        if not targets or self._cancel.is_set():
            return
        current = self._read_robot_observation(use_camera=False)
        starts = {key: float(current[key]) for key in targets if key in current}
        steps = max(1, self.config.startup_pose_steps)
        for index in range(1, steps + 1):
            if self._cancel.is_set():
                return
            ratio = index / steps
            action = {
                key: start + (float(targets[key]) - start) * ratio
                for key, start in starts.items()
            }
            with self.motion_lock:
                self.robot_runtime.robot.send_action(action)
            if self.config.startup_pose_stage_delay_s > 0:
                self._cancel.wait(self.config.startup_pose_stage_delay_s)

    def _wait_for_observation(
        self,
        required_image_keys: tuple[str, ...],
        action_names: tuple[str, ...],
    ) -> dict[str, Any]:
        deadline = time.monotonic() + max(0.0, self.config.camera_ready_timeout_s)
        last_error = "camera observation unavailable"
        while not self._cancel.is_set():
            try:
                return self._policy_observation(required_image_keys, action_names)
            except (KeyError, RuntimeError, ValueError) as exc:
                last_error = str(exc)
            if time.monotonic() >= deadline:
                raise RuntimeError(f"VLA cameras were not ready: {last_error}")
            self._cancel.wait(0.1)
        raise RuntimeError("VLA handoff was cancelled while waiting for cameras.")

    def _policy_observation(
        self,
        required_image_keys: tuple[str, ...],
        action_names: tuple[str, ...],
    ) -> dict[str, Any]:
        import numpy as np

        raw = self._read_robot_observation(use_camera=True)
        missing_state = [name for name in action_names if name not in raw]
        if missing_state:
            raise KeyError(f"robot observation is missing state joints: {missing_state}")
        observation: dict[str, Any] = {
            "observation.state": np.asarray(
                [float(raw[name]) for name in action_names],
                dtype=np.float32,
            )
        }
        for key in required_image_keys:
            camera_name = key.removeprefix("observation.images.")
            if camera_name == "head":
                frame = self.head_frame_provider()
                if frame is None:
                    raise RuntimeError("Orbbec head RGB frame is not available in robot_brain_agent.")
                image = np.frombuffer(frame.rgb, dtype=np.uint8).reshape(
                    int(frame.rgb_height), int(frame.rgb_width), 3
                )
            else:
                if camera_name not in raw:
                    raise KeyError(
                        f"robot camera {camera_name!r} is unavailable; configure it on robot_brain_agent"
                    )
                image = np.asarray(raw[camera_name])
            observation[key] = np.asarray(image).copy()
        return observation

    def _read_robot_observation(self, *, use_camera: bool) -> dict[str, Any]:
        with self.motion_lock:
            self.robot_runtime.connect()
            getter = self.robot_runtime.robot.get_observation
            try:
                return getter(use_camera=use_camera)
            except TypeError:
                return getter()

    def _wait_for_observed_gripper_open(self) -> float:
        deadline = time.monotonic() + max(0.0, self.config.release_observed_open_timeout_s)
        required_samples = max(1, int(self.config.release_observed_open_samples))
        matching_samples = 0
        latest_value = float("nan")
        while not self._cancel.is_set():
            observation = self._read_robot_observation(use_camera=False)
            latest_value = float(observation.get("right_arm_gripper.pos", float("nan")))
            matching_samples = matching_samples + 1 if latest_value >= self.config.release_open_threshold else 0
            if matching_samples >= required_samples:
                return latest_value
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "The policy commanded release, but the observed right gripper did not reach the "
                    f"open threshold {self.config.release_open_threshold:g}; latest={latest_value:.3f}."
                )
            self._cancel.wait(min(0.05, max(deadline - time.monotonic(), 0.0)))
        raise RuntimeError("VLA handoff was cancelled while confirming the gripper release.")

    def _capture_release_wrist_images(self) -> list[str]:
        images: list[str] = []
        count = max(1, int(self.config.release_capture_count))
        for index in range(count):
            if self._cancel.is_set():
                break
            observation = self._read_robot_observation(use_camera=True)
            image = observation.get("right_wrist")
            if image is None:
                raise RuntimeError("The right-wrist camera is unavailable for basket verification.")
            images.append(
                _image_data_url(
                    image,
                    jpeg_quality=self.config.release_capture_jpeg_quality,
                )
            )
            if index + 1 < count and self.config.release_capture_interval_s > 0:
                self._cancel.wait(self.config.release_capture_interval_s)
        if self._cancel.is_set():
            raise RuntimeError("VLA handoff was cancelled while capturing release evidence.")
        return images


def _clamp_right_arm_action(
    action: dict[str, float],
    observation: dict[str, Any],
    *,
    max_joint_delta: float,
    max_gripper_delta: float,
) -> dict[str, float]:
    unknown = set(action).difference(RIGHT_ARM_ACTION_NAMES)
    if unknown:
        raise RuntimeError(f"VLA attempted to command joints outside the right arm: {sorted(unknown)}")
    clamped: dict[str, float] = {}
    for key, raw_value in action.items():
        value = float(raw_value)
        if not math.isfinite(value):
            raise RuntimeError(f"VLA produced a non-finite action for {key}.")
        if key in observation:
            current = float(observation[key])
            max_delta = max_gripper_delta if key.endswith("_gripper.pos") else max_joint_delta
            if math.isfinite(max_delta) and max_delta >= 0:
                value = max(current - max_delta, min(current + max_delta, value))
        clamped[key] = value
    return clamped


def _image_data_url(image: Any, *, jpeg_quality: int) -> str:
    import numpy as np
    from PIL import Image

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError(f"Expected an HxWx3/4 wrist image, received shape {array.shape}.")
    if np.issubdtype(array.dtype, np.floating):
        maximum = float(np.nanmax(array)) if array.size else 0.0
        if maximum <= 1.0:
            array = array * 255.0
    array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
    array = np.clip(array, 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array[:, :, :3], mode="RGB").save(
        buffer,
        format="JPEG",
        quality=max(40, min(95, int(jpeg_quality))),
        optimize=True,
    )
    return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"
