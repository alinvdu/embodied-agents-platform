from __future__ import annotations

import builtins
from contextlib import contextmanager
from dataclasses import dataclass, field
import math
from collections.abc import Iterator
from typing import Any, Sequence

from multido_xlerobot import XLeRobotInterface
from multido_xlerobot.bootstrap import resolve_xlerobot_repo_root
from xlerobot_agent.exploration import Pose2D
from xlerobot_playground.exploration_runtime import (
    NavigationDependency,
    NavigationResult,
    RGBDObservation,
    ScanObservation,
)


@dataclass(frozen=True)
class RealXLeRobotRuntimeConfig:
    repo_root: str = str(resolve_xlerobot_repo_root())
    robot_kind: str = "xlerobot_2wheels"
    port1: str = "/dev/tty.usbmodem5B140330101"
    port2: str = "/dev/tty.usbmodem5B140332271"
    fps: int = 30
    use_degrees: bool = False
    camera_frame: str = "head_camera_link"
    allow_motion_commands: bool = False
    max_linear_m_s: float = 0.20
    max_angular_rad_s: float = 0.50
    debug_motion: bool = False
    calibration_prompt_response: str | None = ""
    cameras: dict[str, Any] = field(default_factory=dict)


class RealXLeRobotDirectRuntime:
    """Direct real-XLeRobot exploration embodiment.

    This class is intentionally a safe skeleton. It names the same contract as
    the ManiSkill runtime, while leaving Orbbec RGB-D, pose tracking, and base
    command conversion unimplemented until those pieces can be tested on
    hardware with a deadman stop.
    """

    def __init__(
        self,
        config: RealXLeRobotRuntimeConfig | None = None,
        *,
        navigation: NavigationDependency | None = None,
        robot: Any | None = None,
    ) -> None:
        self.config = config or RealXLeRobotRuntimeConfig()
        self.navigation = navigation
        self._robot = robot
        self._connected = False

    @property
    def name(self) -> str:
        return "real_xlerobot_direct"

    @property
    def robot(self) -> Any:
        if self._robot is None:
            self._debug("building robot interface")
            interface = XLeRobotInterface(self.config.repo_root)
            if self.config.robot_kind == "xlerobot_2wheels":
                config_cls, robot_cls = interface.robot_2wheels_classes()
            else:
                config_cls, robot_cls = interface.robot_classes()
            self._robot = robot_cls(
                config_cls(
                    port1=self.config.port1,
                    port2=self.config.port2,
                    cameras=self.config.cameras,
                    use_degrees=self.config.use_degrees,
                )
            )
            self._debug("robot interface built")
        return self._robot

    def connect(self) -> None:
        if self._connected:
            return
        try:
            self._debug("connect start")
            with self._calibration_input_context():
                self.robot.connect()
            self._debug("connect ok")
        except Exception as exc:
            if "already connected" not in str(exc).lower():
                self._debug(f"connect failed: {exc}")
                raise
            self._debug("connect recovered from already-connected motor bus")
        self._connected = True

    def close(self) -> None:
        if self._robot is None:
            return
        if not self._connected:
            return
        self._debug("disconnect start")
        self._robot.disconnect()
        self._debug("disconnect ok")
        self._connected = False

    def reset(self) -> None:
        raise NotImplementedError("Real reset/home behavior is not implemented yet.")

    def current_pose(self) -> Pose2D:
        raise NotImplementedError("Real pose tracking requires odometry or localization integration.")

    def capture_rgbd(self) -> RGBDObservation:
        raise NotImplementedError("Real RGB-D capture requires Orbbec depth integration.")

    def latest_scan(self) -> ScanObservation | None:
        raise NotImplementedError("Real scan requires Orbbec depth-to-scan integration.")

    def drive_velocity(self, *, linear_m_s: float, angular_rad_s: float) -> NavigationResult:
        if not self.config.allow_motion_commands:
            return NavigationResult(
                succeeded=False,
                message="Real motion commands are disabled. Set allow_motion_commands=True after hardware checks.",
                metadata={
                    "requested_linear_m_s": linear_m_s,
                    "requested_angular_rad_s": angular_rad_s,
                },
            )
        self.connect()
        action = self._base_velocity_action(linear_m_s=linear_m_s, angular_rad_s=angular_rad_s)
        self._debug(f"send_action start: {action}")
        sent = self.robot.send_action(action)
        self._debug(f"send_action ok: {sent}")
        return NavigationResult(
            succeeded=True,
            message="Velocity command sent to real XLeRobot base.",
            metadata={"requested_action": action, "sent_action": sent},
        )

    def stop(self) -> NavigationResult:
        if not self._connected:
            return NavigationResult(succeeded=True, message="Robot is not connected; nothing to stop.")
        action = self._base_velocity_action(linear_m_s=0.0, angular_rad_s=0.0)
        self._debug(f"zero stop send_action start: {action}")
        sent = self.robot.send_action(action)
        self._debug(f"zero stop send_action ok: {sent}")
        return NavigationResult(
            succeeded=True,
            message="Zero velocity command sent to real XLeRobot base.",
            metadata={"sent_action": sent},
        )

    def rotate_in_place(self, yaw_delta_rad: float) -> NavigationResult:
        raise NotImplementedError("Real rotate-in-place requires the base velocity adapter and odometry feedback.")

    def execute_path(self, path: Sequence[Pose2D]) -> NavigationResult:
        raise NotImplementedError("Real path execution requires navigation dependency wiring and command safety.")

    def _base_velocity_action(self, *, linear_m_s: float, angular_rad_s: float) -> dict[str, float]:
        linear = max(-self.config.max_linear_m_s, min(self.config.max_linear_m_s, float(linear_m_s)))
        angular = max(-self.config.max_angular_rad_s, min(self.config.max_angular_rad_s, float(angular_rad_s)))
        action = {
            "x.vel": linear,
            "theta.vel": math.degrees(angular),
        }
        if self.config.robot_kind != "xlerobot_2wheels":
            action["y.vel"] = 0.0
        return action

    def _debug(self, message: str) -> None:
        if self.config.debug_motion:
            print(f"[real_xlerobot_direct] {message}", flush=True)

    @contextmanager
    def _calibration_input_context(self) -> Iterator[None]:
        if self.config.calibration_prompt_response is None:
            yield
            return
        original_input = builtins.input

        def auto_input(prompt: str = "") -> str:
            if prompt:
                print(prompt, end="", flush=True)
            response = self.config.calibration_prompt_response or ""
            print(response, flush=True)
            return response

        builtins.input = auto_input
        try:
            yield
        finally:
            builtins.input = original_input
