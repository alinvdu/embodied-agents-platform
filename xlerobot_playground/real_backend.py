from __future__ import annotations

import argparse
import asyncio
import builtins
import importlib
import importlib.util
import json
import math
import os
import ssl
import threading
import subprocess
import sys
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from multido_xlerobot import XLeRobotInterface
from multido_xlerobot.bootstrap import bootstrap_xlerobot, resolve_xlerobot_repo_root


@dataclass(frozen=True)
class CameraSpec:
    name: str
    driver: str
    source: str
    width: int | None = None
    height: int | None = None
    fps: int | None = None
    fourcc: str | None = None
    backend: str | None = None


@dataclass
class RecordingSession:
    dataset: Any
    task: str
    dataset_root: Path | None = None
    active: bool = False
    session_episode_count: int = 0
    orbbec_output_dir: Path | None = None
    orbbec_camera_key: str = "head"
    latest_observation_images: dict[str, tuple[Any, float]] = field(default_factory=dict)
    last_missing_features: tuple[str, ...] = ()
    last_missing_feature_warn_t: float = 0.0
    episode_frame_count: int = 0


@dataclass(frozen=True)
class OrbbecRgbConfig:
    enabled: bool
    launch_capture: bool
    capture_bin: Path
    output_dir: Path
    width: int
    height: int
    fps: int
    timeout_ms: int
    log_every: int


@dataclass
class BaseSmoother:
    max_linear: float
    max_angular: float
    linear_accel: float
    angular_accel: float
    deadzone: float
    curve: float
    x_vel: float = 0.0
    theta_vel: float = 0.0
    last_t: float | None = None


@dataclass(frozen=True)
class VrArmTuning:
    ik_mode: str
    vertical_sign: float
    y_gain: float
    z_gain: float
    ik_alpha: float
    yawed_forward_gain: float
    yawed_lateral_gain: float
    yawed_pan_sign: float
    yawed_pan_limit: float
    yawed_pan_step_limit: float
    shoulder_lift_min: float
    shoulder_lift_max: float
    elbow_flex_min: float
    elbow_flex_max: float
    enforce_joint_limits: bool
    debug: bool
    debug_hz: float


@dataclass(frozen=True)
class VrVideoDisplayConfig:
    wrist_gain: float
    wrist_gamma: float
    wrist_bias: float
    orbbec_gain: float
    orbbec_gamma: float
    orbbec_bias: float


@dataclass(frozen=True)
class VrStartupPoseConfig:
    enabled: bool
    stow_wait_s: float
    stow_pose: dict[str, float]
    action_ready_elbow_delta: float
    action_ready_shoulder_delta: float
    action_ready_wrist_delta: float
    steps_per_stage: int
    stage_delay_s: float


_VR_CAMERA_FRAMES: dict[str, tuple[bytes, int, int, float]] = {}
_VR_CAMERA_FRAMES_LOCK = threading.Lock()
_VR_CAMERA_JPEGS: dict[str, tuple[bytes, int, int, float]] = {}
_VR_CAMERA_JPEGS_LOCK = threading.Lock()
_ORBBEC_JPEG_CACHE: tuple[Path, int, int, bytes, float] | None = None
_ORBBEC_JPEG_CACHE_LOCK = threading.Lock()
_VR_SESSION_STATUS: dict[str, Any] = {
    "recording_enabled": False,
    "recording_active": False,
    "recording_missing": [],
    "episode_frame_count": 0,
    "session_episode_count": 0,
    "episode_phase": "idle",
    "held_sides": [],
    "finish_requested": False,
    "menu_open": False,
    "menu_pointer_hand": "left",
    "recording_operation": "",
}
_VR_SESSION_STATUS_LOCK = threading.Lock()
_VR_FINISH_REQUESTED = False
_VR_FINISH_REQUEST_LOCK = threading.Lock()
_VR_MENU_OPEN = False
_VR_MENU_LOCK = threading.Lock()
_VR_RECORDING_CONTROL_REQUESTS: list[str] = []
_VR_RECORDING_CONTROL_LOCK = threading.Lock()
_VR_RECORDING_CONTROL_LAST_ACTION: tuple[str, float] | None = None
_VR_RECORDING_CONTROL_DEBOUNCE_S = 0.45
_RECORDING_IMAGE_CACHE_MAX_AGE_S = 1.0
_VR_CAMERA_ASYNC_READ_TIMEOUT_MS = 500.0
_VR_CAMERA_RECONNECT_MIN_INTERVAL_S = 3.0
_VR_CAMERA_RECONNECT_LAST_ATTEMPT: dict[str, float] = {}
_WEBRTC_LOOP: Any = None
_WEBRTC_LOOP_THREAD: threading.Thread | None = None
_WEBRTC_PEERS: set[Any] = set()
_WEBRTC_PEERS_LOCK = threading.Lock()
_JPEG_QUALITY = 85
_MJPEG_BOUNDARY = b"frame"
_NAV_STOW_ARM_POSE = {
    # Captured from the physical robot's folded right arm on 2026-06-14.
    # Applied to both arms because the captured left/right folded poses were nearly identical.
    "shoulder_pan": -4.8316,
    "shoulder_lift": -99.1708,
    "elbow_flex": 100.0,
    "wrist_flex": 76.2061,
    "wrist_roll": 0.1709,
    "gripper": 0.9466,
}
_ACTION_READY_ELBOW_DELTA = -65.0
_ACTION_READY_SHOULDER_DELTA = 55.0
_ACTION_READY_WRIST_DELTA = -40.0
_VR_ARM_CLUTCH_RELEASE_HOLD_FRAMES = 3
_XLEVR_ORIGINAL_PRINT: Any | None = None
_VR_BASKET_POSE_DEFAULTS = {
    # Captured from physical basket-placement poses on 2026-06-21.
    "left_arm_shoulder_pan.pos": 17.5510,
    "left_arm_shoulder_lift.pos": -38.5593,
    "left_arm_elbow_flex.pos": 72.3118,
    "left_arm_wrist_flex.pos": 64.8387,
    "left_arm_wrist_roll.pos": -5.9111,
    "left_arm_gripper.pos": 1.3889,
    "right_arm_shoulder_pan.pos": -26.8668,
    "right_arm_shoulder_lift.pos": -42.3715,
    "right_arm_elbow_flex.pos": 84.8226,
    "right_arm_wrist_flex.pos": 44.9714,
    "right_arm_wrist_roll.pos": -2.9548,
    "right_arm_gripper.pos": 6.4909,
}
_RIGHT_BASKET_PATH_REFERENCE_GRASP = {
    "right_arm_shoulder_pan.pos": -10.1757,
    "right_arm_shoulder_lift.pos": 15.6716,
    "right_arm_elbow_flex.pos": -25.3705,
    "right_arm_wrist_flex.pos": 35.4865,
    "right_arm_wrist_roll.pos": -2.7595,
}
_RIGHT_BASKET_PATH_CLEARANCE = {
    "right_arm_shoulder_pan.pos": -10.2489,
    "right_arm_shoulder_lift.pos": -47.4295,
    "right_arm_elbow_flex.pos": 9.7441,
    "right_arm_wrist_flex.pos": 54.1292,
    "right_arm_wrist_roll.pos": -3.0525,
}
_RIGHT_BASKET_PATH_OVER_BASKET = {
    "right_arm_shoulder_pan.pos": -23.7921,
    "right_arm_shoulder_lift.pos": -87.8109,
    "right_arm_elbow_flex.pos": 86.2595,
    "right_arm_wrist_flex.pos": 54.1292,
    "right_arm_wrist_roll.pos": -3.0037,
}


@dataclass(frozen=True)
class VRRecordingControls:
    toggle_recording: bool = False
    discard_episode: bool = False
    quit_session: bool = False
    reset_robot: bool = False


@dataclass(frozen=True)
class VRRecordingDecision:
    start_recording: bool = False
    save_episode: bool = False
    discard_episode: bool = False
    quit_session: bool = False
    reset_robot: bool = False


@dataclass(frozen=True)
class VrArmClutchKeys:
    left: str
    right: str


@dataclass(frozen=True)
class VrBasketPoseConfig:
    enabled: bool
    skill_arm: str
    full_reset_key: str
    right_basket_key: str
    right_action_key: str
    left_basket_key: str
    left_action_key: str
    right_basket_button: str
    right_action_button: str
    left_basket_button: str
    left_action_button: str
    targets: dict[str, float]
    release_gripper: float
    basket_motion_s: float
    basket_elbow_lift_deg: float
    basket_shoulder_back_deg: float
    basket_elbow_compensation_deg: float | None
    action_ready_motion_s: float


@dataclass(frozen=True)
class FixedArmMotionState:
    mode: str
    start_targets: dict[str, float]
    goal_targets: dict[str, float]
    started_at: float
    duration_s: float
    waypoint_targets: tuple[dict[str, float], ...] = ()
    segment_weights: tuple[float, ...] = ()


@dataclass
class VrEpisodeBoundaryState:
    phase: str = "idle"
    held_sides: set[str] = field(default_factory=set)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backend launcher for XLeRobot real teleop and local LeRobot recording."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    manipulate = subparsers.add_parser("manipulate", help="Launch real teleoperation.")
    _add_shared_args(manipulate)
    manipulate.add_argument("--controller", choices=("keyboard", "vr"), default="keyboard")
    manipulate.add_argument(
        "--record-training",
        action="store_true",
        help="Enable local LeRobot recording while staying in manipulate mode.",
    )
    _add_recording_args(manipulate)

    record = subparsers.add_parser("record", help="Launch real teleop with local LeRobot recording.")
    _add_shared_args(record)
    record.add_argument("--controller", choices=("keyboard", "vr"), default="keyboard")
    _add_recording_args(record)
    return parser


def _add_recording_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-id", default="local/xlerobot_playground")
    parser.add_argument("--dataset-root", default="./datasets")
    parser.add_argument("--task", default="XLeRobot teleoperation")
    parser.add_argument("--use-videos", action="store_true")
    parser.add_argument("--start-key", default="[")
    parser.add_argument("--stop-key", default="]")
    parser.add_argument("--quit-key", default="\\")
    parser.add_argument(
        "--no-resume-dataset",
        action="store_true",
        help="Create a fresh dataset and fail if the dataset root already exists.",
    )
    parser.add_argument(
        "--record-training-toggle-vr-button",
        default="",
        help="Legacy direct Quest button for training start/save. Empty uses the in-VR recording menu.",
    )
    parser.add_argument(
        "--record-training-discard-vr-button",
        default="",
        help="Legacy direct Quest button for training cancel/discard. Empty uses the in-VR recording menu.",
    )


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", default=str(resolve_xlerobot_repo_root()))
    parser.add_argument("--robot-kind", choices=("xlerobot", "xlerobot_2wheels"), default="xlerobot")
    parser.add_argument("--port1", default="/dev/ttyACM0")
    parser.add_argument("--port2", default="/dev/ttyACM1")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--camera",
        action="append",
        default=[],
        metavar="NAME=DRIVER:SOURCE",
        help=(
            "Camera config. Example: `head=realsense:125322060037` or "
            "`left_wrist=opencv:/dev/video0`. Optional per-camera overrides: "
            "`left_wrist=opencv:0,fps=30`."
        ),
    )
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--use-degrees", action="store_true")
    parser.add_argument("--xlevr-path", default=None)
    parser.add_argument(
        "--manual-calibration-prompt",
        action="store_true",
        help="Show the XLeRobot calibration restore prompt instead of auto-restoring saved calibration.",
    )
    parser.add_argument(
        "--orbbec-rgb-vr",
        action="store_true",
        help="Show the Orbbec Gemini 2 RGB stream in the Quest/XLeVR scene.",
    )
    parser.add_argument(
        "--orbbec-capture-bin",
        default=None,
        help="Path to the native orbbec_rgb_test binary.",
    )
    parser.add_argument(
        "--orbbec-no-launch",
        action="store_true",
        help="Serve the Quest overlay from an already-running Orbbec RGB sidecar.",
    )
    parser.add_argument("--orbbec-output-dir", default="artifacts/orbbec_rgb")
    parser.add_argument("--orbbec-width", type=int, default=640)
    parser.add_argument("--orbbec-height", type=int, default=480)
    parser.add_argument("--orbbec-fps", type=int, default=30)
    parser.add_argument("--orbbec-timeout-ms", type=int, default=1000)
    parser.add_argument(
        "--orbbec-log-every",
        type=int,
        default=0,
        help="Native Orbbec sidecar frame log interval. Use 0 to silence frame logs.",
    )
    parser.add_argument(
        "--vr-input-scale",
        type=float,
        default=0.35,
        help="Scale Quest controller movement before it reaches the robot. Lower is slower.",
    )
    parser.add_argument(
        "--vr-kp",
        type=float,
        default=0.6,
        help="VR joint proportional gain. Lower is smoother/slower.",
    )
    parser.add_argument(
        "--vr-camera-hz",
        type=float,
        default=30.0,
        help="How often to poll wrist cameras for the VR overlay.",
    )
    parser.add_argument(
        "--no-vr-video-streams",
        action="store_true",
        help="Disable Quest WebRTC camera streams while keeping VR controls and the recording menu.",
    )
    parser.add_argument(
        "--vr-left-arm-clutch-key",
        default="1",
        help="Hold this keyboard key to pause/rebaseline left-arm VR IK without moving the robot arm.",
    )
    parser.add_argument(
        "--vr-right-arm-clutch-key",
        default="2",
        help="Hold this keyboard key to pause/rebaseline right-arm VR IK without moving the robot arm.",
    )
    parser.add_argument(
        "--no-vr-squeeze-clutch",
        action="store_true",
        help="Disable Quest squeeze/grip as an IK clutch. Keyboard clutch keys remain available.",
    )
    parser.add_argument(
        "--vr-skill-mode",
        choices=("free", "grab_to_basket"),
        default="free",
        help="Optional higher-level VR manipulation mode. `grab_to_basket` adds a fixed basket placement pose.",
    )
    parser.add_argument(
        "--vr-skill-arm",
        choices=("left", "right", "both"),
        default="both",
        help="Arm(s) active in grab_to_basket mode. The inactive arm is held at ACTION_READY.",
    )
    parser.add_argument(
        "--vr-basket-pose-key",
        default="b",
        help="Keyboard key that moves the right arm from IK grabbing to the fixed basket pose.",
    )
    parser.add_argument(
        "--vr-skill-reset-key",
        default="r",
        help="Keyboard key that resets all grab_to_basket arms back to ACTION_READY.",
    )
    parser.add_argument("--vr-right-action-key", default="a")
    parser.add_argument("--vr-left-basket-key", default="y")
    parser.add_argument("--vr-left-action-key", default="x")
    parser.add_argument(
        "--vr-basket-pose-vr-button",
        default="right:b",
        help="Quest button that moves the right arm to the fixed basket pose, formatted as `left:name` or `right:name`.",
    )
    parser.add_argument("--vr-right-action-vr-button", default="right:a")
    parser.add_argument("--vr-left-basket-vr-button", default="left:y")
    parser.add_argument("--vr-left-action-vr-button", default="left:x")
    parser.add_argument(
        "--vr-basket-target",
        action="append",
        default=[],
        metavar="JOINT=DEG",
        help=(
            "Override a basket pose target, repeatable. Example: "
            "`--vr-basket-target right_arm_shoulder_pan.pos=-30`."
        ),
    )
    parser.add_argument(
        "--vr-basket-release-gripper",
        type=float,
        default=45.0,
        help="Gripper target sent for the pressed trigger while holding the fixed basket pose.",
    )
    parser.add_argument(
        "--vr-basket-motion-s",
        type=float,
        default=4.0,
        help="Total seconds used for the lift, transfer, and descent into the fixed basket pose.",
    )
    parser.add_argument(
        "--vr-basket-elbow-lift-deg",
        type=float,
        default=-25.0,
        help=(
            "Signed elbow-flex offset used first at the current pose and then above the basket "
            "before lowering to the captured basket pose. The current robot uses a negative offset to lift."
        ),
    )
    parser.add_argument(
        "--vr-basket-shoulder-back-deg",
        type=float,
        default=65.0,
        help="Degrees to move shoulder_lift back toward NAV_STOW after the initial elbow lift.",
    )
    parser.add_argument(
        "--vr-basket-elbow-compensation-deg",
        type=float,
        default=None,
        help=(
            "Elbow-flex degrees moved opposite the initial lift while shoulder_lift moves back. "
            "Defaults to 2x the shoulder-back amount."
        ),
    )
    parser.add_argument(
        "--vr-action-ready-motion-s",
        type=float,
        default=2.0,
        help="Seconds used to ramp back to ACTION_READY after pressing the action-ready button.",
    )
    parser.add_argument("--vr-base-max-linear", type=float, default=0.12)
    parser.add_argument("--vr-base-max-angular", type=float, default=35.0)
    parser.add_argument("--vr-base-linear-accel", type=float, default=0.9)
    parser.add_argument("--vr-base-angular-accel", type=float, default=240.0)
    parser.add_argument("--vr-base-deadzone", type=float, default=0.14)
    parser.add_argument("--vr-base-curve", type=float, default=1.5)
    parser.add_argument(
        "--allow-vr-base-while-recording",
        action="store_true",
        help=(
            "Allow right-thumbstick base motion during an active recording. "
            "By default the base is hard-stopped while recording."
        ),
    )
    parser.add_argument("--vr-arm-vertical-sign", type=float, default=1.0)
    parser.add_argument("--vr-arm-y-gain", type=float, default=1.4)
    parser.add_argument("--vr-arm-z-gain", type=float, default=1.0)
    parser.add_argument("--vr-arm-ik-alpha", type=float, default=0.25)
    parser.add_argument(
        "--vr-arm-ik-mode",
        choices=("planar", "yawed"),
        default="planar",
        help="Arm IK mode. `yawed` treats controller side motion as part of a 3D target.",
    )
    parser.add_argument(
        "--vr-arm-yawed-forward-gain",
        type=float,
        default=1.0,
        help="Forward/back gain for yawed 3D arm IK.",
    )
    parser.add_argument(
        "--vr-arm-yawed-lateral-gain",
        type=float,
        default=0.30,
        help="Sideways gain for yawed 3D arm IK. Use a negative value to flip lateral direction.",
    )
    parser.add_argument(
        "--vr-arm-yawed-pan-sign",
        type=float,
        default=1.0,
        help="Sign applied to computed shoulder pan in yawed 3D arm IK.",
    )
    parser.add_argument(
        "--vr-arm-yawed-pan-limit",
        type=float,
        default=120.0,
        help="Absolute shoulder pan limit used by yawed 3D arm IK.",
    )
    parser.add_argument(
        "--vr-arm-yawed-pan-step-limit",
        type=float,
        default=3.0,
        help="Maximum shoulder-pan change per VR update in yawed IK mode.",
    )
    parser.add_argument(
        "--vr-arm-debug",
        action="store_true",
        help="Print throttled per-arm VR IK diagnostics while teleoperating.",
    )
    parser.add_argument("--vr-arm-debug-hz", type=float, default=2.0)
    parser.add_argument(
        "--vr-wrist-video-gain",
        type=float,
        default=1.0,
        help="Display-only brightness gain for wrist camera panels in the Quest overlay.",
    )
    parser.add_argument(
        "--vr-wrist-video-gamma",
        type=float,
        default=1.0,
        help="Display-only gamma for wrist camera panels. Values below 1 brighten shadows.",
    )
    parser.add_argument(
        "--vr-wrist-video-bias",
        type=float,
        default=0.0,
        help="Display-only brightness offset for wrist camera panels.",
    )
    parser.add_argument("--vr-orbbec-video-gain", type=float, default=1.0)
    parser.add_argument("--vr-orbbec-video-gamma", type=float, default=1.0)
    parser.add_argument("--vr-orbbec-video-bias", type=float, default=0.0)
    parser.add_argument("--vr-arm-shoulder-lift-min", type=float, default=-108.0)
    parser.add_argument("--vr-arm-shoulder-lift-max", type=float, default=96.0)
    parser.add_argument(
        "--vr-arm-elbow-flex-min",
        type=float,
        default=-115.0,
        help="Minimum elbow flex target produced by VR IK. Lower values extend the negative/tucked side.",
    )
    parser.add_argument(
        "--vr-arm-elbow-flex-max",
        type=float,
        default=106.0,
        help="Maximum elbow flex target produced by VR IK. Original SO101 IK positive side is about 106 degrees.",
    )
    parser.add_argument(
        "--vr-arm-unbounded-ik-joints",
        action="store_true",
        help=(
            "Do not clip IK shoulder_lift/elbow_flex outputs to the configured software limits. "
            "Physical motor/calibration limits may still apply."
        ),
    )
    parser.add_argument(
        "--no-vr-startup-pose",
        action="store_true",
        help="Skip NAV_STOW -> ACTION_READY startup pose routine.",
    )
    parser.add_argument("--vr-nav-stow-wait-s", type=float, default=5.0)
    parser.add_argument(
        "--vr-action-ready-elbow-delta",
        type=float,
        default=_ACTION_READY_ELBOW_DELTA,
        help="Elbow flex delta applied when moving from NAV_STOW to ACTION_READY.",
    )
    parser.add_argument(
        "--vr-action-ready-shoulder-delta",
        type=float,
        default=_ACTION_READY_SHOULDER_DELTA,
        help="Shoulder lift delta applied after elbow when moving from NAV_STOW to ACTION_READY.",
    )
    parser.add_argument(
        "--vr-action-ready-wrist-delta",
        type=float,
        default=_ACTION_READY_WRIST_DELTA,
        help="Optional wrist flex delta applied during the elbow stage of ACTION_READY.",
    )
    parser.add_argument("--vr-startup-pose-steps", type=int, default=40)
    parser.add_argument("--vr-startup-pose-stage-delay-s", type=float, default=0.02)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bootstrap_xlerobot(args.repo_root)

    recording_enabled = args.mode == "record" or bool(getattr(args, "record_training", False))
    if recording_enabled and args.controller == "vr":
        args.orbbec_rgb_vr = True
    orbbec_rgb = OrbbecRgbConfig(
        enabled=args.orbbec_rgb_vr,
        launch_capture=not args.orbbec_no_launch,
        capture_bin=Path(
            args.orbbec_capture_bin
            or Path(__file__).resolve().parents[1] / "build" / "orbbec_rgb_test" / "orbbec_rgb_test"
        ).expanduser().resolve(),
        output_dir=Path(args.orbbec_output_dir).expanduser().resolve(),
        width=args.orbbec_width,
        height=args.orbbec_height,
        fps=args.orbbec_fps,
        timeout_ms=args.orbbec_timeout_ms,
        log_every=args.orbbec_log_every,
    )

    interface = XLeRobotInterface(args.repo_root)
    if args.robot_kind == "xlerobot_2wheels":
        config_cls, robot_cls = interface.robot_2wheels_classes()
    else:
        config_cls, robot_cls = interface.robot_classes()
    robot_config = config_cls(
        port1=args.port1,
        port2=args.port2,
        cameras=_build_camera_configs(
            args.camera,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
        ),
        use_degrees=args.use_degrees,
    )
    robot = robot_cls(robot_config)

    recording = None
    if recording_enabled:
        extra_observation_features = {}
        orbbec_output_dir = None
        if args.controller == "vr" and orbbec_rgb.enabled:
            extra_observation_features["head"] = (orbbec_rgb.height, orbbec_rgb.width, 3)
            orbbec_output_dir = orbbec_rgb.output_dir
        recording = RecordingSession(
            dataset=_create_dataset(
                robot,
                dataset_id=args.dataset_id,
                dataset_root=args.dataset_root,
                fps=args.fps,
                use_videos=args.use_videos,
                resume=not args.no_resume_dataset,
                extra_observation_features=extra_observation_features,
            ),
            task=args.task,
            dataset_root=Path(args.dataset_root).expanduser().resolve(),
            orbbec_output_dir=orbbec_output_dir,
        )

    if args.controller == "keyboard":
        return _run_keyboard_backend(
            repo_root=Path(args.repo_root).expanduser().resolve(),
            robot=robot,
            fps=args.fps,
            recording=recording,
            start_key=getattr(args, "start_key", "["),
            stop_key=getattr(args, "stop_key", "]"),
            quit_key=getattr(args, "quit_key", "\\"),
            auto_restore_calibration=not args.manual_calibration_prompt,
        )
    return _run_vr_backend(
        interface=interface,
        robot=robot,
        fps=args.fps,
        recording=recording,
        start_key=getattr(args, "start_key", "["),
        stop_key=getattr(args, "stop_key", "]"),
        quit_key=getattr(args, "quit_key", "\\"),
        xlevr_path=args.xlevr_path,
        auto_restore_calibration=not args.manual_calibration_prompt,
        orbbec_rgb=orbbec_rgb,
        training_toggle_vr_button=getattr(args, "record_training_toggle_vr_button", "right:thumbstick"),
        training_discard_vr_button=getattr(args, "record_training_discard_vr_button", "left:thumbstick"),
        vr_input_scale=args.vr_input_scale,
        vr_kp=args.vr_kp,
        vr_camera_hz=args.vr_camera_hz,
        enable_squeeze_clutch=not args.no_vr_squeeze_clutch,
        arm_clutch_keys=VrArmClutchKeys(
            left=args.vr_left_arm_clutch_key,
            right=args.vr_right_arm_clutch_key,
        ),
        basket_pose=VrBasketPoseConfig(
            enabled=args.vr_skill_mode == "grab_to_basket",
            skill_arm=args.vr_skill_arm,
            full_reset_key=args.vr_skill_reset_key,
            right_basket_key=args.vr_basket_pose_key,
            right_action_key=args.vr_right_action_key,
            left_basket_key=args.vr_left_basket_key,
            left_action_key=args.vr_left_action_key,
            right_basket_button=args.vr_basket_pose_vr_button,
            right_action_button=args.vr_right_action_vr_button,
            left_basket_button=args.vr_left_basket_vr_button,
            left_action_button=args.vr_left_action_vr_button,
            targets=_build_vr_basket_pose_targets(args.vr_basket_target),
            release_gripper=args.vr_basket_release_gripper,
            basket_motion_s=args.vr_basket_motion_s,
            basket_elbow_lift_deg=args.vr_basket_elbow_lift_deg,
            basket_shoulder_back_deg=args.vr_basket_shoulder_back_deg,
            basket_elbow_compensation_deg=args.vr_basket_elbow_compensation_deg,
            action_ready_motion_s=args.vr_action_ready_motion_s,
        ),
        base_smoother=BaseSmoother(
            max_linear=args.vr_base_max_linear,
            max_angular=args.vr_base_max_angular,
            linear_accel=args.vr_base_linear_accel,
            angular_accel=args.vr_base_angular_accel,
            deadzone=args.vr_base_deadzone,
            curve=args.vr_base_curve,
        ),
        lock_base_while_recording=not args.allow_vr_base_while_recording,
        arm_tuning=VrArmTuning(
            ik_mode=args.vr_arm_ik_mode,
            vertical_sign=args.vr_arm_vertical_sign,
            y_gain=args.vr_arm_y_gain,
            z_gain=args.vr_arm_z_gain,
            ik_alpha=args.vr_arm_ik_alpha,
            yawed_forward_gain=args.vr_arm_yawed_forward_gain,
            yawed_lateral_gain=args.vr_arm_yawed_lateral_gain,
            yawed_pan_sign=args.vr_arm_yawed_pan_sign,
            yawed_pan_limit=args.vr_arm_yawed_pan_limit,
            yawed_pan_step_limit=args.vr_arm_yawed_pan_step_limit,
            shoulder_lift_min=args.vr_arm_shoulder_lift_min,
            shoulder_lift_max=args.vr_arm_shoulder_lift_max,
            elbow_flex_min=args.vr_arm_elbow_flex_min,
            elbow_flex_max=args.vr_arm_elbow_flex_max,
            enforce_joint_limits=not args.vr_arm_unbounded_ik_joints,
            debug=args.vr_arm_debug,
            debug_hz=args.vr_arm_debug_hz,
        ),
        video_display=VrVideoDisplayConfig(
            wrist_gain=args.vr_wrist_video_gain,
            wrist_gamma=args.vr_wrist_video_gamma,
            wrist_bias=args.vr_wrist_video_bias,
            orbbec_gain=args.vr_orbbec_video_gain,
            orbbec_gamma=args.vr_orbbec_video_gamma,
            orbbec_bias=args.vr_orbbec_video_bias,
        ),
        vr_video_streams=not args.no_vr_video_streams,
        startup_pose=VrStartupPoseConfig(
            enabled=not args.no_vr_startup_pose,
            stow_wait_s=args.vr_nav_stow_wait_s,
            stow_pose=_default_vr_nav_stow_pose(),
            action_ready_elbow_delta=args.vr_action_ready_elbow_delta,
            action_ready_shoulder_delta=args.vr_action_ready_shoulder_delta,
            action_ready_wrist_delta=args.vr_action_ready_wrist_delta,
            steps_per_stage=args.vr_startup_pose_steps,
            stage_delay_s=args.vr_startup_pose_stage_delay_s,
        ),
    )


def _build_camera_configs(
    camera_specs: list[str],
    *,
    width: int,
    height: int,
    fps: int,
) -> dict[str, Any]:
    from lerobot.cameras.configs import ColorMode, Cv2Backends, Cv2Rotation
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

    cameras: dict[str, Any] = {}
    for raw_spec in camera_specs:
        spec = _parse_camera_spec(raw_spec)
        spec_width = spec.width or width
        spec_height = spec.height or height
        spec_fps = spec.fps or fps
        if spec.driver == "opencv":
            source: Any = int(spec.source) if spec.source.isdigit() else spec.source
            try:
                backend = Cv2Backends[spec.backend.upper()] if spec.backend else Cv2Backends.ANY
            except KeyError as exc:
                valid = ", ".join(item.name for item in Cv2Backends)
                raise ValueError(f"Unsupported OpenCV backend `{spec.backend}`. Valid values: {valid}.") from exc
            cameras[spec.name] = OpenCVCameraConfig(
                index_or_path=source,
                fps=spec_fps,
                width=spec_width,
                height=spec_height,
                rotation=Cv2Rotation.NO_ROTATION,
                fourcc=spec.fourcc,
                backend=backend,
            )
            continue
        if spec.driver == "realsense":
            cameras[spec.name] = RealSenseCameraConfig(
                serial_number_or_name=spec.source,
                fps=spec_fps,
                width=spec_width,
                height=spec_height,
                color_mode=ColorMode.BGR,
                rotation=Cv2Rotation.NO_ROTATION,
                use_depth=True,
            )
            continue
        raise ValueError(f"Unsupported camera driver `{spec.driver}` in `{raw_spec}`")
    return cameras


def _parse_camera_spec(raw_spec: str) -> CameraSpec:
    if "=" not in raw_spec or ":" not in raw_spec:
        raise ValueError(
            f"Invalid camera spec `{raw_spec}`. Use `NAME=DRIVER:SOURCE`, "
            "for example `head=realsense:125322060037`."
        )
    name, remainder = raw_spec.split("=", 1)
    driver, source_and_options = remainder.split(":", 1)
    source, *option_parts = source_and_options.split(",")
    int_options: dict[str, int] = {}
    str_options: dict[str, str] = {}
    for option in option_parts:
        if not option.strip():
            continue
        if "=" not in option:
            raise ValueError(f"Invalid camera option `{option}` in `{raw_spec}`. Use key=value.")
        option_key, option_value = option.split("=", 1)
        option_key = option_key.strip()
        option_value = option_value.strip()
        if option_key in {"width", "height", "fps"}:
            int_options[option_key] = int(option_value)
            continue
        if option_key == "fourcc":
            if len(option_value) != 4:
                raise ValueError(f"Invalid fourcc `{option_value}` in `{raw_spec}`. Use 4 characters, e.g. MJPG.")
            str_options[option_key] = option_value
            continue
        if option_key == "backend":
            str_options[option_key] = option_value
            continue
        if option_key not in {"width", "height", "fps", "fourcc", "backend"}:
            raise ValueError(f"Unsupported camera option `{option_key}` in `{raw_spec}`.")
    return CameraSpec(
        name=name.strip(),
        driver=driver.strip(),
        source=source.strip(),
        width=int_options.get("width"),
        height=int_options.get("height"),
        fps=int_options.get("fps"),
        fourcc=str_options.get("fourcc"),
        backend=str_options.get("backend"),
    )


def _default_vr_nav_stow_pose() -> dict[str, float]:
    return {
        f"{side}_arm_{joint}.pos": value
        for side in ("left", "right")
        for joint, value in _NAV_STOW_ARM_POSE.items()
    }


def _start_orbbec_rgb_sidecar(config: OrbbecRgbConfig) -> subprocess.Popen[bytes] | None:
    if not config.enabled or not config.launch_capture:
        return None
    capture_bin = config.capture_bin.resolve()
    if not capture_bin.exists():
        raise FileNotFoundError(
            f"Orbbec RGB capture binary not found: {capture_bin}. "
            "Build it with `cmake -S tools/orbbec_rgb_test -B build/orbbec_rgb_test "
            "-DORBBEC_SDK_ROOT=/Users/alin/orbbec/sdk && cmake --build build/orbbec_rgb_test`."
        )

    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(capture_bin),
        "--frames",
        "0",
        "--latest-only",
        "--output-dir",
        str(output_dir),
        "--width",
        str(config.width),
        "--height",
        str(config.height),
        "--fps",
        str(config.fps),
        "--timeout-ms",
        str(config.timeout_ms),
        "--log-every",
        str(max(0, config.log_every)),
    ]
    print(f"Starting Orbbec RGB sidecar: {' '.join(cmd)}")
    return subprocess.Popen(cmd)


def _stop_orbbec_rgb_sidecar(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def _enable_threaded_vr_http(vr_teleop: Any) -> bool:
    module_prefix = vr_teleop.__class__.__module__.rsplit(".", 1)[0]
    monitor_module = sys.modules.get(f"{module_prefix}.vr_monitor")
    if monitor_module is None or not hasattr(monitor_module, "http"):
        return False
    http_server_module = monitor_module.http.server
    if getattr(http_server_module, "HTTPServer", None) is http_server_module.ThreadingHTTPServer:
        return True
    http_server_module.ThreadingHTTPServer.daemon_threads = True
    http_server_module.HTTPServer = http_server_module.ThreadingHTTPServer
    return True


def _install_orbbec_vr_overlay(
    vr_teleop: Any,
    output_dir: Path,
    *,
    include_orbbec: bool,
    video_display: VrVideoDisplayConfig,
    video_streams_enabled: bool,
) -> bool:
    monitor = getattr(vr_teleop, "vr_monitor", None)
    if monitor is None:
        return False
    handler_cls = getattr(sys.modules.get(monitor.__class__.__module__), "SimpleAPIHandler", None)
    if handler_cls is None:
        return False
    if getattr(handler_cls, "_orbbec_overlay_installed", False):
        handler_cls.orbbec_output_dir = output_dir.resolve()
        handler_cls.orbbec_overlay_include_orbbec = include_orbbec
        handler_cls.robot42_video_display = video_display
        handler_cls.robot42_video_streams_enabled = video_streams_enabled
        return True

    original_do_get = handler_cls.do_GET
    original_do_post = getattr(handler_cls, "do_POST", None)
    original_serve_file = handler_cls.serve_file
    handler_cls.orbbec_output_dir = output_dir.resolve()
    handler_cls.orbbec_overlay_include_orbbec = include_orbbec
    handler_cls.robot42_video_display = video_display
    handler_cls.robot42_video_streams_enabled = video_streams_enabled

    def do_GET(self):
        if self.path.startswith("/api/status"):
            return _serve_xlevr_status(self)
        if self.path.startswith("/api/config"):
            return _serve_xlevr_config(self)
        if self.path.startswith("/orbbec/latest.mjpg"):
            return _serve_orbbec_mjpeg_stream(self)
        if self.path.startswith("/orbbec/latest.jpg"):
            return _serve_orbbec_jpeg_snapshot(self)
        if self.path.startswith("/orbbec/latest.ppm"):
            return _serve_orbbec_file(self, "latest.ppm", "image/x-portable-pixmap")
        if self.path.startswith("/orbbec/latest.json"):
            return _serve_orbbec_file(self, "latest.json", "application/json")
        if self.path.startswith("/vr-camera/"):
            return _serve_vr_camera_file(self)
        return original_do_get(self)

    def do_POST(self):
        if self.path.startswith("/api/robot"):
            return _serve_xlevr_post_ack(self, "robot")
        if self.path.startswith("/api/keyboard"):
            return _serve_xlevr_post_ack(self, "keyboard")
        if self.path.startswith("/api/keypress"):
            return _serve_xlevr_post_ack(self, "keypress")
        if self.path.startswith("/api/config"):
            return _serve_xlevr_post_ack(self, "config")
        if self.path.startswith("/api/restart"):
            return _serve_xlevr_post_ack(self, "restart")
        if self.path.startswith("/api/teleop-menu"):
            return _serve_teleop_menu_request(self)
        if self.path.startswith("/api/recording-control"):
            return _serve_recording_control_request(self)
        if self.path.startswith("/api/finish-collection"):
            return _serve_finish_collection_request(self)
        if self.path.startswith("/webrtc/offer"):
            if not getattr(self.__class__, "robot42_video_streams_enabled", True):
                return _write_json_response(
                    self,
                    json.dumps({"error": "VR video streams are disabled"}),
                    status=404,
                )
            return _serve_webrtc_offer(self)
        if original_do_post is not None:
            return original_do_post(self)
        self.send_error(404, "Unknown API endpoint")

    def serve_file(self, filename, content_type):
        if filename == "web-ui/vr_app.js":
            try:
                web_root = getattr(self.server, "web_root_path", None)
                root = Path(web_root) if web_root else Path.cwd()
                js_path = root / filename
                content = js_path.read_text()
                content = _patch_xlevr_controller_button_js(content)
                content += "\n\n" + _orbbec_vr_overlay_js(
                    include_orbbec=getattr(handler_cls, "orbbec_overlay_include_orbbec", True),
                    video_display=getattr(handler_cls, "robot42_video_display", video_display),
                    video_streams_enabled=getattr(handler_cls, "robot42_video_streams_enabled", True),
                )
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(content.encode("utf-8"))
                return
            except Exception as exc:
                print(f"Error injecting Orbbec VR overlay: {exc}")
        return original_serve_file(self, filename, content_type)

    handler_cls.do_GET = do_GET
    handler_cls.do_POST = do_POST
    handler_cls.serve_file = serve_file
    handler_cls._orbbec_overlay_installed = True
    return True


def _patch_xlevr_controller_button_js(content: str) -> str:
    left_old = """                leftController.buttons = {
                    a: !!leftGamepad.buttons[3]?.pressed,
                    b: !!leftGamepad.buttons[4]?.pressed,
                    squeeze: !!leftGamepad.buttons[1]?.pressed,
                    thumbstick: !!leftGamepad.buttons[2]?.pressed,
                    menu: !!leftGamepad.buttons[6]?.pressed
                };"""
    left_new = """                leftController.buttons = {
                    x: !!leftGamepad.buttons[4]?.pressed,
                    y: !!leftGamepad.buttons[5]?.pressed,
                    a: !!leftGamepad.buttons[4]?.pressed,
                    b: !!leftGamepad.buttons[5]?.pressed,
                    squeeze: !!leftGamepad.buttons[1]?.pressed,
                    thumbstick: !!leftGamepad.buttons[2]?.pressed || !!leftGamepad.buttons[3]?.pressed,
                    menu: !!leftGamepad.buttons[6]?.pressed
                };"""
    right_old = """                rightController.buttons = {
                    a: !!rightGamepad.buttons[3]?.pressed,
                    b: !!rightGamepad.buttons[4]?.pressed,
                    squeeze: !!rightGamepad.buttons[1]?.pressed,
                    thumbstick: !!rightGamepad.buttons[2]?.pressed,
                    menu: !!rightGamepad.buttons[6]?.pressed
                };"""
    right_new = """                rightController.buttons = {
                    a: !!rightGamepad.buttons[4]?.pressed,
                    b: !!rightGamepad.buttons[5]?.pressed,
                    squeeze: !!rightGamepad.buttons[1]?.pressed,
                    thumbstick: !!rightGamepad.buttons[2]?.pressed || !!rightGamepad.buttons[3]?.pressed,
                    menu: !!rightGamepad.buttons[6]?.pressed
                };"""
    return content.replace(left_old, left_new).replace(right_old, right_new)


def _publish_vr_camera_frames(obs: dict[str, Any]) -> None:
    try:
        import cv2
        import numpy as np
    except Exception:
        return

    frames: dict[str, tuple[bytes, int, int, float]] = {}
    jpegs: dict[str, tuple[bytes, int, int, float]] = {}
    for name, value in obs.items():
        if name not in {"left_wrist", "right_wrist"}:
            continue
        array = np.asarray(value)
        if array.ndim != 3 or array.shape[2] < 3:
            continue
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        rgb = np.ascontiguousarray(array[:, :, :3])
        height, width = rgb.shape[:2]
        header = f"P6\n{width} {height}\n255\n".encode("ascii")
        captured_at = time.time()
        frames[name] = (header + rgb.tobytes(), width, height, captured_at)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY])
        if ok:
            jpegs[name] = (encoded.tobytes(), width, height, captured_at)

    if frames:
        with _VR_CAMERA_FRAMES_LOCK:
            _VR_CAMERA_FRAMES.update(frames)
    if jpegs:
        with _VR_CAMERA_JPEGS_LOCK:
            _VR_CAMERA_JPEGS.update(jpegs)


def _get_robot_observation(robot: Any, *, use_camera: bool = True) -> dict[str, Any]:
    try:
        return robot.get_observation(use_camera=use_camera)
    except TypeError:
        return robot.get_observation()


def _get_robot_observation_best_effort(robot: Any) -> tuple[dict[str, Any], Exception | None]:
    obs = _get_robot_observation(robot, use_camera=False)
    cameras = getattr(robot, "cameras", None)
    if not cameras:
        return obs, None

    first_error: Exception | None = None
    for cam_key, cam in cameras.items():
        try:
            obs[cam_key] = cam.async_read(timeout_ms=_VR_CAMERA_ASYNC_READ_TIMEOUT_MS)
        except Exception as exc:
            if first_error is None:
                first_error = exc
            if _camera_read_thread_is_dead(cam) and _try_reconnect_vr_camera(cam_key, cam):
                try:
                    obs[cam_key] = cam.async_read(timeout_ms=_VR_CAMERA_ASYNC_READ_TIMEOUT_MS)
                except Exception as retry_exc:
                    if first_error is None:
                        first_error = retry_exc
            continue
    return obs, first_error


def _camera_read_thread_is_dead(cam: Any) -> bool:
    thread = getattr(cam, "thread", None)
    return thread is not None and not thread.is_alive()


def _try_reconnect_vr_camera(cam_key: str, cam: Any) -> bool:
    now = time.monotonic()
    last_attempt = _VR_CAMERA_RECONNECT_LAST_ATTEMPT.get(cam_key, 0.0)
    if now - last_attempt < _VR_CAMERA_RECONNECT_MIN_INTERVAL_S:
        return False
    _VR_CAMERA_RECONNECT_LAST_ATTEMPT[cam_key] = now

    print(f"VR camera `{cam_key}` read thread is dead; attempting reconnect.")
    disconnect = getattr(cam, "disconnect", None)
    if callable(disconnect):
        try:
            disconnect()
        except Exception as exc:
            print(f"VR camera `{cam_key}` disconnect before reconnect failed: {exc}")

    connect = getattr(cam, "connect", None)
    if not callable(connect):
        print(f"VR camera `{cam_key}` cannot reconnect: camera object has no connect().")
        return False
    try:
        connect(warmup=False)
    except TypeError:
        try:
            connect()
        except Exception as exc:
            print(f"VR camera `{cam_key}` reconnect failed: {exc}")
            return False
    except Exception as exc:
        print(f"VR camera `{cam_key}` reconnect failed: {exc}")
        return False

    print(f"VR camera `{cam_key}` reconnected.")
    return True


def _run_vr_startup_pose(robot: Any, vr_teleop: Any, config: VrStartupPoseConfig) -> None:
    if not config.enabled:
        robot.send_action(vr_teleop.move_to_zero_position(robot))
        _sync_vr_teleop_to_current_pose(robot, vr_teleop)
        return

    if config.stow_pose:
        print("Moving to NAV_STOW.")
        _move_to_joint_targets(robot, config.stow_pose, steps=config.steps_per_stage, delay_s=config.stage_delay_s)
    else:
        print("Using current robot pose as NAV_STOW.")

    print("folded")
    if config.stow_wait_s > 0:
        print(f"Waiting {config.stow_wait_s:.1f}s before moving to ACTION_READY.")
        time.sleep(config.stow_wait_s)

    _move_to_action_ready(robot, vr_teleop, config)


def _move_to_action_ready(robot: Any, vr_teleop: Any, config: VrStartupPoseConfig) -> None:
    if not config.enabled:
        robot.send_action(vr_teleop.move_to_zero_position(robot))
        _sync_vr_teleop_to_current_pose(robot, vr_teleop)
        return

    print(
        "Moving to ACTION_READY: "
        f"elbow delta {config.action_ready_elbow_delta:+.1f}, "
        f"shoulder delta {config.action_ready_shoulder_delta:+.1f}, "
        f"wrist delta {config.action_ready_wrist_delta:+.1f}."
    )

    obs = _get_robot_observation(robot, use_camera=False)
    elbow_targets: dict[str, float] = {}
    for side in ("left", "right"):
        elbow_key = f"{side}_arm_elbow_flex.pos"
        wrist_key = f"{side}_arm_wrist_flex.pos"
        if elbow_key in obs:
            elbow_targets[elbow_key] = float(obs[elbow_key]) + config.action_ready_elbow_delta
        if config.action_ready_wrist_delta and wrist_key in obs:
            elbow_targets[wrist_key] = float(obs[wrist_key]) + config.action_ready_wrist_delta
    if elbow_targets:
        _move_to_joint_targets(robot, elbow_targets, steps=config.steps_per_stage, delay_s=config.stage_delay_s)

    obs = _get_robot_observation(robot, use_camera=False)
    shoulder_targets: dict[str, float] = {}
    for side in ("left", "right"):
        shoulder_key = f"{side}_arm_shoulder_lift.pos"
        if shoulder_key in obs:
            shoulder_targets[shoulder_key] = float(obs[shoulder_key]) + config.action_ready_shoulder_delta
    if shoulder_targets:
        _move_to_joint_targets(robot, shoulder_targets, steps=config.steps_per_stage, delay_s=config.stage_delay_s)

    _sync_vr_teleop_to_current_pose(robot, vr_teleop)


def _move_to_joint_targets(
    robot: Any,
    targets: dict[str, float],
    *,
    steps: int,
    delay_s: float,
) -> None:
    if not targets:
        return
    obs = _get_robot_observation(robot, use_camera=False)
    starts = {key: float(obs[key]) for key in targets if key in obs}
    if not starts:
        return
    steps = max(1, steps)
    for idx in range(1, steps + 1):
        ratio = idx / steps
        action = {
            key: start + (float(targets[key]) - start) * ratio
            for key, start in starts.items()
        }
        robot.send_action(action)
        if delay_s > 0:
            time.sleep(delay_s)


def _sync_vr_teleop_to_current_pose(
    robot: Any,
    vr_teleop: Any,
    *,
    rebaseline_frames: int = _VR_ARM_CLUTCH_RELEASE_HOLD_FRAMES,
) -> None:
    obs = _get_robot_observation(robot, use_camera=False)
    for side in ("left", "right"):
        _sync_vr_arm_to_observation(vr_teleop, side, obs)
        arm = getattr(vr_teleop, f"{side}_arm", None)
        if arm is not None and rebaseline_frames > 0:
            setattr(arm, "ik_clutch_rebaseline_frames", int(rebaseline_frames))

    head = getattr(vr_teleop, "head_control", None)
    if head is not None:
        head_targets = {
            motor: float(obs[f"{motor}.pos"])
            for motor in ("head_motor_1", "head_motor_2")
            if f"{motor}.pos" in obs
        }
        if head_targets:
            head.target_positions = head_targets


def _sync_vr_arm_to_observation(vr_teleop: Any, side: str, obs: dict[str, Any]) -> None:
    arm = getattr(vr_teleop, f"{side}_arm", None)
    if arm is None:
        return
    targets: dict[str, float] = {}
    for joint in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"):
        obs_key = f"{side}_arm_{joint}.pos"
        if obs_key in obs:
            targets[joint] = float(obs[obs_key])
    if targets:
        arm.target_positions = targets
        arm.pitch = targets.get("wrist_flex", 0.0) + targets.get("shoulder_lift", 0.0) + targets.get("elbow_flex", 0.0)
        kin = getattr(arm, "kinematics", None)
        if kin is not None:
            try:
                arm.current_x, arm.current_y = _so101_forward_kinematics(
                    kin,
                    targets.get("shoulder_lift", 0.0),
                    targets.get("elbow_flex", 0.0),
                )
            except Exception:
                pass
        _sync_vr_arm_yawed_target_from_pose(arm, targets)
    _clear_vr_arm_controller_baseline(arm)


def _sync_vr_arm_yawed_target_from_pose(arm: Any, targets: dict[str, float]) -> None:
    planar_x = float(getattr(arm, "current_x", 0.0) or 0.0)
    planar_y = float(getattr(arm, "current_y", 0.0) or 0.0)
    radial = abs(planar_x)
    pan_sign = float(getattr(arm, "_robot42_yawed_pan_sign", 1.0) or 1.0)
    pan_rad = math.radians(float(targets.get("shoulder_pan", 0.0) or 0.0) / pan_sign)
    arm.current_forward = radial * math.cos(pan_rad)
    arm.current_lateral = radial * math.sin(pan_rad)
    arm.current_height = planar_y


def _so101_forward_kinematics(
    kinematics: Any,
    shoulder_lift_deg: float,
    elbow_flex_deg: float,
) -> tuple[float, float]:
    """Invert the SO101 IK branch used by this runtime."""
    l1 = float(getattr(kinematics, "l1", 0.1159))
    l2 = float(getattr(kinematics, "l2", 0.1350))
    theta1_offset = math.atan2(0.028, 0.11257)
    theta2_offset = math.atan2(0.0052, 0.1349) + theta1_offset
    theta1 = math.radians(90.0 - float(shoulder_lift_deg)) - theta1_offset
    theta2 = math.radians(float(elbow_flex_deg) + 90.0) - theta2_offset
    return (
        l1 * math.cos(theta1) + l2 * math.cos(theta1 - theta2),
        l1 * math.sin(theta1) + l2 * math.sin(theta1 - theta2),
    )


def _clear_vr_arm_controller_baseline(arm: Any) -> None:
    for attr in ("prev_vr_pos", "prev_wrist_flex", "prev_wrist_roll"):
        if hasattr(arm, attr):
            delattr(arm, attr)


def _write_response(handler: Any, content: bytes, content_type: str) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(content)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(content)


def _write_json_response(handler: Any, payload: str, *, status: int = 200) -> None:
    content = payload.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(content)))
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(content)


def _serve_xlevr_status(handler: Any) -> None:
    with _VR_SESSION_STATUS_LOCK:
        robot42_status = dict(_VR_SESSION_STATUS)
    _write_json_response(
        handler,
        json.dumps(
            {
                "left_arm_connected": True,
                "right_arm_connected": True,
                "vrConnected": True,
                "keyboardEnabled": False,
                "robotEngaged": True,
                "robot42": robot42_status,
            }
        ),
    )


def _serve_xlevr_config(handler: Any) -> None:
    _write_json_response(
        handler,
        (
            '{"robot":{"left_arm":{"name":"left","port":"","enabled":true},'
            '"right_arm":{"name":"right","port":"","enabled":true},'
            '"vr_to_robot_scale":1.0,"send_interval":0.05},'
            '"network":{"https_port":8443,"websocket_port":8442,"host_ip":"0.0.0.0"},'
            '"control":{"keyboard":{"pos_step":0.01,"angle_step":1.0}}}'
        ),
    )


def _serve_xlevr_post_ack(handler: Any, endpoint: str) -> None:
    try:
        length = int(handler.headers.get("Content-Length", "0") or "0")
    except ValueError:
        length = 0
    if length > 0:
        handler.rfile.read(length)
    _write_json_response(handler, f'{{"success":true,"endpoint":"{endpoint}"}}')


def _read_json_request_body(handler: Any) -> dict[str, Any]:
    try:
        length = int(handler.headers.get("Content-Length", "0") or "0")
    except ValueError:
        length = 0
    if length <= 0:
        return {}
    try:
        payload = json.loads(handler.rfile.read(length).decode("utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _serve_teleop_menu_request(handler: Any) -> None:
    payload = _read_json_request_body(handler)
    open_state = bool(payload.get("open", False))
    _set_vr_menu_open(open_state)
    _write_json_response(handler, json.dumps({"success": True, "menu_open": open_state}))


def _serve_recording_control_request(handler: Any) -> None:
    payload = _read_json_request_body(handler)
    action = str(payload.get("action", "")).strip().lower()
    if action not in {"record", "save", "cancel", "finish"}:
        _write_json_response(handler, json.dumps({"success": False, "error": "invalid action"}), status=400)
        return
    if not _vr_menu_is_open():
        _write_json_response(handler, json.dumps({"success": True, "action": action, "ignored": True}))
        return
    _set_vr_menu_open(False)
    _set_vr_recording_operation(
        {
            "record": "starting",
            "save": "saving",
            "cancel": "cancelling",
            "finish": "finishing",
        }[action]
    )
    _queue_vr_recording_control(action)
    _write_json_response(handler, json.dumps({"success": True, "action": action}))


def _serve_finish_collection_request(handler: Any) -> None:
    _read_json_request_body(handler)
    _request_finish_collection()
    _write_json_response(handler, json.dumps({"success": True, "finish_requested": True}))


def _request_finish_collection() -> None:
    global _VR_FINISH_REQUESTED
    with _VR_FINISH_REQUEST_LOCK:
        _VR_FINISH_REQUESTED = True
    with _VR_SESSION_STATUS_LOCK:
        _VR_SESSION_STATUS["finish_requested"] = True


def _consume_finish_collection_request() -> bool:
    global _VR_FINISH_REQUESTED
    with _VR_FINISH_REQUEST_LOCK:
        requested = _VR_FINISH_REQUESTED
        _VR_FINISH_REQUESTED = False
    return requested


def _finish_collection_requested() -> bool:
    with _VR_FINISH_REQUEST_LOCK:
        return _VR_FINISH_REQUESTED


def _set_vr_menu_open(open_state: bool) -> None:
    global _VR_MENU_OPEN
    with _VR_MENU_LOCK:
        _VR_MENU_OPEN = bool(open_state)
    with _VR_SESSION_STATUS_LOCK:
        _VR_SESSION_STATUS["menu_open"] = bool(open_state)


def _vr_menu_is_open() -> bool:
    with _VR_MENU_LOCK:
        return _VR_MENU_OPEN


def _set_vr_recording_operation(operation: str) -> None:
    with _VR_SESSION_STATUS_LOCK:
        _VR_SESSION_STATUS["recording_operation"] = operation


def _queue_vr_recording_control(action: str) -> None:
    global _VR_RECORDING_CONTROL_LAST_ACTION
    now = time.monotonic()
    with _VR_RECORDING_CONTROL_LOCK:
        if _VR_RECORDING_CONTROL_LAST_ACTION is not None:
            last_action, last_t = _VR_RECORDING_CONTROL_LAST_ACTION
            if action == last_action and now - last_t < _VR_RECORDING_CONTROL_DEBOUNCE_S:
                return
        _VR_RECORDING_CONTROL_LAST_ACTION = (action, now)
        _VR_RECORDING_CONTROL_REQUESTS.append(action)


def _consume_vr_recording_controls() -> list[str]:
    with _VR_RECORDING_CONTROL_LOCK:
        actions = list(_VR_RECORDING_CONTROL_REQUESTS)
        _VR_RECORDING_CONTROL_REQUESTS.clear()
    return actions


def _serve_webrtc_offer(handler: Any) -> None:
    try:
        length = int(handler.headers.get("Content-Length", "0") or "0")
    except ValueError:
        length = 0
    try:
        payload = json.loads(handler.rfile.read(length).decode("utf-8") if length else "{}")
    except Exception as exc:
        return _write_json_response(handler, json.dumps({"error": f"Invalid WebRTC offer: {exc}"}), status=400)

    loop = _ensure_webrtc_loop()
    future = _run_coroutine_threadsafe(
        _create_webrtc_answer(
            payload,
            output_dir=getattr(handler.__class__, "orbbec_output_dir", None),
            include_orbbec=getattr(handler.__class__, "orbbec_overlay_include_orbbec", True),
        ),
        loop,
    )
    try:
        answer = future.result(timeout=10)
    except ModuleNotFoundError as exc:
        return _write_json_response(
            handler,
            json.dumps({"error": f"{exc.name} is not installed. Install aiortc in the xlerobot env."}),
            status=503,
        )
    except Exception as exc:
        return _write_json_response(handler, json.dumps({"error": f"WebRTC offer failed: {exc}"}), status=500)
    _write_json_response(handler, json.dumps(answer))


def _ensure_webrtc_loop() -> Any:
    global _WEBRTC_LOOP, _WEBRTC_LOOP_THREAD
    if _WEBRTC_LOOP is not None:
        return _WEBRTC_LOOP

    import asyncio

    loop = asyncio.new_event_loop()

    def run() -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    thread = threading.Thread(target=run, name="robot42-webrtc", daemon=True)
    thread.start()
    _WEBRTC_LOOP = loop
    _WEBRTC_LOOP_THREAD = thread
    return loop


def _run_coroutine_threadsafe(coro: Any, loop: Any) -> Any:
    import asyncio

    return asyncio.run_coroutine_threadsafe(coro, loop)


async def _create_webrtc_answer(
    offer_payload: dict[str, Any],
    *,
    output_dir: Path | None,
    include_orbbec: bool,
) -> dict[str, Any]:
    from aiortc import RTCPeerConnection, RTCSessionDescription

    offer = RTCSessionDescription(sdp=offer_payload["sdp"], type=offer_payload["type"])
    requested = offer_payload.get("feeds") or ["orbbec", "left_wrist", "right_wrist"]
    feeds = [name for name in requested if name in {"orbbec", "left_wrist", "right_wrist"}]
    if not include_orbbec:
        feeds = [name for name in feeds if name != "orbbec"]

    pc = RTCPeerConnection()
    with _WEBRTC_PEERS_LOCK:
        _WEBRTC_PEERS.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange() -> None:
        if pc.connectionState in {"failed", "closed", "disconnected"}:
            with _WEBRTC_PEERS_LOCK:
                _WEBRTC_PEERS.discard(pc)
            await pc.close()

    await pc.setRemoteDescription(offer)
    for name in feeds:
        pc.addTrack(_make_shared_jpeg_video_track(name, output_dir=output_dir))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "feeds": feeds}


def _make_shared_jpeg_video_track(name: str, *, output_dir: Path | None) -> Any:
    from aiortc import VideoStreamTrack

    class SharedJpegVideoTrack(VideoStreamTrack):
        def __init__(self, feed_name: str, feed_output_dir: Path | None) -> None:
            super().__init__()
            self.feed_name = feed_name
            self.feed_output_dir = feed_output_dir
            self.last_jpeg: bytes | None = None

        async def recv(self) -> Any:
            import cv2
            import numpy as np
            from av import VideoFrame

            pts, time_base = await self.next_timestamp()
            frame = _latest_webrtc_jpeg(self.feed_name, self.feed_output_dir)
            if frame is not None:
                self.last_jpeg = frame[0]

            if self.last_jpeg is None:
                rgb = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                data = np.frombuffer(self.last_jpeg, dtype=np.uint8)
                bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if bgr is None:
                    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
                else:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            video_frame = VideoFrame.from_ndarray(rgb, format="rgb24")
            video_frame.pts = pts
            video_frame.time_base = time_base
            return video_frame

    return SharedJpegVideoTrack(name, output_dir)


def _latest_webrtc_jpeg(name: str, output_dir: Path | None) -> tuple[bytes, int, int, float] | None:
    if name == "orbbec":
        return _latest_orbbec_jpeg_from_dir(output_dir)
    return _latest_vr_camera_jpeg(name)


def _serve_vr_camera_file(handler: Any) -> None:
    parsed = urllib.parse.urlparse(handler.path)
    rel = parsed.path.removeprefix("/vr-camera/")
    name, suffix = Path(rel).stem, Path(rel).suffix
    if not name or suffix not in {".ppm", ".json", ".mjpg", ".jpg"}:
        handler.send_error(404, "Unknown VR camera endpoint")
        return

    if suffix == ".mjpg":
        return _serve_vr_camera_mjpeg_stream(handler, name)
    if suffix == ".jpg":
        return _serve_vr_camera_jpeg_snapshot(handler, name)

    with _VR_CAMERA_FRAMES_LOCK:
        frame = _VR_CAMERA_FRAMES.get(name)
    if frame is None:
        handler.send_error(404, f"VR camera frame not ready: {name}")
        return

    ppm, width, height, captured_at = frame
    if suffix == ".json":
        payload = (
            f'{{"name":"{name}","width":{width},"height":{height},'
            f'"captured_at":{captured_at:.6f}}}'
        ).encode("utf-8")
        content_type = "application/json"
        content = payload
    else:
        content_type = "image/x-portable-pixmap"
        content = ppm

    try:
        _write_response(handler, content, content_type)
    except (BrokenPipeError, ConnectionResetError, ssl.SSLError):
        return
    except Exception as exc:
        print(f"Error serving VR camera frame {name}: {exc}")
        handler.send_error(500, "Could not serve VR camera frame")


def _serve_vr_camera_mjpeg_stream(handler: Any, name: str) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={_MJPEG_BOUNDARY.decode()}")
    handler.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
    handler.send_header("Pragma", "no-cache")
    handler.end_headers()
    _stream_jpeg_frames(handler, lambda: _latest_vr_camera_jpeg(name), fps=30)


def _serve_vr_camera_jpeg_snapshot(handler: Any, name: str) -> None:
    frame = _latest_vr_camera_jpeg(name)
    if frame is None:
        handler.send_error(404, f"VR camera JPEG not ready: {name}")
        return
    jpeg, _width, _height, _captured_at = frame
    try:
        _write_response(handler, jpeg, "image/jpeg")
    except (BrokenPipeError, ConnectionResetError, ssl.SSLError):
        return


def _latest_vr_camera_jpeg(name: str) -> tuple[bytes, int, int, float] | None:
    with _VR_CAMERA_JPEGS_LOCK:
        return _VR_CAMERA_JPEGS.get(name)


def _stream_jpeg_frames(
    handler: Any,
    latest_frame: Any,
    *,
    fps: int,
) -> None:
    last_timestamp = 0.0
    delay = 1.0 / max(1, fps)
    try:
        while True:
            frame = latest_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            jpeg, _width, _height, captured_at = frame
            if captured_at <= last_timestamp:
                time.sleep(min(0.01, delay))
                continue
            last_timestamp = captured_at
            handler.wfile.write(b"--" + _MJPEG_BOUNDARY + b"\r\n")
            handler.wfile.write(b"Content-Type: image/jpeg\r\n")
            handler.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
            handler.wfile.write(jpeg)
            handler.wfile.write(b"\r\n")
            handler.wfile.flush()
            time.sleep(delay)
    except (BrokenPipeError, ConnectionResetError, ssl.SSLError, ConnectionAbortedError):
        return


def _serve_orbbec_file(handler: Any, filename: str, content_type: str) -> None:
    output_dir = getattr(handler.__class__, "orbbec_output_dir", None)
    if output_dir is None:
        handler.send_error(404, "Orbbec RGB output directory is not configured")
        return
    path = Path(output_dir) / filename
    if not path.exists():
        handler.send_error(404, f"Orbbec RGB frame not ready: {filename}")
        return
    try:
        content = path.read_bytes()
        _write_response(handler, content, content_type)
    except (BrokenPipeError, ConnectionResetError, ssl.SSLError):
        return
    except Exception as exc:
        print(f"Error serving Orbbec RGB file {path}: {exc}")
        handler.send_error(500, "Could not serve Orbbec RGB frame")


def _serve_orbbec_mjpeg_stream(handler: Any) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={_MJPEG_BOUNDARY.decode()}")
    handler.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
    handler.send_header("Pragma", "no-cache")
    handler.end_headers()
    _stream_jpeg_frames(handler, lambda: _latest_orbbec_jpeg(handler), fps=30)


def _serve_orbbec_jpeg_snapshot(handler: Any) -> None:
    frame = _latest_orbbec_jpeg(handler)
    if frame is None:
        handler.send_error(404, "Orbbec RGB JPEG not ready")
        return
    jpeg, _width, _height, _captured_at = frame
    try:
        _write_response(handler, jpeg, "image/jpeg")
    except (BrokenPipeError, ConnectionResetError, ssl.SSLError):
        return


def _latest_orbbec_jpeg(handler: Any) -> tuple[bytes, int, int, float] | None:
    output_dir = getattr(handler.__class__, "orbbec_output_dir", None)
    return _latest_orbbec_jpeg_from_dir(output_dir)


def _latest_orbbec_jpeg_from_dir(output_dir: Path | None) -> tuple[bytes, int, int, float] | None:
    if output_dir is None:
        return None
    path = Path(output_dir) / "latest.ppm"
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None

    global _ORBBEC_JPEG_CACHE
    with _ORBBEC_JPEG_CACHE_LOCK:
        if (
            _ORBBEC_JPEG_CACHE is not None
            and _ORBBEC_JPEG_CACHE[0] == path
            and _ORBBEC_JPEG_CACHE[1] == stat.st_mtime_ns
            and _ORBBEC_JPEG_CACHE[2] == stat.st_size
        ):
            return (_ORBBEC_JPEG_CACHE[3], 0, 0, _ORBBEC_JPEG_CACHE[4])

    try:
        jpeg, width, height = _ppm_file_to_jpeg(path)
    except Exception:
        return None

    with _ORBBEC_JPEG_CACHE_LOCK:
        _ORBBEC_JPEG_CACHE = (path, stat.st_mtime_ns, stat.st_size, jpeg, stat.st_mtime)
    return (jpeg, width, height, stat.st_mtime)


def _ppm_file_to_jpeg(path: Path) -> tuple[bytes, int, int]:
    import cv2
    import numpy as np

    data = path.read_bytes()
    offset = 0

    def skip_ws_and_comments() -> None:
        nonlocal offset
        while offset < len(data):
            value = data[offset]
            if value == 35:
                while offset < len(data) and data[offset] != 10:
                    offset += 1
            elif value in (9, 10, 13, 32):
                offset += 1
            else:
                break

    def token() -> bytes:
        nonlocal offset
        skip_ws_and_comments()
        start = offset
        while offset < len(data) and data[offset] not in (9, 10, 13, 32, 35):
            offset += 1
        return data[start:offset]

    magic = token()
    width = int(token())
    height = int(token())
    max_value = int(token())
    skip_ws_and_comments()
    if magic != b"P6" or max_value != 255:
        raise ValueError(f"Unsupported PPM header in {path}")
    rgb = np.frombuffer(data, dtype=np.uint8, count=width * height * 3, offset=offset)
    rgb = rgb.reshape((height, width, 3))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY])
    if not ok:
        raise RuntimeError(f"Could not encode JPEG for {path}")
    return encoded.tobytes(), width, height


def _orbbec_vr_overlay_js(
    *,
    include_orbbec: bool,
    video_display: VrVideoDisplayConfig,
    video_streams_enabled: bool,
) -> str:
    script = r"""
(function () {
  const FEEDS = [
    {
      name: 'orbbec',
      url: '/orbbec/latest.jpg',
      canvasId: 'orbbec-rgb-canvas',
      width: 0.8,
      height: 0.6,
      position: [0, -0.18, -1.05],
      markerPosition: [-0.42, 0.32, -1.04],
      logName: 'Orbbec RGB',
      displayGain: __ORBBEC_GAIN__,
      displayGamma: __ORBBEC_GAMMA__,
      displayBias: __ORBBEC_BIAS__,
      pollMs: 120
    },
    {
      name: 'left_wrist',
      url: '/vr-camera/left_wrist.jpg',
      canvasId: 'left-wrist-rgb-canvas',
      width: 0.32,
      height: 0.24,
      position: [-0.17, -0.56, -1.02],
      markerPosition: [-0.34, -0.42, -1.01],
      logName: 'Left wrist',
      displayGain: __WRIST_GAIN__,
      displayGamma: __WRIST_GAMMA__,
      displayBias: __WRIST_BIAS__,
      pollMs: 160
    },
    {
      name: 'right_wrist',
      url: '/vr-camera/right_wrist.jpg',
      canvasId: 'right-wrist-rgb-canvas',
      width: 0.32,
      height: 0.24,
      position: [0.17, -0.56, -1.02],
      markerPosition: [0.0, -0.42, -1.01],
      logName: 'Right wrist',
      displayGain: __WRIST_GAIN__,
      displayGamma: __WRIST_GAMMA__,
      displayBias: __WRIST_BIAS__,
      pollMs: 160
    }
  ];
  const INCLUDE_ORBBEC = __INCLUDE_ORBBEC__;
  const VIDEO_STREAMS_ENABLED = __VIDEO_STREAMS_ENABLED__;
  const ACTIVE_FEEDS = VIDEO_STREAMS_ENABLED ? FEEDS.filter(feed => INCLUDE_ORBBEC || feed.name !== 'orbbec') : [];
  const overlays = new Map();
  let lastStatusLog = 0;
  let webRtcStarted = false;
  let webRtcFailed = false;
  let webRtcPc = null;
  let lastRobot42StatusText = '';
  let robot42LatestStatus = {};
  let menuOpen = false;
  let menuButtons = [];
  let menuSignature = '';
  let hoveredMenuAction = null;
  let activePointerHand = null;
  let pointerReticle = null;
  let pendingRecordingAction = null;

  function statusTextFromPayload(payload) {
    const status = payload && payload.robot42 ? payload.robot42 : {};
    robot42LatestStatus = status;
    const missing = Array.isArray(status.recording_missing) ? status.recording_missing : [];
    let text = '';
    if (status.recording_operation === 'saving') text = 'Saving...';
    else if (status.recording_operation === 'starting') text = 'Starting...';
    else if (status.recording_operation === 'cancelling') text = 'Cancelling...';
    else if (status.recording_operation === 'finishing') text = 'FINISHING';
    else if (status.finish_requested) text = 'FINISHING';
    if (!text && status.recording_active && missing.length > 0) {
      text = status.menu_open ? `MENU - Missing ${missing[0]}` : `REC BLOCKED: ${missing[0]}`;
    } else if (!text && status.recording_active) {
      text = status.menu_open ? 'MENU - Recording...' : 'Recording...';
    } else if (!text && status.episode_phase === 'await_finish') {
      text = status.menu_open ? 'MENU - Save or cancel' : 'Action ready - open menu';
    } else if (!text && status.episode_phase === 'await_start') {
      text = status.menu_open ? 'MENU - Record or finish' : 'Ready - open menu';
    } else if (!text && status.menu_open) {
      text = status.recording_enabled ? 'MENU - Idle' : 'MENU';
    } else if (!text) {
      text = status.recording_enabled ? 'Idle' : 'Teleop';
    }
    const count = Math.max(0, Number.parseInt(status.session_episode_count || 0, 10) || 0);
    return status.recording_enabled ? `${text} | Saved: ${count}` : text;
  }

  function statusColor(text) {
    if (text.includes('BLOCKED') || text.includes('Missing')) return '#b91c1c';
    if (text.includes('Saving') || text.includes('Starting') || text.includes('Cancelling')) return '#b7791f';
    if (text.includes('Recording')) return '#cf2e2e';
    if (text.startsWith('FINISHING')) return '#6b21a8';
    if (text.includes('Save') || text.includes('Action ready')) return '#b7791f';
    if (text.includes('Ready')) return '#2563eb';
    if (text.startsWith('MENU')) return '#111827';
    return '#333';
  }

  function ensurePageStatus() {
    if (document.getElementById('robot42-recording-status')) return;
    const badge = document.createElement('div');
    badge.id = 'robot42-recording-status';
    badge.textContent = 'IDLE';
    badge.style.position = 'fixed';
    badge.style.left = '18px';
    badge.style.bottom = '18px';
    badge.style.zIndex = '10000';
    badge.style.padding = '10px 14px';
    badge.style.borderRadius = '8px';
    badge.style.background = '#333';
    badge.style.color = '#fff';
    badge.style.font = '700 14px system-ui, -apple-system, BlinkMacSystemFont, sans-serif';
    badge.style.boxShadow = '0 2px 10px rgba(0,0,0,0.25)';
    document.body.appendChild(badge);
  }

  function ensureVrStatus() {
    if (document.getElementById('robot42-recording-status-vr')) return;
    const headset = document.querySelector('#headset');
    if (!headset) return;
    const label = document.createElement('a-text');
    label.id = 'robot42-recording-status-vr';
    label.setAttribute('value', 'IDLE');
    label.setAttribute('align', 'center');
    label.setAttribute('width', '2.2');
    label.setAttribute('color', '#fff');
    label.setAttribute('position', '0 0.16 -0.99');
    label.setAttribute('side', 'double');
    label.setAttribute('geometry', 'primitive: plane; width: 1.08; height: 0.075');
    label.setAttribute('material', 'color: #333; shader: flat; side: double');
    headset.appendChild(label);
  }

  async function postJson(url, payload) {
    try {
      await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload || {})
      });
    } catch (error) {
      console.warn(`[Robot42] request failed: ${url}`, error);
    }
  }

  function setLocalMenuOpen(open) {
    menuOpen = !!open;
    applyMenuVisualState();
    ensureRecordingMenu();
    updateControllerPointer();
    updateRecordingMenu();
  }

  function setMenuOpen(open) {
    setLocalMenuOpen(open);
    postJson('/api/teleop-menu', { open: menuOpen });
    setTimeout(pollRobot42Status, 80);
  }

  async function requestRecordingAction(action) {
    if (!action || pendingRecordingAction) return;
    pendingRecordingAction = action;
    setMenuHover(null);
    setLocalMenuOpen(false);
    const localStatus = {
      record: 'Starting...',
      save: 'Saving...',
      cancel: 'Cancelling...',
      finish: 'FINISHING'
    }[action];
    if (localStatus) applyRobot42Status(localStatus);
    try {
      await postJson('/api/recording-control', { action });
    } finally {
      setTimeout(() => {
        pendingRecordingAction = null;
        pollRobot42Status();
      }, 450);
    }
  }

  function recordingMenuSpecs() {
    const status = robot42LatestStatus || {};
    if (!status.recording_enabled) {
      return [
        { action: 'record', label: 'Resume', color: '#2563eb' },
        { action: 'finish', label: 'Exit', color: '#6b21a8' }
      ];
    }
    if (status.recording_active) {
      return [
        { action: 'save', label: 'Save', color: '#1f8b4c' },
        { action: 'cancel', label: 'Cancel', color: '#b91c1c' },
        { action: 'finish', label: 'Finish', color: '#6b21a8' }
      ];
    }
    return [
      { action: 'record', label: 'Record', color: '#2563eb' },
      { action: 'finish', label: 'Finish', color: '#6b21a8' }
    ];
  }

  function ensureRecordingMenu() {
    if (document.getElementById('robot42-recording-menu')) return;
    const headset = document.querySelector('#headset');
    if (!headset) return;

    const root = document.createElement('a-entity');
    root.id = 'robot42-recording-menu';
    root.setAttribute('position', '0 0.055 -0.985');
    root.setAttribute('visible', 'false');

    const backdrop = document.createElement('a-plane');
    backdrop.id = 'robot42-recording-menu-backdrop';
    backdrop.setAttribute('width', '0.88');
    backdrop.setAttribute('height', '0.21');
    backdrop.setAttribute('position', '0 0 0');
    backdrop.setAttribute('material', 'color: #050816; opacity: 0.82; transparent: true; shader: flat; side: double');
    root.appendChild(backdrop);

    const hint = document.createElement('a-text');
    hint.id = 'robot42-recording-menu-hint';
    hint.setAttribute('value', 'Episodes saved this session: 0');
    hint.setAttribute('align', 'center');
    hint.setAttribute('width', '1.4');
    hint.setAttribute('color', '#cbd5e1');
    hint.setAttribute('position', '0 0.065 0.006');
    hint.setAttribute('side', 'double');
    root.appendChild(hint);

    headset.appendChild(root);
    updateRecordingMenu();
  }

  function tagMenuButtonObject(el, action) {
    const assign = () => {
      if (!el.object3D) return;
      el.object3D.traverse(obj => {
        obj.userData.robot42MenuAction = action;
      });
    };
    assign();
    el.addEventListener('loaded', assign);
  }

  function updateRecordingMenu() {
    const root = document.getElementById('robot42-recording-menu');
    if (!root) return;
    root.setAttribute('visible', menuOpen ? 'true' : 'false');

    const specs = recordingMenuSpecs();
    const hint = document.getElementById('robot42-recording-menu-hint');
    if (hint) {
      const count = Math.max(
        0,
        Number.parseInt((robot42LatestStatus || {}).session_episode_count || 0, 10) || 0
      );
      hint.setAttribute('value', `Episodes saved this session: ${count}`);
    }
    const signature = specs.map(spec => `${spec.action}:${spec.label}`).join('|');
    if (signature === menuSignature && menuButtons.length > 0) {
      setMenuHover(hoveredMenuAction);
      return;
    }
    menuSignature = signature;

    const existing = root.querySelectorAll('.robot42-menu-button');
    existing.forEach(el => el.parentNode && el.parentNode.removeChild(el));
    menuButtons = [];

    const spacing = 0.25;
    const startX = -spacing * (specs.length - 1) / 2;
    specs.forEach((spec, index) => {
      const button = document.createElement('a-plane');
      button.classList.add('robot42-menu-button', 'clickable');
      button.setAttribute('width', '0.22');
      button.setAttribute('height', '0.065');
      button.setAttribute('position', `${startX + spacing * index} -0.035 0.008`);
      button.setAttribute('material', `color: ${spec.color}; opacity: 0.95; transparent: true; shader: flat; side: double`);
      button.addEventListener('click', event => {
        event.stopPropagation();
        requestRecordingAction(spec.action);
      });
      button.addEventListener('mouseenter', () => setMenuHover(spec.action));
      button.addEventListener('mouseleave', () => {
        if (hoveredMenuAction === spec.action) setMenuHover(null);
      });
      tagMenuButtonObject(button, spec.action);

      const label = document.createElement('a-text');
      label.setAttribute('value', spec.label);
      label.setAttribute('align', 'center');
      label.setAttribute('width', '0.72');
      label.setAttribute('color', '#fff');
      label.setAttribute('position', '0 0 0.006');
      label.setAttribute('side', 'double');
      button.appendChild(label);
      root.appendChild(button);
      menuButtons.push({ el: button, action: spec.action, color: spec.color });
    });
    setMenuHover(hoveredMenuAction);
    if (activePointerHand) {
      const el = controllerEl(activePointerHand);
      if (el && el.components && el.components.raycaster) {
        try {
          el.components.raycaster.refreshObjects();
        } catch (error) {
          // The raycaster will refresh on its next tick if the scene is not ready yet.
        }
      }
    }
  }

  function setMenuHover(action) {
    hoveredMenuAction = action;
    menuButtons.forEach(button => {
      const isHovered = button.action === action;
      button.el.setAttribute(
        'material',
        `color: ${isHovered ? '#f8fafc' : button.color}; opacity: 0.96; transparent: true; shader: flat; side: double`
      );
      const label = button.el.querySelector('a-text');
      if (label) label.setAttribute('color', isHovered ? '#111827' : '#fff');
    });
  }

  function applyMenuVisualState() {
    overlays.forEach(nodes => {
      if (nodes.mesh && nodes.mesh.material && nodes.mesh.material.uniforms) {
        const baseGain = nodes.baseDisplayGain || 1.0;
        nodes.mesh.material.uniforms.displayGain.value = menuOpen ? baseGain * 0.32 : baseGain;
      }
      if (nodes.marker && nodes.marker.material) {
        nodes.marker.material.transparent = true;
        nodes.marker.material.opacity = menuOpen ? 0.25 : 1.0;
      }
    });
    const root = document.getElementById('robot42-recording-menu');
    if (root) root.setAttribute('visible', menuOpen ? 'true' : 'false');
  }

  function applyRobot42Status(text) {
    if (!text || text === lastRobot42StatusText) return;
    lastRobot42StatusText = text;
    const color = statusColor(text);
    const pageBadge = document.getElementById('robot42-recording-status');
    if (pageBadge) {
      pageBadge.textContent = text;
      pageBadge.style.background = color;
    }
    const vrBadge = document.getElementById('robot42-recording-status-vr');
    if (vrBadge) {
      vrBadge.setAttribute('value', text);
      vrBadge.setAttribute('material', `color: ${color}; shader: flat; side: double`);
    }
  }

  async function pollRobot42Status() {
    try {
      const response = await fetch('/api/status', { cache: 'no-store' });
      if (!response.ok) return;
      const payload = await response.json();
      const status = payload && payload.robot42 ? payload.robot42 : {};
      if (typeof status.menu_open === 'boolean' && status.menu_open !== menuOpen) {
        setLocalMenuOpen(status.menu_open);
      }
      applyRobot42Status(statusTextFromPayload(payload));
      updateRecordingMenu();
    } catch (error) {
      // Keep the last visible status during transient network hiccups.
    }
  }

  function controllerGamepad(hand) {
    const el = document.querySelector(hand === 'right' ? '#rightHand' : '#leftHand');
    return el && el.components && el.components['tracked-controls']
      ? el.components['tracked-controls'].controller?.gamepad
      : null;
  }

  function controllerEl(hand) {
    return document.querySelector(hand === 'right' ? '#rightHand' : '#leftHand');
  }

  function ensureAFramePointer(hand) {
    const el = controllerEl(hand);
    if (!el) return null;
    el.setAttribute(
      'raycaster',
      'objects: .robot42-menu-button; showLine: true; far: 3.0; interval: 0; lineColor: #38bdf8; lineOpacity: 0.95'
    );
    el.setAttribute(
      'cursor',
      'rayOrigin: entity; fuse: false; downEvents: triggerdown; upEvents: triggerup'
    );
    return el;
  }

  function ensurePointerReticle() {
    if (pointerReticle) return pointerReticle;
    if (typeof THREE === 'undefined') return null;
    const scene = document.querySelector('a-scene');
    if (!scene || !scene.object3D) return null;
    pointerReticle = new THREE.Mesh(
      new THREE.RingGeometry(0.011, 0.018, 32),
      new THREE.MeshBasicMaterial({
        color: 0x38bdf8,
        transparent: true,
        opacity: 0.95,
        side: THREE.DoubleSide,
        depthTest: false
      })
    );
    pointerReticle.name = 'robot42-recording-menu-pointer-reticle';
    pointerReticle.visible = false;
    pointerReticle.renderOrder = 5000;
    scene.object3D.add(pointerReticle);
    return pointerReticle;
  }

  function disableAFramePointer(hand) {
    const el = controllerEl(hand);
    if (!el) return;
    if (el.components && el.components.raycaster) {
      el.setAttribute('raycaster', 'enabled', false);
    }
    if (el.components && el.components.cursor) {
      el.setAttribute('cursor', 'enabled', false);
    }
  }

  function enableAFramePointer(hand) {
    const el = ensureAFramePointer(hand);
    if (!el) return;
    el.setAttribute('raycaster', 'enabled', true);
    el.setAttribute('cursor', 'enabled', true);
    if (el.components && el.components.raycaster) {
      try {
        el.components.raycaster.refreshObjects();
      } catch (error) {
        // A-Frame may refresh automatically depending on lifecycle timing.
      }
    }
  }

  function updateControllerPointer() {
    const reticle = ensurePointerReticle();
    if (!menuOpen) {
      disableAFramePointer('left');
      disableAFramePointer('right');
      activePointerHand = null;
      setMenuHover(null);
      if (reticle) reticle.visible = false;
      return;
    }

    const hand = robot42LatestStatus.menu_pointer_hand === 'right' ? 'right' : 'left';
    if (activePointerHand && activePointerHand !== hand) {
      disableAFramePointer(activePointerHand);
    }
    activePointerHand = hand;
    enableAFramePointer(hand);
    updatePointerReticle(hand);
  }

  function updatePointerReticle(hand) {
    const reticle = ensurePointerReticle();
    const el = controllerEl(hand);
    const raycaster = el && el.components ? el.components.raycaster : null;
    if (!reticle || !raycaster || !raycaster.intersectedEls || raycaster.intersectedEls.length === 0) {
      if (reticle) reticle.visible = false;
      return;
    }

    let closest = null;
    raycaster.intersectedEls.forEach(target => {
      const intersection = raycaster.getIntersection(target);
      if (intersection && (!closest || intersection.distance < closest.distance)) {
        closest = intersection;
      }
    });
    if (!closest || !closest.point) {
      reticle.visible = false;
      return;
    }

    reticle.visible = true;
    reticle.position.copy(closest.point);
    const headset = document.querySelector('#headset');
    if (headset && headset.object3D) {
      reticle.quaternion.copy(headset.object3D.quaternion);
    }
  }

  function ensurePanel(feed) {
    const headset = document.querySelector('#headset');
    if (!headset || !headset.object3D || typeof THREE === 'undefined') return null;

    let canvas = document.getElementById(feed.canvasId);
    if (!canvas) {
      canvas = document.createElement('canvas');
      canvas.id = feed.canvasId;
      canvas.width = 640;
      canvas.height = 480;
      canvas.style.display = 'none';
      document.body.appendChild(canvas);
    }

    if (!overlays.has(feed.name)) {
      const video = document.createElement('video');
      video.id = `${feed.name}-webrtc-video`;
      video.autoplay = true;
      video.muted = true;
      video.playsInline = true;
      video.setAttribute('playsinline', '');
      video.setAttribute('webkit-playsinline', '');
      video.style.display = 'none';
      document.body.appendChild(video);

      const texture = new THREE.VideoTexture(video);
      texture.minFilter = THREE.LinearFilter;
      texture.magFilter = THREE.LinearFilter;
      texture.generateMipmaps = false;
      if ('SRGBColorSpace' in THREE) texture.colorSpace = THREE.SRGBColorSpace;

      const material = new THREE.ShaderMaterial({
        uniforms: {
          map: { value: texture },
          displayGain: { value: feed.displayGain || 1.0 },
          displayGamma: { value: feed.displayGamma || 1.0 },
          displayBias: { value: feed.displayBias || 0.0 }
        },
        vertexShader: `
          varying vec2 vUv;
          void main() {
            vUv = uv;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          }
        `,
        fragmentShader: `
          uniform sampler2D map;
          uniform float displayGain;
          uniform float displayGamma;
          uniform float displayBias;
          varying vec2 vUv;
          void main() {
            vec4 color = texture2D(map, vUv);
            vec3 rgb = max(color.rgb * displayGain + vec3(displayBias), vec3(0.0));
            rgb = pow(rgb, vec3(max(displayGamma, 0.01)));
            gl_FragColor = vec4(clamp(rgb, 0.0, 1.0), color.a);
          }
        `,
        side: THREE.DoubleSide,
        transparent: false,
        toneMapped: false
      });
      const mesh = new THREE.Mesh(new THREE.PlaneGeometry(feed.width, feed.height), material);
      mesh.name = `${feed.name}-rgb-three-overlay`;
      mesh.position.set(feed.position[0], feed.position[1], feed.position[2]);
      mesh.renderOrder = 999;
      headset.object3D.add(mesh);

      const marker = new THREE.Mesh(
        new THREE.PlaneGeometry(feed.width * 0.1, feed.width * 0.1),
        new THREE.MeshBasicMaterial({ color: 0x00ff00, side: THREE.DoubleSide })
      );
      marker.name = `${feed.name}-rgb-marker`;
      marker.position.set(feed.markerPosition[0], feed.markerPosition[1], feed.markerPosition[2]);
      marker.renderOrder = 1000;
      headset.object3D.add(marker);

	      overlays.set(feed.name, {
	        canvas,
	        video,
	        texture,
	        mesh,
	        marker,
	        loading: false,
	        lastRequestAt: 0,
	        frameCount: 0,
	        baseDisplayGain: feed.displayGain || 1.0
	      });
      applyMenuVisualState();
      console.log(`[${feed.logName}] WebRTC headset overlay created`);
    }

    return overlays.get(feed.name);
  }

  function updateVideoTexture(feed) {
    const nodes = ensurePanel(feed);
    if (!nodes) return;
    const videoReady = nodes.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA;
    if (videoReady) {
      nodes.texture.needsUpdate = true;
      nodes.frameCount += 1;
      if (nodes.marker) nodes.marker.material.color.setHex(0x00ff00);
      if (Date.now() - lastStatusLog > 3000) {
        console.log(`[${feed.logName}] WebRTC video ${nodes.video.videoWidth || 0}x${nodes.video.videoHeight || 0}`);
        lastStatusLog = Date.now();
      }
    } else if (nodes.marker) {
      nodes.marker.material.color.setHex(webRtcFailed ? 0xff0000 : 0xffff00);
    }
  }

  function waitForIceGatheringComplete(pc) {
    if (pc.iceGatheringState === 'complete') return Promise.resolve();
    return new Promise(resolve => {
      const check = () => {
        if (pc.iceGatheringState === 'complete') {
          pc.removeEventListener('icegatheringstatechange', check);
          resolve();
        }
      };
      pc.addEventListener('icegatheringstatechange', check);
    });
  }

  async function startWebRtc() {
    if (!VIDEO_STREAMS_ENABLED || ACTIVE_FEEDS.length === 0 || webRtcStarted || webRtcFailed) return;
    webRtcStarted = true;
    try {
      ACTIVE_FEEDS.forEach(ensurePanel);
      const pc = new RTCPeerConnection({ iceServers: [] });
      webRtcPc = pc;
      let trackIndex = 0;

      ACTIVE_FEEDS.forEach(() => {
        pc.addTransceiver('video', { direction: 'recvonly' });
      });

      pc.ontrack = (event) => {
        const feed = ACTIVE_FEEDS[trackIndex++];
        if (!feed) return;
        const nodes = ensurePanel(feed);
        if (!nodes) return;
        nodes.video.srcObject = new MediaStream([event.track]);
        nodes.video.play().catch(error => {
          console.warn(`[${feed.logName}] Video play was delayed`, error);
        });
        console.log(`[${feed.logName}] WebRTC track attached`);
      };

      pc.onconnectionstatechange = () => {
        console.log(`[Robot42 WebRTC] connection state: ${pc.connectionState}`);
        if (['failed', 'disconnected', 'closed'].includes(pc.connectionState)) {
          webRtcFailed = true;
        }
      };

      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      await waitForIceGatheringComplete(pc);

      const response = await fetch('/webrtc/offer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sdp: pc.localDescription.sdp,
          type: pc.localDescription.type,
          feeds: ACTIVE_FEEDS.map(feed => feed.name)
        })
      });
      const answer = await response.json();
      if (!response.ok) {
        throw new Error(answer.error || `WebRTC signaling failed with HTTP ${response.status}`);
      }
      await pc.setRemoteDescription(answer);
      console.log('[Robot42 WebRTC] video answer installed', answer.feeds || []);
    } catch (error) {
      webRtcFailed = true;
      console.error('[Robot42 WebRTC] video setup failed', error);
    }
  }

  function start() {
    ensurePageStatus();
    ensureVrStatus();
    ensureRecordingMenu();
    pollRobot42Status();
    setInterval(() => {
      ensurePageStatus();
      ensureVrStatus();
      ensureRecordingMenu();
      pollRobot42Status();
    }, 500);
    startWebRtc();
    function renderStreams() {
      ensureVrStatus();
      ensureRecordingMenu();
      updateControllerPointer();
      ACTIVE_FEEDS.forEach(updateVideoTexture);
      requestAnimationFrame(renderStreams);
    }
    renderStreams();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }
})();
"""
    replacements = {
        "__INCLUDE_ORBBEC__": "true" if include_orbbec else "false",
        "__VIDEO_STREAMS_ENABLED__": "true" if video_streams_enabled else "false",
        "__WRIST_GAIN__": _js_float(video_display.wrist_gain),
        "__WRIST_GAMMA__": _js_float(video_display.wrist_gamma),
        "__WRIST_BIAS__": _js_float(video_display.wrist_bias),
        "__ORBBEC_GAIN__": _js_float(video_display.orbbec_gain),
        "__ORBBEC_GAMMA__": _js_float(video_display.orbbec_gamma),
        "__ORBBEC_BIAS__": _js_float(video_display.orbbec_bias),
    }
    for key, value in replacements.items():
        script = script.replace(key, value)
    return script


def _js_float(value: float) -> str:
    return format(float(value), ".6g")


def _run_keyboard_backend(
    *,
    repo_root: Path,
    robot: Any,
    fps: int,
    recording: RecordingSession | None,
    start_key: str,
    stop_key: str,
    quit_key: str,
    auto_restore_calibration: bool,
) -> int:
    from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardTeleopConfig
    from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
    from lerobot.utils.errors import DeviceNotConnectedError
    from lerobot.utils.robot_utils import precise_sleep
    from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
    import numpy as np

    keyboard_module = _load_example_module(
        repo_root / "software" / "examples" / "4_xlerobot_teleop_keyboard.py",
        "xlerobot_playground._real_keyboard_example",
    )

    keyboard_config = KeyboardTeleopConfig()
    keyboard = KeyboardTeleop(keyboard_config)
    previous_pressed_keys: set[str] = set()

    _connect_robot(robot, auto_restore_calibration=auto_restore_calibration)
    init_rerun(session_name="xlerobot_real_keyboard_playground")
    keyboard.connect()

    obs = robot.get_observation()
    kin_left = keyboard_module.SO101Kinematics()
    kin_right = keyboard_module.SO101Kinematics()
    left_arm = keyboard_module.SimpleTeleopArm(kin_left, keyboard_module.LEFT_JOINT_MAP, obs, prefix="left")
    right_arm = keyboard_module.SimpleTeleopArm(kin_right, keyboard_module.RIGHT_JOINT_MAP, obs, prefix="right")
    head_control = keyboard_module.SimpleHeadControl(obs)

    left_arm.move_to_zero_position(robot)
    right_arm.move_to_zero_position(robot)
    head_control.move_to_zero_position(robot)
    _print_recording_guide(recording, start_key=start_key, stop_key=stop_key, quit_key=quit_key)

    try:
        while True:
            start_loop_t = time.perf_counter()
            try:
                pressed_keys = set(keyboard.get_action().keys())
            except DeviceNotConnectedError:
                break
            newly_pressed = pressed_keys - previous_pressed_keys
            previous_pressed_keys = pressed_keys
            if quit_key in newly_pressed:
                break
            if _consume_finish_collection_request():
                print("Finish dataset requested from VR UI.")
                break
            _handle_recording_hotkeys(
                recording,
                newly_pressed,
                start_key=start_key,
                stop_key=stop_key,
            )

            left_key_state = {
                action: (key in pressed_keys) for action, key in keyboard_module.LEFT_KEYMAP.items()
            }
            right_key_state = {
                action: (key in pressed_keys) for action, key in keyboard_module.RIGHT_KEYMAP.items()
            }

            if left_key_state.get("triangle"):
                left_arm.execute_rectangular_trajectory(robot, fps=fps)
                continue
            if right_key_state.get("triangle"):
                right_arm.execute_rectangular_trajectory(robot, fps=fps)
                continue
            if left_key_state.get("reset"):
                left_arm.move_to_zero_position(robot)
                continue
            if right_key_state.get("reset"):
                right_arm.move_to_zero_position(robot)
                continue
            if "?" in pressed_keys:
                head_control.move_to_zero_position(robot)
                continue

            left_arm.handle_keys(left_key_state)
            right_arm.handle_keys(right_key_state)
            head_control.handle_keys(left_key_state)

            left_action = left_arm.p_control_action(robot)
            right_action = right_arm.p_control_action(robot)
            head_action = head_control.p_control_action(robot)
            keyboard_keys = np.array(list(pressed_keys))
            base_action = robot._from_keyboard_to_base_action(keyboard_keys) or {}

            action = {**left_action, **right_action, **head_action, **base_action}
            sent_action = robot.send_action(action)

            obs = robot.get_observation()
            log_rerun_data(obs, sent_action)
            _record_frame_if_needed(recording, obs, sent_action)

            dt_s = time.perf_counter() - start_loop_t
            precise_sleep(max(0.0, 1 / fps - dt_s))
    finally:
        _finalize_recording(recording)
        try:
            robot.disconnect()
        finally:
            if keyboard.is_connected:
                keyboard.disconnect()
    return 0


def _run_vr_backend(
    *,
    interface: XLeRobotInterface,
    robot: Any,
    fps: int,
    recording: RecordingSession | None,
    start_key: str,
    stop_key: str,
    quit_key: str,
    xlevr_path: str | None,
    auto_restore_calibration: bool,
    orbbec_rgb: OrbbecRgbConfig,
    training_toggle_vr_button: str,
    training_discard_vr_button: str,
    vr_input_scale: float,
    vr_kp: float,
    vr_camera_hz: float,
    enable_squeeze_clutch: bool,
    arm_clutch_keys: VrArmClutchKeys,
    basket_pose: VrBasketPoseConfig,
    base_smoother: BaseSmoother,
    lock_base_while_recording: bool,
    arm_tuning: VrArmTuning,
    video_display: VrVideoDisplayConfig,
    vr_video_streams: bool,
    startup_pose: VrStartupPoseConfig,
) -> int:
    from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardTeleopConfig
    from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
    from lerobot.utils.errors import DeviceNotConnectedError
    from lerobot.utils.robot_utils import precise_sleep
    from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

    hotkeys = KeyboardTeleop(KeyboardTeleopConfig())
    previous_pressed_keys: set[str] = set()
    _set_vr_menu_open(False)
    _consume_vr_recording_controls()
    _consume_finish_collection_request()

    orbbec_process = _start_orbbec_rgb_sidecar(orbbec_rgb)
    _connect_robot(robot, auto_restore_calibration=auto_restore_calibration)
    init_rerun(session_name="xlerobot_real_vr_playground")
    hotkeys.connect()

    vr_overrides = {"kp": vr_kp}
    if xlevr_path is not None:
        vr_overrides["xlevr_path"] = xlevr_path
    if _install_xlevr_console_filter():
        print("XLeVR headset pose console spam suppressed.")
    vr_teleop = interface.make_vr_teleop(**vr_overrides)
    if enable_squeeze_clutch and _install_xlevr_squeeze_metadata_patch(vr_teleop, xlevr_path=xlevr_path):
        print("Quest squeeze metadata patch enabled for VR IK clutch.")
    if _enable_threaded_vr_http(vr_teleop):
        print("XLeVR HTTPS server patched to ThreadingHTTPServer for camera streams.")
    vr_teleop.connect(robot=robot)
    _configure_vr_runtime(vr_teleop, input_scale=vr_input_scale)
    _install_vr_arm_tuning(vr_teleop, arm_tuning)
    print(
        "VR base smoothing: "
        f"max {base_smoother.max_linear:.2f} m/s, "
        f"{base_smoother.max_angular:.0f} deg/s; "
        f"accel {base_smoother.linear_accel:.2f} m/s^2, "
        f"{base_smoother.angular_accel:.0f} deg/s^2."
    )
    if recording is not None and lock_base_while_recording:
        print("VR base safety: wheel commands are hard-stopped while an episode is recording.")
    installed = _install_orbbec_vr_overlay(
        vr_teleop,
        orbbec_rgb.output_dir,
        include_orbbec=orbbec_rgb.enabled,
        video_display=video_display,
        video_streams_enabled=vr_video_streams,
    )
    if installed:
        if vr_video_streams and orbbec_rgb.enabled:
            print("Orbbec RGB VR overlay enabled. Reload the Quest page if it was already open.")
        if vr_video_streams:
            print("VR arm camera panels enabled for camera names: left_wrist and right_wrist.")
        else:
            print("VR camera video streams disabled; recording menu/status overlay remains enabled.")
    elif orbbec_rgb.enabled:
        print("Orbbec RGB sidecar is running, but the XLeVR web overlay could not be installed.")
    _run_vr_startup_pose(robot, vr_teleop, startup_pose)
    print("Robot moved to ACTION_READY. Open the Quest page and start controller tracking.")
    _print_recording_guide(
        recording,
        start_key=start_key,
        stop_key=stop_key,
        quit_key=quit_key,
        controller="vr",
    )
    if basket_pose.enabled:
        print(
            "VR skill mode grab_to_basket: "
            f"active_arm={basket_pose.skill_arm}; "
            f"right B -> right basket, right A -> right ACTION_READY; "
            f"left Y -> left basket, left X -> left ACTION_READY; "
            f"`{basket_pose.full_reset_key}` or left thumbstick down resets both."
        )

    action_ready_targets = _capture_arm_pose_targets(robot)
    camera_interval_s = 1.0 / max(0.1, vr_camera_hz)
    next_camera_t = 0.0
    last_camera_warn_t = 0.0
    arm_clutch_state = {"left": False, "right": False}
    previous_vr_buttons: set[str] = set()
    previous_menu_open = _vr_menu_is_open()
    menu_pointer_hand = _vr_menu_pointer_hand(basket_pose)
    fixed_arm_modes, episode_boundary = _initial_vr_episode_boundary(recording)
    fixed_arm_motion_states: dict[str, FixedArmMotionState] = {}
    if episode_boundary.phase == "await_start":
        print("ACTION_READY locked while waiting for the first episode. Press left thumbstick or Record to begin.")
    _update_vr_session_status(recording, episode_boundary, menu_pointer_hand=menu_pointer_hand)

    try:
        while True:
            start_loop_t = time.perf_counter()
            try:
                pressed_keys = set(hotkeys.get_action().keys())
            except DeviceNotConnectedError:
                break
            newly_pressed = pressed_keys - previous_pressed_keys
            previous_pressed_keys = pressed_keys
            if quit_key in newly_pressed:
                break
            if _consume_finish_collection_request():
                print("Finish dataset requested from VR UI/controller.")
                break
            _handle_recording_hotkeys(
                recording,
                newly_pressed,
                start_key=start_key,
                stop_key=stop_key,
            )

            current_vr_buttons = _current_vr_button_state(vr_teleop)
            newly_pressed_vr_buttons = current_vr_buttons - previous_vr_buttons
            previous_vr_buttons = current_vr_buttons
            if newly_pressed_vr_buttons:
                print(f"VR buttons pressed: {', '.join(sorted(newly_pressed_vr_buttons))}")
            if _vr_button_spec_pressed("right:thumbstick", newly_pressed_vr_buttons):
                _set_vr_menu_open(not _vr_menu_is_open())
                print(f"VR recording menu {'opened' if _vr_menu_is_open() else 'closed'} from right thumbstick.")
            vr_controls = _map_vr_events_to_recording_controls(vr_teleop.get_vr_events())
            vr_decision = VRRecordingDecision(reset_robot=vr_controls.reset_robot)

            obs = _get_robot_observation(robot, use_camera=False)
            menu_actions = _consume_vr_recording_controls()
            if menu_actions:
                _apply_vr_recording_menu_actions(
                    menu_actions,
                    recording,
                    episode_boundary,
                    fixed_arm_modes,
                    fixed_arm_motion_states,
                    vr_teleop,
                    obs,
                    action_ready_targets,
                )

            left_thumb_recording_shortcut = (
                not _vr_menu_is_open()
                and _vr_button_spec_pressed("left:thumbstick", newly_pressed_vr_buttons)
            )
            left_thumb_shortcut_consumed = False
            if left_thumb_recording_shortcut:
                left_thumb_shortcut_consumed = _apply_left_thumb_recording_shortcut(
                    recording,
                    episode_boundary,
                    fixed_arm_modes,
                    fixed_arm_motion_states,
                    vr_teleop,
                    obs,
                    action_ready_targets,
                )
                if left_thumb_shortcut_consumed:
                    vr_decision = VRRecordingDecision()

            menu_open = _vr_menu_is_open()
            menu_pause_this_loop = menu_open or previous_menu_open
            if menu_open and not previous_menu_open:
                _sync_vr_teleop_to_current_pose(robot, vr_teleop)
                print("VR teleoperation paused for recording menu.")
            elif previous_menu_open and not menu_open:
                _sync_vr_teleop_to_current_pose(robot, vr_teleop)
                print("VR teleoperation menu closed; controls re-baselined.")
            previous_menu_open = menu_open

            if not menu_pause_this_loop and not left_thumb_shortcut_consumed:
                boundary_button_consumed = _handle_episode_boundary_buttons(
                    episode_boundary,
                    recording,
                    fixed_arm_modes,
                    fixed_arm_motion_states,
                    vr_teleop,
                    obs,
                    action_ready_targets,
                    newly_pressed_vr_buttons,
                    toggle_button=training_toggle_vr_button,
                    discard_button=training_discard_vr_button,
                )
                if not boundary_button_consumed:
                    if (
                        recording is not None
                        and _vr_button_spec_pressed(training_discard_vr_button, newly_pressed_vr_buttons)
                    ):
                        _discard_or_finish_recording_from_vr_button(recording, training_discard_vr_button)
                    elif (
                        recording is not None
                        and _vr_button_spec_pressed(training_toggle_vr_button, newly_pressed_vr_buttons)
                    ):
                        _toggle_recording_from_vr_button(recording, training_toggle_vr_button)
            if recording is not None:
                if left_thumb_shortcut_consumed:
                    vr_decision = VRRecordingDecision()
                else:
                    vr_decision = _decide_vr_recording_action(
                        recording.active,
                        vr_controls,
                    )
                _apply_vr_recording_decision(recording, vr_decision)
                if vr_decision.quit_session:
                    break
            if _consume_finish_collection_request():
                print("Finish dataset requested from VR UI/controller.")
                break

            if basket_pose.enabled and not menu_pause_this_loop:
                _update_fixed_arm_modes_from_controls(
                    fixed_arm_modes,
                    basket_pose,
                    newly_pressed,
                    newly_pressed_vr_buttons,
                )
            basket_reset_requested = basket_pose.enabled and basket_pose.full_reset_key in newly_pressed
            if menu_pause_this_loop:
                _set_fixed_arm_ik_pauses(vr_teleop, {"left": "menu", "right": "menu"})
                action = {}
            elif vr_decision.reset_robot or basket_reset_requested:
                _move_to_action_ready(robot, vr_teleop, startup_pose)
                action_ready_targets = _capture_arm_pose_targets(robot)
                fixed_arm_modes.clear()
                fixed_arm_motion_states.clear()
                episode_boundary.phase = "idle"
                episode_boundary.held_sides.clear()
                _set_fixed_arm_ik_pauses(vr_teleop, {})
                action = {}
                if basket_reset_requested:
                    print("Grab-to-basket reset: both arms returned to ACTION_READY.")
            else:
                effective_fixed_arm_modes = _effective_fixed_arm_modes(fixed_arm_modes, basket_pose)
                _set_fixed_arm_ik_pauses(vr_teleop, effective_fixed_arm_modes)
                _update_vr_arm_clutch(
                    vr_teleop,
                    obs,
                    arm_clutch_keys,
                    pressed_keys,
                    arm_clutch_state,
                )
                action = vr_teleop.get_action(obs, robot)
                action = _freeze_vr_clutched_arm_actions(action, obs, vr_teleop)
                action = _apply_fixed_arm_pose_action(
                    action,
                    vr_teleop,
                    basket_pose,
                    effective_fixed_arm_modes,
                    action_ready_targets,
                    fixed_arm_motion_states,
                    obs,
                )
                _clear_reached_action_ready_modes(
                    fixed_arm_modes,
                    fixed_arm_motion_states,
                    vr_teleop,
                    obs,
                    action_ready_targets,
                    episode_boundary,
                )
            base_locked = menu_pause_this_loop or (
                lock_base_while_recording
                and recording is not None
                and recording.active
            )
            if base_locked:
                action = _force_vr_base_stop(action, base_smoother)
            else:
                action = _smooth_vr_base_action(action, base_smoother)
            if action:
                sent_action = robot.send_action(action)
            else:
                sent_action = {}
            now = time.perf_counter()
            if now >= next_camera_t:
                next_camera_t = now + camera_interval_s
                camera_obs, camera_error = _get_robot_observation_best_effort(robot)
                if camera_error is not None and now - last_camera_warn_t >= 30.0:
                    print(f"VR camera read skipped: {camera_error}")
                    last_camera_warn_t = now
                if vr_video_streams:
                    _publish_vr_camera_frames(camera_obs)
                if recording is not None:
                    obs = _augment_recording_observation(recording, camera_obs)
            log_rerun_data(obs, sent_action)
            if not menu_pause_this_loop:
                _record_frame_if_needed(recording, obs, sent_action)
            _update_vr_session_status(recording, episode_boundary, menu_pointer_hand=menu_pointer_hand)

            dt_s = time.perf_counter() - start_loop_t
            precise_sleep(max(0.0, 1 / fps - dt_s))
    finally:
        _set_vr_menu_open(False)
        _set_vr_recording_operation("")
        _update_vr_session_status(None, VrEpisodeBoundaryState(), menu_pointer_hand=menu_pointer_hand)
        _consume_finish_collection_request()
        _finalize_recording(recording)
        _stop_orbbec_rgb_sidecar(orbbec_process)
        try:
            try:
                robot.send_action({"x.vel": 0.0, "theta.vel": 0.0})
            except Exception:
                pass
            robot.disconnect()
        finally:
            _force_stop_vr_runtime(vr_teleop)
            try:
                vr_teleop.disconnect()
            except Exception:
                pass
            _restore_xlevr_console_filter()
            if hotkeys.is_connected:
                hotkeys.disconnect()
    return 0


def _force_stop_vr_runtime(vr_teleop: Any) -> None:
    monitor = getattr(vr_teleop, "vr_monitor", None)
    if monitor is None:
        return

    print("Stopping VR monitor/runtime...")
    try:
        monitor.is_running = False
    except Exception:
        pass

    vr_server = getattr(monitor, "vr_server", None)
    if vr_server is not None:
        _force_stop_vr_websocket_server(vr_server)

    https_server = getattr(monitor, "https_server", None)
    if https_server is not None:
        _force_stop_vr_https_server(https_server)

    vr_thread = getattr(vr_teleop, "vr_thread", None)
    if vr_thread is not None and hasattr(vr_thread, "join"):
        try:
            vr_thread.join(timeout=2.0)
        except Exception:
            pass


def _force_stop_vr_websocket_server(vr_server: Any) -> None:
    async def stop_server() -> None:
        try:
            await asyncio.wait_for(vr_server.stop(), timeout=2.0)
        except Exception as exc:
            print(f"VR websocket stop skipped/failed: {exc}")

    try:
        asyncio.run(stop_server())
    except RuntimeError:
        server = getattr(vr_server, "server", None)
        if server is not None:
            try:
                server.close()
            except Exception:
                pass


def _force_stop_vr_https_server(https_server: Any) -> None:
    httpd = getattr(https_server, "httpd", None)
    if httpd is not None:
        try:
            httpd.shutdown()
        except Exception as exc:
            print(f"VR HTTPS shutdown skipped/failed: {exc}")
        try:
            httpd.server_close()
        except Exception:
            pass

    server_thread = getattr(https_server, "server_thread", None)
    if server_thread is not None and hasattr(server_thread, "join"):
        try:
            server_thread.join(timeout=2.0)
        except Exception:
            pass


def _configure_vr_runtime(vr_teleop: Any, *, input_scale: float) -> None:
    monitor = getattr(vr_teleop, "vr_monitor", None)
    config = getattr(monitor, "config", None)
    if config is not None and hasattr(config, "vr_to_robot_scale"):
        config.vr_to_robot_scale = input_scale
    print(f"VR input scale set to {input_scale:.3f}.")


def _install_xlevr_console_filter() -> bool:
    global _XLEVR_ORIGINAL_PRINT
    if _XLEVR_ORIGINAL_PRINT is not None:
        return True

    original_print = builtins.print

    def filtered_print(*args: Any, **kwargs: Any) -> None:
        if args and isinstance(args[0], str) and args[0].startswith("[VR_WS] Headset - Position:"):
            return
        original_print(*args, **kwargs)

    _XLEVR_ORIGINAL_PRINT = original_print
    builtins.print = filtered_print
    return True


def _restore_xlevr_console_filter() -> None:
    global _XLEVR_ORIGINAL_PRINT
    if _XLEVR_ORIGINAL_PRINT is None:
        return
    builtins.print = _XLEVR_ORIGINAL_PRINT
    _XLEVR_ORIGINAL_PRINT = None


def _install_xlevr_squeeze_metadata_patch(vr_teleop: Any, *, xlevr_path: str | None) -> bool:
    module_prefix = vr_teleop.__class__.__module__.rsplit(".", 1)[0]
    monitor_module = sys.modules.get(f"{module_prefix}.vr_monitor")
    root_value = xlevr_path or getattr(monitor_module, "XLEVR_PATH", None)
    if root_value:
        xlevr_root = Path(root_value).expanduser().resolve()
        if str(xlevr_root) not in sys.path:
            sys.path.insert(0, str(xlevr_root))

    try:
        ws_module = importlib.import_module("xlevr.inputs.vr_ws_server")
    except Exception as exc:
        print(f"Could not patch Quest squeeze metadata: {exc}")
        return False

    server_cls = getattr(ws_module, "VRWebSocketServer", None)
    if server_cls is None:
        return False
    if getattr(server_cls, "_robot42_squeeze_metadata_patched", False):
        return True

    original_process_controller_data = server_cls.process_controller_data
    original_process_single_controller = server_cls.process_single_controller
    original_handle_grip_release = getattr(server_cls, "handle_grip_release", None)
    original_send_goal = server_cls.send_goal

    def set_button_state(
        server: Any,
        hand: str,
        *,
        buttons: dict[str, Any] | None = None,
        grip_active: bool = False,
        trigger: float = 0.0,
        trigger_active: bool = False,
    ) -> None:
        if hand not in {"left", "right"}:
            return
        button_state = dict(buttons or {})
        squeeze_active = bool(button_state.get("squeeze", False) or grip_active)
        if not hasattr(server, "_robot42_controller_button_state"):
            server._robot42_controller_button_state = {}
        server._robot42_controller_button_state[hand] = {
            "buttons": button_state,
            "grip_active": bool(grip_active),
            "squeeze": squeeze_active,
            "trigger": float(trigger),
            "trigger_active": bool(trigger_active),
        }

    async def process_controller_data(self: Any, data: dict[str, Any]) -> Any:
        if isinstance(data, dict):
            hand = data.get("hand")
            if hand in {"left", "right"} and data.get("gripReleased", False):
                set_button_state(self, hand, grip_active=False)
                if original_handle_grip_release is not None:
                    await original_handle_grip_release(self, hand)
                return None
        return await original_process_controller_data(self, data)

    async def process_single_controller(self: Any, hand: str, data: dict[str, Any]) -> Any:
        buttons = dict(data.get("buttons") or {}) if isinstance(data, dict) else {}
        grip_active = bool(data.get("gripActive", False)) if isinstance(data, dict) else False
        trigger = float(data.get("trigger", 0.0) or 0.0) if isinstance(data, dict) else 0.0
        set_button_state(
            self,
            hand,
            buttons=buttons,
            grip_active=grip_active,
            trigger=trigger,
            trigger_active=trigger > 0.5,
        )
        return await original_process_single_controller(self, hand, data)

    async def handle_grip_release(self: Any, hand: str) -> Any:
        set_button_state(self, hand, grip_active=False)
        if original_handle_grip_release is not None:
            return await original_handle_grip_release(self, hand)
        return None

    async def send_goal(self: Any, goal: Any) -> Any:
        arm = getattr(goal, "arm", None)
        if arm in {"left", "right"}:
            state = getattr(self, "_robot42_controller_button_state", {}).get(arm, {})
            metadata = dict(getattr(goal, "metadata", None) or {})
            buttons = dict(metadata.get("buttons") or {})
            buttons.update(state.get("buttons") or {})
            if buttons:
                metadata["buttons"] = buttons
            metadata["grip_active"] = bool(state.get("grip_active", False))
            metadata["squeeze"] = bool(state.get("squeeze", False))
            metadata["trigger"] = float(state.get("trigger", metadata.get("trigger", 0.0)) or 0.0)
            metadata["trigger_active"] = bool(
                state.get("trigger_active", metadata.get("trigger_active", False))
            )
            goal.metadata = metadata
        return await original_send_goal(self, goal)

    server_cls.process_controller_data = process_controller_data
    server_cls.process_single_controller = process_single_controller
    server_cls.handle_grip_release = handle_grip_release
    server_cls.send_goal = send_goal
    server_cls._robot42_squeeze_metadata_patched = True
    return True


def _update_vr_arm_clutch(
    vr_teleop: Any,
    obs: dict[str, Any],
    keys: VrArmClutchKeys,
    pressed_keys: set[str],
    state: dict[str, bool],
) -> None:
    key_by_side = {"left": keys.left, "right": keys.right}
    for side, key in key_by_side.items():
        key_active = bool(key) and key in pressed_keys
        squeeze_active = _vr_arm_squeeze_active(vr_teleop, side)
        active = key_active or squeeze_active
        was_active = state.get(side, False)
        arm = getattr(vr_teleop, f"{side}_arm", None)
        if arm is None:
            state[side] = active
            continue
        if active:
            _sync_vr_arm_to_observation(vr_teleop, side, obs)
            setattr(arm, "ik_clutch_paused", True)
            if not was_active:
                source = "Quest squeeze" if squeeze_active else f"keyboard `{key}`"
                print(f"{side.capitalize()} arm IK clutch engaged from {source}; freezing observed pose.")
        elif was_active and not active:
            _sync_vr_arm_to_observation(vr_teleop, side, obs)
            setattr(arm, "ik_clutch_rebaseline_frames", _VR_ARM_CLUTCH_RELEASE_HOLD_FRAMES)
            _clear_vr_arm_controller_baseline(arm)
            setattr(arm, "ik_clutch_paused", False)
            print(f"{side.capitalize()} arm IK clutch released; freezing observed pose and re-baselining controller.")
        state[side] = active


def _vr_arm_squeeze_active(vr_teleop: Any, side: str) -> bool:
    monitor = getattr(vr_teleop, "vr_monitor", None)
    if monitor is None:
        return False
    server = getattr(monitor, "vr_server", None)
    controller_state = getattr(server, "_robot42_controller_button_state", {}) if server is not None else {}
    if isinstance(controller_state, dict):
        state = controller_state.get(side)
        if isinstance(state, dict):
            buttons = state.get("buttons", {}) or {}
            return bool(state.get("squeeze", False) or state.get("grip_active", False) or buttons.get("squeeze", False))
    if not hasattr(monitor, "get_latest_goal_nowait"):
        return False
    try:
        goal = monitor.get_latest_goal_nowait(side)
    except Exception:
        return False
    metadata = getattr(goal, "metadata", {}) if goal is not None else {}
    if not isinstance(metadata, dict):
        return False
    buttons = metadata.get("buttons", {}) or {}
    return bool(metadata.get("squeeze", False) or metadata.get("grip_active", False) or buttons.get("squeeze", False))


def _freeze_vr_clutched_arm_actions(action: dict[str, Any], obs: dict[str, Any], vr_teleop: Any) -> dict[str, Any]:
    if not action:
        return action
    frozen = dict(action)
    for side in ("left", "right"):
        arm = getattr(vr_teleop, f"{side}_arm", None)
        if arm is None:
            continue
        should_freeze = bool(getattr(arm, "ik_clutch_paused", False)) or int(
            getattr(arm, "ik_clutch_rebaseline_frames", 0) or 0
        ) > 0
        if not should_freeze:
            continue
        for joint in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"):
            key = f"{side}_arm_{joint}.pos"
            if key in obs:
                frozen[key] = float(obs[key])
    return frozen


def _build_vr_basket_pose_targets(raw_targets: list[str]) -> dict[str, float]:
    targets = dict(_VR_BASKET_POSE_DEFAULTS)
    for raw in raw_targets:
        if "=" not in raw:
            raise ValueError(f"Invalid --vr-basket-target `{raw}`. Use JOINT=DEG.")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key.endswith(".pos"):
            key = f"{key}.pos"
        if not (key.startswith("left_arm_") or key.startswith("right_arm_")):
            raise ValueError(f"Invalid --vr-basket-target joint `{key}`. Use left_arm_* or right_arm_* position keys.")
        targets[key] = float(value)
    return targets


def _apply_vr_recording_menu_actions(
    actions: list[str],
    recording: RecordingSession | None,
    state: VrEpisodeBoundaryState,
    fixed_modes: dict[str, str],
    motion_states: dict[str, FixedArmMotionState],
    vr_teleop: Any,
    obs: dict[str, Any],
    action_ready_targets: dict[str, float],
) -> None:
    for action in actions:
        if action == "record":
            if recording is not None and not recording.active:
                _start_recording_session(recording)
                print("Training menu: recording started.")
            elif recording is not None:
                print("Training menu: recording is already active.")
            else:
                print("Training menu: resume selected without dataset recording.")
            if state.phase in {"await_start", "await_finish"}:
                _release_episode_boundary_hold(
                    state,
                    fixed_modes,
                    motion_states,
                    vr_teleop,
                    obs,
                    action_ready_targets,
                )
                print("Training menu: ACTION_READY hold released; IK resumed.")
            _set_vr_menu_open(False)
            _set_vr_recording_operation("")
        elif action == "save":
            if recording is not None and recording.active:
                print("Training menu: saving episode.")
                saved = _save_episode(recording)
            else:
                print("Training menu: save selected, but no episode is active.")
                saved = False
            _set_vr_menu_open(False)
            _set_vr_recording_operation("")
            if saved and state.phase == "await_finish":
                state.phase = "await_start"
                print("ACTION_READY remains locked. Press left thumbstick to start the next episode, or open the menu.")
        elif action == "cancel":
            if recording is not None and recording.active:
                print("Training menu: cancelling active episode.")
                _discard_episode(recording)
            else:
                print("Training menu: cancel selected, but no episode is active.")
            _set_vr_menu_open(False)
            _set_vr_recording_operation("")
            if state.phase == "await_finish":
                state.phase = "await_start"
                print("ACTION_READY remains locked. Press left thumbstick to start the next episode, or open the menu.")
        elif action == "finish":
            print("Training menu: finishing dataset collection.")
            _set_vr_menu_open(False)
            _request_finish_collection()


def _apply_left_thumb_recording_shortcut(
    recording: RecordingSession | None,
    state: VrEpisodeBoundaryState,
    fixed_modes: dict[str, str],
    motion_states: dict[str, FixedArmMotionState],
    vr_teleop: Any,
    obs: dict[str, Any],
    action_ready_targets: dict[str, float],
) -> bool:
    if recording is None:
        return False

    if recording.active:
        _set_vr_recording_operation("saving")
        try:
            print("Left thumb recording shortcut: saving episode.")
            saved = _save_episode(recording)
        finally:
            _set_vr_recording_operation("")
        if saved and state.phase == "await_finish":
            state.phase = "await_start"
            print("ACTION_READY remains locked. Press left thumbstick to start the next episode, or open the menu.")
        return True

    _set_vr_recording_operation("starting")
    try:
        _start_recording_session(recording)
        print("Left thumb recording shortcut: recording started.")
        if state.phase in {"await_start", "await_finish"}:
            _release_episode_boundary_hold(
                state,
                fixed_modes,
                motion_states,
                vr_teleop,
                obs,
                action_ready_targets,
            )
            print("Left thumb recording shortcut: ACTION_READY hold released; IK resumed.")
    finally:
        _set_vr_recording_operation("")
    return True


def _handle_episode_boundary_buttons(
    state: VrEpisodeBoundaryState,
    recording: RecordingSession | None,
    fixed_modes: dict[str, str],
    motion_states: dict[str, FixedArmMotionState],
    vr_teleop: Any,
    obs: dict[str, Any],
    action_ready_targets: dict[str, float],
    newly_pressed_vr_buttons: set[str],
    *,
    toggle_button: str,
    discard_button: str,
) -> bool:
    if state.phase == "idle":
        return False

    toggle_pressed = _vr_button_spec_pressed(toggle_button, newly_pressed_vr_buttons)
    discard_pressed = _vr_button_spec_pressed(discard_button, newly_pressed_vr_buttons)
    if not toggle_pressed and not discard_pressed:
        return False

    if state.phase == "await_finish":
        if discard_pressed:
            if recording is not None and recording.active:
                _discard_recording_from_vr_button(recording, discard_button)
            else:
                print("Episode boundary acknowledged as cancelled; no dataset recording was active.")
        else:
            if recording is not None and recording.active:
                _toggle_recording_from_vr_button(recording, toggle_button)
            else:
                print("Episode boundary acknowledged as complete; no dataset recording was active.")
        state.phase = "await_start"
        print("ACTION_READY remains locked. Press left thumbstick to start the next episode, or open the menu.")
        return True

    if state.phase == "await_start":
        if discard_pressed:
            print("Training discard pressed while no episode is active: finalizing dataset collection.")
            _request_finish_collection()
            return True
        if recording is not None and not recording.active:
            _start_recording_session(recording)
            print(f"Training recording toggle `{toggle_button}` pressed: recording started.")
        _release_episode_boundary_hold(
            state,
            fixed_modes,
            motion_states,
            vr_teleop,
            obs,
            action_ready_targets,
        )
        print("Next episode started from ACTION_READY; IK resumed.")
        return True

    return False


def _update_vr_session_status(
    recording: RecordingSession | None,
    state: VrEpisodeBoundaryState,
    *,
    menu_pointer_hand: str = "left",
) -> None:
    finish_requested = _finish_collection_requested()
    menu_open = _vr_menu_is_open()
    with _VR_SESSION_STATUS_LOCK:
        _VR_SESSION_STATUS.update(
            {
                "recording_enabled": recording is not None,
                "recording_active": bool(recording is not None and recording.active),
                "recording_missing": list(recording.last_missing_features) if recording is not None else [],
                "episode_frame_count": recording.episode_frame_count if recording is not None else 0,
                "session_episode_count": recording.session_episode_count if recording is not None else 0,
                "episode_phase": state.phase,
                "held_sides": sorted(state.held_sides),
                "finish_requested": finish_requested,
                "menu_open": menu_open,
                "menu_pointer_hand": menu_pointer_hand,
            }
        )


def _vr_menu_pointer_hand(config: VrBasketPoseConfig) -> str:
    if not config.enabled:
        return "left"
    if config.skill_arm == "right":
        return "left"
    if config.skill_arm == "left":
        return "right"
    return "left"


def _release_episode_boundary_hold(
    state: VrEpisodeBoundaryState,
    fixed_modes: dict[str, str],
    motion_states: dict[str, FixedArmMotionState],
    vr_teleop: Any,
    obs: dict[str, Any],
    action_ready_targets: dict[str, float],
) -> None:
    for side in tuple(state.held_sides):
        fixed_modes.pop(side, None)
        motion_states.pop(side, None)
        sync_obs = dict(action_ready_targets)
        sync_obs.update(obs)
        _sync_vr_arm_to_observation(vr_teleop, side, sync_obs)
        arm = getattr(vr_teleop, f"{side}_arm", None)
        if arm is not None:
            setattr(arm, "ik_clutch_rebaseline_frames", _VR_ARM_CLUTCH_RELEASE_HOLD_FRAMES)
            setattr(arm, "ik_fixed_pose_paused", False)
    state.held_sides.clear()
    state.phase = "idle"


def _initial_vr_episode_boundary(
    recording: RecordingSession | None,
) -> tuple[dict[str, str], VrEpisodeBoundaryState]:
    if recording is None:
        return {}, VrEpisodeBoundaryState()
    held_sides = {"left", "right"}
    return (
        {side: "episode_hold" for side in held_sides},
        VrEpisodeBoundaryState(phase="await_start", held_sides=held_sides),
    )


def _update_fixed_arm_modes_from_controls(
    fixed_modes: dict[str, str],
    config: VrBasketPoseConfig,
    newly_pressed_keys: set[str],
    newly_pressed_vr_buttons: set[str],
) -> None:
    allowed_sides = _vr_skill_allowed_sides(config)
    requests = (
        ("right", "basket", config.right_basket_key, config.right_basket_button),
        ("right", "action_ready", config.right_action_key, config.right_action_button),
        ("left", "basket", config.left_basket_key, config.left_basket_button),
        ("left", "action_ready", config.left_action_key, config.left_action_button),
    )
    for side, mode, key, button in requests:
        if side not in allowed_sides:
            continue
        if fixed_modes.get(side) in {"action_ready", "episode_hold"}:
            continue
        if key in newly_pressed_keys or _vr_button_spec_pressed(button, newly_pressed_vr_buttons):
            fixed_modes[side] = mode
            if mode == "basket":
                print(f"{side.capitalize()} arm basket pose active: IK disabled for that arm. Press trigger to release.")
            else:
                print(f"{side.capitalize()} arm returning to ACTION_READY.")


def _effective_fixed_arm_modes(fixed_modes: dict[str, str], config: VrBasketPoseConfig) -> dict[str, str]:
    if not config.enabled:
        return dict(fixed_modes)
    effective = dict(fixed_modes)
    for side in {"left", "right"} - _vr_skill_allowed_sides(config):
        effective[side] = "parked_action_ready"
    return effective


def _vr_skill_allowed_sides(config: VrBasketPoseConfig) -> set[str]:
    if config.skill_arm == "both":
        return {"left", "right"}
    return {config.skill_arm}


def _set_fixed_arm_ik_pauses(vr_teleop: Any, fixed_modes: dict[str, str]) -> None:
    for side in ("left", "right"):
        arm = getattr(vr_teleop, f"{side}_arm", None)
        if arm is not None:
            setattr(arm, "ik_fixed_pose_paused", side in fixed_modes)


def _apply_fixed_arm_pose_action(
    action: dict[str, Any],
    vr_teleop: Any,
    config: VrBasketPoseConfig,
    fixed_modes: dict[str, str],
    action_ready_targets: dict[str, float],
    motion_states: dict[str, FixedArmMotionState],
    obs: dict[str, Any],
) -> dict[str, Any]:
    if not fixed_modes:
        return action
    fixed_action = dict(action)
    now = time.perf_counter()
    for side, mode in fixed_modes.items():
        gripper_key = f"{side}_arm_gripper.pos"
        if mode == "basket":
            targets = _side_arm_targets(config.targets, side)
            targets.pop(gripper_key, None)
            fixed_action.update(
                _fixed_arm_motion_targets(
                    side,
                    mode,
                    targets,
                    motion_states,
                    obs,
                    fixed_action,
                    now=now,
                    duration_s=config.basket_motion_s,
                    basket_elbow_lift_deg=config.basket_elbow_lift_deg,
                    basket_shoulder_back_deg=config.basket_shoulder_back_deg,
                    basket_elbow_compensation_deg=config.basket_elbow_compensation_deg,
                )
            )
        elif mode == "action_ready":
            targets = _side_arm_targets(action_ready_targets, side)
            targets.pop(gripper_key, None)
            fixed_action.update(
                _fixed_arm_motion_targets(
                    side,
                    mode,
                    targets,
                    motion_states,
                    obs,
                    fixed_action,
                    now=now,
                    duration_s=config.action_ready_motion_s,
                )
            )
        elif mode == "parked_action_ready":
            motion_states.pop(side, None)
            targets = _side_arm_targets(action_ready_targets, side)
            targets.pop(gripper_key, None)
            fixed_action.update(targets)
        elif mode == "episode_hold":
            motion_states.pop(side, None)
            targets = _side_arm_targets(action_ready_targets, side)
            targets.pop(gripper_key, None)
            fixed_action.update(targets)

        if gripper_key in action:
            fixed_action[gripper_key] = float(action[gripper_key])
        elif _vr_trigger_active(vr_teleop, side):
            fixed_action[gripper_key] = config.release_gripper
        elif gripper_key in obs:
            fixed_action[gripper_key] = float(obs[gripper_key])
    return fixed_action


def _fixed_arm_motion_targets(
    side: str,
    mode: str,
    targets: dict[str, float],
    motion_states: dict[str, FixedArmMotionState],
    obs: dict[str, Any],
    action: dict[str, Any],
    *,
    now: float,
    duration_s: float,
    basket_elbow_lift_deg: float = 0.0,
    basket_shoulder_back_deg: float = 0.0,
    basket_elbow_compensation_deg: float | None = None,
) -> dict[str, float]:
    state = motion_states.get(side)
    if state is None or state.mode != mode or not _same_joint_targets(state.goal_targets, targets):
        state = _start_fixed_arm_motion(
            side,
            mode,
            targets,
            obs,
            action,
            now=now,
            duration_s=duration_s,
            basket_elbow_lift_deg=basket_elbow_lift_deg,
            basket_shoulder_back_deg=basket_shoulder_back_deg,
            basket_elbow_compensation_deg=basket_elbow_compensation_deg,
        )
        motion_states[side] = state
        if state.waypoint_targets:
            elbow_key = f"{side}_arm_elbow_flex.pos"
            shoulder_key = f"{side}_arm_shoulder_lift.pos"
            first_elbow = state.waypoint_targets[0].get(elbow_key)
            last_elbow = state.waypoint_targets[-1].get(elbow_key)
            last_shoulder = state.waypoint_targets[-1].get(shoulder_key)
            final_elbow = state.goal_targets.get(elbow_key)
            print(
                f"{side.capitalize()} arm {mode.replace('_', ' ')} motion over {state.duration_s:.1f}s: "
                f"elbow {float(state.start_targets.get(elbow_key, 0.0)):.1f}"
                f" -> first {float(first_elbow):.1f}; "
                f"last waypoint elbow {float(last_elbow):.1f}, shoulder {float(last_shoulder):.1f}; "
                f"{len(state.waypoint_targets)} waypoints; final elbow {float(final_elbow):.1f}."
            )
        else:
            print(f"{side.capitalize()} arm {mode.replace('_', ' ')} motion over {state.duration_s:.1f}s.")
    return _interpolate_fixed_arm_motion(state, now, obs=obs)


def _start_fixed_arm_motion(
    side: str,
    mode: str,
    targets: dict[str, float],
    obs: dict[str, Any],
    action: dict[str, Any],
    *,
    now: float,
    duration_s: float,
    basket_elbow_lift_deg: float = 0.0,
    basket_shoulder_back_deg: float = 0.0,
    basket_elbow_compensation_deg: float | None = None,
) -> FixedArmMotionState:
    start_targets: dict[str, float] = {}
    for key, target in targets.items():
        value = obs.get(key, action.get(key, target))
        try:
            start_targets[key] = float(value)
        except (TypeError, ValueError):
            start_targets[key] = float(target)
    waypoint_targets: tuple[dict[str, float], ...] = ()
    segment_weights: tuple[float, ...] = ()
    if mode == "basket":
        waypoint_targets = _build_basket_joint_waypoints(
            side,
            start_targets,
            targets,
            elbow_lift_deg=basket_elbow_lift_deg,
            shoulder_back_deg=basket_shoulder_back_deg,
            elbow_compensation_deg=basket_elbow_compensation_deg,
        )
        if waypoint_targets:
            segment_weights = _basket_motion_segment_weights(len(waypoint_targets))
    elif mode == "action_ready":
        waypoint_targets = _build_action_ready_joint_waypoints(side, start_targets)
        if waypoint_targets:
            segment_weights = _action_ready_motion_segment_weights(len(waypoint_targets))
    return FixedArmMotionState(
        mode=mode,
        start_targets=start_targets,
        goal_targets=dict(targets),
        started_at=now,
        duration_s=max(0.0, float(duration_s)),
        waypoint_targets=waypoint_targets,
        segment_weights=segment_weights,
    )


def _interpolate_fixed_arm_motion(
    state: FixedArmMotionState,
    now: float,
    *,
    obs: dict[str, Any] | None = None,
) -> dict[str, float]:
    if state.duration_s <= 0:
        return dict(state.goal_targets)
    progress = _clip((now - state.started_at) / state.duration_s, 0.0, 1.0)
    points = (state.start_targets, *state.waypoint_targets, state.goal_targets)
    weights = state.segment_weights
    if len(weights) != len(points) - 1 or sum(weights) <= 0:
        weights = tuple(1.0 for _ in range(len(points) - 1))
    total_weight = sum(weights)
    weighted_progress = progress * total_weight
    completed_weight = 0.0
    segment_index = len(weights) - 1
    segment_progress = 1.0
    for index, weight in enumerate(weights):
        next_weight = completed_weight + weight
        if weighted_progress <= next_weight or index == len(weights) - 1:
            segment_index = index
            segment_progress = (
                1.0 if weight <= 0 else _clip((weighted_progress - completed_weight) / weight, 0.0, 1.0)
            )
            break
        completed_weight = next_weight
    eased = segment_progress * segment_progress * (3.0 - 2.0 * segment_progress)
    segment_start = points[segment_index]
    segment_goal = points[segment_index + 1]
    return {
        key: float(segment_start[key]) + (float(segment_goal[key]) - float(segment_start[key])) * eased
        for key in state.start_targets
    }


def _build_basket_joint_waypoints(
    side: str,
    start_targets: dict[str, float],
    goal_targets: dict[str, float],
    *,
    elbow_lift_deg: float,
    shoulder_back_deg: float,
    elbow_compensation_deg: float | None,
) -> tuple[dict[str, float], ...]:
    if side == "right":
        return _build_right_captured_basket_waypoints(start_targets, goal_targets)

    elbow_key = f"{side}_arm_elbow_flex.pos"
    shoulder_key = f"{side}_arm_shoulder_lift.pos"
    if elbow_key not in start_targets or elbow_key not in goal_targets:
        return ()

    lift_offset = float(elbow_lift_deg)
    lift_targets = dict(start_targets)
    lift_targets[elbow_key] = _clip(
        float(start_targets[elbow_key]) + lift_offset,
        -115.0,
        106.0,
    )

    compensation_deg = (
        abs(float(shoulder_back_deg)) * 2.0
        if elbow_compensation_deg is None
        else abs(float(elbow_compensation_deg))
    )
    clearance_targets = dict(lift_targets)
    if shoulder_key in clearance_targets:
        clearance_targets[shoulder_key] = _clip(
            _step_toward(
                float(lift_targets[shoulder_key]),
                float(_NAV_STOW_ARM_POSE["shoulder_lift"]),
                abs(float(shoulder_back_deg)),
            ),
            -108.0,
            96.0,
        )
    if lift_offset:
        clearance_targets[elbow_key] = _clip(
            float(lift_targets[elbow_key]) - math.copysign(compensation_deg, lift_offset),
            -115.0,
            106.0,
        )

    travel_targets = dict(clearance_targets)
    for suffix in ("shoulder_pan.pos", "wrist_roll.pos"):
        joint_key = f"{side}_arm_{suffix}"
        if joint_key in travel_targets and joint_key in goal_targets:
            travel_targets[joint_key] = float(goal_targets[joint_key])

    raised_basket_targets = dict(goal_targets)
    raised_basket_targets[elbow_key] = _clip(
        float(goal_targets[elbow_key]) + lift_offset,
        -115.0,
        106.0,
    )
    return (lift_targets, clearance_targets, travel_targets, raised_basket_targets)


def _basket_motion_segment_weights(waypoint_count: int) -> tuple[float, ...]:
    if waypoint_count == 4:
        return (0.20, 0.30, 0.25, 0.15, 0.10)
    if waypoint_count == 3:
        return (0.25, 0.35, 0.25, 0.15)
    if waypoint_count == 2:
        return (0.35, 0.40, 0.25)
    return tuple(1.0 for _ in range(max(1, waypoint_count + 1)))


def _action_ready_motion_segment_weights(waypoint_count: int) -> tuple[float, ...]:
    if waypoint_count == 2:
        return (0.35, 0.35, 0.30)
    return tuple(1.0 for _ in range(max(1, waypoint_count + 1)))


def _build_right_captured_basket_waypoints(
    start_targets: dict[str, float],
    goal_targets: dict[str, float],
) -> tuple[dict[str, float], ...]:
    clearance_targets = dict(start_targets)
    for key, reference_value in _RIGHT_BASKET_PATH_REFERENCE_GRASP.items():
        if key not in clearance_targets or key not in _RIGHT_BASKET_PATH_CLEARANCE:
            continue
        delta = float(_RIGHT_BASKET_PATH_CLEARANCE[key]) - float(reference_value)
        clearance_targets[key] = _clip_arm_joint_target(key, float(start_targets[key]) + delta)

    over_basket_targets = dict(goal_targets)
    for key, value in _RIGHT_BASKET_PATH_OVER_BASKET.items():
        if key in over_basket_targets:
            over_basket_targets[key] = _clip_arm_joint_target(key, float(value))

    return (clearance_targets, over_basket_targets)


def _build_action_ready_joint_waypoints(
    side: str,
    start_targets: dict[str, float],
) -> tuple[dict[str, float], ...]:
    if side != "right":
        return ()

    over_basket_targets = dict(start_targets)
    for key, value in _RIGHT_BASKET_PATH_OVER_BASKET.items():
        if key in over_basket_targets:
            over_basket_targets[key] = _clip_arm_joint_target(key, float(value))

    clearance_targets = dict(start_targets)
    for key, value in _RIGHT_BASKET_PATH_CLEARANCE.items():
        if key in clearance_targets:
            clearance_targets[key] = _clip_arm_joint_target(key, float(value))

    return (over_basket_targets, clearance_targets)


def _clip_arm_joint_target(key: str, value: float) -> float:
    if key.endswith("_shoulder_lift.pos"):
        return _clip(value, -108.0, 96.0)
    if key.endswith("_elbow_flex.pos"):
        return _clip(value, -115.0, 106.0)
    return value


def _step_toward(current: float, target: float, amount: float) -> float:
    if amount <= 0:
        return current
    if current < target:
        return min(target, current + amount)
    return max(target, current - amount)


def _same_joint_targets(left: dict[str, float], right: dict[str, float], *, tolerance: float = 1e-4) -> bool:
    if left.keys() != right.keys():
        return False
    return all(abs(float(left[key]) - float(right[key])) <= tolerance for key in left)


def _clear_reached_action_ready_modes(
    fixed_modes: dict[str, str],
    motion_states: dict[str, FixedArmMotionState],
    vr_teleop: Any,
    obs: dict[str, Any],
    action_ready_targets: dict[str, float],
    episode_boundary: VrEpisodeBoundaryState,
    *,
    tolerance_deg: float = 2.0,
) -> None:
    now = time.perf_counter()
    for side, mode in list(fixed_modes.items()):
        if mode != "action_ready":
            continue
        targets = _side_arm_targets(action_ready_targets, side)
        motion_done = _fixed_arm_motion_done(motion_states.get(side), now)
        reached = _arm_targets_reached(obs, targets, tolerance_deg=tolerance_deg)
        if not (motion_done or reached):
            continue
        fixed_modes[side] = "episode_hold"
        motion_states.pop(side, None)
        episode_boundary.phase = "await_finish"
        episode_boundary.held_sides.add(side)
        print(
            f"{side.capitalize()} arm ACTION_READY motion complete; arm locked at ACTION_READY. "
            "Press left thumbstick to save, or open the menu to Save/Cancel. "
            "Press left thumbstick again to start the next episode."
        )


def _fixed_arm_motion_done(state: FixedArmMotionState | None, now: float, *, settle_s: float = 0.15) -> bool:
    if state is None or state.mode != "action_ready":
        return False
    return now >= state.started_at + state.duration_s + settle_s


def _arm_targets_reached(obs: dict[str, Any], targets: dict[str, float], *, tolerance_deg: float) -> bool:
    if not targets:
        return False
    for key, target in targets.items():
        if key.endswith("_gripper.pos"):
            continue
        if key not in obs:
            continue
        if abs(float(obs[key]) - float(target)) > tolerance_deg:
            return False
    return True


def _side_arm_targets(targets: dict[str, float], side: str) -> dict[str, float]:
    prefix = f"{side}_arm_"
    return {key: value for key, value in targets.items() if key.startswith(prefix)}


def _capture_arm_pose_targets(robot: Any) -> dict[str, float]:
    obs = _get_robot_observation(robot, use_camera=False)
    return {
        key: float(value)
        for key, value in obs.items()
        if key.endswith(".pos") and (key.startswith("left_arm_") or key.startswith("right_arm_"))
    }


def _current_vr_button_state(vr_teleop: Any) -> set[str]:
    pressed: set[str] = set()
    monitor = getattr(vr_teleop, "vr_monitor", None)
    server = getattr(monitor, "vr_server", None) if monitor is not None else None
    controller_state = getattr(server, "_robot42_controller_button_state", {}) if server is not None else {}
    for side in ("left", "right"):
        state = controller_state.get(side) if isinstance(controller_state, dict) else None
        if isinstance(state, dict):
            _add_vr_buttons_from_metadata(pressed, side, state)
        metadata = _vr_controller_metadata(vr_teleop, side)
        if metadata:
            _add_vr_buttons_from_metadata(pressed, side, metadata)
    return pressed


def _add_vr_buttons_from_metadata(pressed: set[str], side: str, metadata: dict[str, Any]) -> None:
    buttons = metadata.get("buttons", {}) or {}
    if isinstance(buttons, dict):
        for name, is_pressed in buttons.items():
            if is_pressed:
                button_name = str(name).strip().lower()
                pressed.add(f"{side}:{button_name}")
                if side == "left":
                    if button_name == "a":
                        pressed.add("left:x")
                    elif button_name == "b":
                        pressed.add("left:y")
    if metadata.get("squeeze", False):
        pressed.add(f"{side}:squeeze")
    if metadata.get("grip_active", False):
        pressed.add(f"{side}:grip")


def _vr_button_spec_pressed(spec: str, newly_pressed: set[str]) -> bool:
    normalized = _normalize_vr_button_spec(spec)
    return bool(normalized and normalized in newly_pressed)


def _normalize_vr_button_spec(spec: str) -> str:
    value = (spec or "").strip().lower()
    if not value:
        return ""
    if ":" not in value:
        return f"right:{value}"
    side, button = value.split(":", 1)
    return f"{side.strip()}:{button.strip()}"


def _vr_trigger_active(vr_teleop: Any, side: str) -> bool:
    monitor = getattr(vr_teleop, "vr_monitor", None)
    server = getattr(monitor, "vr_server", None) if monitor is not None else None
    controller_state = getattr(server, "_robot42_controller_button_state", {}) if server is not None else {}
    if isinstance(controller_state, dict):
        state = controller_state.get(side)
        if isinstance(state, dict):
            return bool(
                float(state.get("trigger", 0.0) or 0.0) > 0.5
                or state.get("trigger_active", False)
            )
    metadata = _vr_controller_metadata(vr_teleop, side)
    if not metadata:
        return False
    return bool(float(metadata.get("trigger", 0.0) or 0.0) > 0.5 or metadata.get("trigger_active", False))


def _vr_controller_metadata(vr_teleop: Any, side: str) -> dict[str, Any]:
    monitor = getattr(vr_teleop, "vr_monitor", None)
    if monitor is None or not hasattr(monitor, "get_latest_goal_nowait"):
        return {}
    try:
        goal = monitor.get_latest_goal_nowait(side)
    except Exception:
        return {}
    metadata = getattr(goal, "metadata", {}) if goal is not None else {}
    return metadata if isinstance(metadata, dict) else {}


def _install_vr_arm_tuning(vr_teleop: Any, tuning: VrArmTuning) -> None:
    import types

    for arm_name in ("left_arm", "right_arm"):
        arm = getattr(vr_teleop, arm_name, None)
        if arm is None:
            continue
        arm._robot42_yawed_pan_sign = tuning.yawed_pan_sign
        arm.handle_vr_input = types.MethodType(_tuned_vr_arm_input(tuning), arm)
    print(
        "VR arm tuning: "
        f"ik_mode={tuning.ik_mode}, "
        f"vertical_sign={tuning.vertical_sign:g}, "
        f"y_gain={tuning.y_gain:.2f}, "
        f"z_gain={tuning.z_gain:.2f}, "
        f"ik_alpha={tuning.ik_alpha:.2f}, "
        f"yawed_forward_gain={tuning.yawed_forward_gain:.2f}, "
        f"yawed_lateral_gain={tuning.yawed_lateral_gain:.2f}, "
        f"yawed_pan_sign={tuning.yawed_pan_sign:g}, "
        f"yawed_pan_step_limit={tuning.yawed_pan_step_limit:.1f}, "
        f"shoulder_lift=[{tuning.shoulder_lift_min:.1f}, {tuning.shoulder_lift_max:.1f}], "
        f"elbow_flex=[{tuning.elbow_flex_min:.1f}, {tuning.elbow_flex_max:.1f}], "
        f"joint_limits={'on' if tuning.enforce_joint_limits else 'off'}."
    )


def _tuned_vr_arm_input(tuning: VrArmTuning) -> Any:
    def handle_vr_input(self: Any, vr_goal: Any, gripper_state: Any) -> None:
        if vr_goal is None or not hasattr(vr_goal, "target_position") or vr_goal.target_position is None:
            return

        current_vr_pos = vr_goal.target_position
        metadata = getattr(vr_goal, "metadata", {}) or {}
        rebaseline_frames = int(getattr(self, "ik_clutch_rebaseline_frames", 0) or 0)
        if rebaseline_frames > 0:
            setattr(self, "ik_clutch_rebaseline_frames", rebaseline_frames - 1)
            setattr(self, "ik_clutch_rebaseline_once", False)
            self.prev_vr_pos = current_vr_pos
            if hasattr(vr_goal, "wrist_flex_deg") and vr_goal.wrist_flex_deg is not None:
                self.prev_wrist_flex = vr_goal.wrist_flex_deg
            if hasattr(vr_goal, "wrist_roll_deg") and vr_goal.wrist_roll_deg is not None:
                self.prev_wrist_roll = vr_goal.wrist_roll_deg
            self.target_positions["gripper"] = 45 if metadata.get("trigger", 0) > 0.5 else 0.0
            _maybe_print_vr_arm_debug(self, tuning, phase="rebaseline")
            return

        if getattr(self, "ik_clutch_rebaseline_once", False):
            setattr(self, "ik_clutch_rebaseline_once", False)
            self.prev_vr_pos = current_vr_pos
            if hasattr(vr_goal, "wrist_flex_deg") and vr_goal.wrist_flex_deg is not None:
                self.prev_wrist_flex = vr_goal.wrist_flex_deg
            if hasattr(vr_goal, "wrist_roll_deg") and vr_goal.wrist_roll_deg is not None:
                self.prev_wrist_roll = vr_goal.wrist_roll_deg
            self.target_positions["gripper"] = 45 if metadata.get("trigger", 0) > 0.5 else 0.0
            _maybe_print_vr_arm_debug(self, tuning, phase="rebaseline")
            return

        if getattr(self, "ik_clutch_paused", False) or getattr(self, "ik_fixed_pose_paused", False):
            self.prev_vr_pos = current_vr_pos
            if hasattr(vr_goal, "wrist_flex_deg") and vr_goal.wrist_flex_deg is not None:
                self.prev_wrist_flex = vr_goal.wrist_flex_deg
            if hasattr(vr_goal, "wrist_roll_deg") and vr_goal.wrist_roll_deg is not None:
                self.prev_wrist_roll = vr_goal.wrist_roll_deg
            self.target_positions["gripper"] = 45 if metadata.get("trigger", 0) > 0.5 else 0.0
            phase = "fixed_pose" if getattr(self, "ik_fixed_pose_paused", False) else "paused"
            _maybe_print_vr_arm_debug(self, tuning, phase=phase)
            return

        if not hasattr(self, "prev_vr_pos"):
            self.prev_vr_pos = current_vr_pos
            return

        vr_x = (current_vr_pos[0] - self.prev_vr_pos[0]) * 170
        vr_y = (current_vr_pos[1] - self.prev_vr_pos[1]) * 80 * tuning.y_gain
        vr_z = (current_vr_pos[2] - self.prev_vr_pos[2]) * 80 * tuning.z_gain
        self.prev_vr_pos = current_vr_pos

        pos_scale = 0.015
        angle_scale = 3.0
        delta_limit = 0.02
        angle_limit = 6.0

        delta_x = _deadzone_clip(vr_x * pos_scale, deadzone=0.001, limit=delta_limit)
        delta_y = _deadzone_clip(vr_y * pos_scale, deadzone=0.001, limit=delta_limit)
        delta_z = _deadzone_clip(vr_z * pos_scale, deadzone=0.001, limit=delta_limit)

        if tuning.ik_mode == "yawed":
            _update_yawed_vr_arm_target(self, tuning, delta_x=delta_x, delta_y=delta_y, delta_z=delta_z)
        else:
            self.current_x += -delta_z
            self.current_y += tuning.vertical_sign * delta_y

        if hasattr(vr_goal, "wrist_flex_deg") and vr_goal.wrist_flex_deg is not None:
            if not hasattr(self, "prev_wrist_flex"):
                self.prev_wrist_flex = vr_goal.wrist_flex_deg
            else:
                delta_pitch = (vr_goal.wrist_flex_deg - self.prev_wrist_flex) * angle_scale
                delta_pitch = _deadzone_clip(delta_pitch, deadzone=1.0, limit=angle_limit)
                self.pitch += delta_pitch
                self.pitch = max(-90, min(90, self.pitch))
                self.prev_wrist_flex = vr_goal.wrist_flex_deg

        if hasattr(vr_goal, "wrist_roll_deg") and vr_goal.wrist_roll_deg is not None:
            if not hasattr(self, "prev_wrist_roll"):
                self.prev_wrist_roll = vr_goal.wrist_roll_deg
            else:
                delta_roll = (vr_goal.wrist_roll_deg - self.prev_wrist_roll) * angle_scale
                delta_roll = _deadzone_clip(delta_roll, deadzone=1.0, limit=angle_limit)
                current_roll = self.target_positions.get("wrist_roll", 0.0)
                self.target_positions["wrist_roll"] = max(-90, min(90, current_roll + delta_roll))
                self.prev_wrist_roll = vr_goal.wrist_roll_deg

        if tuning.ik_mode != "yawed" and abs(delta_x) > 0.001:
            delta_pan = max(-angle_limit, min(angle_limit, delta_x * 200.0))
            current_pan = self.target_positions.get("shoulder_pan", 0.0)
            self.target_positions["shoulder_pan"] = max(-180, min(180, current_pan + delta_pan))

        joint2_target = None
        joint3_target = None
        ik_error = None
        try:
            joint2_target, joint3_target = _so101_inverse_kinematics_with_limits(
                self.kinematics,
                self.current_x,
                self.current_y,
                tuning,
            )
            alpha = max(0.01, min(1.0, tuning.ik_alpha))
            self.target_positions["shoulder_lift"] = (
                (1 - alpha) * self.target_positions.get("shoulder_lift", 0.0) + alpha * joint2_target
            )
            self.target_positions["elbow_flex"] = (
                (1 - alpha) * self.target_positions.get("elbow_flex", 0.0) + alpha * joint3_target
            )
        except Exception as exc:
            ik_error = str(exc)
            print(f"[{self.prefix}] VR IK failed: {exc}")

        self.target_positions["wrist_flex"] = (
            -self.target_positions["shoulder_lift"] - self.target_positions["elbow_flex"] + self.pitch
        )
        self.target_positions["gripper"] = 45 if metadata.get("trigger", 0) > 0.5 else 0.0
        _maybe_print_vr_arm_debug(
            self,
            tuning,
            phase="track",
            delta_x=delta_x,
            delta_y=delta_y,
            delta_z=delta_z,
            ik_shoulder=joint2_target,
            ik_elbow=joint3_target,
            ik_error=ik_error,
        )

    return handle_vr_input


def _update_yawed_vr_arm_target(
    arm: Any,
    tuning: VrArmTuning,
    *,
    delta_x: float,
    delta_y: float,
    delta_z: float,
) -> None:
    _ensure_yawed_vr_arm_state(arm)

    candidate_forward = arm.current_forward + (-delta_z) * tuning.yawed_forward_gain
    candidate_lateral = arm.current_lateral + delta_x * tuning.yawed_lateral_gain
    arm.current_height += tuning.vertical_sign * delta_y

    radial = math.hypot(candidate_forward, candidate_lateral)
    pan_sign = float(tuning.yawed_pan_sign)
    if abs(pan_sign) < 1e-6:
        pan_sign = 1.0
    desired_pan = math.degrees(math.atan2(candidate_lateral, candidate_forward)) * pan_sign
    pan_limit = abs(float(tuning.yawed_pan_limit))
    desired_pan = _clip(desired_pan, -pan_limit, pan_limit)
    current_pan = float(arm.target_positions.get("shoulder_pan", desired_pan))
    pan_step_limit = abs(float(tuning.yawed_pan_step_limit))
    pan_target = (
        _step_toward(current_pan, desired_pan, pan_step_limit)
        if pan_step_limit > 0.0
        else desired_pan
    )
    arm.target_positions["shoulder_pan"] = pan_target

    pan_rad = math.radians(pan_target / pan_sign)
    arm.current_forward = radial * math.cos(pan_rad)
    arm.current_lateral = radial * math.sin(pan_rad)

    arm.current_x = radial
    arm.current_y = arm.current_height


def _ensure_yawed_vr_arm_state(arm: Any) -> None:
    if all(hasattr(arm, attr) for attr in ("current_forward", "current_lateral", "current_height")):
        return
    radial = abs(float(getattr(arm, "current_x", 0.0) or 0.0))
    pan_rad = math.radians(float(getattr(arm, "target_positions", {}).get("shoulder_pan", 0.0) or 0.0))
    arm.current_forward = radial * math.cos(pan_rad)
    arm.current_lateral = radial * math.sin(pan_rad)
    arm.current_height = float(getattr(arm, "current_y", 0.0) or 0.0)


def _maybe_print_vr_arm_debug(
    arm: Any,
    tuning: VrArmTuning,
    *,
    phase: str,
    delta_x: float = 0.0,
    delta_y: float = 0.0,
    delta_z: float = 0.0,
    ik_shoulder: float | None = None,
    ik_elbow: float | None = None,
    ik_error: str | None = None,
) -> None:
    if not tuning.debug:
        return
    now = time.perf_counter()
    interval_s = 1.0 / max(0.1, tuning.debug_hz)
    last_t = float(getattr(arm, "_robot42_last_ik_debug_t", 0.0) or 0.0)
    if now - last_t < interval_s:
        return
    setattr(arm, "_robot42_last_ik_debug_t", now)

    x = float(getattr(arm, "current_x", 0.0) or 0.0)
    y = float(getattr(arm, "current_y", 0.0) or 0.0)
    r = math.hypot(x, y)
    kin = getattr(arm, "kinematics", None)
    l1 = float(getattr(kin, "l1", 0.1159))
    l2 = float(getattr(kin, "l2", 0.1350))
    r_min = abs(l1 - l2)
    r_max = l1 + l2

    notes: list[str] = []
    if r <= r_min + 0.003:
        notes.append("near_min_radius")
    if r >= r_max - 0.003:
        notes.append("near_max_radius")
    limit_prefix = "" if tuning.enforce_joint_limits else "would_"
    if ik_shoulder is not None and ik_shoulder <= tuning.shoulder_lift_min + 0.5:
        notes.append(f"{limit_prefix}shoulder_min")
    if ik_shoulder is not None and ik_shoulder >= tuning.shoulder_lift_max - 0.5:
        notes.append(f"{limit_prefix}shoulder_max")
    if ik_elbow is not None and ik_elbow <= tuning.elbow_flex_min + 0.5:
        notes.append(f"{limit_prefix}elbow_min")
    if ik_elbow is not None and ik_elbow >= tuning.elbow_flex_max - 0.5:
        notes.append(f"{limit_prefix}elbow_max")
    if ik_error:
        notes.append(f"ik_error={ik_error}")

    targets = getattr(arm, "target_positions", {}) or {}
    prefix = getattr(arm, "prefix", "arm")
    ik_text = (
        f"ik=({ik_shoulder:.1f},{ik_elbow:.1f})"
        if ik_shoulder is not None and ik_elbow is not None
        else "ik=(none)"
    )
    yawed_text = ""
    if tuning.ik_mode == "yawed" and all(
        hasattr(arm, attr) for attr in ("current_forward", "current_lateral", "current_height")
    ):
        yawed_text = (
            f" target3d=(fwd={float(getattr(arm, 'current_forward')):.3f},"
            f"lat={float(getattr(arm, 'current_lateral')):.3f},"
            f"h={float(getattr(arm, 'current_height')):.3f},"
            f"pan={float(targets.get('shoulder_pan', 0.0)):.1f})"
        )
    print(
        f"[VR_IK {prefix}] mode={tuning.ik_mode} phase={phase} "
        f"xy=({x:.3f},{y:.3f}) r={r:.3f} "
        f"d=({delta_x:+.3f},{delta_y:+.3f},{delta_z:+.3f}) "
        f"{ik_text} "
        f"target=(shoulder={float(targets.get('shoulder_lift', 0.0)):.1f},"
        f"elbow={float(targets.get('elbow_flex', 0.0)):.1f},"
        f"wrist={float(targets.get('wrist_flex', 0.0)):.1f}) "
        f"pitch={float(getattr(arm, 'pitch', 0.0) or 0.0):.1f} "
        f"{yawed_text} "
        f"notes={','.join(notes) if notes else 'ok'}"
    )


def _so101_inverse_kinematics_with_limits(
    kinematics: Any,
    x: float,
    y: float,
    tuning: VrArmTuning,
) -> tuple[float, float]:
    l1 = float(getattr(kinematics, "l1", 0.1159))
    l2 = float(getattr(kinematics, "l2", 0.1350))

    theta1_offset = math.atan2(0.028, 0.11257)
    theta2_offset = math.atan2(0.0052, 0.1349) + theta1_offset

    r = math.sqrt(x**2 + y**2)
    r_max = l1 + l2
    if r > r_max:
        scale_factor = r_max / r
        x *= scale_factor
        y *= scale_factor
        r = r_max

    r_min = abs(l1 - l2)
    if 0 < r < r_min:
        scale_factor = r_min / r
        x *= scale_factor
        y *= scale_factor
        r = r_min

    cos_theta2 = -(r**2 - l1**2 - l2**2) / (2 * l1 * l2)
    cos_theta2 = max(-1.0, min(1.0, cos_theta2))

    theta2 = math.pi - math.acos(cos_theta2)
    beta = math.atan2(y, x)
    gamma = math.atan2(l2 * math.sin(theta2), l1 + l2 * math.cos(theta2))
    theta1 = beta + gamma

    shoulder_lift = 90.0 - math.degrees(theta1 + theta1_offset)
    elbow_flex = math.degrees(theta2 + theta2_offset) - 90.0

    if tuning.enforce_joint_limits:
        shoulder_lift = _clip(shoulder_lift, tuning.shoulder_lift_min, tuning.shoulder_lift_max)
        elbow_flex = _clip(elbow_flex, tuning.elbow_flex_min, tuning.elbow_flex_max)
    return shoulder_lift, elbow_flex


def _clip(value: float, lower: float, upper: float) -> float:
    if lower > upper:
        lower, upper = upper, lower
    return max(lower, min(upper, value))


def _deadzone_clip(value: float, *, deadzone: float, limit: float) -> float:
    if -deadzone < value < deadzone:
        return 0.0
    return max(-limit, min(limit, value))


def _smooth_vr_base_action(action: dict[str, Any], smoother: BaseSmoother) -> dict[str, Any]:
    if not action:
        return action
    if "x.vel" not in action and "theta.vel" not in action:
        return action

    now = time.perf_counter()
    dt = 1.0 / 30.0 if smoother.last_t is None else max(0.001, min(0.1, now - smoother.last_t))
    smoother.last_t = now

    target_x = _shape_axis(
        float(action.get("x.vel", 0.0) or 0.0),
        source_max=0.5,
        target_max=smoother.max_linear,
        deadzone=smoother.deadzone,
        curve=smoother.curve,
    )
    target_theta = _shape_axis(
        float(action.get("theta.vel", 0.0) or 0.0),
        source_max=120.0,
        target_max=smoother.max_angular,
        deadzone=smoother.deadzone,
        curve=smoother.curve,
    )

    smoother.x_vel = _slew(smoother.x_vel, target_x, smoother.linear_accel * dt)
    smoother.theta_vel = _slew(smoother.theta_vel, target_theta, smoother.angular_accel * dt)

    smoothed = dict(action)
    smoothed["x.vel"] = smoother.x_vel
    smoothed["theta.vel"] = smoother.theta_vel
    return smoothed


def _force_vr_base_stop(action: dict[str, Any], smoother: BaseSmoother) -> dict[str, Any]:
    stopped = dict(action)
    stopped["x.vel"] = 0.0
    stopped["theta.vel"] = 0.0
    smoother.x_vel = 0.0
    smoother.theta_vel = 0.0
    smoother.last_t = None
    return stopped


def _shape_axis(
    value: float,
    *,
    source_max: float,
    target_max: float,
    deadzone: float,
    curve: float,
) -> float:
    if source_max <= 0 or target_max <= 0:
        return 0.0
    normalized = max(-1.0, min(1.0, value / source_max))
    magnitude = abs(normalized)
    if magnitude <= deadzone:
        return 0.0
    scaled = (magnitude - deadzone) / max(1e-6, 1.0 - deadzone)
    shaped = scaled ** max(1.0, curve)
    return (1.0 if normalized >= 0 else -1.0) * shaped * target_max


def _slew(current: float, target: float, max_step: float) -> float:
    if max_step <= 0:
        return target
    delta = target - current
    if delta > max_step:
        return current + max_step
    if delta < -max_step:
        return current - max_step
    return target


def _connect_robot(robot: Any, *, auto_restore_calibration: bool) -> None:
    if not auto_restore_calibration:
        robot.connect()
        return

    original_input = builtins.input

    def auto_input(prompt: str = "") -> str:
        if "restore calibration from file" in prompt:
            print(prompt)
            return ""
        return original_input(prompt)

    try:
        builtins.input = auto_input
        robot.connect()
    finally:
        builtins.input = original_input


def _create_dataset(
    robot: Any,
    *,
    dataset_id: str,
    dataset_root: str,
    fps: int,
    use_videos: bool,
    resume: bool,
    extra_observation_features: dict[str, type | tuple] | None = None,
) -> Any:
    from lerobot.datasets.feature_utils import hw_to_dataset_features
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import INFO_PATH
    from lerobot.utils.constants import ACTION, OBS_STR

    # XLeVR changes the process working directory while running. Keep every
    # dataset read and write anchored to the directory selected at startup.
    dataset_root_path = Path(dataset_root).expanduser().resolve()
    if resume and (dataset_root_path / INFO_PATH).exists():
        print(f"Resuming LeRobot dataset `{dataset_id}` at {dataset_root_path}.")
        return LeRobotDataset.resume(
            dataset_id,
            root=dataset_root_path,
        )

    observation_features = dict(robot.observation_features)
    if extra_observation_features:
        observation_features.update(extra_observation_features)
    action_features = hw_to_dataset_features(robot.action_features, ACTION, use_video=use_videos)
    obs_features = hw_to_dataset_features(observation_features, OBS_STR, use_video=use_videos)
    dataset_features = {**action_features, **obs_features}
    print(f"Creating LeRobot dataset `{dataset_id}` at {dataset_root_path}.")
    return LeRobotDataset.create(
        dataset_id,
        fps,
        root=dataset_root_path,
        robot_type=robot.name,
        features=dataset_features,
        use_videos=use_videos,
    )


def _record_frame_if_needed(recording: RecordingSession | None, observation: dict[str, Any], action: dict[str, Any]) -> None:
    if recording is None or not recording.active:
        return
    observation = _recording_observation_with_cached_images(recording, observation)
    missing = _missing_dataset_observation_values(recording.dataset.features, observation)
    recording.last_missing_features = tuple(missing)
    if missing:
        now = time.time()
        if now - recording.last_missing_feature_warn_t >= 5.0:
            print(f"Recording frame skipped; missing observation values: {', '.join(missing)}")
            recording.last_missing_feature_warn_t = now
        return

    from lerobot.datasets.feature_utils import build_dataset_frame
    from lerobot.utils.constants import ACTION, OBS_STR

    observation_frame = build_dataset_frame(recording.dataset.features, observation, prefix=OBS_STR)
    complete_action = dict(action)
    for action_name in recording.dataset.features[ACTION]["names"]:
        if action_name in complete_action:
            continue
        if action_name in observation:
            complete_action[action_name] = observation[action_name]
        elif action_name.endswith(".vel"):
            complete_action[action_name] = 0.0
        else:
            complete_action[action_name] = 0.0

    action_frame = build_dataset_frame(recording.dataset.features, complete_action, prefix=ACTION)
    frame = {
        **observation_frame,
        **action_frame,
        "task": recording.task,
    }
    recording.dataset.add_frame(frame)
    recording.episode_frame_count += 1
    recording.last_missing_features = ()


def _augment_recording_observation(recording: RecordingSession, observation: dict[str, Any]) -> dict[str, Any]:
    if recording.orbbec_output_dir is None:
        return observation
    augmented = dict(observation)
    if recording.orbbec_camera_key not in augmented:
        frame = _latest_orbbec_rgb_array(recording.orbbec_output_dir)
        if frame is not None:
            augmented[recording.orbbec_camera_key] = frame
    return augmented


def _recording_observation_with_cached_images(
    recording: RecordingSession,
    observation: dict[str, Any],
) -> dict[str, Any]:
    image_keys = _dataset_image_observation_keys(recording.dataset.features)
    if not image_keys:
        return observation

    now = time.monotonic()
    augmented = dict(observation)
    for key in image_keys:
        if key in augmented:
            recording.latest_observation_images[key] = (_copy_observation_image(augmented[key]), now)
            continue
        cached = recording.latest_observation_images.get(key)
        if cached is None:
            continue
        value, captured_at = cached
        if now - captured_at <= _RECORDING_IMAGE_CACHE_MAX_AGE_S:
            augmented[key] = value
    return augmented


def _copy_observation_image(value: Any) -> Any:
    try:
        import numpy as np

        return np.asarray(value).copy()
    except Exception:
        return value


def _dataset_image_observation_keys(features: dict[str, dict]) -> list[str]:
    from lerobot.utils.constants import OBS_STR

    prefix = f"{OBS_STR}.images."
    return [
        key.removeprefix(prefix)
        for key, ft in features.items()
        if key.startswith(prefix) and ft["dtype"] in {"image", "video"}
    ]


def _latest_orbbec_rgb_array(output_dir: Path) -> Any | None:
    path = output_dir / "latest.ppm"
    if not path.exists():
        return None
    try:
        return _ppm_file_to_rgb_array(path)
    except Exception:
        return None


def _ppm_file_to_rgb_array(path: Path) -> Any:
    import numpy as np

    data = path.read_bytes()
    offset = 0

    def skip_ws_and_comments() -> None:
        nonlocal offset
        while offset < len(data):
            value = data[offset]
            if value == 35:
                while offset < len(data) and data[offset] != 10:
                    offset += 1
            elif value in (9, 10, 13, 32):
                offset += 1
            else:
                break

    def token() -> bytes:
        nonlocal offset
        skip_ws_and_comments()
        start = offset
        while offset < len(data) and data[offset] not in (9, 10, 13, 32, 35):
            offset += 1
        return data[start:offset]

    magic = token()
    width = int(token())
    height = int(token())
    max_value = int(token())
    skip_ws_and_comments()
    if magic != b"P6" or max_value != 255:
        raise ValueError(f"Unsupported PPM header in {path}")
    rgb = np.frombuffer(data, dtype=np.uint8, count=width * height * 3, offset=offset)
    return np.ascontiguousarray(rgb.reshape((height, width, 3)))


def _missing_dataset_observation_values(features: dict[str, dict], observation: dict[str, Any]) -> list[str]:
    from lerobot.utils.constants import OBS_STR

    missing: list[str] = []
    for key, ft in features.items():
        if not key.startswith(OBS_STR):
            continue
        if ft["dtype"] == "float32" and len(ft["shape"]) == 1:
            missing.extend(name for name in ft["names"] if name not in observation)
        elif ft["dtype"] in {"image", "video"}:
            raw_key = key.removeprefix(f"{OBS_STR}.images.")
            if raw_key not in observation:
                missing.append(raw_key)
    return missing


def _handle_recording_hotkeys(
    recording: RecordingSession | None,
    pressed_keys: set[str],
    *,
    start_key: str,
    stop_key: str,
) -> None:
    if recording is None:
        return

    if start_key in pressed_keys and not recording.active:
        _start_recording_session(recording)
        print(f"Recording started. Press `{stop_key}` to save the current episode.")
        return

    if stop_key in pressed_keys and recording.active:
        _save_episode(recording)


def _toggle_recording_from_vr_button(recording: RecordingSession, button_spec: str) -> None:
    if recording.active:
        print(f"Training recording toggle `{button_spec}` pressed: saving episode.")
        _save_episode(recording)
        return
    _start_recording_session(recording)
    print(f"Training recording toggle `{button_spec}` pressed: recording started.")


def _discard_recording_from_vr_button(recording: RecordingSession, button_spec: str) -> None:
    if not recording.active:
        print(f"Training discard `{button_spec}` pressed, but no episode is active.")
        return
    print(f"Training discard `{button_spec}` pressed: discarding active episode.")
    _discard_episode(recording)


def _discard_or_finish_recording_from_vr_button(recording: RecordingSession, button_spec: str) -> None:
    if recording.active:
        _discard_recording_from_vr_button(recording, button_spec)
        return
    print(f"Training discard `{button_spec}` pressed while no episode is active: finalizing dataset collection.")
    _request_finish_collection()


def _start_recording_session(recording: RecordingSession) -> None:
    recording.active = True
    recording.episode_frame_count = 0
    recording.last_missing_features = ()


def _save_episode(recording: RecordingSession) -> bool:
    if not _episode_buffer_has_frames(recording.dataset):
        recording.active = False
        if recording.last_missing_features:
            print(
                "Recording stopped. No frames captured; last missing observation values: "
                f"{', '.join(recording.last_missing_features)}. Skipping save."
            )
        else:
            print("Recording stopped. No frames captured, skipping save.")
        recording.episode_frame_count = 0
        return False

    recording.dataset.save_episode()
    _restore_dataset_owner_after_sudo(recording)
    recording.session_episode_count += 1
    recording.active = False
    recording.episode_frame_count = 0
    recording.last_missing_features = ()
    print(
        f"Saved episode {recording.dataset.meta.total_episodes - 1} "
        f"({recording.session_episode_count} saved this session)."
    )
    return True


def _finalize_recording(recording: RecordingSession | None) -> None:
    if recording is None:
        return
    if recording.active:
        print("Discarding the active unsaved episode before exit.")
        _discard_episode(recording)
    finalize = getattr(recording.dataset, "finalize", None)
    if callable(finalize):
        print("Finalizing LeRobot dataset.")
        finalize()
        _restore_dataset_owner_after_sudo(recording)
        total_episodes = getattr(getattr(recording.dataset, "meta", None), "total_episodes", None)
        if total_episodes is not None:
            print(f"LeRobot dataset finalized with {total_episodes} episode(s).")


def _restore_dataset_owner_after_sudo(recording: RecordingSession) -> None:
    if os.geteuid() != 0:
        return
    sudo_uid = os.environ.get("SUDO_UID")
    sudo_gid = os.environ.get("SUDO_GID")
    if not sudo_uid or not sudo_gid:
        return
    try:
        uid = int(sudo_uid)
        gid = int(sudo_gid)
    except ValueError:
        return
    root = recording.dataset_root or Path(getattr(recording.dataset, "root", ""))
    if not root:
        return
    root = Path(root).expanduser().resolve()
    if not root.exists():
        return
    try:
        os.chown(root, uid, gid)
        for path in root.rglob("*"):
            try:
                os.chown(path, uid, gid)
            except OSError:
                continue
    except OSError as exc:
        print(f"Warning: could not restore dataset ownership for {root}: {exc}")


def _discard_episode(recording: RecordingSession) -> None:
    clear_episode_buffer = getattr(recording.dataset, "clear_episode_buffer", None)
    if callable(clear_episode_buffer):
        clear_episode_buffer()
    else:
        buffer = _dataset_episode_buffer(recording.dataset)
        if isinstance(buffer, dict):
            for key, value in buffer.items():
                if key == "size":
                    buffer[key] = 0
                elif hasattr(value, "clear"):
                    value.clear()
    recording.active = False
    recording.episode_frame_count = 0
    recording.last_missing_features = ()
    print("Discarded the current episode.")


def _episode_buffer_has_frames(dataset: Any) -> bool:
    buffer = _dataset_episode_buffer(dataset)
    return bool(buffer and buffer.get("size", 0) > 0)


def _dataset_episode_buffer(dataset: Any) -> dict[str, Any] | None:
    buffer = getattr(dataset, "episode_buffer", None)
    if isinstance(buffer, dict):
        return buffer
    writer = getattr(dataset, "writer", None)
    buffer = getattr(writer, "episode_buffer", None) if writer is not None else None
    return buffer if isinstance(buffer, dict) else None


def _print_recording_guide(
    recording: RecordingSession | None,
    *,
    start_key: str,
    stop_key: str,
    quit_key: str,
    controller: str = "keyboard",
) -> None:
    print(f"Quit key: `{quit_key}`")
    if controller == "vr":
        print(
            "VR controls: left thumbstick press starts/saves recording; "
            "right thumbstick opens the recording menu for Record, Save, Cancel, and Finish."
        )
        print(
            "VR IK clutch: hold Quest squeeze/grip to freeze that arm; "
            "keyboard `1`/`2` remain left/right fallbacks."
        )
    if recording is None:
        return
    if controller == "keyboard":
        print(f"Recording hotkeys: `{start_key}` start, `{stop_key}` stop and save")


def _load_example_module(file_path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to build import spec for {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _map_vr_events_to_recording_controls(vr_events: dict[str, bool] | None) -> VRRecordingControls:
    if not vr_events:
        return VRRecordingControls()

    return VRRecordingControls(
        reset_robot=bool(vr_events.get("reset_position")),
    )


def _decide_vr_recording_action(active: bool, controls: VRRecordingControls) -> VRRecordingDecision:
    toggle_requested = (
        controls.toggle_recording
        and not controls.discard_episode
        and not controls.quit_session
    )
    return VRRecordingDecision(
        start_recording=toggle_requested and not active,
        save_episode=toggle_requested and active,
        discard_episode=controls.discard_episode and active,
        quit_session=controls.quit_session,
        reset_robot=controls.reset_robot,
    )


def _apply_vr_recording_decision(
    recording: RecordingSession,
    decision: VRRecordingDecision,
) -> None:
    if decision.start_recording:
        _start_recording_session(recording)
        print("Recording started from VR. Press left thumbstick to save, or open the menu to cancel.")
    if decision.save_episode:
        _save_episode(recording)
    if decision.discard_episode:
        _discard_episode(recording)
    if decision.quit_session:
        print("Stopping the VR recording session.")


if __name__ == "__main__":
    raise SystemExit(main())
