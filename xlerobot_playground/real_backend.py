from __future__ import annotations

import argparse
import builtins
import importlib.util
import json
import ssl
import threading
import subprocess
import sys
import time
import urllib.parse
from dataclasses import dataclass
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


@dataclass
class RecordingSession:
    dataset: Any
    task: str
    active: bool = False


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
    vertical_sign: float
    y_gain: float
    z_gain: float
    ik_alpha: float


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backend launcher for XLeRobot real teleop and local LeRobot recording."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    manipulate = subparsers.add_parser("manipulate", help="Launch real teleoperation.")
    _add_shared_args(manipulate)
    manipulate.add_argument("--controller", choices=("keyboard", "vr"), default="keyboard")

    record = subparsers.add_parser("record", help="Launch real teleop with local LeRobot recording.")
    _add_shared_args(record)
    record.add_argument("--controller", choices=("keyboard", "vr"), default="keyboard")
    record.add_argument("--dataset-id", default="local/xlerobot_playground")
    record.add_argument("--dataset-root", default="./datasets")
    record.add_argument("--task", default="XLeRobot teleoperation")
    record.add_argument("--use-videos", action="store_true")
    record.add_argument("--start-key", default="[")
    record.add_argument("--stop-key", default="]")
    record.add_argument("--quit-key", default="\\")
    return parser


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
            "`left_wrist=opencv:/dev/video0`."
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
    parser.add_argument("--vr-base-max-linear", type=float, default=0.25)
    parser.add_argument("--vr-base-max-angular", type=float, default=75.0)
    parser.add_argument("--vr-base-linear-accel", type=float, default=0.9)
    parser.add_argument("--vr-base-angular-accel", type=float, default=240.0)
    parser.add_argument("--vr-base-deadzone", type=float, default=0.14)
    parser.add_argument("--vr-base-curve", type=float, default=1.5)
    parser.add_argument("--vr-arm-vertical-sign", type=float, default=1.0)
    parser.add_argument("--vr-arm-y-gain", type=float, default=1.4)
    parser.add_argument("--vr-arm-z-gain", type=float, default=1.0)
    parser.add_argument("--vr-arm-ik-alpha", type=float, default=0.25)
    parser.add_argument(
        "--no-vr-startup-pose",
        action="store_true",
        help="Skip NAV_STOW -> ACTION_READY startup pose routine.",
    )
    parser.add_argument("--vr-nav-stow-wait-s", type=float, default=10.0)
    parser.add_argument(
        "--vr-action-ready-elbow-delta",
        type=float,
        default=-80.0,
        help="Elbow flex delta applied when moving from NAV_STOW to ACTION_READY.",
    )
    parser.add_argument(
        "--vr-action-ready-shoulder-delta",
        type=float,
        default=90.0,
        help="Shoulder lift delta applied after elbow when moving from NAV_STOW to ACTION_READY.",
    )
    parser.add_argument(
        "--vr-action-ready-wrist-delta",
        type=float,
        default=-40.0,
        help="Optional wrist flex delta applied during the elbow stage of ACTION_READY.",
    )
    parser.add_argument("--vr-startup-pose-steps", type=int, default=40)
    parser.add_argument("--vr-startup-pose-stage-delay-s", type=float, default=0.02)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bootstrap_xlerobot(args.repo_root)

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
    if args.mode == "record":
        recording = RecordingSession(
            dataset=_create_dataset(
                robot,
                dataset_id=args.dataset_id,
                dataset_root=args.dataset_root,
                fps=args.fps,
                use_videos=args.use_videos,
            ),
            task=args.task,
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
        orbbec_rgb=OrbbecRgbConfig(
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
        ),
        vr_input_scale=args.vr_input_scale,
        vr_kp=args.vr_kp,
        vr_camera_hz=args.vr_camera_hz,
        base_smoother=BaseSmoother(
            max_linear=args.vr_base_max_linear,
            max_angular=args.vr_base_max_angular,
            linear_accel=args.vr_base_linear_accel,
            angular_accel=args.vr_base_angular_accel,
            deadzone=args.vr_base_deadzone,
            curve=args.vr_base_curve,
        ),
        arm_tuning=VrArmTuning(
            vertical_sign=args.vr_arm_vertical_sign,
            y_gain=args.vr_arm_y_gain,
            z_gain=args.vr_arm_z_gain,
            ik_alpha=args.vr_arm_ik_alpha,
        ),
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
    from lerobot.cameras.configs import ColorMode, Cv2Rotation
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

    cameras: dict[str, Any] = {}
    for raw_spec in camera_specs:
        spec = _parse_camera_spec(raw_spec)
        if spec.driver == "opencv":
            source: Any = int(spec.source) if spec.source.isdigit() else spec.source
            cameras[spec.name] = OpenCVCameraConfig(
                index_or_path=source,
                fps=fps,
                width=width,
                height=height,
                rotation=Cv2Rotation.NO_ROTATION,
            )
            continue
        if spec.driver == "realsense":
            cameras[spec.name] = RealSenseCameraConfig(
                serial_number_or_name=spec.source,
                fps=fps,
                width=width,
                height=height,
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
    driver, source = remainder.split(":", 1)
    return CameraSpec(name=name.strip(), driver=driver.strip(), source=source.strip())


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


def _install_orbbec_vr_overlay(vr_teleop: Any, output_dir: Path, *, include_orbbec: bool) -> bool:
    monitor = getattr(vr_teleop, "vr_monitor", None)
    if monitor is None:
        return False
    handler_cls = getattr(sys.modules.get(monitor.__class__.__module__), "SimpleAPIHandler", None)
    if handler_cls is None:
        return False
    if getattr(handler_cls, "_orbbec_overlay_installed", False):
        handler_cls.orbbec_output_dir = output_dir.resolve()
        handler_cls.orbbec_overlay_include_orbbec = include_orbbec
        return True

    original_do_get = handler_cls.do_GET
    original_do_post = getattr(handler_cls, "do_POST", None)
    original_serve_file = handler_cls.serve_file
    handler_cls.orbbec_output_dir = output_dir.resolve()
    handler_cls.orbbec_overlay_include_orbbec = include_orbbec

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
        if self.path.startswith("/webrtc/offer"):
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
                content += "\n\n" + _orbbec_vr_overlay_js(
                    include_orbbec=getattr(handler_cls, "orbbec_overlay_include_orbbec", True)
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
            obs[cam_key] = cam.async_read()
        except Exception as exc:
            if first_error is None:
                first_error = exc
            continue
    return obs, first_error


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


def _sync_vr_teleop_to_current_pose(robot: Any, vr_teleop: Any) -> None:
    obs = _get_robot_observation(robot, use_camera=False)
    for side in ("left", "right"):
        arm = getattr(vr_teleop, f"{side}_arm", None)
        if arm is None:
            continue
        targets: dict[str, float] = {}
        for joint in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"):
            obs_key = f"{side}_arm_{joint}.pos"
            if obs_key in obs:
                targets[joint] = float(obs[obs_key])
        if targets:
            arm.target_positions = targets
            arm.pitch = targets.get("wrist_flex", 0.0) + targets.get("shoulder_lift", 0.0) + targets.get("elbow_flex", 0.0)
            kin = getattr(arm, "kinematics", None)
            if kin is not None and hasattr(kin, "forward_kinematics"):
                try:
                    arm.current_x, arm.current_y = kin.forward_kinematics(
                        targets.get("shoulder_lift", 0.0),
                        targets.get("elbow_flex", 0.0),
                    )
                except Exception:
                    pass
        for attr in ("prev_vr_pos", "prev_wrist_flex", "prev_wrist_roll"):
            if hasattr(arm, attr):
                delattr(arm, attr)

    head = getattr(vr_teleop, "head_control", None)
    if head is not None:
        head_targets = {
            motor: float(obs[f"{motor}.pos"])
            for motor in ("head_motor_1", "head_motor_2")
            if f"{motor}.pos" in obs
        }
        if head_targets:
            head.target_positions = head_targets


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
    _write_json_response(
        handler,
        (
            '{"left_arm_connected":true,"right_arm_connected":true,'
            '"vrConnected":true,"keyboardEnabled":false,"robotEngaged":true}'
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


def _orbbec_vr_overlay_js(*, include_orbbec: bool) -> str:
    return r"""
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
      pollMs: 160
    }
  ];
  const INCLUDE_ORBBEC = __INCLUDE_ORBBEC__;
  const ACTIVE_FEEDS = FEEDS.filter(feed => INCLUDE_ORBBEC || feed.name !== 'orbbec');
  const overlays = new Map();
  let lastStatusLog = 0;
  let webRtcStarted = false;
  let webRtcFailed = false;
  let webRtcPc = null;

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

      const material = new THREE.MeshBasicMaterial({
        map: texture,
        side: THREE.DoubleSide,
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
        frameCount: 0
      });
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
    if (webRtcStarted || webRtcFailed) return;
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
    startWebRtc();
    function renderStreams() {
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
""".replace("__INCLUDE_ORBBEC__", "true" if include_orbbec else "false")


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
    vr_input_scale: float,
    vr_kp: float,
    vr_camera_hz: float,
    base_smoother: BaseSmoother,
    arm_tuning: VrArmTuning,
    startup_pose: VrStartupPoseConfig,
) -> int:
    from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardTeleopConfig
    from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
    from lerobot.utils.errors import DeviceNotConnectedError
    from lerobot.utils.robot_utils import precise_sleep
    from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

    hotkeys = KeyboardTeleop(KeyboardTeleopConfig())
    previous_pressed_keys: set[str] = set()

    orbbec_process = _start_orbbec_rgb_sidecar(orbbec_rgb)
    _connect_robot(robot, auto_restore_calibration=auto_restore_calibration)
    init_rerun(session_name="xlerobot_real_vr_playground")
    hotkeys.connect()

    vr_overrides = {"kp": vr_kp}
    if xlevr_path is not None:
        vr_overrides["xlevr_path"] = xlevr_path
    vr_teleop = interface.make_vr_teleop(**vr_overrides)
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
    installed = _install_orbbec_vr_overlay(
        vr_teleop,
        orbbec_rgb.output_dir,
        include_orbbec=orbbec_rgb.enabled,
    )
    if installed:
        if orbbec_rgb.enabled:
            print("Orbbec RGB VR overlay enabled. Reload the Quest page if it was already open.")
        print("VR arm camera panels enabled for camera names: left_wrist and right_wrist.")
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

    camera_interval_s = 1.0 / max(0.1, vr_camera_hz)
    next_camera_t = 0.0
    last_camera_warn_t = 0.0

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
            _handle_recording_hotkeys(
                recording,
                newly_pressed,
                start_key=start_key,
                stop_key=stop_key,
            )

            vr_controls = _map_vr_events_to_recording_controls(vr_teleop.get_vr_events())
            vr_decision = VRRecordingDecision(reset_robot=vr_controls.reset_robot)
            if recording is not None:
                vr_decision = _decide_vr_recording_action(
                    recording.active,
                    vr_controls,
                )
                _apply_vr_recording_decision(recording, vr_decision)
                if vr_decision.quit_session:
                    break

            obs = _get_robot_observation(robot, use_camera=False)
            if vr_decision.reset_robot:
                _move_to_action_ready(robot, vr_teleop, startup_pose)
                action = {}
            else:
                action = vr_teleop.get_action(obs, robot)
            action = _smooth_vr_base_action(action, base_smoother)
            if action:
                sent_action = robot.send_action(action)
            else:
                sent_action = {}
            now = time.perf_counter()
            if now >= next_camera_t:
                next_camera_t = now + camera_interval_s
                camera_obs, camera_error = _get_robot_observation_best_effort(robot)
                if camera_error is not None and now - last_camera_warn_t >= 5.0:
                    print(f"VR camera read skipped: {camera_error}")
                    last_camera_warn_t = now
                _publish_vr_camera_frames(camera_obs)
                if recording is not None:
                    obs = camera_obs
            log_rerun_data(obs, sent_action)
            _record_frame_if_needed(recording, obs, sent_action)

            dt_s = time.perf_counter() - start_loop_t
            precise_sleep(max(0.0, 1 / fps - dt_s))
    finally:
        _finalize_recording(recording)
        _stop_orbbec_rgb_sidecar(orbbec_process)
        try:
            robot.disconnect()
        finally:
            try:
                vr_teleop.disconnect()
            except Exception:
                pass
            if hotkeys.is_connected:
                hotkeys.disconnect()
    return 0


def _configure_vr_runtime(vr_teleop: Any, *, input_scale: float) -> None:
    monitor = getattr(vr_teleop, "vr_monitor", None)
    config = getattr(monitor, "config", None)
    if config is not None and hasattr(config, "vr_to_robot_scale"):
        config.vr_to_robot_scale = input_scale
    print(f"VR input scale set to {input_scale:.3f}.")


def _install_vr_arm_tuning(vr_teleop: Any, tuning: VrArmTuning) -> None:
    import types

    for arm_name in ("left_arm", "right_arm"):
        arm = getattr(vr_teleop, arm_name, None)
        if arm is None:
            continue
        arm.handle_vr_input = types.MethodType(_tuned_vr_arm_input(tuning), arm)
    print(
        "VR arm tuning: "
        f"vertical_sign={tuning.vertical_sign:g}, "
        f"y_gain={tuning.y_gain:.2f}, "
        f"z_gain={tuning.z_gain:.2f}, "
        f"ik_alpha={tuning.ik_alpha:.2f}."
    )


def _tuned_vr_arm_input(tuning: VrArmTuning) -> Any:
    def handle_vr_input(self: Any, vr_goal: Any, gripper_state: Any) -> None:
        if vr_goal is None or not hasattr(vr_goal, "target_position") or vr_goal.target_position is None:
            return

        current_vr_pos = vr_goal.target_position
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

        if abs(delta_x) > 0.001:
            delta_pan = max(-angle_limit, min(angle_limit, delta_x * 200.0))
            current_pan = self.target_positions.get("shoulder_pan", 0.0)
            self.target_positions["shoulder_pan"] = max(-180, min(180, current_pan + delta_pan))

        try:
            joint2_target, joint3_target = self.kinematics.inverse_kinematics(self.current_x, self.current_y)
            alpha = max(0.01, min(1.0, tuning.ik_alpha))
            self.target_positions["shoulder_lift"] = (
                (1 - alpha) * self.target_positions.get("shoulder_lift", 0.0) + alpha * joint2_target
            )
            self.target_positions["elbow_flex"] = (
                (1 - alpha) * self.target_positions.get("elbow_flex", 0.0) + alpha * joint3_target
            )
        except Exception as exc:
            print(f"[{self.prefix}] VR IK failed: {exc}")

        self.target_positions["wrist_flex"] = (
            -self.target_positions["shoulder_lift"] - self.target_positions["elbow_flex"] + self.pitch
        )
        self.target_positions["gripper"] = 45 if vr_goal.metadata.get("trigger", 0) > 0.5 else 0.0

    return handle_vr_input


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
) -> Any:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import hw_to_dataset_features
    from lerobot.utils.constants import ACTION, OBS_STR

    action_features = hw_to_dataset_features(robot.action_features, ACTION, use_video=use_videos)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=use_videos)
    dataset_features = {**action_features, **obs_features}
    return LeRobotDataset.create(
        dataset_id,
        fps,
        root=dataset_root,
        robot_type=robot.name,
        features=dataset_features,
        use_videos=use_videos,
    )


def _record_frame_if_needed(recording: RecordingSession | None, observation: dict[str, Any], action: dict[str, Any]) -> None:
    if recording is None or not recording.active:
        return

    from lerobot.datasets.utils import build_dataset_frame
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
        "timestamp": time.time(),
    }
    recording.dataset.add_frame(frame)


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
        recording.active = True
        print(f"Recording started. Press `{stop_key}` to save the current episode.")
        return

    if stop_key in pressed_keys and recording.active:
        _save_episode(recording)


def _save_episode(recording: RecordingSession) -> None:
    if not _episode_buffer_has_frames(recording.dataset):
        recording.active = False
        print("Recording stopped. No frames captured, skipping save.")
        return

    recording.dataset.save_episode()
    recording.active = False
    print(f"Saved episode {recording.dataset.meta.total_episodes - 1}.")


def _finalize_recording(recording: RecordingSession | None) -> None:
    if recording is None or not recording.active:
        return
    print("Saving the active episode before exit.")
    _save_episode(recording)


def _discard_episode(recording: RecordingSession) -> None:
    clear_episode_buffer = getattr(recording.dataset, "clear_episode_buffer", None)
    if callable(clear_episode_buffer):
        clear_episode_buffer()
    else:
        buffer = getattr(recording.dataset, "episode_buffer", None)
        if isinstance(buffer, dict):
            for key, value in buffer.items():
                if key == "size":
                    buffer[key] = 0
                elif hasattr(value, "clear"):
                    value.clear()
    recording.active = False
    print("Discarded the current episode.")


def _episode_buffer_has_frames(dataset: Any) -> bool:
    buffer = getattr(dataset, "episode_buffer", None)
    return bool(buffer and buffer.get("size", 0) > 0)


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
            "VR controls: left thumbstick right start/stop and save, "
            "left thumbstick left discard, left thumbstick up save and quit, "
            "left thumbstick down reset robot pose"
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
        toggle_recording=bool(vr_events.get("exit_early")),
        discard_episode=bool(vr_events.get("rerecord_episode")),
        quit_session=bool(vr_events.get("stop_recording")),
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
        recording.active = True
        print("Recording started from VR. Push left thumbstick right again to save.")
    if decision.save_episode:
        _save_episode(recording)
    if decision.discard_episode:
        _discard_episode(recording)
    if decision.quit_session:
        print("Stopping the VR recording session.")


if __name__ == "__main__":
    raise SystemExit(main())
