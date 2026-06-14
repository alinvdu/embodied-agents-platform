# XLeRobot VR Teleoperation

This document describes the current Robot42 VR teleoperation setup for XLeRobot VLA data collection.

## Standard Command

Run this on the robot brain machine:

```bash
sudo /Users/alindumitru/miniconda3/envs/xlerobot/bin/python -m xlerobot_playground.real_backend manipulate \
  --repo-root /Users/alindumitru/XLeRobot \
  --robot-kind xlerobot_2wheels \
  --controller vr \
  --port1 /dev/tty.usbmodem5B140330101 \
  --port2 /dev/tty.usbmodem5B140332271 \
  --xlevr-path /Users/alindumitru/XLeRobot/XLeVR \
  --orbbec-rgb-vr \
  --camera left_wrist=opencv:0 \
  --camera right_wrist=opencv:1
```

The backend prints the Quest URL after startup, usually:

```text
https://<robot-brain-ip>:8443
```

Open that URL in the Quest browser and start controller tracking.

## Cameras

The current VR setup uses three camera feeds:

- Orbbec Gemini 2 RGB as the center/head view.
- `opencv:0` as the left wrist camera.
- `opencv:1` as the right wrist camera.

Controller pose data uses WebSocket on port `8442`.
Camera video uses WebRTC video tracks rendered into the A-Frame/Three.js scene.

## Startup Pose

VR teleoperation no longer starts from the old all-middle/zero pose.

Startup now runs:

```text
NAV_STOW -> wait -> ACTION_READY -> VR teleop
```

`NAV_STOW` is baked into the backend from the captured folded pose. One captured arm pose is applied to both arms:

```text
shoulder_pan  -4.8316
shoulder_lift -99.1708
elbow_flex    100.0
wrist_flex    76.2061
wrist_roll    0.1709
gripper       0.9466
```

After reaching `NAV_STOW`, the backend prints:

```text
folded
```

It then waits 10 seconds and moves to `ACTION_READY` using staged deltas:

```text
elbow_flex    -80
shoulder_lift +90
wrist_flex    -40
```

The order is intentional:

```text
1. Elbow and wrist move first.
2. Shoulder lift moves second.
3. VR IK state is synced to the reached pose.
```

This avoids a direct sweep from folded navigation pose to an IK-friendly pose that may collide with the surface.

## Tuning Startup Pose

The default ACTION_READY deltas are already baked in. Override them only when testing:

```bash
--vr-action-ready-elbow-delta -80 \
--vr-action-ready-shoulder-delta 90 \
--vr-action-ready-wrist-delta -40
```

To change the wait after folded pose:

```bash
--vr-nav-stow-wait-s 5
```

To skip the startup pose routine and use the old zero/middle behavior:

```bash
--no-vr-startup-pose
```

## Base Control

The right thumbstick controls the differential base. Robot42 smooths base velocity commands before sending them to the wheels.

Default base tuning:

```text
max linear speed    0.25 m/s
max angular speed   75 deg/s
linear accel        0.9 m/s^2
angular accel       240 deg/s^2
deadzone            0.14
curve               1.5
```

## VR Recording Controls

In `record` mode, left controller thumbstick events map to recording controls:

```text
left thumbstick right  start / stop and save episode
left thumbstick left   discard active episode
left thumbstick up     save and quit session
left thumbstick down   reset robot to ACTION_READY
```

In plain `manipulate` mode, left thumbstick down also resets the robot to `ACTION_READY`.

## Dataset Practice

For VLA data collection:

```text
1. Let the robot navigate folded in NAV_STOW.
2. Move to ACTION_READY outside the recorded episode.
3. Start recording.
4. Demonstrate grab and basket drop.
5. Stop/save immediately after success.
6. Reset outside the episode.
```

Do not include reset/stow motions inside the task episode unless intentionally training a reset skill.

## Pose Capture Utility

To capture a new folded pose from the physical robot:

```bash
/Users/alindumitru/miniconda3/envs/xlerobot/bin/python /Users/alindumitru/Robot42/scripts/capture_xlerobot_pose.py
```

Position the robot first, then run the script. It prints the observed joint values and reusable pose flags.
