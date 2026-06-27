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

It then waits 5 seconds and moves to `ACTION_READY` using staged deltas:

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

## Tuning VR IK Limits

Robot42's VR path uses a local SO101 IK override with tunable final joint limits. The original SO101 IK positive elbow side already reaches about `106` degrees, but the negative side was capped around `-85.3` degrees.

For basket tuning, the negative elbow side is the likely limit to test. Try small changes cautiously:

```bash
--vr-arm-elbow-flex-min -118
```

The related limit flags are:

```bash
--vr-arm-shoulder-lift-min -108
--vr-arm-shoulder-lift-max 96
--vr-arm-elbow-flex-min -115
--vr-arm-elbow-flex-max 106
```

To temporarily disable the software shoulder/elbow clipping:

```bash
--vr-arm-unbounded-ik-joints
```

This only removes Robot42's final IK joint clamp. The IK workspace radius clamp remains so the math stays valid, and the robot's physical, servo, firmware, and calibration limits may still prevent motion.

To print throttled IK diagnostics while testing:

```bash
--vr-arm-debug --vr-arm-debug-hz 2
```

Useful debug notes:

```text
elbow_min / elbow_max        IK is asking for the configured elbow limit
shoulder_min / shoulder_max  IK is asking for the configured shoulder limit
would_elbow_min / max        same as above, but software limits are disabled
near_min_radius              controller target is near the folded inside workspace boundary
near_max_radius              controller target is near full extension
```

## Experimental Yawed 3D IK

The default `planar` mode keeps the original behavior: controller side motion mostly changes shoulder pan, while forward/back and up/down drive the 2D arm IK.

For basket placement, test `yawed` mode. It treats the controller target as `forward + lateral + height`, computes shoulder pan from the lateral target, then solves the same arm IK using the radial distance. This should make side aiming and basket drops feel more continuous.

```bash
--vr-arm-ik-mode yawed --vr-arm-debug --vr-arm-debug-hz 2
```

In debug output, look for:

```text
mode=yawed
target3d=(fwd=...,lat=...,h=...,pan=...)
```

Forward/back controller motion should change `fwd`, side motion should change `lat` and `pan`, and up/down should change `h`. If side motion points the wrong way, flip the lateral gain:

```bash
--vr-arm-yawed-lateral-gain -0.5
```

If the arm pans opposite to the desired lateral target, flip the pan sign:

```bash
--vr-arm-yawed-pan-sign -1
```

To skip the startup pose routine and use the old zero/middle behavior:

```bash
--no-vr-startup-pose
```

## Tuning VR Camera Display

The wrist camera panels use display-only gain/gamma correction in the Quest overlay. This does not alter the raw camera frames or recorded dataset images.

Defaults:

```bash
--vr-wrist-video-gain 1.65
--vr-wrist-video-gamma 0.78
--vr-wrist-video-bias 0.02
```

If the wrist panels are too bright or washed out, reduce `--vr-wrist-video-gain`. If the shadows are still too dark, lower `--vr-wrist-video-gamma` slightly.

The Quest page also adds a `Hide video feeds` / `Show video feeds` button. It hides the headset video panels and green status markers without stopping the WebRTC streams.

If you need a specific wrist camera frame rate later, add an override directly in the camera spec:

```bash
--camera left_wrist=opencv:0,fps=30
--camera right_wrist=opencv:1,fps=30
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

In training recording mode, thumbstick button presses are the primary episode controls:

```text
right thumbstick press start / stop and save episode
left thumbstick press  discard active episode
```

The older left-thumbstick direction events remain available as fallbacks:

```text
left thumbstick right  fallback start / stop and save episode
left thumbstick left   fallback discard active episode
left thumbstick up     save and quit session
left thumbstick down   reset robot to ACTION_READY
```

In plain `manipulate` mode, left thumbstick down also resets the robot to `ACTION_READY`.

## IK Clutch

Use the IK clutch when your real hand/controller reaches an awkward pose but the robot arm is in a useful position.

```text
hold left squeeze/grip    freeze left-arm IK at observed pose
hold right squeeze/grip   freeze right-arm IK at observed pose
release squeeze/grip      rebaseline controller and resume from held robot pose
keyboard 1 / 2            left/right fallback clutch keys
```

While clutch is held, Robot42 refreshes the arm IK target from the observed motor pose and overwrites outgoing arm joint commands with the observed pose. On release, it refreshes the target one more time and swallows a few controller frames so accumulated controller motion is not applied.

To disable Quest squeeze/grip clutch and keep only keyboard fallback:

```bash
--no-vr-squeeze-clutch
```

## Grab To Basket Mode

For VLA data collection, `grab_to_basket` mode keeps normal VR IK for grabbing. When you press an arm's basket control, Robot42 disables IK for that arm and follows a direct joint-space path: increase elbow flex at the current pose, move toward an elbow-raised version of the basket pose, then lower into the captured basket pose. Pressing that arm's action-ready control sends the arm back to the captured `ACTION_READY` pose and then resumes IK.

```bash
--vr-skill-mode grab_to_basket --vr-skill-arm right
```

Use `--vr-skill-arm left`, `right`, or `both`. When one side is selected, the inactive arm is held at the captured `ACTION_READY` pose.

Controls:

```text
right B             right arm lifts, transfers above basket, then descends
right A             right arm back to ACTION_READY, then boundary-lock IK
left Y              left arm to fixed basket pose
left X              left arm back to ACTION_READY, then boundary-lock IK
keyboard r          reset both arms to ACTION_READY
left thumbstick down reset both arms to ACTION_READY
trigger             control gripper during any fixed motion or hold
```

Button-triggered fixed-pose motions are ramped by default so carrying an object to the basket is not a snap move:

```bash
--vr-basket-motion-s 4.0
--vr-basket-elbow-lift-deg -25
--vr-action-ready-motion-s 2.0
```

The basket trajectory uses 25% of its duration for the elbow-only lift, 50% for the raised transfer, and 25% for lowering into the final pose. The phases advance on time rather than observed servo error, so object weight cannot leave the sequence waiting forever. Adjust the signed `--vr-basket-elbow-lift-deg` to tune clearance; this robot currently uses a negative value to lift.

These settings only affect the `B/Y` basket move. `--vr-action-ready-motion-s` controls the `A/X` return-to-`ACTION_READY` move. None of them change the startup `NAV_STOW` -> `ACTION_READY` routine.

The baked basket poses are the captured right/left placement poses for the current physical basket. If you capture a better pose later, override any joint with repeated target flags:

```bash
--vr-basket-target right_arm_shoulder_pan.pos=-21.0835 \
--vr-basket-target right_arm_shoulder_lift.pos=-30.9287
```

While recording, all three joint-space stages are produced inside the main control loop, so the full trajectory is captured frame-by-frame in the episode. Live trigger/gripper commands pass through during the automatic move, ACTION_READY return, and boundary hold.

After `A/X` reaches `ACTION_READY`, the arm stays locked there. Press the right thumbstick once to save/end the current episode, or the left thumbstick once to cancel it. The next right-thumbstick press starts the next episode and releases IK from the captured `ACTION_READY` pose. This two-step boundary also works without dataset recording.

## Dataset Practice

For VLA data collection:

```text
1. Let the robot navigate folded in NAV_STOW.
2. Move to ACTION_READY outside the recorded episode.
3. Start recording.
4. Demonstrate grab and basket drop.
5. Return to ACTION_READY and let the boundary lock engage.
6. Save or cancel with one thumbstick press.
7. Reset the object while the arm remains locked.
8. Start the next episode with the right thumbstick; IK resumes.
```

## Pose Utilities

To read the current arm/head pose without moving the robot:

```bash
sudo /Users/alindumitru/miniconda3/envs/xlerobot/bin/python /Users/alindumitru/Robot42/scripts/capture_xlerobot_pose.py \
  --repo-root /Users/alindumitru/XLeRobot \
  --robot-kind xlerobot_2wheels \
  --port1 /dev/tty.usbmodem5B140330101 \
  --port2 /dev/tty.usbmodem5B140332271
```

Position the robot first, then run the script. It prints the observed joint values and reusable pose flags.

To move to a test pose and print the readings afterward:

```bash
sudo /Users/alindumitru/miniconda3/envs/xlerobot/bin/python /Users/alindumitru/Robot42/scripts/set_xlerobot_pose.py \
  --repo-root /Users/alindumitru/XLeRobot \
  --robot-kind xlerobot_2wheels \
  --port1 /dev/tty.usbmodem5B140330101 \
  --port2 /dev/tty.usbmodem5B140332271 \
  --target right:shoulder_pan=-21.0835 \
  --target right:shoulder_lift=-30.9287 \
  --target right:elbow_flex=70.2739 \
  --target right:wrist_flex=63.2052 \
  --target right:wrist_roll=-2.7595 \
  --target right:gripper=1.7579
```
