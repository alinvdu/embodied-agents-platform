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
shoulder_lift +80
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
--vr-action-ready-shoulder-delta 80 \
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

The wrist camera panels default to neutral display values. Optional display-only gain/gamma correction can be enabled in the Quest overlay if needed. This does not alter the raw camera frames or recorded dataset images.

Defaults:

```bash
--vr-wrist-video-gain 1.0
--vr-wrist-video-gamma 1.0
--vr-wrist-video-bias 0.0
```

If the wrist panels are too dark, raise `--vr-wrist-video-gain` or lower `--vr-wrist-video-gamma` slightly.

If you need a specific wrist camera frame rate later, add an override directly in the camera spec:

```bash
--camera left_wrist=opencv:0,fps=30
--camera right_wrist=opencv:1,fps=30
```

## Base Control

The right thumbstick axis controls the differential base. Pressing the right thumbstick opens the recording menu. Robot42 smooths base velocity commands before sending them to the wheels.

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

In training recording mode, left thumbstick press is the fast start/save shortcut. The right thumbstick opens the in-headset recording menu and pauses teleop. The camera panels darken while the menu is open. Use the controller ray from the non-teleop hand and press trigger to select a menu action.

```text
left thumbstick press  start recording if idle, save episode if recording
right thumbstick press open / close recording menu
Record                 start recording or resume from ACTION_READY hold
Save                   save active episode
Cancel                 discard active episode
Finish                 finalize dataset and exit
```

Cancel and Finish remain menu-only so accidental shortcut presses cannot discard or finalize the dataset.

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

For VLA data collection, `grab_to_basket` mode keeps normal VR IK for grabbing. When you press the right arm's basket control, Robot42 disables IK for that arm and follows a captured waypoint path: a relative clearance pose from the actual grasp, a fixed over-basket pose close to the base, then the final basket placement pose. Pressing that arm's action-ready control follows a reverse-style captured path through the over-basket and clearance poses before settling at `ACTION_READY`.

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
--vr-basket-shoulder-back-deg 65
--vr-action-ready-motion-s 2.0
```

The right-arm basket trajectory uses 35% of its duration for the relative clearance pose, 40% for the over-basket transfer pose, and 25% for lowering into the final basket pose. The phases advance on time rather than observed servo error, so object weight cannot leave the sequence waiting forever. The shoulder/elbow heuristic flags are kept for fallback paths, but the right-arm dataset path now follows the captured waypoints.

These settings only affect the `B/Y` basket move. `--vr-action-ready-motion-s` controls the `A/X` return-to-`ACTION_READY` move; the right arm return uses captured waypoints so it does not sweep directly through the table. None of them change the startup `NAV_STOW` -> `ACTION_READY` routine.

The baked basket poses are the captured right/left placement poses for the current physical basket. If you capture a better final pose later, override any joint with repeated target flags:

```bash
--vr-basket-target right_arm_shoulder_pan.pos=-26.8668 \
--vr-basket-target right_arm_shoulder_lift.pos=-42.3715
```

While recording, the waypointed joint-space stages are produced inside the main control loop, so the full trajectory is captured frame-by-frame in the episode. Live trigger/gripper commands pass through during the automatic move, ACTION_READY return, and boundary hold.

After `A/X` reaches `ACTION_READY`, the arm stays locked there. Press the left thumbstick to save the active episode, or press the right thumbstick and choose Cancel from the menu. Press the left thumbstick again to start the next episode and release IK from the captured `ACTION_READY` pose. Choose Finish from the menu when collection is complete. This boundary also works without dataset recording; in that mode the menu behaves as a pause/resume control.

## Dataset Practice

For VLA data collection:

```text
1. Let the robot navigate folded in NAV_STOW.
2. Move to ACTION_READY outside the recorded episode.
3. Press the left thumbstick to start recording.
4. Demonstrate grab and basket drop.
5. Return to ACTION_READY and let the boundary lock engage.
6. Press the left thumbstick to save, or open the menu and select Cancel.
7. Reset the object while the arm remains locked.
8. Press the left thumbstick; IK resumes.
9. Finish collection from the recording menu.
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
