# VR Teleoperation Training Data

This is the operator checklist for collecting the first VLA skill dataset:

```text
grab object from a surface -> put object in the robot basket
```

The current recommended setup is right-arm-only collection, with Orbbec as the head view and two wrist cameras.

Each recorded frame includes:

```text
observation.images.head        Orbbec Gemini 2 RGB
observation.images.left_wrist  OpenCV camera 0
observation.images.right_wrist OpenCV camera 1
observation.state              left arm, right arm, head, base velocity state
action                         full robot action vector
```

The full left/right arm action/state schema is recorded even when `--vr-skill-arm right` is used.

## Practice Command

Use this first when testing controls without writing dataset episodes:

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
  --camera right_wrist=opencv:1 \
  --vr-arm-ik-mode yawed \
  --vr-skill-mode grab_to_basket \
  --vr-skill-arm right \
  --vr-basket-motion-s 2.5 \
  --vr-action-ready-motion-s 2.0 \
  --vr-arm-debug \
  --vr-arm-debug-hz 2
```

Open the printed Quest URL, usually:

```text
https://<robot-brain-ip>:8443
```

Reload the Quest browser page after every backend restart, then press `Start Controller Tracking`.

## Recording Command

Use this when collecting LeRobot episodes. This uses the same teleop path as practice mode, but enables local LeRobot recording:

```bash
sudo /Users/alindumitru/miniconda3/envs/xlerobot/bin/python -m xlerobot_playground.real_backend manipulate \
  --repo-root /Users/alindumitru/XLeRobot \
  --robot-kind xlerobot_2wheels \
  --controller vr \
  --port1 /dev/tty.usbmodem5B140330101 \
  --port2 /dev/tty.usbmodem5B140332271 \
  --xlevr-path /Users/alindumitru/XLeRobot/XLeVR \
  --record-training \
  --dataset-id local/robot42_grab_to_basket_v0 \
  --dataset-root /Users/alindumitru/Robot42/datasets/robot42_grab_to_basket_v0 \
  --task "grab the object and put it in the robot basket" \
  --use-videos \
  --orbbec-rgb-vr \
  --camera left_wrist=opencv:0 \
  --camera right_wrist=opencv:1 \
  --vr-arm-ik-mode yawed \
  --vr-skill-mode grab_to_basket \
  --vr-skill-arm right \
  --vr-basket-motion-s 2.5 \
  --vr-action-ready-motion-s 2.0 \
  --vr-arm-debug \
  --vr-arm-debug-hz 2
```

If the dataset already exists at `--dataset-root`, Robot42 resumes it and appends new episodes. Use `--no-resume-dataset` only when intentionally starting a fresh dataset directory.

For object-specific sessions, change `--task` to the user-facing instruction, for example:

```bash
--task "grab the small water bottle and put it in the robot basket"
```

## Startup

On launch, Robot42 moves through:

```text
NAV_STOW -> 5 second wait -> ACTION_READY -> VR teleop
```

Do not start recording during startup. Wait until `ACTION_READY` is complete and controller tracking is live.

## Quest Controls

Right-arm grab-to-basket controls:

```text
right B       move right arm to fixed basket pose
trigger       open gripper while in basket pose
right A       return right arm to ACTION_READY, then resume IK
right grip    IK clutch/rebaseline for the right arm
```

Recording controls:

```text
right thumbstick press  start episode, or stop and save active episode
left thumbstick press   cancel/discard active episode
left thumbstick right   fallback start/save episode control
left thumbstick left    fallback discard active episode
left thumbstick up      save active episode and quit session
left thumbstick down    reset robot pose to ACTION_READY
```

The fixed basket move is intentionally slowed to `2.5s`, and the button-triggered return to `ACTION_READY` is `2.0s`. Startup pose timing is separate.

## Episode Recipe

For each successful episode:

```text
1. Place the target object on the surface.
2. Make sure the robot is at ACTION_READY.
3. Press the right thumbstick to start recording.
4. Use IK to grab the object.
5. Press right B to move to the basket pose.
6. Press trigger to release the object into the basket.
7. Press the right thumbstick immediately after success to save.
8. Outside the episode, press right A to return to ACTION_READY.
9. Put the object back by hand and repeat.
```

Keep reset, return-to-ready, and object replacement outside the recorded episode unless intentionally training a reset skill.

## Dataset Shape

For the first dataset, collect:

```text
5 object families
5 target poses per object family
4-6 repeats per object-pose pair
```

Recommended target pose spread:

```text
center
slightly left
slightly right
about 15 degrees rotated
about 30 degrees rotated
```

The exploration/object-centering phase should make the target object the main centered object. For v0, keep the language instruction as the target grounding signal and rely on centering rather than adding bounding-box targeting.

## Good Episode Rules

Save episodes where:

```text
object is grasped cleanly
object reaches basket without collision
release succeeds
all three camera feeds are live: head, left_wrist, right_wrist
motion is smooth enough to be learnable
```

Discard episodes where:

```text
wrong object is grabbed
object is dropped before basket
arm collides hard with table/basket/robot
camera feed freezes
operator accidentally records reset or object replacement
```

## Restarting

Clean exit:

```text
Ctrl+C in terminal
wait for "Stopping VR monitor/runtime..."
reload the Quest browser page before reconnecting
```

If a previous frozen run seems to be holding the VR ports:

```bash
lsof -nP -iTCP:8442 -iTCP:8443 -sTCP:LISTEN
```

## Pose Utilities

Read current arm/head pose:

```bash
sudo /Users/alindumitru/miniconda3/envs/xlerobot/bin/python /Users/alindumitru/Robot42/scripts/capture_xlerobot_pose.py \
  --repo-root /Users/alindumitru/XLeRobot \
  --robot-kind xlerobot_2wheels \
  --port1 /dev/tty.usbmodem5B140330101 \
  --port2 /dev/tty.usbmodem5B140332271
```

Move to a test pose and print readings:

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
