# Cherry Juice Bottle Dataset Progress

## Dataset

- Dataset ID: `alindumitru/robot42_cherry_juice_bottle_to_basket_v0`
- Public Hub repo: `alindumitru/small-juice-bottle-to-basket`
- Canonical root: `/Users/alindumitru/embodied-agents-platform/datasets/robot42_cherry_juice_bottle_to_basket_v0`
- Task: `Pick up the small bottle of cherry juice and put it in the robot basket.`
- Cameras: Orbbec head and right wrist, 640x480 at 30 FPS
- Manipulation: right arm only; left arm held at ACTION_READY
- Collection strategy: staged curriculum; expand only after the current stage is stable
- Stage 1 target: 50 episodes across five fixed 2D positions
- Current: 50 saved episodes, 30,384 frames
- Next episode index: 50

Only episodes confirmed by the dataset metadata count toward progress. An episode
is persisted only after explicitly pressing Save.

## Position Convention

- Keep the table and basket fixed.
- Treat 4 cm from the table edge as the nominal bottle depth. Measure depth to
  the bottle center and use the same convention for every block.
- Keep the robot parallel to the table edge unless a block says otherwise.
- Bottle `left` and `right` are measured from the robot's point of view.
- Mark every robot and bottle position with tape before recording.
- Keep bottle orientation, lighting, camera mounts, ACTION_READY, grasp strategy,
  basket trajectory, and release strategy consistent.

## Curriculum

| Stage | Training distribution | Goal | Status |
| --- | --- | --- | --- |
| 1 | Five fixed positions in a plus layout | Learn a stable basic policy | Complete: 50/50; ready for upload and training |
| 1b | More repetitions of the same five positions | Use only if stage 1 is unstable | Conditional |
| 2 | Evaluate untrained positions between the five training locations | Measure interpolation before adding data | Locked until stage 1 passes |
| 3 | Add demonstrations only at a weak position or depth | Expand one variable at a time | Conditional on stage 2 |
| 4 | Add small robot-position changes one variable at a time | Expand approach tolerance | Locked until earlier stages pass |

Do not introduce random position jitter, positions beyond the five planned
markers, or robot-position changes during stage 1. First establish that the
policy can reliably reproduce a small, clean distribution. The final episode
target will be chosen from evaluation results, not fixed at 180 in advance.

## Completed Block 1

Robot position: nominal centered approach at the normal table distance.

| Episode range | Bottle position | Count | Status |
| --- | --- | ---: | --- |
| 0-9 | Center | 10 | Complete |
| 10-19 | 5 cm right | 10 | Complete |
| 20-29 | 5 cm left | 10 | Complete |

## Completed Stage 1

Keep the robot, cameras, basket, bottle orientation, and grasp strategy
unchanged. Add one centered position farther from the robot and one centered
position closer to the robot. Together with the completed center, left, and
right positions, this creates a simple five-position plus layout.

| Episode range | Bottle position | Count | Status |
| --- | --- | ---: | --- |
| 30-39 | Centered laterally, closer to robot | 10 | Complete |
| 40-45 | Centered laterally, farther from robot | 6 | Complete |
| 46-49 | Centered laterally, farther from robot | 4 | Complete |

The finalized dataset reports 50 episodes and 30,384 frames. All finalized
camera videos decode through their last registered frame.

## Stage 1 Evaluation

Train an ACT baseline on episodes 0-49, then test five physical attempts at each
of the five recorded positions with the robot pose unchanged.

| Test position | Attempts | Passing target |
| --- | ---: | ---: |
| Center, 4 cm depth | 5 | At least 4 successful placements |
| 5 cm left, 4 cm depth | 5 | At least 4 successful placements |
| 5 cm right, 4 cm depth | 5 | At least 4 successful placements |
| Centered laterally, farther marker | 5 | At least 4 successful placements |
| Centered laterally, closer marker | 5 | At least 4 successful placements |

If the model fails at the recorded positions, add clean demonstrations to the
specific weak positions and retrain. If it passes, first test untrained positions
between the five training locations. Add new demonstrations only where that
evaluation exposes a meaningful weakness.

## Session Log

| Date | Episodes added | Robot position | Bottle positions | Result | Notes |
| --- | ---: | --- | --- | --- | --- |
| 2026-08-08 | 30 | Nominal centered approach | Center, 5 cm left, 5 cm right | Complete | Consolidated and validated at 30 episodes / 18,802 frames. |
| 2026-08-09 | 16 | Nominal centered approach | 10 closer, 6 farther | Partial | Dataset finalized at 46 episodes / 28,025 frames after an encoding worker crash; the active unsaved episode was discarded and both final camera videos decode cleanly. |
| 2026-08-09 | 4 | Nominal centered approach | Farther | Complete | Finalized at 50 episodes / 30,384 frames. Pre-upload audit passed after removing 652 stale, unregistered frame rows left by the earlier failed save and quarantining its temporary MP4 files. |

## References

- [LeRobot real-world imitation-learning guide](https://huggingface.co/docs/lerobot/il_robots)
- [LeRobot SmolVLA data-collection guide](https://github.com/huggingface/lerobot/blob/main/docs/source/smolvla.mdx)

The downloaded reference top-view video was inspected across all 50 episode
starts. It shows five broadly separated groups distributed across the reachable
2D area, with placement variation inside each group. They are not a measured
grid or five closely spaced lateral points. The outer groups are separated from
the center by roughly 2.5 apparent object widths laterally, and the dataset also
contains meaningful depth variation. These are image-space estimates, not
published physical measurements.
