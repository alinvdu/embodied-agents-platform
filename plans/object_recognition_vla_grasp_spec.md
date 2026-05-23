# Robot42 Object Recognition And Mock Grasp Spec

## Goal

Add object search on top of region exploration:

1. Navigate to planned observation stops inside a semantic region.
2. Capture RGB-D shots from each planned shot cone. The current implementation captures RGB only; depth/3D target extraction is the next step.
3. Run object recognition for the user-requested target.
4. Stop exploration as soon as a confident match is found.
5. Center the object in the camera view.
6. Approach slowly until the object is in grasp range.
7. Call a mocked `grab_object` tool. This later becomes the VLA skill entrypoint.

Example command:

> Go to the kitchen and bring me the Coke from the shelf.

## Key Constraints

- The object detector should produce bounding boxes.
- RGB-D depth should convert the selected box into an approximate 3D target in the robot/base frame.
- The robot must not approach blindly from detection alone.
- Approach must use small validated increments and local free-space checks.
- The VLA model should not need to be resident while object search is running.
- For a 16GB GPU, object recognition and VLA should be treated as staged workloads unless the VLA is very small or quantized.

## Recommended Runtime Strategy

Use a separate perception worker:

```text
Agent
  -> execute_region_exploration_plan(region, object_label)
  -> PerceptionWorker.detect_object(rgb, depth, object_label)
  -> focus_detected_object(detection)
  -> approach_detected_object(detection)
  -> grab_object_mock(object_description, detection_context)
```

The perception worker can run in one of three modes:

```text
local_light
  Small local detector. Best default for development.

local_heavy
  Stronger local detector. Use when VLA is not resident.

remote
  Hosted detector service. Use when local GPU memory is reserved for VLA.
```

Current implementation status:

```text
Implemented:
  - provider interface
  - mock detector
  - Replicate Grounding DINO provider
  - detection after each region-exploration RGB shot
  - abort remaining shots when a match is found
  - abort region exploration if the configured detector fails or is unavailable
  - detection metadata saved into the vision report manifest

Not implemented yet:
  - depth-based 3D pose extraction
  - focus_detected_object
  - approach_detected_object
  - grab_object mock/VLA entrypoint
```

## Model Options

### Option A: YOLOE Small Or Medium

Use for the first local prototype.

Why:

- Open-vocabulary / promptable detection.
- Designed for real-time use.
- Small variants should fit comfortably on a 16GB GPU.
- Ultralytics docs say YOLOE keeps YOLO-like inference speed and parameter count while supporting text/image prompts.

Tradeoff:

- May be weaker than Grounding DINO on unusual phrases.
- For very specific objects like a particular Coke can, performance may depend on prompt quality or visual prompting.

Source:

- https://docs.ultralytics.com/models/yoloe/

Recommended role:

```text
Primary local detector for v1.
```

### Option B: Grounding DINO SwinT

Use for stronger open-vocabulary detection.

Why:

- Good language-conditioned detector.
- Accepts `(image, text)` input and outputs boxes for prompted phrases.
- The SwinT checkpoint is around 694MB on Hugging Face, so the model is not huge on disk.

Tradeoff:

- More memory and latency than YOLOE.
- More annoying dependency stack.
- Better as on-demand inference, not always resident beside a VLA.

Sources:

- https://github.com/IDEA-Research/GroundingDINO
- https://huggingface.co/Flynn12/grounding-dino/blob/main/groundingdino_swint_ogc.pth

Recommended role:

```text
Second local detector or remote detector fallback.
```

### Option C: Florence-2 Base

Use as a compact vision-language fallback or hosted detector.

Why:

- Can do captioning, object detection, grounding, OCR, and segmentation from prompts.
- Florence-2 base is commonly listed as a compact model around 232M parameters; large is around 771M.

Tradeoff:

- Detection boxes may need more post-processing than YOLO-style detectors.
- Useful as a verifier / semantic fallback, not necessarily the fastest detector.

Sources:

- https://huggingface.co/microsoft/Florence-2-base
- https://arxiv.org/abs/2311.06242

Recommended role:

```text
Compact detector/verifier fallback.
```

### Option D: OpenAI Vision As Verifier

Use online for semantic verification, not as the only box detector.

Why:

- Strong at answering "is this crop a Coke can?" or "is this the newly printed object?"
- Useful after a local model proposes boxes.

Tradeoff:

- Not ideal as the primary detector because Robot42 needs reliable box outputs.
- Network dependency and latency.

Source:

- https://platform.openai.com/docs/guides/vision

Recommended role:

```text
Optional crop verifier after local detection.
```

## Configuring Online Detection

The first runnable online provider is Replicate Grounding DINO.

Environment:

```bash
export REPLICATE_API_TOKEN="r8_..."
export ROBOT42_OBJECT_DETECTOR_PROVIDER=replicate_grounding_dino
export ROBOT42_OBJECT_DETECTOR_MODEL=adirik/grounding-dino
export ROBOT42_OBJECT_DETECTOR_BOX_THRESHOLD=0.25
export ROBOT42_OBJECT_DETECTOR_TEXT_THRESHOLD=0.25
export ROBOT42_OBJECT_DETECTOR_MIN_CONFIDENCE=0.25
export ROBOT42_OBJECT_DETECTOR_TIMEOUT_S=90
```

Or CLI:

```bash
python examples/robot42_agent_backend.py \
  --memory-root ./artifacts/memories \
  --exploration-backend-url http://127.0.0.1:8770 \
  --provider openai \
  --model gpt-5.5 \
  --navigation-waypoint-horizon-m 2.0 \
  --max-turns 80 \
  --object-detector-provider replicate_grounding_dino \
  --object-detector-api-key "$REPLICATE_API_TOKEN" \
  --object-detector-model adirik/grounding-dino \
  --object-detector-box-threshold 0.25 \
  --object-detector-text-threshold 0.25 \
  --object-detector-min-confidence 0.25
```

For local dry tests without spending API calls:

```bash
--object-detector-provider mock
```

Replicate Grounding DINO inputs:

```json
{
  "image": "data:image/png;base64,...",
  "query": "coke can",
  "box_threshold": 0.25,
  "text_threshold": 0.25,
  "show_visualisation": true
}
```

Replicate accepts HTTP URLs or data URLs for image inputs. Robot42 currently sends the original saved RGB shot as a data URL because local agent artifacts are not reachable from Replicate.

Robot42 normalizes provider output into:

```json
{
  "status": "matched",
  "provider": "replicate_grounding_dino",
  "object_label": "coke can",
  "detections": [
    {
      "detection_id": "kitchen_stop_1_shot_1_det_1",
      "label": "coke can",
      "confidence": 0.72,
      "bbox_xyxy": [320, 180, 420, 360]
    }
  ],
  "selected_detection_id": "kitchen_stop_1_shot_1_det_1"
}
```

## 16GB GPU Recommendation

Do not plan on running a large VLA model and a heavy detector resident at the same time.

Recommended policy:

```text
Object search phase:
  load detector
  run detection on shots
  unload detector or keep only if enough VRAM remains

Manipulation phase:
  load VLA/grasp skill model
  execute grab
```

Likely practical combinations:

```text
YOLOE small + small/quantized VLA:
  probably feasible on 16GB, but test VRAM.

Grounding DINO + large VLA:
  not recommended resident together.

Florence-2 base + small/quantized VLA:
  possible, but still benchmark.

Remote detector + local VLA:
  safest if VLA needs most of the GPU.
```

The first implementation should support both:

```text
ROBOT42_DETECTOR_PROVIDER=local_yoloe
ROBOT42_DETECTOR_PROVIDER=remote
```

## Detection Contract

Input:

```json
{
  "shot_id": "kitchen_stop_2_shot_left",
  "rgb_path": "artifacts/agent_runs/run_x/vision_report/kitchen_stop_2_left.jpg",
  "depth_path": "optional",
  "object_label": "coke can",
  "region_label": "kitchen",
  "camera_frame": "camera_head",
  "robot_pose_map": {"x": 2.1, "y": -0.4, "yaw": 0.1}
}
```

Output:

```json
{
  "status": "matched",
  "shot_id": "kitchen_stop_2_shot_left",
  "target_label": "coke can",
  "detections": [
    {
      "detection_id": "det_001",
      "label": "coke can",
      "confidence": 0.78,
      "bbox_xyxy": [320, 180, 420, 360],
      "depth_m": 0.82,
      "estimated_pose_base": {"x": 0.78, "y": 0.10, "z": 0.35},
      "crop_path": "artifacts/agent_runs/run_x/vision_report/det_001_crop.jpg",
      "annotated_image_path": "artifacts/agent_runs/run_x/vision_report/kitchen_stop_2_left_annotated.jpg"
    }
  ],
  "selected_detection_id": "det_001"
}
```

## Region Exploration With Detection

Current behavior:

```text
execute_region_exploration_plan
  -> navigate stop
  -> rotate shot
  -> save RGB debug shot
```

New behavior:

```text
execute_region_exploration_plan
  -> navigate stop
  -> rotate shot
  -> save RGB-D debug shot
  -> detect target object
  -> if match found:
       abort remaining shots
       return matched detection
     else:
       continue
```

Current return statuses:

```text
detection_status="not_configured"
  No detector provider is configured.

detection_status="not_found"
  Detection ran on all completed shots but found no match above threshold.

detection_status="matched"
  A matching detection was found and remaining shots were aborted.

detection_status="unavailable"
  A configured detector could not run, for example because an API token is missing. Region exploration is aborted.

detection_status="failed"
  At least one detection request failed.
```

The agent must not claim success unless the detection result is `matched`.

## Object Focus

Tool:

```text
focus_detected_object(detection_id, constraints_json)
```

Behavior:

1. Capture a fresh RGB frame from the exploration backend.
2. Re-detect or track the selected object.
3. Compare box center to image center.
4. Rotate the robot in small bounded increments.
5. Stop when the object center is inside a configurable normalized tolerance.
6. Return the updated tracked detection.

Suggested defaults:

```json
{
  "center_tolerance_norm": 0.08,
  "max_attempts": 5,
  "max_yaw_step_deg": 12,
  "horizontal_fov_deg": 65
}
```

## Safe Approach

Tool:

```text
approach_detected_object(detection_id, constraints_json)
```

Behavior:

1. Capture RGB and re-run detection.
2. Send the bbox to the exploration backend.
3. The backend solves median object position from the latest organized RGB-D point cloud.
4. If distance is already in grasp range, stop.
5. If too far, check a local swept footprint corridor from RGB-D geometry.
6. Move forward in small increments.
7. Re-capture and re-detect after every increment.
8. Stop if the object is lost, depth is invalid, or the footprint corridor is unsafe.

Suggested defaults:

```json
{
  "target_min_m": 0.35,
  "target_max_m": 0.45,
  "step_m": 0.08,
  "robot_width_m": 0.459,
  "clearance_m": 0.06,
  "max_attempts": 12
}
```

Safety checks:

- Object depth must be valid.
- Shelf/wall must not intersect the robot body path.
- Robot footprint corridor must be clear.
- If the object appears too high/low for the arm, stop and report `not_reachable`.
- If detection confidence drops below threshold, stop and report `object_lost`.

## Mock Grasp Tool

Tool:

```text
grab_object(object_label, detection_id, object_description, constraints_json)
```

V1 behavior:

```json
{
  "status": "mock_succeeded",
  "reason": "Mock grab_object completed. This is the future VLA skill entrypoint.",
  "object_label": "coke can",
  "detection_id": "det_001"
}
```

Later behavior:

```text
SkillRegistry.run_vla_skill(
  skill_id="grab_object",
  target_detection=detection,
  task_language="grab the Coke can from the shelf"
)
```

## Agent Instructions

The agent should understand:

- `execute_region_exploration_plan` searches a region by moving through planned observation stops.
- Object detection runs after each shot.
- If a match is found, remaining shots are aborted.
- `focus_detected_object` centers the object.
- `approach_detected_object` moves slowly using local safety checks.
- `grab_object` is mocked until VLA is connected.
- While `grab_object` is mocked, report a mock VLA handoff, not a real physical pickup.
- If detection fails, summarize where the robot looked and what debug images were saved.

Example:

```text
User: Go to the kitchen and bring me a Coke from the shelf.

Agent:
1. execute_region_exploration_plan("kitchen", "coke can")
2. If matched: focus_detected_object(detection_id)
3. approach_detected_object(detection_id)
4. grab_object("coke can", detection_id, description)
5. Report result.
```

## Debug Report

Every run should save:

```text
artifacts/agent_runs/<run_id>/vision_report/
  manifest.json
  shots/<region>_<stop>_<shot>.png
  shots/<region>_<stop>_<shot>.json
  shots/focus_detected_object_<object>_<attempt>.png
  shots/approach_detected_object_<object>_<attempt>.png
```

Manifest:

```json
{
  "run_id": "run_...",
  "capture_count": 3,
  "captures": [
    {
      "object_label": "coke can",
      "artifact_url": "/api/artifacts/...",
      "detection": {
        "status": "matched",
        "selected_detection": {
          "bbox_xyxy": [10, 20, 110, 220],
          "confidence": 0.81
        }
      }
    }
  ]
}
```

## Implementation Phases

### Phase 1: Detection Adapter

- Done: add `ObjectDetector` adapter.
- Done: add local mock detector for tests.
- Done: add provider config:
  - `none`
  - `mock`
  - `replicate_grounding_dino`
- Done: save RGB shots and manifest entries with detection metadata.

### Phase 2: Wire Detection Into Region Exploration

- Done: run detector after every RGB shot.
- Done: abort remaining shots on match.
- Done: emit UI events with detection results.
- Done: show saved RGB images and detection status in the React debug report.

### Phase 3: Focus And Approach

- Done: add `focus_detected_object`.
- Done: add `approach_detected_object`.
- Done: solve depth median inside bbox from the backend's organized RGB-D point cloud.
- Done: add a first-pass local footprint corridor safety check.

### Phase 4: Mock Grab

- Done: add `grab_object` tool.
- Done: return mocked result.
- Done: keep selected detection tracking state for future VLA dataset wiring.

### Phase 5: VLA Entry Point

- Replace mock grab with `SkillRegistry.run_vla_skill`.
- Pass object label, selected crop, RGB-D frame, base pose, and task language.
- Record success/failure for later training.

## Open Questions

- Which detector should be first in local tests: YOLOE small or Grounding DINO?
- Should remote detection be required before VLA is loaded, or only used when local detector fails?
- What exact grasp distance should be used for the arm: `0.35m`, `0.40m`, or `0.45m`?
- Should the robot move base-only during approach, or allow camera/head correction between increments?
