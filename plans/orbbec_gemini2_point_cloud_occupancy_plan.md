# Orbbec Gemini 2 Point Cloud Occupancy Mapping Plan

## Goal

Replace the current narrow depth-row-to-`LaserScan` mapping path with a point-cloud-based occupancy mapper for real Orbbec Gemini 2 scans.

The target result is a more perceptible 2D map of indoor spaces:

- continuous wall and furniture obstacle evidence
- conservative handling of glass, missing depth, and reflective surfaces
- useful free-space/frontier boundaries for exploration
- compatibility with the existing review UI and Nav2 integration

The existing synthetic scan path should remain available as a debug/fallback mode.

## Current Problem

The current real mapping pipeline converts a thin horizontal band of the depth image into a fake planar laser scan, then fuses that scan into a 2D occupancy grid.

Relevant files:

- `tools/orbbec_rgb_test/main.cpp`: Orbbec SDK sidecar currently captures RGB, depth, optional depth registration, and IMU.
- `xlerobot_playground/real_ros_bridge.py`: publishes RGB, depth, TF, IMU, and depth-derived `/scan`.
- `xlerobot_playground/scan_fusion.py`: ray-carves a 2D grid from `LaserScan` ranges.
- `xlerobot_playground/sim_exploration_backend.py`: integrates ROS scan observations into the `fused_scan` map.

The bad maps are expected from this design because each beam contributes many free cells and only one occupied endpoint. Sparse or unstable depth endpoints become broken dots, while free-space carving creates stripe artifacts.

## Proposed Architecture

```text
Orbbec Gemini 2
  -> SDK RGB-D frames
  -> SDK PointCloudFilter
  -> binary point cloud payload
  -> robot_brain_agent / real_ros_bridge
  -> point cloud projection + height filtering
  -> occupancy evidence grid
  -> review UI + Nav2 map publishing
```

Use the SDK point cloud as the primary geometry source. Keep the depth image and synthetic scan for diagnostics.

## Phase 1: Capture SDK Point Cloud

Add point cloud support to `tools/orbbec_rgb_test/main.cpp`.

Use the Orbbec SDK v2 sample pattern:

```cpp
auto pointCloud = std::make_shared<ob::PointCloudFilter>();
auto align = std::make_shared<ob::Align>(OB_STREAM_COLOR);

auto frameset = pipeline.waitForFrameset(timeout_ms);
auto alignedFrameset = align->process(frameset);

pointCloud->setCreatePointFormat(OB_FORMAT_POINT);
std::shared_ptr<ob::Frame> cloud = pointCloud->process(alignedFrameset);

auto points = reinterpret_cast<OBPoint *>(cloud->data());
int count = cloud->dataSize() / sizeof(OBPoint);
```

Add sidecar flags:

- `--enable-point-cloud`
- `--point-cloud-format xyz|xyzrgb`
- `--point-cloud-stride N`
- `--point-cloud-max-points N`
- `--point-cloud-min-z-m`
- `--point-cloud-max-z-m`

Start with depth-only `OB_FORMAT_POINT`. Add `OB_FORMAT_RGB_POINT` later only if semantic coloring/debug visualization needs it.

## Phase 2: Define A Transport Format

Extend the existing RGB-D HTTP payload rather than creating a second service at first.

Recommended binary layout:

```text
magic: XLRGBDPC
version: u32
frame_index: u64
timestamp_us: u64
rgb_width, rgb_height, rgb_size
depth_width, depth_height, depth_size
point_format: u32  # 1=xyz_float32, 2=xyzrgb_float32_u8
point_count: u32
point_stride_bytes: u32
point_payload_size: u64
metadata_json_size: u32
rgb bytes
depth bytes
point cloud bytes
metadata json
```

For `xyz_float32`, store each point as:

```text
float32 x
float32 y
float32 z
```

For `xyzrgb`, store:

```text
float32 x
float32 y
float32 z
uint8 r
uint8 g
uint8 b
uint8 padding
```

Keep units explicit in metadata. Orbbec `OBPoint` coordinates may be in millimeters depending on SDK configuration/sample path; normalize to meters before sending if possible.

## Phase 3: Parse And Publish Point Cloud In Python

Update:

- `xlerobot_playground/rgbd_transport.py`
- `xlerobot_playground/robot_brain_agent.py`
- `xlerobot_playground/real_ros_bridge.py`

Add fields to `RgbdFrame`:

- `point_cloud_format`
- `point_cloud_points`
- `point_cloud_count`
- `point_cloud_stride`
- `point_cloud_units`

In `real_ros_bridge.py`, publish a ROS `sensor_msgs/PointCloud2` topic:

```text
/camera/head/points
frame_id: head_camera_link
```

This is useful for RViz validation and later Nav2 voxel-layer experiments.

## Phase 4: Implement Point Cloud To Occupancy Fusion

Create a new module:

```text
xlerobot_playground/point_cloud_fusion.py
```

Core function:

```python
integrate_point_cloud_observation(
    pose,
    points_xyz_camera,
    camera_to_base_tf,
    map_resolution_m,
    known_cells,
    evidence_scores,
    config,
)
```

Processing steps:

1. Drop invalid points: NaN, zero range, outside min/max range.
2. Transform points from camera frame to base/map frame.
3. Remove floor and ceiling points.
4. Keep obstacle/wall height bands.
5. Project surviving points into 2D grid cells as occupied evidence.
6. Raycast free space only to valid point endpoints.
7. Do not clear through missing depth or invalid point regions.

Initial filter defaults:

```text
range_min_m = 0.25
range_max_m = 4.0
floor_ignore_z_m = 0.05
obstacle_min_z_m = 0.08
obstacle_max_z_m = 1.80
free_ray_max_m = 4.0
occupied_enter_threshold = 2.0
free_weight = -0.20
occupied_weight = 1.0
```

Use a coarser voxel/downsample stage before grid insertion:

```text
voxel_size_m = occupancy_resolution_m / 2
```

## Phase 5: Add A New Map Source

Extend `--ros-navigation-map-source` choices:

```text
fused_scan
fused_point_cloud
external
```

When `fused_point_cloud` is selected:

- consume `/camera/head/points` or internal `RgbdFrame` point cloud payload
- update `scan_known_cells` equivalent, probably renamed later to `sensor_known_cells`
- publish the fused map to the Nav2 adapter just like `fused_scan`
- keep frontiers and UI payload format unchanged

This minimizes UI and frontier code churn.

## Phase 6: Wall And Room Feature Extraction

After the basic point-cloud occupancy map works, add a post-processing layer for human-perceptible structure.

Recommended first pass:

- cluster occupied cells
- extract long line segments using Hough or RANSAC
- merge near-collinear wall fragments
- snap almost-horizontal/vertical lines when the local evidence supports it
- mark glass/invalid-depth regions as `unknown_boundary`, not occupied

Do not make wall fitting part of the safety map at first. Use it for visualization and semantic room understanding until it is well validated.

## Phase 7: Validation Workflow

Add debug outputs:

- raw point cloud stats: point count, valid count, min/max range
- height-filtered point count
- occupied cell count
- free cell count
- invalid/no-return count
- per-frame processing time

Add RViz checks:

```bash
ros2 topic echo /camera/head/points --once
ros2 topic hz /camera/head/points
ros2 run rviz2 rviz2
```

Visual checks:

- point cloud aligns with RGB/depth image
- head pan rotates points correctly in TF
- floor is not marked as wall
- transparent sliding door remains unknown or weak evidence
- left/right walls become continuous after multiple scan poses
- kitchen entrance remains open but has side boundary evidence

## Phase 8: Tuning Strategy

Start conservative:

```text
range_max_m = 3.5 to 4.0
occupancy_resolution = 0.10 to 0.15 for debugging
occupancy_resolution = 0.20 to 0.25 for exploration UI
point_cloud_stride = 2 or 4
voxel_size_m = 0.05 to 0.10
```

Tune in this order:

1. Confirm point cloud units and coordinate axes.
2. Confirm TF during head pan.
3. Tune floor removal.
4. Tune obstacle height band.
5. Tune raycast free-space weight.
6. Tune occupied evidence threshold.
7. Add line/wall fitting.

## Risks

Glass and reflective surfaces will remain hard. The correct behavior is conservative unknown space, not false free space.

Pose/TF errors can ruin the map even with perfect point cloud data. Pan direction, degree/radian mode, timestamping, and `head_camera_link` transform must be validated before tuning fusion.

Large point clouds can overload Python if sent at full resolution and full frame rate. Downsample in C++ first and publish at a controlled rate.

## Acceptance Criteria

The implementation is ready when:

- `/camera/head/points` publishes valid `PointCloud2` data from Gemini 2.
- RViz shows point clouds aligned to `head_camera_link` and rotating correctly with head pan.
- `fused_point_cloud` creates fewer stripe artifacts than `fused_scan`.
- left/right living-room walls appear as mostly continuous occupied evidence after a 360 scan.
- open kitchen entrance remains open, with boundary evidence around the entrance.
- transparent sliding door does not get incorrectly carved as clear free space.
- Nav2 can still receive a published occupancy map from the exploration runtime.

## First Implementation Milestone

Implement only:

1. `--enable-point-cloud` in `tools/orbbec_rgb_test/main.cpp`.
2. binary `xyz_float32` payload in meters.
3. parser support in `rgbd_transport.py`.
4. `/camera/head/points` publisher in `real_ros_bridge.py`.
5. one manual RViz validation command path.

Only after this works should occupancy fusion be changed.
