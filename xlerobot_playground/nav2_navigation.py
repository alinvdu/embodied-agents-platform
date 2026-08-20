# Copyright 2026 Alin Vasile Dumitru
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import math
from typing import Any

from xlerobot_agent.exploration import Pose2D
from xlerobot_playground.exploration_runtime import (
    NavigationPath,
    NavigationResult,
    ScanObservation,
)


def scan_observation_to_payload(scan: ScanObservation | dict[str, Any] | None) -> dict[str, Any] | None:
    if scan is None:
        return None
    if isinstance(scan, dict):
        return dict(scan)
    return {
        "frame_id": scan.frame_id,
        "reference_frame": "",
        "pose": scan.pose,
        "range_min": float(scan.range_min),
        "range_max": float(scan.range_max),
        "angle_min": float(scan.angle_min),
        "angle_increment": float(scan.angle_increment),
        "ranges": tuple(float(item) for item in scan.ranges),
    }


def path_length_m(poses: list[Pose2D] | tuple[Pose2D, ...]) -> float:
    total = 0.0
    for previous, current in zip(poses, poses[1:]):
        total += math.hypot(current.x - previous.x, current.y - previous.y)
    return total


class Nav2RouterNavigationDependency:
    """NavigationDependency implementation backed by the existing Nav2 router.

    The robot embodiment remains outside this class. A ManiSkill or real-XLeRobot
    runtime feeds map/scan/pose here, then asks for a Nav2 path. The router owns
    the ROS/Nav2 action calls.
    """

    def __init__(
        self,
        client: Any,
        *,
        planner_id: str = "",
        controller_id: str = "",
        behavior_tree: str = "",
    ) -> None:
        self.client = client
        self.planner_id = planner_id
        self.controller_id = controller_id
        self.behavior_tree = behavior_tree
        self._occupancy_map: Any | None = None
        self._scan: ScanObservation | dict[str, Any] | None = None
        self._image_data_url: str | None = None
        self._latest_pose: Pose2D | None = None

    def update_map(self, occupancy_map: Any) -> None:
        self._occupancy_map = occupancy_map
        self._publish_state_if_ready()

    def update_scan(self, scan: ScanObservation | dict[str, Any]) -> None:
        self._scan = scan
        self._publish_state_if_ready()

    def update_pose(self, pose: Pose2D) -> None:
        self._latest_pose = pose
        self._publish_state_if_ready()

    def update_image(self, image_data_url: str | None) -> None:
        self._image_data_url = image_data_url
        self._publish_state_if_ready()

    def update_state(
        self,
        *,
        occupancy_map: Any | None = None,
        pose: Pose2D | None = None,
        scan: ScanObservation | dict[str, Any] | None = None,
        image_data_url: str | None = None,
    ) -> None:
        if occupancy_map is not None:
            self._occupancy_map = occupancy_map
        if pose is not None:
            self._latest_pose = pose
        if scan is not None:
            self._scan = scan
        if image_data_url is not None:
            self._image_data_url = image_data_url
        self._publish_state_if_ready()

    def plan_path(self, start: Pose2D, goal: Pose2D) -> NavigationPath:
        self.update_pose(start)
        status, path_poses, status_label = self.client.compute_path(
            goal_pose=goal,
            planner_id=self.planner_id,
        )
        poses = tuple(path_poses)
        return NavigationPath(
            poses=poses,
            source="ros_nav2_router",
            cost_m=path_length_m(list(poses)) if poses else None,
            metadata={
                "status": status,
                "status_label": status_label,
                "planner_id": self.planner_id,
            },
        )

    def navigate_to_pose(self, goal: Pose2D) -> NavigationResult:
        if not hasattr(self.client, "navigate_to_pose"):
            return NavigationResult(
                succeeded=False,
                message="Nav2 router client only supports path planning; execution belongs to the robot runtime.",
                final_pose=None,
                metadata={"goal": goal.to_dict(), "source": "ros_nav2_router"},
            )
        outcome, feedback_samples = self.client.navigate_to_pose(
            goal_pose=goal,
            behavior_tree=self.behavior_tree,
        )
        status = outcome.get("status") if isinstance(outcome, dict) else None
        succeeded = str(outcome.get("status_label", status)).lower() == "succeeded" if isinstance(outcome, dict) else False
        return NavigationResult(
            succeeded=succeeded,
            message="Nav2 navigate_to_pose succeeded." if succeeded else f"Nav2 navigate_to_pose returned {status}.",
            final_pose=goal if succeeded else None,
            metadata={
                "outcome": outcome,
                "feedback_samples": feedback_samples,
                "behavior_tree": self.behavior_tree,
            },
        )

    def cancel(self) -> None:
        if hasattr(self.client, "cancel"):
            self.client.cancel()

    def snapshot(self) -> dict[str, Any]:
        if hasattr(self.client, "snapshot"):
            return self.client.snapshot()
        return {}

    def _publish_state_if_ready(self) -> None:
        if self._latest_pose is None:
            return
        if not hasattr(self.client, "update_state"):
            return
        self.client.update_state(
            occupancy_map=self._occupancy_map,
            pose=self._latest_pose,
            scan_observation=scan_observation_to_payload(self._scan),
            image_data_url=self._image_data_url,
        )
