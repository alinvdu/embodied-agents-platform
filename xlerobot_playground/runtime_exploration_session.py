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

from dataclasses import dataclass, field
from typing import Any

from xlerobot_agent.exploration import Pose2D
from xlerobot_playground.exploration_runtime import (
    ExplorationRuntime,
    NavigationDependency,
    NavigationPath,
    NavigationResult,
    RGBDObservation,
    ScanObservation,
)


@dataclass
class RuntimeExplorationSnapshot:
    runtime_name: str
    pose: Pose2D | None
    latest_scan: ScanObservation | None
    latest_rgbd: RGBDObservation | None
    last_path: NavigationPath | None
    last_result: NavigationResult | None
    errors: list[str] = field(default_factory=list)


class RuntimeNeutralExplorationSession:
    """Small coordinator shared by simulated and real robot embodiments.

    The runtime owns sensors and actuation. The navigation dependency owns path
    planning/controller calls such as ROS/Nav2. This class only refreshes state
    across that boundary and asks the active runtime to execute.
    """

    def __init__(self, runtime: ExplorationRuntime, navigation: NavigationDependency) -> None:
        self.runtime = runtime
        self.navigation = navigation
        self.latest_scan: ScanObservation | None = None
        self.latest_rgbd: RGBDObservation | None = None
        self.last_path: NavigationPath | None = None
        self.last_result: NavigationResult | None = None
        self.errors: list[str] = []

    def reset(self) -> RuntimeExplorationSnapshot:
        self.runtime.reset()
        self.latest_scan = None
        self.latest_rgbd = None
        self.last_path = None
        self.last_result = None
        self.errors = []
        return self.refresh_navigation_state()

    def refresh_navigation_state(self, *, occupancy_map: Any | None = None, image_data_url: str | None = None) -> RuntimeExplorationSnapshot:
        pose = self._try_current_pose()
        scan = self._try_latest_scan()
        rgbd = self._try_capture_rgbd()
        if occupancy_map is not None:
            self.navigation.update_map(occupancy_map)
        if scan is not None:
            self.navigation.update_scan(scan)
        if hasattr(self.navigation, "update_state"):
            self.navigation.update_state(
                occupancy_map=occupancy_map,
                pose=pose,
                scan=scan,
                image_data_url=image_data_url,
            )
        elif pose is not None and hasattr(self.navigation, "update_pose"):
            self.navigation.update_pose(pose)
        return self.snapshot(pose=pose)

    def plan_to(self, goal: Pose2D) -> NavigationPath:
        pose = self._require_pose()
        self.refresh_navigation_state()
        self.last_path = self.navigation.plan_path(pose, goal)
        return self.last_path

    def navigate_to(self, goal: Pose2D) -> NavigationResult:
        path = self.plan_to(goal)
        if path.poses:
            self.last_result = self.runtime.execute_path(path.poses)
        else:
            self.last_result = NavigationResult(
                succeeded=False,
                message="Navigation dependency returned an empty path.",
                metadata={"goal": goal.to_dict()},
            )
        return self.last_result

    def stop(self) -> NavigationResult:
        self.navigation.cancel()
        self.last_result = self.runtime.stop()
        return self.last_result

    def snapshot(self, *, pose: Pose2D | None = None) -> RuntimeExplorationSnapshot:
        if pose is None:
            pose = self._try_current_pose()
        return RuntimeExplorationSnapshot(
            runtime_name=self.runtime.name,
            pose=pose,
            latest_scan=self.latest_scan,
            latest_rgbd=self.latest_rgbd,
            last_path=self.last_path,
            last_result=self.last_result,
            errors=list(self.errors),
        )

    def _require_pose(self) -> Pose2D:
        pose = self._try_current_pose()
        if pose is None:
            raise RuntimeError("Runtime did not provide a current pose.")
        return pose

    def _try_current_pose(self) -> Pose2D | None:
        try:
            return self.runtime.current_pose()
        except NotImplementedError as exc:
            self._remember_error(str(exc))
            return None

    def _try_latest_scan(self) -> ScanObservation | None:
        try:
            self.latest_scan = self.runtime.latest_scan()
        except NotImplementedError as exc:
            self._remember_error(str(exc))
            self.latest_scan = None
        return self.latest_scan

    def _try_capture_rgbd(self) -> RGBDObservation | None:
        try:
            self.latest_rgbd = self.runtime.capture_rgbd()
        except NotImplementedError as exc:
            self._remember_error(str(exc))
            self.latest_rgbd = None
        return self.latest_rgbd

    def _remember_error(self, message: str) -> None:
        if message and message not in self.errors:
            self.errors.append(message)
