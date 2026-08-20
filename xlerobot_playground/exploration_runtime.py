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

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from xlerobot_agent.exploration import Pose2D


@dataclass(frozen=True)
class RGBDObservation:
    """One robot-head RGB-D observation in the runtime's native camera frame."""

    rgb: Any
    depth: Any
    pose: Pose2D
    frame_id: str
    intrinsics: dict[str, float] | None = None
    timestamp_s: float | None = None


@dataclass(frozen=True)
class ScanObservation:
    pose: Pose2D
    ranges: Sequence[float]
    angle_min: float
    angle_increment: float
    range_min: float
    range_max: float
    frame_id: str
    timestamp_s: float | None = None


@dataclass(frozen=True)
class NavigationPath:
    poses: tuple[Pose2D, ...]
    source: str
    cost_m: float | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class NavigationResult:
    succeeded: bool
    message: str
    final_pose: Pose2D | None = None
    travelled_distance_m: float | None = None
    metadata: dict[str, Any] | None = None


class NavigationDependency(Protocol):
    """Planner/controller service used by an exploration runtime.

    ROS/Nav2 belongs behind this boundary: it can plan, validate, or execute
    navigation, but it is not the embodiment that owns sensors and actuation.
    """

    def update_map(self, occupancy_map: Any) -> None:
        ...

    def update_scan(self, scan: ScanObservation | dict[str, Any]) -> None:
        ...

    def plan_path(self, start: Pose2D, goal: Pose2D) -> NavigationPath:
        ...

    def navigate_to_pose(self, goal: Pose2D) -> NavigationResult:
        ...

    def cancel(self) -> None:
        ...


class ExplorationRuntime(Protocol):
    """Embodiment boundary shared by ManiSkill and the real XLeRobot."""

    @property
    def name(self) -> str:
        ...

    def reset(self) -> None:
        ...

    def current_pose(self) -> Pose2D:
        ...

    def capture_rgbd(self) -> RGBDObservation:
        ...

    def latest_scan(self) -> ScanObservation | None:
        ...

    def drive_velocity(self, *, linear_m_s: float, angular_rad_s: float) -> NavigationResult:
        ...

    def stop(self) -> NavigationResult:
        ...

    def rotate_in_place(self, yaw_delta_rad: float) -> NavigationResult:
        ...

    def execute_path(self, path: Sequence[Pose2D]) -> NavigationResult:
        ...
