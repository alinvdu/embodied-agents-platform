from __future__ import annotations

import os
from pathlib import Path


DEFAULT_NAV2_BEHAVIOR_TREE_FILENAME = "navigate_to_pose_replanning_no_recovery.xml"


def default_nav2_behavior_tree() -> str:
    repo_candidate = Path(__file__).resolve().parent / "nav2_behavior_trees" / DEFAULT_NAV2_BEHAVIOR_TREE_FILENAME
    if repo_candidate.exists():
        return str(repo_candidate)
    ros_distro = os.environ.get("ROS_DISTRO", "humble")
    candidate = (
        Path("/opt/ros")
        / ros_distro
        / "share"
        / "nav2_bt_navigator"
        / "behavior_trees"
        / DEFAULT_NAV2_BEHAVIOR_TREE_FILENAME
    )
    if candidate.exists():
        return str(candidate)
    return DEFAULT_NAV2_BEHAVIOR_TREE_FILENAME
